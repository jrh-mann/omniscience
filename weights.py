import torch
import torch.nn as nn
from safetensors import safe_open
from pathlib import Path
import logging
from .config import ProbeConfig, OmniscienceConfig

logger = logging.getLogger(__name__)

class ProbeWeightSet(nn.Module):
    """
    Manages probe weights for a single layer.
    
    CRITICAL: Inherits from nn.Module so weights become part of model graph.
    This ensures CUDA graph compatibility.
    """
    
    def __init__(
        self,
        layer_idx: int,
        probe_configs: list[ProbeConfig],
        device: str,
        dtype: torch.dtype
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.device = device
        self.dtype = dtype
        
        # Load and stack all probes for this layer into single matrix
        weights = []
        biases = []
        self.probe_names = []
        self.activation_fns = []
        
        for probe in probe_configs:
            # Fast validation using header-only read
            self._validate_probe_shape(probe.weight_path, expected_dim=None)
            
            w, b = self._load_single_probe(probe.weight_path)
            
            # Validate shape
            if w.dim() != 1:
                raise ValueError(
                    f"Probe {probe.name} has shape {w.shape}, expected [hidden_dim]"
                )
            
            weights.append(w)
            biases.append(torch.tensor(b if b is not None else probe.bias or 0.0))
            self.probe_names.append(probe.name)
            self.activation_fns.append(probe.activation)
        
        # Stack into [num_probes, hidden_dim] for efficient matmul
        # Using @ operator: [batch, hidden_dim] @ [hidden_dim, num_probes]
        #   → Need weights as [hidden_dim, num_probes]
        weight_matrix = torch.stack(weights).T.to(dtype=dtype)  # [hidden_dim, num_probes]
        bias_vector = torch.stack(biases).to(dtype=dtype)       # [num_probes]
        
        # Register as nn.Parameter (not regular tensor) for graph compatibility
        self.weight_matrix = nn.Parameter(weight_matrix, requires_grad=False)
        self.bias_vector = nn.Parameter(bias_vector, requires_grad=False)
        
        # Move to device AFTER registering as parameters
        self.to(device=device)
        
        logger.info(
            f"Layer {layer_idx}: Loaded {len(weights)} probes, "
            f"weight matrix shape {self.weight_matrix.shape}"
        )
    
    def _validate_probe_shape(self, path: Path, expected_dim: int | None):
        """
        Fast validation using safetensors header (no tensor data loaded).
        
        Optimization: safetensors stores metadata in header.
        """
        with safe_open(str(path), framework="pt", device="cpu") as f:
            # Get shape without loading tensor data
            # Check for required keys
            keys = f.keys()
            if "weight" not in keys:
                raise ValueError(f"Probe file {path} missing 'weight' tensor")
            
            weight_slice = f.get_slice("weight")
            weight_shape = weight_slice.get_shape()
            
            if len(weight_shape) != 1:
                raise ValueError(
                    f"Probe weight must be 1D, got shape {weight_shape}"
                )
            
            if expected_dim is not None and weight_shape[0] != expected_dim:
                raise ValueError(
                    f"Probe weight has dim {weight_shape[0]}, expected {expected_dim}"
                )
    
    def _load_single_probe(self, path: Path) -> tuple[torch.Tensor, float | None]:
        """Load probe from safetensors. Expected keys: 'weight', optionally 'bias'"""
        with safe_open(str(path), framework="pt", device="cpu") as f:
            weight = f.get_tensor("weight")
            keys = f.keys()
            bias = f.get_tensor("bias").item() if "bias" in keys else None
        return weight, bias
    
    @property
    def num_probes(self) -> int:
        return len(self.probe_names)
    
    @property
    def hidden_dim(self) -> int:
        return self.weight_matrix.shape[0]
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Compute probe scores. Graph-compatible operation.
        
        Args:
            hidden_states: [num_tokens, hidden_dim]
        
        Returns:
            probe_scores: [num_tokens, num_probes]
        """
        # Core matmul: [num_tokens, hidden_dim] @ [hidden_dim, num_probes]
        scores = torch.addmm(
            self.bias_vector,      # bias [num_probes], broadcasted
            hidden_states,         # [num_tokens, hidden_dim]
            self.weight_matrix,    # [hidden_dim, num_probes]
        )  # Result: [num_tokens, num_probes]
        
        # Apply activation (must be graph-safe)
        scores = self._apply_activations(scores)
        
        return scores
    
    def _apply_activations(self, scores: torch.Tensor) -> torch.Tensor:
        """Apply per-probe activation functions (graph-safe)"""
        # If all probes use same activation, batch apply
        if len(set(self.activation_fns)) == 1:
            return self._activation_fn(scores, self.activation_fns[0])
        
        # Mixed activations: apply column-wise
        # NOTE: This is graph-safe but slower
        result = scores.clone()
        for i, fn_name in enumerate(self.activation_fns):
            result[:, i] = self._activation_fn(scores[:, i], fn_name)
        return result
    
    @staticmethod
    def _activation_fn(x: torch.Tensor, name: str) -> torch.Tensor:
        """Graph-safe activation functions"""
        if name == "linear":
            return x
        elif name == "sigmoid":
            return torch.sigmoid(x)
        elif name == "tanh":
            return torch.tanh(x)
        else:
            raise ValueError(f"Unknown activation: {name}")

class ProbeWeightManager:
    """Manages all probe weights across all layers"""
    
    def __init__(self, config: OmniscienceConfig):
        self.config = config
        self.layer_weights: dict[int, ProbeWeightSet] = {}
        self._device = config.compute_device
        self._dtype = self._parse_dtype(config.probe_dtype)
        
        self._load_all_weights()
    
    def _parse_dtype(self, dtype_str: str) -> torch.dtype:
        return {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[dtype_str]
    
    def _load_all_weights(self):
        """Group probes by layer and load"""
        layers_with_probes = set(p.layer_idx for p in self.config.probes)
        
        for layer_idx in sorted(layers_with_probes):
            probe_configs = self.config.get_probes_for_layer(layer_idx)
            self.layer_weights[layer_idx] = ProbeWeightSet(
                layer_idx=layer_idx,
                probe_configs=probe_configs,
                device=self._device,
                dtype=self._dtype
            )
        
        logger.info(f"Loaded probes for {len(self.layer_weights)} layers")
    
    def get_layer_weights(self, layer_idx: int) -> ProbeWeightSet | None:
        return self.layer_weights.get(layer_idx)
    
    def has_probes_for_layer(self, layer_idx: int) -> bool:
        return layer_idx in self.layer_weights

