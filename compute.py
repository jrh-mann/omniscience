import torch
import torch.nn as nn
import logging
from .weights import ProbeWeightSet, ProbeWeightManager
from .config import OmniscienceConfig

logger = logging.getLogger(__name__)

class ProbePassthroughLayer(nn.Module):
    """
    CUDA Graph-compatible probe computation layer.
    
    This module is injected into the model's layer stack. It:
    1. Receives hidden_states from previous layer
    2. Computes probe scores
    3. Writes to persistent GPU buffer
    4. Returns hidden_states unchanged (passthrough)
    
    CRITICAL: All operations must be graph-capturable:
    - No Python control flow based on tensor values
    - No dynamic memory allocation
    - No .cpu() / .item() / .numpy() calls
    - Fixed buffer addresses
    """
    
    def __init__(
        self,
        layer_idx: int,
        probe_weights: ProbeWeightSet,  # The nn.Module with weights
        output_buffer: torch.Tensor,    # Pre-allocated [max_tokens, num_probes]
        max_tokens: int,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.probe_weights = probe_weights  # Register as submodule
        self.max_tokens = max_tokens
        
        # Register buffer (persistent GPU tensor with fixed address)
        self.register_buffer("output_buffer", output_buffer, persistent=True)
        
        logger.info(
            f"ProbePassthroughLayer[{layer_idx}]: "
            f"max_tokens={max_tokens}, num_probes={probe_weights.num_probes}"
        )
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Graph-safe forward pass.
        
        Args:
            hidden_states: [num_tokens, hidden_dim] where num_tokens varies
        
        Returns:
            hidden_states: Unchanged (passthrough)
        """
        # Get current batch size (dynamic but graph-safe)
        num_tokens = hidden_states.shape[0]
        
        # Safety check (will compile into graph)
        # If num_tokens > max_tokens, we'll silently truncate (logged elsewhere)
        valid_tokens = min(num_tokens, self.max_tokens)
        
        # Compute probe scores using the weights module
        # This calls ProbeWeightSet.forward()
        # We slice input to valid_tokens to handle cases where batch > max (truncation)
        # or simply to pass correct shape.
        # Note: hidden_states is [num_tokens, hidden_dim]
        
        if valid_tokens > 0:
             probe_scores = self.probe_weights(hidden_states[:valid_tokens])
             # Result: [valid_tokens, num_probes]
             
             # Write to persistent buffer (fixed address = graph-safe)
             # CRITICAL: Using slice assignment, not creating new tensor
             self.output_buffer[:valid_tokens] = probe_scores
        
        # If batch is smaller than max, zero out unused rows to prevent stale data
        # This is important for variable batch sizes in graph replay
        if valid_tokens < self.max_tokens:
            self.output_buffer[valid_tokens:].zero_()
        
        # Return hidden_states unchanged (passthrough)
        return hidden_states

class ProbeBufferManager:
    """
    Manages GPU output buffers and CPU pinned memory for transfers.
    
    Separates concerns:
    - GPU buffers: Written by ProbePassthroughLayer during forward pass
    - CPU buffers: Destination for async transfers after forward pass
    """
    
    def __init__(
        self,
        weight_manager: ProbeWeightManager,
        config: OmniscienceConfig
    ):
        self.config = config
        self._device = config.compute_device
        self._dtype = weight_manager._dtype
        
        # GPU buffers (written during forward pass)
        self.gpu_buffers: dict[int, torch.Tensor] = {}
        
        # CPU pinned buffers (for async transfers)
        self.cpu_pinned_buffers: dict[int, torch.Tensor] = {}
        
        for layer_idx, weights in weight_manager.layer_weights.items():
            # GPU buffer: where ProbePassthroughLayer writes
            gpu_buf = torch.zeros(
                (config.max_batch_size, weights.num_probes),
                device=self._device,
                dtype=self._dtype,
            )
            self.gpu_buffers[layer_idx] = gpu_buf
            
            # CPU pinned buffer: destination for async copy
            # NOTE: pin_memory only works for CPU tensors
            cpu_buf = torch.zeros(
                (config.max_batch_size, weights.num_probes),
                dtype=self._dtype,
            )
            
            if config.use_pinned_memory and torch.cuda.is_available():
                 cpu_buf = cpu_buf.pin_memory()

            self.cpu_pinned_buffers[layer_idx] = cpu_buf
            
            logger.debug(
                f"Allocated buffers for layer {layer_idx}: "
                f"GPU {gpu_buf.shape}, CPU pinned={cpu_buf.is_pinned()}"
            )
    
    def get_gpu_buffer(self, layer_idx: int) -> torch.Tensor:
        """Get GPU buffer for a specific layer"""
        return self.gpu_buffers[layer_idx]
    
    def get_cpu_buffer(self, layer_idx: int) -> torch.Tensor:
        """Get CPU pinned buffer for a specific layer"""
        return self.cpu_pinned_buffers[layer_idx]

class ProbeModuleFactory:
    """Creates ProbePassthroughLayer instances for model injection"""
    
    def __init__(
        self,
        weight_manager: ProbeWeightManager,
        buffer_manager: ProbeBufferManager,
        config: OmniscienceConfig
    ):
        self.weight_manager = weight_manager
        self.buffer_manager = buffer_manager
        self.config = config
    
    def create_probe_layer(self, layer_idx: int) -> ProbePassthroughLayer | None:
        """
        Create a ProbePassthroughLayer for a specific layer.
        
        Returns None if no probes configured for this layer.
        """
        weights = self.weight_manager.get_layer_weights(layer_idx)
        if weights is None:
            return None
        
        gpu_buffer = self.buffer_manager.get_gpu_buffer(layer_idx)
        
        probe_layer = ProbePassthroughLayer(
            layer_idx=layer_idx,
            probe_weights=weights,
            output_buffer=gpu_buffer,
            max_tokens=self.config.max_batch_size
        )
        
        return probe_layer
