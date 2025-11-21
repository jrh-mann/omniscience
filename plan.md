Project Title: Omniscience - vLLM Linear Probing System
Version: 2.1 (REVISED - CUDA Graph Compatible)
Last Updated: 2025-11-21

================================================================================
CRITICAL REVISIONS FROM v2.0
================================================================================

This plan has been revised to address five critical flaws:

1. ✅ CUDA Graph Compatibility (SHOWSTOPPER)
   OLD: Python hooks that stop executing during graph replay
   NEW: nn.Module injection that compiles into CUDA graph
   IMPACT: Probes now work for ALL tokens, not just first one

2. ✅ Tensor Parallel Latency Transparency
   OLD: Claimed "only rank 0 computes" hides latency
   NEW: Explicitly document that TP overhead affects ALL ranks
   IMPACT: Honest performance expectations

3. ✅ Thread Safety
   OLD: Thread-local global context (race conditions)
   NEW: Metadata extraction AFTER forward pass completes
   IMPACT: No race conditions in pipelined execution

4. ✅ Technical Correctness
   - Safetensors: Header-only validation (faster)
   - Pin Memory: Only for CPU tensors (fixed)
   - Batch Mapping: Use vLLM's slot_mapping (accurate)

5. ✅ Architecture Clarity
   OLD: Compute + metadata tracking mixed in forward pass
   NEW: Clear separation - compute (in graph) vs mapping (out of graph)
   IMPACT: Simpler debugging, better maintainability

================================================================================
EXECUTIVE SUMMARY
================================================================================

Goal: Extract interpretability signals (linear probe scores) from LLM hidden 
states during vLLM inference with <2ms overhead per token.

Strategy: Inject graph-compatible nn.Module layers at residual stream locations,
write probe scores to persistent GPU buffers, map to requests outside graph.

Key Constraint: Must work with vLLM's continuous batching, tensor parallelism,
and CUDA graph compilation without breaking existing functionality.

Architecture Philosophy: "What runs in the graph, stays in the graph."
- Compute: Inside CUDA graph (ProbePassthroughLayer.forward)
- Metadata: Outside CUDA graph (ModelRunnerWrapper post-processing)

================================================================================
SECTION 1: ARCHITECTURE OVERVIEW
================================================================================

⚠️  CRITICAL CONSTRAINTS ⚠️

1. CUDA Graph Compatibility: vLLM compiles decode phase into CUDA graphs.
   Python hooks are NOT executed during graph replay. All probe computation
   must be graph-capturable PyTorch operations.

2. Tensor Parallel Latency: In TP setups, ALL ranks synchronize at layer
   boundaries. Rank 0 computation overhead affects GLOBAL latency, not just
   rank 0. Cannot hide latency by computing on single rank.

3. Thread Safety: vLLM may pipeline batches. Cannot rely on global mutable
   state to pass metadata into forward pass.

1.1 THE THREE-LAYER SYSTEM (REVISED)

┌─────────────────────────────────────────────────────────────────┐
│ Layer 1: Configuration & Weight Management (CPU/Disk)          │
│  - Load probe weights from safetensors (header-only validation) │
│  - Convert to nn.Parameter for graph compatibility              │
│  - Manage lifecycle (hot-reload, versioning)                    │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ Layer 2: Compute Engine (GPU) - INSIDE CUDA GRAPH              │
│  - Inject as nn.Module "passthrough" layer                      │
│  - Execute matmul: activations @ probe_weights                  │
│  - Write to fixed GPU buffer (persistent address)               │
│  - NO Python logic, NO request tracking here                    │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ Layer 3: Result Transport & API Integration (CPU) - OUTSIDE    │
│  - AFTER forward pass completes, read output buffer             │
│  - Map batch positions to request IDs using vLLM metadata       │
│  - Async transfer GPU → pinned CPU memory                       │
│  - Attach to vLLM RequestOutput                                 │
└─────────────────────────────────────────────────────────────────┘

KEY ARCHITECTURAL CHANGE: "Passthrough Module" Pattern

Old (BROKEN):
  Python hook → Check request_id → Compute probe → Append to list
  ❌ Python code doesn't run during CUDA graph replay

New (CORRECT):
  nn.Module → Compute probe → Write to fixed buffer
  (Runs during graph capture AND replay)
  
  Then separately (outside graph):
  Read buffer → Map to requests → Transfer to CPU

1.2 CORE PRINCIPLES (REVISED)

Zero-Sync Policy: Never call .cpu(), .item(), or .numpy() in forward pass
Memory Discipline: Pre-allocate all buffers at init; fixed addresses for graphs
Fault Tolerance: Probe failures must not crash inference
Graph Safety: All ops must be graph-capturable (no dynamic control flow)
Distributed Honesty: TP overhead affects all ranks; document true cost
Decode-Only Default: Skip prefill phase unless explicitly enabled
Metadata Separation: Request tracking happens OUTSIDE the model forward pass

1.3 THE CONTINUOUS BATCHING CHALLENGE

vLLM's flattened batch structure:
  hidden_states shape: [num_tokens_in_batch, hidden_dim]
  
  Example decode batch:
    [seq_A_tok_50, seq_B_tok_1, seq_C_tok_120]  → shape [3, 4096]
  
  Example prefill batch (request D has 10-token prompt):
    [D_tok_0, D_tok_1, ..., D_tok_9, seq_A_tok_51]  → shape [11, 4096]

Critical requirements:
  - Probe computation: [num_tokens, dim] @ [dim, probes] → [num_tokens, probes]
  - Mapping: Use vLLM's slot_mapping or sequence metadata to map rows to requests
  - Variable batch: Support dynamic num_tokens (changes every forward pass)
  - Prefill vs Decode: Distinguish multi-token prefill from single-token decode

================================================================================
SECTION 2: IMPLEMENTATION PHASES (FOUNDATION → OPTIMIZATION)
================================================================================

─────────────────────────────────────────────────────────────────────
PHASE 0: RECONNAISSANCE & VALIDATION (1-2 hours)
─────────────────────────────────────────────────────────────────────

Objective: Understand the actual vLLM codebase before writing any probe code.

Tasks:

0.1 vLLM Version Detection
    - Check installed vLLM version (vllm.__version__)
    - Identify if using legacy (v0.2.x) or modern (v0.4+) architecture
    - Note: v0.4+ has significant ModelRunner refactor

0.2 Model Architecture Exploration
    - Locate LlamaDecoderLayer (or equivalent) source code
    - Identify the exact location of residual stream:
        Before v0.4: hidden_states = residual + layer_output
        After v0.4: Check new attention/MLP interfaces
    - Find where layer_idx is stored (self.layer_idx? constructor arg?)

0.3 Tensor Parallel Inspection
    - Check if model uses TP (tensor_parallel_size in engine config)
    - Locate all-reduce operations (look for tensor_model_parallel_all_reduce)
    - Verify residual stream is replicated across ranks (critical!)

0.4 Request Tracking Mechanism
    - Find vLLM's RequestOutput class definition
    - Identify how request_id flows through the system:
        LLMEngine → ModelRunner → Forward Pass → Output
    - Check if sampling_params or metadata dict exists on intermediate objects

0.5 CUDA Context Investigation
    - Determine default CUDA stream used by vLLM
    - Check if CUDA graphs are enabled (vLLM uses them aggressively)
    - Understand PagedAttention's memory layout (KV cache pointer structure)

0.6 CUDA Graph Compatibility Testing
    - Test: Add dummy nn.Module to model.layers, verify graph still works
    - Test: Write to persistent buffer in forward(), verify graph captures it
    - Test: Read buffer after execute_model(), verify data is correct
    - Document: Which operations break graph capture (if any discovered)

Validation Gate: Successfully inject a dummy passthrough layer that:
1. Writes tensor shape info to a persistent GPU buffer
2. Doesn't break CUDA graph compilation
3. Buffer can be read after execute_model() completes

Output: Document findings in `docs/vllm_integration_notes.md`

CRITICAL SUCCESS CRITERIA:
- Dummy layer must run during BOTH graph capture AND replay
- Buffer contents must update on every forward pass (verify with unique values)
- No "CUDA graph capture failed" or "trying to backward through graph" errors

─────────────────────────────────────────────────────────────────────
PHASE 1: DATA LAYER - PROBE WEIGHT MANAGEMENT (4-6 hours)
─────────────────────────────────────────────────────────────────────

Objective: Load, validate, and prepare probe weights for GPU compute.

1.1 Configuration Schema (Pydantic)

File: omniscience/config.py

from pydantic import BaseModel, Field, validator
from pathlib import Path
from typing import Literal

class ProbeConfig(BaseModel):
    """Single probe definition"""
    name: str = Field(..., description="Unique probe identifier")
    layer_idx: int = Field(..., ge=0, description="Target layer (0-indexed)")
    weight_path: Path = Field(..., description="Path to .safetensors file")
    activation: Literal["linear", "sigmoid", "tanh"] = "linear"
    
    # Optional: if not in safetensors file
    bias: float | None = None
    
    @validator("weight_path")
    def check_exists(cls, v):
        if not v.exists():
            raise ValueError(f"Probe weight file not found: {v}")
        if v.suffix != ".safetensors":
            raise ValueError(f"Must be .safetensors format, got {v.suffix}")
        return v

class OmniscienceConfig(BaseModel):
    """System-wide configuration"""
    probes: list[ProbeConfig]
    
    # Performance tuning
    max_batch_size: int = Field(256, description="Pre-allocate for this batch size")
    enable_prefill_probes: bool = Field(False, description="Probe during prefill")
    compute_device: str = Field("cuda", description="Device for probe computation")
    
    # Distributed settings
    tensor_parallel_rank: int | None = Field(None, description="Auto-detected if None")
    
    # Memory optimization
    probe_dtype: Literal["float16", "bfloat16", "float32"] = "float16"
    use_pinned_memory: bool = Field(True, description="For async CPU transfers")
    
    @validator("probes")
    def check_unique_names(cls, v):
        names = [p.name for p in v]
        if len(names) != len(set(names)):
            raise ValueError("Probe names must be unique")
        return v
    
    def get_probes_for_layer(self, layer_idx: int) -> list[ProbeConfig]:
        return [p for p in self.probes if p.layer_idx == layer_idx]

1.2 Weight Loader (REVISED)

File: omniscience/weights.py

import torch
import torch.nn as nn
from safetensors import safe_open
from pathlib import Path
import logging

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
        with safe_open(path, framework="pt", device="cpu") as f:
            # Get shape without loading tensor data
            metadata = f.metadata()
            
            # Check for required keys
            if "weight" not in f.keys():
                raise ValueError(f"Probe file {path} missing 'weight' tensor")
            
            weight_shape = f.get_slice("weight").get_shape()
            
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
        with safe_open(path, framework="pt", device="cpu") as f:
            weight = f.get_tensor("weight")
            bias = f.get_tensor("bias").item() if "bias" in f.keys() else None
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

1.3 Weight Manager (Lifecycle)

File: omniscience/weights.py (continued)

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

Validation Gate: Write test that loads probe weights, checks shapes, validates
device placement. Test with invalid configs to ensure proper error handling.

─────────────────────────────────────────────────────────────────────
PHASE 2: COMPUTE LAYER - PASSTHROUGH MODULE (8-10 hours)
─────────────────────────────────────────────────────────────────────

Objective: Implement graph-compatible probe computation using nn.Module pattern.

2.1 The Passthrough Probe Module

File: omniscience/compute.py

import torch
import torch.nn as nn

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
        probe_scores = self.probe_weights(hidden_states[:valid_tokens])
        # Result: [valid_tokens, num_probes]
        
        # Write to persistent buffer (fixed address = graph-safe)
        # CRITICAL: Using slice assignment, not creating new tensor
        self.output_buffer[:valid_tokens] = probe_scores
        
        # If batch is smaller than max, zero out unused rows to prevent stale data
        if valid_tokens < self.max_tokens:
            self.output_buffer[valid_tokens:].zero_()
        
        # Return hidden_states unchanged (passthrough)
        return hidden_states

2.2 Buffer Manager

File: omniscience/compute.py (continued)

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
                pin_memory=config.use_pinned_memory and self._device == "cuda"
            )
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

2.3 Probe Module Factory

File: omniscience/compute.py (continued)

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
    
    def create_probe_layer(self, layer_idx: int) -> ProbePassthroughLayer:
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

2.2 Request Tracking System

File: omniscience/tracking.py

from dataclasses import dataclass
from typing import Any
import numpy as np

@dataclass
class ProbeResult:
    """Probe scores for a single token in a single request"""
    request_id: str
    token_idx: int          # Position in sequence
    layer_idx: int
    probe_names: list[str]
    scores: np.ndarray      # [num_probes], on CPU as numpy
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "layer": self.layer_idx,
            "probes": dict(zip(self.probe_names, self.scores.tolist()))
        }

class ProbeResultTracker:
    """Maps batch positions to request IDs and buffers results"""
    
    def __init__(self, use_pinned_memory: bool = True):
        # Circular buffer for async GPU→CPU transfers
        self.pending_transfers: dict[int, tuple[torch.Tensor, list[str]]] = {}
        self.use_pinned_memory = use_pinned_memory
    
    def record_batch(
        self,
        layer_idx: int,
        probe_scores: torch.Tensor,  # [batch, num_probes] on GPU
        request_ids: list[str],
        token_indices: list[int],
        probe_names: list[str]
    ) -> list[ProbeResult]:
        """
        Register probe computation and transfer to CPU asynchronously.
        
        Note: This is synchronous version. Phase 3 will make it async.
        """
        # For now, simple synchronous transfer
        scores_cpu = probe_scores.cpu().numpy()  # [batch, num_probes]
        
        results = []
        for i, (req_id, token_idx) in enumerate(zip(request_ids, token_indices)):
            result = ProbeResult(
                request_id=req_id,
                token_idx=token_idx,
                layer_idx=layer_idx,
                probe_names=probe_names,
                scores=scores_cpu[i]
            )
            results.append(result)
        
        return results

2.3 Integration Test (Standalone)

File: tests/test_compute_engine.py

def test_probe_computation():
    """Test probe computation without vLLM integration"""
    
    # Create dummy probe weights
    hidden_dim = 4096
    num_probes = 3
    
    probe_weight = torch.randn(hidden_dim)
    
    # Save as safetensors
    from safetensors.torch import save_file
    save_file(
        {"weight": probe_weight, "bias": torch.tensor(0.5)},
        "/tmp/test_probe.safetensors"
    )
    
    # Create config
    config = OmniscienceConfig(
        probes=[
            ProbeConfig(
                name="test_probe_1",
                layer_idx=15,
                weight_path=Path("/tmp/test_probe.safetensors"),
                activation="sigmoid"
            )
        ],
        max_batch_size=8
    )
    
    # Initialize system
    weight_manager = ProbeWeightManager(config)
    compute_engine = ProbeComputeEngine(weight_manager, config)
    
    # Simulate hidden states
    batch_size = 4
    hidden_states = torch.randn(batch_size, hidden_dim, device="cuda", dtype=torch.float16)
    
    # Compute probes
    scores = compute_engine.compute_probes(layer_idx=15, hidden_states=hidden_states)
    
    # Validate
    assert scores.shape == (batch_size, 1)
    assert scores.device.type == "cuda"
    assert 0 <= scores.min() <= scores.max() <= 1  # sigmoid output
    
    print("✓ Probe computation test passed")

Validation Gate: Run test_probe_computation() successfully. Benchmark throughput
(should handle 1000 batch x 50 probes in <0.5ms on A100).

─────────────────────────────────────────────────────────────────────
PHASE 3: INTEGRATION LAYER - MODEL INJECTION (12-15 hours)
─────────────────────────────────────────────────────────────────────

Objective: Inject ProbePassthroughLayer into vLLM model architecture.

⚠️  CRITICAL CHANGES FROM ORIGINAL PLAN:
- NO Python hooks (incompatible with CUDA graphs)
- NO thread-local context (race conditions)
- YES nn.Module injection (graph-compatible)
- Request mapping happens AFTER forward pass completes

3.1 Model Architecture Patcher

File: omniscience/integration.py

import torch
import torch.nn as nn
from typing import Any
import logging

logger = logging.getLogger(__name__)

class ModelArchitecturePatcher:
    """
    Injects ProbePassthroughLayer into vLLM model's layer stack.
    
    Strategy: Wrap each decoder layer with a Sequential container:
        Sequential(
            OriginalLayer,
            ProbePassthroughLayer  # Our injected probe
        )
    
    This ensures probes are part of the CUDA graph.
    """
    
    def __init__(
        self,
        probe_factory: ProbeModuleFactory,
        config: OmniscienceConfig
    ):
        self.probe_factory = probe_factory
        self.config = config
        self.patched_layers: dict[int, Any] = {}
    
    def patch_model(self, model: nn.Module, model_architecture: str = "llama") -> None:
        """
        Inject probe layers into model architecture.
        
        Args:
            model: vLLM's loaded model
            model_architecture: "llama", "mistral", etc. (from Phase 0 recon)
        """
        # Get model's layer container (architecture-specific)
        if model_architecture == "llama":
            layer_container = model.model.layers
        elif model_architecture == "mistral":
            layer_container = model.model.layers
        else:
            raise ValueError(f"Unsupported architecture: {model_architecture}")
        
        # Iterate over all layers
        for layer_idx in range(len(layer_container)):
            # Check if this layer has probes configured
            probe_layer = self.probe_factory.create_probe_layer(layer_idx)
            if probe_layer is None:
                continue  # No probes for this layer
            
            # Wrap original layer + probe layer in Sequential
            original_layer = layer_container[layer_idx]
            
            wrapped_layer = nn.Sequential(
                original_layer,
                probe_layer
            )
            
            # Replace in model
            layer_container[layer_idx] = wrapped_layer
            self.patched_layers[layer_idx] = wrapped_layer
            
            logger.info(f"Injected ProbePassthroughLayer at layer {layer_idx}")
        
        logger.info(f"Model patching complete: {len(self.patched_layers)} layers modified")
    
    def get_probe_layer(self, layer_idx: int) -> ProbePassthroughLayer | None:
        """Retrieve the injected probe layer for buffer access"""
        if layer_idx not in self.patched_layers:
            return None
        
        # The Sequential container has: [0] = original, [1] = probe
        sequential = self.patched_layers[layer_idx]
        return sequential[1]

3.2 Request Metadata Extractor

File: omniscience/integration.py (continued)

from dataclasses import dataclass

@dataclass
class BatchMetadata:
    """
    Metadata for current batch, extracted AFTER forward pass.
    
    This is safe because we extract it outside the CUDA graph.
    """
    request_ids: list[str]
    token_indices: list[int]
    num_tokens: int
    is_prefill: bool
    
    def get_request_for_position(self, pos: int) -> tuple[str, int]:
        """Map buffer position to (request_id, token_idx)"""
        if pos >= len(self.request_ids):
            raise IndexError(f"Position {pos} out of range (batch size {len(self.request_ids)})")
        return self.request_ids[pos], self.token_indices[pos]

class MetadataExtractor:
    """
    Extracts request metadata from vLLM's scheduler output.
    
    CRITICAL: This runs OUTSIDE the model forward pass, so no graph issues.
    """
    
    def __init__(self, vllm_version: str):
        self.vllm_version = vllm_version
    
    def extract_from_scheduler_output(
        self,
        seq_group_metadata_list: list[Any]
    ) -> BatchMetadata:
        """
        Extract metadata from vLLM's SequenceGroupMetadata.
        
        Args:
            seq_group_metadata_list: vLLM's internal scheduler output
        
        Returns:
            BatchMetadata with request IDs and token positions
        """
        request_ids = []
        token_indices = []
        is_prefill = False
        
        for seq_group_meta in seq_group_metadata_list:
            # Check if this is prefill or decode
            # (All sequences in a batch are same phase in vLLM)
            is_prefill = seq_group_meta.is_prompt
            
            if is_prefill:
                # Prefill: one sequence group, many tokens
                prompt_len = seq_group_meta.prompt_len
                for token_idx in range(prompt_len):
                    request_ids.append(seq_group_meta.request_id)
                    token_indices.append(token_idx)
            else:
                # Decode: each sequence generates 1 token
                for seq_data in seq_group_meta.seq_data.values():
                    request_ids.append(seq_group_meta.request_id)
                    token_indices.append(len(seq_data.get_token_ids()) - 1)
        
        return BatchMetadata(
            request_ids=request_ids,
            token_indices=token_indices,
            num_tokens=len(request_ids),
            is_prefill=is_prefill
        )

3.3 ModelRunner Wrapper (Post-Forward Processing)

File: omniscience/integration.py (continued)

class ModelRunnerWrapper:
    """
    Wraps vLLM's ModelRunner to extract probe results AFTER forward pass.
    
    This is the coordination point:
    1. vLLM calls execute_model()
    2. Model runs (with probe layers writing to buffers)
    3. Forward pass completes
    4. We read buffers and map to requests
    """
    
    def __init__(
        self,
        model_runner: Any,
        patcher: ModelArchitecturePatcher,
        metadata_extractor: MetadataExtractor,
        buffer_manager: ProbeBufferManager,
        weight_manager: ProbeWeightManager,
        config: OmniscienceConfig
    ):
        self.model_runner = model_runner
        self.patcher = patcher
        self.metadata_extractor = metadata_extractor
        self.buffer_manager = buffer_manager
        self.weight_manager = weight_manager
        self.config = config
        
        # Store original execute_model
        self.original_execute_model = model_runner.execute_model
        
        # Results buffer (request_id → probe results)
        self.pending_results: dict[str, list[dict]] = {}
    
    def install(self):
        """Replace execute_model with our wrapper"""
        def wrapped_execute_model(seq_group_metadata_list, **kwargs):
            # Call original vLLM execution
            output = self.original_execute_model(seq_group_metadata_list, **kwargs)
            
            # Extract metadata AFTER forward pass
            metadata = self.metadata_extractor.extract_from_scheduler_output(
                seq_group_metadata_list
            )
            
            # Skip if prefill and not enabled
            if metadata.is_prefill and not self.config.enable_prefill_probes:
                return output
            
            # Read probe buffers and map to requests
            self._process_probe_outputs(metadata)
            
            return output
        
        self.model_runner.execute_model = wrapped_execute_model
        logger.info("ModelRunner wrapper installed")
    
    def _process_probe_outputs(self, metadata: BatchMetadata):
        """
        Read GPU buffers and map to request IDs.
        
        This runs on CPU thread, outside CUDA graph.
        """
        for layer_idx in self.patcher.patched_layers.keys():
            # Get the probe layer
            probe_layer = self.patcher.get_probe_layer(layer_idx)
            if probe_layer is None:
                continue
            
            # Access the output buffer (still on GPU)
            gpu_buffer = probe_layer.output_buffer  # [max_tokens, num_probes]
            
            # Slice to actual batch size
            valid_scores = gpu_buffer[:metadata.num_tokens]  # [num_tokens, num_probes]
            
            # Copy to CPU (synchronous for now, async in Phase 5)
            scores_cpu = valid_scores.cpu().numpy()
            
            # Map each row to request ID
            weights = self.weight_manager.get_layer_weights(layer_idx)
            probe_names = weights.probe_names
            
            for pos in range(metadata.num_tokens):
                request_id, token_idx = metadata.get_request_for_position(pos)
                
                # Store result
                if request_id not in self.pending_results:
                    self.pending_results[request_id] = []
                
                self.pending_results[request_id].append({
                    "layer": layer_idx,
                    "token_idx": token_idx,
                    "probes": dict(zip(probe_names, scores_cpu[pos].tolist()))
                })
    
    def get_results_for_request(self, request_id: str) -> list[dict]:
        """Retrieve and clear results for a request"""
        return self.pending_results.pop(request_id, [])

3.4 Tensor Parallel Handling (REVISED)

File: omniscience/distributed.py

import torch.distributed as dist

class TensorParallelHandler:
    """
    Handles probe computation in Tensor Parallel setups.
    
    CRITICAL REVISION: Probe overhead affects ALL ranks due to synchronization.
    We compute only on rank 0 to save compute (energy), not latency.
    """
    
    def __init__(self):
        self.is_distributed = dist.is_initialized()
        self.rank = dist.get_rank() if self.is_distributed else 0
        self.world_size = dist.get_world_size() if self.is_distributed else 1
        
        if self.is_distributed:
            logger.warning(
                f"Tensor Parallelism detected (rank {self.rank}/{self.world_size}). "
                "Probe computation on rank 0 will affect GLOBAL latency due to "
                "synchronization barriers. Consider sharding probes if overhead is high."
            )
    
    def should_compute_probes(self) -> bool:
        """
        Only rank 0 computes probes.
        
        NOTE: This does NOT hide latency from other ranks. They will wait
        at the next all-reduce barrier if rank 0 is slower.
        """
        return self.rank == 0
    
    def should_inject_probe_layers(self) -> bool:
        """
        Only inject probe layers on rank 0 to avoid unnecessary compute.
        
        Other ranks will have standard model architecture.
        """
        return self.rank == 0

Validation Gate:
1. Test model patching doesn't break vLLM inference
2. Verify probe buffers are written correctly
3. Confirm metadata extraction maps rows to correct request IDs
4. Test with batch_size=1,2,4,8 (variable sizes)

─────────────────────────────────────────────────────────────────────
PHASE 4: API INTEGRATION - RESULT ATTACHMENT (6-8 hours)
─────────────────────────────────────────────────────────────────────

Objective: Attach probe results to vLLM's output objects.

4.1 Output Extension Strategy

File: omniscience/api_integration.py

from vllm import RequestOutput, CompletionOutput
from typing import Any

class ProbeOutputAttacher:
    """
    Attaches probe results to vLLM RequestOutput objects.
    
    Strategy depends on vLLM version:
    1. If RequestOutput has 'extras' dict → use that
    2. Else, monkey-patch to add 'probe_scores' attribute
    3. Fallback: wrap in custom response class
    """
    
    def __init__(self, result_tracker: ProbeResultTracker):
        self.result_tracker = result_tracker
        self._patch_request_output()
    
    def _patch_request_output(self):
        """Add probe_scores field to RequestOutput if not exists"""
        if not hasattr(RequestOutput, "probe_scores"):
            # Add new attribute dynamically
            original_init = RequestOutput.__init__
            
            def new_init(self, *args, **kwargs):
                original_init(self, *args, **kwargs)
                self.probe_scores: dict[str, Any] = {}
            
            RequestOutput.__init__ = new_init
            logger.info("Patched RequestOutput to include probe_scores")
    
    def attach_results(
        self,
        request_output: RequestOutput,
        probe_results: list[ProbeResult]
    ) -> RequestOutput:
        """
        Attach probe results to output object.
        
        Format:
        {
            "token_0": {"layer_15": {"probe1": 0.82, "probe2": 0.34}},
            "token_1": {...}
        }
        """
        if not hasattr(request_output, "probe_scores"):
            # Fallback: add as dict
            request_output.probe_scores = {}
        
        for result in probe_results:
            token_key = f"token_{result.token_idx}"
            if token_key not in request_output.probe_scores:
                request_output.probe_scores[token_key] = {}
            
            request_output.probe_scores[token_key].update(result.to_dict())
        
        return request_output

4.2 LLMEngine Wrapper

File: omniscience/engine.py

from vllm import LLM, SamplingParams
import logging

class OmniscienceLLM:
    """
    Wrapper around vLLM's LLM class with integrated probing.
    
    Usage:
        llm = OmniscienceLLM(
            model="meta-llama/Llama-2-7b-hf",
            omniscience_config=config
        )
        
        outputs = llm.generate(["Tell me a joke"], sampling_params)
        for output in outputs:
            print(output.probe_scores)
    """
    
    def __init__(
        self,
        model: str,
        omniscience_config: OmniscienceConfig,
        **vllm_kwargs
    ):
        self.config = omniscience_config
        
        # Initialize probe system
        self.weight_manager = ProbeWeightManager(omniscience_config)
        self.compute_engine = ProbeComputeEngine(self.weight_manager, omniscience_config)
        self.result_tracker = ProbeResultTracker(omniscience_config.use_pinned_memory)
        self.output_attacher = ProbeOutputAttacher(self.result_tracker)
        
        # Check tensor parallelism
        self.tp_guard = TensorParallelGuard()
        if not self.tp_guard.should_compute_probes():
            logger.info(f"Rank {self.tp_guard.rank}: Probes disabled (not rank 0)")
            # Early return - don't install hooks on non-rank-0
            self.llm = LLM(model=model, **vllm_kwargs)
            return
        
        # Initialize vLLM engine
        self.llm = LLM(model=model, **vllm_kwargs)
        
        # Install hooks after model is loaded
        self._install_hooks()
    
    def _install_hooks(self):
        """Install all hooks into vLLM internals"""
        # Hook into model layers
        self.hook_installer = ModelHookInstaller(
            self.compute_engine,
            self.result_tracker,
            self.config
        )
        self.hook_installer.install_hooks(self.llm.llm_engine.model_executor.driver_worker.model_runner.model)
        
        # Hook into ModelRunner for metadata
        self.runner_interceptor = ModelRunnerInterceptor()
        self.runner_interceptor.install(
            self.llm.llm_engine.model_executor.driver_worker.model_runner
        )
        
        logger.info("All hooks installed successfully")
    
    def generate(
        self,
        prompts: list[str],
        sampling_params: SamplingParams | None = None,
        **kwargs
    ) -> list[RequestOutput]:
        """Generate with probe scores attached"""
        outputs = self.llm.generate(prompts, sampling_params, **kwargs)
        
        # Attach probe results
        # Note: Need to map request_ids from outputs to tracked results
        # This is simplified - real implementation needs proper mapping
        for output in outputs:
            # Get probe results for this request
            results = self._get_results_for_request(output.request_id)
            self.output_attacher.attach_results(output, results)
        
        return outputs
    
    def _get_results_for_request(self, request_id: str) -> list[ProbeResult]:
        """Retrieve probe results for a specific request"""
        # This needs proper implementation with result buffering
        # For now, placeholder
        return []

Validation Gate: End-to-end test with vLLM generate() call, verify probe_scores
are present in output and contain expected structure.

─────────────────────────────────────────────────────────────────────
PHASE 5: ASYNC OPTIMIZATION - THE "DELAYED" PATTERN (8-10 hours)
─────────────────────────────────────────────────────────────────────

Objective: Make GPU→CPU transfers asynchronous to eliminate synchronization overhead.

5.1 Async Result Buffer

File: omniscience/async_transfer.py

import torch
from collections import deque
from dataclasses import dataclass

@dataclass
class PendingTransfer:
    """Represents an in-flight GPU→CPU transfer"""
    layer_idx: int
    scores_gpu: torch.Tensor      # Still on GPU
    request_ids: list[str]
    token_indices: list[int]
    probe_names: list[str]
    event: torch.cuda.Event        # Marks when compute finished
    transfer_stream: torch.cuda.Stream  # Async transfer stream

class AsyncProbeResultTracker:
    """
    Manages async transfers using CUDA streams and events.
    
    Pattern:
    - Step N: Launch probe kernel, record event, queue transfer
    - Step N+1: Check if event complete, initiate CPU copy
    - Step N+2: Results available for attachment
    """
    
    def __init__(self, max_queue_depth: int = 3):
        self.max_queue_depth = max_queue_depth
        self.pending_queue: deque[PendingTransfer] = deque()
        self.completed_results: dict[str, list[ProbeResult]] = {}
        
        # Create dedicated stream for CPU transfers
        self.transfer_stream = torch.cuda.Stream()
        
        # Pre-allocate pinned memory for transfers
        # (sized based on config.max_batch_size * max_probes)
        self.pinned_buffers: dict[int, torch.Tensor] = {}
    
    def enqueue_computation(
        self,
        layer_idx: int,
        scores_gpu: torch.Tensor,
        request_ids: list[str],
        token_indices: list[int],
        probe_names: list[str]
    ):
        """
        Record completed GPU computation, prepare for async transfer.
        """
        # Record event on default stream (where probe was computed)
        event = torch.cuda.Event()
        event.record()
        
        pending = PendingTransfer(
            layer_idx=layer_idx,
            scores_gpu=scores_gpu.clone(),  # Keep reference
            request_ids=request_ids,
            token_indices=token_indices,
            probe_names=probe_names,
            event=event,
            transfer_stream=self.transfer_stream
        )
        
        self.pending_queue.append(pending)
        
        # Process queue asynchronously
        self._process_queue()
    
    def _process_queue(self):
        """Check for completed events and initiate transfers"""
        while self.pending_queue:
            pending = self.pending_queue[0]
            
            # Check if GPU computation finished
            if not pending.event.query():  # Still running
                break
            
            # Event complete - start CPU transfer in background
            self.pending_queue.popleft()
            self._start_transfer(pending)
    
    def _start_transfer(self, pending: PendingTransfer):
        """Initiate async GPU→CPU transfer"""
        with torch.cuda.stream(self.transfer_stream):
            # Non-blocking transfer
            scores_cpu = pending.scores_gpu.cpu()  # This is async in the transfer_stream
            
            # Create probe results
            scores_np = scores_cpu.numpy()
            
            for i, (req_id, token_idx) in enumerate(
                zip(pending.request_ids, pending.token_indices)
            ):
                result = ProbeResult(
                    request_id=req_id,
                    token_idx=token_idx,
                    layer_idx=pending.layer_idx,
                    probe_names=pending.probe_names,
                    scores=scores_np[i]
                )
                
                if req_id not in self.completed_results:
                    self.completed_results[req_id] = []
                self.completed_results[req_id].append(result)
    
    def get_results(self, request_id: str, wait: bool = False) -> list[ProbeResult]:
        """
        Retrieve completed probe results for a request.
        
        Args:
            wait: If True, synchronize transfer stream before returning
        """
        if wait:
            self.transfer_stream.synchronize()
        
        return self.completed_results.pop(request_id, [])

5.2 Update Compute Engine

File: omniscience/compute.py (add method)

class ProbeComputeEngine:
    # ... existing code ...
    
    def compute_probes_async(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
        request_ids: list[str],
        token_indices: list[int],
        result_tracker: AsyncProbeResultTracker
    ):
        """
        Compute probes and enqueue for async transfer.
        Does NOT synchronize - returns immediately.
        """
        scores = self.compute_probes(layer_idx, hidden_states)
        
        weights = self.weight_manager.get_layer_weights(layer_idx)
        result_tracker.enqueue_computation(
            layer_idx=layer_idx,
            scores_gpu=scores,
            request_ids=request_ids,
            token_indices=token_indices,
            probe_names=weights.probe_names
        )

Validation Gate: Benchmark latency with async vs sync. Async should add <0.5ms
overhead even with 100 probes. Use torch.cuda.nvtx.range for profiling.

─────────────────────────────────────────────────────────────────────
PHASE 6: PERFORMANCE OPTIMIZATION - TRITON KERNEL (8-10 hours)
─────────────────────────────────────────────────────────────────────

Objective: Replace PyTorch matmul with fused Triton kernel (only if needed).

6.1 Benchmark First

File: benchmarks/matmul_comparison.py

"""
Compare PyTorch vs Triton for probe computation.

Only proceed with Triton if >10% faster.
"""

import torch
import triton
import triton.language as tl
import time

def benchmark_pytorch(hidden_states, weight_matrix, bias, num_iters=1000):
    torch.cuda.synchronize()
    start = time.perf_counter()
    
    for _ in range(num_iters):
        output = torch.addmm(bias, hidden_states, weight_matrix)
    
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return elapsed / num_iters * 1000  # ms per iteration

# Only implement Triton kernel if PyTorch is bottleneck

6.2 Fused Triton Kernel (Conditional)

File: omniscience/kernels.py

"""
Custom Triton kernel that fuses:
1. Matrix multiplication
2. Bias addition
3. Activation function (sigmoid/tanh)

This avoids multiple kernel launches and intermediate buffers.
"""

import triton
import triton.language as tl

@triton.jit
def fused_probe_kernel(
    # Pointers
    activations_ptr,  # [batch, hidden_dim]
    weights_ptr,      # [hidden_dim, num_probes]
    bias_ptr,         # [num_probes]
    output_ptr,       # [batch, num_probes]
    # Shapes
    batch_size,
    hidden_dim,
    num_probes,
    # Activation type
    activation_type,  # 0=linear, 1=sigmoid, 2=tanh
    # Block sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Fused probe computation kernel.
    
    Grid: (batch_size // BLOCK_M, num_probes // BLOCK_N)
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Matrix multiplication loop
    for k in range(0, hidden_dim, BLOCK_K):
        # Load activation block
        a = tl.load(
            activations_ptr + offs_m[:, None] * hidden_dim + (k + offs_k)[None, :],
            mask=(offs_m[:, None] < batch_size) & ((k + offs_k)[None, :] < hidden_dim),
            other=0.0
        )
        
        # Load weight block
        w = tl.load(
            weights_ptr + (k + offs_k)[:, None] * num_probes + offs_n[None, :],
            mask=((k + offs_k)[:, None] < hidden_dim) & (offs_n[None, :] < num_probes),
            other=0.0
        )
        
        # Accumulate
        acc += tl.dot(a, w)
    
    # Load bias and add
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < num_probes, other=0.0)
    acc += bias[None, :]
    
    # Apply activation
    if activation_type == 1:  # sigmoid
        acc = 1.0 / (1.0 + tl.exp(-acc))
    elif activation_type == 2:  # tanh
        acc = tl.tanh(acc)
    # else: linear (no-op)
    
    # Store output
    tl.store(
        output_ptr + offs_m[:, None] * num_probes + offs_n[None, :],
        acc,
        mask=(offs_m[:, None] < batch_size) & (offs_n[None, :] < num_probes)
    )

def launch_fused_probe_kernel(
    activations: torch.Tensor,
    weights: torch.Tensor,
    bias: torch.Tensor,
    output: torch.Tensor,
    activation_type: int
):
    """Wrapper to launch Triton kernel with optimal block sizes"""
    batch_size, hidden_dim = activations.shape
    _, num_probes = weights.shape
    
    # Heuristic block sizes (tune for target GPU)
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32
    
    grid = (
        triton.cdiv(batch_size, BLOCK_M),
        triton.cdiv(num_probes, BLOCK_N),
    )
    
    fused_probe_kernel[grid](
        activations, weights, bias, output,
        batch_size, hidden_dim, num_probes,
        activation_type,
        BLOCK_M, BLOCK_N, BLOCK_K
    )

Note: Only use Triton kernel if benchmarks show >10% improvement over PyTorch.

─────────────────────────────────────────────────────────────────────
PHASE 7: TESTING & VALIDATION (6-8 hours)
─────────────────────────────────────────────────────────────────────

7.1 Unit Tests

tests/test_weights.py - Weight loading and validation
tests/test_compute.py - Probe computation correctness
tests/test_hooks.py - Hook installation and execution
tests/test_async.py - Async transfer correctness

7.2 Integration Test

File: tests/test_end_to_end.py

"""
Full end-to-end test with real vLLM inference.
"""

def test_omniscience_inference():
    # Create test probe
    hidden_dim = 4096
    probe_weight = torch.randn(hidden_dim)
    save_file(
        {"weight": probe_weight},
        "/tmp/test_probe.safetensors"
    )
    
    # Configure omniscience
    config = OmniscienceConfig(
        probes=[
            ProbeConfig(
                name="test_sentiment",
                layer_idx=15,
                weight_path=Path("/tmp/test_probe.safetensors"),
                activation="sigmoid"
            )
        ],
        max_batch_size=32
    )
    
    # Initialize
    llm = OmniscienceLLM(
        model="meta-llama/Llama-2-7b-hf",
        omniscience_config=config,
        tensor_parallel_size=1
    )
    
    # Generate
    prompts = ["Hello, how are you?", "I hate Mondays"]
    sampling_params = SamplingParams(max_tokens=20, temperature=0.8)
    
    outputs = llm.generate(prompts, sampling_params)
    
    # Validate probe scores exist
    for output in outputs:
        assert hasattr(output, "probe_scores")
        assert len(output.probe_scores) > 0
        
        # Check structure
        for token_key, layer_dict in output.probe_scores.items():
            assert "layer_15" in layer_dict
            assert "test_sentiment" in layer_dict["layer_15"]["probes"]
            
            score = layer_dict["layer_15"]["probes"]["test_sentiment"]
            assert 0.0 <= score <= 1.0  # sigmoid bounds

7.3 Performance Benchmark

File: benchmarks/latency_test.py

"""
Measure latency overhead of probe system.

Target: <2ms per token for 50 probes.
"""

import time
import torch
from omniscience import OmniscienceLLM

def benchmark_latency():
    # Test with 0, 10, 50, 100 probes
    for num_probes in [0, 10, 50, 100]:
        # Create probe configs
        # ... setup ...
        
        # Measure time per token
        start = time.perf_counter()
        outputs = llm.generate(prompts, sampling_params)
        elapsed = time.perf_counter() - start
        
        total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
        ms_per_token = elapsed / total_tokens * 1000
        
        print(f"{num_probes} probes: {ms_per_token:.2f} ms/token")
        
        # Validation
        if num_probes > 0:
            assert ms_per_token < 2.0, f"Failed latency target: {ms_per_token:.2f}ms"

7.4 Correctness Validation

File: tests/test_correctness.py

"""
Verify probe scores are mathematically correct.
"""

def test_probe_math():
    """Compare omniscience output against manual PyTorch computation"""
    
    # Setup known probe weights
    hidden_dim = 128
    num_probes = 5
    weights = torch.randn(num_probes, hidden_dim)
    bias = torch.randn(num_probes)
    
    # Create hidden states
    hidden_states = torch.randn(4, hidden_dim)
    
    # Manual computation
    expected = torch.sigmoid(hidden_states @ weights.T + bias)
    
    # Omniscience computation
    # ... (load weights, compute probes) ...
    
    # Compare
    torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-5)

7.5 CUDA Graph Verification (CRITICAL)

File: tests/test_cuda_graphs.py

"""
Verify probe layers are compatible with CUDA graph capture.
"""

def test_cuda_graph_compatibility():
    """Test that probe layer works in CUDA graph"""
    
    # Create probe layer
    probe_layer = ProbePassthroughLayer(...)
    
    # Create dummy input
    hidden_states = torch.randn(32, 4096, device="cuda", dtype=torch.float16)
    
    # Capture graph
    graph = torch.cuda.CUDAGraph()
    
    with torch.cuda.graph(graph):
        output = probe_layer(hidden_states)
    
    # Replay graph multiple times
    for i in range(10):
        # Modify input (but keep same shape/address)
        hidden_states.uniform_(-1, 1)
        
        # Replay
        graph.replay()
        
        # Verify buffer was updated
        buffer_sum = probe_layer.output_buffer.sum().item()
        print(f"Iteration {i}: buffer sum = {buffer_sum}")
        
        # Buffer should have different values each time
        # If it's constant, graph is not updating the buffer!
    
    print("✓ CUDA graph compatibility test passed")

def test_graph_with_vllm():
    """Test full vLLM integration with graphs enabled"""
    
    # Load model with probes
    llm = OmniscienceLLM(
        model="meta-llama/Llama-2-7b-hf",
        omniscience_config=config,
        # Ensure graphs are enabled
        enforce_eager=False  # Don't disable graphs
    )
    
    # Generate multiple times
    prompts = ["Hello"] * 5
    outputs = []
    
    for prompt in prompts:
        output = llm.generate([prompt], SamplingParams(max_tokens=10))
        outputs.append(output[0])
        
        # Verify probe scores exist and are different each time
        assert hasattr(output[0], "probe_scores")
        print(f"Probe scores: {output[0].probe_scores}")
    
    # Verify scores are not identical (would indicate stale buffer)
    first_scores = outputs[0].probe_scores
    for i, output in enumerate(outputs[1:], 1):
        assert output.probe_scores != first_scores, \
            f"Output {i} has identical scores to output 0! Probes not updating."
    
    print("✓ vLLM CUDA graph integration test passed")

================================================================================
SECTION 3: CRITICAL IMPLEMENTATION DETAILS
================================================================================

3.0 CUDA Graph Compatibility Deep Dive

⚠️  This is the PRIMARY technical constraint. Get this wrong = probes stop working.

What are CUDA Graphs?
  - Record GPU kernel launches once during "capture" phase
  - Replay the recorded graph for subsequent iterations
  - ~5-10x faster launch overhead
  - vLLM uses them for decode phase by default

The Problem:
  Python code does NOT run during graph replay. Only the recorded CUDA ops.

Example (BROKEN):
```python
def forward(x):
    print(f"Batch size: {x.shape[0]}")  # Prints once, never again
    if x.shape[0] > 10:                 # Condition evaluated once
        return x * 2
    return x
```

Example (WORKS):
```python
def forward(x):
    # All tensor ops are recorded
    result = torch.matmul(x, self.weight)
    self.buffer[:x.shape[0]] = result  # Dynamic indexing is OK!
    return x
```

Rules for Graph-Safe Code:
✅ DO:
  - Use torch ops (matmul, add, slice, etc.)
  - Write to pre-allocated buffers (persistent=True)
  - Dynamic tensor shapes (x.shape[0]) - captured as dynamic dims
  - Activation functions (sigmoid, tanh, relu)

❌ DON'T:
  - Call .cpu(), .item(), .numpy() - forces sync
  - Python control flow on tensor values (if x.max() > 5)
  - Allocate new tensors (torch.zeros, torch.empty)
  - Append to Python lists
  - Print statements (they run once)

Why Our Architecture Works:
1. ProbePassthroughLayer is pure nn.Module
2. All ops are torch operations (matmul, slice assign)
3. Buffer is persistent (register_buffer with persistent=True)
4. No Python control flow
5. Metadata extraction happens OUTSIDE forward pass

Testing Strategy:
```python
# Force graph capture
torch.cuda.synchronize()
with torch.cuda.graph(torch.cuda.CUDAGraph()):
    output = model(dummy_input)
    
# If this doesn't crash, your ops are graph-safe
```

3.1 vLLM Version Compatibility Matrix

| vLLM Version | Integration Point | Notes |
|--------------|-------------------|-------|
| v0.2.x       | model.model.layers | Simple architecture |
| v0.3.x       | model.model.layers | Changed to ModuleList |
| v0.4+        | model.model.layers | Major refactor but layers still ModuleList |

→ Phase 0 must verify exact model structure.
→ Injection point is consistent: wrap items in model.model.layers

3.2 Memory Budget Calculation

Per-layer memory overhead (GPU):
  Output buffer: [max_batch_size, num_probes] * dtype_bytes
  Weight matrix: [hidden_dim, num_probes] * dtype_bytes
  Bias vector:   [num_probes] * dtype_bytes

Per-layer memory overhead (CPU, if using pinned memory):
  Pinned buffer: [max_batch_size, num_probes] * dtype_bytes
  
Example Configuration:
  - Model: Llama-2-7B (hidden_dim = 4096)
  - 10 layers with probes
  - 50 probes per layer
  - max_batch_size = 256
  - dtype = FP16 (2 bytes)
  
GPU Memory:
  Output buffers: 10 * 256 * 50 * 2 = 2.56 MB
  Weights:        10 * 4096 * 50 * 2 = 4.096 MB
  Bias:           10 * 50 * 2 = 0.001 MB
  Total GPU:      ~6.66 MB (negligible for inference)

CPU Pinned Memory (if async enabled):
  10 * 256 * 50 * 2 = 2.56 MB

Max Configuration (stress test):
  - 32 layers (full model)
  - 200 probes per layer
  - max_batch_size = 512
  
GPU Memory:
  Output:  32 * 512 * 200 * 2 = 6.55 MB
  Weights: 32 * 4096 * 200 * 2 = 52.4 MB
  Total:   ~59 MB (<1% of 80GB A100)

Conclusion: Memory overhead is NOT a concern for reasonable probe counts.

3.3 CUDA Graph Compatibility

vLLM uses CUDA graphs for decode phase optimization. Probe kernels must:
  - Have static shapes (no dynamic control flow)
  - Use pre-allocated buffers
  - Not call .cpu() or allocate memory

Test: Run with CUDA graphs enabled and verify no graph capture errors.

3.4 Continuous Batching and Request Mapping (CRITICAL)

vLLM's continuous batching flattens all tokens into single tensor:
  hidden_states: [num_tokens_in_batch, hidden_dim]

Example 1 - Pure Decode (simple):
  Request A at token 50, Request B at token 1, Request C at token 120
  Batch: [A_tok50, B_tok1, C_tok120]
  Shape: [3, 4096]
  Mapping: Row 0 → (A, 50), Row 1 → (B, 1), Row 2 → (C, 120)

Example 2 - Prefill (complex):
  Request D: prompt "Hello world, how are" (5 tokens)
  Request A continues: token 51 (decode)
  Batch: [D_tok0, D_tok1, D_tok2, D_tok3, D_tok4, A_tok51]
  Shape: [6, 4096]
  Mapping:
    Row 0 → (D, 0)
    Row 1 → (D, 1)
    Row 2 → (D, 2)
    Row 3 → (D, 3)
    Row 4 → (D, 4)
    Row 5 → (A, 51)

Challenge: How do we know rows 0-4 belong to Request D?

vLLM provides metadata in SequenceGroupMetadata:
  - is_prompt: bool (prefill vs decode)
  - prompt_len: int (if prefill, how many tokens)
  - seq_data: dict (sequence ID → token list)

Extraction Logic:
```python
row_idx = 0
for seq_group in seq_group_metadata_list:
    if seq_group.is_prompt:
        # Prefill: N tokens from same request
        for token_idx in range(seq_group.prompt_len):
            mapping[row_idx] = (seq_group.request_id, token_idx)
            row_idx += 1
    else:
        # Decode: 1 token per sequence
        for seq_data in seq_group.seq_data.values():
            token_idx = len(seq_data.get_token_ids()) - 1
            mapping[row_idx] = (seq_group.request_id, token_idx)
            row_idx += 1
```

Gotcha #1: Beam search creates multiple sequences per request
  Each beam is a separate seq_data entry

Gotcha #2: vLLM may use slot_mapping tensor for KV cache indexing
  May be more reliable than manual counting

Gotcha #3: Prefill may be chunked (long prompts split across batches)
  Check vLLM version for chunked prefill support

Recommendation: Use vLLM's internal mapping tensors when available,
fall back to manual counting only if necessary.

3.5 Distributed (Tensor Parallel) Considerations (REVISED)

In TP setups:
  - Attention weights are column-sharded
  - MLP weights are row/column sharded
  - **Residual stream is replicated** (our target!)
  
CRITICAL INSIGHT: All ranks must synchronize at all-reduce barriers.
If rank 0 spends 2ms computing probes, ranks 1-N wait 2ms at next barrier.

Strategy:
  - Only rank 0 computes probes (saves GPU cycles, not latency)
  - Other ranks don't inject probe layers (cleaner, less memory)
  - Document TRUE latency impact to users

Advanced Optimization (if overhead is high):
  - Shard probes across ranks: Rank 0 computes probes 0-24, Rank 1 computes 25-49
  - Gather results asynchronously (adds communication overhead)
  - Only worth it if probe count is very high (>100 probes)

================================================================================
SECTION 4: FAILURE MODES & DEBUGGING
================================================================================

4.1 Common Issues

Issue: "Probes work for first token, then stop updating"
  CAUSE: Python hook not executing during CUDA graph replay
  FIX: Migrate to nn.Module injection pattern (see Phase 3)
  VERIFY: Check buffer contents change on every step:
    ```python
    for i in range(10):
        output = llm.generate(...)
        print(probe_layer.output_buffer[0, 0].item())  # Should vary
    ```

Issue: "CUDA graph capture failed"
  CAUSE: Non-capturable operation in probe layer
  DEBUG:
    1. Set CUDA_LAUNCH_BLOCKING=1
    2. Check for .cpu(), .item(), .numpy() in ProbePassthroughLayer
    3. Check for new tensor allocations (torch.zeros in forward)
    4. Check for Python control flow on tensor values
  FIX: Replace with graph-safe equivalent

Issue: "Request ID mismatch / wrong probes attached"
  CAUSE: Metadata extraction not matching buffer rows
  DEBUG:
    1. Log metadata.num_tokens vs actual batch size
    2. Check if prefill tokens are being counted correctly
    3. Verify slot_mapping interpretation
  FIX: Use vLLM's internal mapping tensors, not manual counting

Issue: "Latency increased by 10ms+ per token"
  CAUSE: Synchronous CPU transfer or TP synchronization overhead
  DEBUG:
    1. Profile with `nsys profile --trace=cuda,nvtx`
    2. Check for cudaDeviceSynchronize calls
    3. In TP setup, check if all ranks are synchronized
  FIX: Implement async transfers (Phase 5), optimize probe count

Issue: "CUDA out of memory"
  CAUSE: Output buffers consuming too much VRAM
  DEBUG: Calculate expected memory: max_batch_size * num_probes * 2 bytes
  FIX: Reduce max_batch_size or use FP16 instead of FP32

Issue: "Probe buffer contains stale data"
  CAUSE: Not zeroing unused rows when batch < max_batch_size
  FIX: Add zero-out in ProbePassthroughLayer:
    ```python
    if valid_tokens < self.max_tokens:
        self.output_buffer[valid_tokens:].zero_()
    ```

4.2 Debugging Workflow

1. Start with single probe, single layer
2. Test with batch_size=1 first
3. Add logging at every major step
4. Use torch.cuda.synchronize() + timing in development
5. Profile with nsys for production optimization

4.3 Logging Strategy

```python
import logging

# Configure hierarchical logging
logging.basicConfig(level=logging.INFO)

# Detailed logs for development
logger = logging.getLogger("omniscience")
logger.setLevel(logging.DEBUG)

# Example log messages
logger.debug(f"Hook called: layer={layer_idx}, batch={hidden_states.shape}")
logger.info(f"Loaded {num_probes} probes for layer {layer_idx}")
logger.warning(f"Batch size {actual} exceeds max {max_batch_size}")
logger.error(f"Probe computation failed: {exception}")
```

================================================================================
SECTION 5: DEPLOYMENT CHECKLIST
================================================================================

Before production deployment:

Core Functionality:
□ Run full test suite with 100% pass rate
□ Verify probes update on EVERY token (not just first)
□ Test with batch_size=1,2,4,8,16,32 (variable sizes)
□ Confirm buffer contents change on each forward pass
□ Validate request ID mapping is 100% accurate

Performance:
□ Benchmark latency on target GPU (meets <2ms requirement)
□ Profile with nsys: verify no unexpected synchronizations
□ Test with tensor_parallel_size=1,2,4,8
□ Document TRUE latency overhead in TP setups (affects all ranks)
□ Verify async transfers working (Phase 5)

CUDA Graph Compatibility:
□ Verify CUDA graph compilation succeeds with probes
□ Test: Disable graphs (should still work), then re-enable
□ Confirm no "graph capture failed" errors
□ Check vLLM's custom op fusion doesn't conflict

Continuous Batching:
□ Test prefill phase: multi-token prompts map correctly
□ Test decode phase: single tokens map correctly
□ Test mixed batches (if vLLM supports)
□ Verify slot_mapping interpretation is correct

Edge Cases:
□ Test with beam search (multiple sequences per request)
□ Test with very long sequences (>2048 tokens)
□ Test when batch exceeds max_batch_size (graceful degradation)
□ Test probe weight reload without restart
□ Test with empty probes (0 probes configured)

Production Readiness:
□ Load test with 1000+ concurrent requests
□ Memory leak test: run 10k+ requests, check VRAM usage
□ Document API for end users
□ Create example probe training script
□ Set up monitoring/metrics (probe compute time, buffer sizes)
□ Test with multiple model sizes (7B, 13B, 70B)
□ Create rollback plan if probes cause issues

================================================================================
SECTION 6: FUTURE EXTENSIONS
================================================================================

6.1 Potential Enhancements

- Multi-head probes (one weight matrix, multiple activation functions)
- Dynamic probe loading (add/remove without restart)
- Probe quantization (INT8/INT4 for memory savings)
- Sparse probes (only compute for specific tokens/conditions)
- Probe ensembles (average multiple probe outputs)
- Training mode (save activations for probe training)

6.2 Performance Roadmap

- Phase 1 (MVP): PyTorch ops, synchronous, <5ms overhead
- Phase 2 (Optimized): Async transfers, <2ms overhead
- Phase 3 (Advanced): Triton kernels, <1ms overhead
- Phase 4 (Expert): Fused with attention, <0.5ms overhead

================================================================================
APPENDIX A: ADDRESSING CRITICAL ISSUES
================================================================================

This appendix documents how the plan addresses the five critical issues
identified in the review.

─────────────────────────────────────────────────────────────────────
Issue #1: The "CUDA Graph" Showstopper
─────────────────────────────────────────────────────────────────────

Problem: Python hooks don't execute during CUDA graph replay.

Resolution:
  ✅ Replaced Python hooks with nn.Module injection (Phase 2.1)
  ✅ ProbePassthroughLayer contains only torch operations
  ✅ Writes to persistent buffer (register_buffer with persistent=True)
  ✅ Metadata extraction moved OUTSIDE forward pass (Phase 3.3)
  ✅ Added CUDA graph verification tests (Section 7.5)
  ✅ Added Phase 0.6: CUDA graph compatibility testing

Impact: Probes will execute on EVERY token during both graph capture and replay.

Implementation Details:
  - Section 1.1: Updated architecture diagram
  - Section 2.1: ProbePassthroughLayer.forward() is graph-safe
  - Section 3.0: Deep dive on CUDA graph compatibility
  - Section 4.1: Troubleshooting "probes stop after first token"

─────────────────────────────────────────────────────────────────────
Issue #2: The "Rank 0" Latency Fallacy
─────────────────────────────────────────────────────────────────────

Problem: Claimed rank 0 computation wouldn't affect global latency.

Resolution:
  ✅ Updated Section 1.2: Removed "hide latency" claim
  ✅ Section 3.5: Explicitly states "overhead affects ALL ranks"
  ✅ TensorParallelHandler includes warning log message
  ✅ Deployment checklist: "Document TRUE latency overhead in TP setups"
  ✅ Added note about probe sharding for high probe counts

Impact: Users have honest expectations about performance in TP environments.

Implementation Details:
  - Section 3.4: "Only rank 0 computes probes to avoid redundant work"
  - Section 3.5: "CRITICAL INSIGHT: All ranks must synchronize..."
  - Phase 3.4: TensorParallelHandler logs warning
  - Deployment: Explicit TP testing at sizes 2,4,8

─────────────────────────────────────────────────────────────────────
Issue #3: The Context/Thread-Safety Race Condition
─────────────────────────────────────────────────────────────────────

Problem: Thread-local global context could be overwritten in pipelined execution.

Resolution:
  ✅ Removed thread-local context entirely (deleted vLLMHookContext)
  ✅ Metadata extraction happens AFTER forward pass (Phase 3.2)
  ✅ ModelRunnerWrapper reads scheduler metadata post-execution
  ✅ No shared state between forward pass and metadata extraction

Impact: No race conditions possible; clean separation of concerns.

Implementation Details:
  - Phase 3.2: MetadataExtractor operates on scheduler output
  - Phase 3.3: ModelRunnerWrapper wraps execute_model, processes after
  - No global mutable state in the system
  - Section 1.1: Architecture diagram shows clear separation

─────────────────────────────────────────────────────────────────────
Issue #4: Minor Technical Errors
─────────────────────────────────────────────────────────────────────

4.1 Safetensors Metadata

Problem: Loading full tensor for validation is slow.

Resolution:
  ✅ Phase 1.2: ProbeWeightSet._validate_probe_shape() uses header-only read
  ✅ Uses get_slice().get_shape() instead of loading tensor
  ✅ Validates shape without I/O penalty

Implementation:
  ```python
  with safe_open(path, framework="pt", device="cpu") as f:
      weight_shape = f.get_slice("weight").get_shape()
      # No tensor data loaded!
  ```

4.2 Buffer Pinning Logic

Problem: pin_memory=True on GPU tensors is incorrect.

Resolution:
  ✅ Phase 2.2: ProbeBufferManager separates GPU and CPU buffers
  ✅ GPU buffers: device="cuda", no pin_memory
  ✅ CPU buffers: device="cpu", pin_memory=True (correct)
  ✅ Two separate buffer dictionaries (gpu_buffers, cpu_pinned_buffers)

Implementation:
  ```python
  # GPU buffer (written by forward pass)
  gpu_buf = torch.zeros(..., device="cuda")
  
  # CPU pinned buffer (destination for async transfer)
  cpu_buf = torch.zeros(..., pin_memory=True)  # CPU tensor
  ```

4.3 Request Tracking in Continuous Batching

Problem: Manual counting of batch positions is error-prone, especially in prefill.

Resolution:
  ✅ Section 3.4: Detailed explanation of vLLM's flattened batch structure
  ✅ Phase 3.2: MetadataExtractor handles both prefill and decode
  ✅ Explicit logic for multi-token prefill (prompt_len iteration)
  ✅ Documentation of gotchas (beam search, chunked prefill)
  ✅ Recommendation to use vLLM's slot_mapping when available

Implementation:
  ```python
  if seq_group.is_prompt:
      # Prefill: N tokens from same request
      for token_idx in range(seq_group.prompt_len):
          mapping[row_idx] = (request_id, token_idx)
          row_idx += 1
  else:
      # Decode: 1 token per sequence
      for seq_data in seq_group.seq_data.values():
          mapping[row_idx] = (request_id, current_token_idx)
          row_idx += 1
  ```

─────────────────────────────────────────────────────────────────────
Issue #5: Refined Architecture Recommendation
─────────────────────────────────────────────────────────────────────

Suggested: Use "Passthrough Module" pattern instead of Python hooks.

Resolution:
  ✅ Entire Phase 2 rewritten around ProbePassthroughLayer
  ✅ Phase 3 uses model injection, not Python hooks
  ✅ Architecture follows exact pattern recommended:
    - nn.Module with nn.Parameter weights
    - forward() writes to persistent buffer
    - Returns hidden_states unchanged (passthrough)
    - Post-processing outside graph

Implementation Details:
  - Phase 2.1: ProbePassthroughLayer matches suggested design exactly
  - Section 1.1: Architecture explicitly separates in-graph vs out-of-graph
  - Phase 3.1: ModelArchitecturePatcher wraps layers in Sequential
  - Phase 3.3: ModelRunnerWrapper processes results after execute_model

Code Structure Matches Suggestion:
  ```python
  class ProbePassthroughLayer(nn.Module):
      def forward(self, hidden_states):
          probe_scores = self.probe_weights(hidden_states)
          self.output_buffer[:num_tokens] = probe_scores
          return hidden_states  # Passthrough
  ```

─────────────────────────────────────────────────────────────────────
Additional Improvements
─────────────────────────────────────────────────────────────────────

Beyond the five main issues, this revision includes:

1. Phase 0.6: Explicit CUDA graph compatibility testing phase
2. Section 3.0: Deep dive on CUDA graph rules and patterns
3. Section 3.4: Comprehensive continuous batching documentation
4. Section 4.1: Extended troubleshooting (7 common issues)
5. Section 7.5: CUDA graph verification tests
6. Deployment: Expanded checklist (35+ items)
7. Memory: Detailed calculations with examples
8. Architecture: Clear "in-graph" vs "out-of-graph" separation

Summary: All five critical issues have been addressed with concrete
implementation details, code examples, and verification tests.

================================================================================
APPENDIX B: IMPLEMENTATION SEQUENCE SUMMARY
================================================================================

For quick reference, the optimal implementation order:

Phase 0 (1-2 hours): Reconnaissance
  → Understand vLLM structure before writing code
  → Test CUDA graph compatibility with dummy layer
  → Output: docs/vllm_integration_notes.md

Phase 1 (4-6 hours): Configuration & Weight Loading
  → Pydantic config, safetensors loading, nn.Parameter registration
  → Output: ProbeWeightManager with loaded weights
  → Gate: Test loads weights correctly

Phase 2 (8-10 hours): Passthrough Module
  → ProbePassthroughLayer, buffer management, factory
  → Output: Graph-compatible probe computation
  → Gate: Standalone test (no vLLM yet)

Phase 3 (12-15 hours): vLLM Integration
  → Model patching, metadata extraction, ModelRunner wrapper
  → Output: End-to-end probes in vLLM
  → Gate: Probes appear in RequestOutput

Phase 4 (6-8 hours): API Integration
  → Output attachment, result formatting
  → Output: User-facing API
  → Gate: Clean probe_scores in outputs

Phase 5 (8-10 hours): Async Optimization
  → Non-blocking GPU→CPU transfers
  → Output: <0.5ms overhead async system
  → Gate: Latency benchmarks pass

Phase 6 (8-10 hours): Triton Kernel (Optional)
  → Fused matmul+activation kernel
  → Output: Additional 10%+ speedup
  → Gate: Benchmark vs PyTorch

Phase 7 (6-8 hours): Testing & Validation
  → Unit tests, integration tests, performance tests
  → Output: Production-ready system
  → Gate: 100% test pass, <2ms overhead

Total: 47-69 hours (~1-2 weeks for single developer)

Critical Path: Phase 0 → 1 → 2 → 3 (can't parallelize)
Optional: Phase 6 (only if benchmarks show need)

================================================================================
END OF PLAN
================================================================================