import torch
import torch.nn as nn
from typing import Any, TYPE_CHECKING
import logging
from dataclasses import dataclass

from .compute import ProbeModuleFactory, ProbePassthroughLayer, ProbeBufferManager
from .weights import ProbeWeightManager
from .config import OmniscienceConfig
from .async_transfer import AsyncProbeResultTracker

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

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
            model: vLLM's loaded model (nn.Module)
            model_architecture: "llama", "mistral", etc. (from Phase 0 recon)
        """
        # Get model's layer container (architecture-specific)
        # Based on vLLM v1 structure
        if hasattr(model, "model") and hasattr(model.model, "layers"):
             layer_container = model.model.layers
        elif hasattr(model, "layers"):
             # Some architectures might be flat
             layer_container = model.layers
        else:
            logger.warning(f"Could not find 'layers' in model {type(model)}. Patching failed.")
            return
        
        # Iterate over all layers
        for layer_idx in range(len(layer_container)):
            # Check if this layer has probes configured
            probe_layer = self.probe_factory.create_probe_layer(layer_idx)
            if probe_layer is None:
                continue  # No probes for this layer
            
            # Wrap original layer + probe layer in Sequential
            original_layer = layer_container[layer_idx]
            
            # Check if already patched (avoid double wrapping on reload)
            if isinstance(original_layer, nn.Sequential) and isinstance(original_layer[-1], ProbePassthroughLayer):
                logger.info(f"Layer {layer_idx} already patched, skipping.")
                continue

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
        # Verify structure just in case
        if len(sequential) > 1 and isinstance(sequential[1], ProbePassthroughLayer):
            return sequential[1]
        return None

@dataclass
class BatchMetadata:
    """
    Metadata for current batch, extracted AFTER forward pass.
    
    This is safe because we extract it outside the CUDA graph.
    """
    request_ids: list[str]
    token_indices: list[int]
    num_tokens: int
    
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
    
    def __init__(self):
        pass
    
    def extract(
        self,
        scheduler_output: "SchedulerOutput",
        req_ids_order: list[str]
    ) -> BatchMetadata:
        """
        Extract metadata from vLLM's SchedulerOutput.
        
        Args:
            scheduler_output: vLLM's internal scheduler output
            req_ids_order: The ordered list of request IDs from InputBatch
        
        Returns:
            BatchMetadata with request IDs and token positions
        """
        request_ids = []
        token_indices = []
        
        # Build lookup for start_token_idx
        start_indices = {}
        
        # 1. New requests
        for req in scheduler_output.scheduled_new_reqs:
             start_indices[req.req_id] = req.num_computed_tokens
             
        # 2. Cached requests
        cached = scheduler_output.scheduled_cached_reqs
        for i, req_id in enumerate(cached.req_ids):
             start_indices[req_id] = cached.num_computed_tokens[i]
             
        # 3. Build flat lists based on execution order
        for req_id in req_ids_order:
             if req_id is None: # Should not happen in active batch but safety check
                 continue
                 
             count = scheduler_output.num_scheduled_tokens.get(req_id, 0)
             start = start_indices.get(req_id, 0)
             
             for offset in range(count):
                 request_ids.append(req_id)
                 token_indices.append(start + offset)
        
        return BatchMetadata(
            request_ids=request_ids,
            token_indices=token_indices,
            num_tokens=len(request_ids)
        )

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
        model_runner: "GPUModelRunner",
        patcher: ModelArchitecturePatcher,
        metadata_extractor: MetadataExtractor,
        buffer_manager: ProbeBufferManager,
        weight_manager: ProbeWeightManager,
        result_tracker: AsyncProbeResultTracker, # Updated type
        config: OmniscienceConfig
    ):
        self.model_runner = model_runner
        self.patcher = patcher
        self.metadata_extractor = metadata_extractor
        self.buffer_manager = buffer_manager
        self.weight_manager = weight_manager
        self.result_tracker = result_tracker
        self.config = config
        
        # Store original execute_model
        self.original_execute_model = model_runner.execute_model
        
    def install(self):
        """Replace execute_model with our wrapper"""
        def wrapped_execute_model(scheduler_output, *args, **kwargs):
            # Call original vLLM execution
            output = self.original_execute_model(scheduler_output, *args, **kwargs)
            
            try:
                # Extract metadata AFTER forward pass
                # Access input_batch from model_runner to get execution order
                req_ids_order = self.model_runner.input_batch.req_ids
                
                metadata = self.metadata_extractor.extract(
                    scheduler_output,
                    req_ids_order
                )
                
                # Read probe buffers and map to requests
                self._process_probe_outputs(metadata)
            except Exception as e:
                logger.error(f"Error processing probe outputs: {e}", exc_info=True)
            
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
            gpu_buffer = probe_layer.output_buffer
            
            # Access CPU pinned buffer
            cpu_buffer = self.buffer_manager.get_cpu_buffer(layer_idx)
            
            # Get probe names
            weights = self.weight_manager.get_layer_weights(layer_idx)
            probe_names = weights.probe_names
            
            # Initiate async transfer
            self.result_tracker.record_batch_async(
                layer_idx=layer_idx,
                gpu_buffer=gpu_buffer,
                cpu_buffer=cpu_buffer,
                metadata_num_tokens=metadata.num_tokens,
                request_ids=metadata.request_ids,
                token_indices=metadata.token_indices,
                probe_names=probe_names
            )
    
    def get_results_for_request(self, request_id: str) -> list[dict]:
        """Retrieve and clear results for a request"""
        # The tracker manages the buffer of results now
        return self.result_tracker.get_results(request_id)
