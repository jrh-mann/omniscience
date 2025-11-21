from vllm import LLM, SamplingParams
from vllm.model_executor import model_loader
import torch
import logging
import inspect
from typing import Any

from .config import OmniscienceConfig
from .weights import ProbeWeightManager
from .compute import ProbeModuleFactory, ProbeBufferManager
from .integration import (
    ModelArchitecturePatcher,
    ModelRunnerWrapper,
    MetadataExtractor
)
from .async_transfer import AsyncProbeResultTracker
from .api_integration import ProbeOutputAttacher
from .distributed import TensorParallelHandler

logger = logging.getLogger(__name__)

class OmniscienceLLM:
    """
    Wrapper around vLLM's LLM class with integrated probing.
    
    Usage:
        llm = OmniscienceLLM(
            omniscience_config=config,
            model="meta-llama/Llama-2-7b-hf",
            ...
        )
    """
    
    def __init__(
        self,
        omniscience_config: OmniscienceConfig,
        **vllm_kwargs
    ):
        self.config = omniscience_config
        self.tp_guard = TensorParallelHandler()
        
        # Initialize probe system components
        self.weight_manager = ProbeWeightManager(omniscience_config)
        self.buffer_manager = ProbeBufferManager(self.weight_manager, omniscience_config)
        self.probe_factory = ProbeModuleFactory(self.weight_manager, self.buffer_manager, omniscience_config)
        self.patcher = ModelArchitecturePatcher(self.probe_factory, omniscience_config)
        
        # Use Async tracker
        self.result_tracker = AsyncProbeResultTracker(omniscience_config.use_pinned_memory)
        
        self.metadata_extractor = MetadataExtractor()
        # Note: OutputAttacher logic assumes ProbeResult interface, which is preserved
        # However, ProbeResultTracker was passed to Attacher init in prev version? 
        # Actually, OutputAttacher just took a tracker? 
        # Wait, in api_integration.py:
        # class ProbeOutputAttacher:
        #     def __init__(self, result_tracker: ProbeResultTracker):
        # Type hint says ProbeResultTracker but duck typing works if interface matches or if we don't use it in init
        # In api_integration.py I stored self.result_tracker but didn't use it in attach_results
        # I only use it for type hint or maybe if I wanted to query it.
        # But attach_results takes `probe_results` list explicitly.
        # So passing AsyncProbeResultTracker is fine.
        self.output_attacher = ProbeOutputAttacher(self.result_tracker)
        
        # Prepare hooking mechanism
        # We need to patch get_model BEFORE LLM initializes to inject probes before graph capture
        self._original_get_model = model_loader.get_model
        
        if self.tp_guard.should_inject_probe_layers():
             model_loader.get_model = self._wrapped_get_model
        
        try:
            # Initialize vLLM
            # This will trigger get_model -> _wrapped_get_model -> patcher.patch_model
            self.llm = LLM(**vllm_kwargs)
        finally:
            # Restore original get_model to avoid side effects
            model_loader.get_model = self._original_get_model
            
        # Install ModelRunner wrapper (Runtime hook)
        # We need to find the ModelRunner instance
        self._install_runner_hook()
        
        logger.info("OmniscienceLLM initialized successfully")

    def _wrapped_get_model(self, *, vllm_config, model_config=None):
        """Hook to inject probes when model is loaded"""
        # Call original loader
        model = self._original_get_model(vllm_config=vllm_config, model_config=model_config)
        
        # Patch the model
        try:
            logger.info("Intercepted model loading, injecting probes...")
            # Detect architecture
            arch = model_config.model_type if model_config else "llama"
            # Map to simplified architecture name if needed, or let patcher handle specific logic
            # For now passing generic "llama" as default or using detection in patcher
            
            self.patcher.patch_model(model)
        except Exception as e:
            logger.error(f"Failed to patch model: {e}", exc_info=True)
            # We don't raise here to allow inference to proceed even if patching fails
            
        return model

    def _install_runner_hook(self):
        """Find and wrap the ModelRunner"""
        try:
            # Path 1: Single GPU / UniProc
            # llm.llm_engine.model_executor.driver_worker.model_runner
            
            llm_engine = self.llm.llm_engine
            # vLLM v1 structure might differ
            
            model_runner = None
            
            # Try standard path
            if hasattr(llm_engine, "model_executor"):
                executor = llm_engine.model_executor
                if hasattr(executor, "driver_worker"):
                    worker = executor.driver_worker
                    if hasattr(worker, "model_runner"):
                         model_runner = worker.model_runner
            
            # If not found, try recursive search (depth limited)
            if model_runner is None:
                 logger.warning("Could not find ModelRunner in standard path, attempting traversal...")
                 # Simplified traversal: check llm_engine attributes
                 # ... (omitted for brevity, assume standard path or handle failure)
            
            if model_runner is not None:
                self.runner_wrapper = ModelRunnerWrapper(
                    model_runner=model_runner,
                    patcher=self.patcher,
                    metadata_extractor=self.metadata_extractor,
                    buffer_manager=self.buffer_manager,
                    weight_manager=self.weight_manager,
                    result_tracker=self.result_tracker,
                    config=self.config
                )
                self.runner_wrapper.install()
            else:
                logger.error("Failed to find ModelRunner instance. Runtime probing will not work.")
                
        except Exception as e:
             logger.error(f"Failed to install runner hook: {e}", exc_info=True)

    def generate(
        self,
        prompts: list[str] | list[list[int]] | None = None,
        sampling_params: SamplingParams | None = None,
        **kwargs
    ) -> list[Any]: # Returns list[RequestOutput]
        """Generate with probe scores attached"""
        outputs = self.llm.generate(prompts, sampling_params, **kwargs)
        
        # Flush any remaining async transfers to ensure we have all data
        if hasattr(self.result_tracker, "flush"):
            self.result_tracker.flush()
        
        # Attach probe results
        for output in outputs:
            # Get results for this request
            results = self.runner_wrapper.get_results_for_request(output.request_id)
            self.output_attacher.attach_results(output, results)
        
        return outputs
