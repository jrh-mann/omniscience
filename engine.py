from vllm import LLM, SamplingParams
from vllm.model_executor import model_loader
import torch
import logging
import multiprocessing as mp
import os
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
    """
    
    def __init__(
        self,
        omniscience_config: OmniscienceConfig,
        **vllm_kwargs
    ):
        self.config = omniscience_config
        self.tp_guard = TensorParallelHandler()
        
        # Communication channel
        self.result_queue = mp.Queue()
        
        # Initialize result processor (Consumer)
        # We reuse AsyncProbeResultTracker as a container for results on Main side too
        # But mainly we use it to reassemble results from the queue.
        # Wait, AsyncProbeResultTracker is designed to be the Producer (GPU->CPU->Queue).
        # We need a simple consumer logic in generate().
        
        self.output_attacher = ProbeOutputAttacher(None) # Tracker not needed for init anymore
        
        # Patch get_model to run our hook in the Worker process
        self._original_get_model = model_loader.get_model
        
        if self.tp_guard.should_inject_probe_layers():
             model_loader.get_model = self._wrapped_get_model
        
        # Attempt to force fork to ensure our patch survives if possible
        # This only works if CUDA is not initialized yet.
        # We haven't imported ProbeWeightManager or touched CUDA in this __init__.
        if not torch.cuda.is_initialized():
             os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'fork'
             logger.info("Attempting to force 'fork' start method for vLLM workers.")
        else:
             logger.warning("CUDA is already initialized. vLLM will likely use 'spawn', which may break probing hooks.")

        try:
            self.llm = LLM(**vllm_kwargs)
        finally:
            # Restore original get_model
            model_loader.get_model = self._original_get_model
            
        logger.info("OmniscienceLLM initialized successfully")

    def _wrapped_get_model(self, *, vllm_config, model_config=None):
        """
        Hook to inject probes when model is loaded.
        This runs inside the Worker process.
        """
        # Call original loader first to get the model
        model = self._original_get_model(vllm_config=vllm_config, model_config=model_config)
        
        try:
            logger.info("Intercepted model loading in Worker process. Initializing probes...")
            
            # Initialize components LOCAL to the worker
            # This avoids pickling CUDA tensors from Main
            self._init_worker_components()
            
            # Patch the model
            self.patcher.patch_model(model)
            
            # Install runner hook (which is now local)
            self._install_runner_hook_worker()
            
        except Exception as e:
            logger.error(f"Failed to inject probes in worker: {e}", exc_info=True)
            
        return model

    def _init_worker_components(self):
        """Initialize probing components inside the worker process"""
        # Load weights (CUDA context is valid here)
        self.weight_manager = ProbeWeightManager(self.config)
        self.buffer_manager = ProbeBufferManager(self.weight_manager, self.config)
        self.probe_factory = ProbeModuleFactory(self.weight_manager, self.buffer_manager, self.config)
        self.patcher = ModelArchitecturePatcher(self.probe_factory, self.config)
        self.metadata_extractor = MetadataExtractor()
        
        # Result tracker writes to the multiprocessing queue
        self.result_tracker = AsyncProbeResultTracker(
            self.config.use_pinned_memory, 
            queue=self.result_queue
        )

    def _install_runner_hook_worker(self):
        """Find and wrap the ModelRunner (Worker side)"""
        # Logic similar to before, but we don't need self.llm reference
        # We can find ModelRunner via inspection of the stack or singleton?
        # Actually, we don't have easy access to the ModelRunner instance from inside get_model.
        # get_model returns the model. The ModelRunner calls get_model.
        # So the ModelRunner has 'model' attribute.
        
        # Strategy: We can't wrap ModelRunner instance easily because it's the one calling us (indirectly).
        # BUT, ModelRunner calls `self.model = get_model(...)`.
        # So we are patching the model that ModelRunner owns.
        
        # To intercept `execute_model`, we need to patch the ModelRunner class or instance.
        # If we are inside `get_model`, we might be able to inspect the stack to find the `ModelRunner` instance.
        
        import inspect
        runner_instance = None
        try:
            # Walk up stack to find 'self' that is a ModelRunner
            for frame_info in inspect.stack():
                frame = frame_info.frame
                # Check 'self' arg
                if 'self' in frame.f_locals:
                    obj = frame.f_locals['self']
                    # Check if it looks like a ModelRunner (has execute_model)
                    if hasattr(obj, 'execute_model') and hasattr(obj, 'input_batch'):
                        runner_instance = obj
                        break
        except Exception:
            pass
            
        if runner_instance:
            logger.info(f"Found ModelRunner instance via stack inspection: {type(runner_instance)}")
            self.runner_wrapper = ModelRunnerWrapper(
                model_runner=runner_instance,
                patcher=self.patcher,
                metadata_extractor=self.metadata_extractor,
                buffer_manager=self.buffer_manager,
                weight_manager=self.weight_manager,
                result_tracker=self.result_tracker,
                config=self.config
            )
            self.runner_wrapper.install()
        else:
            logger.error("Could not find ModelRunner instance via stack inspection. Probes will run but results won't be collected.")

    def generate(
        self,
        prompts: list[str] | list[list[int]] | None = None,
        sampling_params: SamplingParams | None = None,
        **kwargs
    ) -> list[Any]:
        """Generate with probe scores attached"""
        # Start generation
        outputs = self.llm.generate(prompts, sampling_params, **kwargs)
        
        # Consume queue results
        results_by_req = {}
        
        # Drain queue
        # Note: this assumes generation is finished and all results are in queue.
        # vLLM generate() blocks, so this is true.
        # BUT, we might read from queue while generating if we used async iterator.
        # Here we just drain at the end.
        while not self.result_queue.empty():
            try:
                result_dict = self.result_queue.get_nowait()
                req_id = result_dict["request_id"]
                if req_id not in results_by_req:
                    results_by_req[req_id] = []
                results_by_req[req_id].append(result_dict)
            except mp.queues.Empty:
                break
        
        # Attach to outputs
        for output in outputs:
            if output.request_id in results_by_req:
                self.output_attacher.attach_results(output, results_by_req[output.request_id])
        
        return outputs
