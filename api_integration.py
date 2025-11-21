from vllm import RequestOutput
from typing import Any
import logging
from .tracking import ProbeResultTracker, ProbeResult

logger = logging.getLogger(__name__)

class ProbeOutputAttacher:
    """
    Attaches probe results to vLLM RequestOutput objects.
    
    Strategy:
    Monkey-patch RequestOutput to add 'probe_scores' attribute.
    """
    
    def __init__(self, result_tracker: ProbeResultTracker):
        self.result_tracker = result_tracker
        self._patch_request_output()
    
    def _patch_request_output(self):
        """Add probe_scores field to RequestOutput if not exists"""
        # Check if we've already patched it (or if vLLM added it natively, unlikely)
        # Note: Checking the class itself for the attribute doesn't work for instance vars added in init
        # So we check if we've already wrapped init
        
        if getattr(RequestOutput, "_omniscience_patched", False):
            return

        original_init = RequestOutput.__init__
        
        def new_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            # Initialize probe_scores dictionary
            self.probe_scores: dict[str, Any] = {}
        
        RequestOutput.__init__ = new_init
        RequestOutput._omniscience_patched = True
        logger.info("Patched RequestOutput to include probe_scores")
    
    def attach_results(
        self,
        request_output: RequestOutput,
        probe_results: list[dict]  # These are dicts from ProbeResult.to_dict()
    ) -> RequestOutput:
        """
        Attach probe results to output object.
        
        Format:
        {
            "token_0": {"layer_15": {"probe1": 0.82, "probe2": 0.34}},
            "token_1": {...}
        }
        """
        # Ensure attribute exists (in case patch didn't work or object created before patch)
        if not hasattr(request_output, "probe_scores"):
            request_output.probe_scores = {}
        
        for result_dict in probe_results:
            # result_dict has 'token_idx', 'layer', 'probes'
            token_idx = result_dict["token_idx"]
            layer_idx = result_dict["layer"]
            probes = result_dict["probes"]
            
            token_key = f"token_{token_idx}"
            
            if token_key not in request_output.probe_scores:
                request_output.probe_scores[token_key] = {}
            
            layer_key = f"layer_{layer_idx}"
            
            if layer_key not in request_output.probe_scores[token_key]:
                 request_output.probe_scores[token_key][layer_key] = {}
                 
            request_output.probe_scores[token_key][layer_key].update(probes)
        
        return request_output
