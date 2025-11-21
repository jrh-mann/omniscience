from dataclasses import dataclass
from typing import Any
import numpy as np
import torch

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
            "token_idx": self.token_idx,
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
        # Check if probe_scores is actually on GPU before calling cpu()
        if probe_scores.device.type != "cpu":
             scores_cpu = probe_scores.cpu().numpy()  # [batch, num_probes]
        else:
             scores_cpu = probe_scores.numpy()

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
