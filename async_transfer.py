import torch
import logging
import multiprocessing as mp
from collections import deque
from dataclasses import dataclass
from .tracking import ProbeResult

logger = logging.getLogger(__name__)

@dataclass
class PendingTransfer:
    """Represents an in-flight GPU→CPU transfer"""
    layer_idx: int
    scores_cpu_pinned: torch.Tensor # The destination buffer (pinned)
    request_ids: list[str]
    token_indices: list[int]
    probe_names: list[str]
    num_rows: int
    transfer_event: torch.cuda.Event # Marks when transfer is done

class AsyncProbeResultTracker:
    """
    Manages async transfers using CUDA streams and events.
    """
    
    def __init__(self, use_pinned_memory: bool = True, queue: mp.Queue | None = None):
        self.pending_queue: deque[PendingTransfer] = deque()
        self.completed_results: dict[str, list[dict]] = {} # Buffer for final results
        self.use_pinned_memory = use_pinned_memory
        self.queue = queue
        
        # Create dedicated stream for CPU transfers
        if torch.cuda.is_available():
            self.transfer_stream = torch.cuda.Stream()
        else:
            self.transfer_stream = None
            
    def record_batch_async(
        self,
        layer_idx: int,
        gpu_buffer: torch.Tensor, # [max_tokens, num_probes] on GPU
        cpu_buffer: torch.Tensor, # [max_tokens, num_probes] Pinned CPU
        metadata_num_tokens: int,
        request_ids: list[str],
        token_indices: list[int],
        probe_names: list[str]
    ):
        """
        Initiate async transfer for this batch.
        """
        if not torch.cuda.is_available():
            # Fallback to sync
            self._process_sync(layer_idx, gpu_buffer, metadata_num_tokens, request_ids, token_indices, probe_names)
            return

        # 1. Slice valid data on GPU (view)
        valid_gpu = gpu_buffer[:metadata_num_tokens]
        valid_cpu = cpu_buffer[:metadata_num_tokens]
        
        # 2. Record event on current stream (Compute stream)
        # This marks "GPU is done writing to gpu_buffer"
        compute_done_event = torch.cuda.Event()
        compute_done_event.record()
        
        # 3. Queue transfer on transfer stream
        # Wait for compute to finish
        self.transfer_stream.wait_event(compute_done_event)
        
        # Async copy D2H
        with torch.cuda.stream(self.transfer_stream):
            valid_cpu.copy_(valid_gpu, non_blocking=True)
            
            # Record event that transfer is done
            transfer_done_event = torch.cuda.Event()
            transfer_done_event.record()
        
        # 4. Store pending transfer state
        pending = PendingTransfer(
            layer_idx=layer_idx,
            scores_cpu_pinned=valid_cpu, # Keep reference to pinned view
            request_ids=request_ids,
            token_indices=token_indices,
            probe_names=probe_names,
            num_rows=metadata_num_tokens,
            transfer_event=transfer_done_event
        )
        
        self.pending_queue.append(pending)
        
        # 5. Process completed transfers
        self.process_queue()
        
    def _process_sync(self, layer_idx, gpu_buffer, num_tokens, req_ids, tok_indices, probe_names):
        """Synchronous fallback"""
        scores = gpu_buffer[:num_tokens].cpu().numpy()
        self._store_results(scores, layer_idx, req_ids, tok_indices, probe_names)

    def process_queue(self, wait_all: bool = False):
        """
        Check for completed transfers and process them.
        Args:
            wait_all: If True, synchronize and process EVERYTHING (e.g. at end of generation)
        """
        if wait_all and torch.cuda.is_available():
             self.transfer_stream.synchronize()
             
        while self.pending_queue:
            pending = self.pending_queue[0]
            
            # Check if transfer is done
            if not wait_all and not pending.transfer_event.query():
                # Not done yet, and we are strictly FIFO for simplicity
                # (Actually streams define order, so if 0 is not done, 1 is not done)
                break
            
            # Pop from queue
            self.pending_queue.popleft()
            
            # Convert to numpy (fast because it's pinned and synced)
            scores_np = pending.scores_cpu_pinned.numpy()
            
            self._store_results(
                scores_np,
                pending.layer_idx,
                pending.request_ids,
                pending.token_indices,
                pending.probe_names
            )
            
    def _store_results(self, scores_np, layer_idx, request_ids, token_indices, probe_names):
        for i, (req_id, token_idx) in enumerate(zip(request_ids, token_indices)):
            result = ProbeResult(
                request_id=req_id,
                token_idx=token_idx,
                layer_idx=layer_idx,
                probe_names=probe_names,
                scores=scores_np[i]
            )
            
            if self.queue is not None:
                # If using queue, send immediately
                self.queue.put(result.to_dict())
            else:
                # Store locally (fallback)
                result_dict = result.to_dict()
                # Add request_id to dict for easier processing by consumer
                result_dict["request_id"] = req_id
                
                if req_id not in self.completed_results:
                    self.completed_results[req_id] = []
                self.completed_results[req_id].append(result_dict)

    def get_results(self, request_id: str) -> list[dict]:
        return self.completed_results.pop(request_id, [])

    def flush(self):
        """Ensure all pending results are processed"""
        self.process_queue(wait_all=True)
