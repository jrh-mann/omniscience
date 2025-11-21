import torch.distributed as dist
import logging

logger = logging.getLogger(__name__)

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
