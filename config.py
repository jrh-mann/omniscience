from pydantic import BaseModel, Field, field_validator
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
    
    @field_validator("weight_path")
    @classmethod
    def check_exists(cls, v: Path) -> Path:
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
    
    @field_validator("probes")
    @classmethod
    def check_unique_names(cls, v: list[ProbeConfig]) -> list[ProbeConfig]:
        names = [p.name for p in v]
        if len(names) != len(set(names)):
            raise ValueError("Probe names must be unique")
        return v
    
    def get_probes_for_layer(self, layer_idx: int) -> list[ProbeConfig]:
        return [p for p in self.probes if p.layer_idx == layer_idx]
