from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml
from pydantic import BaseModel, Field, ConfigDict, model_validator

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


class MemoryType(str, Enum):
    KV = "kv"
    PARAMETER = "parameter"
    HYBRID = "hybrid"


class KVMemoryConfig(BaseModel):
    """KV-Memory v0 configuration"""

    M: int = Field(256, description="Number of memory slots")
    g_write: float = Field(0.9, ge=0.0, le=1.0, description="EMA rate")
    tau_reuse: float = Field(0.8, ge=0.0, le=1.0, description="Reuse threshold")
    tau_novel: float = Field(0.3, ge=0.0, le=1.0, description="Novelty threshold")
    L: int = Field(4096, description="Attention window length (tokenizer truncation)")
    epsilon: float = Field(1e-8, description="Numerical guard")

    @model_validator(mode="after")
    def _check_thresholds(self):
        if self.tau_novel >= self.tau_reuse:
            raise ValueError("tau_novel must be < tau_reuse")
        return self


class ParameterMemoryConfig(BaseModel):
    """TTT/Surprise-gated parameter memory configuration"""

    trust_radius: float = Field(0.05, ge=0.0, description="Max parameter drift")
    learn_threshold: float = Field(0.1, ge=0.0, description="Surprise threshold")
    eta_base: float = Field(1e-3, ge=0.0, description="Base learning rate")
    lambda_base: float = Field(0.01, ge=0.0, le=1.0, description="Base retention")
    alpha: float = Field(0.9, ge=0.0, le=1.0, description="EMA smoothing")

    gate_type: str = Field("sigmoid", description="budget|sigmoid|miras")
    beta_gate: float = Field(1.0, description="Gate steepness")
    tau_gate: float = Field(0.5, description="Gate threshold")


class VerificationConfig(BaseModel):
    """CBench verification configuration"""

    thresholds: Dict[str, float] = Field(
        default_factory=lambda: {
            "epsilon_S": 0.2,
            "epsilon_I": 0.2,
            "epsilon_F": 0.2,
            "max_rti_violations": 0.0,
        }
    )
    schema_path: Optional[Path] = None
    attestation_key: Optional[str] = None
    require_signature: bool = Field(True, description="Require cryptographic signatures")

    @model_validator(mode="after")
    def _check_thresholds(self):
        required = {"epsilon_S", "epsilon_I", "epsilon_F", "max_rti_violations"}
        if not required.issubset(set(self.thresholds.keys())):
            raise ValueError(f"Thresholds must contain {required}")
        return self


class RegistryConfig(BaseModel):
    """Merkle registry configuration"""

    merkle_depth: int = Field(32, ge=1, description="Merkle tree depth")
    storage_type: str = Field("sqlite", description="sqlite|postgres|memory")
    storage_path: Optional[Path] = Field(None, description="Database path")
    backup_interval: int = Field(1000, ge=1, description="Steps between backups")
    witness_interval: int = Field(100, ge=1, description="Steps between full witnesses")


class VALMConfig(BaseModel):
    """Complete VALM configuration"""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    system_name: str = Field("valm-unified", description="System identifier")
    version: str = Field("1.0.0", description="System version")

    memory_type: MemoryType = Field(MemoryType.HYBRID, description="Memory system type")
    kv_memory: KVMemoryConfig = Field(default_factory=KVMemoryConfig)
    param_memory: ParameterMemoryConfig = Field(default_factory=ParameterMemoryConfig)
    verification: VerificationConfig = Field(default_factory=VerificationConfig)
    registry: RegistryConfig = Field(default_factory=RegistryConfig)

    kv_to_param_threshold: float = Field(0.7, ge=0.0, le=1.0)
    param_to_kv_threshold: float = Field(0.9, ge=0.0, le=1.0)

    batch_size: int = Field(1, ge=1)
    device: str = Field(default_factory=lambda: ("cuda" if (torch and torch.cuda.is_available()) else "cpu"))
    dtype: str = Field(default_factory=lambda: ("float16" if (torch and torch.cuda.is_available()) else "float32"))

    log_level: str = Field("INFO")
    log_memory_events: bool = Field(True)
    log_verification: bool = Field(True)

    validate_on_update: bool = Field(True)
    revert_on_failure: bool = Field(True)

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "VALMConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)

    def to_yaml(self, path: Union[str, Path]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.model_dump(mode="json"), f, sort_keys=False)

    def update_from_dict(self, updates: Dict[str, Any]) -> "VALMConfig":
        data = self.model_dump(mode="json")
        data.update(updates)
        return self.__class__(**data)
