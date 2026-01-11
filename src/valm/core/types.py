from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import torch
from pydantic import BaseModel


@dataclass
class MemoryEvent:
    """Unified memory operation event"""

    event_type: str
    timestamp: int
    surprise: float
    content_hash: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": self.event_type,
            "timestamp": int(self.timestamp),
            "surprise": float(self.surprise),
            "content_hash": self.content_hash,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryEvent":
        return cls(**data)


@dataclass
class KVState:
    """KV-Memory state snapshot"""

    keys: torch.Tensor  # [M, d]
    values: torch.Tensor  # [M, d]
    ages: torch.Tensor  # [M]
    slot_mask: torch.Tensor  # [M] bool

    @property
    def occupied_slots(self) -> int:
        return int(self.slot_mask.sum().item())

    @property
    def state_digest(self) -> str:
        import hashlib

        # Deterministic byte layout: float32/float16 etc. come from tensor dtype.
        data = b"".join(
            [
                self.keys.detach().cpu().contiguous().numpy().tobytes(),
                self.values.detach().cpu().contiguous().numpy().tobytes(),
                self.ages.detach().cpu().contiguous().numpy().tobytes(),
                self.slot_mask.detach().cpu().contiguous().numpy().tobytes(),
            ]
        )
        return hashlib.sha256(data).hexdigest()


@dataclass
class ParameterState:
    """Parameter memory state snapshot"""

    parameters: Dict[str, torch.Tensor]
    theta_0: Dict[str, torch.Tensor]
    step: int
    total_drift: float

    @property
    def state_digest(self) -> str:
        import hashlib, struct

        hasher = hashlib.sha256()
        for name in sorted(self.parameters.keys()):
            t = self.parameters[name].detach().cpu().contiguous()
            hasher.update(name.encode("utf-8"))
            hasher.update(t.numpy().tobytes())
        hasher.update(struct.pack("<Q", int(self.step)))
        hasher.update(struct.pack("<d", float(self.total_drift)))
        return hasher.hexdigest()


@dataclass
class VerificationResult:
    overall_pass: bool
    final_pass: bool
    module_results: Dict[str, bool]
    proof: Optional[Dict[str, Any]] = None
    witness_hash: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_pass": bool(self.overall_pass),
            "final_pass": bool(self.final_pass),
            "module_results": dict(self.module_results),
            "witness_hash": self.witness_hash,
        }


@dataclass
class VerifiedOutput:
    text: str
    tokens: torch.Tensor
    kv_state_digest: str
    param_state_digest: str
    surprise: float
    memory_events: List[MemoryEvent]
    verification_result: VerificationResult
    step: int
    timestamp: datetime = field(default_factory=datetime.now)

    def summary(self) -> Dict[str, Any]:
        return {
            "step": int(self.step),
            "text_preview": (self.text[:100] + "...") if len(self.text) > 100 else self.text,
            "surprise": float(self.surprise),
            "verification_passed": bool(self.verification_result.final_pass),
            "kv_state": self.kv_state_digest[:16],
            "param_state": self.param_state_digest[:16],
            "event_count": len(self.memory_events),
            "event_types": [e.event_type for e in self.memory_events],
        }


class ModelState(BaseModel):
    step: int
    parameters: Dict[str, List[float]]
    metadata: Dict[str, Any]
    parent_hash: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True
