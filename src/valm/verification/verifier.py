from __future__ import annotations

from typing import Any, Dict, Optional

from ..core.types import VerificationResult


class VerificationSystem:
    """CBench verification system (Phase 1 stub)."""

    def __init__(self, config):
        self.config = config
        self.last_result: Optional[VerificationResult] = None

    def verify(self, bundle: Dict[str, Any]) -> VerificationResult:
        # Phase 1: always pass; Phase 2 will implement CIA/RST/RTI modules.
        res = VerificationResult(
            overall_pass=True,
            final_pass=True,
            module_results={"CIA": True, "RST": True, "RTI": True},
            witness_hash="stub_witness_hash",
        )
        self.last_result = res
        return res

    def create_initial_witness(self, model: Any, config: Any) -> Dict[str, Any]:
        return {
            "type": "initial_witness",
            "model_hash": "stub_model_hash",
            "config_hash": "stub_config_hash",
        }
