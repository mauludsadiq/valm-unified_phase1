from __future__ import annotations

import json
from pathlib import Path

from valm.integrated.run import make_integrated_run_witness


def test_integrated_run_witness_shape() -> None:
    lw = Path("runs/unified_load_witness.json")
    assert lw.exists()

    w = make_integrated_run_witness(str(lw))
    assert w["witness_type"] == "valm_integrated_run_witness"
    assert w["witness_version"] == "0.1.0"
    assert "load_witness" in w and "sha256" in w["load_witness"]
    assert "adapters" in w
    for k in ["adaptation", "capability", "specialization", "verification"]:
        assert k in w["adapters"]
        assert "action" in w["adapters"][k]
