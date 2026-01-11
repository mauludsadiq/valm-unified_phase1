from __future__ import annotations

import json
from pathlib import Path

from valm.integrated.run import make_integrated_run_witness


def test_adaptation_action_status_is_deterministic() -> None:
    lw = Path("runs/unified_load_witness.json")
    assert lw.exists()

    w = make_integrated_run_witness(str(lw))
    a = w["adapters"]["adaptation"]["action"]["status"]
    assert a in ["ok", "skipped", "error"]

    if a == "error":
        err = w["adapters"]["adaptation"]["action"]["error"]
        assert isinstance(err, str) and len(err) > 0
