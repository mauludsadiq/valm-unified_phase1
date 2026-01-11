from __future__ import annotations

import json
import subprocess
from pathlib import Path

def test_six_systems_witness_smoke(tmp_path: Path) -> None:
    repo = Path.cwd()
    load_witness = repo / "runs" / "unified_load_witness.json"
    out = tmp_path / "six_systems_witness.json"

    if not load_witness.exists():
        subprocess.check_call(
            ["python", "scripts/unified_witness.py", "--config", "configs/unified.yaml", "--out", str(load_witness)]
        )

    subprocess.check_call(["python", "scripts/six_systems_witness.py", "--load-witness", str(load_witness), "--out", str(out)])

    d = json.loads(out.read_text(encoding="utf-8"))
    assert "systems" in d
    for k in ["llm_nature", "smallchat", "kv_memory", "mlp_xor", "cbench", "surprise_ttt"]:
        assert k in d["systems"]
        assert "action" in d["systems"][k]
        assert "status" in d["systems"][k]["action"]
        assert "executed" in d["systems"][k]["action"]
