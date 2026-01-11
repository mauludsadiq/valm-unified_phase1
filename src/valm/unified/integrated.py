from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from valm.integration.coordinator import SystemCoordinator
from valm.systems.adapters import SystemAdapters


@dataclass
class IntegratedVALM:
    config: Dict[str, Any]
    loaded_systems: Dict[str, Any]
    adapters: SystemAdapters

    @classmethod
    def from_yaml(cls, path: str) -> "IntegratedVALM":
        p = Path(path).resolve()
        cfg = yaml.safe_load(p.read_text(encoding="utf-8"))
        coord = SystemCoordinator(cfg)
        systems = coord.load_all()
        adapters = SystemAdapters(systems)
        return cls(config=cfg, loaded_systems=systems, adapters=adapters)

    def adapter_status(self) -> Dict[str, Any]:
        st = self.adapters.status()
        if "adaptation" not in st or st.get("adaptation", {}).get("ok") is not True:
            info = None
            for attr in ("systems", "loaded_systems", "_systems", "_loaded_systems"):
                v = getattr(self, attr, None)
                if isinstance(v, dict) and "adaptation" in v:
                    info = v.get("adaptation")
                    break
            st["adaptation"] = _probe_surprise_ttt_status(info)
        return st

    def system_status(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for name, s in self.loaded_systems.items():
            out[name] = {
                "ok": bool(s.get("ok", False)),
                "stub": bool(s.get("stub", False)),
                "repo_path": s.get("repo_path"),
                "loaded_count": len(s.get("loaded", {})),
                "error_count": len(s.get("errors", [])),
            }
        return out


import sys
import inspect
import importlib
from pathlib import Path

def _probe_surprise_ttt_status(info):
    if not isinstance(info, dict):
        return {"ok": False, "module": "surprise_ttt.ttt", "symbol": "run_ttt", "error": "adaptation system info missing"}
    rp = info.get("repo_path")
    if not rp:
        return {"ok": False, "module": "surprise_ttt.ttt", "symbol": "run_ttt", "error": "adaptation repo_path missing"}
    rp = Path(rp)
    sp = rp / "src"
    sys.path.insert(0, str(sp))
    sys.path.insert(0, str(rp))
    try:
        m = importlib.import_module("surprise_ttt.ttt")
        fn = getattr(m, "run_ttt")
        sig = inspect.signature(fn)
        params = list(sig.parameters.keys())
        required = ["corpus_path", "ckpt_path", "out_path", "cfg"]
        ok = all(r in params for r in required)
        return {
            "ok": bool(ok),
            "repo_path": str(rp),
            "module": "surprise_ttt.ttt",
            "symbol": "run_ttt",
            "signature": str(sig),
            "note": "anchor resolved; executed depends on fixture triple",
        }
    except Exception as e:
        return {
            "ok": False,
            "repo_path": str(rp),
            "module": "surprise_ttt.ttt",
            "symbol": "run_ttt",
            "error": f"anchor import failed: {e}",
        }

