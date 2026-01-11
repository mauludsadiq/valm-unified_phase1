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
        return self.adapters.status()

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
