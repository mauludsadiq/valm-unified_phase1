from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml

from valm.integration.coordinator import SystemCoordinator


@dataclass
class VALMUnified:
    config: Dict[str, Any]
    coordinator: SystemCoordinator
    systems: Dict[str, Dict[str, Any]]

    @classmethod
    def from_yaml(cls, path: str) -> "VALMUnified":
        p = Path(path)
        cfg = yaml.safe_load(p.read_text())
        coord = SystemCoordinator(cfg)
        systems = coord.load_all()
        return cls(config=cfg, coordinator=coord, systems=systems)

    def report_loaded(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for name, s in self.systems.items():
            out[name] = {
                "ok": bool(s.get("ok", False)),
                "stub": bool(s.get("stub", False)),
                "repo_path": s.get("repo_path"),
                "loaded_count": len(s.get("loaded", {})),
                "error_count": len(s.get("errors", [])),
            }
        return out
