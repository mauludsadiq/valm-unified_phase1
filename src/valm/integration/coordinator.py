from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from .importer import load_system_from_spec


@dataclass
class SystemCoordinator:
    """
    Pure loader. No repo assumptions. Config is the only truth.
    """
    config: Dict[str, Any]
    systems: Dict[str, Dict[str, Any]]

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.systems = {}

    def load_all(self) -> Dict[str, Dict[str, Any]]:
        systems_cfg = self.config.get("systems", {})
        for name, spec in systems_cfg.items():
            if not spec.get("enabled", True):
                self.systems[name] = {"enabled": False}
                continue

            try:
                self.systems[name] = load_system_from_spec(spec)
            except Exception as e:
                if bool(spec.get("allow_stub", False)):
                    self.systems[name] = {
                        "repo_path": spec.get("repo_path"),
                        "ok": False,
                        "stub": True,
                        "errors": [{"error": repr(e)}],
                        "loaded": {},
                    }
                else:
                    raise
        return self.systems
