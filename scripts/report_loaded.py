from __future__ import annotations
import sys

import json
from pathlib import Path

import yaml

from valm.unified.system import VALMUnified


def main() -> None:
    cfg_path = Path("configs/unified.yaml")
    cfg = yaml.safe_load(cfg_path.read_text())

    def _syspath_add_repo(repo_path: str) -> None:
        repo_root = Path(repo_path).resolve()
        src_dir = repo_root / "src"
        if src_dir.is_dir():
            sys.path.insert(0, str(src_dir))
        sys.path.insert(0, str(repo_root))

    for _name, _sys in cfg.get("systems", {}).items():
        _rp = _sys.get("repo_path")
        if _rp:
            _syspath_add_repo(_rp)


    u = VALMUnified.from_yaml(str(cfg_path))
    rpt = u.report_loaded()

    # Print stable JSON so CI / humans can diff it.
    print(json.dumps(rpt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
