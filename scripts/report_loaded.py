from __future__ import annotations

import json
from pathlib import Path

import yaml

from valm.unified.system import VALMUnified


def main() -> None:
    cfg_path = Path("configs/unified.yaml")
    cfg = yaml.safe_load(cfg_path.read_text())

    u = VALMUnified.from_yaml(str(cfg_path))
    rpt = u.report_loaded()

    # Print stable JSON so CI / humans can diff it.
    print(json.dumps(rpt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
