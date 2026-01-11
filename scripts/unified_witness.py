from __future__ import annotations

import argparse
import json
from pathlib import Path

from valm.unified.witness import write_unified_load_witness


def main() -> int:
    ap = argparse.ArgumentParser(prog="unified_witness")
    ap.add_argument("--config", default="configs/unified.yaml")
    ap.add_argument("--out", default="runs/unified_load_witness.json")
    args = ap.parse_args()

    w = write_unified_load_witness(args.config, args.out)
    p = Path(args.out)
    print(str(p.resolve()))
    print(json.dumps({"ok": True, "systems": {k: v.get("status", {}) for k, v in w.get("systems", {}).items()}}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
