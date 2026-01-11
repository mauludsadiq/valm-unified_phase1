from __future__ import annotations

import argparse
import json
from pathlib import Path

from valm.integrated.run import write_integrated_run_witness


def main() -> int:
    ap = argparse.ArgumentParser(prog="integrated_run_witness")
    ap.add_argument("--load-witness", default="runs/unified_load_witness.json")
    ap.add_argument("--out", default="runs/integrated_run_witness.json")
    args = ap.parse_args()

    w = write_integrated_run_witness(args.load_witness, args.out)
    p = Path(args.out)
    print(str(p.resolve()))
    print(json.dumps({"ok": True, "load_witness_sha256": w["load_witness"]["sha256"], "adapters": {k: v["action"]["status"] for k, v in w["adapters"].items()}}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
