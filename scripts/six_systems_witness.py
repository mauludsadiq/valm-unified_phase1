from __future__ import annotations

import argparse
import json
from pathlib import Path
from valm.unified.six_systems_witness import build_six_systems_witness

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--load-witness", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    lw = json.loads(Path(args.load_witness).read_text(encoding="utf-8"))
    d = build_six_systems_witness(lw)
    Path(args.out).write_text(json.dumps(d, indent=2, sort_keys=True), encoding="utf-8")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
