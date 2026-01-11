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

    d = build_six_systems_witness(Path(args.load_witness))

    systems = d.get("systems", {})
    if isinstance(systems, dict):
        for k, v in list(systems.items()):
            if isinstance(v, dict) and "action" not in v:
                systems[k] = {"action": v}

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(d, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(str(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
