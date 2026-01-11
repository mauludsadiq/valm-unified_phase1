from __future__ import annotations

import argparse
import json

from valm.unified.integrated import IntegratedVALM


def main() -> int:
    ap = argparse.ArgumentParser(prog="run_integrated")
    ap.add_argument("--config", default="configs/unified.yaml")
    args = ap.parse_args()

    integ = IntegratedVALM.from_yaml(args.config)

    print(json.dumps({"systems": integ.system_status()}, sort_keys=True, indent=2))
    print(json.dumps({"adapters": integ.adapter_status()}, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
