from __future__ import annotations

import argparse
import json

from valm.unified.system import VALMUnified


def main() -> int:
    ap = argparse.ArgumentParser(prog="run_unified")
    ap.add_argument("--config", default="configs/unified.yaml")
    args = ap.parse_args()

    u = VALMUnified.from_yaml(args.config)
    print(json.dumps(u.report_loaded(), sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
