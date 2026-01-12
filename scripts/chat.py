from __future__ import annotations

import argparse
from valm.chat.runtime import run_chat, DecodeCfg

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    args = ap.parse_args()

    cfg = DecodeCfg(max_new_tokens=args.max_new, temperature=args.temp, top_p=args.top_p)
    run_chat(args.config, cfg)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
