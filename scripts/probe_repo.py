from __future__ import annotations

import argparse
import importlib
import pkgutil
import sys
from pathlib import Path
from typing import Any, Dict, List


def list_top_level_modules(repo_path: Path) -> List[str]:
    sys.path.insert(0, str(repo_path))
    try:
        return sorted({m.name for m in pkgutil.iter_modules([str(repo_path)])})
    finally:
        sys.path.pop(0)


def try_import(repo_path: Path, module: str) -> Dict[str, Any]:
    sys.path.insert(0, str(repo_path))
    try:
        mod = importlib.import_module(module)
        return {"module": module, "import_ok": True, "file": getattr(mod, "__file__", None)}
    except Exception as e:
        return {"module": module, "import_ok": False, "error": repr(e)}
    finally:
        sys.path.pop(0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("repo_path", type=str)
    ap.add_argument("--try", dest="try_modules", nargs="*", default=[])
    args = ap.parse_args()

    repo = Path(args.repo_path).expanduser().resolve()
    print(f"repo_path={repo}")
    if not repo.exists():
        raise SystemExit(f"ERROR: path does not exist: {repo}")

    tops = list_top_level_modules(repo)
    print("top_level_modules:")
    for n in tops:
        print(f"  - {n}")

    if args.try_modules:
        print("import_tests:")
        for m in args.try_modules:
            r = try_import(repo, m)
            if r["import_ok"]:
                print(f"  OK  {m}  file={r.get('file')}")
            else:
                print(f"  BAD {m}  error={r.get('error')}")


if __name__ == "__main__":
    main()
