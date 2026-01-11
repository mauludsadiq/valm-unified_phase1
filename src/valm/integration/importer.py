from __future__ import annotations

import importlib
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@contextmanager
def sys_path_prepend(path: Path):
    p = str(path)
    sys.path.insert(0, p)
    try:
        yield
    finally:
        for i, x in enumerate(list(sys.path)):
            if x == p:
                sys.path.pop(i)
                break


def import_symbol(module: str, symbol: str) -> Any:
    mod = importlib.import_module(module)
    return getattr(mod, symbol)


@dataclass(frozen=True)
class LoadedSymbol:
    module: str
    symbol: str
    ok: bool
    value: Optional[Any] = None
    error: Optional[str] = None


def load_system_from_spec(spec: Dict[str, Any]) -> Dict[str, Any]:
    repo_path = Path(spec["repo_path"]).expanduser().resolve()
    if not repo_path.exists():
        raise FileNotFoundError(f"repo_path does not exist: {repo_path}")

    import_list = spec.get("import", [])
    out: Dict[str, Any] = {"repo_path": str(repo_path), "loaded": {}, "errors": []}

    with sys_path_prepend(repo_path):
        for item in import_list:
            m = item["module"]
            for sym in item.get("symbols", []):
                key = f"{m}:{sym}"
                try:
                    val = import_symbol(m, sym)
                    out["loaded"][key] = LoadedSymbol(m, sym, True, val, None).__dict__
                except Exception as e:
                    out["loaded"][key] = LoadedSymbol(m, sym, False, None, repr(e)).__dict__
                    out["errors"].append({"module": m, "symbol": sym, "error": repr(e)})

    out["ok"] = (len(out["errors"]) == 0)
    return out
