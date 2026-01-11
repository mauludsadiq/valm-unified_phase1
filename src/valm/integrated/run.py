from __future__ import annotations

import hashlib
import importlib
import json
import platform
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _sha256_bytes(b: bytes) -> str:
    h = hashlib.sha256()
    h.update(b)
    return h.hexdigest()

def _sha256_file(p: Path) -> str:
    return _sha256_bytes(p.read_bytes())

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

@contextmanager
def _prepend_syspath(paths: List[str]) -> Iterator[None]:
    """
    Prepend paths to sys.path for the duration of the context, then restore.
    This avoids "sticky" sys.path mutations across adapters/tests.
    """
    old = list(sys.path)
    try:
        for x in reversed(paths):
            if x and x not in sys.path:
                sys.path.insert(0, x)
        yield
    finally:
        sys.path[:] = old

def _import_symbol(module: str, symbol: str) -> Tuple[Optional[Any], Optional[str]]:
    try:
        m = importlib.import_module(module)
    except Exception as e:
        return None, f"{type(e).__name__}({e!r})"
    try:
        obj = getattr(m, symbol)
    except Exception as e:
        return None, f"{type(e).__name__}({e!r})"
    return obj, None

def _format_sig(obj: Any) -> str:
    try:
        import inspect  # local import safe
        return str(inspect.signature(obj))
    except Exception:
        return "<unavailable>"

# ---------------------------------------------------------------------
# Adapter actions
# ---------------------------------------------------------------------

def _action_import_only(module: str, symbol: str) -> Dict[str, Any]:
    obj, err = _import_symbol(module, symbol)
    if err is not None:
        return {"status": "error", "error": err}
    return {"status": "ok", "error": None}

def _action_adaptation_run_ttt(module: str, symbol: str) -> Dict[str, Any]:
    """
    Integrated smoke action for Surprise-TTT.

    Current Surprise-TTT anchor is a training-style entrypoint:
      run_ttt(corpus_path, ckpt_path, out_path, cfg, device="cpu") -> TTTResult

    In this repo (valm-unified_phase1), we do NOT assume:
    - a default checkpoint exists
    - a default corpus exists
    - a stable default cfg constructor exists

    So: verify import + symbol resolution, then SKIP execution with a
    deterministic reason that preserves the legacy "no compatible ..." phrase.
    """
    fn, err = _import_symbol(module, symbol)
    if err is not None:
        return {"status": "error", "error": err}

    sig = _format_sig(fn)

    # Legacy phrase retained so old tests/greps don't break.
    msg = (
        "no compatible run_ttt signature for integrated smoke; "
        f"got signature={sig}. "
        "This anchor requires (corpus_path, ckpt_path, out_path, cfg[, device]) "
        "and valm-unified_phase1 does not ship a canonical ckpt/corpus/cfg for execution."
    )
    return {"status": "skipped", "error": msg}

def _adapter_action(name: str, anchor: Dict[str, Any]) -> Dict[str, Any]:
    module = str(anchor.get("module") or "")
    symbol = str(anchor.get("symbol") or "")
    if not module or not symbol:
        return {"status": "error", "error": "missing module/symbol in adapter anchor"}

    if name == "adaptation" and module == "surprise_ttt.ttt" and symbol == "run_ttt":
        return _action_adaptation_run_ttt(module, symbol)

    # Default: import + symbol presence is enough for capability/specialization/verification
    return _action_import_only(module, symbol)

# ---------------------------------------------------------------------
# Public API (used by tests + scripts)
# ---------------------------------------------------------------------

def make_integrated_run_witness(load_witness_path: str) -> Dict[str, Any]:
    lw_p = Path(load_witness_path)
    lw_bytes = lw_p.read_bytes()
    lw_sha = _sha256_bytes(lw_bytes)
    lw = json.loads(lw_bytes.decode("utf-8"))

    adapters_in = lw.get("adapters", {}) or {}
    out_adapters: Dict[str, Any] = {}

    for name, a in adapters_in.items():
        syspath_prefix = a.get("syspath_prefix", []) or []
        anchor = a.get("anchor", {}) or {}

        with _prepend_syspath(list(syspath_prefix)):
            action = _adapter_action(str(name), dict(anchor))

        out_adapters[str(name)] = {
            "name": str(name),
            "repo_path": a.get("repo_path"),
            "vcs": a.get("vcs", {}),
            "anchor": anchor,
            "syspath_prefix": list(syspath_prefix),
            "action": action,
        }

    ok = True
    for _, a in out_adapters.items():
        if a.get("action", {}).get("status") == "error":
            ok = False
            break

    return {
        "witness_type": "valm_integrated_run_witness",
        "witness_version": "0.1.0",
        "created_utc": _utc_now_iso(),
        "python": {
            "version": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
        },
        "load_witness": {
            "path": str(lw_p),
            "sha256": lw_sha,
            "witness_type": lw.get("witness_type"),
            "witness_version": lw.get("witness_version"),
            "created_utc": lw.get("created_utc"),
        },
        "adapters": out_adapters,
        "ok": bool(ok),
    }

def write_integrated_run_witness(load_witness_path: str, out_path: str) -> Dict[str, Any]:
    w = make_integrated_run_witness(load_witness_path)
    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(w, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return w


import hashlib
from pathlib import Path
from typing import Any, Dict, Optional

def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def _read_cfg_data(cfg_path: Path) -> Dict[str, Any]:
    if cfg_path.suffix.lower() in (".json",):
        import json
        return json.loads(cfg_path.read_text(encoding="utf-8"))
    if cfg_path.suffix.lower() in (".yml", ".yaml"):
        import yaml
        return yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    raise ValueError(f"unsupported cfg file type: {cfg_path.name}")

def _resolve_ttt_fixture(load_cfg: Dict[str, Any]) -> Dict[str, Optional[str]]:
    integ = (load_cfg or {}).get("integrated", {}) or {}
    a = (integ.get("adaptation_smoke", {}) or {})
    return {
        "corpus_path": a.get("corpus_path"),
        "ckpt_path": a.get("ckpt_path"),
        "cfg_path": a.get("cfg_path"),
        "device": a.get("device", "cpu"),
    }

def _run_adaptation_action(anchor: Dict[str, Any], load_cfg: Dict[str, Any]) -> Dict[str, Any]:
    import importlib
    import inspect
    import tempfile

    mod = importlib.import_module(anchor["module"])
    fn = getattr(mod, anchor["symbol"])
    sig = inspect.signature(fn)
    params = list(sig.parameters.keys())

    required = ["corpus_path", "ckpt_path", "out_path", "cfg"]
    if not all(r in params for r in required):
        return {
            "status": "skipped",
            "executed": False,
            "signature": str(sig),
            "error": f"unexpected run_ttt signature; need params {required}",
        }

    fx = _resolve_ttt_fixture(load_cfg)
    root = Path.cwd()

    corpus = (root / fx["corpus_path"]).resolve() if fx["corpus_path"] else None
    ckpt = (root / fx["ckpt_path"]).resolve() if fx["ckpt_path"] else None
    cfgp = (root / fx["cfg_path"]).resolve() if fx["cfg_path"] else None

    missing = []
    for name, p in [("corpus_path", corpus), ("ckpt_path", ckpt), ("cfg_path", cfgp)]:
        if p is None or not p.exists():
            missing.append(name)

    if missing:
        return {
            "status": "ok",
            "executed": False,
            "signature": str(sig),
            "note": "anchor resolved; canonical ttt fixture triple not present",
            "missing": missing,
            "expected": {
                "corpus_path": str(corpus) if corpus else None,
                "ckpt_path": str(ckpt) if ckpt else None,
                "cfg_path": str(cfgp) if cfgp else None,
            },
        }

    cfg_data = _read_cfg_data(cfgp)

    cfg_cls = getattr(mod, "TTTcfg", None)
    if cfg_cls is None:
        return {
            "status": "ok",
            "executed": False,
            "signature": str(sig),
            "note": "anchor resolved; TTTcfg class not found in module",
            "cfg_path": str(cfgp),
        }

    cfg_obj = cfg_cls(**cfg_data)

    out_dir = Path(tempfile.mkdtemp(prefix="valm_ttt_smoke_"))
    out_path = out_dir / "ttt_out.json"

    device = fx.get("device") or "cpu"

    try:
        res = fn(
            corpus_path=str(corpus),
            ckpt_path=str(ckpt),
            out_path=str(out_path),
            cfg=cfg_obj,
            device=device,
        )
    except Exception as e:
        return {
            "status": "error",
            "executed": False,
            "signature": str(sig),
            "error": f"run_ttt execution failed: {e}",
        }

    out_sha = _sha256_file(out_path) if out_path.exists() else None

    return {
        "status": "ok",
        "executed": True,
        "signature": str(sig),
        "inputs": {
            "corpus_path": str(corpus),
            "corpus_sha256": _sha256_file(corpus),
            "ckpt_path": str(ckpt),
            "ckpt_sha256": _sha256_file(ckpt),
            "cfg_path": str(cfgp),
            "cfg_sha256": _sha256_file(cfgp),
            "device": device,
        },
        "output": {
            "out_path": str(out_path),
            "out_sha256": out_sha,
        },
        "result_type": type(res).__name__,
    }
