from __future__ import annotations

import hashlib
import importlib
import json
import platform
import sys
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from valm.unified.system import VALMUnified
from valm.unified.adapters_witness import adapter_anchor_witness


def _sha256_bytes(b: bytes) -> str:
    h = hashlib.sha256()
    h.update(b)
    return h.hexdigest()


def _sha256_file(p: Path) -> str:
    return _sha256_bytes(p.read_bytes())


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _safe_module_file(mod: Any) -> Optional[str]:
    return getattr(mod, "__file__", None)


def _git_info(repo: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {"head": None, "describe": None, "dirty": None, "error": None}
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        ).stdout.strip()
        out["head"] = head

        desc = subprocess.run(
            ["git", "describe", "--tags", "--always", "--dirty=-dirty"],
            cwd=str(repo),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        ).stdout.strip()
        out["describe"] = desc

        dirty_rc = subprocess.run(
            ["git", "diff", "--quiet"],
            cwd=str(repo),
        ).returncode
        out["dirty"] = bool(dirty_rc != 0)
    except Exception as e:
        out["error"] = f"{type(e).__name__}({e!r})"
    return out


def _import_and_locate(module: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    try:
        m = importlib.import_module(module)
    except Exception as e:
        return None, None, f"{type(e).__name__}({e!r})"
    f = _safe_module_file(m)
    if not f:
        return None, None, None
    fp = Path(f)
    try:
        digest = _sha256_file(fp)
    except Exception as e:
        return str(fp), None, f"{type(e).__name__}({e!r})"
    return str(fp), digest, None


def _symbol_presence(module: str, sym: str) -> Tuple[bool, Optional[str]]:
    try:
        m = importlib.import_module(module)
    except Exception as e:
        return False, f"{type(e).__name__}({e!r})"
    return hasattr(m, sym), None


def make_unified_load_witness(config_path: str) -> Dict[str, Any]:
    cfg_p = Path(config_path)
    cfg_text = cfg_p.read_text(encoding="utf-8")
    cfg = yaml.safe_load(cfg_text)

    u = VALMUnified.from_yaml(str(cfg_p))

    systems_cfg = cfg.get("systems", {})
    systems_report = u.report_loaded()

    per_system: Dict[str, Any] = {}
    for name, scfg in systems_cfg.items():
        imports = scfg.get("import", []) or []
        import_rows: List[Dict[str, Any]] = []
        for imp in imports:
            mod = imp.get("module")
            syms = imp.get("symbols", []) or []
            mod_file, mod_sha256, mod_import_error = _import_and_locate(mod)
            sym_rows: List[Dict[str, Any]] = []
            for s in syms:
                present, sym_err = _symbol_presence(mod, s)
                sym_rows.append(
                    {
                        "symbol": s,
                        "present": bool(present),
                        "error": sym_err,
                    }
                )
            import_rows.append(
                {
                    "module": mod,
                    "module_file": mod_file,
                    "module_sha256": mod_sha256,
                    "import_error": mod_import_error,
                    "symbols": sym_rows,
                }
            )

        rep = systems_report.get(name, {})
        per_system[name] = {
            "enabled": bool(scfg.get("enabled", False)),
            "allow_stub": bool(scfg.get("allow_stub", False)),
            "repo_path": scfg.get("repo_path"),
            "vcs": _git_info(Path(scfg.get("repo_path")).resolve()),
            "status": rep,
            "imports": import_rows,
        }

    out: Dict[str, Any] = {
        "witness_type": "valm_unified_load_witness",
        "witness_version": "0.1.0",
        "created_utc": _utc_now_iso(),
        "python": {
            "version": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
        },
        "config": {
            "path": str(cfg_p),
            "sha256": _sha256_bytes(cfg_text.encode("utf-8")),
        },
        "systems": per_system,
        "adapters": adapter_anchor_witness(systems_report),
    }
    return out


def write_unified_load_witness(config_path: str, out_path: str) -> Dict[str, Any]:
    w = make_unified_load_witness(config_path)
    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(w, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return w
