from __future__ import annotations

import hashlib
import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _sha256_bytes(b: bytes) -> str:
    h = hashlib.sha256()
    h.update(b)
    return h.hexdigest()


def _sha256_file(p: Path) -> str:
    return _sha256_bytes(p.read_bytes())


@dataclass(frozen=True)
class RepoPaths:
    repo_root: Path
    src_dir: Path

    @classmethod
    def from_repo_path(cls, repo_path: str) -> "RepoPaths":
        rr = Path(repo_path).resolve()
        return cls(repo_root=rr, src_dir=rr / "src")

    def syspath_prefix(self) -> List[str]:
        out: List[str] = []
        if self.src_dir.is_dir():
            out.append(str(self.src_dir))
        out.append(str(self.repo_root))
        return out


class RepoSysPath:
    def __init__(self, repo_path: str):
        self.paths = RepoPaths.from_repo_path(repo_path)
        self._saved: Optional[List[str]] = None

    def __enter__(self) -> "RepoSysPath":
        self._saved = list(sys.path)
        prefix = self.paths.syspath_prefix()
        sys.path = prefix + [p for p in sys.path if p not in prefix]
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._saved is not None:
            sys.path = self._saved


def _safe_module_file(mod: Any) -> Optional[str]:
    return getattr(mod, "__file__", None)


def import_locate(module: str, repo_path: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    try:
        with RepoSysPath(repo_path):
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


def symbol_presence(module: str, sym: str, repo_path: str) -> Tuple[bool, Optional[str]]:
    try:
        with RepoSysPath(repo_path):
            m = importlib.import_module(module)
    except Exception as e:
        return False, f"{type(e).__name__}({e!r})"
    return hasattr(m, sym), None


def adapter_anchor_witness(loaded_systems: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    anchors: Dict[str, Dict[str, str]] = {
        "adaptation": {"module": "surprise_ttt.ttt", "symbol": "run_ttt"},
        "capability": {"module": "llm_nature", "symbol": "__file__"},
        "specialization": {"module": "smallchat_sft_dpo", "symbol": "__file__"},
        "verification": {"module": "cbench.verifier_v0_1", "symbol": "verify_artifacts"},
    }

    for name, a in anchors.items():
        sysinfo = loaded_systems.get(name) or {}
        repo_path = str(sysinfo.get("repo_path") or "")
        rp = RepoPaths.from_repo_path(repo_path)

        module = a["module"]
        symbol = a["symbol"]

        mod_file, mod_sha256, import_err = import_locate(module, repo_path)
        sym_present, sym_err = symbol_presence(module, symbol, repo_path)

        out[name] = {
            "repo_path": repo_path,
            "syspath_prefix": rp.syspath_prefix(),
            "anchor": {"module": module, "symbol": symbol},
            "module_file": mod_file,
            "module_sha256": mod_sha256,
            "import_error": import_err,
            "symbol_present": bool(sym_present),
            "symbol_error": sym_err,
        }

    return out
