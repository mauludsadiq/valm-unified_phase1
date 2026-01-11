from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class RepoPaths:
    repo_root: Path
    src_dir: Path

    @classmethod
    def from_repo_path(cls, repo_path: str) -> "RepoPaths":
        rr = Path(repo_path).resolve()
        return cls(repo_root=rr, src_dir=rr / "src")

    def as_syspath_prefix(self) -> List[str]:
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
        prefix = self.paths.as_syspath_prefix()
        sys.path = prefix + [p for p in sys.path if p not in prefix]
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._saved is not None:
            sys.path = self._saved


def import_module_safely(module: str, repo_path: str) -> Tuple[Optional[Any], Optional[str]]:
    try:
        with RepoSysPath(repo_path):
            m = importlib.import_module(module)
        return m, None
    except Exception as e:
        return None, f"{type(e).__name__}({e!r})"


def require_symbol(module: str, symbol: str, repo_path: str) -> Tuple[Optional[Any], Optional[str]]:
    m, err = import_module_safely(module, repo_path)
    if err is not None or m is None:
        return None, err
    if not hasattr(m, symbol):
        return None, f"AttributeError({symbol!r})"
    return getattr(m, symbol), None


@dataclass
class AdapterStatus:
    ok: bool
    name: str
    repo_path: str
    module: Optional[str] = None
    symbol: Optional[str] = None
    error: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "ok": bool(self.ok),
            "name": self.name,
            "repo_path": self.repo_path,
            "module": self.module,
            "symbol": self.symbol,
            "error": self.error,
        }


class SurpriseTTTAdapter:
    name = "adaptation"

    def __init__(self, system_info: Dict[str, Any]):
        self.system_info = system_info
        self.repo_path = str(system_info.get("repo_path") or "")

    def status(self) -> AdapterStatus:
        fn, err = require_symbol("surprise_ttt.ttt", "run_ttt", self.repo_path)
        ok = (err is None) and callable(fn)
        return AdapterStatus(
            ok=ok,
            name=self.name,
            repo_path=self.repo_path,
            module="surprise_ttt.ttt",
            symbol="run_ttt",
            error=err,
        )

    def run_ttt(self, *args: Any, **kwargs: Any) -> Any:
        fn, err = require_symbol("surprise_ttt.ttt", "run_ttt", self.repo_path)
        if err is not None or fn is None:
            raise RuntimeError(err or "missing run_ttt")
        with RepoSysPath(self.repo_path):
            return fn(*args, **kwargs)


class LLMNatureAdapter:
    name = "capability"

    def __init__(self, system_info: Dict[str, Any]):
        self.system_info = system_info
        self.repo_path = str(system_info.get("repo_path") or "")

    def status(self) -> AdapterStatus:
        m, err = import_module_safely("llm_nature", self.repo_path)
        ok = (err is None) and (m is not None)
        return AdapterStatus(
            ok=ok,
            name=self.name,
            repo_path=self.repo_path,
            module="llm_nature",
            symbol="__file__",
            error=err,
        )

    def module_public(self) -> List[str]:
        m, err = import_module_safely("llm_nature", self.repo_path)
        if err is not None or m is None:
            raise RuntimeError(err or "missing llm_nature")
        return sorted([x for x in dir(m) if not x.startswith("_")])


class SmallChatAdapter:
    name = "specialization"

    def __init__(self, system_info: Dict[str, Any]):
        self.system_info = system_info
        self.repo_path = str(system_info.get("repo_path") or "")

    def _try_import(self) -> Tuple[Optional[Any], Optional[str], Optional[str]]:
        for mod in ["smallchat_sft_dpo", "scripts"]:
            m, err = import_module_safely(mod, self.repo_path)
            if err is None and m is not None:
                return m, mod, None
        return None, None, "ImportError('no importable module found for specialization')"

    def status(self) -> AdapterStatus:
        m, mod, err = self._try_import()
        ok = (err is None) and (m is not None)
        return AdapterStatus(
            ok=ok,
            name=self.name,
            repo_path=self.repo_path,
            module=mod,
            symbol="__file__",
            error=err,
        )

    def entrypoints(self) -> List[str]:
        m, _, err = self._try_import()
        if err is not None or m is None:
            raise RuntimeError(err or "missing specialization module")
        out: List[str] = []
        for a in dir(m):
            if a.startswith("_"):
                continue
            obj = getattr(m, a)
            if callable(obj):
                out.append(a)
        return sorted(out)


class CBenchAdapter:
    name = "verification"

    def __init__(self, system_info: Dict[str, Any]):
        self.system_info = system_info
        self.repo_path = str(system_info.get("repo_path") or "")

    def status(self) -> AdapterStatus:
        fn, err = require_symbol("cbench.verifier_v0_1", "verify_artifacts", self.repo_path)
        ok = (err is None) and callable(fn)
        return AdapterStatus(
            ok=ok,
            name=self.name,
            repo_path=self.repo_path,
            module="cbench.verifier_v0_1",
            symbol="verify_artifacts",
            error=err,
        )

    def verify_artifacts(self, *args: Any, **kwargs: Any) -> Any:
        fn, err = require_symbol("cbench.verifier_v0_1", "verify_artifacts", self.repo_path)
        if err is not None or fn is None:
            raise RuntimeError(err or "missing verify_artifacts")
        with RepoSysPath(self.repo_path):
            return fn(*args, **kwargs)


class SystemAdapters:
    def __init__(self, loaded_systems: Dict[str, Any]):
        self.loaded_systems = loaded_systems
        self.adapters: Dict[str, Any] = {}
        self._create()

    def _create(self) -> None:
        s = self.loaded_systems

        if s.get("adaptation", {}).get("ok"):
            self.adapters["adaptation"] = SurpriseTTTAdapter(s["adaptation"])
        if s.get("capability", {}).get("ok"):
            self.adapters["capability"] = LLMNatureAdapter(s["capability"])
        if s.get("specialization", {}).get("ok"):
            self.adapters["specialization"] = SmallChatAdapter(s["specialization"])
        if s.get("verification", {}).get("ok"):
            self.adapters["verification"] = CBenchAdapter(s["verification"])

    def get(self, name: str) -> Optional[Any]:
        return self.adapters.get(name)

    def status(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for k, a in self.adapters.items():
            out[k] = a.status().as_dict()
        return out
