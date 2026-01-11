
from __future__ import annotations

import sys
import inspect
import importlib
from pathlib import Path

def _probe_surprise_ttt_anchor(repo_path):
    if not repo_path:
        return {
            "status": "error",
            "executed": False,
            "error": "repo_path missing for surprise_ttt",
        }

    rp = Path(str(repo_path))
    sp = rp / "src"
    sys.path.insert(0, str(sp))
    sys.path.insert(0, str(rp))

    try:
        m = importlib.import_module("surprise_ttt.ttt")
        fn = getattr(m, "run_ttt")
        sig = inspect.signature(fn)
        params = list(sig.parameters.keys())
        required = ["corpus_path", "ckpt_path", "out_path", "cfg"]
        ok = all(r in params for r in required)
        return {
            "status": "ok" if ok else "error",
            "executed": False,
            "module": "surprise_ttt.ttt",
            "symbol": "run_ttt",
            "signature": str(sig),
            "note": "anchor resolved; executed depends on canonical fixture triple (corpus, ckpt, cfg)",
            "repo_path": str(rp),
            "missing": [] if ok else ["signature_fields"],
        }
    except Exception as e:
        return {
            "status": "error",
            "executed": False,
            "error": f"anchor import failed: {e}",
            "repo_path": str(rp),
        }



def _normalize_action_wrappers(systems: dict) -> dict:
    out = {}
    for k, v in (systems or {}).items():
        if isinstance(v, dict) and "action" in v:
            out[k] = v
        else:
            out[k] = {"action": v}
    return out


import sys
import inspect
import importlib
from pathlib import Path


def _resolve_surprise_ttt_anchor_from_repo(repo_path: str):
    rp = Path(repo_path)
    sp = rp / "src"
    sys.path.insert(0, str(sp))
    sys.path.insert(0, str(rp))
    m = importlib.import_module("surprise_ttt.ttt")
    fn = getattr(m, "run_ttt")
    sig = inspect.signature(fn)
    return {"module": "surprise_ttt.ttt", "symbol": "run_ttt", "signature": str(sig)}

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import hashlib
import importlib
import inspect
import json
import tempfile

def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def read_cfg_data(cfg_path: Path) -> Dict[str, Any]:
    suf = cfg_path.suffix.lower()
    if suf == ".json":
        return json.loads(cfg_path.read_text(encoding="utf-8"))
    if suf in (".yml", ".yaml"):
        import yaml
        return yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    raise ValueError(f"unsupported cfg file type: {cfg_path.name}")

def import_anchor(anchor: Dict[str, Any]) -> Tuple[Optional[Any], Optional[str], Optional[str]]:
    mod_name = anchor.get("module")
    sym = anchor.get("symbol")
    if not mod_name or not sym:
        return None, None, "anchor missing module or symbol"
    try:
        mod = importlib.import_module(mod_name)
    except Exception as e:
        return None, None, f"import failed: {e}"
    try:
        obj = getattr(mod, sym)
    except Exception as e:
        return None, None, f"symbol lookup failed: {e}"
    sig = None
    try:
        if callable(obj):
            sig = str(inspect.signature(obj))
    except Exception:
        sig = None
    return obj, sig, None

def resolve_fixture_block(load_cfg: Dict[str, Any]) -> Dict[str, Any]:
    integ = (load_cfg or {}).get("integrated", {}) or {}
    a = (integ.get("six_systems_smoke", {}) or {})
    return {
        "surprise_ttt": (a.get("surprise_ttt", {}) or {}),
        "cbench": (a.get("cbench", {}) or {}),
    }

def run_surprise_ttt_action(anchor: Dict[str, Any], load_cfg: Dict[str, Any]) -> Dict[str, Any]:
    obj, sig, err = import_anchor(anchor)
    if err:
        return {"status": "error", "executed": False, "error": err, "signature": sig}

    if not callable(obj):
        return {"status": "error", "executed": False, "error": "anchor symbol not callable", "signature": sig}

    params = list(inspect.signature(obj).parameters.keys())
    required = ["corpus_path", "ckpt_path", "out_path", "cfg"]
    if not all(r in params for r in required):
        return {
            "status": "skipped",
            "executed": False,
            "signature": sig,
            "error": f"unexpected signature; need params {required}",
        }

    fx = resolve_fixture_block(load_cfg).get("surprise_ttt", {}) or {}
    root = Path.cwd()
    corpus = (root / fx["corpus_path"]).resolve() if fx.get("corpus_path") else None
    ckpt = (root / fx["ckpt_path"]).resolve() if fx.get("ckpt_path") else None
    cfgp = (root / fx["cfg_path"]).resolve() if fx.get("cfg_path") else None
    device = fx.get("device", "cpu")

    missing = []
    for name, p in [("corpus_path", corpus), ("ckpt_path", ckpt), ("cfg_path", cfgp)]:
        if p is None or not p.exists():
            missing.append(name)

    if missing:
        return {
            "status": "ok",
            "executed": False,
            "signature": sig,
            "note": "anchor resolved; fixture triple not present",
            "missing": missing,
            "expected": {
                "corpus_path": str(corpus) if corpus else None,
                "ckpt_path": str(ckpt) if ckpt else None,
                "cfg_path": str(cfgp) if cfgp else None,
            },
        }

    mod = importlib.import_module(anchor["module"])
    cfg_cls = getattr(mod, "TTTcfg", None)
    if cfg_cls is None:
        return {
            "status": "ok",
            "executed": False,
            "signature": sig,
            "note": "anchor resolved; TTTcfg not found in module",
            "cfg_path": str(cfgp),
        }

    cfg_data = read_cfg_data(cfgp)
    cfg_obj = cfg_cls(**cfg_data)

    out_dir = Path(tempfile.mkdtemp(prefix="six_smoke_ttt_"))
    out_path = out_dir / "ttt_out.json"

    try:
        res = obj(
            corpus_path=str(corpus),
            ckpt_path=str(ckpt),
            out_path=str(out_path),
            cfg=cfg_obj,
            device=device,
        )
    except Exception as e:
        return {"status": "error", "executed": False, "signature": sig, "error": f"run_ttt failed: {e}"}

    return {
        "status": "ok",
        "executed": True,
        "signature": sig,
        "inputs": {
            "corpus_path": str(corpus),
            "corpus_sha256": sha256_file(corpus),
            "ckpt_path": str(ckpt),
            "ckpt_sha256": sha256_file(ckpt),
            "cfg_path": str(cfgp),
            "cfg_sha256": sha256_file(cfgp),
            "device": device,
        },
        "output": {
            "out_path": str(out_path),
            "out_sha256": sha256_file(out_path) if out_path.exists() else None,
        },
        "result_type": type(res).__name__,
    }

def resolve_anchor_only(anchor: Dict[str, Any]) -> Dict[str, Any]:
    obj, sig, err = import_anchor(anchor)
    if err:
        return {"status": "error", "executed": False, "error": err, "signature": sig}
    return {"status": "ok", "executed": False, "signature": sig, "note": "anchor resolved"}

def build_six_systems_witness(load_witness: Dict[str, Any]) -> Dict[str, Any]:
    from pathlib import Path
    import json
    if isinstance(load_witness, (str, Path)):
        load_witness = json.loads(Path(load_witness).read_text(encoding="utf-8"))
    cfg = (load_witness or {}).get("config", {}) or {}
    anchors = (load_witness or {}).get("anchors", {}) or {}
    adapters = (load_witness or {}).get("adapters", {}) or {}

    def pick_anchor(key: str) -> Optional[Dict[str, Any]]:
        if key in anchors and isinstance(anchors[key], dict):
            return anchors[key]
        if key in adapters and isinstance(adapters[key], dict):
            a = adapters[key].get("anchor")
            if isinstance(a, dict):
                return a
        return None

    out: Dict[str, Any] = {"systems": {}, "version": "0.1.0"}

    for name in ["llm_nature", "smallchat", "kv_memory", "mlp_xor", "cbench", "surprise_ttt"]:
        anchor = pick_anchor(name)
        if not anchor:
            out["systems"][name] = {"status": "error", "executed": False, "error": "missing anchor"}
            continue

        if name == "surprise_ttt":
            out["systems"][name] = {"anchor": anchor, "action": run_surprise_ttt_action(anchor, cfg)}
        else:
            out["systems"][name] = {"anchor": anchor, "action": resolve_anchor_only(anchor)}

    systems = out.get("systems", {})
    if isinstance(systems, dict):
        cur = systems.get("surprise_ttt")
        if isinstance(cur, dict):
            act = cur.get("action") if isinstance(cur.get("action"), dict) else cur
            if isinstance(act, dict) and act.get("status") == "error" and act.get("error") == "missing anchor":
                rp = None
                lw = load_witness
                if isinstance(lw, dict):
                    sysinfo = (lw.get("systems") or {}).get("adaptation") or (lw.get("systems") or {}).get("surprise_ttt")
                    if isinstance(sysinfo, dict):
                        rp = sysinfo.get("repo_path")
                cur["action"] = _probe_surprise_ttt_anchor(rp)
                systems["surprise_ttt"] = cur

    return out
