from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import importlib
import json
from pathlib import Path

@dataclass(frozen=True)
class DecodeCfg:
    max_new_tokens: int = 128
    temperature: float = 0.8
    top_p: float = 0.95

class ChatBackend:
    def encode(self, text: str) -> List[int]:
        raise NotImplementedError

    def decode(self, ids: List[int]) -> str:
        raise NotImplementedError

    def step(self, prompt_ids: List[int], state: Optional[Dict[str, Any]], cfg: DecodeCfg) -> Tuple[int, Dict[str, Any], Dict[str, Any]]:
        raise NotImplementedError

def _load_backend(backend_cfg: Dict[str, Any]) -> ChatBackend:
    mod_name = backend_cfg.get("module")
    cls_name = backend_cfg.get("class")
    if not mod_name or not cls_name:
        raise ValueError("chat.backend must specify module and class")
    m = importlib.import_module(mod_name)
    cls = getattr(m, cls_name)
    return cls(**(backend_cfg.get("kwargs") or {}))

def run_chat(config_path: str, cfg: DecodeCfg) -> None:
    cfg_path = Path(config_path)
    data = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.suffix == ".json" else None
    if data is None:
        try:
            import yaml
        except Exception as e:
            raise RuntimeError("YAML config requires pyyaml") from e
        data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}

    chat_cfg = (data or {}).get("chat", {}) or {}
    backend_cfg = (chat_cfg.get("backend") or {})
    backend = _load_backend(backend_cfg)

    history: List[Tuple[str, str]] = []
    state: Optional[Dict[str, Any]] = None

    while True:
        try:
            user = input("user> ").strip()
        except EOFError:
            print()
            break
        if user == "":
            continue
        if user in ("/quit", "/exit"):
            break

        if hasattr(backend, "render_prompt"):
            prompt = backend.render_prompt(history, user)
        else:
            prompt = ""
            for u, a in history[-8:]:
                prompt += f"User: {u}\nAssistant: {a}\n"
            prompt += f"User: {user}\nAssistant: "

        prompt_ids = backend.encode(prompt)
        out_ids: List[int] = []
        meta_last: Dict[str, Any] = {}

        for _ in range(cfg.max_new_tokens):
            tok, state, meta = backend.step(prompt_ids + out_ids, state, cfg)
            meta_last = meta or {}
            out_ids.append(tok)
            text = backend.decode(out_ids)
            if text.endswith("\n") or text.endswith("User:"):
                break

        assistant = backend.decode(out_ids).strip()
        print("assistant>", assistant)
        history.append((user, assistant))

        _ = meta_last
