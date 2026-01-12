from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import math
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _top_p_filtering(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    # logits: [vocab]
    if top_p is None or top_p >= 1.0:
        return logits
    probs = torch.softmax(logits, dim=-1)
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cum = torch.cumsum(sorted_probs, dim=-1)

    # Keep smallest set with cumulative prob >= top_p
    cutoff = cum > top_p
    # shift right to keep at least 1 token
    cutoff[..., 1:] = cutoff[..., :-1].clone()
    cutoff[..., 0] = False

    sorted_probs = sorted_probs.masked_fill(cutoff, 0.0)
    # renormalize
    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
    # sample in sorted space, map back to vocab id
    pick = torch.multinomial(sorted_probs, num_samples=1)  # [1]
    tok = sorted_idx.gather(dim=-1, index=pick).item()
    return tok


class HFSmallchatBackend:
    """
    HF-backed chat backend compatible with valm.chat.runtime.ChatBackend.

    model_id:
      - HF hub id, OR
      - local path to output_dir produced by smallchat scripts (train_sft / train_dpo).
    """

    def __init__(
        self,
        model_id: str,
        device: str = "cpu",
        dtype: str = "auto",
        trust_remote_code: bool = False,
    ):
        self.model_id = model_id
        self.device = device

        self.tok = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=trust_remote_code)
        if self.tok.pad_token is None:
            # match smallchat scripts/common.py intent
            self.tok.pad_token = self.tok.eos_token

        torch_dtype = None
        if dtype == "auto":
            torch_dtype = None
        else:
            # e.g. "float16", "bfloat16", "float32"
            torch_dtype = getattr(torch, dtype)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            device_map=None,
            trust_remote_code=trust_remote_code,
        )
        self.model.eval()
        self.model.to(torch.device(device))

        # If tokenizer has chat template, we use it.
        self.has_chat_template = bool(getattr(self.tok, "chat_template", None))

    # --- required interface ---
    def encode(self, text: str) -> List[int]:
        ids = self.tok.encode(text, add_special_tokens=False)
        return list(map(int, ids))

    def decode(self, ids: List[int]) -> str:
        return self.tok.decode(ids, skip_special_tokens=True)

    # --- optional helper used by runtime if present ---
    def render_prompt(self, history: List[Tuple[str, str]], user: str) -> str:
        if self.has_chat_template:
            messages: List[Dict[str, str]] = []
            for u, a in history:
                messages.append({"role": "user", "content": u})
                messages.append({"role": "assistant", "content": a})
            messages.append({"role": "user", "content": user})
            # add_generation_prompt=True makes the assistant prefix appear per-template
            return self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # fallback (your current runtime format)
        prompt = ""
        for u, a in history[-8:]:
            prompt += f"User: {u}\nAssistant: {a}\n"
        prompt += f"User: {user}\nAssistant: "
        return prompt

    def step(
        self,
        prompt_ids: List[int],
        state: Optional[Dict[str, Any]],
        cfg,  # DecodeCfg from runtime
    ) -> Tuple[int, Dict[str, Any], Dict[str, Any]]:
        # state tracks how many ids already ingested and KV cache
        if state is None:
            state = {"n_ingested": 0, "past_key_values": None}

        n0 = int(state.get("n_ingested", 0))
        past = state.get("past_key_values", None)

        # Only feed the delta since last call
        new_ids = prompt_ids[n0:]
        if len(new_ids) == 0:
            # should not happen in normal loop, but be safe
            new_ids = [prompt_ids[-1]]

        input_ids = torch.tensor([new_ids], dtype=torch.long, device=self.model.device)

        with torch.no_grad():
            out = self.model(input_ids=input_ids, use_cache=True, past_key_values=past)
            logits = out.logits[0, -1, :]  # [vocab]
            past2 = out.past_key_values

        # temperature + top_p sampling
        temp = float(getattr(cfg, "temperature", 1.0) or 1.0)
        top_p = float(getattr(cfg, "top_p", 1.0) or 1.0)

        if temp <= 0.0:
            tok = int(torch.argmax(logits).item())
        else:
            scaled = logits / temp
            tok = _top_p_filtering(scaled, top_p=top_p)

        # update state
        state["past_key_values"] = past2
        state["n_ingested"] = n0 + len(new_ids)

        # lightweight meta for llm_nature style instrumentation later
        with torch.no_grad():
            probs = torch.softmax(logits, dim=-1)
            top1 = float(torch.max(probs).item())
            entropy = float((-torch.sum(probs * torch.log(probs + 1e-12))).item())
        meta = {"top1": top1, "entropy": entropy}

        return tok, state, meta
