from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def _detach_tensor(x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if x is None:
        return None
    return x.detach().clone()


@dataclass(frozen=True)
class KVProjector:
    kind: str
    w_q: torch.Tensor
    b_q: Optional[torch.Tensor]
    w_k: torch.Tensor
    b_k: Optional[torch.Tensor]
    w_v: torch.Tensor
    b_v: Optional[torch.Tensor]

    @property
    def d_kv(self) -> int:
        return int(self.w_k.shape[0])

    def project(self, hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = hidden.device
        dtype = hidden.dtype
        wq = self.w_q.to(device=device, dtype=dtype)
        wk = self.w_k.to(device=device, dtype=dtype)
        wv = self.w_v.to(device=device, dtype=dtype)
        bq = None if self.b_q is None else self.b_q.to(device=device, dtype=dtype)
        bk = None if self.b_k is None else self.b_k.to(device=device, dtype=dtype)
        bv = None if self.b_v is None else self.b_v.to(device=device, dtype=dtype)
        q = F.linear(hidden, wq, bq)
        k = F.linear(hidden, wk, bk)
        v = F.linear(hidden, wv, bv)
        return q, k, v

    @staticmethod
    def _from_qkv_linear(q, k, v) -> Optional["KVProjector"]:
        if q is None or k is None or v is None:
            return None
        if not (hasattr(q, "weight") and hasattr(k, "weight") and hasattr(v, "weight")):
            return None
        wq = _detach_tensor(q.weight)
        wk = _detach_tensor(k.weight)
        wv = _detach_tensor(v.weight)
        bq = _detach_tensor(getattr(q, "bias", None))
        bk = _detach_tensor(getattr(k, "bias", None))
        bv = _detach_tensor(getattr(v, "bias", None))
        if wq is None or wk is None or wv is None:
            return None
        return KVProjector(kind="qkv_proj", w_q=wq, b_q=bq, w_k=wk, b_k=bk, w_v=wv, b_v=bv)

    @staticmethod
    def _from_concat_linear(weight: torch.Tensor, bias: Optional[torch.Tensor]) -> Optional["KVProjector"]:
        if weight is None:
            return None
        out = int(weight.shape[0])
        if out % 3 != 0:
            return None
        n = out // 3
        wq = _detach_tensor(weight[0:n, :])
        wk = _detach_tensor(weight[n:2 * n, :])
        wv = _detach_tensor(weight[2 * n:3 * n, :])
        bq = None
        bk = None
        bv = None
        if bias is not None and int(bias.shape[0]) == out:
            bq = _detach_tensor(bias[0:n])
            bk = _detach_tensor(bias[n:2 * n])
            bv = _detach_tensor(bias[2 * n:3 * n])
        if wq is None or wk is None or wv is None:
            return None
        return KVProjector(kind="concat_qkv", w_q=wq, b_q=bq, w_k=wk, b_k=bk, w_v=wv, b_v=bv)

    @classmethod
    def from_model(cls, model) -> Optional["KVProjector"]:
        for m in model.modules():
            q = getattr(m, "q_proj", None)
            k = getattr(m, "k_proj", None)
            v = getattr(m, "v_proj", None)
            proj = cls._from_qkv_linear(q, k, v)
            if proj is not None:
                return proj

        for m in model.modules():
            c_attn = getattr(m, "c_attn", None)
            if c_attn is not None and hasattr(c_attn, "weight"):
                w = getattr(c_attn, "weight", None)
                b = getattr(c_attn, "bias", None)
                proj = cls._from_concat_linear(w, b)
                if proj is not None:
                    return proj

        for m in model.modules():
            w = getattr(m, "in_proj_weight", None)
            b = getattr(m, "in_proj_bias", None)
            if w is not None:
                proj = cls._from_concat_linear(w, b)
                if proj is not None:
                    return proj

        return None
