from __future__ import annotations

import hashlib
from typing import Dict, Optional

import torch
import torch.nn as nn


class ParameterMemorySystem:
    """Surprise-gated parameter memory (Phase 1 stub).

    - Keeps `theta_0` snapshot.
    - Applies a small drift update only if surprise crosses threshold.
    - Tracks drift deterministically (L2 norm over params vs theta_0).
    """

    def __init__(self, config):
        self.config = config
        self.theta_0: Optional[Dict[str, torch.Tensor]] = None
        self.current_state: Optional[Dict[str, torch.Tensor]] = None

        self.last_surprise: float = 0.0
        self.last_gamma: float = 0.0
        self.last_lambda: float = 0.0
        self.update_count: int = 0
        self.total_drift: float = 0.0

    def initialize(self, theta_0: Dict[str, torch.Tensor]):
        self.theta_0 = {k: v.clone() for k, v in theta_0.items()}
        self.current_state = {k: v.clone() for k, v in theta_0.items()}
        self.total_drift = 0.0
        self.update_count = 0

    def update(self, model: nn.Module, loss: Optional[torch.Tensor], surprise: float):
        # Phase 1: do not require grads; apply a tiny deterministic nudge gated by gamma.
        self.last_surprise = float(surprise)
        gamma = float(self._compute_gamma(float(surprise)))
        lam = float(self._compute_lambda(float(surprise)))
        self.last_gamma = gamma
        self.last_lambda = lam

        if self.theta_0 is None:
            return

        with torch.no_grad():
            for name, p in model.named_parameters():
                if name not in self.theta_0:
                    continue
                # deterministic update: scale parameter by (1 - gamma*lam) as a placeholder
                # (real TTT uses gradients; this preserves a drift concept without requiring them)
                p.mul_(1.0 - (gamma * lam))

        # Refresh current_state snapshot + drift
        self.current_state = {k: v.clone() for k, v in model.state_dict().items() if torch.is_tensor(v)}
        self.total_drift = self._compute_total_drift(self.current_state, self.theta_0)
        self.update_count += 1

    def _compute_total_drift(self, cur: Dict[str, torch.Tensor], base: Dict[str, torch.Tensor]) -> float:
        tot = 0.0
        for k, v in cur.items():
            if k in base:
                d = (v - base[k]).norm().item()
                tot += d * d
        return tot ** 0.5

    def _compute_gamma(self, surprise: float) -> float:
        if self.config.gate_type == "budget":
            return float(self.config.eta_base) * min(1.0, max(0.0, float(self.config.tau_gate) / (surprise + 1e-8)))
        if self.config.gate_type == "sigmoid":
            z = float(self.config.beta_gate) * (float(self.config.tau_gate) - surprise)
            return float(self.config.eta_base) * torch.sigmoid(torch.tensor(z)).item()
        if self.config.gate_type == "miras":
            base = float(self.config.eta_base) / (1.0 + surprise)
            return min(base, float(self.config.eta_base))
        return float(self.config.eta_base)

    def _compute_lambda(self, surprise: float) -> float:
        return float(self.config.lambda_base) * (1.0 - min(1.0, surprise))

    def check_stability(self) -> bool:
        return float(self.total_drift) <= float(self.config.trust_radius)

    def state_digest(self) -> str:
        if self.current_state is None:
            return "uninitialized"
        h = hashlib.sha256()
        for name in sorted(self.current_state.keys()):
            t = self.current_state[name].detach().cpu().contiguous()
            h.update(name.encode("utf-8"))
            h.update(t.numpy().tobytes())
        return h.hexdigest()
