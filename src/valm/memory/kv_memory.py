import torch
import torch.nn.functional as F
from typing import Optional
from ..core.types import KVState


class KVMemorySystem:
    def __init__(self, config):
        self.config = config
        self.state: Optional[KVState] = None
        self.last_retrieval_count = 0

    def initialize(self, d_model: int):
        M = self.config.M
        self.state = KVState(
            keys=torch.zeros(M, d_model),
            values=torch.zeros(M, d_model),
            ages=torch.zeros(M, dtype=torch.long),
            slot_mask=torch.zeros(M, dtype=torch.bool),
        )
        self.last_retrieval_count = 0

    def should_write(self, k_vec: torch.Tensor) -> bool:
        if self.state is None:
            return False
        novelty = float(self.compute_novelty(k_vec))
        return novelty >= float(self.config.tau_novel)

    def compute_novelty(self, k_vec: torch.Tensor) -> float:
        if self.state is None or self.state.occupied_slots == 0:
            return 1.0

        k = k_vec.detach().mean(dim=0)
        sims = []
        for i in range(self.config.M):
            if bool(self.state.slot_mask[i]):
                sims.append(float(self._cosine_similarity(k, self.state.keys[i]).item()))
        if len(sims) == 0:
            return 1.0
        return 1.0 - max(sims)

    def select_slot(self, k_vec: torch.Tensor) -> int:
        if self.state is None:
            return 0

        k = k_vec.detach().mean(dim=0)
        best_sim = -1.0
        best_idx = 0

        for i in range(self.config.M):
            if bool(self.state.slot_mask[i]):
                sim = float(self._cosine_similarity(k, self.state.keys[i]).item())
                if sim > best_sim:
                    best_sim = sim
                    best_idx = i

        if best_sim >= float(self.config.tau_reuse):
            return int(best_idx)

        return int(self.state.ages.argmax().item())

    def write(self, slot_idx: int, k_vec: torch.Tensor, v_vec: torch.Tensor):
        if self.state is None:
            return

        k = k_vec.detach().mean(dim=0)
        v = v_vec.detach().mean(dim=0)

        if bool(self.state.slot_mask[slot_idx]):
            g = float(self.config.g_write)
            self.state.keys[slot_idx] = (1.0 - g) * self.state.keys[slot_idx] + g * k
            self.state.values[slot_idx] = (1.0 - g) * self.state.values[slot_idx] + g * v
        else:
            self.state.keys[slot_idx] = k
            self.state.values[slot_idx] = v
            self.state.slot_mask[slot_idx] = True

        self.state.ages[slot_idx] = 0
        self.state.ages += 1

    def retrieve(self, q_vec: torch.Tensor) -> torch.Tensor:
        if self.state is None or self.state.occupied_slots == 0:
            return torch.zeros_like(q_vec)

        q = q_vec.detach().mean(dim=0)
        sims = []
        idxs = []

        for i in range(self.config.M):
            if bool(self.state.slot_mask[i]):
                sims.append(float(self._cosine_similarity(q, self.state.keys[i]).item()))
                idxs.append(i)

        if len(sims) == 0:
            return torch.zeros_like(q_vec)

        sim_t = torch.tensor(sims, device=q_vec.device, dtype=torch.float32)
        weights = F.softmax(sim_t, dim=0)

        out = torch.zeros_like(q)
        for i, w in zip(idxs, weights):
            out = out + w * self.state.values[i].to(out.device, dtype=out.dtype)

        self.last_retrieval_count = len(idxs)
        return out.view(1, -1).expand_as(q_vec)

    def state_digest(self) -> str:
        if self.state is None:
            return "empty"
        return self.state.state_digest

    @property
    def occupied_slots(self) -> int:
        if self.state is None:
            return 0
        return self.state.occupied_slots

    def _cosine_similarity(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        eps = float(self.config.epsilon)
        na = a.norm() + eps
        nb = b.norm() + eps
        return (a @ b) / (na * nb)
