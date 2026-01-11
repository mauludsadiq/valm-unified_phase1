from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence


@dataclass
class ToyNextTokenModel:
    vocab_size: int = 256

    def loss_next_token(self, ids: Sequence[int]) -> float:
        if len(ids) < 2:
            return 0.0

        n = 0
        ce = 0.0
        for i in range(len(ids) - 1):
            x = int(ids[i]) & 0xFFFFFFFF
            y = int(ids[i + 1]) % self.vocab_size
            pred = (1103515245 * x + 12345) % self.vocab_size
            p = 0.9 if pred == y else 0.1 / (self.vocab_size - 1)
            ce += -math.log(p)
            n += 1
        return ce / max(n, 1)


def make_toy_ids(n: int = 64) -> List[int]:
    ids: List[int] = []
    x = 1
    for _ in range(n):
        x = (1664525 * x + 1013904223) & 0xFFFFFFFF
        ids.append(int(x % 256))
    return ids
