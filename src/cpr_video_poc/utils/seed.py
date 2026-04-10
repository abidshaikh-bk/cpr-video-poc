from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None



def resolve_seed(seed: Optional[int] = None) -> int:
    if seed is not None:
        return int(seed)
    return int.from_bytes(os.urandom(4), "big")



def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)



def build_torch_generator(seed: int, device: str = "cpu"):
    if torch is None:
        return None
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    return gen
