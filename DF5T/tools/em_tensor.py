"""Single source for tensor / numpy conversions and basic array clamps."""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch

EPS = 1e-6


def tensor_is_m11(x: torch.Tensor) -> bool:
    return bool(torch.is_tensor(x) and (float(x.min()) < -0.05 or float(x.max()) > 1.05))


def tensor_to_gray01(x: torch.Tensor) -> np.ndarray:
    t = x.detach().float().cpu()
    if t.dim() == 4:
        t = t[0]
    if t.dim() == 3:
        if t.shape[0] == 1:
            t = t[0]
        else:
            if tensor_is_m11(x):
                t = (t + 1.0) * 0.5
            t = t.clamp(0.0, 1.0)
            t = 0.299 * t[0] + 0.587 * t[1] + 0.114 * t[2]
    if tensor_is_m11(x):
        t = (t + 1.0) * 0.5
    return t.clamp(0.0, 1.0).numpy().astype(np.float32)


def gray01_to_tensor(gray: np.ndarray, ref: Optional[torch.Tensor] = None, device: Optional[torch.device] = None) -> torch.Tensor:
    out = torch.from_numpy(np.clip(gray, 0.0, 1.0).astype(np.float32))[None, None]
    if ref is not None and ref.dim() == 4 and ref.shape[1] > 1:
        out = out.repeat(1, ref.shape[1], 1, 1)
    out = out * 2.0 - 1.0
    if ref is not None:
        return out.to(device=ref.device, dtype=ref.dtype)
    if device is not None:
        return out.to(device=device)
    return out


def gray01_to_three_m11(gray: np.ndarray, device: torch.device) -> torch.Tensor:
    x = torch.from_numpy(np.clip(gray, 0.0, 1.0).astype(np.float32))[None, None].to(device=device)
    x = x.repeat(1, 3, 1, 1)
    return x * 2.0 - 1.0


def clip01(x: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(x, dtype=np.float32), 0.0, 1.0)


def norm01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    mn = float(x.min())
    mx = float(x.max())
    if mx - mn < EPS:
        return np.zeros_like(x, dtype=np.float32)
    return (x - mn) / (mx - mn + EPS)
