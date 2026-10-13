"""Z-stack TIFF I/O and resize helpers for EM volume processing."""
from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import tifffile


def _infer_resize_hw(h: int, w: int) -> Tuple[float, int, int]:
    """Match app.resize_image_if_needed geometry policy (per first slice)."""
    scale_factor = 1.0
    if w > 2048 or h > 2048:
        scale_factor = 0.4
    elif w > 1024 or h > 1024:
        scale_factor = 0.7
    new_w = max(1, int(round(w * scale_factor)))
    new_h = max(1, int(round(h * scale_factor)))
    return scale_factor, new_w, new_h


def resize_z_stack_hw(stack: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """
    Resize every Z-slice if H/W exceed thresholds (same scale for whole stack).
    stack: (Z, H, W) uint8 or (Z, H, W, C) uint8 with C in (3, 4).
    Returns resized stack, scale_factor, (orig_h, orig_w).
    """
    if stack.ndim != 3 and not (stack.ndim == 4 and stack.shape[-1] in (3, 4)):
        raise ValueError(f"resize_z_stack_hw: expected ZHW or ZHWC, got {stack.shape}")
    z = int(stack.shape[0])
    h, w = int(stack.shape[1]), int(stack.shape[2])
    sf, nw, nh = _infer_resize_hw(h, w)
    if sf >= 1.0 - 1e-9:
        return stack, 1.0, (h, w)
    out = np.empty((z, nh, nw) + stack.shape[3:], dtype=stack.dtype)
    for i in range(z):
        sl = stack[i]
        if sl.ndim == 2:
            out[i] = cv2.resize(sl, (nw, nh), interpolation=cv2.INTER_AREA)
        else:
            out[i] = cv2.resize(sl, (nw, nh), interpolation=cv2.INTER_AREA)
    return out, sf, (h, w)


def _volume_to_z_hw_u8(arr: np.ndarray) -> np.ndarray:
    """Normalize arbitrary dtype/shape to (Z, H, W) uint8."""
    a = np.asarray(arr)
    if a.ndim == 2:
        a = a[np.newaxis, ...]
    elif a.ndim == 4:
        # Z,H,W,C
        c = a.shape[-1]
        if c == 1:
            a = a[..., 0]
        elif c >= 3:
            z, h, w, _ = a.shape
            gray = np.empty((z, h, w), dtype=np.float32)
            for i in range(z):
                rgb = a[i][..., :3]
                if rgb.dtype != np.uint8:
                    rgb = np.clip(rgb, 0, None)
                    if np.nanmax(rgb) <= 1.5:
                        rgb = (rgb * 255.0).astype(np.uint8)
                    else:
                        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
                gray[i] = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
            a = gray
        else:
            raise ValueError(f"Unsupported channel count in Z-stack: {a.shape}")
    elif a.ndim != 3:
        raise ValueError(f"Unsupported stack shape: {a.shape}")

    if a.dtype == np.uint8:
        return np.clip(a, 0, 255).astype(np.uint8)

    a = a.astype(np.float32, copy=False)
    finite = np.isfinite(a)
    if not np.any(finite):
        return np.zeros(a.shape, dtype=np.uint8)
    lo, hi = np.percentile(a[finite], [0.8, 99.4])
    span = float(max(hi - lo, 1e-6))
    u8 = np.clip((a - lo) / span * 255.0, 0.0, 255.0).astype(np.uint8)
    return u8


def load_z_stack(path: str) -> Tuple[np.ndarray, Dict[str, object]]:
    """
    Read TIFF as (Z, H, W) uint8: either multi-page (one slice per page) or a single
    IFD with shape (Z, Y, X) / (Z, Y, X, C).
    """
    path = os.path.abspath(path)
    with tifffile.TiffFile(path) as tf:
        n = len(tf.pages)
        if n < 1:
            raise ValueError(f"No pages in TIFF: {path}")
        if n > 1:
            slices = [tf.pages[i].asarray() for i in range(n)]
            stack_raw = np.stack(slices, axis=0)
        else:
            arr = tf.pages[0].asarray()
            if arr.ndim == 2:
                stack_raw = arr[np.newaxis, ...]
            elif arr.ndim == 3:
                stack_raw = arr
            elif arr.ndim == 4:
                stack_raw = arr
            else:
                raise ValueError(f"Unsupported TIFF array shape: {arr.shape}")
    stack = _volume_to_z_hw_u8(stack_raw)
    meta: Dict[str, object] = {
        "path": path,
        "num_pages": int(stack.shape[0]),
        "shape_u8": tuple(int(x) for x in stack.shape),
    }
    return stack, meta


def save_z_stack_tiff(path: str, stack: np.ndarray) -> None:
    """Write (Z,H,W) or (Z,H,W,C) uint8/uint16 volume as TIFF series."""
    stack = np.asarray(stack)
    if stack.ndim not in (3, 4):
        raise ValueError(f"save_z_stack_tiff: need ZHW or ZHWC, got {stack.shape}")
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tifffile.imwrite(
        path,
        stack,
        imagej=True,
        metadata={"axes": "ZYX" if stack.ndim == 3 else "ZYXC"},
    )


def detect_single_multipage_tiff(folder: str) -> Optional[str]:
    """
    If folder contains exactly one .tif/.tiff and it represents a Z-stack
    (multiple pages OR single IFD with depth > 1), return its path.
    """
    folder = os.path.abspath(folder)
    if not os.path.isdir(folder):
        return None
    tiffs = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith((".tif", ".tiff"))
    ]
    if len(tiffs) != 1:
        return None
    p = tiffs[0]
    try:
        with tifffile.TiffFile(p) as tf:
            if len(tf.pages) > 1:
                return p
            arr = tf.pages[0].asarray()
            if arr.ndim == 3 and arr.shape[0] > 1:
                return p
            if arr.ndim == 4 and arr.shape[0] > 1:
                return p
    except Exception:
        return None
    return None
