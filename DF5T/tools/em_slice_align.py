"""2D alignment between consecutive EM slices (phase correlation + dense optical flow)."""
from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from tools.em_tensor import clip01


def _to_gray01(img: np.ndarray) -> np.ndarray:
    x = np.asarray(img)
    if x.ndim == 3:
        x = cv2.cvtColor(x.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    if x.dtype != np.float32:
        x = x.astype(np.float32)
    if x.max() > 1.5:
        x = x / 255.0
    return clip01(x)


def align_prev_to_current(
    prev: np.ndarray,
    cur: np.ndarray,
    max_shift_frac: float = 0.12,
    prev_shift: Optional[Tuple[float, float]] = None,
) -> Tuple[np.ndarray, float, Tuple[float, float], np.ndarray]:
    """
    Phase-correlate `prev` onto `cur` grid (same shape as `cur`), return warped prev in [0,1].

    Returns:
        warped_prev: float32 [0,1], shape == cur.shape
        confidence: [0,1] from phase peak quality (low -> skip blending)
        (dx, dy): translation applied to prev (positive dx shifts content right)
        M: 2x3 affine warp matrix used for warping
    """
    cur01 = _to_gray01(cur)
    h, w = cur01.shape[:2]
    prev01 = _to_gray01(prev)
    if prev01.shape != (h, w):
        prev01 = cv2.resize(prev01, (w, h), interpolation=cv2.INTER_LINEAR)

    max_side = 512
    scale = 1.0
    cur_c = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8)).apply(
        np.clip(np.round(_to_gray01(cur01) * 255.0), 0, 255).astype(np.uint8)
    ).astype(np.float32)
    prev_c = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8)).apply(
        np.clip(np.round(_to_gray01(prev01) * 255.0), 0, 255).astype(np.uint8)
    ).astype(np.float32)
    if max(h, w) > max_side:
        scale = max_side / float(max(h, w))
        nh, nw = max(8, int(round(h * scale))), max(8, int(round(w * scale)))
        cur_s = cv2.resize(cur_c, (nw, nh), interpolation=cv2.INTER_AREA)
        prev_s = cv2.resize(prev_c, (nw, nh), interpolation=cv2.INTER_AREA)
    else:
        cur_s, prev_s = cur_c, prev_c

    (dx_s, dy_s), response = cv2.phaseCorrelate(cur_s, prev_s)
    resp = float(response) if np.isfinite(response) else 0.0
    phase_conf = float(np.clip((resp - 2.8) / 10.0, 0.0, 1.0))

    dx = float(dx_s) / scale
    dy = float(dy_s) / scale
    max_shift = max(4.0, float(max_shift_frac) * float(min(h, w)))
    dx = float(np.clip(dx, -max_shift, max_shift))
    dy = float(np.clip(dy, -max_shift, max_shift))
    if prev_shift is not None:
        pdx, pdy = float(prev_shift[0]), float(prev_shift[1])
        dx = 0.62 * dx + 0.38 * pdx
        dy = 0.62 * dy + 0.38 * pdy

    M = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
    prev_shifted = cv2.warpAffine(
        prev01,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )

    ecc_conf = 0.0
    try:
        cur_u8 = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8)).apply(
            np.clip(np.round(cur01 * 255.0), 0, 255).astype(np.uint8)
        )
        prev_u8 = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8)).apply(
            np.clip(np.round(prev_shifted * 255.0), 0, 255).astype(np.uint8)
        )
        M_ecc = M.copy()
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 80, 1e-5)
        cc, M_ecc = cv2.findTransformECC(cur_u8, prev_u8, M_ecc, cv2.MOTION_EUCLIDEAN, criteria, None, 5)
        if np.isfinite(cc) and cc > 0.0:
            M = M_ecc.astype(np.float32)
            dx = float(M[0, 2])
            dy = float(M[1, 2])
            ecc_conf = float(np.clip((float(cc) - 0.55) / 0.40, 0.0, 1.0))
    except Exception:
        pass

    confidence = float(np.clip(0.45 * phase_conf + 0.55 * max(phase_conf, ecc_conf), 0.0, 1.0))
    warped = cv2.warpAffine(
        prev01,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    return clip01(warped.astype(np.float32)), confidence, (dx, dy), M


def warp_float01_with_shift(
    prev: np.ndarray,
    dx: float,
    dy: float,
    h: int,
    w: int,
    M: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Apply the same 2D affine as align_prev_to_current to an arbitrary
    single-channel map in [0,1], resizing to (h, w) if needed.
    """
    prev01 = _to_gray01(prev)
    if prev01.shape[0] != h or prev01.shape[1] != w:
        prev01 = cv2.resize(prev01, (w, h), interpolation=cv2.INTER_LINEAR)
    if M is None:
        M = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
    else:
        M = np.asarray(M, dtype=np.float32)
    out = cv2.warpAffine(
        prev01.astype(np.float32),
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    return clip01(out.astype(np.float32))


def _u8_for_flow(x01: np.ndarray, h: int, w: int) -> np.ndarray:
    g = _to_gray01(x01)
    if g.shape[0] != h or g.shape[1] != w:
        g = cv2.resize(g, (w, h), interpolation=cv2.INTER_LINEAR)
    u8 = np.clip(np.round(g * 255.0), 0, 255).astype(np.uint8)
    return cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8)).apply(u8)


def dense_optical_flow_prev_to_cur(
    prev: np.ndarray,
    cur: np.ndarray,
    max_side: int = 512,
    max_disp_frac: float = 0.12,
    prev_shift: Optional[Tuple[float, float]] = None,
) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    Dense Farneback flow: first image = ``cur``, second = ``prev`` so flow is defined
    on the **current** grid (displacement from cur pixel toward matching prev pixel).
    Warp ``prev`` onto ``cur`` grid via inverse sampling.

    Returns:
        warped_prev: float32 [0,1], shape == cur
        confidence: [0,1] (lower when median flow magnitude is large / unstable)
        flow_hw2: (H,W,2) float32 displacement (cur_x + fx -> prev sample), same as used for remap
    """
    cur01 = _to_gray01(cur)
    h, w = int(cur01.shape[0]), int(cur01.shape[1])
    prev01 = _to_gray01(prev)
    if prev01.shape != (h, w):
        prev01 = cv2.resize(prev01, (w, h), interpolation=cv2.INTER_LINEAR)

    max_side_pc = 512
    scale_pc = 1.0
    cur_pc = _u8_for_flow(cur01, h, w).astype(np.float32)
    prev_pc = _u8_for_flow(prev01, h, w).astype(np.float32)
    if max(h, w) > max_side_pc:
        scale_pc = max_side_pc / float(max(h, w))
        nh, nw = max(8, int(round(h * scale_pc))), max(8, int(round(w * scale_pc)))
        cur_s = cv2.resize(cur_pc, (nw, nh), interpolation=cv2.INTER_AREA)
        prev_s = cv2.resize(prev_pc, (nw, nh), interpolation=cv2.INTER_AREA)
    else:
        cur_s, prev_s = cur_pc, prev_pc
    (cdx_s, cdy_s), response = cv2.phaseCorrelate(cur_s, prev_s)
    resp = float(response) if np.isfinite(response) else 0.0
    phase_conf = float(np.clip((resp - 2.8) / 10.0, 0.0, 1.0))
    coarse_dx = float(cdx_s) / scale_pc
    coarse_dy = float(cdy_s) / scale_pc
    max_shift = max(4.0, 0.14 * float(min(h, w)))
    coarse_dx = float(np.clip(coarse_dx, -max_shift, max_shift))
    coarse_dy = float(np.clip(coarse_dy, -max_shift, max_shift))
    if prev_shift is not None:
        pdx, pdy = float(prev_shift[0]), float(prev_shift[1])
        coarse_dx = 0.62 * coarse_dx + 0.38 * pdx
        coarse_dy = 0.62 * coarse_dy + 0.38 * pdy

    M_coarse = np.array([[1.0, 0.0, coarse_dx], [0.0, 1.0, coarse_dy]], dtype=np.float32)
    prev_pre = cv2.warpAffine(
        prev01,
        M_coarse,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )

    scale = 1.0
    cur_u8 = _u8_for_flow(cur01, h, w)
    prev_u8 = _u8_for_flow(prev_pre, h, w)
    try:
        if max(h, w) > max_side:
            scale = max_side / float(max(h, w))
            nh, nw = max(8, int(round(h * scale))), max(8, int(round(w * scale)))
            cur_s = cv2.resize(cur_u8, (nw, nh), interpolation=cv2.INTER_AREA)
            prev_s = cv2.resize(prev_u8, (nw, nh), interpolation=cv2.INTER_AREA)
            win_s = max(5, min(21, ((min(nh, nw) // 18) | 1)))
            lv_s = min(5, max(3, int(np.log2(max(float(nh), float(nw))) + 1e-6)))
            flow_s = cv2.calcOpticalFlowFarneback(
                cur_s,
                prev_s,
                None,
                0.5,
                int(lv_s),
                int(win_s),
                3,
                5,
                1.15,
                0,
            )
            flow = np.zeros((h, w, 2), dtype=np.float32)
            flow[:, :, 0] = cv2.resize(flow_s[:, :, 0], (w, h), interpolation=cv2.INTER_LINEAR) / scale
            flow[:, :, 1] = cv2.resize(flow_s[:, :, 1], (w, h), interpolation=cv2.INTER_LINEAR) / scale
        else:
            win = max(5, min(21, ((min(h, w) // 18) | 1)))
            lv = min(5, max(3, int(np.log2(max(float(h), float(w))) + 1e-6)))
            flow = cv2.calcOpticalFlowFarneback(
                cur_u8,
                prev_u8,
                None,
                0.5,
                int(lv),
                int(win),
                3,
                5,
                1.15,
                0,
            ).astype(np.float32)
    except Exception:
        flow = np.zeros((h, w, 2), dtype=np.float32)

    max_disp = max(2.0, float(max_disp_frac) * float(min(h, w)))
    flow[:, :, 0] = np.clip(flow[:, :, 0], -max_disp, max_disp)
    flow[:, :, 1] = np.clip(flow[:, :, 1], -max_disp, max_disp)

    mag = np.sqrt(flow[:, :, 0] ** 2 + flow[:, :, 1] ** 2)
    med = float(np.median(mag)) if mag.size else 0.0
    p95 = float(np.percentile(mag, 95)) if mag.size else 0.0
    span = float(max(med, p95 * 0.60, 1e-6))
    flow_conf = float(np.clip(1.0 - span / max(8.0, 0.10 * float(min(h, w))), 0.0, 1.0))
    confidence = float(np.clip(0.38 * phase_conf + 0.62 * flow_conf, 0.0, 1.0))
    if phase_conf > 0.35 and flow_conf > 0.25:
        confidence = float(np.clip(confidence * 1.10, 0.0, 1.0))

    flow[:, :, 0] += np.float32(coarse_dx)
    flow[:, :, 1] += np.float32(coarse_dy)

    gx = np.arange(w, dtype=np.float32)[None, :].repeat(h, axis=0)
    gy = np.arange(h, dtype=np.float32)[:, None].repeat(w, axis=1)
    map_x = gx + flow[:, :, 0]
    map_y = gy + flow[:, :, 1]
    warped = cv2.remap(
        prev01.astype(np.float32),
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    return clip01(warped.astype(np.float32)), confidence, flow


def warp_float01_with_flow(prev: np.ndarray, flow_hw2: np.ndarray, h: int, w: int) -> np.ndarray:
    """Apply the same (H,W,2) flow field (cur-grid) to another single-channel [0,1] map."""
    prev01 = _to_gray01(prev)
    if prev01.shape[0] != h or prev01.shape[1] != w:
        prev01 = cv2.resize(prev01, (w, h), interpolation=cv2.INTER_LINEAR)
    if flow_hw2.shape[0] != h or flow_hw2.shape[1] != w:
        fx = cv2.resize(flow_hw2[:, :, 0], (w, h), interpolation=cv2.INTER_LINEAR)
        fy = cv2.resize(flow_hw2[:, :, 1], (w, h), interpolation=cv2.INTER_LINEAR)
        flow_hw2 = np.stack([fx, fy], axis=-1)
    gx = np.arange(w, dtype=np.float32)[None, :].repeat(h, axis=0)
    gy = np.arange(h, dtype=np.float32)[:, None].repeat(w, axis=1)
    map_x = gx + flow_hw2[:, :, 0]
    map_y = gy + flow_hw2[:, :, 1]
    out = cv2.remap(
        prev01.astype(np.float32),
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    return clip01(out.astype(np.float32))
