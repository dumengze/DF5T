"""Fallback nonlinear reverse when no diffusion backbone is available."""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from tools.em_maps import build_inp_mask, gauss, task_support_maps
from tools.em_tensor import EPS, clip01


def heuristic_reverse_from_svd(
    obs_gray: np.ndarray,
    linear_gray: np.ndarray,
    svd_gray: np.ndarray,
    task: str,
    strength: float,
    metrics: Optional[Dict[str, float]] = None,
    linear_meta: Optional[Dict[str, Any]] = None,
    svd_meta: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    from tools.em_maps import analyze_em_image

    metrics = metrics or analyze_em_image(obs_gray)
    linear_meta = linear_meta or {}
    svd_meta = svd_meta or {}
    maps = linear_meta.get("maps") or task_support_maps(linear_gray)
    guidance = svd_meta.get("guidance_map")
    if guidance is None:
        guidance = maps["membrane"]
    detail = linear_gray - svd_gray
    membrane = maps["membrane"]
    ridge = maps["ridge"]
    anis_map = maps["anis_map"]

    if task == "deno_em":
        recovered = svd_gray + (0.28 + 0.26 * strength) * guidance * detail + (0.06 + 0.10 * strength) * membrane * (linear_gray - gauss(linear_gray, 0.75))
        recovered = np.clip(0.88 * recovered + 0.12 * gauss(recovered, 0.42), 0.0, 1.0)
    elif task == "deblur_em":
        recovered = svd_gray + (0.40 + 0.32 * strength) * guidance * detail + (0.06 + 0.10 * strength) * membrane * np.sign(detail) * np.sqrt(np.abs(detail) + EPS)
        recovered = np.clip(recovered, 0.0, 1.0)
    elif task.startswith("sr"):
        recovered = svd_gray + (0.44 + 0.34 * strength) * guidance * detail + (0.10 + 0.14 * strength) * ridge * (linear_gray - gauss(linear_gray, 0.55))
        recovered = np.clip(recovered, 0.0, 1.0)
    elif task == "inp_em":
        hole_mask = svd_meta.get("mask", linear_meta.get("mask"))
        if not isinstance(hole_mask, np.ndarray) or hole_mask.shape != obs_gray.shape:
            hole_mask, _ = build_inp_mask(linear_gray)
        hole_mask = clip01(hole_mask.astype(np.float32))
        conf = svd_meta.get("hole_confidence", linear_meta.get("hole_confidence"))
        if not isinstance(conf, np.ndarray) or conf.shape != obs_gray.shape:
            conf = hole_mask
        conf = clip01(conf.astype(np.float32))
        mem_boost = clip01(membrane + 0.82 * ridge)
        hi_freq = linear_gray - gauss(linear_gray, 0.85)
        lc = maps.get("line_closure")
        lp = maps.get("lumen_protect")
        brk = maps.get("break_saliency")
        bridge_target = linear_meta.get("bridge_target", maps.get("bridge_target", svd_meta.get("bridge_target")))
        if not isinstance(bridge_target, np.ndarray) or bridge_target.shape != obs_gray.shape:
            bridge_target = linear_gray
        else:
            bridge_target = clip01(bridge_target.astype(np.float32))
        bridge_need = clip01(np.maximum(linear_gray - bridge_target, 0.0) / 0.12)
        repair = hole_mask
        if isinstance(lc, np.ndarray) and lc.shape == obs_gray.shape:
            lc = clip01(lc.astype(np.float32))
            repair = clip01(np.maximum(repair, 0.55 * lc))
        if isinstance(brk, np.ndarray) and brk.shape == obs_gray.shape:
            brk = clip01(brk.astype(np.float32))
            repair = clip01(np.maximum(repair, 0.52 * brk))
        if isinstance(lp, np.ndarray) and lp.shape == obs_gray.shape:
            lp = clip01(lp.astype(np.float32))
            repair = clip01(repair * (1.0 - 0.70 * lp))
        repair = clip01(np.maximum(repair, 0.92 * bridge_need) * (0.40 + 0.60 * mem_boost))
        repair_soft = clip01(np.maximum(0.0, gauss(repair, 0.78) - 0.34 * repair))
        side_ring = clip01(np.maximum(0.0, gauss(repair, 1.05) - 0.90 * repair))
        bridge_pull = np.clip(bridge_target - svd_gray, -0.46, 0.10)
        fill = (
            svd_gray
            + (0.14 + 0.12 * strength) * conf * repair * detail
            + (0.86 + 0.42 * strength) * repair * bridge_pull
            + (0.04 + 0.06 * strength) * mem_boost * repair * np.clip(hi_freq, -0.18, 0.12)
        )
        recovered = linear_gray * (1.0 - repair_soft) + np.clip(fill, 0.0, 1.0) * repair_soft
        recovered = np.clip(recovered * (1.0 - 0.74 * side_ring) + linear_gray * (0.74 * side_ring), 0.0, 1.0)
        if isinstance(lp, np.ndarray) and lp.shape == obs_gray.shape:
            recovered = np.clip(recovered * (1.0 - 0.82 * lp) + linear_gray * (0.82 * lp), 0.0, 1.0)
        recovered = np.clip(recovered + 0.34 * repair * np.clip(bridge_target - recovered, -0.12, 0.08), 0.0, 1.0)
    else:
        recovered = svd_gray + (0.32 + 0.24 * strength) * guidance * detail + (0.12 + 0.10 * strength) * anis_map * (linear_gray - gauss(linear_gray, 1.0))
        recovered = np.clip(recovered, 0.0, 1.0)

    return recovered.astype(np.float32), {"mode": "heuristic_reverse_fallback", "guidance_map": guidance.astype(np.float32)}
