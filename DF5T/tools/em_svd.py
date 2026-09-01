"""SVD-based nonlinear degradation surrogate (forward branch)."""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from tools.em_maps import (
    build_inp_mask,
    gauss,
    inter_membrane_lumen_protect,
    membrane_break_saliency,
    membrane_bridge_target,
    membrane_line_gap_map,
    task_support_maps,
)
from tools.em_task_spec import resolve_svd_keep, svd_keep_for_task
from tools.em_tensor import EPS, clip01, norm01


def _resize_for_svd(gray: np.ndarray, max_side: int = 128) -> Tuple[np.ndarray, Tuple[int, int]]:
    h, w = gray.shape[:2]
    if max(h, w) <= max_side:
        return gray.copy(), (h, w)
    if h >= w:
        nh = max_side
        nw = max(8, int(round(w * max_side / h)))
    else:
        nw = max_side
        nh = max(8, int(round(h * max_side / w)))
    return cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_AREA), (h, w)


def observation_svd_spectrum(gray: np.ndarray, max_side: int = 128) -> Dict[str, Any]:
    """
    One SVD decomposition of the observation (resized). Reused for all adaptive probes on the same patch.
    cum_energy: cumulative normalized singular-value energy; keep is read from this curve, not from constants.
    decay_slow: how slowly energy decays (0=fast/blur-like, 1=slow/noise-or-alias-like spectrum).
    """
    g = np.asarray(gray, dtype=np.float32)
    if not np.isfinite(g).all():
        g = np.nan_to_num(g, nan=0.5, posinf=1.0, neginf=0.0)
    g = clip01(g)
    small, orig_hw = _resize_for_svd(g, max_side=max_side)
    h, w = small.shape[:2]
    if min(h, w) < 4:
        return {
            "cum_energy": np.array([1.0], dtype=np.float32),
            "decay_slow": 0.0,
            "n_sv": 1,
            "orig_hw": orig_hw,
            "U": None,
            "S": None,
            "Vh": None,
        }

    u, s, vh = np.linalg.svd(small, full_matrices=False)
    power = (s * s).astype(np.float64)
    total = float(power.sum() + EPS)
    cum = np.cumsum(power / total).astype(np.float32)
    # Use a reference singular value safely inside bounds. The old max(4, ...)
    # expression could index power[len(s)] when the SVD rank was exactly 4.
    k_ref = int(min(max(1, len(s) // 4), min(16, len(s) - 1)))
    # Larger power[0] / power[k_ref] means faster spectral decay (low-rank / blur-like).
    # The routing code expects decay_slow=1 for flat, slow-decaying spectra
    # (noise/alias-like), so invert the normalized fast-decay score.
    fast_decay = float(np.clip(np.log((power[0] + EPS) / (power[k_ref] + EPS)) / np.log(40.0), 0.0, 1.0))
    decay_slow = float(1.0 - fast_decay)
    return {
        "cum_energy": cum,
        "decay_slow": decay_slow,
        "n_sv": int(len(s)),
        "orig_hw": orig_hw,
        "U": u,
        "S": s,
        "Vh": vh,
    }


def _lowrank_reuse(
    gray: np.ndarray,
    spectrum: Dict[str, Any],
    keep: float,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Reconstruct low-rank image from a precomputed observation spectrum (no second SVD)."""
    gray = clip01(np.asarray(gray, dtype=np.float32))
    orig_hw = spectrum.get("orig_hw", gray.shape[:2])
    u, s, vh = spectrum.get("U"), spectrum.get("S"), spectrum.get("Vh")
    cum = spectrum.get("cum_energy")
    if u is None or s is None or vh is None or cum is None:
        return _lowrank(gray, keep)

    keep = float(np.clip(keep, 0.62, 0.995))
    rank = int(np.searchsorted(np.asarray(cum, dtype=np.float64), keep) + 1)
    rank = max(4, min(rank, len(s)))
    low_small = (u[:, :rank] * s[:rank]) @ vh[:rank, :]
    low = low_small.astype(np.float32)
    if low.shape[:2] != orig_hw:
        low = cv2.resize(low, (orig_hw[1], orig_hw[0]), interpolation=cv2.INTER_CUBIC)
    low = clip01(low)
    power = (s * s).astype(np.float64)
    total = float(power.sum() + EPS)
    tail = float(np.sum(power[rank:]) / total)
    return low, {
        "rank": float(rank),
        "tail_energy": tail,
        "keep_resolved": keep,
    }


def _lowrank(gray: np.ndarray, keep: float) -> Tuple[np.ndarray, Dict[str, float]]:
    gray = np.asarray(gray, dtype=np.float32)
    if not np.isfinite(gray).all():
        gray = np.nan_to_num(gray, nan=0.5, posinf=1.0, neginf=0.0)
    gray = clip01(gray)
    spec = observation_svd_spectrum(gray)
    return _lowrank_reuse(gray, spec, keep)


def svd_nonlinear_degrade(
    linear: np.ndarray,
    task: str,
    strength: float,
    metrics: Optional[Dict[str, float]] = None,
    linear_meta: Optional[Dict[str, Any]] = None,
    spectrum: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    from tools.em_maps import analyze_em_image

    g = clip01(np.asarray(linear, dtype=np.float32))
    metrics = metrics or analyze_em_image(g)
    linear_meta = linear_meta or {}
    maps = linear_meta.get("maps") or task_support_maps(g)
    membrane = maps["membrane"]
    ridge = maps["ridge"]
    anis_map = maps["anis_map"]
    weak_vertical = bool(float(maps["weak_axis_vertical"]))

    spec_use = spectrum if spectrum is not None and spectrum.get("cum_energy") is not None else observation_svd_spectrum(g)
    keep, keep_diag = resolve_svd_keep(task, strength, metrics, spec_use)
    if task == "inp_em":
        st = float(max(0.0, min(1.55, strength)))
        keep = float(np.clip(keep + 0.03 + 0.05 * (1.0 - st), 0.74, 0.95))
    if spectrum is not None and spectrum.get("U") is not None:
        lowrank, svd_diag = _lowrank_reuse(g, spectrum, keep)
    else:
        lowrank, svd_diag = _lowrank_reuse(g, spec_use, keep)
    svd_diag = {**svd_diag, **keep_diag}
    residual = g - lowrank
    residual_band = residual + 0.38 * (gauss(g, 0.85) - gauss(g, 2.05))

    if task == "deno_em":
        rng = np.random.default_rng(2026 + int(g.shape[0]) * 17 + int(g.shape[1]))
        rand = rng.normal(0.0, 1.0, g.shape).astype(np.float32)
        forward_noise = norm01(np.abs(rand))
        degraded = lowrank + (0.14 + 0.16 * strength) * membrane * residual_band + (0.07 + 0.12 * strength) * (1.0 - membrane) * (rand * 0.055)
        guidance = clip01(0.62 * membrane + 0.38 * ridge)
    elif task == "deblur_em":
        blurred = cv2.GaussianBlur(lowrank, (0, 0), sigmaX=1.15 + 1.0 * strength, sigmaY=1.15 + 1.0 * strength)
        degraded = 0.74 * blurred + 0.26 * lowrank + (0.10 + 0.12 * strength) * membrane * residual_band
        forward_noise = np.abs(blurred - lowrank)
        guidance = clip01(0.72 * membrane + 0.28 * ridge)
    elif task.startswith("sr"):
        low = cv2.resize(lowrank, (max(8, g.shape[1] // 2), max(8, g.shape[0] // 2)), interpolation=cv2.INTER_AREA)
        alias = cv2.resize(low, (g.shape[1], g.shape[0]), interpolation=cv2.INTER_LINEAR)
        degraded = 0.64 * alias + 0.36 * lowrank + (0.08 + 0.10 * strength) * membrane * residual_band
        forward_noise = np.abs(alias - lowrank)
        guidance = clip01(0.52 * ridge + 0.28 * membrane + 0.20 * norm01(forward_noise))
    elif task == "inp_em":
        # New inp_em forward corruption (rewrite):
        # Make the corridor look *more missing* (strong local removal) while exporting a clear guidance gate.
        maps_in = (linear_meta.get("maps") or {}) if isinstance(linear_meta, dict) else {}
        protect = inter_membrane_lumen_protect(g, ridge, membrane).astype(np.float32)

        hole_mask = linear_meta.get("mask") if isinstance(linear_meta, dict) else None
        conf = linear_meta.get("hole_confidence") if isinstance(linear_meta, dict) else None
        if not isinstance(hole_mask, np.ndarray) or hole_mask.shape != g.shape:
            hole_mask, conf = build_inp_mask(g)
        hole_mask = clip01(np.asarray(hole_mask, dtype=np.float32))
        conf = clip01(np.asarray(conf, dtype=np.float32)) if isinstance(conf, np.ndarray) and conf.shape == g.shape else hole_mask

        corridor_core = maps_in.get("corridor_core")
        corridor_soft = maps_in.get("corridor_soft")
        if not isinstance(corridor_soft, np.ndarray) or corridor_soft.shape != g.shape:
            corridor_soft = clip01(gauss(hole_mask * (0.55 + 0.45 * conf), 1.15))
        else:
            corridor_soft = clip01(np.asarray(corridor_soft, dtype=np.float32))
        if not isinstance(corridor_core, np.ndarray) or corridor_core.shape != g.shape:
            corridor_core = clip01(gauss((corridor_soft > 0.55).astype(np.float32), 0.65))
        else:
            corridor_core = clip01(np.asarray(corridor_core, dtype=np.float32))
        # Prefer narrow trajectory trace if provided by new linear inp rewrite.
        trace = maps_in.get("trace")
        if isinstance(trace, np.ndarray) and trace.shape == g.shape:
            trace = clip01(np.asarray(trace, dtype=np.float32))
            corridor_core = clip01(np.maximum(corridor_core, trace))
            k = 3 if st < 0.55 else 5
            it = 1 if st < 0.62 else 2
            corridor_soft = clip01(np.maximum(corridor_soft, cv2.dilate((trace > (0.08 - 0.02 * st)).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)), iterations=it).astype(np.float32)))

        line_closure = maps_in.get("line_closure")
        if not isinstance(line_closure, np.ndarray) or line_closure.shape != g.shape:
            line_closure = membrane_line_gap_map(g, ridge, membrane)
        else:
            line_closure = clip01(np.asarray(line_closure, dtype=np.float32))
        break_sal = maps_in.get("break_saliency")
        if not isinstance(break_sal, np.ndarray) or break_sal.shape != g.shape:
            break_sal = membrane_break_saliency(g, ridge, membrane, protect)
        else:
            break_sal = clip01(np.asarray(break_sal, dtype=np.float32))

        bridge_target = maps_in.get("bridge_target")
        if not isinstance(bridge_target, np.ndarray) and isinstance(linear_meta, dict):
            bridge_target = linear_meta.get("bridge_target")
        if not isinstance(bridge_target, np.ndarray) or bridge_target.shape != g.shape:
            repair_hint = clip01(corridor_soft * (1.0 - 0.96 * protect))
            bridge_target = membrane_bridge_target(g, ridge, membrane, repair_hint=repair_hint, lumen_protect=protect)
        bridge_target = clip01(np.asarray(bridge_target, dtype=np.float32))

        mem_r = clip01(0.46 * membrane + 0.54 * ridge)
        st = float(np.clip(strength, 0.0, 1.55))
        gen_gate = clip01(corridor_soft * (0.55 + 0.45 * mem_r) * (1.0 - 0.985 * protect))
        gen_gate = clip01(gen_gate * (0.72 + 0.78 * st))
        # Make the core gate sharper; soft ring keeps transition smooth.
        core_gate = clip01(gauss((gen_gate > (0.38 - 0.10 * st)).astype(np.float32), 0.42 + 0.36 * st))
        ring = clip01(np.maximum(0.0, gauss(core_gate, 1.15) - 0.88 * core_gate))

        # Strong missing-signal corruption inside the core: move toward local background.
        bg = gauss(g, 5.8)
        fill = clip01(np.maximum(bg, gauss(g, 2.2)))
        degraded = g.copy()
        # Inside core, erase membrane (toward fill) and blur to remove high-frequency ridge.
        core_mix = clip01(1.00 * core_gate + 0.35 * ring)
        degraded = clip01(degraded * (1.0 - core_mix) + fill * core_mix)
        degraded = clip01(degraded * (1.0 - 0.70 * core_gate) + gauss(degraded, 1.55) * (0.70 * core_gate))
        # Keep lumen nearly unchanged.
        degraded = clip01(degraded * (1.0 - 0.90 * protect) + g * (0.90 * protect))

        forward_noise = clip01(core_gate * (0.75 + 0.25 * norm01(np.abs(fill - bridge_target))))
        guidance = clip01(core_gate * 0.88 + ring * 0.32 + 0.10 * line_closure + 0.10 * break_sal)
        guidance = clip01(guidance * (1.0 - 0.98 * protect))

        svd_diag = {
            **svd_diag,
            "mask": hole_mask.astype(np.float32),
            "hole_confidence": conf.astype(np.float32),
            "line_closure": line_closure.astype(np.float32),
            "break_saliency": break_sal.astype(np.float32),
            "bridge_target": bridge_target.astype(np.float32),
            "corridor_core": corridor_core.astype(np.float32),
            "corridor_soft": corridor_soft.astype(np.float32),
            "gen_gate": gen_gate.astype(np.float32),
            "blend_ring": ring.astype(np.float32),
        }
    else:
        axis_blur = cv2.GaussianBlur(lowrank, (0, 0), sigmaX=0.1 if weak_vertical else 2.5, sigmaY=2.5 if weak_vertical else 0.1)
        degraded = 0.70 * axis_blur + 0.30 * lowrank + (0.08 + 0.12 * strength) * ridge * residual_band
        forward_noise = np.abs(axis_blur - lowrank)
        guidance = clip01(0.58 * anis_map + 0.22 * membrane + 0.20 * ridge)

    if task == "inp_em":
        degraded = np.clip(0.92 * degraded + 0.08 * lowrank, 0.0, 1.0)
    else:
        degraded = np.clip(0.86 * degraded + 0.14 * lowrank, 0.0, 1.0)
    meta = {
        **svd_diag,
        "guidance_map": guidance.astype(np.float32),
        "forward_noise": norm01(forward_noise).astype(np.float32),
        "residual_energy": float(np.mean(np.abs(residual))),
    }
    if task == "inp_em" and "mask" not in meta:
        hm, cf = build_inp_mask(g)
        meta["mask"] = hm.astype(np.float32)
        meta["hole_confidence"] = cf.astype(np.float32)
    return degraded.astype(np.float32), meta
