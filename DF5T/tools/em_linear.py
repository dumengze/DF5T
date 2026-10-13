"""Linear (deterministic) enhancement branch per EM task."""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from tools.em_maps import (
    analyze_em_image,
    build_inp_mask,
    clahe,
    gauss,
    inter_membrane_lumen_protect,
    membrane_break_saliency,
    membrane_bridge_target,
    membrane_line_gap_map,
    membrane_map,
    normalize_em,
    ridge_map,
    task_support_maps,
)
from tools.em_tensor import EPS, clip01, norm01


def _inp_linear_bridge_extend(g: np.ndarray, strength: float, maps: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    New inp_em linear stage (rewrite):
    - Detect a narrow break corridor (core + soft ring) in membrane context.
    - Predict a continuation path (bridge_target) and imprint it visibly inside the corridor.
    - Export rich maps for SVD-forward corruption, diffusion reverse guidance, and fusion.
    """
    g = clip01(np.asarray(g, dtype=np.float32))
    st = float(np.clip(strength, 0.0, 1.55))
    # Processing-degree-aware scale: low degree => smaller/local inp edits.
    scale = 0.35 + 0.95 * st

    membrane = clip01(np.asarray(maps["membrane"], dtype=np.float32))
    ridge = clip01(np.asarray(maps["ridge"], dtype=np.float32))
    mem_r = clip01(0.46 * membrane + 0.54 * ridge)
    dark_mem = clip01((1.0 - g) * (0.30 + 0.70 * mem_r))
    lumen = inter_membrane_lumen_protect(g, ridge, membrane).astype(np.float32)

    def _skeletonize_binary(bin01: np.ndarray) -> np.ndarray:
        """Morphological skeletonization (0/1 input)."""
        img = (bin01 > 0).astype(np.uint8)
        if img.sum() == 0:
            return img
        skel = np.zeros_like(img, dtype=np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        it = 0
        while True:
            eroded = cv2.erode(img, kernel)
            temp = cv2.dilate(eroded, kernel)
            temp = cv2.subtract(img, temp)
            skel = cv2.bitwise_or(skel, temp)
            img = eroded
            it += 1
            if cv2.countNonZero(img) == 0 or it > 128:
                break
        return (skel > 0).astype(np.uint8)

    def _endpoint_mask(skel01: np.ndarray) -> np.ndarray:
        sk = (skel01 > 0).astype(np.uint8)
        if sk.sum() == 0:
            return sk
        nb = cv2.filter2D(sk, -1, np.ones((3, 3), np.uint8), borderType=cv2.BORDER_CONSTANT)
        return ((sk == 1) & (nb == 2)).astype(np.uint8)

    def _remove_small_components(bin01: np.ndarray, min_area: int) -> np.ndarray:
        b = (bin01 > 0).astype(np.uint8)
        num, labels, stats, _ = cv2.connectedComponentsWithStats(b, connectivity=8)
        out = np.zeros_like(b, dtype=np.uint8)
        for lab in range(1, num):
            if int(stats[lab, cv2.CC_STAT_AREA]) >= int(min_area):
                out[labels == lab] = 1
        return out

    # Local tangent orientation (for direction-consistent endpoint pairing).
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    tangent_x = -gy
    tangent_y = gx

    def _connect_endpoints(endpoint01: np.ndarray, support: np.ndarray, max_dist: int = 120) -> np.ndarray:
        ys, xs = np.where(endpoint01 > 0)
        if len(ys) < 2:
            return np.zeros_like(support, dtype=np.float32)
        pts = np.stack([ys, xs], axis=1).astype(np.int32)
        vals = support[ys, xs]
        order = np.argsort(-vals)
        canvas = np.zeros_like(support, dtype=np.float32)
        used = np.zeros(len(pts), dtype=bool)
        # Pair endpoints greedily; one partner each to avoid global dot spraying.
        for i in order:
            if used[i]:
                continue
            p = pts[i]
            best_j = -1
            best_d = 10**9
            for j in order:
                if i == j or used[j]:
                    continue
                q = pts[j]
                # Direction consistency: connection direction should align with local membrane tangent
                vy = float(q[0] - p[0])
                vx = float(q[1] - p[1])
                vn = float(np.hypot(vx, vy)) + 1e-6
                dx = vx / vn
                dy = vy / vn
                txp = float(tangent_x[p[0], p[1]])
                typ = float(tangent_y[p[0], p[1]])
                txq = float(tangent_x[q[0], q[1]])
                tyq = float(tangent_y[q[0], q[1]])
                tpn = float(np.hypot(txp, typ)) + 1e-6
                tqn = float(np.hypot(txq, tyq)) + 1e-6
                # absolute because tangent has no polarity
                align_p = abs((txp / tpn) * dx + (typ / tpn) * dy)
                align_q = abs((txq / tqn) * dx + (tyq / tqn) * dy)
                if min(align_p, align_q) < 0.45:
                    continue
                d2 = int((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2)
                if d2 < best_d and d2 <= max_dist * max_dist and d2 >= 12 * 12:
                    best_d = d2
                    best_j = j
            if best_j < 0:
                continue
            used[i] = True
            used[best_j] = True
            q = pts[best_j]
            cv2.line(canvas, (int(p[1]), int(p[0])), (int(q[1]), int(q[0])), 1.0, 1, cv2.LINE_8)
        return clip01(canvas)

    # Base break detectors.
    hole, conf = build_inp_mask(g)
    hole = clip01(hole.astype(np.float32))
    conf = clip01(conf.astype(np.float32))
    line_closure = membrane_line_gap_map(g, ridge, membrane)
    break_sal = membrane_break_saliency(g, ridge, membrane, lumen)

    # Trajectory-first corridor:
    # 1) build a thin membrane skeleton, 2) find endpoints, 3) connect endpoint pairs directly with 1px bridge lines.
    mem_seed_thr = float(np.quantile(dark_mem, 0.82)) if float(dark_mem.max()) > 1e-6 else 1.0
    mem_seed = ((dark_mem >= max(mem_seed_thr, 0.18)) & (lumen < 0.65)).astype(np.uint8)
    mem_seed = cv2.morphologyEx(mem_seed, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    mem_seed = _remove_small_components(mem_seed, min_area=24)
    skel = _skeletonize_binary(mem_seed)
    # Hard mask: only allow bridge generation near dark membrane skeleton neighborhood.
    nbr_k = 3 if st < 0.55 else 5
    skeleton_nbr = cv2.dilate(skel.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (nbr_k, nbr_k)), iterations=1)
    dark_gate = ((dark_mem > max(float(np.quantile(dark_mem, 0.76)), 0.16)) & (lumen < 0.65)).astype(np.uint8)
    hard_track_mask = cv2.bitwise_and(skeleton_nbr, dark_gate)
    if float(hard_track_mask.mean()) < 0.002:
        # fallback: keep at least skeleton neighborhood if dark gating is too strict
        hard_track_mask = skeleton_nbr
    endpoint = _endpoint_mask(skel)
    support = clip01((0.52 * line_closure + 0.48 * break_sal) * (0.30 + 0.70 * dark_mem) * (1.0 - 0.98 * lumen))
    bridge_line = _connect_endpoints(endpoint, support, max_dist=int(30 + 46 * scale))
    if float(bridge_line.mean()) < 1e-5:
        # Fallback to top support trajectory if endpoint pairing failed.
        thrf = float(np.quantile(support, 0.992)) if float(support.max()) > 1e-6 else 1.0
        bridge_line = (support >= max(thrf, 0.08)).astype(np.float32)
    # Enforce hard track mask to avoid global dot artifacts.
    bridge_line = clip01(bridge_line * hard_track_mask.astype(np.float32))
    # Keep the core thin (length extension over width expansion).
    bridge_bin = _remove_small_components((bridge_line > 0.02).astype(np.uint8), min_area=18).astype(np.float32)
    core_sigma = 0.20 + 0.18 * st
    corridor_core = clip01(gauss(bridge_bin, core_sigma) * (0.30 + 0.70 * dark_mem) * (1.0 - 0.98 * lumen))
    soft_k = 3 if st < 0.50 else 5
    soft_it = 1 if st < 0.62 else 2
    corridor_soft = clip01(
        cv2.dilate(
            (corridor_core > (0.06 - 0.02 * st)).astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (soft_k, soft_k)),
            iterations=soft_it,
        ).astype(np.float32)
    )
    corridor_soft = clip01(corridor_soft * (1.0 - 0.92 * lumen))

    endpoint_seed = clip01(endpoint.astype(np.float32) * gauss(mem_r, 0.9) * (1.0 - 0.95 * lumen))

    # Predictive continuation target inside the corridor.
    repair_hint = clip01(corridor_soft * (0.35 + 0.65 * mem_r))
    bridge_target = membrane_bridge_target(g, ridge, membrane, repair_hint=repair_hint, lumen_protect=lumen).astype(np.float32)

    # Thin trajectory imprint: prioritize extending membrane length, avoid width bloom.
    trace = clip01(np.maximum(corridor_core, (g - bridge_target) / max(0.08, 0.16 - 0.06 * st)))
    trace = clip01(trace * (0.30 + 0.70 * dark_mem) * (1.0 - 0.98 * lumen))
    shoulder = clip01(np.maximum(0.0, cv2.dilate((trace > 0.08).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1).astype(np.float32) - (trace > 0.08).astype(np.float32)))

    # Linear corridor imprint: strong, but strictly localized.
    base_corr = clip01((0.20 - 0.08 * st) * g + (0.80 + 0.08 * st) * bridge_target)
    out_corr = clip01(base_corr - (0.42 + 0.42 * st) * trace + (0.03 + 0.02 * st) * shoulder)
    out = clip01(g * (1.0 - corridor_soft) + out_corr * corridor_soft)
    # Keep lumen and side walls conservative.
    out = clip01(out * (1.0 - 0.90 * lumen) + g * (0.90 * lumen))

    maps_out = dict(maps)
    maps_out["lumen_protect"] = lumen.astype(np.float32)
    maps_out["line_closure"] = line_closure.astype(np.float32)
    maps_out["break_saliency"] = break_sal.astype(np.float32)
    maps_out["endpoint_seed"] = endpoint_seed.astype(np.float32)
    maps_out["corridor_core"] = corridor_core.astype(np.float32)
    maps_out["corridor_soft"] = corridor_soft.astype(np.float32)
    maps_out["bridge_target"] = bridge_target.astype(np.float32)
    maps_out["trace"] = trace.astype(np.float32)

    return out, {
        "support": clip01(0.60 * mem_r + 0.40 * corridor_soft),
        "mask": clip01(corridor_soft).astype(np.float32),
        "hole_confidence": clip01(np.maximum(conf, corridor_soft)).astype(np.float32),
        "bridge_target": bridge_target.astype(np.float32),
        "maps": maps_out,
        "task_signature": "rewrite_inp_linear_bridge_extend_v1",
    }


def _force_endpoint_connectivity(
    endpoint_seed: np.ndarray,
    repair_core: np.ndarray,
    membrane: np.ndarray,
    ridge: np.ndarray,
    lumen: np.ndarray,
    strength: float,
) -> np.ndarray:
    """
    Explicit endpoint-to-endpoint corridor construction for aggressive inp mode.
    This creates narrow candidate bridges so reverse diffusion has a concrete target path.
    """
    ep = clip01(np.asarray(endpoint_seed, dtype=np.float32))
    core = clip01(np.asarray(repair_core, dtype=np.float32))
    mem_r = clip01(0.46 * np.asarray(membrane, dtype=np.float32) + 0.54 * np.asarray(ridge, dtype=np.float32))
    lum = clip01(np.asarray(lumen, dtype=np.float32))

    score = clip01((0.72 * ep + 0.28 * core) * (0.35 + 0.65 * mem_r) * (1.0 - 0.96 * lum))
    if float(score.max()) < 1e-6:
        return np.zeros_like(score, dtype=np.float32)

    # Candidate endpoints from high-percentile seed.
    thr = float(np.percentile(score, 99.4))
    mask = (score >= max(thr, 0.10)).astype(np.uint8)
    ys, xs = np.where(mask > 0)
    if len(ys) < 2:
        thr = float(np.percentile(score, 98.8))
        mask = (score >= max(thr, 0.07)).astype(np.uint8)
        ys, xs = np.where(mask > 0)
    if len(ys) < 2:
        return np.zeros_like(score, dtype=np.float32)

    pts = np.stack([ys, xs], axis=1).astype(np.int32)
    vals = score[ys, xs]
    order = np.argsort(-vals)
    # Keep only strongest sparse points.
    selected: list[np.ndarray] = []
    min_sep = 18
    for idx in order:
        p = pts[idx]
        if not selected:
            selected.append(p)
            continue
        d2 = [int((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2) for q in selected]
        if min(d2) >= min_sep * min_sep:
            selected.append(p)
        if len(selected) >= 14:
            break
    if len(selected) < 2:
        # fallback: pick top-2 strongest points directly
        flat_idx = np.argpartition(score.reshape(-1), -2)[-2:]
        h, w = score.shape
        p0 = np.array([int(flat_idx[0] // w), int(flat_idx[0] % w)], dtype=np.int32)
        p1 = np.array([int(flat_idx[1] // w), int(flat_idx[1] % w)], dtype=np.int32)
        selected = [p0, p1]

    selected_arr = np.stack(selected, axis=0).astype(np.int32)
    used = np.zeros(len(selected_arr), dtype=bool)
    corr = np.zeros_like(score, dtype=np.float32)
    max_dist = int(95 + 65 * float(np.clip(strength, 0.0, 1.55)))
    thickness = 1 + int(float(strength) > 0.45)

    for i in range(len(selected_arr)):
        if used[i]:
            continue
        p = selected_arr[i]
        best_j = -1
        best_d = 10**9
        for j in range(len(selected_arr)):
            if i == j or used[j]:
                continue
            q = selected_arr[j]
            d2 = int((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2)
            if d2 < best_d and d2 <= max_dist * max_dist:
                best_d = d2
                best_j = j
        if best_j < 0:
            continue
        used[i] = True
        used[best_j] = True
        q = selected_arr[best_j]
        cv2.line(corr, (int(p[1]), int(p[0])), (int(q[1]), int(q[0])), color=1.0, thickness=thickness, lineType=cv2.LINE_AA)

    # hard fallback: if no pairs were drawn, connect two strongest points.
    if float(corr.max()) <= 0.0 and len(selected_arr) >= 2:
        order2 = np.argsort(-score[selected_arr[:, 0], selected_arr[:, 1]])
        a = selected_arr[order2[0]]
        b = selected_arr[order2[1]]
        cv2.line(corr, (int(a[1]), int(a[0])), (int(b[1]), int(b[0])), color=1.0, thickness=max(1, thickness), lineType=cv2.LINE_AA)

    corr = clip01(gauss(corr, 0.85) * (0.40 + 0.60 * mem_r) * (1.0 - 0.96 * lum))
    # ensure non-empty corridor for downstream gating
    if float(corr.mean()) < 1e-6 and float(score.max()) > 1e-6:
        thr2 = float(np.percentile(score, 99.8))
        seed2 = (score >= max(thr2, 0.08)).astype(np.float32)
        corr = clip01(gauss(seed2, 0.9) * (0.45 + 0.55 * mem_r) * (1.0 - 0.96 * lum))
    return corr.astype(np.float32)


def _endpoint_guided_membrane_bridge(
    g: np.ndarray,
    repair_core: np.ndarray,
    membrane: np.ndarray,
    ridge: np.ndarray,
    lumen: np.ndarray,
    strength: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Endpoint-guided linear bridge prior for inp:
    - only acts in break corridor
    - propagates dark membrane evidence along local tangent direction
    - keeps bilayer lumen protected
    Returns: (bridge_prior, endpoint_seed, bridge_corridor).
    """
    g = clip01(np.asarray(g, dtype=np.float32))
    core = clip01(np.asarray(repair_core, dtype=np.float32))
    membrane = clip01(np.asarray(membrane, dtype=np.float32))
    ridge = clip01(np.asarray(ridge, dtype=np.float32))
    lumen = clip01(np.asarray(lumen, dtype=np.float32))
    mem_r = clip01(0.48 * membrane + 0.52 * ridge)

    core_u8 = np.clip(core * 255.0, 0, 255).astype(np.uint8)
    ring = cv2.dilate(core_u8, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1).astype(np.float32) / 255.0 - core
    ring = clip01(ring)
    endpoint_seed = clip01(ring * gauss(mem_r, 0.7) * (1.0 - 0.95 * lumen))
    endpoint_seed = gauss(endpoint_seed, 0.45)

    corridor = clip01(gauss(core, 1.1) * (0.32 + 0.68 * gauss(mem_r, 0.85)) * (1.0 - 0.96 * lumen))
    g_s = gauss(g, 0.85)
    gx = cv2.Sobel(g_s, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g_s, cv2.CV_32F, 0, 1, ksize=3)
    denom = np.sqrt(gx * gx + gy * gy) + EPS
    tx = -gy / denom
    ty = gx / denom

    h, w = g.shape[:2]
    xx, yy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    x = g.copy()
    steps = max(6, min(20, 8 + int(9 * float(np.clip(strength, 0.0, 1.55)))))
    alpha = 0.22 + 0.28 * float(np.clip(strength, 0.0, 1.55))

    for _ in range(steps):
        mxp = np.clip(xx + tx, 0.0, float(w - 1))
        myp = np.clip(yy + ty, 0.0, float(h - 1))
        mxm = np.clip(xx - tx, 0.0, float(w - 1))
        mym = np.clip(yy - ty, 0.0, float(h - 1))
        p_pos = cv2.remap(x, mxp, myp, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        p_neg = cv2.remap(x, mxm, mym, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        # Dark membrane continuation should follow tangent and prefer darker candidate.
        along = np.minimum(p_pos, p_neg)
        local_bg = gauss(x, 1.15)
        target = clip01(0.86 * along + 0.14 * np.minimum(local_bg, x))
        pull = clip01(alpha * corridor * (0.55 + 0.45 * gauss(endpoint_seed, 0.7)))
        x = clip01(x * (1.0 - pull) + target * pull)
        # Keep lumen clean.
        x = clip01(x * (1.0 - 0.84 * lumen) + g * (0.84 * lumen))

    bridge_prior = clip01(g * (1.0 - corridor) + x * corridor)
    return bridge_prior.astype(np.float32), endpoint_seed.astype(np.float32), corridor.astype(np.float32)


def _membrane_structure_completion(
    g: np.ndarray,
    w: np.ndarray,
    membrane: np.ndarray,
    ridge: np.ndarray,
    strength: float,
) -> np.ndarray:
    """
    Bridge membrane gaps inside inp holes: multi-scale morphological closure + ridge/black-hat reinjection.
    EM membranes appear as thin dark curves — closing along dominant axes reconnects broken bilayers.
    """
    g = clip01(np.asarray(g, dtype=np.float32))
    w = clip01(np.asarray(w, dtype=np.float32))
    membrane = clip01(np.asarray(membrane, dtype=np.float32))
    ridge = clip01(np.asarray(ridge, dtype=np.float32))
    mem_r = clip01(0.48 * membrane + 0.52 * ridge)
    lum = inter_membrane_lumen_protect(g, ridge, membrane)
    g8 = (g * 255.0).astype(np.uint8)

    kernels = [
        cv2.getStructuringElement(cv2.MORPH_RECT, (11, 3)),
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 11)),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
    ]
    acc = np.zeros_like(g, dtype=np.float32)
    for kk in kernels:
        acc += cv2.morphologyEx(g8, cv2.MORPH_CLOSE, kk).astype(np.float32) / 255.0
    morph_blend = acc / float(len(kernels))

    bh = cv2.morphologyEx(g8, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))).astype(np.float32) / 255.0
    bh_n = norm01(np.abs(bh))

    pull = clip01(w * (0.32 + 0.68 * gauss(mem_r, 0.8)))
    pull = gauss(pull, 0.75)
    pull = clip01(pull * (1.0 - 0.82 * gauss(lum, 0.45)))
    coef = float(np.clip(0.10 + 0.14 * strength, 0.10, 0.24))
    out = clip01(g * (1.0 - pull * coef) + morph_blend * (pull * coef))

    reinj = clip01(w * mem_r * bh_n * (1.0 - 0.88 * lum))
    ref = gauss(g, 0.48)
    out = clip01(out + (0.04 + 0.06 * strength) * reinj * (ref - out))

    lap = cv2.Laplacian(g8, cv2.CV_32F, ksize=3)
    lap_e = clip01(norm01(np.abs(lap)))
    rim_w = clip01(w * lap_e * gauss(mem_r, 1.1) * (1.0 - 0.85 * lum))
    out = clip01(out + (0.03 + 0.05 * strength) * rim_w * (morph_blend - out))
    return out

# Richardson–Lucy (same as legacy EMSVD)
def _richardson_lucy(gray: np.ndarray, sigma: float, iters: int) -> np.ndarray:
    sigma = max(0.6, float(sigma))
    size = int(round(sigma * 6.0)) | 1
    ax = np.arange(-(size // 2), size // 2 + 1, dtype=np.float32)
    xx, yy = np.meshgrid(ax, ax)
    psf = np.exp(-(xx * xx + yy * yy) / (2.0 * sigma * sigma)).astype(np.float32)
    psf /= np.sum(psf) + EPS
    est = np.clip(gray.astype(np.float32), 1e-4, 1.0)
    psf_flip = np.flip(psf)
    for _ in range(max(2, int(iters))):
        conv = cv2.filter2D(est, -1, psf, borderType=cv2.BORDER_REFLECT)
        est = est * cv2.filter2D(gray / np.clip(conv, 1e-4, None), -1, psf_flip, borderType=cv2.BORDER_REFLECT)
        est = np.clip(est, 0.0, 1.0)
    return est


def linear_enhance(
    gray: np.ndarray,
    task: str,
    strength: float,
    metrics: Optional[Dict[str, float]] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    g = normalize_em(gray, strength)
    metrics = metrics or analyze_em_image(g)
    maps = task_support_maps(g)
    membrane = maps["membrane"]
    ridge = maps["ridge"]
    noise = maps["noise"]
    hole = maps["hole"]
    hole_conf = maps.get("hole_confidence", hole)
    anis_map = maps["anis_map"]
    weak_vertical = bool(float(maps["weak_axis_vertical"]))

    if task == "deno_em":
        g8 = (g * 255.0).astype(np.uint8)
        nlm = cv2.fastNlMeansDenoising(g8, None, h=8 + int(9 * strength), templateWindowSize=7, searchWindowSize=21).astype(np.float32) / 255.0
        bil = cv2.bilateralFilter(g8, d=0, sigmaColor=18 + int(20 * strength), sigmaSpace=3 + int(3 * strength)).astype(np.float32) / 255.0
        base = 0.52 * nlm + 0.36 * bil + 0.12 * g
        hp = g - gauss(g, 0.75)
        preserve = membrane * (0.62 + 0.38 * ridge)
        out = base + (0.06 + 0.20 * strength) * preserve * hp - (0.06 + 0.14 * strength) * noise * hp
        out = np.clip(0.86 * out + 0.14 * clahe(out, clip=1.35 + 1.35 * strength), 0.0, 1.0)
        return out, {"support": preserve, "maps": maps, "task_signature": "variance_stabilized_membrane_denoise"}

    if task == "deblur_em":
        rl = _richardson_lucy(g, sigma=1.65 - 0.32 * strength, iters=5 + int(5 * strength))
        detail = rl - gauss(rl, 0.82)
        shock = np.sign(detail) * np.sqrt(np.abs(detail) + EPS)
        preserve = clip01(0.52 * membrane + 0.48 * ridge)
        out = rl + (0.12 + 0.24 * strength) * preserve * detail + (0.04 + 0.10 * strength) * preserve * shock
        out = np.clip(0.78 * out + 0.22 * clahe(out, clip=1.75 + 1.15 * strength), 0.0, 1.0)
        return out, {"support": preserve, "maps": maps, "task_signature": "membrane_boundary_defocus_recovery"}

    if task.startswith("sr"):
        low = cv2.resize(g, (max(8, g.shape[1] // 2), max(8, g.shape[0] // 2)), interpolation=cv2.INTER_AREA)
        up = cv2.resize(low, (g.shape[1], g.shape[0]), interpolation=cv2.INTER_CUBIC)
        alias = g - up
        band1 = g - gauss(g, 0.65)
        band2 = gauss(g, 0.65) - gauss(g, 1.75)
        preserve = clip01(0.48 * ridge + 0.32 * membrane + 0.20 * norm01(np.abs(alias)))
        out = g + (0.26 + 0.28 * strength) * preserve * band1 + (0.12 + 0.22 * strength) * preserve * band2 + (0.22 + 0.24 * strength) * preserve * alias
        out = np.clip(0.72 * out + 0.28 * clahe(out, clip=2.15 + 1.65 * strength), 0.0, 1.0)
        return out, {"support": preserve, "maps": maps, "task_signature": "subcellular_microtexture_boost"}

    if task == "inp_em":
        return _inp_linear_bridge_extend(g, strength, maps)

    blur_axis = cv2.GaussianBlur(g, (0, 0), sigmaX=0.10 if weak_vertical else 2.35, sigmaY=2.35 if weak_vertical else 0.10)
    orth = cv2.GaussianBlur(g, (0, 0), sigmaX=2.05 if weak_vertical else 0.10, sigmaY=0.10 if weak_vertical else 2.05)
    preserve = clip01(0.58 * anis_map + 0.22 * membrane + 0.20 * ridge)
    out = g + (0.18 + 0.26 * strength) * preserve * (g - blur_axis) + (0.10 + 0.14 * strength) * preserve * (orth - gauss(orth, 1.05))
    out = np.clip(0.82 * out + 0.18 * clahe(out, clip=1.75 + 1.05 * strength), 0.0, 1.0)
    return out, {"support": preserve, "maps": maps, "task_signature": "directional_anisotropy_equalization"}
