"""EM-specific maps, metrics, and inpainting hole detection."""
from __future__ import annotations

from typing import Any, Dict, Tuple

import cv2
import numpy as np

from tools.em_tensor import EPS, clip01, norm01

# Re-export for EMSVD compatibility
__all__ = [
    "analyze_em_image",
    "build_inp_mask",
    "membrane_bridge_target",
    "membrane_line_gap_map",
    "membrane_break_saliency",
    "inter_membrane_lumen_protect",
    "task_support_maps",
    "membrane_map",
    "ridge_map",
    "noise_map",
    "hole_mask_legacy",
    "anisotropy_map",
    "gauss",
    "lap",
    "sobel_mag",
    "scharr",
    "clahe",
    "normalize_em",
]


def gauss(gray: np.ndarray, sigma: float) -> np.ndarray:
    return cv2.GaussianBlur(np.asarray(gray, dtype=np.float32), (0, 0), sigmaX=max(float(sigma), 0.0), sigmaY=max(float(sigma), 0.0))


def lap(gray: np.ndarray) -> np.ndarray:
    return cv2.Laplacian(np.asarray(gray, dtype=np.float32), cv2.CV_32F, ksize=3)


def sobel_mag(gray: np.ndarray) -> np.ndarray:
    g = np.asarray(gray, dtype=np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx * gx + gy * gy)


def scharr(gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    g = np.asarray(gray, dtype=np.float32)
    gx = cv2.Scharr(g, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(g, cv2.CV_32F, 0, 1)
    return gx, gy


def clahe(gray: np.ndarray, clip: float = 2.0, tile: int = 8) -> np.ndarray:
    g8 = (clip01(gray) * 255.0).astype(np.uint8)
    return cv2.createCLAHE(clipLimit=max(1.0, float(clip)), tileGridSize=(tile, tile)).apply(g8).astype(np.float32) / 255.0


def normalize_em(gray: np.ndarray, strength: float) -> np.ndarray:
    lo, hi = np.percentile(gray, [0.8, 99.2])
    norm = np.clip((gray - lo) / max(float(hi - lo), EPS), 0.0, 1.0)
    alpha = 0.18 + 0.22 * float(strength)
    return np.clip((1.0 - alpha) * gray + alpha * norm, 0.0, 1.0).astype(np.float32)


def membrane_map(gray: np.ndarray) -> np.ndarray:
    g = clip01(gray)
    blackhat = cv2.morphologyEx((g * 255.0).astype(np.uint8), cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))).astype(np.float32) / 255.0
    dog = np.abs(gauss(g, 0.8) - gauss(g, 2.2))
    edge = norm01(sobel_mag(g))
    ridge = np.maximum(0.0, gauss(1.0 - g, 0.8) - gauss(1.0 - g, 2.8))
    score = 0.34 * norm01(blackhat) + 0.30 * norm01(dog) + 0.22 * edge + 0.14 * norm01(ridge)
    return gauss(clip01(score), 0.8)


def ridge_map(gray: np.ndarray) -> np.ndarray:
    g = clip01(gray)
    gx, gy = scharr(g)
    mag = np.sqrt(gx * gx + gy * gy)
    fine = np.abs(g - gauss(g, 0.65))
    coarse = np.abs(gauss(g, 0.65) - gauss(g, 1.8))
    return gauss(clip01(0.55 * norm01(mag) + 0.30 * norm01(fine) + 0.15 * norm01(coarse)), 0.7)


def noise_map(gray: np.ndarray) -> np.ndarray:
    g = clip01(gray)
    hp = g - gauss(g, 0.9)
    local_mean = gauss(hp, 2.0)
    local_var = gauss((hp - local_mean) ** 2, 2.0)
    return norm01(local_var)


def hole_mask_legacy(gray: np.ndarray) -> np.ndarray:
    """Legacy dark+flat heuristic (kept for metrics compatibility)."""
    g = clip01(gray)
    g8 = (g * 255.0).astype(np.uint8)
    thr = int(np.percentile(g8, 7.5))
    dark = (g8 <= thr).astype(np.uint8) * 255
    flat = (norm01(1.0 - sobel_mag(g)) > 0.72).astype(np.uint8) * 255
    mask = cv2.bitwise_and(dark, flat)
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k7)
    return mask


def membrane_bridge_target(
    gray: np.ndarray,
    ridge: np.ndarray,
    membrane: np.ndarray,
    repair_hint: np.ndarray | None = None,
    lumen_protect: np.ndarray | None = None,
) -> np.ndarray:
    """
    Dark bridge hypothesis for broken membranes.
    Only darkens inside biologically plausible repair zones and respects bilayer lumen.

    Updated for inp_em: use longer directional closings so the target represents
    endpoint-to-endpoint continuation across a real membrane break rather than a tiny local blur.
    """
    g = clip01(np.asarray(gray, dtype=np.float32))
    r = clip01(np.asarray(ridge, dtype=np.float32))
    m = clip01(np.asarray(membrane, dtype=np.float32))
    lum = inter_membrane_lumen_protect(g, r, m) if lumen_protect is None else clip01(np.asarray(lumen_protect, dtype=np.float32))
    if repair_hint is None:
        gap = membrane_line_gap_map(g, r, m)
        brk = membrane_break_saliency(g, r, m, lum)
        repair = clip01(0.58 * gap + 0.42 * brk)
    else:
        repair = clip01(np.asarray(repair_hint, dtype=np.float32))
    repair = clip01(gauss(repair, 0.75) * (1.0 - 0.94 * lum))
    inv8 = ((1.0 - g) * 255.0).astype(np.uint8)
    kernels = [
        cv2.getStructuringElement(cv2.MORPH_RECT, (11, 3)),
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 11)),
        cv2.getStructuringElement(cv2.MORPH_RECT, (17, 5)),
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 17)),
        cv2.getStructuringElement(cv2.MORPH_RECT, (21, 3)),
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 21)),
        cv2.getStructuringElement(cv2.MORPH_RECT, (15, 7)),
        cv2.getStructuringElement(cv2.MORPH_RECT, (7, 15)),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
    ]
    acc = np.zeros_like(g, dtype=np.float32)
    for kk in kernels:
        acc = np.maximum(acc, cv2.morphologyEx(inv8, cv2.MORPH_CLOSE, kk).astype(np.float32) / 255.0)
    dark_target = clip01(np.maximum((1.0 - g) * (0.22 + 0.78 * r), gauss(acc, 0.42)))
    bridge = clip01(1.0 - dark_target)
    mem_ctx = gauss(clip01(0.46 * m + 0.54 * r), 0.90)
    bridge = clip01(np.minimum(bridge, g - 0.22 * repair * (0.28 + 0.72 * mem_ctx)))
    bridge = clip01(g * (1.0 - repair) + bridge * repair)
    return bridge.astype(np.float32)


def build_inp_mask(gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Targeted EM inp mask: emphasize *true membrane breaks / local missing signal* rather than
    the whole dark membrane centerline. Also keeps compact beam-damage voids / dead pixels.
    Returns (hole [0,1], confidence [0,1]).
    """
    def _zhang_suen_thinning(bin01: np.ndarray, max_iter: int = 48) -> np.ndarray:
        """Simple Zhang–Suen thinning for binary images (0/1)."""
        img = (bin01 > 0).astype(np.uint8)
        if img.sum() == 0:
            return img
        h, w = img.shape
        it = 0
        changed = True
        while changed and it < max_iter:
            changed = False
            it += 1
            for step in (0, 1):
                m = []
                P = img
                # neighbors
                p2 = np.zeros_like(P); p2[1:, :] = P[:-1, :]
                p3 = np.zeros_like(P); p3[1:, :-1] = P[:-1, 1:]
                p4 = np.zeros_like(P); p4[:, :-1] = P[:, 1:]
                p5 = np.zeros_like(P); p5[:-1, :-1] = P[1:, 1:]
                p6 = np.zeros_like(P); p6[:-1, :] = P[1:, :]
                p7 = np.zeros_like(P); p7[:-1, 1:] = P[1:, :-1]
                p8 = np.zeros_like(P); p8[:, 1:] = P[:, :-1]
                p9 = np.zeros_like(P); p9[1:, 1:] = P[:-1, :-1]
                nb = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
                # transitions 0->1 in ordered neighborhood
                seq = [p2, p3, p4, p5, p6, p7, p8, p9, p2]
                trans = np.zeros_like(P, dtype=np.uint8)
                for a, b in zip(seq[:-1], seq[1:]):
                    trans += ((a == 0) & (b == 1)).astype(np.uint8)
                c1 = (nb >= 2) & (nb <= 6)
                c2 = (trans == 1)
                if step == 0:
                    c3 = (p2 * p4 * p6 == 0)
                    c4 = (p4 * p6 * p8 == 0)
                else:
                    c3 = (p2 * p4 * p8 == 0)
                    c4 = (p2 * p6 * p8 == 0)
                cond = (P == 1) & c1 & c2 & c3 & c4
                ys, xs = np.where(cond)
                if len(ys):
                    m = list(zip(ys.tolist(), xs.tolist()))
                if m:
                    for y, x in m:
                        img[y, x] = 0
                    changed = True
        return img

    def _skeleton_endpoints(skel01: np.ndarray) -> np.ndarray:
        """Return endpoint mask (0/1) for a 1-pixel wide skeleton."""
        sk = (skel01 > 0).astype(np.uint8)
        if sk.sum() == 0:
            return sk
        # 8-neighborhood count
        k = np.ones((3, 3), np.uint8)
        nb = cv2.filter2D(sk, -1, k, borderType=cv2.BORDER_CONSTANT)
        # nb includes self, so endpoint has nb == 2 (self + 1 neighbor)
        end = ((sk == 1) & (nb == 2)).astype(np.uint8)
        return end

    def _pairwise_corridor(end_mask: np.ndarray, mem_ctx: np.ndarray, max_dist: int = 42) -> np.ndarray:
        """Connect nearby endpoints with a soft corridor mask."""
        ys, xs = np.where(end_mask > 0)
        if len(ys) < 2:
            return np.zeros_like(mem_ctx, dtype=np.float32)
        pts = np.stack([ys, xs], axis=1).astype(np.int32)
        used = np.zeros(len(pts), dtype=bool)
        canvas = np.zeros_like(mem_ctx, dtype=np.float32)
        # Greedy pairing by distance, prefer high membrane context at endpoints.
        scores = mem_ctx[ys, xs].astype(np.float32)
        order = np.argsort(-scores)
        for ii in order:
            if used[ii]:
                continue
            p = pts[ii]
            best_j = -1
            best_d = 1e9
            for jj in order:
                if ii == jj or used[jj]:
                    continue
                q = pts[jj]
                dy = int(q[0] - p[0])
                dx = int(q[1] - p[1])
                d2 = dy * dy + dx * dx
                if d2 < best_d and d2 <= max_dist * max_dist:
                    best_d = d2
                    best_j = jj
            if best_j < 0:
                continue
            used[ii] = True
            used[best_j] = True
            q = pts[best_j]
            # draw thin line then blur into corridor
            cv2.line(canvas, (int(p[1]), int(p[0])), (int(q[1]), int(q[0])), color=1.0, thickness=1, lineType=cv2.LINE_AA)
        if float(canvas.max()) <= 0.0:
            return canvas
        canvas = gauss(canvas, 1.25)
        canvas = clip01(canvas * (0.35 + 0.65 * gauss(mem_ctx, 0.9)))
        return canvas.astype(np.float32)

    def _ensure_corridor_coverage(
        hole: np.ndarray,
        conf: np.ndarray,
        score: np.ndarray,
        mem_ctx: np.ndarray,
        lum: np.ndarray,
        min_ratio: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Guarantee a minimum inp corridor coverage so nonlinear reverse is actually triggered.
        This intentionally prioritizes break continuity over conservative sparsity.
        """
        hole = clip01(np.asarray(hole, dtype=np.float32))
        conf = clip01(np.asarray(conf, dtype=np.float32))
        score = clip01(np.asarray(score, dtype=np.float32))
        mem_ctx = clip01(np.asarray(mem_ctx, dtype=np.float32))
        lum = clip01(np.asarray(lum, dtype=np.float32))
        cur = float(np.mean(hole))
        if cur >= float(min_ratio):
            return hole, conf
        target = clip01(score * (0.35 + 0.65 * mem_ctx) * (1.0 - 0.98 * lum))
        if float(target.max()) <= 1e-6:
            return hole, conf
        # choose top-k pixels by target score
        ratio = float(np.clip(min_ratio, 0.002, 0.03))
        thr = float(np.quantile(target, max(0.0, 1.0 - ratio)))
        force_bin = (target >= thr).astype(np.uint8)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        force_bin = cv2.morphologyEx(force_bin, cv2.MORPH_CLOSE, k)
        force_bin = cv2.dilate(force_bin, k, iterations=1)
        force = gauss(force_bin.astype(np.float32), 1.2)
        hole = clip01(np.maximum(hole, force))
        conf = clip01(np.maximum(conf, gauss(target, 0.9)))
        return hole, conf

    g = clip01(np.asarray(gray, dtype=np.float32))
    if not np.isfinite(g).all():
        g = np.nan_to_num(g, nan=0.5, posinf=1.0, neginf=0.0)
    h, w = g.shape[:2]
    npx = float(h * w)

    membrane = membrane_map(g)
    ridge = ridge_map(g)
    lum = inter_membrane_lumen_protect(g, ridge, membrane)

    # Break-focused deficit: if membrane context is strong but local ridge is weak,
    # it's a candidate for a missing segment (break corridor).
    dark_ridge = clip01((1.0 - g) * (0.22 + 0.78 * ridge))
    envelope = gauss(dark_ridge, 3.0)
    deficit = clip01((envelope - dark_ridge) * 5.2)
    gap = membrane_line_gap_map(g, ridge, membrane)
    brk = membrane_break_saliency(g, ridge, membrane, lum)
    ctx = gauss(clip01(0.58 * membrane + 0.42 * ridge), 1.1)
    # Strongly weight true break detectors; keep lumen suppressed.
    bridge_seed = clip01((0.52 * gap + 0.38 * brk + 0.30 * deficit * ctx) * (1.0 - 0.98 * lum))
    # Lower threshold to avoid "almost empty" masks on subtle breaks.
    thr_bridge = float(max(np.percentile(bridge_seed, 72.0), 0.05))
    bridge_bin = (bridge_seed > thr_bridge).astype(np.uint8)
    bridge_bin = cv2.morphologyEx(bridge_bin, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    bridge = gauss(bridge_bin.astype(np.float32), 0.70)
    bridge_conf = gauss(np.maximum(bridge_seed, bridge * 0.85).astype(np.float32), 0.55)

    # If mask is still too sparse, fall back to skeleton endpoint connection.
    # This explicitly targets "broken ends should continue", matching inp requirement.
    if float(bridge.mean()) < 0.0012:
        mem_ctx = clip01(gauss(clip01(0.62 * membrane + 0.38 * ridge), 0.85) * (1.0 - 0.98 * lum))
        # binarize membrane context into candidate curves, thin to skeleton, find endpoints
        thr = float(max(np.percentile(mem_ctx, 78.0), 0.22))
        binm = (mem_ctx > thr).astype(np.uint8)
        binm = cv2.morphologyEx(binm, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        sk = _zhang_suen_thinning(binm, max_iter=42)
        end = _skeleton_endpoints(sk)
        corr = _pairwise_corridor(end, mem_ctx, max_dist=46)
        # Merge corridor into bridge mask with high confidence.
        bridge = clip01(np.maximum(bridge, corr))
        bridge_conf = clip01(np.maximum(bridge_conf, gauss(corr, 0.85)))

    g_med = cv2.medianBlur((g * 255.0).astype(np.uint8), 5).astype(np.float32) / 255.0
    med_dev = np.abs(g - g_med)
    outlier = norm01(med_dev)
    p10 = float(np.percentile(g, 10.0))
    p22 = float(np.percentile(g, 22.0))
    dark = (g <= p22).astype(np.float32)
    very_dark = (g <= p10).astype(np.float32)
    edge = norm01(sobel_mag(g))
    smooth_interior = np.clip(1.0 - edge * 2.4, 0.0, 1.0)
    local_bg = gauss(g, 6.0)
    ring_n = norm01(np.abs(g - local_bg))
    coarse_tex = np.abs(g - gauss(g, 1.4))
    tex_n = norm01(coarse_tex)
    large_shadow = ((gauss(tex_n, 12.0) < 0.06) & (g < float(np.percentile(g, 35.0)))).astype(np.float32)
    membrane_ctx = gauss(clip01(membrane + 0.85 * ridge), 0.9)
    cand = np.clip(0.44 * dark * smooth_interior + 0.28 * very_dark * ring_n + 0.28 * outlier, 0.0, 1.0)
    cand = cand * (1.0 - 0.90 * large_shadow) * (1.0 - 0.97 * membrane_ctx)

    binary = (cand > 0.38).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    void = np.zeros((h, w), dtype=np.float32)
    void_conf = np.zeros((h, w), dtype=np.float32)
    max_area = 0.08 * npx

    for lab in range(1, num):
        area = int(stats[lab, cv2.CC_STAT_AREA])
        if area < 1 or area > max_area:
            continue
        comp = (labels == lab).astype(np.float32)
        bx, by, bw, bh = stats[lab, 0], stats[lab, 1], stats[lab, 2], stats[lab, 3]
        roi = g[by : by + bh, bx : bx + bw].astype(np.float32)
        if roi.size > 0 and float(np.var(roi)) < 2e-5 and area > 0.02 * npx:
            continue
        aspect = max(float(bw) / max(float(bh), 1.0), float(bh) / max(float(bw), 1.0))
        comp_ctx = float(np.mean(membrane_ctx[comp > 0.5])) if np.any(comp > 0.5) else 0.0
        extent = float(area) / max(float(bw * bh), 1.0)
        if comp_ctx > 0.28 or (aspect > 3.0 and extent < 0.48):
            continue
        c = 0.58 * float(np.mean(cand[comp > 0.5])) + 0.22 * float(np.mean(outlier[comp > 0.5])) + 0.20 * float(1.0 - comp_ctx)
        c = float(np.clip(c, 0.0, 1.0))
        void = np.maximum(void, comp)
        void_conf = np.maximum(void_conf, comp * c)

    if float(void.mean()) < 0.00015:
        spike = (outlier > 0.84) & (med_dev > 0.08) & (membrane_ctx < 0.14)
        void = np.maximum(void, spike.astype(np.float32) * 0.85)
        void_conf = np.maximum(void_conf, spike.astype(np.float32) * 0.55)

    # For inp_em, "hole" primarily means break corridor; voids are secondary.
    hole = np.maximum(gauss(bridge, 0.55), gauss(void, 0.25))
    conf = np.maximum(gauss(bridge_conf, 0.55), gauss(void_conf, 0.25))

    # Hard floor for inp activation:
    # if mask is too sparse, force-select top break-score within membrane context
    # so the reverse stage has a real corridor to generate continuity.
    mem_ctx_f = clip01(gauss(0.62 * membrane + 0.38 * ridge, 0.9) * (1.0 - 0.97 * lum))
    forced_score = clip01(0.56 * gap + 0.44 * brk)
    # Global floor for subtle breaks.
    hole, conf = _ensure_corridor_coverage(hole, conf, forced_score, mem_ctx_f, lum, min_ratio=0.0075)
    # Stronger floor for high-contrast membrane scenes.
    if float(np.percentile(mem_ctx_f, 95.0)) > 0.30:
        hole, conf = _ensure_corridor_coverage(hole, conf, forced_score, mem_ctx_f, lum, min_ratio=0.012)
    hole = clip01(hole)
    conf = clip01(conf)
    return hole.astype(np.float32), conf.astype(np.float32)


def membrane_break_saliency(
    gray: np.ndarray,
    ridge: np.ndarray,
    membrane: np.ndarray,
    lumen_protect: np.ndarray | None = None,
) -> np.ndarray:
    """
    Salient **break / weak-ridge** sites for inp (not the thick dark membrane core that SVD low-rank retains).
    Uses ridge-vs-envelope deficit + edge support, suppressed on double-membrane lumina.
    """
    g = clip01(np.asarray(gray, dtype=np.float32))
    r = clip01(np.asarray(ridge, dtype=np.float32))
    m = clip01(np.asarray(membrane, dtype=np.float32))
    # Break saliency should peak at missing segments, not along intact dark cores.
    mem_r = clip01(0.48 * m + 0.52 * r)
    env = gauss(mem_r, 3.2)
    deficit = clip01((env - mem_r) * 4.8)
    inv = clip01(1.0 - g)
    dark = clip01(inv * (0.25 + 0.75 * mem_r))
    # If dark context exists nearby but local dark ridge is weak -> break.
    dark_env = gauss(dark, 2.8)
    dark_def = clip01((dark_env - dark) * 5.0)
    edge = norm01(sobel_mag(g))
    sal = clip01((0.52 * deficit + 0.48 * dark_def) * (0.30 + 0.70 * edge) * gauss(mem_r, 1.25))
    if lumen_protect is not None:
        lu = clip01(np.asarray(lumen_protect, dtype=np.float32))
        if lu.shape == sal.shape:
            sal = clip01(sal * (1.0 - 0.98 * lu))
    return gauss(sal, 0.60).astype(np.float32)


def inter_membrane_lumen_protect(gray: np.ndarray, ridge: np.ndarray, membrane: np.ndarray) -> np.ndarray:
    """
    High (≈1) on thin bright **luminal** slits between parallel membrane ridges (double-membrane gap).
    Used to block inp/SVD/fusion from smearing dark signal into those gaps.
    """
    g = clip01(np.asarray(gray, dtype=np.float32))
    r = clip01(np.asarray(ridge, dtype=np.float32))
    m = clip01(np.asarray(membrane, dtype=np.float32))
    lo = gauss(g, 3.8)
    bright = clip01(np.maximum(0.0, g - lo) * 4.5)
    dr = clip01((1.0 - g) * (0.25 + 0.75 * r))
    p_dr = float(np.percentile(dr, 74.0)) if dr.size else 0.35
    dark_core = (dr > max(p_dr, 0.18)).astype(np.uint8) * 255
    el = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    shell = cv2.dilate(dark_core, el, iterations=2).astype(np.float32) / 255.0
    dark_f = dark_core.astype(np.float32) / 255.0
    between = clip01(shell * (1.0 - gauss(dark_f, 0.55)))
    thr_r = float(np.percentile(r, 70.0)) if r.size else 0.4
    rk = (r > thr_r).astype(np.uint8) * 255
    ridge_ctx = gauss(cv2.dilate(rk, el, iterations=1).astype(np.float32) / 255.0, 1.1)
    prot = clip01(bright * between * ridge_ctx * (0.45 + 0.55 * gauss(m + r, 1.15)))
    return gauss(prot, 0.32).astype(np.float32)


def membrane_line_gap_map(gray: np.ndarray, ridge: np.ndarray, membrane: np.ndarray) -> np.ndarray:
    """
    Conservative map of likely *true* membrane breaks (not double-membrane lumina).
    Uses short morphological closings to avoid bridging paired dark lines.
    """
    g = clip01(np.asarray(gray, dtype=np.float32))
    r = clip01(np.asarray(ridge, dtype=np.float32))
    m = clip01(np.asarray(membrane, dtype=np.float32))
    lum = inter_membrane_lumen_protect(g, r, m)
    inv = clip01(1.0 - g)
    mem_r = clip01(0.48 * m + 0.52 * r)
    d = clip01(inv * (0.18 + 0.82 * mem_r))
    d = gauss(d, 0.35)
    p = float(np.percentile(d, 78.0))
    thr = float(max(p, 0.08))
    seed = (d > thr).astype(np.uint8) * 255
    kernels = [
        cv2.getStructuringElement(cv2.MORPH_RECT, (11, 3)),
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 11)),
        cv2.getStructuringElement(cv2.MORPH_RECT, (9, 4)),
        cv2.getStructuringElement(cv2.MORPH_RECT, (4, 9)),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
    ]
    acc = np.zeros_like(g, dtype=np.float32)
    for kk in kernels:
        acc = np.maximum(acc, cv2.morphologyEx(seed, cv2.MORPH_CLOSE, kk).astype(np.float32) / 255.0)
    dil = cv2.dilate(seed, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    dil_f = dil.astype(np.float32) / 255.0
    gap = clip01(np.maximum(0.0, acc - dil_f))
    ctx = gauss(clip01(mem_r), 1.35)
    gap = clip01(gap * ctx)
    sm = norm01(sobel_mag(g))
    rim = clip01(gauss(sm * clip01(mem_r), 1.05))
    gap = clip01(gap * (0.28 + 0.72 * rim))
    gap = clip01(gap * (1.0 - 0.96 * lum))
    return gauss(gap, 0.55).astype(np.float32)


def anisotropy_map(gray: np.ndarray) -> Tuple[np.ndarray, bool]:
    g = clip01(gray)
    gx = np.abs(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3))
    gy = np.abs(cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3))
    denom = gx + gy + EPS
    amap = np.abs(gx - gy) / denom
    weak_vertical = float(gx.mean()) > float(gy.mean())
    return gauss(clip01(amap * 1.7), 1.0), bool(weak_vertical)


def task_support_maps(gray: np.ndarray) -> Dict[str, Any]:
    membrane = membrane_map(gray)
    ridge = ridge_map(gray)
    hole_u8 = hole_mask_legacy(gray)
    hole_legacy = hole_u8.astype(np.float32) / 255.0
    hole_auto, hole_conf = build_inp_mask(gray)
    anis_map, weak_vertical = anisotropy_map(gray)
    noise = noise_map(gray)
    smooth = clip01(1.0 - ridge)
    return {
        "membrane": membrane,
        "ridge": ridge,
        "hole": hole_auto,
        "hole_confidence": hole_conf,
        "hole_legacy": hole_legacy,
        "anis_map": anis_map,
        "noise": noise,
        "smooth": smooth,
        "weak_axis_vertical": np.array(float(weak_vertical), dtype=np.float32),
    }


def analyze_em_image(gray: np.ndarray) -> Dict[str, float]:
    g = clip01(gray)
    hp = g - gauss(g, 1.0)
    noise = float(np.clip(np.median(np.abs(hp - np.median(hp))) / (0.6745 + EPS) * 3.2, 0.0, 1.0))
    blur = float(np.clip(1.0 / (1.0 + 135.0 * float(np.var(lap(g)))), 0.0, 1.0))
    hole_m, _ = build_inp_mask(g)
    hole = float(np.clip(float(hole_m.mean()) * 5.5, 0.0, 1.0))
    gx = float(np.mean(np.abs(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3))))
    gy = float(np.mean(np.abs(cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3))))
    anisotropy = float(np.clip(abs(gx - gy) / (gx + gy + EPS) * 2.1, 0.0, 1.0))
    edge = float(np.clip(np.mean(ridge_map(g)) * 1.9, 0.0, 1.0))
    contrast = float(np.clip((np.percentile(g, 95.0) - np.percentile(g, 5.0)) * 1.3, 0.0, 1.0))
    block = 0.0
    if g.shape[0] >= 16 and g.shape[1] >= 16:
        v_idx = np.arange(8, g.shape[1], 8)
        h_idx = np.arange(8, g.shape[0], 8)
        vb = float(np.mean(np.abs(g[:, v_idx] - g[:, v_idx - 1]))) if len(v_idx) else 0.0
        hb = float(np.mean(np.abs(g[h_idx, :] - g[h_idx - 1, :]))) if len(h_idx) else 0.0
        neigh = (float(np.mean(np.abs(g[:, 1:] - g[:, :-1]))) + float(np.mean(np.abs(g[1:, :] - g[:-1, :]))) ) * 0.5 + EPS
        block = float(np.clip((vb + hb) * 0.5 / neigh - 0.60, 0.0, 1.0))
    texture = float(np.clip(np.mean(np.abs(g - gauss(g, 0.8))) * 3.5, 0.0, 1.0))
    return {
        "noise": noise,
        "blur": blur,
        "block": block,
        "hole": hole,
        "anisotropy": anisotropy,
        "edge": edge,
        "contrast": contrast,
        "texture": texture,
    }
