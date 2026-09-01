"""Scalar image quality metrics for EM restoration (no skimage dependency)."""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from tools.em_tensor import EPS, clip01


def _gray_u8(gray: np.ndarray) -> np.ndarray:
    g = np.asarray(gray, dtype=np.float32)
    if g.max() <= 1.0 + 1e-6:
        g = np.clip(g * 255.0, 0.0, 255.0)
    return g.astype(np.uint8)


def single_image_scalar_metrics(gray01: np.ndarray) -> Dict[str, float]:
    """Sharpness / contrast / high-freq noise proxies on float [0,1] grayscale."""
    g = clip01(np.asarray(gray01, dtype=np.float32))
    u8 = _gray_u8(g)
    lap = cv2.Laplacian(u8, cv2.CV_64F, ksize=3)
    lap_var = float(np.var(lap))
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    grad_mean = float(np.mean(np.sqrt(gx * gx + gy * gy + EPS)))
    p_lo, p_hi = float(np.percentile(g, 5.0)), float(np.percentile(g, 95.0))
    contrast = float(np.clip(p_hi - p_lo, 0.0, 1.0))
    smooth = cv2.GaussianBlur(g, (5, 5), 1.0)
    noise_std = float(np.std(g - smooth))
    edges = cv2.Canny(u8, 40, 120)
    edge_density = float(np.mean(edges > 0))
    return {
        "laplacian_variance": lap_var,
        "gradient_mean": grad_mean,
        "percentile_contrast_5_95": contrast,
        "highfreq_noise_std": noise_std,
        "edge_density": edge_density,
    }


def assess_quality_2d(patch: np.ndarray, eps: float = 1e-8) -> Tuple[Dict[str, float], Tuple[float, ...]]:
    """
    Legacy-style 2D quality scores + 5-tuple weights (deblur, deno, inp, sr, iso).
    Compatible with oldvision EMSVD naming intent.
    """
    img = clip01(np.asarray(patch, dtype=np.float32))
    if img.ndim == 3:
        img = np.mean(img, axis=2)
    h, w = img.shape
    u8 = _gray_u8(img)
    lap = cv2.Laplacian(u8, cv2.CV_64F, ksize=3)
    lap_var = float(np.var(lap) + eps)
    ref_lap = 500.0
    blur_score = float(1.0 / (1.0 + lap_var / ref_lap))
    smoothed = cv2.GaussianBlur(u8, (5, 5), 1.0)
    residual = np.abs(u8.astype(np.float32) - smoothed.astype(np.float32))
    noise_score = float(min(1.0, float(np.std(residual) + eps) / 25.0))
    edges = cv2.Canny(u8, 30, 120)
    edge_density = float(np.mean(edges > 0) + eps)
    ref_edge = 0.15
    resolution_score = float(1.0 / (1.0 + edge_density / ref_edge))
    block_size = max(4, min(32, h // 4, w // 4))
    local_std = (cv2.blur(residual ** 2, (block_size, block_size))) ** 0.5
    membrane_score = float(min(1.0, float(np.mean(local_std) + eps) / 20.0))
    iso_score = float(membrane_score * 0.5 + (1.0 - blur_score) * 0.5)
    scores = {
        "blur": blur_score,
        "noise": noise_score,
        "resolution": resolution_score,
        "membrane": membrane_score,
        "iso": iso_score,
    }
    raw = np.array([blur_score, noise_score, membrane_score, resolution_score, iso_score], dtype=np.float64)
    raw = np.clip(raw, 0.05, 1.0)
    weights = tuple(float(x) for x in (raw / raw.sum()).tolist())
    return scores, weights


def restoration_quality_report(
    obs01: np.ndarray,
    final01: np.ndarray,
    linear01: Optional[np.ndarray] = None,
    nonlinear01: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Before/after metrics and simple gains for JSON / logging."""
    obs01 = clip01(np.asarray(obs01, dtype=np.float32))
    final01 = clip01(np.asarray(final01, dtype=np.float32))
    m_in = single_image_scalar_metrics(obs01)
    m_out = single_image_scalar_metrics(final01)
    report: Dict[str, Any] = {
        "input": m_in,
        "output": m_out,
        "delta_laplacian_variance": m_out["laplacian_variance"] - m_in["laplacian_variance"],
        "delta_gradient_mean": m_out["gradient_mean"] - m_in["gradient_mean"],
        "delta_contrast": m_out["percentile_contrast_5_95"] - m_in["percentile_contrast_5_95"],
        "delta_noise_std": m_out["highfreq_noise_std"] - m_in["highfreq_noise_std"],
    }
    o64 = obs01.astype(np.float64).ravel()
    f64 = final01.astype(np.float64).ravel()
    num = float(np.dot(o64 - o64.mean(), f64 - f64.mean()))
    den = float(np.linalg.norm(o64 - o64.mean()) * np.linalg.norm(f64 - f64.mean()) + EPS)
    report["global_correlation"] = float(np.clip(num / den, -1.0, 1.0))
    if linear01 is not None:
        report["linear"] = single_image_scalar_metrics(linear01)
    if nonlinear01 is not None:
        report["nonlinear"] = single_image_scalar_metrics(nonlinear01)
    return report
