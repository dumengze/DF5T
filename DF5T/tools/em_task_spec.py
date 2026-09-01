"""
Unified EM task semantics: data-driven SVD keep, diffusion start ratios, fusion priors.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from tools.em_tensor import EPS

TASKS = ("deblur_em", "deno_em", "sr2", "inp_em", "isotropic_em")

# Safety bounds for resolved keep (not task-specific priors).
SVD_KEEP_MIN = 0.62
SVD_KEEP_MAX = 0.995
SVD_TAIL_BUDGET_MIN = 0.05
SVD_TAIL_BUDGET_MAX = 0.45


@dataclass(frozen=True)
class TaskPhysicsSpec:
    """EM imaging analogy + numeric anchors for each degradation/restoration path."""

    key: str
    em_semantics: str
    svd_keep_low: float
    svd_keep_high: float
    reverse_start_base: float
    fusion_linear_cap: float
    fusion_nonlinear_boost_in_support: float


SPECS: Dict[str, TaskPhysicsSpec] = {
    "deno_em": TaskPhysicsSpec(
        "deno_em",
        "low-dose shot noise / camera readout on organelle contrast",
        0.88,
        0.96,
        0.44,
        0.92,
        1.0,
    ),
    "deblur_em": TaskPhysicsSpec(
        "deblur_em",
        "CTF/defocus + small beam drift blurring membrane boundaries",
        0.74,
        0.90,
        0.62,
        0.88,
        1.05,
    ),
    "sr2": TaskPhysicsSpec(
        "sr2",
        "pixel-size / upsampling alias and staircasing on fine texture",
        0.68,
        0.86,
        0.68,
        0.85,
        1.12,
    ),
    "inp_em": TaskPhysicsSpec(
        "inp_em",
        "local missing signal at membranes; reverse completes thin dark lines, fusion preserves bilayer lumen",
        0.72,
        0.88,
        0.66,
        0.94,
        1.32,
    ),
    "isotropic_em": TaskPhysicsSpec(
        "isotropic_em",
        "astigmatism / asymmetric illumination causing directional smear",
        0.72,
        0.88,
        0.58,
        0.90,
        1.0,
    ),
}


def svd_keep_for_task(task: str, strength: float) -> float:
    """Legacy fixed-interval keep (fallback when no spectrum is available)."""
    s = float(max(0.0, min(1.55, strength)))
    spec = SPECS.get(task, SPECS["deblur_em"])
    hi = spec.svd_keep_high
    lo = spec.svd_keep_low
    return float(hi - (hi - lo) * s)


def tail_budget_for_task(
    task: str,
    metrics: Dict[str, float],
    spectrum: Optional[Dict[str, Any]] = None,
) -> float:
    """
    Target fraction of singular-value energy to leave in the residual (truncated tail),
    derived from image-quality metrics and optional spectral shape — not fixed per-task keep constants.

    Higher tail_budget -> more aggressive low-rank truncation -> lower keep ratio on this spectrum.
    """
    noise = float(metrics.get("noise", 0.5))
    blur = float(metrics.get("blur", 0.5))
    edge = float(metrics.get("edge", 0.5))
    block = float(metrics.get("block", 0.0))
    contrast = float(metrics.get("contrast", 0.5))
    hole = float(metrics.get("hole", 0.0))
    anis = float(metrics.get("anisotropy", 0.5))

    if task == "deno_em":
        # Noisy, low-contrast fields: more HF attributed to noise -> larger tail budget.
        tau = 0.06 + 0.24 * noise + 0.10 * (1.0 - contrast) - 0.08 * edge
    elif task == "deblur_em":
        # Blur without edge support: lost HF at boundaries -> moderate tail for forward blur probe.
        tau = 0.08 + 0.26 * blur - 0.14 * edge
    elif task.startswith("sr"):
        # Block/alias + blur: corrupt HF -> larger tail for alias forward probe.
        tau = 0.10 + 0.22 * block + 0.16 * blur * (1.0 - edge)
    elif task == "inp_em":
        tau = 0.11 + 0.18 * hole
    else:
        tau = 0.09 + 0.20 * anis

    if spectrum is not None:
        decay_slow = float(spectrum.get("decay_slow", 0.0))
        # Slow singular-value decay: more distributed HF (noise/alias); fast decay: already smooth (blur).
        if task == "deno_em" or task.startswith("sr"):
            tau += 0.07 * decay_slow
        elif task == "deblur_em":
            tau -= 0.06 * decay_slow

    return float(np.clip(tau, SVD_TAIL_BUDGET_MIN, SVD_TAIL_BUDGET_MAX))


def keep_from_spectrum(cum_energy: np.ndarray, tail_budget: float) -> Tuple[float, int]:
    """
    Map tail budget to keep ratio using the observation's cumulative energy curve.
    keep is the actual energy fraction retained at the chosen rank (data-dependent, not a constant).
    """
    cum = np.asarray(cum_energy, dtype=np.float64)
    if cum.size == 0:
        return 0.88, 1
    tail_budget = float(np.clip(tail_budget, SVD_TAIL_BUDGET_MIN, SVD_TAIL_BUDGET_MAX))
    target_keep = 1.0 - tail_budget
    rank_idx = int(np.searchsorted(cum, target_keep))
    rank_idx = int(np.clip(rank_idx, 0, cum.size - 1))
    keep = float(np.clip(cum[rank_idx], SVD_KEEP_MIN, SVD_KEEP_MAX))
    return keep, rank_idx + 1


def resolve_svd_keep(
    task: str,
    strength: float,
    metrics: Dict[str, float],
    spectrum: Optional[Dict[str, Any]] = None,
) -> Tuple[float, Dict[str, float]]:
    """
    Data-driven SVD keep: tail budget from metrics (+ spectral shape) -> keep from cum_energy curve.
    Processing degree scales tail budget (stronger -> more truncation). Falls back to legacy keep if no spectrum.
    """
    diag: Dict[str, float] = {"mode": 0.0}
    if spectrum is None or "cum_energy" not in spectrum:
        keep = svd_keep_for_task(task, strength)
        diag.update({"mode": 0.0, "keep": keep, "tail_budget": float(1.0 - keep), "tail_budget_eff": float(1.0 - keep)})
        return keep, diag

    tau_base = tail_budget_for_task(task, metrics, spectrum)
    s_norm = float(np.clip(strength / 1.55, 0.0, 1.0))
    tau_eff = float(np.clip(tau_base * (0.82 + 0.55 * s_norm), SVD_TAIL_BUDGET_MIN, SVD_TAIL_BUDGET_MAX))
    keep, rank = keep_from_spectrum(np.asarray(spectrum["cum_energy"]), tau_eff)
    tail_actual = float(1.0 - keep)
    diag.update(
        {
            "mode": 1.0,
            "keep": keep,
            "rank": float(rank),
            "tail_budget": tau_base,
            "tail_budget_eff": tau_eff,
            "tail_actual": tail_actual,
            "decay_slow": float(spectrum.get("decay_slow", 0.0)),
        }
    )
    return keep, diag


def reverse_start_base(task: str) -> float:
    return SPECS.get(task, SPECS["deblur_em"]).reverse_start_base


def fusion_caps(task: str) -> Tuple[float, float]:
    sp = SPECS.get(task, SPECS["deblur_em"])
    return sp.fusion_linear_cap, sp.fusion_nonlinear_boost_in_support
