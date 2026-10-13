"""Adaptive routing for deblur/deno/sr2 using observation-first SVD forward probes."""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from tools.em_maps import analyze_em_image, membrane_map, ridge_map, task_support_maps
from tools.em_svd import observation_svd_spectrum, svd_nonlinear_degrade
from tools.em_tensor import EPS, clip01

ADAPTIVE_TASKS = ("deblur_em", "deno_em", "sr2")
ADAPTIVE_WEIGHT_GAIN = 6.5
ADAPTIVE_WEIGHT_MAX = 2.0
# Fusion weights below ~1.0 visibly down-weight a branch; keep a floor after scaling (still respect vmax).
ADAPTIVE_WEIGHT_FLOOR = 1.0


def _score_vector(metrics: Dict[str, float], svd_assessment: Dict[str, float], svd_response: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    return {
        # deno_em: noise-dominant artifacts + SVD energy loss response.
        "deno_em": (
            1.55 * metrics["noise"]
            + 0.34 * (1.0 - metrics["contrast"])
            + 0.22 * svd_response["deno_em"]["tail"]
            + 0.26 * svd_assessment["deno_em"]
        ),
        # deblur_em: blur/edge weakness + SVD response in structure support.
        "deblur_em": (
            1.58 * metrics["blur"]
            + 0.30 * (1.0 - metrics["edge"])
            + 0.22 * svd_response["deblur_em"]["support_inv"]
            + 0.26 * svd_assessment["deblur_em"]
        ),
        # sr2: aliasing/blocky textures + high-frequency response collapse.
        "sr2": (
            1.28 * metrics["block"]
            + 0.58 * metrics["blur"]
            + 0.20 * (1.0 - metrics["edge"])
            + 0.22 * svd_response["sr2"]["tail"]
            + 0.24 * svd_assessment["sr2"]
        ),
    }


def _softmax_dict(scores: Dict[str, float], temp: float = 0.55) -> Dict[str, float]:
    keys = list(scores.keys())
    vals = np.array([scores[k] for k in keys], dtype=np.float32) / max(float(temp), 1e-3)
    vals -= float(vals.max())
    probs = np.exp(vals)
    probs /= float(probs.sum()) + EPS
    return {k: float(v) for k, v in zip(keys, probs.tolist())}


def _amplify_weights(
    weights: Dict[str, float],
    gain: float = ADAPTIVE_WEIGHT_GAIN,
    vmax: float = ADAPTIVE_WEIGHT_MAX,
    wfloor: float = ADAPTIVE_WEIGHT_FLOOR,
) -> Dict[str, float]:
    """Scale softmax by gain; if max(raw) > vmax, scale down proportionally; then lift weak branches.

    Per-key clip to vmax kills nuance; proportional cap preserves ordering. A post floor keeps no task
    stuck far below 1.0 in fusion (user-tunable). If the floor pushes max above vmax, rescale once.
    """
    raw = {k: float(v * gain) for k, v in weights.items()}
    if not raw:
        return {}
    mx = max(raw.values())
    if mx <= EPS:
        return {k: 0.0 for k in raw}
    if mx > vmax:
        scale = vmax / mx
        out = {k: float(v * scale) for k, v in raw.items()}
    else:
        out = dict(raw)
    out = {k: float(max(wfloor, v)) for k, v in out.items()}
    mx2 = max(out.values())
    if mx2 > vmax and mx2 > EPS:
        s2 = vmax / mx2
        out = {k: float(v * s2) for k, v in out.items()}
    return out


def assess_observation_svd_response(
    gray: np.ndarray,
    task: str,
    strength: float,
    metrics: Dict[str, float],
    maps: Dict[str, Any],
    spectrum: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, float], Dict[str, Any], np.ndarray]:
    """
    Observation-first routing probe: apply task forward degradation directly on the raw
    observation (no linear preconditioning), then derive task-specific response scalars.

    Physical reading:
      - forward_gap  : mean |obs - forward(obs)| under the task's degradation model
      - support_gap  : forward effect weighted on structure-bearing regions
      - background_gap (deno): forward effect in smooth / low-ridge background
      - tail         : discarded singular-value energy after low-rank truncation
      - alias_sens (sr): forward gap amplified when block/alias metrics are high
    """
    obs = clip01(np.asarray(gray, dtype=np.float32))
    svd_deg, svd_meta = svd_nonlinear_degrade(
        obs, task, strength, metrics=metrics, spectrum=spectrum
    )
    svd_deg = clip01(np.asarray(svd_deg, dtype=np.float32))
    diff = np.abs(obs - svd_deg)

    membrane = clip01(np.asarray(maps["membrane"], dtype=np.float32))
    ridge = clip01(np.asarray(maps["ridge"], dtype=np.float32))
    smooth = clip01(np.asarray(maps.get("smooth", 1.0 - ridge), dtype=np.float32))

    forward_gap = float(np.mean(diff))
    tail = float(svd_meta.get("tail_energy", 0.0))
    rank = float(svd_meta.get("rank", 1.0))
    rank_n = float(np.clip(rank / 48.0, 0.0, 1.0))
    support_inv = float(1.0 - np.clip(np.mean(membrane), 0.0, 1.0))

    response: Dict[str, float] = {
        "gap": forward_gap,
        "tail": tail,
        "support_inv": support_inv,
        "rank_n": rank_n,
    }
    if isinstance(svd_meta.get("keep"), (int, float)):
        response["keep_resolved"] = float(svd_meta["keep"])
    if isinstance(svd_meta.get("tail_budget_eff"), (int, float)):
        response["tail_budget_eff"] = float(svd_meta["tail_budget_eff"])
    if isinstance(svd_meta.get("decay_slow"), (int, float)):
        response["decay_slow"] = float(svd_meta["decay_slow"])

    if task == "deno_em":
        bg_w = clip01(smooth * (1.0 - membrane * 0.85))
        struct_w = clip01(0.45 * membrane + 0.55 * ridge)
        background_gap = float(np.sum(diff * bg_w) / (float(bg_w.sum()) + EPS))
        structure_gap = float(np.sum(diff * struct_w) / (float(struct_w.sum()) + EPS))
        assessment = (
            0.36 * forward_gap
            + 0.34 * background_gap
            + 0.22 * tail
            + 0.08 * structure_gap
        )
        response["background_gap"] = background_gap
        response["structure_gap"] = structure_gap
    elif task == "deblur_em":
        support_gap = float(np.sum(diff * membrane) / (float(membrane.sum()) + EPS))
        assessment = (
            0.44 * support_gap
            + 0.28 * forward_gap
            + 0.18 * support_inv
            + 0.10 * tail
        )
        response["support_gap"] = support_gap
    else:  # sr2
        detail_w = clip01(0.62 * ridge + 0.38 * membrane)
        support_gap = float(np.sum(diff * detail_w) / (float(detail_w.sum()) + EPS))
        block = float(metrics.get("block", 0.0))
        alias_sens = forward_gap * (0.55 + 0.45 * block)
        assessment = (
            0.34 * forward_gap
            + 0.30 * tail
            + 0.22 * alias_sens
            + 0.14 * support_gap
        )
        response["support_gap"] = support_gap
        response["alias_sens"] = alias_sens

    response["assessment"] = float(assessment)
    return response, svd_meta, svd_deg


def build_adaptive_plan(gray: np.ndarray, strength: float, fast_preview: bool = False) -> Dict[str, Any]:
    """
    Routing-only plan: global metrics + observation-first SVD forward probes.
    Does NOT run linear enhancement or diffusion; selected tasks are executed later
    in run_adaptive_chain via the full linear → SVD → nonlinear → fusion pipeline.
    """
    metrics = analyze_em_image(gray)
    maps = task_support_maps(gray)
    spectrum = observation_svd_spectrum(gray)
    svd_assessment: Dict[str, float] = {}
    svd_response: Dict[str, Dict[str, float]] = {}
    probes: Dict[str, Dict[str, Any]] = {}

    for task in ADAPTIVE_TASKS:
        response, svd_meta, svd_deg = assess_observation_svd_response(
            gray, task, strength, metrics=metrics, maps=maps, spectrum=spectrum
        )
        svd_assessment[task] = float(response["assessment"])
        svd_response[task] = {k: v for k, v in response.items() if k != "assessment"}
        probes[task] = {
            "svd_degraded": svd_deg,
            "svd_meta": svd_meta,
            "response": dict(response),
        }

    raw_scores = _score_vector(metrics, svd_assessment, svd_response)
    base_weights = _softmax_dict(raw_scores)
    weights = _amplify_weights(base_weights)

    # Keep routing decision stable on softmax weights; amplify only affects processing strength.
    selection_threshold = 0.18
    top_ranked = [k for k, _ in sorted(base_weights.items(), key=lambda kv: kv[1], reverse=True)]
    selected = [k for k in top_ranked if base_weights[k] >= selection_threshold][:3]
    if not selected:
        selected = [top_ranked[0]]

    selected_weights = {t: float(weights.get(t, 0.0)) for t in selected}
    selection_reasons = {
        t: {
            "rank": int(top_ranked.index(t) + 1),
            "softmax_weight": float(base_weights.get(t, 0.0)),
            "fusion_weight": float(weights.get(t, 0.0)),
            "threshold": float(selection_threshold),
            "selected": bool(t in selected),
            "reason": ("softmax>=threshold" if t in selected and base_weights.get(t, 0.0) >= selection_threshold else
                       "top_fallback" if t in selected else "below_threshold"),
        }
        for t in top_ranked
    }

    w_maps = {
        "deno_em": clip01(maps["membrane"]),
        "deblur_em": clip01(maps["membrane"]),
        "sr2": clip01(maps["ridge"]),
    }

    mode_suffix = "_fast_no_model" if fast_preview else ""
    return {
        "metrics": metrics,
        "svd_scores": svd_assessment,
        "svd_response": svd_response,
        "raw_scores": raw_scores,
        "supported_tasks": list(ADAPTIVE_TASKS),
        "routing_softmax_weights": dict(base_weights),
        "global_weights": weights,
        "selected_weights": selected_weights,
        "selected_tasks": selected,
        "selection_threshold": float(selection_threshold),
        "selection_reasons": selection_reasons,
        "probes": probes,
        "previews": {},  # legacy key; routing no longer caches linear/SVD for reuse
        "w_maps": {k: np.clip(v, 0.0, 1.0).astype(np.float32) for k, v in w_maps.items()},
        "svd_spectrum": {
            "decay_slow": float(spectrum.get("decay_slow", 0.0)),
            "n_sv": int(spectrum.get("n_sv", 0)),
        },
        "routing_mode": f"obs_svd_forward_probe_data_driven_keep{mode_suffix}",
    }
