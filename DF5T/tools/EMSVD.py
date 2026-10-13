from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import time

from tools.em_adaptive import ADAPTIVE_WEIGHT_MAX, build_adaptive_plan
from tools.em_fusion import organic_fuse
from tools.em_linear import linear_enhance
from tools.em_maps import analyze_em_image
from tools.em_quality import assess_quality_2d, restoration_quality_report
from tools.em_svd import svd_nonlinear_degrade
from tools.em_task_spec import TASKS
from tools.em_slice_align import (
    align_prev_to_current,
    dense_optical_flow_prev_to_cur,
    warp_float01_with_flow,
    warp_float01_with_shift,
)
from tools.em_tensor import EPS, clip01, gray01_to_tensor, tensor_to_gray01

NonlinearSolver = Callable[..., Tuple[np.ndarray, Dict[str, Any]]]

# Matches Diffusion / app cap for adaptive runs (sigma scaled before hitting the engine).
ADAPTIVE_PROCESSING_DEGREE_MAX = 1.55


@dataclass
class StageResult:
    linear: torch.Tensor
    svd_degraded: torch.Tensor
    nonlinear: torch.Tensor
    final: torch.Tensor
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EMTaskOperator:
    task: str
    strength: float = 0.5
    obs_tensor: Optional[torch.Tensor] = None
    patch_idx: int = 0
    adaptive_debug: Dict[str, Any] = field(default_factory=dict)
    last_result: Optional[StageResult] = None
    metrics: Dict[str, float] = field(default_factory=dict)


def process_em_patch(
    obs_tensor: torch.Tensor,
    task: str,
    strength: float = 0.5,
    nonlinear_solver: Optional[NonlinearSolver] = None,
    preview: Optional[Dict[str, Any]] = None,
) -> StageResult:
    gray = tensor_to_gray01(obs_tensor)
    metrics = analyze_em_image(gray)
    preview = preview or {}
    linear = preview.get("linear")
    linear_meta = preview.get("linear_meta")
    svd_deg = preview.get("svd")
    svd_meta = preview.get("svd_meta")

    reuse_linear = isinstance(linear, np.ndarray) and linear.shape == gray.shape and isinstance(linear_meta, dict)
    if not reuse_linear:
        linear, linear_meta = linear_enhance(gray, task, strength, metrics=metrics)
    else:
        linear = np.clip(linear.astype(np.float32), 0.0, 1.0)
        linear_meta = dict(linear_meta)
    if task == "isotropic_em" and not reuse_linear:
        pg = preview.get("prev_gray")
        pl = preview.get("prev_linear")
        ps = preview.get("prev_svd")
        if isinstance(pg, np.ndarray) and pg.shape == gray.shape and isinstance(pl, np.ndarray) and pl.shape == gray.shape:
            h, w = int(gray.shape[0]), int(gray.shape[1])
            st = float(np.clip(strength, 0.0, ADAPTIVE_PROCESSING_DEGREE_MAX))
            prev_sa = preview.get("prev_slice_align")
            prev_shift = None
            if isinstance(prev_sa, dict) and "dx" in prev_sa and "dy" in prev_sa:
                prev_shift = (float(prev_sa["dx"]), float(prev_sa["dy"]))
            align_M: Optional[np.ndarray] = None
            try:
                _wg, conf_f, flow = dense_optical_flow_prev_to_cur(pg, gray, prev_shift=prev_shift)
            except Exception:
                conf_f, flow = 0.0, np.zeros((h, w, 2), dtype=np.float32)
            if conf_f >= 0.05:
                wpl = warp_float01_with_flow(pl, flow, h, w)
                conf = conf_f
                mean_mag = float(np.mean(np.sqrt(flow[:, :, 0] ** 2 + flow[:, :, 1] ** 2)))
                dx = float(np.median(flow[:, :, 0]))
                dy = float(np.median(flow[:, :, 1]))
                linear_meta["slice_align"] = {
                    "align_mode": "dense_farneback",
                    "flow_confidence": float(conf_f),
                    "confidence": float(conf_f),
                    "mean_flow_mag": mean_mag,
                    "dx": dx,
                    "dy": dy,
                }
            else:
                _warped_pg, conf, (dx, dy), align_M = align_prev_to_current(pg, gray, prev_shift=prev_shift)
                wpl = warp_float01_with_shift(pl, dx, dy, h, w, M=align_M)
                linear_meta["slice_align"] = {
                    "align_mode": "phase_correlation_fallback",
                    "flow_confidence": float(conf_f),
                    "confidence": float(conf),
                    "dx": float(dx),
                    "dy": float(dy),
                }
            conf_eff = float(conf)
            if conf_eff > 0.45:
                conf_eff = min(1.0, conf_eff * 1.10)
            w_blend = float(np.clip(conf_eff * (0.18 + 0.32 * st), 0.0, 0.55))
            linear = clip01((1.0 - w_blend) * linear.astype(np.float32) + w_blend * wpl.astype(np.float32))
            linear_meta["slice_align"]["blend_w"] = w_blend
            linear_meta["iso_prev_linear"] = wpl.astype(np.float32)
            if isinstance(ps, np.ndarray) and ps.shape == gray.shape:
                if conf_f >= 0.05:
                    linear_meta["iso_prev_svd"] = warp_float01_with_flow(ps, flow, h, w).astype(np.float32)
                else:
                    linear_meta["iso_prev_svd"] = warp_float01_with_shift(
                        ps,
                        float(linear_meta["slice_align"]["dx"]),
                        float(linear_meta["slice_align"]["dy"]),
                        h,
                        w,
                        M=align_M,
                    ).astype(np.float32)
    if task == "inp_em":
        # Hard activation mode: ensure inp corridor always has visible spatial support.
        maps = linear_meta.get("maps") or {}
        mask = linear_meta.get("mask")
        if not isinstance(mask, np.ndarray) or mask.shape != gray.shape:
            mask = np.zeros_like(gray, dtype=np.float32)
        else:
            mask = np.clip(mask.astype(np.float32), 0.0, 1.0)
        line = maps.get("line_closure")
        brk = maps.get("break_saliency")
        mem = maps.get("membrane")
        rid = maps.get("ridge")
        lum = maps.get("lumen_protect")
        if isinstance(line, np.ndarray) and line.shape == gray.shape:
            score = line.astype(np.float32)
        else:
            score = np.zeros_like(gray, dtype=np.float32)
        if isinstance(brk, np.ndarray) and brk.shape == gray.shape:
            score = np.maximum(score, brk.astype(np.float32))
        if isinstance(mem, np.ndarray) and mem.shape == gray.shape and isinstance(rid, np.ndarray) and rid.shape == gray.shape:
            score = score * np.clip(0.45 * mem.astype(np.float32) + 0.55 * rid.astype(np.float32), 0.0, 1.0)
        if isinstance(lum, np.ndarray) and lum.shape == gray.shape:
            score = score * (1.0 - 0.95 * np.clip(lum.astype(np.float32), 0.0, 1.0))
        score = np.clip(score, 0.0, 1.0)
        # enforce minimum support in a *narrow* break corridor (avoid global blotches)
        if float(np.mean(mask)) < 0.012:
            thr = float(np.quantile(score, 0.992)) if float(np.max(score)) > 1e-6 else 1.0
            forced = (score >= thr).astype(np.float32) if thr < 1.0 else np.zeros_like(score, dtype=np.float32)
            if np.any(forced > 0):
                import cv2
                # remove tiny speckles before expanding corridor
                bin_u8 = (forced > 0.5).astype(np.uint8)
                num, labels, stats, _ = cv2.connectedComponentsWithStats(bin_u8, connectivity=8)
                keep = np.zeros_like(bin_u8, dtype=np.uint8)
                for lab in range(1, num):
                    area = int(stats[lab, cv2.CC_STAT_AREA])
                    if area >= 24:
                        keep[labels == lab] = 1
                forced = keep.astype(np.float32)
                k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                forced = cv2.morphologyEx(forced, cv2.MORPH_CLOSE, k)
                forced = cv2.GaussianBlur(forced.astype(np.float32), (0, 0), 0.9)
            mask = np.clip(np.maximum(mask, forced), 0.0, 1.0)
            linear_meta["mask"] = mask.astype(np.float32)
            hc = linear_meta.get("hole_confidence")
            if isinstance(hc, np.ndarray) and hc.shape == gray.shape:
                linear_meta["hole_confidence"] = np.clip(np.maximum(hc.astype(np.float32), mask * 0.9), 0.0, 1.0).astype(np.float32)
    reuse_svd = isinstance(svd_deg, np.ndarray) and svd_deg.shape == gray.shape and isinstance(svd_meta, dict)
    if not reuse_svd:
        svd_deg, svd_meta = svd_nonlinear_degrade(linear, task, strength, metrics=metrics, linear_meta=linear_meta)
    else:
        svd_deg = np.clip(svd_deg.astype(np.float32), 0.0, 1.0)
        svd_meta = dict(svd_meta)

    if nonlinear_solver is None:
        raise RuntimeError(
            "External diffusion model reverse is required for the nonlinear stage; "
            "no nonlinear_solver was provided. Pass EMGenerativeBackbone.restore_from_svd "
            "or another model-backed solver. Heuristic fallback is disabled."
        )
    nonlinear, non_meta = nonlinear_solver(
        task=task,
        obs_tensor=obs_tensor,
        obs_gray=gray,
        linear_gray=linear,
        svd_gray=svd_deg,
        strength=strength,
        metrics=metrics,
        linear_meta=linear_meta,
        svd_meta=svd_meta,
    )
    if not isinstance(non_meta, dict):
        raise RuntimeError("External nonlinear_solver must return (ndarray, dict).")
    mode = str(non_meta.get("mode", ""))
    if "heuristic" in mode.lower() or "fallback" in mode.lower():
        raise RuntimeError(
            f"External diffusion model reverse is required; solver returned disallowed mode: {mode!r}."
        )
    nonlinear = np.clip(nonlinear, 0.0, 1.0)

    fuse_meta = dict(metrics)
    maps_meta = linear_meta.get("maps") or {}
    fuse_meta.update({
        "support": linear_meta.get("support"),
        "mask": linear_meta.get("mask", svd_meta.get("mask")),
        "hole_confidence": linear_meta.get("hole_confidence", svd_meta.get("hole_confidence")),
        "anisotropy": metrics.get("anisotropy", 0.5),
        "line_closure": maps_meta.get("line_closure"),
        "lumen_protect": maps_meta.get("lumen_protect"),
        "break_saliency": maps_meta.get("break_saliency"),
        "bridge_target": linear_meta.get("bridge_target", maps_meta.get("bridge_target", svd_meta.get("bridge_target"))),
        "bridge_need": maps_meta.get("bridge_need"),
        "endpoint_seed": maps_meta.get("endpoint_seed"),
        "bridge_corridor": maps_meta.get("bridge_corridor"),
        "forced_corridor": maps_meta.get("forced_corridor"),
        # inp_em rewrite gates (prefer svd_meta if present, else maps)
        "gen_gate": svd_meta.get("gen_gate", maps_meta.get("gen_gate")),
        "blend_ring": svd_meta.get("blend_ring", maps_meta.get("blend_ring")),
    })
    final, fusion_weight, fusion_map = organic_fuse(gray, linear, nonlinear, task, strength, fuse_meta)
    qa_scores, qa_weights = assess_quality_2d(final)
    qm = restoration_quality_report(gray, final, linear, nonlinear)
    stage_diff_stats = {
        "linear_svd_l1_mean": float(np.mean(np.abs(linear - svd_deg))),
        "svd_nonlinear_l1_mean": float(np.mean(np.abs(svd_deg - nonlinear))),
        "nonlinear_final_l1_mean": float(np.mean(np.abs(nonlinear - final))),
    }
    inp_diag: Dict[str, float] = {}
    if task == "inp_em":
        mask_arr = fuse_meta.get("mask")
        if isinstance(mask_arr, np.ndarray):
            inp_diag["mask_mean"] = float(np.mean(mask_arr))
            inp_diag["mask_max"] = float(np.max(mask_arr))
        bc = fuse_meta.get("bridge_corridor")
        if isinstance(bc, np.ndarray):
            inp_diag["corridor_mean"] = float(np.mean(bc))
            inp_diag["corridor_max"] = float(np.max(bc))
    meta = {
        "metrics": metrics,
        "linear_meta": linear_meta,
        "svd_meta": svd_meta,
        "nonlinear_meta": non_meta,
        "reused_preview": {
            "linear": bool(reuse_linear),
            "svd": bool(reuse_svd),
        },
        "fusion_weight": fusion_weight,
        "fusion_map": fusion_map,
        "stage_diff_stats": stage_diff_stats,
        "inp_diag": inp_diag,
        "quality_metrics": qm,
        "quality_assessment": {
            "scores": qa_scores,
            "weights_deblur_deno_membrane_sr_iso": qa_weights,
        },
    }
    return StageResult(
        gray01_to_tensor(linear, obs_tensor),
        gray01_to_tensor(svd_deg, obs_tensor),
        gray01_to_tensor(nonlinear, obs_tensor),
        gray01_to_tensor(final, obs_tensor),
        meta,
    )


def _adaptive_weight_map_tensor(weight_map: np.ndarray, ref: torch.Tensor) -> torch.Tensor:
    arr = np.asarray(weight_map, dtype=np.float32)
    t = torch.from_numpy(arr).to(device=ref.device, dtype=ref.dtype)
    return t.unsqueeze(0).unsqueeze(0)


def _adaptive_parallel_fuse(
    obs_tensor: torch.Tensor,
    task_results: Dict[str, StageResult],
    selected_weights: Dict[str, float],
    w_maps: Dict[str, np.ndarray],
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    stage_keys = ("linear", "svd_degraded", "nonlinear", "final")
    weighted_sum = {k: torch.zeros_like(obs_tensor) for k in stage_keys}
    weight_sum = torch.zeros_like(obs_tensor)
    max_weight = torch.zeros_like(obs_tensor)
    per_task_mean: Dict[str, float] = {}

    for task, res in task_results.items():
        wm = np.clip(np.asarray(w_maps.get(task, np.ones(obs_tensor.shape[-2:], dtype=np.float32)), dtype=np.float32), 0.0, 1.0)
        wm_t = _adaptive_weight_map_tensor(float(selected_weights.get(task, 0.0)) * wm, obs_tensor)
        per_task_mean[task] = float(wm_t.mean().detach().cpu().item())
        weight_sum = weight_sum + wm_t
        max_weight = torch.maximum(max_weight, wm_t)
        for key in stage_keys:
            weighted_sum[key] = weighted_sum[key] + getattr(res, key) * wm_t

    anchor = torch.clamp(0.025 + 0.10 * (1.0 - max_weight), 0.025, 0.14)
    denom = weight_sum + anchor + EPS
    fused = {
        "linear": torch.clamp((weighted_sum["linear"] + obs_tensor * anchor) / denom, -1.0, 1.0),
        "svd_degraded": torch.clamp((weighted_sum["svd_degraded"] + obs_tensor * anchor) / denom, -1.0, 1.0),
        "nonlinear": torch.clamp((weighted_sum["nonlinear"] + obs_tensor * anchor) / denom, -1.0, 1.0),
        "final": torch.clamp((weighted_sum["final"] + obs_tensor * anchor) / denom, -1.0, 1.0),
    }
    fusion_diag = {
        "pixel_weight_sum_mean": float(weight_sum.mean().detach().cpu().item()),
        "pixel_weight_sum_max": float(weight_sum.max().detach().cpu().item()),
        "anchor_mean": float(anchor.mean().detach().cpu().item()),
        "per_task_pixel_mean": per_task_mean,
    }
    return fused, fusion_diag


def run_adaptive_chain(
    obs_tensor: torch.Tensor,
    strength: float,
    nonlinear_solver: Optional[NonlinearSolver] = None,
    fast_preview: bool = False,
    plan: Optional[Dict[str, Any]] = None,
) -> Tuple[StageResult, Dict[str, Any]]:
    gray = tensor_to_gray01(obs_tensor)
    strength = float(np.clip(strength, 0.0, ADAPTIVE_PROCESSING_DEGREE_MAX))
    if plan is None:
        plan = build_adaptive_plan(gray, strength, fast_preview=fast_preview)
    selected = plan["selected_tasks"]
    selected_weights = dict(plan.get("selected_weights", {}))
    global_weights = dict(plan.get("global_weights", {}))
    # Robust fallback: adaptive plan may only provide global_weights.
    if not selected_weights:
        selected_weights = {t: float(global_weights.get(t, 0.0)) for t in selected}
    else:
        for t in selected:
            if t not in selected_weights:
                selected_weights[t] = float(global_weights.get(t, 0.0))

    task_results: Dict[str, StageResult] = {}
    previews = dict(plan.get("previews", {}))
    preview_reuse: Dict[str, Dict[str, bool]] = {}
    per_task_s: Dict[str, float] = {}
    per_task_local_strength: Dict[str, float] = {}
    t_chain0 = time.perf_counter()
    for task in selected:
        task_weight = float(selected_weights.get(task, global_weights.get(task, 0.0)))
        w_n = float(np.clip(task_weight / max(float(ADAPTIVE_WEIGHT_MAX), 1e-6), 0.0, 1.0))
        floor = 0.28 + 0.52 * strength
        raw = strength * (0.40 + 1.12 * w_n)
        local_strength = float(np.clip(max(floor, raw), 0.45, 1.0))
        per_task_local_strength[task] = local_strength
        task_preview = previews.get(task, {}) if isinstance(previews.get(task, {}), dict) else {}
        t0 = time.perf_counter()
        res = process_em_patch(
            obs_tensor.detach().clone(),
            task,
            local_strength,
            nonlinear_solver=nonlinear_solver,
            preview=task_preview,
        )
        per_task_s[task] = float(time.perf_counter() - t0)
        pr = res.meta.get("reused_preview", {}) if isinstance(res.meta, dict) else {}
        preview_reuse[task] = {
            "linear": bool(pr.get("linear", False)),
            "svd": bool(pr.get("svd", False)),
        }
        task_results[task] = res

    fused, fusion_diag = _adaptive_parallel_fuse(obs_tensor, task_results, selected_weights, plan.get("w_maps", {}))

    f01 = tensor_to_gray01(fused["final"])
    lin01 = tensor_to_gray01(fused["linear"])
    non01 = tensor_to_gray01(fused["nonlinear"])
    qa_scores, qa_weights = assess_quality_2d(f01)
    qm = restoration_quality_report(gray, f01, lin01, non01)

    result = StageResult(
        fused["linear"],
        fused["svd_degraded"],
        fused["nonlinear"],
        fused["final"],
        {
            "metrics": plan["metrics"],
            "selected_tasks": selected,
            "selected_weights": selected_weights,
            "quality_metrics": qm,
            "quality_assessment": {"scores": qa_scores, "weights_deblur_deno_membrane_sr_iso": qa_weights},
            "fusion_diag": fusion_diag,
        },
    )
    debug = {
        **plan,
        "quality_metrics": qm,
        "quality_assessment": {"scores": qa_scores, "weights_deblur_deno_membrane_sr_iso": qa_weights},
        "selected_weights": selected_weights,
        "task_results": task_results,
        "x_svd_stage": result.svd_degraded.detach().clone(),
        "x_non_stage": result.nonlinear.detach().clone(),
        "r_non_fused": result.nonlinear.detach().clone() - result.svd_degraded.detach().clone(),
        "lambda_non": float(np.mean([float(selected_weights.get(t, global_weights.get(t, 0.0))) for t in selected])) if selected else 0.0,
        "restoration_diag": {
            "selected": selected,
            "selected_weights": selected_weights,
            "global_weights": global_weights,
            "processing_degree_passed": strength,
            "per_task_local_strength": per_task_local_strength,
            "fusion_diag": fusion_diag,
            "preview_reuse": preview_reuse,
            "per_task_s": per_task_s,
            "chain_total_s": float(time.perf_counter() - t_chain0),
            "parallel_from_same_observation": True,
        },
    }
    return result, debug


def _adaptive_run(obs_tensor: torch.Tensor, strength: float, nonlinear_solver: Optional[NonlinearSolver] = None) -> Tuple[StageResult, Dict[str, Any]]:
    return run_adaptive_chain(obs_tensor, strength, nonlinear_solver=nonlinear_solver, fast_preview=False)


def make_operator(task: str, obs_tensor: Optional[torch.Tensor] = None, strength: float = 0.5, patch_idx: int = 0) -> EMTaskOperator:
    smax = ADAPTIVE_PROCESSING_DEGREE_MAX if task == "adaptive" else 1.0
    op = EMTaskOperator(task=task, strength=float(np.clip(strength, 0.0, smax)), obs_tensor=obs_tensor, patch_idx=patch_idx)
    if obs_tensor is not None:
        op.metrics = analyze_em_image(tensor_to_gray01(obs_tensor))
    return op


def em_task_observation(H_funcs: Optional[EMTaskOperator], patch: torch.Tensor, task: str) -> torch.Tensor:
    del H_funcs, task
    return patch


def adaptive_fused_linear_restore(H_funcs: EMTaskOperator, y_0: torch.Tensor, x_prior: Optional[torch.Tensor] = None) -> torch.Tensor:
    del x_prior
    if H_funcs.last_result is None or H_funcs.task != "adaptive":
        H_funcs.last_result, H_funcs.adaptive_debug = _adaptive_run(y_0, H_funcs.strength)
    return H_funcs.last_result.linear.detach().clone()


def direct_em_physics_restore(
    H_funcs: Optional[EMTaskOperator],
    y_0: torch.Tensor,
    x_prior: Optional[torch.Tensor],
    processing_degree: float = 0.5,
    task_name: Optional[str] = None,
) -> torch.Tensor:
    del x_prior
    task = task_name or (H_funcs.task if isinstance(H_funcs, EMTaskOperator) else None) or "deno_em"
    raw_deg = processing_degree if processing_degree is not None else getattr(H_funcs, "strength", 0.5)
    if task == "adaptive":
        strength = float(np.clip(raw_deg, 0.0, ADAPTIVE_PROCESSING_DEGREE_MAX))
    else:
        strength = float(np.clip(raw_deg, 0.0, 1.0))
    if task == "adaptive":
        if H_funcs is None:
            H_funcs = make_operator("adaptive", y_0, strength)
        if H_funcs.last_result is None:
            H_funcs.last_result, H_funcs.adaptive_debug = _adaptive_run(y_0, strength)
        return H_funcs.last_result.final.detach().clone()
    res = process_em_patch(y_0, task, strength)
    if isinstance(H_funcs, EMTaskOperator):
        H_funcs.last_result = res
        H_funcs.metrics = res.meta.get("metrics", {})
    return res.final.detach().clone()


class EMDenoising(EMTaskOperator):
    def __init__(self, *args, **kwargs):
        super().__init__(task="deno_em", strength=float(kwargs.get("strength", 0.5)))


class EMDeblurring(EMTaskOperator):
    def __init__(self, *args, **kwargs):
        super().__init__(task="deblur_em", strength=float(kwargs.get("strength", 0.5)))


class SuperResolutionEM(EMTaskOperator):
    def __init__(self, *args, **kwargs):
        super().__init__(task="sr2", strength=float(kwargs.get("strength", 0.5)))


class Inpainting(EMTaskOperator):
    def __init__(self, *args, **kwargs):
        super().__init__(task="inp_em", strength=float(kwargs.get("strength", 0.5)))


class IsotropicEM(EMTaskOperator):
    def __init__(self, *args, **kwargs):
        super().__init__(task="isotropic_em", strength=float(kwargs.get("strength", 0.5)))

    def process(self, patch: torch.Tensor) -> torch.Tensor:
        return direct_em_physics_restore(self, patch, patch, processing_degree=self.strength, task_name="isotropic_em")

    def process_with_prev_info(self, prev_patch: torch.Tensor, current_patch: torch.Tensor) -> torch.Tensor:
        del prev_patch
        return self.process(current_patch)
