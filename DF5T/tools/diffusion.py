from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

from DF5T_guided_diffusion.dist_util import load_state_dict
from DF5T_guided_diffusion.gaussian_diffusion import _extract_into_tensor
from DF5T_guided_diffusion.script_util import create_model_and_diffusion
from tools.EMSVD import (
    EMTaskOperator,
    adaptive_fused_linear_restore,
    direct_em_physics_restore,
    make_operator,
    process_em_patch,
    run_adaptive_chain,
)
from tools.em_adaptive import build_adaptive_plan
from tools.em_guided_torch import diffusion_start_ratio, em_guided_prediction
from tools.em_tensor import clip01, gray01_to_tensor, gray01_to_three_m11, norm01, tensor_to_gray01
from tools.em_volume_io import save_z_stack_tiff

_BACKBONE_CACHE: Dict[str, Tuple[torch.nn.Module, Any, Dict[str, Any]]] = {}
EPS = 1e-6


def _save_stage(path: Path, arr: np.ndarray, color_hint: Optional[np.ndarray]):
    out = arr if arr.ndim == 3 else _gray_to_style(arr.astype(np.uint8), color_hint)
    cv2.imwrite(str(path), out)


# -----------------------------------------------------------------------------
# Config / path helpers
# -----------------------------------------------------------------------------

def _cfgv(cfg: Any, *keys: str, default: Any = None) -> Any:
    cur = cfg
    for key in keys:
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(key, None)
        else:
            cur = getattr(cur, key, None)
    return default if cur is None else cur



def _resolve_model_path(explicit: Optional[str], repo_root: Path) -> Path:
    candidates: List[Path] = []
    if explicit:
        p = Path(explicit)
        candidates.extend([p, repo_root / explicit])
    candidates.extend([
        repo_root / "model_2562.pt",
        repo_root / "exp" / "model" / "MitEM" / "model_2562.pt",
        repo_root / "exp" / "model" / "MitEM" / "model_y.pt",
        Path.cwd() / "model_2562.pt",
        Path.cwd() / "exp" / "model" / "MitEM" / "model_2562.pt",
    ])
    seen: List[Path] = []
    for c in candidates:
        if c not in seen:
            seen.append(c)
        if c.is_file():
            return c.resolve()
    for pattern in ("model_2562.pt", "model_y.pt"):
        for found in repo_root.rglob(pattern):
            if found.is_file():
                return found.resolve()
    raise FileNotFoundError(
        "Base diffusion model was not found. Expected model_2562.pt. Searched: "
        + "; ".join(str(p) for p in seen)
    )



def _clean_state_dict(state: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(state, dict):
        for key in ("state_dict", "model", "ema", "weights"):
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break
    cleaned: Dict[str, Any] = {}
    for k, v in state.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[7:]
        cleaned[nk] = v
    return cleaned


# -----------------------------------------------------------------------------
# Diffusion backbone: real model-driven nonlinear reverse process
# -----------------------------------------------------------------------------

class EMGenerativeBackbone:
    def __init__(self, args: Any, config: Any, device: torch.device):
        self.args = args
        self.config = config
        self.device = device
        self.repo_root = Path(__file__).resolve().parents[1]
        self.image_size = int(_cfgv(config, "data", "image_size", default=256))
        self.sample_steps = int(np.clip(int(getattr(args, "timesteps", 24) or 24), 8, 64))
        self.model_path = _resolve_model_path(getattr(args, "model_path", None), self.repo_root)
        self.model, self.diffusion, self.load_diag = self._load_model()
        self.tile = self.image_size
        self._last_reverse_diag: Dict[str, Any] = {}

    def _tile_overlap(self, task: str) -> int:
        t = self.tile
        if task == "inp_em":
            return min(max(int(t * 0.44), 88), t - 28)
        if task == "deblur_em" or (isinstance(task, str) and task.startswith("sr")):
            return min(max(int(t * 0.36), 72), t - 28)
        return min(max(int(t * 0.30), 56), t - 32)

    def _load_model(self):
        cache_key = f"{self.model_path}|{self.sample_steps}|{self.device.type}"
        if cache_key in _BACKBONE_CACHE:
            return _BACKBONE_CACHE[cache_key]
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"External diffusion model checkpoint not found: {self.model_path}"
            )

        model_cfg = {
            "image_size": int(_cfgv(self.config, "model", "image_size", default=self.image_size)),
            "learn_sigma": bool(_cfgv(self.config, "model", "learn_sigma", default=True)),
            "num_channels": int(_cfgv(self.config, "model", "num_channels", default=256)),
            "num_res_blocks": int(_cfgv(self.config, "model", "num_res_blocks", default=2)),
            "channel_mult": _cfgv(self.config, "model", "channel_mult", default=""),
            "num_heads": int(_cfgv(self.config, "model", "num_heads", default=4)),
            "num_head_channels": int(_cfgv(self.config, "model", "num_head_channels", default=64)),
            "attention_resolutions": str(_cfgv(self.config, "model", "attention_resolutions", default="32,16,8")),
            "dropout": float(_cfgv(self.config, "model", "dropout", default=0.0)),
            "diffusion_steps": int(_cfgv(self.config, "diffusion", "num_diffusion_timesteps", default=1500)),
            "noise_schedule": str(_cfgv(self.config, "diffusion", "beta_schedule", default="linear")),
            "timestep_respacing": f"ddim{self.sample_steps}",
            "use_kl": False,
            "predict_xstart": False,
            "rescale_timesteps": False,
            "rescale_learned_sigmas": False,
            "use_checkpoint": False,
            "use_scale_shift_norm": bool(_cfgv(self.config, "model", "use_scale_shift_norm", default=True)),
            "resblock_updown": bool(_cfgv(self.config, "model", "resblock_updown", default=True)),
            "use_fp16": bool(_cfgv(self.config, "model", "use_fp16", default=False)) and self.device.type == "cuda",
            "use_new_attention_order": bool(_cfgv(self.config, "model", "use_new_attention_order", default=True)),
        }
        model, diffusion = create_model_and_diffusion(**model_cfg)
        state = load_state_dict(str(self.model_path), map_location="cpu")
        state = _clean_state_dict(state)
        try:
            missing, unexpected = model.load_state_dict(state, strict=False)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load external diffusion model checkpoint {self.model_path}: {exc}"
            ) from exc
        allow_partial = bool(getattr(self.args, "allow_partial_model_load", False))
        if (missing or unexpected) and not allow_partial:
            raise RuntimeError(
                "External diffusion model checkpoint does not match the configured architecture. "
                f"missing_keys={list(missing)[:20]}, unexpected_keys={list(unexpected)[:20]}. "
                "Set args.allow_partial_model_load=True only if this mismatch is intentional."
            )
        if not callable(getattr(model, "forward", None)):
            raise RuntimeError("External diffusion model must be a torch module with a callable forward().")
        if not hasattr(diffusion, "p_mean_variance") or not hasattr(diffusion, "q_sample"):
            raise RuntimeError("Diffusion object must provide p_mean_variance() and q_sample().")
        model.to(self.device)
        model.eval()
        diag = {
            "model_path": str(self.model_path),
            "missing_keys": list(missing),
            "unexpected_keys": list(unexpected),
            "sample_steps": self.sample_steps,
            "external_model_required": True,
            "heuristic_fallback_enabled": False,
        }
        _BACKBONE_CACHE[cache_key] = (model, diffusion, diag)
        return model, diffusion, diag

    def restore_from_svd(
        self,
        *,
        task: str,
        obs_tensor: torch.Tensor,
        obs_gray: np.ndarray,
        linear_gray: np.ndarray,
        svd_gray: np.ndarray,
        strength: float,
        metrics: Dict[str, float],
        linear_meta: Dict[str, Any],
        svd_meta: Dict[str, Any],
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        nonlinear = self._restore_gray(task, obs_gray, linear_gray, svd_gray, strength, linear_meta, svd_meta)
        diag = dict(self.load_diag)
        if task == "inp_em":
            diag["inp_reverse_diag"] = self._last_reverse_diag
        return nonlinear, {
            "mode": "model_2562_reverse_from_svd",
            "model_path": str(self.model_path),
            "load_diag": diag,
        }

    def _restore_gray(
        self,
        task: str,
        obs_gray: np.ndarray,
        linear_gray: np.ndarray,
        svd_gray: np.ndarray,
        strength: float,
        linear_meta: Dict[str, Any],
        svd_meta: Dict[str, Any],
    ) -> np.ndarray:
        maps = linear_meta.get("maps") or {}
        guide_map = svd_meta.get("guidance_map")
        if guide_map is None:
            guide_map = linear_meta.get("support")
        if guide_map is None:
            guide_map = norm01(np.abs(linear_gray - svd_gray))
        hole_map = linear_meta.get("mask", svd_meta.get("mask"))
        if hole_map is None:
            hole_map = np.zeros_like(guide_map, dtype=np.float32)
        conf_map = linear_meta.get("hole_confidence", svd_meta.get("hole_confidence"))
        if not isinstance(conf_map, np.ndarray) or conf_map.shape != hole_map.shape:
            conf_map = np.ones_like(hole_map, dtype=np.float32)
        else:
            conf_map = clip01(conf_map.astype(np.float32))
        anis_map = maps.get("anis_map", np.zeros_like(guide_map, dtype=np.float32))
        weak_vertical = bool(float(maps.get("weak_axis_vertical", 0.0))) if "weak_axis_vertical" in maps else False

        membrane_guide = None
        line_closure_pad: Optional[np.ndarray] = None
        lumen_pad: Optional[np.ndarray] = None
        break_pad: Optional[np.ndarray] = None
        bridge_pad: Optional[np.ndarray] = None
        if task == "inp_em":
            mo = maps.get("membrane")
            ro = maps.get("ridge")
            if isinstance(mo, np.ndarray) and isinstance(ro, np.ndarray) and mo.shape == obs_gray.shape and ro.shape == obs_gray.shape:
                membrane_guide = clip01(0.46 * mo.astype(np.float32) + 0.54 * ro.astype(np.float32))
            elif isinstance(mo, np.ndarray) and mo.shape == obs_gray.shape:
                membrane_guide = clip01(mo.astype(np.float32))
            lp = maps.get("lumen_protect")
            if isinstance(lp, np.ndarray) and lp.shape == obs_gray.shape:
                lumen_pad = clip01(lp.astype(np.float32))
            lc = maps.get("line_closure")
            if isinstance(lc, np.ndarray) and lc.shape == obs_gray.shape:
                lc = clip01(lc.astype(np.float32))
                if lumen_pad is not None:
                    lc = clip01(lc * (1.0 - 0.80 * lumen_pad))
                line_closure_pad = lc
                conf_map = clip01(np.maximum(conf_map, lc * 0.45))
                if membrane_guide is not None:
                    membrane_guide = clip01(0.62 * membrane_guide + 0.38 * lc)
            if lumen_pad is not None:
                conf_map = clip01(conf_map * (1.0 - 0.58 * lumen_pad))
                if membrane_guide is not None:
                    membrane_guide = clip01(membrane_guide * (1.0 - 0.38 * lumen_pad))
            bs = maps.get("break_saliency")
            if isinstance(bs, np.ndarray) and bs.shape == obs_gray.shape:
                break_pad = clip01(bs.astype(np.float32))
                if lumen_pad is not None:
                    break_pad = clip01(break_pad * (1.0 - 0.88 * lumen_pad))
            bt = linear_meta.get("bridge_target", maps.get("bridge_target", svd_meta.get("bridge_target")))
            if isinstance(bt, np.ndarray) and bt.shape == obs_gray.shape:
                bridge_pad = clip01(bt.astype(np.float32))
                if lumen_pad is not None:
                    bridge_pad = clip01(bridge_pad * (1.0 - 0.70 * lumen_pad) + linear_gray * (0.70 * lumen_pad))

        h, w = obs_gray.shape
        iso_prev_lin_pad: Optional[np.ndarray] = None
        iso_prev_svd_pad: Optional[np.ndarray] = None
        iso_anchor_w = 0.0
        if task == "isotropic_em":
            sa = linear_meta.get("slice_align")
            conf = 0.0
            if isinstance(sa, dict):
                conf = float(sa.get("confidence", sa.get("flow_confidence", 0.0)))
            pl = linear_meta.get("iso_prev_linear")
            ps = linear_meta.get("iso_prev_svd")
            if isinstance(pl, np.ndarray) and pl.shape == (h, w) and conf > 0.05:
                conf_eff = float(conf)
                if conf_eff > 0.45:
                    conf_eff = min(1.0, conf_eff * 1.10)
                tdiff = norm01(np.abs(linear_gray.astype(np.float32) - pl.astype(np.float32)))
                boost = float(np.clip((0.14 + 0.28 * float(np.clip(strength, 0.0, 1.55))) * conf_eff, 0.0, 0.52))
                guide_map = clip01(guide_map.astype(np.float32) * (1.0 + boost * tdiff))
                iso_prev_lin_pad = pl.astype(np.float32)
                if isinstance(ps, np.ndarray) and ps.shape == (h, w):
                    iso_prev_svd_pad = ps.astype(np.float32)
                iso_anchor_w = float(np.clip(conf_eff * (0.08 + 0.16 * float(np.clip(strength, 0.0, 1.55))), 0.0, 0.26))

        tile = self.tile
        overlap = self._tile_overlap(task)
        step = max(20, tile - overlap)
        pad_h = max(tile - h, 0)
        pad_w = max(tile - w, 0)
        if pad_h or pad_w:
            obs_gray = np.pad(obs_gray, ((0, pad_h), (0, pad_w)), mode="reflect")
            linear_gray = np.pad(linear_gray, ((0, pad_h), (0, pad_w)), mode="reflect")
            svd_gray = np.pad(svd_gray, ((0, pad_h), (0, pad_w)), mode="reflect")
            guide_map = np.pad(guide_map, ((0, pad_h), (0, pad_w)), mode="reflect")
            hole_map = np.pad(hole_map, ((0, pad_h), (0, pad_w)), mode="reflect")
            conf_map = np.pad(conf_map, ((0, pad_h), (0, pad_w)), mode="reflect")
            anis_map = np.pad(anis_map, ((0, pad_h), (0, pad_w)), mode="reflect")
            if membrane_guide is not None:
                membrane_guide = np.pad(membrane_guide, ((0, pad_h), (0, pad_w)), mode="reflect")
            if line_closure_pad is not None:
                line_closure_pad = np.pad(line_closure_pad, ((0, pad_h), (0, pad_w)), mode="reflect")
            if lumen_pad is not None:
                lumen_pad = np.pad(lumen_pad, ((0, pad_h), (0, pad_w)), mode="reflect")
            if break_pad is not None:
                break_pad = np.pad(break_pad, ((0, pad_h), (0, pad_w)), mode="reflect")
            if bridge_pad is not None:
                bridge_pad = np.pad(bridge_pad, ((0, pad_h), (0, pad_w)), mode="reflect")
            if iso_prev_lin_pad is not None:
                iso_prev_lin_pad = np.pad(iso_prev_lin_pad, ((0, pad_h), (0, pad_w)), mode="reflect")
            if iso_prev_svd_pad is not None:
                iso_prev_svd_pad = np.pad(iso_prev_svd_pad, ((0, pad_h), (0, pad_w)), mode="reflect")

        H, W = obs_gray.shape
        ys = list(range(0, max(H - tile, 0) + 1, step))
        xs = list(range(0, max(W - tile, 0) + 1, step))
        if ys[-1] != H - tile:
            ys.append(H - tile)
        if xs[-1] != W - tile:
            xs.append(W - tile)

        acc = np.zeros((H, W), dtype=np.float32)
        weight = np.zeros((H, W), dtype=np.float32)
        wy = np.hanning(tile) if tile > 1 else np.ones(1, dtype=np.float32)
        wx = np.hanning(tile) if tile > 1 else np.ones(1, dtype=np.float32)
        win = (wy[:, None] * wx[None, :]).astype(np.float32)
        win = np.maximum(win, 1e-3)

        inp_gate_means: List[float] = []
        inp_pull_inside: List[float] = []
        inp_pull_outside: List[float] = []
        for yi in ys:
            for xi in xs:
                obs_patch = obs_gray[yi:yi + tile, xi:xi + tile]
                lin_patch = linear_gray[yi:yi + tile, xi:xi + tile]
                svd_patch = svd_gray[yi:yi + tile, xi:xi + tile]
                g_patch = guide_map[yi:yi + tile, xi:xi + tile]
                h_patch = hole_map[yi:yi + tile, xi:xi + tile]
                c_patch = conf_map[yi:yi + tile, xi:xi + tile]
                a_patch = anis_map[yi:yi + tile, xi:xi + tile]
                mg_patch = membrane_guide[yi:yi + tile, xi:xi + tile] if membrane_guide is not None else None
                lc_patch = line_closure_pad[yi:yi + tile, xi:xi + tile] if line_closure_pad is not None else None
                lp_patch = lumen_pad[yi:yi + tile, xi:xi + tile] if lumen_pad is not None else None
                br_patch = break_pad[yi:yi + tile, xi:xi + tile] if break_pad is not None else None
                bt_patch = bridge_pad[yi:yi + tile, xi:xi + tile] if bridge_pad is not None else None
                pl_patch = (
                    iso_prev_lin_pad[yi : yi + tile, xi : xi + tile] if iso_prev_lin_pad is not None else None
                )
                ps_patch = (
                    iso_prev_svd_pad[yi : yi + tile, xi : xi + tile] if iso_prev_svd_pad is not None else None
                )
                out, patch_diag = self._reverse_patch(
                    task,
                    obs_patch,
                    lin_patch,
                    svd_patch,
                    g_patch,
                    h_patch,
                    c_patch,
                    a_patch,
                    strength,
                    weak_vertical,
                    mg_patch,
                    lc_patch,
                    lp_patch,
                    br_patch,
                    bt_patch,
                    iso_prev_lin_patch=pl_patch,
                    iso_prev_svd_patch=ps_patch,
                    iso_anchor_w=iso_anchor_w,
                )
                acc[yi:yi + tile, xi:xi + tile] += out * win
                weight[yi:yi + tile, xi:xi + tile] += win
                if task == "inp_em":
                    if "gate_mean" in patch_diag:
                        inp_gate_means.append(float(patch_diag["gate_mean"]))
                    if "pull_inside" in patch_diag:
                        inp_pull_inside.append(float(patch_diag["pull_inside"]))
                    if "pull_outside" in patch_diag:
                        inp_pull_outside.append(float(patch_diag["pull_outside"]))

        fused = acc / np.maximum(weight, 1e-6)
        if task == "inp_em":
            self._last_reverse_diag = {
                "generateGate_mean": float(np.mean(inp_gate_means)) if inp_gate_means else 0.0,
                "inp_pull_inside": float(np.mean(inp_pull_inside)) if inp_pull_inside else 0.0,
                "inp_pull_outside": float(np.mean(inp_pull_outside)) if inp_pull_outside else 0.0,
            }
        else:
            self._last_reverse_diag = {}
        return clip01(fused[:h, :w])

    def _reverse_patch(
        self,
        task: str,
        obs_patch: np.ndarray,
        lin_patch: np.ndarray,
        svd_patch: np.ndarray,
        guide_patch: np.ndarray,
        hole_patch: np.ndarray,
        conf_patch: np.ndarray,
        anis_patch: np.ndarray,
        strength: float,
        weak_vertical: bool,
        membrane_patch: Optional[np.ndarray] = None,
        line_closure_patch: Optional[np.ndarray] = None,
        lumen_protect_patch: Optional[np.ndarray] = None,
        break_saliency_patch: Optional[np.ndarray] = None,
        bridge_patch: Optional[np.ndarray] = None,
        iso_prev_lin_patch: Optional[np.ndarray] = None,
        iso_prev_svd_patch: Optional[np.ndarray] = None,
        iso_anchor_w: float = 0.0,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        tail = float(np.mean(np.abs(lin_patch - svd_patch)))
        residual_energy = float(np.mean(np.abs(obs_patch - svd_patch)))
        start_idx = diffusion_start_ratio(task, strength, tail, residual_energy, self.diffusion.num_timesteps)

        x_obs = gray01_to_three_m11(obs_patch, self.device)
        x_linear = gray01_to_three_m11(lin_patch, self.device)
        x_svd = gray01_to_three_m11(svd_patch, self.device)

        guide_map = torch.from_numpy(clip01(guide_patch))[None, None].to(self.device)
        hole_map = torch.from_numpy(clip01(hole_patch))[None, None].to(self.device)
        hole_conf_t = torch.from_numpy(clip01(conf_patch))[None, None].to(self.device)
        anis_map = torch.from_numpy(clip01(anis_patch))[None, None].to(self.device)
        membrane_guide_t = None
        if membrane_patch is not None:
            membrane_guide_t = torch.from_numpy(clip01(membrane_patch))[None, None].to(self.device)
        line_closure_t = None
        if line_closure_patch is not None:
            line_closure_t = torch.from_numpy(clip01(line_closure_patch))[None, None].to(self.device)
        lumen_protect_t = None
        if lumen_protect_patch is not None:
            lumen_protect_t = torch.from_numpy(clip01(lumen_protect_patch))[None, None].to(self.device)
        break_saliency_t = None
        if break_saliency_patch is not None:
            break_saliency_t = torch.from_numpy(clip01(break_saliency_patch))[None, None].to(self.device)
        bridge_target_t = None
        if bridge_patch is not None:
            bridge_target_t = gray01_to_three_m11(clip01(bridge_patch), self.device)

        detail_seed = x_linear - x_svd
        detail_seed = detail_seed / (detail_seed.abs().amax(dim=(1, 2, 3), keepdim=True) + 1e-6)
        rand_noise = torch.randn_like(x_svd)
        noise = (0.68 * rand_noise + 0.32 * detail_seed).clamp(-3.0, 3.0)
        t0 = torch.tensor([start_idx], device=self.device, dtype=torch.long)
        x = self.diffusion.q_sample(x_svd, t0, noise=noise)

        gate_mean = 0.0
        pull_inside = 0.0
        pull_outside = 0.0
        with torch.no_grad():
            for i in range(start_idx, -1, -1):
                t = torch.tensor([i], device=self.device, dtype=torch.long)
                out = self.diffusion.p_mean_variance(self.model, x, t, clip_denoised=True)
                pred = em_guided_prediction(
                    out["pred_xstart"],
                    x_obs,
                    x_linear,
                    x_svd,
                    guide_map,
                    hole_map,
                    anis_map,
                    task,
                    strength,
                    i,
                    start_idx,
                    weak_vertical,
                    hole_conf=hole_conf_t,
                    membrane_guide=membrane_guide_t,
                    line_closure=line_closure_t,
                    lumen_protect=lumen_protect_t,
                    break_saliency=break_saliency_t,
                    bridge_target=bridge_target_t,
                )
                if i == 0:
                    x = pred
                    continue
                eps = self.diffusion._predict_eps_from_xstart(x, t, pred)
                alpha_prev = _extract_into_tensor(self.diffusion.alphas_cumprod_prev, t, x.shape)
                x = pred * torch.sqrt(alpha_prev) + torch.sqrt(torch.clamp(1.0 - alpha_prev, min=0.0)) * eps
                if task == "inp_em":
                    pull = 0.0005 + 0.0035 * (float(i) / max(float(start_idx), 1.0))
                    anchor = 0.10 * x_svd + 0.12 * x_linear + 0.78 * x_obs
                    repair_gate = hole_map
                    if line_closure_t is not None:
                        repair_gate = torch.clamp(torch.maximum(repair_gate, 1.00 * line_closure_t), 0.0, 1.0)
                    if break_saliency_t is not None:
                        repair_gate = torch.clamp(torch.maximum(repair_gate, 0.92 * break_saliency_t), 0.0, 1.0)
                    if bridge_target_t is not None:
                        bridge_need_t = torch.clamp((x_linear - bridge_target_t) / 0.18, 0.0, 1.0)
                        repair_gate = torch.clamp(torch.maximum(repair_gate, 0.96 * bridge_need_t), 0.0, 1.0)
                    if lumen_protect_t is not None:
                        repair_gate = repair_gate * (1.0 - 0.85 * lumen_protect_t)
                    # hard floor for corridor gate
                    if float(repair_gate.mean().detach().cpu()) < 0.06:
                        q = torch.quantile(guide_map.flatten(1), 0.93, dim=1, keepdim=True).view(guide_map.shape[0], 1, 1, 1)
                        forced_gate = (guide_map >= q).float()
                        forced_gate = torch.nn.functional.avg_pool2d(forced_gate, kernel_size=9, stride=1, padding=4).clamp(0.0, 1.0)
                        repair_gate = torch.maximum(repair_gate, 0.98 * forced_gate).clamp(0.0, 1.0)
                    # Strong anchoring only outside repair corridor; inside, allow nonlinear generation.
                    pull_map = torch.clamp(pull * (1.0 - 0.999 * repair_gate), 0.0, 1.0)
                    gate_mean = max(gate_mean, float(repair_gate.mean().detach().cpu()))
                    pull_inside = max(pull_inside, float((pull_map * repair_gate).mean().detach().cpu()))
                    pull_outside = max(pull_outside, float((pull_map * (1.0 - repair_gate)).mean().detach().cpu()))
                    x = x * (1.0 - pull_map) + anchor * pull_map
                else:
                    pull = 0.03 + 0.05 * (float(i) / max(float(start_idx), 1.0))
                    anchor = 0.65 * x_svd + 0.25 * x_linear + 0.10 * x_obs
                    if (
                        task == "isotropic_em"
                        and iso_anchor_w > 1e-6
                        and iso_prev_lin_patch is not None
                        and iso_prev_lin_patch.shape == lin_patch.shape
                    ):
                        x_prev_lin = gray01_to_three_m11(clip01(iso_prev_lin_patch), self.device)
                        anchor = (1.0 - iso_anchor_w) * anchor + iso_anchor_w * (
                            0.52 * x_svd + 0.22 * x_linear + 0.10 * x_obs + 0.16 * x_prev_lin
                        )
                        if iso_prev_svd_patch is not None and iso_prev_svd_patch.shape == svd_patch.shape:
                            x_prev_svd = gray01_to_three_m11(clip01(iso_prev_svd_patch), self.device)
                            anchor = anchor + (iso_anchor_w * 0.08) * (x_svd - x_prev_svd)
                    x = x * (1.0 - pull) + anchor * pull
                x = x.clamp(-1.0, 1.0)

        gray = ((x.clamp(-1.0, 1.0) + 1.0) * 0.5).mean(dim=1, keepdim=False)[0]
        return gray.detach().cpu().numpy().astype(np.float32), {
            "gate_mean": gate_mean,
            "pull_inside": pull_inside,
            "pull_outside": pull_outside,
        }


# -----------------------------------------------------------------------------
# Lightweight public wrappers
# -----------------------------------------------------------------------------

def linear_nonlinear_joint_restore_with_stages(
    x_prior: torch.Tensor,
    H_funcs: Optional[EMTaskOperator],
    y_0: torch.Tensor,
    deg: Optional[str] = None,
    u_map: Optional[torch.Tensor] = None,
    nonlinear_solver=None,
):
    del x_prior, u_map
    task = deg or (H_funcs.task if isinstance(H_funcs, EMTaskOperator) else None) or "deno_em"
    strength = getattr(H_funcs, "strength", 0.5)
    if task == "adaptive":
        if H_funcs is None:
            H_funcs = make_operator("adaptive", y_0, strength)
        final = direct_em_physics_restore(H_funcs, y_0, y_0, processing_degree=strength, task_name="adaptive")
        res = H_funcs.last_result
        return final, {"linear": res.linear, "svd_degraded": res.svd_degraded, "nonlinear": res.nonlinear, "final": res.final}
    res = process_em_patch(y_0, task, strength, nonlinear_solver=nonlinear_solver)
    return res.final, {"linear": res.linear, "svd_degraded": res.svd_degraded, "nonlinear": res.nonlinear, "final": res.final}



def linear_nonlinear_joint_restore(
    x_prior: torch.Tensor,
    H_funcs: Optional[EMTaskOperator],
    y_0: torch.Tensor,
    deg: Optional[str] = None,
    u_map: Optional[torch.Tensor] = None,
    nonlinear_solver=None,
):
    out, _ = linear_nonlinear_joint_restore_with_stages(x_prior, H_funcs, y_0, deg=deg, u_map=u_map, nonlinear_solver=nonlinear_solver)
    return out



def build_adaptive_fused_h_and_y0(
    patch: torch.Tensor,
    channels: int,
    patch_h: int,
    patch_w: int,
    device: torch.device,
    processing_degree: float,
    args,
    patch_idx: int,
    input_is_gray: bool = True,
    is_grayscale: bool = True,
):
    del channels, patch_h, patch_w, device, args, input_is_gray, is_grayscale
    op = make_operator("adaptive", patch, strength=processing_degree, patch_idx=patch_idx)
    y_0 = patch.detach().clone()
    sigma = float(np.clip(0.03 + 0.16 * processing_degree, 0.02, 0.20))
    return op, y_0, sigma



def finalize_em_output_uint8(image: np.ndarray, task: str, is_grayscale: bool = False) -> np.ndarray:
    img = np.asarray(image)
    gray = img if img.ndim == 2 else cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)
    g_raw = gray.astype(np.float32) / 255.0
    g = g_raw.copy()
    lo, hi = np.percentile(g, [0.8, 99.4])
    g_stretch = np.clip((g - lo) / max(float(hi - lo), 1e-6), 0.0, 1.0)
    # Keep postprocess mild: mostly preserve model output, only apply light tone normalization.
    g = np.clip(0.80 * g_raw + 0.20 * g_stretch, 0.0, 1.0)
    if task == "deblur_em":
        g = np.clip(g + 0.025 * (g - cv2.GaussianBlur(g, (0, 0), 0.9)), 0.0, 1.0)
    elif task.startswith("sr"):
        g = np.clip(g + 0.03 * (g - cv2.GaussianBlur(g, (0, 0), 0.8)), 0.0, 1.0)
    elif task == "deno_em":
        g = cv2.GaussianBlur(g, (0, 0), 0.15)
    elif task == "inp_em":
        g = np.clip(g + 0.018 * (g - cv2.GaussianBlur(g, (0, 0), 0.40)), 0.0, 1.0)
    out = (g * 255.0).round().astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_GRAY2BGR) if (is_grayscale or img.ndim == 3) else out


# -----------------------------------------------------------------------------
# IO helpers
# -----------------------------------------------------------------------------

def _read_list_file(txt_path: str, root: str) -> List[Path]:
    root_p = Path(root)
    items: List[Path] = []
    txt = Path(txt_path)
    if txt.is_file():
        for line in txt.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            stem = line.split()[0]
            for ext in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"):
                p = root_p / f"{stem}{ext}"
                if p.exists():
                    items.append(p)
                    break
    if items:
        return items
    return sorted([p for p in root_p.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}])



def _to_gray_and_hint(img: np.ndarray):
    if img.ndim == 2:
        return img.astype(np.uint8), None
    ycrcb = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2YCrCb)
    return ycrcb[:, :, 0].astype(np.uint8), ycrcb



def _gray_to_style(gray_u8: np.ndarray, color_hint: Optional[np.ndarray]):
    if color_hint is None:
        return gray_u8
    ycrcb = color_hint.copy()
    ycrcb[:, :, 0] = gray_u8
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)



def _jsonable(obj: Any) -> Any:
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, np.ndarray):
        return {
            "shape": list(obj.shape),
            "min": float(np.min(obj)),
            "max": float(np.max(obj)),
            "mean": float(np.mean(obj)),
        }
    if torch.is_tensor(obj):
        t = obj.detach().float().cpu()
        return {
            "shape": list(t.shape),
            "min": float(t.min()),
            "max": float(t.max()),
            "mean": float(t.mean()),
        }
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return str(obj)


@dataclass
class _ProcessedSample:
    stem: str
    linear: np.ndarray
    svd: np.ndarray
    nonlinear: np.ndarray
    final: np.ndarray
    debug: Dict[str, object]


class Diffusion(object):
    def __init__(self, args, config, device: Optional[torch.device] = None):
        self.args = args
        self.config = config
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.root = getattr(getattr(config, "data", object()), "root", getattr(args, "exp", "."))
        self.list_file = getattr(getattr(config, "data", object()), "txt", "")
        self.output_dir = Path(args.image_folder)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.deg = args.deg
        # Adaptive mode may use >1.0 to make per-task restoration scale visibly stronger.
        self.processing_degree = float(np.clip(getattr(args, "sigma_0", 0.55), 0.0, 1.55))
        self.apply_result_light_enhance = bool(getattr(args, "apply_result_light_enhance", False))
        self.backbone: Optional[EMGenerativeBackbone] = None
        self.status_cb = getattr(args, "status_cb", None)
        self._iso_prev_gray: Optional[np.ndarray] = None
        self._iso_prev_linear: Optional[np.ndarray] = None
        self._iso_prev_svd: Optional[np.ndarray] = None
        self._iso_prev_slice_align: Optional[Dict[str, Any]] = None

    def _status(self, msg: str) -> None:
        cb = self.status_cb
        if callable(cb):
            try:
                cb(msg)
            except Exception:
                pass

    def _get_backbone(self) -> EMGenerativeBackbone:
        if self.backbone is None:
            self._status("Loading diffusion backbone (model init)...")
            self.backbone = EMGenerativeBackbone(self.args, self.config, self.device)
        return self.backbone

    def _process_single_task(
        self,
        obs: torch.Tensor,
        task: str,
        strength: float,
        preview: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, object]]:
        backbone = self._get_backbone()
        res = process_em_patch(obs, task, strength, nonlinear_solver=backbone.restore_from_svd, preview=preview)
        inp_gate_map = None
        if task == "inp_em":
            try:
                lm = res.meta.get("linear_meta", {}) if isinstance(res.meta, dict) else {}
                maps = (lm.get("maps") or {}) if isinstance(lm, dict) else {}
                # Prefer corridor_soft; fall back to mask.
                cand = maps.get("corridor_soft", lm.get("mask"))
                if isinstance(cand, np.ndarray):
                    inp_gate_map = np.asarray(cand, dtype=np.float32)
            except Exception:
                inp_gate_map = None
        debug = {
            "metrics": res.meta.get("metrics", {}),
            "fusion_weight": res.meta.get("fusion_weight", 0.0),
            "stage_diff_stats": res.meta.get("stage_diff_stats", {}),
            "inp_diag": res.meta.get("inp_diag", {}),
            "linear_meta": _jsonable(res.meta.get("linear_meta", {})),
            "svd_meta": _jsonable(res.meta.get("svd_meta", {})),
            "nonlinear_meta": _jsonable(res.meta.get("nonlinear_meta", {})),
            "quality_metrics": res.meta.get("quality_metrics", {}),
            "quality_assessment": _jsonable(res.meta.get("quality_assessment", {})),
            "backbone": backbone.load_diag,
        }
        if inp_gate_map is not None:
            debug["inp_gate_map"] = inp_gate_map
            debug["inp_gate_ratio"] = float((inp_gate_map > 0.02).mean())
        inp_rev = (((res.meta.get("nonlinear_meta", {}) or {}).get("load_diag", {}) or {}).get("inp_reverse_diag", {}))
        if isinstance(inp_rev, dict) and inp_rev:
            debug["inp_reverse_diag"] = _jsonable(inp_rev)
        return res.final, {
            "linear": res.linear,
            "svd_degraded": res.svd_degraded,
            "nonlinear": res.nonlinear,
            "final": res.final,
        }, debug

    def _process_adaptive_tensor(self, obs: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, object]]:
        fast = bool(getattr(self.args, "fast_adaptive", False))
        # Routing stage: global metrics + observation-first SVD forward probes only (no linear, no diffusion).
        t0 = time.perf_counter()
        self._status(f"Adaptive routing probe (fast={fast})...")
        plan = build_adaptive_plan(tensor_to_gray01(obs), self.processing_degree, fast_preview=fast)
        t_plan = time.perf_counter()
        sel = plan.get("selected_tasks", [])
        w = plan.get("global_weights", {}) or {}
        w_txt = ", ".join([f"{k}:{float(w.get(k, 0.0)):.3f}" for k in sel]) if sel else "none"
        self._status(f"Routing selected: {sel} | weights: {w_txt}")
        backbone = self._get_backbone()
        t_bb = time.perf_counter()
        result, debug = run_adaptive_chain(
            obs,
            self.processing_degree,
            nonlinear_solver=backbone.restore_from_svd,
            fast_preview=fast,
            plan=plan,
        )
        t_done = time.perf_counter()
        dbg = {
            **debug,
            "backbone": backbone.load_diag,
            "timing": {
                "routing_plan_s": float(t_plan - t0),
                "backbone_init_s": float(t_bb - t_plan),
                "adaptive_chain_s": float(t_done - t_bb),
                "total_s": float(t_done - t0),
            },
        }
        rm = dbg.get("routing_mode", "")
        dbg["routing_mode"] = f"{rm}_model_reverse" if rm else "model_reverse"
        return result.final, {
            "linear": result.linear,
            "svd_degraded": result.svd_degraded,
            "nonlinear": result.nonlinear,
            "final": result.final,
        }, dbg

    def _process_tensor(self, gray_u8: np.ndarray, preview: Optional[Dict[str, Any]] = None):
        obs = gray01_to_tensor(gray_u8.astype(np.float32) / 255.0, device=self.device)
        if self.deg == "adaptive":
            return self._process_adaptive_tensor(obs)
        return self._process_single_task(obs, self.deg, self.processing_degree, preview=preview)

    def _tensor_to_u8(self, t: torch.Tensor) -> np.ndarray:
        return (tensor_to_gray01(t) * 255.0).round().astype(np.uint8)

    def _save_adaptive_debug(self, tag: str, debug: Dict[str, object], color_hint: Optional[np.ndarray]):
        compact = {
            "supported_tasks": debug.get("supported_tasks", []),
            "selected_tasks": debug.get("selected_tasks", []),
            "selected_weights": debug.get("selected_weights", {}),
            "routing_softmax_weights": debug.get("routing_softmax_weights", {}),
            "global_weights": debug.get("global_weights", {}),
            "svd_scores": debug.get("svd_scores", {}),
            "svd_response": debug.get("svd_response", {}),
            "raw_scores": debug.get("raw_scores", {}),
            "task_probe": _jsonable(debug.get("task_probe", {})),
            "selection_reasons": debug.get("selection_reasons", {}),
            "preview_strength": debug.get("preview_strength", 0.0),
            "routing_mode": debug.get("routing_mode", ""),
            "metrics": debug.get("metrics", {}),
            "timing": _jsonable(debug.get("timing", {})),
            "quality_metrics": debug.get("quality_metrics", {}),
            "quality_assessment": _jsonable(debug.get("quality_assessment", {})),
            "restoration_diag": _jsonable(debug.get("restoration_diag", {})),
            "backbone": debug.get("backbone", {}),
        }
        (self.output_dir / f"routing_{tag}.json").write_text(json.dumps(_jsonable(compact), ensure_ascii=False, indent=2), encoding="utf-8")
        for task, res in (debug.get("task_results", {}) or {}).items():
            _save_stage(self.output_dir / f"adaptive_{tag}_{task}_linear.png", self._tensor_to_u8(res.linear), color_hint)
            _save_stage(self.output_dir / f"adaptive_{tag}_{task}_svd_degraded.png", self._tensor_to_u8(res.svd_degraded), color_hint)
            _save_stage(self.output_dir / f"adaptive_{tag}_{task}_nonlinear.png", self._tensor_to_u8(res.nonlinear), color_hint)
            _save_stage(self.output_dir / f"adaptive_{tag}_{task}_final.png", self._tensor_to_u8(res.final), color_hint)
            # Optional inp_em gate visualization
            if task == "inp_em":
                try:
                    lm = res.meta.get("linear_meta", {}) if isinstance(res.meta, dict) else {}
                    maps = (lm.get("maps") or {}) if isinstance(lm, dict) else {}
                    gate = maps.get("corridor_soft", lm.get("mask"))
                    if isinstance(gate, np.ndarray):
                        gate_u8 = (np.clip(gate.astype(np.float32), 0.0, 1.0) * 255.0).round().astype(np.uint8)
                        _save_stage(self.output_dir / f"inp_gate_{tag}.png", gate_u8, color_hint=None)
                except Exception:
                    pass

    def _gray_to_u8_stages(
        self,
        gray_u8: np.ndarray,
        color_hint: Optional[np.ndarray],
        preview: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, object], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        final_t, stages, debug = self._process_tensor(gray_u8, preview)
        linear = self._tensor_to_u8(stages["linear"])
        svd = self._tensor_to_u8(stages["svd_degraded"])
        nonlinear = self._tensor_to_u8(stages["nonlinear"])
        final = self._tensor_to_u8(final_t)
        return final_t, stages, debug, linear, svd, nonlinear, final

    def _finalize_main_bgr(self, final: np.ndarray, color_hint: Optional[np.ndarray]) -> np.ndarray:
        final_main = _gray_to_style(final, color_hint)
        if self.apply_result_light_enhance:
            final_main = finalize_em_output_uint8(final_main, self.deg, is_grayscale=(color_hint is None))
        return final_main

    def _process_path(self, path: Path, idx: int):
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Unable to load {path}")
        if img.ndim == 3 and img.shape[2] == 4:
            img = img[:, :, :3]
        gray_u8, color_hint = _to_gray_and_hint(img)
        preview: Optional[Dict[str, Any]] = None
        if self.deg == "isotropic_em" and self._iso_prev_gray is not None:
            preview = {
                "prev_gray": self._iso_prev_gray,
                "prev_linear": self._iso_prev_linear,
                "prev_svd": self._iso_prev_svd,
                "prev_slice_align": self._iso_prev_slice_align,
            }
        final_t, stages, debug, linear, svd, nonlinear, final = self._gray_to_u8_stages(gray_u8, color_hint, preview=preview)
        if self.deg == "isotropic_em":
            self._iso_prev_gray = np.clip(gray_u8.astype(np.float32) / 255.0, 0.0, 1.0)
            self._iso_prev_linear = tensor_to_gray01(stages["linear"]).astype(np.float32)
            self._iso_prev_svd = tensor_to_gray01(stages["svd_degraded"]).astype(np.float32)
            lm = debug.get("linear_meta") if isinstance(debug, dict) else None
            sa = lm.get("slice_align") if isinstance(lm, dict) else None
            self._iso_prev_slice_align = dict(sa) if isinstance(sa, dict) else None
        tag = str(idx)
        _save_stage(self.output_dir / f"linear_{idx}.png", linear, color_hint)
        _save_stage(self.output_dir / f"svd_degraded_{idx}.png", svd, color_hint)
        _save_stage(self.output_dir / f"nonlinear_{idx}.png", nonlinear, color_hint)
        _save_stage(self.output_dir / f"final_{idx}.png", final, color_hint)
        final_main = self._finalize_main_bgr(final, color_hint)
        cv2.imwrite(str(self.output_dir / f"{path.stem}_-1.png"), final_main)
        if self.deg == "adaptive":
            self._save_adaptive_debug(tag, debug, color_hint)
        else:
            # Optional inp_em gate visualization
            try:
                if self.deg == "inp_em" and isinstance(debug, dict):
                    gate = debug.get("inp_gate_map")
                    if isinstance(gate, np.ndarray):
                        gate_u8 = (np.clip(gate.astype(np.float32), 0.0, 1.0) * 255.0).round().astype(np.uint8)
                        _save_stage(self.output_dir / f"inp_gate_{tag}.png", gate_u8, color_hint=None)
            except Exception:
                pass
            (self.output_dir / f"debug_{tag}.json").write_text(json.dumps(_jsonable(debug), ensure_ascii=False, indent=2), encoding="utf-8")
        return _ProcessedSample(path.stem, linear, svd, nonlinear, final, debug)

    def sample_z_stack(self, stack_z_hw: np.ndarray, stem: str) -> List[_ProcessedSample]:
        """
        Process each Z slice with the same 2D pipeline, stack linear/svd/nonlinear/final
        into multi-page TIFFs, and write a middle-slice PNG for Qt preview.
        """
        self._iso_prev_gray = None
        self._iso_prev_linear = None
        self._iso_prev_svd = None
        self._iso_prev_slice_align = None
        vol = np.asarray(stack_z_hw)
        if vol.ndim != 3:
            raise ValueError(f"sample_z_stack expects (Z,H,W) uint8, got {vol.shape}")
        if vol.dtype != np.uint8:
            vol = np.clip(vol, 0, 255).astype(np.uint8)
        z_depth = int(vol.shape[0])
        mid = z_depth // 2
        lin_l: List[np.ndarray] = []
        svd_l: List[np.ndarray] = []
        non_l: List[np.ndarray] = []
        fin_gray_l: List[np.ndarray] = []
        fin_main_l: List[np.ndarray] = []
        last_debug: Dict[str, object] = {}
        prev_gray: Optional[np.ndarray] = None
        prev_lin: Optional[np.ndarray] = None
        prev_svd: Optional[np.ndarray] = None
        prev_slice_align: Optional[Dict[str, Any]] = None
        for z in range(z_depth):
            self._status(f"Z-stack slice {z + 1}/{z_depth} ...")
            gray = vol[z]
            preview: Optional[Dict[str, Any]] = None
            if self.deg == "isotropic_em" and prev_gray is not None:
                preview = {
                    "prev_gray": prev_gray,
                    "prev_linear": prev_lin,
                    "prev_svd": prev_svd,
                    "prev_slice_align": prev_slice_align,
                }
            _final_t, _stages, debug, linear, svd, nonlinear, final = self._gray_to_u8_stages(gray, None, preview=preview)
            if self.deg == "isotropic_em":
                prev_gray = np.clip(gray.astype(np.float32) / 255.0, 0.0, 1.0)
                prev_lin = tensor_to_gray01(_stages["linear"]).astype(np.float32)
                prev_svd = tensor_to_gray01(_stages["svd_degraded"]).astype(np.float32)
                lm = debug.get("linear_meta") if isinstance(debug, dict) else None
                sa = lm.get("slice_align") if isinstance(lm, dict) else None
                prev_slice_align = dict(sa) if isinstance(sa, dict) else None
            final_main = self._finalize_main_bgr(final, None)
            lin_l.append(linear)
            svd_l.append(svd)
            non_l.append(nonlinear)
            fin_gray_l.append(final)
            fin_main_l.append(np.asarray(final_main))
            if z == mid:
                last_debug = debug
            vol_cb = getattr(self.args, "volume_progress", None)
            if callable(vol_cb):
                try:
                    vol_cb(z + 1, z_depth)
                except Exception:
                    pass
            if self.deg == "adaptive" and z == mid:
                self._save_adaptive_debug("vol_mid", debug, None)
            elif self.deg != "adaptive" and z == mid:
                try:
                    (self.output_dir / "debug_vol_mid.json").write_text(
                        json.dumps(_jsonable(debug), ensure_ascii=False, indent=2), encoding="utf-8"
                    )
                    if self.deg == "inp_em" and isinstance(debug, dict):
                        gate = debug.get("inp_gate_map")
                        if isinstance(gate, np.ndarray):
                            gate_u8 = (np.clip(gate.astype(np.float32), 0.0, 1.0) * 255.0).round().astype(np.uint8)
                            _save_stage(self.output_dir / "inp_gate_vol_mid.png", gate_u8, color_hint=None)
                except Exception:
                    pass
        vol_lin = np.stack(lin_l, axis=0)
        vol_svd = np.stack(svd_l, axis=0)
        vol_non = np.stack(non_l, axis=0)
        vol_fin_g = np.stack(fin_gray_l, axis=0)
        vol_fin_m = np.stack(fin_main_l, axis=0)
        save_z_stack_tiff(str(self.output_dir / f"{stem}_linear.tif"), vol_lin)
        save_z_stack_tiff(str(self.output_dir / f"{stem}_svd_degraded.tif"), vol_svd)
        save_z_stack_tiff(str(self.output_dir / f"{stem}_nonlinear.tif"), vol_non)
        save_z_stack_tiff(str(self.output_dir / f"{stem}_final.tif"), vol_fin_g)
        save_z_stack_tiff(str(self.output_dir / f"{stem}_-1.tif"), vol_fin_m)
        mid_main = fin_main_l[mid]
        cv2.imwrite(str(self.output_dir / f"{stem}_-1_mid.png"), mid_main)
        return [
            _ProcessedSample(
                stem,
                lin_l[mid],
                svd_l[mid],
                non_l[mid],
                fin_gray_l[mid],
                last_debug,
            )
        ]

    def sample(self):
        # Clears isotropic_em temporal state; folder order follows _read_list_file (natural sort).
        self._iso_prev_gray = None
        self._iso_prev_linear = None
        self._iso_prev_svd = None
        self._iso_prev_slice_align = None
        paths = _read_list_file(self.list_file, self.root)
        if not paths:
            raise ValueError(f"No images found in {self.root}")
        out: List[_ProcessedSample] = []
        for i, p in enumerate(paths):
            self._status(f"Processing {i+1}/{len(paths)}: {p.name} ...")
            out.append(self._process_path(p, i))
        return out
