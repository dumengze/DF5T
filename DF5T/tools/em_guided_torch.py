"""Diffusion reverse-step guidance; keeps task-specific behavior in one module."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from tools.em_task_spec import reverse_start_base


def _shift2d(x: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
    y = torch.roll(x, shifts=(dy, dx), dims=(2, 3))
    if dy > 0:
        y[:, :, :dy, :] = y[:, :, dy:dy + 1, :]
    elif dy < 0:
        y[:, :, dy:, :] = y[:, :, dy - 1:dy, :]
    if dx > 0:
        y[:, :, :, :dx] = y[:, :, :, dx:dx + 1]
    elif dx < 0:
        y[:, :, :, dx:] = y[:, :, :, dx - 1:dx]
    return y


def em_guided_prediction(
    pred_xstart: torch.Tensor,
    x_obs: torch.Tensor,
    x_linear: torch.Tensor,
    x_svd: torch.Tensor,
    guide_map: torch.Tensor,
    hole_map: torch.Tensor,
    anis_map: torch.Tensor,
    task: str,
    strength: float,
    step_idx: int,
    start_idx: int,
    weak_vertical: bool,
    hole_conf: torch.Tensor | None = None,
    membrane_guide: torch.Tensor | None = None,
    line_closure: torch.Tensor | None = None,
    lumen_protect: torch.Tensor | None = None,
    break_saliency: torch.Tensor | None = None,
    bridge_target: torch.Tensor | None = None,
) -> torch.Tensor:
    progress = 1.0 - float(step_idx) / max(float(start_idx), 1.0)
    g = guide_map.repeat(1, 3, 1, 1)
    hole = hole_map.repeat(1, 3, 1, 1)
    anis = anis_map.repeat(1, 3, 1, 1)
    detail = x_linear - x_svd
    obs_residual = x_obs - x_svd
    base_anchor = 0.56 * x_svd + 0.30 * x_linear + 0.14 * x_obs
    s = float(max(0.0, min(1.55, strength)))

    if task == "deno_em":
        lam_anchor = 0.16 + 0.14 * (1.0 - progress)
        lam_detail = (0.05 + 0.12 * s) * progress
        pred = pred_xstart * (1.0 - lam_anchor) + base_anchor * lam_anchor
        pred = pred + lam_detail * g * detail + 0.05 * (1.0 - g) * obs_residual
        pred = 0.95 * pred + 0.05 * F.avg_pool2d(pred, kernel_size=3, stride=1, padding=1)
    elif task == "deblur_em":
        lam_anchor = 0.08 + 0.10 * (1.0 - progress)
        lam_detail = (0.18 + 0.16 * s) * (0.30 + 0.70 * progress)
        pred = pred_xstart * (1.0 - lam_anchor) + base_anchor * lam_anchor
        pred = pred + lam_detail * g * detail + 0.07 * g * obs_residual
    elif task.startswith("sr"):
        lam_anchor = 0.06 + 0.08 * (1.0 - progress)
        lam_detail = (0.22 + 0.20 * s) * (0.28 + 0.72 * progress)
        pred = pred_xstart * (1.0 - lam_anchor) + base_anchor * lam_anchor
        pred = pred + lam_detail * g * detail + 0.08 * g * obs_residual
    elif task == "inp_em":
        # inp_em (rewrite): two-level gate + strong corridor generation + outside hard anchor.
        # Goal: produce visible membrane continuity in a narrow break corridor while keeping everything else unchanged.
        conf = hole if hole_conf is None else hole_conf.clamp(0.0, 1.0)
        hole_u = hole.clamp(0.0, 1.0)
        lp = lumen_protect.clamp(0.0, 1.0) if lumen_protect is not None else None
        mem = membrane_guide.clamp(0.0, 1.0) if membrane_guide is not None else g

        # Base corridor from hole/conf + break detectors (if present).
        core = (hole_u * (0.62 + 0.38 * conf)).clamp(0.0, 1.0)
        if line_closure is not None:
            core = torch.maximum(core, 0.88 * line_closure.clamp(0.0, 1.0))
        if break_saliency is not None:
            core = torch.maximum(core, 0.78 * break_saliency.clamp(0.0, 1.0))
        core = (core * (0.35 + 0.65 * mem)).clamp(0.0, 1.0)

        # Two-level gates: gen_gate (core) and blend_ring (transition).
        soft = F.avg_pool2d(core, kernel_size=7, stride=1, padding=3).clamp(0.0, 1.0)
        gen_gate = torch.maximum(core, 0.55 * soft).clamp(0.0, 1.0)
        # sharpen gen_gate
        gen_gate = (gen_gate * (0.55 + 0.45 * gen_gate)).clamp(0.0, 1.0)
        ring = (F.avg_pool2d(gen_gate, kernel_size=9, stride=1, padding=4) - gen_gate).clamp(0.0, 1.0)
        blend_ring = (0.35 + 0.30 * s) * ring
        blend_ring = blend_ring.clamp(0.0, 1.0)
        # Keep gate narrow: favor length extension along trajectory over width increase.
        gen_gate = F.avg_pool2d(gen_gate, kernel_size=3, stride=1, padding=1).clamp(0.0, 1.0)

        if lp is not None:
            gen_gate = gen_gate * (1.0 - 0.92 * lp)
            blend_ring = blend_ring * (1.0 - 0.92 * lp)

        outside = x_linear  # corridor_only policy: outside stays linear
        if bridge_target is None:
            bridge_target = 0.70 * x_linear + 0.30 * x_svd

        # Membrane-line prior (differentiable): emphasize dark ridges inside the corridor.
        # pred_xstart is [-1,1]; convert to [0,1] approx.
        px = (pred_xstart + 1.0) * 0.5
        k1 = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], device=px.device, dtype=px.dtype).view(1, 1, 3, 3)
        k2 = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], device=px.device, dtype=px.dtype).view(1, 1, 3, 3)
        gx = F.conv2d(px[:, :1], k1, padding=1)
        gy = F.conv2d(px[:, :1], k2, padding=1)
        edge = torch.sqrt(gx * gx + gy * gy + 1e-6).clamp(0.0, 3.0)
        # Encourage a crisp ridge line: high edge + low intensity along the target path.
        ridge_prior = (0.65 * edge + 0.35 * (1.0 - px[:, :1])).clamp(0.0, 1.0)

        # Corridor anchor: pull toward bridge_target inside gen_gate, allow model freedom to create structure.
        bridge_anchor = (0.82 * bridge_target + 0.18 * x_linear).clamp(-1.0, 1.0)
        pred = outside * (1.0 - gen_gate) + pred_xstart * gen_gate

        # Strong early generation, then converge.
        early = float(step_idx) / max(float(start_idx), 1.0)
        gen_strength = (0.80 + 1.00 * s) * (0.65 + 0.35 * early)
        pred = pred + gen_strength * gen_gate * (bridge_anchor - pred)

        # Thin-line encouragement inside corridor: sharpen + darken along ridge prior.
        line_push = (0.28 + 0.82 * s) * (0.40 + 0.60 * early)
        pred = pred + line_push * gen_gate * (ridge_prior.repeat(1, 3, 1, 1) - 0.52) * 0.45
        # Local non-maximum suppression style pull to keep centerline thin (length > width).
        n1 = torch.minimum(_shift2d(pred, -1, 0), _shift2d(pred, 1, 0))
        n2 = torch.minimum(_shift2d(pred, 0, -1), _shift2d(pred, 0, 1))
        thin_target = torch.minimum(n1, n2)
        pred = pred + (0.20 + 0.20 * s) * gen_gate * (thin_target - pred)

        # Transition ring: partially follow anchor, but keep continuity.
        ring_anchor = (0.90 * x_linear + 0.10 * bridge_anchor).clamp(-1.0, 1.0)
        pred = pred * (1.0 - blend_ring) + ring_anchor * blend_ring

        # Hard lumen protection.
        if lp is not None:
            pred = pred * (1.0 - 0.96 * lp) + x_linear * (0.96 * lp)
    else:
        lam_anchor = 0.10 + 0.10 * (1.0 - progress)
        lam_detail = (0.15 + 0.14 * s) * (0.32 + 0.68 * progress)
        pred = pred_xstart * (1.0 - lam_anchor) + base_anchor * lam_anchor
        pred = pred + lam_detail * (0.62 * anis + 0.38 * g) * detail
        if weak_vertical:
            pred[:, :, :, 1:-1] = 0.91 * pred[:, :, :, 1:-1] + 0.09 * pred[:, :, :, :-2]
        else:
            pred[:, :, 1:-1, :] = 0.91 * pred[:, :, 1:-1, :] + 0.09 * pred[:, :, :-2, :]

    return pred.clamp(-1.0, 1.0)


def diffusion_start_ratio(task: str, strength: float, tail: float, residual_energy: float, num_timesteps: int) -> int:
    base = reverse_start_base(task)
    s = float(max(0.0, min(1.55, strength)))
    if task == "inp_em":
        # Start from a noisier point to unlock stronger nonlinear completion for broken membranes.
        ratio = base + 0.28 * s + 0.24 * float(tail) + 0.12 * float(residual_energy)
        ratio = float(max(0.34, min(0.94, ratio)))
    else:
        ratio = base + 0.18 * s + 0.22 * float(tail) + 0.10 * float(residual_energy)
        ratio = float(max(0.20, min(0.92, ratio)))
    return int(max(2, min(round(ratio * (num_timesteps - 1)), num_timesteps - 1)))
