"""Fusion of linear vs nonlinear branches."""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from tools.em_maps import build_inp_mask, gauss
from tools.em_task_spec import fusion_caps
from tools.em_tensor import EPS, clip01


def organic_fuse(
    obs: np.ndarray,
    linear: np.ndarray,
    nonlinear: np.ndarray,
    task: str,
    strength: float,
    meta: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, float, np.ndarray]:
    meta = meta or {}
    support = meta.get("support")
    if support is None:
        from tools.em_maps import membrane_map

        support = membrane_map(linear)
    support = clip01(np.asarray(support, dtype=np.float32))
    agreement = clip01(1.0 - np.abs(linear - nonlinear) / (0.11 + 0.19 * strength))
    lin_cap, nl_boost = fusion_caps(task)

    if task == "deno_em":
        w = clip01((0.22 + 0.40 * (1.0 - meta.get("noise", 0.5)) + 0.20 * agreement + 0.18 * support) * nl_boost)
        w = np.minimum(w, lin_cap)
        fused = (1.0 - w) * linear + w * nonlinear
        fused = 0.93 * fused + 0.07 * obs
    elif task == "deblur_em":
        w = clip01((0.42 + 0.28 * support + 0.18 * agreement + 0.12 * meta.get("blur", 0.5)) * nl_boost)
        w = np.minimum(w, lin_cap)
        fused = (1.0 - w) * linear + w * nonlinear
    elif task.startswith("sr"):
        w = clip01((0.50 + 0.24 * support + 0.20 * meta.get("edge", 0.5) + 0.12 * agreement) * nl_boost)
        w = np.minimum(w, lin_cap)
        fused = (1.0 - w) * linear + w * nonlinear
    elif task == "inp_em":
        # inp_em (rewrite): gate-driven fusion that preserves strong nonlinear generation inside corridor core.
        mask = meta.get("mask")
        if not isinstance(mask, np.ndarray) or mask.shape != obs.shape:
            mask, _ = build_inp_mask(obs)
        mask = clip01(np.asarray(mask, dtype=np.float32))
        conf = meta.get("hole_confidence")
        conf = clip01(np.asarray(conf, dtype=np.float32)) if isinstance(conf, np.ndarray) and conf.shape == mask.shape else np.ones_like(mask, dtype=np.float32)

        lp = meta.get("lumen_protect")
        lp_f = clip01(np.asarray(lp, dtype=np.float32)) if isinstance(lp, np.ndarray) and lp.shape == obs.shape else np.zeros_like(mask, dtype=np.float32)

        gen_gate = meta.get("gen_gate")
        blend_ring = meta.get("blend_ring")
        if isinstance(gen_gate, np.ndarray) and gen_gate.shape == obs.shape:
            core = clip01(np.asarray(gen_gate, dtype=np.float32))
        else:
            core = clip01(mask * (0.60 + 0.40 * conf))
        if isinstance(blend_ring, np.ndarray) and blend_ring.shape == obs.shape:
            ring = clip01(np.asarray(blend_ring, dtype=np.float32))
        else:
            ring = clip01(np.maximum(0.0, gauss(core, 1.10) - 0.88 * core))

        core = clip01(core * (1.0 - 0.92 * lp_f))
        ring = clip01(ring * (1.0 - 0.92 * lp_f))

        # Primary fusion weights: keep nonlinear almost fully inside core, partial in ring.
        w = clip01((1.00 * core + 0.75 * ring) * nl_boost)
        w = np.minimum(w, 0.98)  # allow near-full nonlinear in corridor core

        # Keep outside conservative: small nonlinear only if it agrees with linear.
        agreement = clip01(1.0 - np.abs(linear - nonlinear) / 0.14)
        w_out = clip01(0.03 * agreement * (1.0 - core) * (1.0 - ring))
        w = clip01(w + w_out)

        # Bridge target pull (optional): helps keep generated membrane on the predicted path.
        bridge_target = meta.get("bridge_target")
        if isinstance(bridge_target, np.ndarray) and bridge_target.shape == obs.shape:
            bt = clip01(np.asarray(bridge_target, dtype=np.float32))
            nonlinear = clip01(nonlinear + 0.35 * core * (bt - nonlinear))

        fused = linear * (1.0 - w) + nonlinear * w
        # Hard lumen protection.
        fused = clip01(fused * (1.0 - 0.94 * lp_f) + linear * (0.94 * lp_f))
    else:
        anis = meta.get("anisotropy", 0.5)
        w = clip01((0.36 + 0.30 * support + 0.20 * anis + 0.14 * agreement) * nl_boost)
        w = np.minimum(w, lin_cap)
        fused = (1.0 - w) * linear + w * nonlinear

    return np.clip(fused, 0.0, 1.0), float(np.mean(w)), w.astype(np.float32)
