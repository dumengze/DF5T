import sys
from pathlib import Path
import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.diffusion import build_adaptive_fused_h_and_y0, finalize_em_output_uint8
from tools.EMSVD import adaptive_fused_linear_restore, direct_em_physics_restore


OUT = ROOT / "outputs" / "adaptive_preview"
OUT.mkdir(parents=True, exist_ok=True)


def repo_sample(limit: int = 1):
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    for p in ROOT.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in exts:
            continue
        if "outputs" in p.parts or ".venv" in p.parts:
            continue
        g = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if g is None or g.size == 0:
            continue
        g = cv2.resize(g, (384, 384), interpolation=cv2.INTER_AREA)
        return p.stem, g
    # fallback synth
    h = w = 384
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    base = 148 + 14 * np.sin(x / 19.0) + 11 * np.cos(y / 23.0)
    for k in range(10):
        yy = 35 + k * 34 + 4 * np.sin((x / 31.0) + k)
        stripe = np.exp(-((y - yy) ** 2) / (2 * (1.6 + 0.2 * (k % 3)) ** 2))
        base -= (38 + 6 * (k % 2)) * stripe
    g = np.clip(base + np.random.default_rng(2026).normal(0, 14, (h, w)), 0, 255).astype(np.uint8)
    return "synth", g


def gray_to_tensor(gray: np.ndarray) -> torch.Tensor:
    t = torch.from_numpy(gray.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
    return t * 2.0 - 1.0


def tensor_to_gray(t: torch.Tensor) -> np.ndarray:
    x = t.detach().cpu().clamp(-1.0, 1.0)
    x = ((x + 1.0) * 127.5).round().to(torch.uint8)
    return x[0, 0].numpy()


def save_map_u8(path: Path, m: torch.Tensor):
    mm = m.detach().cpu()
    if mm.dim() == 4:
        mm = mm[0, 0]
    arr = mm.numpy().astype(np.float32)
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-6)
    cv2.imwrite(str(path), (arr * 255.0).astype(np.uint8))


def main():
    name, gray = repo_sample()
    patch = gray_to_tensor(gray)
    # minimal args namespace
    class Args:
        seed = 1234
        deblur_defocus_base = -4500.0
        deblur_bfactor_base = 8.0
        deblur_alpha = 4e-5
        deblur_pinv_max_amplify = 60.0
        deblur_pixel_size = 0.7
    args = Args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    patch = patch.to(device)
    processing_degree = 0.85

    H_funcs, y_0, _sigma = build_adaptive_fused_h_and_y0(
        patch, channels=1, patch_h=patch.shape[2], patch_w=patch.shape[3],
        device=device, processing_degree=processing_degree, args=args,
        patch_idx=0, input_is_gray=True, is_grayscale=True,
    )
    x_lin = adaptive_fused_linear_restore(H_funcs, y_0, patch)
    x_final = direct_em_physics_restore(H_funcs, y_0, patch, processing_degree=processing_degree, task_name="adaptive")
    dbg = getattr(H_funcs, "adaptive_debug", {}) or {}

    orig = gray
    lin = tensor_to_gray(x_lin)
    fin = tensor_to_gray(x_final)
    non = None
    if isinstance(dbg, dict) and torch.is_tensor(dbg.get("x_non_stage", None)):
        non = tensor_to_gray(dbg["x_non_stage"])
    fin_u8 = finalize_em_output_uint8(cv2.cvtColor(fin, cv2.COLOR_GRAY2BGR), "adaptive", is_grayscale=True)
    fin = cv2.cvtColor(fin_u8, cv2.COLOR_BGR2GRAY)

    cv2.imwrite(str(OUT / f"{name}_input.png"), orig)
    cv2.imwrite(str(OUT / f"{name}_fused_linear.png"), lin)
    if non is not None:
        cv2.imwrite(str(OUT / f"{name}_nonlinear.png"), non)
    cv2.imwrite(str(OUT / f"{name}_final.png"), fin)

    # weights and residual vis
    if "w_maps" in dbg:
        for t, wm in dbg["w_maps"].items():
            save_map_u8(OUT / f"{name}_wmap_{t}.png", wm)
    if "r_non_fused" in dbg and torch.is_tensor(dbg["r_non_fused"]):
        r = dbg["r_non_fused"].detach().cpu()
        r = r[0].mean(dim=0).numpy()
        rv = np.clip((r / (np.percentile(np.abs(r), 99.0) + 1e-6)) * 0.5 + 0.5, 0.0, 1.0)
        cv2.imwrite(str(OUT / f"{name}_nonlinear_residual_vis.png"), (rv * 255.0).astype(np.uint8))

    # write routing summary
    sel = dbg.get("selected_tasks", [])
    mode = dbg.get("routing_mode", "")
    gw = dbg.get("global_weights", {})
    lam = dbg.get("lambda_non", None)
    rdiag = dbg.get("restoration_diag", None)
    (OUT / f"{name}_routing.txt").write_text(
        "mode=" + str(mode) + "\n" +
        "selected=" + str(sel) + "\n" +
        "global_weights=" + str(gw) + "\n" +
        "lambda_non=" + str(lam) + "\n" +
        "restoration_diag=" + str(rdiag) + "\n",
        encoding="utf-8",
    )
    print("saved", OUT)


if __name__ == "__main__":
    main()

