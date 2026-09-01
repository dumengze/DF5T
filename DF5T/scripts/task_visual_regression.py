import os
from pathlib import Path
import numpy as np
import cv2
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
OUT_DIR = ROOT / "outputs" / "task_regression"
OUT_DIR.mkdir(parents=True, exist_ok=True)

from tools.diffusion import linear_nonlinear_joint_restore, finalize_em_output_uint8


def find_repo_images():
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    files = []
    for p in ROOT.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts and "outputs" not in p.parts and ".venv" not in p.parts:
            files.append(p)
    return files


def synth_images():
    imgs = []
    for idx in range(3):
        h, w = 384, 384
        y, x = np.mgrid[0:h, 0:w].astype(np.float32)
        base = 148 + 16 * np.sin(x / 19.0) + 13 * np.cos(y / 23.0)
        for k in range(8):
            yy = 40 + k * 38 + 5 * np.sin((x / 33.0) + k)
            stripe = np.exp(-((y - yy) ** 2) / (2 * (1.8 + 0.2 * (k % 3)) ** 2))
            base -= (45 + 8 * (k % 2)) * stripe
        noise = np.random.default_rng(1234 + idx).normal(0, 18 + 4 * idx, size=(h, w))
        salt = (np.random.default_rng(777 + idx).random((h, w)) > 0.996).astype(np.float32) * 80.0
        img = np.clip(base + noise + salt, 0, 255).astype(np.uint8)
        imgs.append((f"synth_{idx+1}", img))
    return imgs


def to_bgr(gray):
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def gray_to_tensor(gray: np.ndarray) -> torch.Tensor:
    t = torch.from_numpy(gray.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
    return t * 2.0 - 1.0


def tensor_to_gray(t: torch.Tensor) -> np.ndarray:
    x = t.detach().cpu().clamp(-1.0, 1.0)
    x = ((x + 1.0) * 127.5).round().to(torch.uint8)
    return x[0, 0].numpy()


def make_prior(obs_gray: np.ndarray, task: str) -> np.ndarray:
    g = obs_gray.astype(np.float32)
    rng = np.random.default_rng(2026)
    if task == "deno_em":
        return np.clip(g + rng.normal(0, 12.0, g.shape), 0, 255).astype(np.uint8)
    if task == "inp_em":
        inv = 255 - obs_gray
        closed = cv2.morphologyEx(inv, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)), iterations=1)
        return np.clip(255 - closed, 0, 255).astype(np.uint8)
    if task == "isotropic_em":
        gx = cv2.sepFilter2D(g, cv2.CV_32F, np.array([1, 2, 1], np.float32) / 4.0, np.array([1], np.float32))
        return np.clip(gx, 0, 255).astype(np.uint8)
    if task.startswith("sr"):
        return np.clip(g + 0.25 * (g - cv2.GaussianBlur(g, (0, 0), 1.3)), 0, 255).astype(np.uint8)
    if task == "adaptive":
        den = cv2.GaussianBlur(g, (0, 0), 1.2)
        shp = g + 0.22 * (g - cv2.GaussianBlur(g, (0, 0), 1.0))
        return np.clip(0.5 * den + 0.5 * shp, 0, 255).astype(np.uint8)
    if task == "deblur_em":
        blur = cv2.GaussianBlur(g, (0, 0), 1.6)
        return np.clip(blur, 0, 255).astype(np.uint8)
    return obs_gray.copy()


def task_apply(gray: np.ndarray, task: str):
    obs = gray_to_tensor(gray)
    prior_gray = make_prior(gray, task)
    x_prior = gray_to_tensor(prior_gray)
    with torch.no_grad():
        out = linear_nonlinear_joint_restore(x_prior, None, obs, deg=task, u_map=None)
    out_gray = tensor_to_gray(out)
    out_bgr = cv2.cvtColor(out_gray, cv2.COLOR_GRAY2BGR)
    out_final = finalize_em_output_uint8(out_bgr, task, is_grayscale=True)
    return cv2.cvtColor(out_final, cv2.COLOR_BGR2GRAY)


def mad(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32))))


def run():
    repo_imgs = find_repo_images()
    samples = []
    if repo_imgs:
        for p in repo_imgs[:5]:
            im = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if im is not None and im.size > 0:
                samples.append((p.stem, im))
    else:
        samples = synth_images()

    tasks = ["deno_em", "deblur_em", "inp_em", "isotropic_em", "sr2", "adaptive"]
    rows = []
    stats = []
    for name, gray in samples:
        gray = cv2.resize(gray, (384, 384), interpolation=cv2.INTER_AREA)
        row = [to_bgr(gray)]
        sample_stats = [f"sample={name}"]
        for t in tasks:
            out = task_apply(gray, t)
            row.append(to_bgr(out))
            m = mad(out, gray)
            sample_stats.append(f"{t}_mad={m:.2f}")
            cv2.imwrite(str(OUT_DIR / f"{name}_{t}.png"), out)
        rows.append(row)
        stats.append(", ".join(sample_stats))

    # Build contact sheet
    margin = 8
    h, w = rows[0][0].shape[:2]
    cols = 1 + len(tasks)
    sheet = np.full((len(rows) * (h + margin) + margin, cols * (w + margin) + margin, 3), 235, np.uint8)
    for r, row in enumerate(rows):
        for c, img in enumerate(row):
            y0 = margin + r * (h + margin)
            x0 = margin + c * (w + margin)
            sheet[y0:y0 + h, x0:x0 + w] = img
    cv2.imwrite(str(OUT_DIR / "comparison_sheet.png"), sheet)

    with open(OUT_DIR / "stats.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(stats))

    print("Saved:", OUT_DIR)
    print("\n".join(stats))


if __name__ == "__main__":
    run()
