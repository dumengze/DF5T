import cv2
import numpy as np
import torch
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.diffusion import linear_nonlinear_joint_restore_with_stages, finalize_em_output_uint8

OUT = ROOT / "outputs" / "phase1_preview"
OUT.mkdir(parents=True, exist_ok=True)


def repo_samples(limit: int = 2):
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    items = []
    for p in ROOT.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in exts:
            continue
        if "outputs" in p.parts or ".venv" in p.parts:
            continue
        im = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if im is None or im.size == 0:
            continue
        items.append((p.stem, cv2.resize(im, (384, 384), interpolation=cv2.INTER_AREA)))
        if len(items) >= limit:
            break
    return items


def synth_samples():
    imgs = []
    for idx in range(2):
        h = w = 384
        y, x = np.mgrid[0:h, 0:w].astype(np.float32)
        base = 145 + 15 * np.sin(x / 19.0) + 10 * np.cos(y / 25.0)
        for k in range(9):
            yy = 35 + k * 37 + 4.5 * np.sin((x / 29.0) + k)
            stripe = np.exp(-((y - yy) ** 2) / (2 * (1.6 + 0.2 * (k % 3)) ** 2))
            base -= (40 + 7 * (k % 2)) * stripe
        noise = np.random.default_rng(2026 + idx).normal(0, 13 + 2 * idx, (h, w))
        img = np.clip(base + noise, 0, 255).astype(np.uint8)
        imgs.append((f"synth_{idx+1}", img))
    return imgs


def gray_to_tensor(gray: np.ndarray) -> torch.Tensor:
    t = torch.from_numpy(gray.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
    return t * 2.0 - 1.0


def tensor_to_gray(t: torch.Tensor) -> np.ndarray:
    x = t.detach().cpu().clamp(-1.0, 1.0)
    x = ((x + 1.0) * 127.5).round().to(torch.uint8)
    return x[0, 0].numpy()


def make_prior(gray: np.ndarray, task: str) -> np.ndarray:
    g = gray.astype(np.float32)
    if task == "deblur_em":
        return np.clip(cv2.GaussianBlur(g, (0, 0), 1.9), 0, 255).astype(np.uint8)
    if task == "deno_em":
        n = np.random.default_rng(3030).normal(0, 8.0, gray.shape)
        return np.clip(g + n, 0, 255).astype(np.uint8)
    if task == "isotropic_em":
        gx = cv2.sepFilter2D(g, cv2.CV_32F, np.array([1, 2, 1], np.float32) / 4.0, np.array([1], np.float32))
        return np.clip(0.8 * g + 0.2 * gx, 0, 255).astype(np.uint8)
    if task.startswith("sr"):
        ds = cv2.resize(gray, (gray.shape[1] // 2, gray.shape[0] // 2), interpolation=cv2.INTER_AREA)
        up = cv2.resize(ds, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_LINEAR)
        return up.astype(np.uint8)
    inv = 255 - gray
    closed = cv2.morphologyEx(inv, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)), iterations=1)
    return np.clip(255 - closed, 0, 255).astype(np.uint8)


def run_task(gray: np.ndarray, task: str):
    obs = gray_to_tensor(gray)
    prior = gray_to_tensor(make_prior(gray, task))
    with torch.no_grad():
        out, stages = linear_nonlinear_joint_restore_with_stages(prior, None, obs, deg=task, u_map=None)
    lin = tensor_to_gray(stages["linear"])
    non = tensor_to_gray(stages["nonlinear"])
    fin = tensor_to_gray(out)
    fin_u8 = finalize_em_output_uint8(cv2.cvtColor(fin, cv2.COLOR_GRAY2BGR), task, is_grayscale=True)
    fin = cv2.cvtColor(fin_u8, cv2.COLOR_BGR2GRAY)
    return lin, non, fin


def main():
    samples = repo_samples(limit=2)
    if not samples:
        samples = synth_samples()
    tasks = ["deblur_em", "inp_em", "deno_em", "isotropic_em", "sr2"]
    rows = []
    lines = []
    for name, gray in samples:
        for task in tasks:
            lin, non, fin = run_task(gray, task)
            cv2.imwrite(str(OUT / f"{name}_{task}_orig.png"), gray)
            cv2.imwrite(str(OUT / f"{name}_{task}_linear.png"), lin)
            cv2.imwrite(str(OUT / f"{name}_{task}_nonlinear.png"), non)
            cv2.imwrite(str(OUT / f"{name}_{task}_final.png"), fin)
            quad = np.concatenate([
                cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR),
                cv2.cvtColor(lin, cv2.COLOR_GRAY2BGR),
                cv2.cvtColor(non, cv2.COLOR_GRAY2BGR),
                cv2.cvtColor(fin, cv2.COLOR_GRAY2BGR),
            ], axis=1)
            cv2.imwrite(str(OUT / f"{name}_{task}_quad.png"), quad)
            rows.append(quad)
            m_lin = float(np.mean(np.abs(lin.astype(np.float32) - gray.astype(np.float32))))
            m_fin = float(np.mean(np.abs(fin.astype(np.float32) - gray.astype(np.float32))))
            lines.append(f"{name},{task},lin_mad={m_lin:.3f},final_mad={m_fin:.3f}")
    if rows:
        sheet = np.concatenate(rows, axis=0)
        cv2.imwrite(str(OUT / "phase1_contact_sheet.png"), sheet)
    (OUT / "phase1_stats.txt").write_text("\n".join(lines), encoding="utf-8")
    print("saved", OUT)


if __name__ == "__main__":
    main()
