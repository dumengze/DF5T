import os
import logging
from typing import List, Tuple, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def _clip01(x):
    return np.clip(x, 0.0, 1.0)


def _small_image_factor_hw(h: int, w: int) -> float:
    short = float(max(1, min(h, w)))
    if short >= 1024:
        return 0.0
    if short <= 256:
        return 1.0
    return float((1024.0 - short) / (1024.0 - 256.0))


def _internal_blockiness_score(gray: np.ndarray) -> float:
    g = gray.astype(np.float32)
    gx = np.abs(np.diff(g, axis=1))
    gy = np.abs(np.diff(g, axis=0))
    if gx.size == 0 or gy.size == 0:
        return 0.0
    col = gx.mean(axis=0)
    row = gy.mean(axis=1)
    if col.size < 4 or row.size < 4:
        return 0.0
    score_x = float(np.percentile(col, 95) - np.median(col)) / (float(np.mean(col)) + 1e-6)
    score_y = float(np.percentile(row, 95) - np.median(row)) / (float(np.mean(row)) + 1e-6)
    return float(np.clip(0.25 * max(score_x, score_y), 0.0, 1.0))


def _micro_detail(gray: np.ndarray, amount: float = 0.12) -> np.ndarray:
    g = gray.astype(np.float32)
    blur = cv2.GaussianBlur(g, (0, 0), 0.8)
    detail = g - blur
    out = g + float(amount) * detail
    return np.clip(out, 0, 255).astype(np.uint8)


def _sr_detail_boost_gray(gray: np.ndarray, strength: float = 0.35) -> np.ndarray:
    g = gray.astype(np.float32)
    # Multi-band details: avoid "no effect" while suppressing ringing in flat background.
    b1 = cv2.GaussianBlur(g, (0, 0), 0.75)
    b2 = cv2.GaussianBlur(g, (0, 0), 1.7)
    hp1 = np.clip(g - b1, -18.0, 18.0)
    hp2 = np.clip(b1 - b2, -14.0, 14.0)

    gn = g / 255.0
    gx = cv2.Scharr(gn, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(gn, cv2.CV_32F, 0, 1)
    edge = cv2.magnitude(gx, gy)
    edge = edge / (float(np.percentile(edge, 99.0)) + 1e-6)
    edge = np.clip(edge, 0.0, 1.0)
    edge = cv2.GaussianBlur(edge, (0, 0), 0.9)
    flat = np.clip(1.0 - edge, 0.0, 1.0)

    amount = float(np.clip(strength, 0.08, 0.95))
    out = g + (0.28 + 0.82 * amount) * edge * hp1 + (0.18 + 0.58 * amount) * edge * hp2
    # Background anti-ringing damping.
    out = out - (0.10 + 0.20 * amount) * flat * np.clip(hp1 + 0.65 * hp2, -16.0, 16.0)
    out = np.clip(out, 0.0, 255.0)
    return out.astype(np.uint8)


def _subpixel_granular_gray(gray: np.ndarray, amount: float = 0.25) -> np.ndarray:
    g = gray.astype(np.float32)
    blur1 = cv2.GaussianBlur(g, (0, 0), 0.55)
    blur2 = cv2.GaussianBlur(g, (0, 0), 1.15)
    hp = (g - blur1) + 0.5 * (blur1 - blur2)
    out = g + float(amount) * hp
    return np.clip(out, 0, 255).astype(np.uint8)


def _dark_membrane_maps(gray: np.ndarray):
    g = gray.astype(np.float32) / 255.0
    blur = cv2.GaussianBlur(g, (0, 0), 1.1)
    gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
    edge = np.sqrt(gx * gx + gy * gy)
    edge = edge / (float(edge.max()) + 1e-6)
    dark = 1.0 - blur
    dark_rel = _clip01(dark - cv2.GaussianBlur(dark, (0, 0), 2.2))
    blackhat = cv2.morphologyEx((blur * 255.0).astype(np.uint8), cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))).astype(np.float32) / 255.0
    lap = cv2.Laplacian(blur, cv2.CV_32F, ksize=3)
    gap = _clip01((-lap - 0.02) * 8.0) * (0.35 + 0.65 * edge)
    membrane = _clip01(0.46 * dark + 0.24 * dark_rel + 0.18 * blackhat + 0.12 * edge)
    return dark, edge, dark_rel, blackhat, gap, membrane


def _bridge_gray(gray: np.ndarray, strength: float = 1.0) -> np.ndarray:
    dark, edge, dark_rel, blackhat, gap, membrane = _dark_membrane_maps(gray)
    score = _clip01(0.36 * membrane + 0.22 * dark_rel + 0.18 * blackhat + 0.14 * edge + 0.10 * gap)
    thr = float(np.percentile(score, 78.0))
    mem_mask = ((score > thr) & (dark > 0.12)).astype(np.uint8) * 255
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mem_mask = cv2.morphologyEx(mem_mask, cv2.MORPH_CLOSE, k3, iterations=1)
    bridge = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    soft = cv2.GaussianBlur((mem_mask > 0).astype(np.float32), (0, 0), 1.0)
    out = gray.astype(np.float32) * (1.0 - soft) + bridge.astype(np.float32) * (soft * float(strength))
    return np.clip(out, 0, 255).astype(np.uint8)


def resize_image_if_needed(image_path: str):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f'Unable to read image: {image_path}')
    original_height, original_width = img.shape[:2]
    scale_factor = 1.0
    if original_width > 2048 or original_height > 2048:
        scale_factor = 0.4
        img = cv2.resize(img, (int(original_width * scale_factor), int(original_height * scale_factor)), interpolation=cv2.INTER_AREA)
    elif original_width > 1024 or original_height > 1024:
        scale_factor = 0.7
        img = cv2.resize(img, (int(original_width * scale_factor), int(original_height * scale_factor)), interpolation=cv2.INTER_AREA)
    return img, scale_factor, (original_width, original_height)


def _prepare_inp_image(img: np.ndarray) -> np.ndarray:
    src = np.asarray(img)
    if src.ndim == 2:
        gray = src.astype(np.uint8)
        p1, p99 = np.percentile(gray, [0.4, 99.7])
        base = gray if p99 <= p1 + 1e-3 else np.clip((gray.astype(np.float32) - p1) * (255.0 / (p99 - p1 + 1e-6)), 0, 255).astype(np.uint8)
        drive = float(max(_small_image_factor_hw(*base.shape[:2]), _internal_blockiness_score(base)))
        base = _subpixel_granular_gray(base, amount=0.18 + 0.18 * drive)
        strong = _bridge_gray(base, strength=1.0 + 0.12 * drive)
        strong = _subpixel_granular_gray(strong, amount=0.38 + 0.22 * drive)
        if drive > 0.08:
            strong = _micro_detail(strong, amount=0.16 + 0.12 * drive)
        return strong
    work = src.astype(np.uint8)
    lab = cv2.cvtColor(work, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    p1, p99 = np.percentile(l, [0.4, 99.7])
    base_l = l if p99 <= p1 + 1e-3 else np.clip((l.astype(np.float32) - p1) * (255.0 / (p99 - p1 + 1e-6)), 0, 255).astype(np.uint8)
    drive = float(max(_small_image_factor_hw(*base_l.shape[:2]), _internal_blockiness_score(base_l)))
    base_l = _subpixel_granular_gray(base_l, amount=0.18 + 0.18 * drive)
    strong_l = _bridge_gray(base_l, strength=1.0 + 0.12 * drive)
    strong_l = _subpixel_granular_gray(strong_l, amount=0.38 + 0.22 * drive)
    if drive > 0.08:
        strong_l = _micro_detail(strong_l, amount=0.16 + 0.12 * drive)
    return cv2.cvtColor(cv2.merge((strong_l, a, b)), cv2.COLOR_LAB2BGR)


def prepare_input_for_diffusion(
    input_folder: str,
    mild_clahe: bool = True,
    clip_percentile: Tuple[float, float] = (1.0, 99.0),
    clahe_clip_max: float = 0.85,
    auxiliary_blend_max: float = 0.25,
    deg: Optional[str] = None,
    process_degree: Optional[float] = None,
) -> Tuple[List[Tuple[str, np.ndarray]], List[Tuple[int, int]]]:
    valid_extensions = ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp']
    result, original_sizes = [], []
    for filename in os.listdir(input_folder):
        if not any(filename.lower().endswith(ext) for ext in valid_extensions):
            continue
        img_path = os.path.join(input_folder, filename)
        try:
            img, _scale_factor, original_size = resize_image_if_needed(img_path)
        except Exception as e:
            logger.warning(f'Failed to load {filename}: {e}')
            continue
        if img is None:
            continue
        img = np.asarray(img)
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        strength = float(np.clip(process_degree if process_degree is not None else 0.32, 0.0, 1.0))
        if deg == 'inp_em':
            # Bridge / granular pre-stretch for display; live inpainting uses tools.em_maps.build_inp_mask in EMSVD.
            out = _prepare_inp_image(img)
            if out.ndim == 3:
                lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                l = _micro_detail(l, amount=0.06 + 0.10 * strength)
                out = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
            else:
                out = _micro_detail(out, amount=0.06 + 0.10 * strength)
        elif isinstance(deg, str) and deg.startswith('sr'):
            if img.ndim == 3:
                lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                l = _subpixel_granular_gray(l, amount=0.20 + 0.18 * strength)
                l = _sr_detail_boost_gray(l, strength=0.22 + 0.58 * strength)
                l = _micro_detail(l, amount=0.08 + 0.10 * strength)
                out = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
            else:
                g = _subpixel_granular_gray(img, amount=0.20 + 0.18 * strength)
                g = _sr_detail_boost_gray(g, strength=0.22 + 0.58 * strength)
                out = _micro_detail(g, amount=0.08 + 0.10 * strength)
        else:
            out = img.copy()
        if out.ndim == 3:
            out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
        result.append((filename, out))
        original_sizes.append(original_size)
    if not result:
        raise ValueError('No valid images found for Diffusion input')
    return result, original_sizes


def preprocess_image(image: np.ndarray, membrane_gray_min: int = 50, top_percent: int = 10, is_rgb: bool = False):
    if image is None or image.size == 0:
        raise ValueError('Invalid image provided')
    src = np.asarray(image).copy()
    if src.ndim == 3:
        gray = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY if is_rgb else cv2.COLOR_BGR2GRAY)
    else:
        gray = src.copy()
    gray = gray.astype(np.uint8)
    drive = float(max(_small_image_factor_hw(*gray.shape[:2]), _internal_blockiness_score(gray)))
    prepared = _bridge_gray(gray, strength=0.90 + 0.06 * drive)
    prepared = _subpixel_granular_gray(prepared, amount=0.30 + 0.18 * drive)
    if drive > 0.08:
        prepared = _micro_detail(prepared, amount=0.10 + 0.08 * drive)
    dark, edge, dark_rel, blackhat, gap, membrane = _dark_membrane_maps(prepared)
    score = _clip01(0.34 * membrane + 0.22 * dark_rel + 0.18 * blackhat + 0.14 * edge + 0.12 * gap)
    thr = float(np.percentile(score, max(56, min(92, 100 - int(top_percent) - 6))))
    membrane_mask = ((score > thr) & (dark > 0.14)).astype(np.uint8) * 255
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    membrane_mask = cv2.morphologyEx(membrane_mask, cv2.MORPH_CLOSE, k3, iterations=1)
    membrane_mask = cv2.dilate(membrane_mask, k3, iterations=1)
    membrane_mask = cv2.bitwise_or(membrane_mask, ((gap > 0.10).astype(np.uint8) * 255))
    membrane_mask = cv2.morphologyEx(membrane_mask, cv2.MORPH_CLOSE, k5, iterations=1)
    membrane_gray = cv2.bitwise_and(prepared, prepared, mask=membrane_mask)
    enhanced = cv2.cvtColor(prepared, cv2.COLOR_GRAY2RGB) if is_rgb else prepared
    return enhanced, membrane_mask, membrane_gray
