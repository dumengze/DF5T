import os
import sys
if getattr(sys, 'frozen', False):
    base_path = sys._MEIPASS
else:
    base_path = os.path.dirname(os.path.abspath(__file__))
def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_path, relative_path)
import shutil
import argparse
import yaml
import torch
import json
import logging
import cv2
import numpy as np
import mrcfile
import warnings
from typing import List, Tuple, Optional, Dict
from pathlib import Path
from natsort import natsorted
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton, QLabel,
    QFileDialog, QScrollArea, QGridLayout, QComboBox, QSlider, QHBoxLayout,
    QProgressBar, QGroupBox, QDialog, QTextEdit, QSizePolicy, QCheckBox,
    QRadioButton, QButtonGroup, QLineEdit, QMessageBox, QFrame, QToolButton, QStatusBar
)
from PyQt6.QtGui import QPixmap, QColor, QIcon, QFont, QPainter, QPen
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QPropertyAnimation, QEasingCurve, QPoint
from PIL import Image
from scipy.ndimage import gaussian_filter1d
from PyQt6.QtCore import QSize
from tools.diffusion import Diffusion
from tools.em_volume_io import detect_single_multipage_tiff, load_z_stack, resize_z_stack_hw, save_z_stack_tiff

# Configure logging
_log_dir = os.path.join(base_path, 'outputs', 'logs')
os.makedirs(_log_dir, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(_log_dir, 'image_processor.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=RuntimeWarning, module='mrcfile')

# Theme styles
STYLES = {
    "light": {
        "background": "qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #f5f7fa, stop:1 #e9ecef)",
        "text": "#2d3436",
        "button": "#1B9AAA",
        "button_hover": "#128F88",
        "panel": "#ffffff",
        "shadow": "0 6px 20px rgba(0,0,0,0.08)",
        "border": "#dfe6e9",
        "accent": "#ff6b6b"
    },
    "dark": {
        "background": "qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #2d3436, stop:1 #636e72)",
        "text": "#dfe6e9",
        "button": "#1B9AAA",
        "button_hover": "#128F88",
        "panel": "#353b48",
        "shadow": "0 6px 20px rgba(0,0,0,0.25)",
        "border": "#57606f",
        "accent": "#ff6b6b"
    },
    "high_contrast": {
        "background": "#000000",
        "text": "#ffffff",
        "button": "#00ccff",
        "button_hover": "#00b8e6",
        "panel": "#1a1a1a",
        "shadow": "0 6px 20px rgba(255,255,255,0.1)",
        "border": "#ffffff",
        "accent": "#ff3333"
    }
}

CONFIG_FILE = "app_config.json"

def load_config() -> Dict:
    """Load configuration from file or return default."""
    try:
        default_config = {
            "theme": "light", 
            "last_folder": "", 
            "sidebar_collapsed": False,
            "input_type": "images",
            "enable_postprocessing": False
        }
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r') as f:
                return json.load(f)
        return default_config
    except Exception as e:
        logger.error(f"Error loading config: {str(e)}")
        return default_config

def save_config(config: Dict) -> None:
    """Save configuration to file."""
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=4)
    except Exception as e:
        logger.error(f"Error saving config: {str(e)}")

def resize_image_if_needed(image_path: str) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """
    Resize image if dimensions exceed thresholds and return resized image, scale factor and original size.
    If any dimension exceeds 2048, resize to 1/4 of original.
    If any dimension exceeds 1024, resize to 1/2 of original.
    Otherwise, keep original size.
    """
    try:
        # Read image
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Unable to read image: {image_path}")
        
        original_height, original_width = img.shape[:2]
        scale_factor = 1.0
        
        # Check if resizing is needed
        if original_width > 2048 or original_height > 2048:
            scale_factor = 0.4
            new_width = int(original_width * scale_factor)
            new_height = int(original_height * scale_factor)
            img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
            logger.info(f"Resized image from {original_width}x{original_height} to {new_width}x{new_height} (scale: 1/4)")
        elif original_width > 1024 or original_height > 1024:
            scale_factor = 0.7
            new_width = int(original_width * scale_factor)
            new_height = int(original_height * scale_factor)
            img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
            logger.info(f"Resized image from {original_width}x{original_height} to {new_width}x{new_height} (scale: 1/2)")
        else:
            logger.info(f"Image size {original_width}x{original_height} is within limits, no resizing needed")
        
        return img, scale_factor, (original_width, original_height)
    except Exception as e:
        logger.error(f"Error in resize_image_if_needed: {str(e)}")
        raise

def _is_cryoet_like_gray(gray: np.ndarray) -> bool:
    """Heuristic: cryo-ET often has low contrast, narrow dynamic range, and near-grayscale tone."""
    if gray is None or gray.size == 0:
        return False
    g = gray.astype(np.float32)
    p1, p99 = np.percentile(g, [1, 99])
    dyn = float(p99 - p1)
    std = float(np.std(g))
    mean = float(np.mean(g))
    return (dyn < 105.0 and std < 34.0 and 60.0 < mean < 205.0)


def _tone_safe_input_normalize(
    img: np.ndarray,
    lo: float,
    hi: float,
    deg: Optional[str] = None,
) -> np.ndarray:
    """
    Conservative input normalization that preserves cryo-ET tone / grayscale character.

    Instead of replacing the input with a full percentile stretch, build a mild normalized
    proposal and blend it back toward the raw image with capped gain and mean/std anchoring.
    """
    if img is None or img.size == 0:
        return img

    raw = img.astype(np.float32)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    cryo_like = _is_cryoet_like_gray(gray)

    span = float(max(hi - lo, 1e-6))
    proposal = np.clip((raw - lo) / span, 0.0, 1.0) * 255.0

    raw_mean = float(np.mean(raw))
    raw_std = float(np.std(raw))
    prop_mean = float(np.mean(proposal))
    prop_std = float(np.std(proposal))

    # Match proposal back to raw tone so normalization does not create a large brightness jump.
    scale = raw_std / max(prop_std, 1e-6)
    if cryo_like:
        scale = float(np.clip(scale, 0.94, 1.06))
    else:
        scale = float(np.clip(scale, 0.88, 1.12))
    proposal = proposal * scale + (raw_mean - scale * prop_mean)

    p5, p95 = np.percentile(gray.astype(np.float32), [5, 95])
    contrast = float((p95 - p5) / 255.0)
    contrast = float(np.clip(contrast, 0.0, 1.0))

    if cryo_like:
        # Cryo-ET should stay very close to the raw observation.
        # inp/deblur/adaptive are especially sensitive to tone drift.
        if deg in ('inp_em', 'deblur_em', 'adaptive', 'deno_em', 'isotropic_em'):
            alpha = 0.015
        else:
            alpha = 0.03
    elif deg == 'deno_em':
        alpha = 0.10
    elif deg == 'deblur_em':
        alpha = 0.12
    elif deg == 'adaptive':
        alpha = 0.14
    elif deg == 'inp_em':
        alpha = 0.10
    elif deg and deg[:2] == 'sr':
        alpha = 0.0
    else:
        alpha = 0.12

    # Lower contrast images need some help, but keep the blend small and bounded.
    alpha *= float(np.clip(1.18 - 0.70 * contrast, 0.55, 1.0))
    if cryo_like:
        alpha *= 0.35
        # If the proposal would noticeably shift tone, nearly disable normalization.
        mean_shift = abs(float(np.mean(proposal)) - raw_mean)
        std_shift = abs(float(np.std(proposal)) - raw_std)
        if mean_shift > 6.0 or std_shift > 5.0:
            alpha *= 0.15

    out = (1.0 - alpha) * raw + alpha * proposal
    out = np.clip(out, 0.0, 255.0).astype(np.uint8)
    return out


def _adaptive_preprocess_strength(gray: np.ndarray) -> Tuple[float, float]:
    """
    Compute mild, auxiliary strength from image: 轻微的、辅助的、自适应的.
    Returns (blend_weight, clahe_clip): blend of enhanced with original, and CLAHE clipLimit.
    High contrast -> less enhancement; low contrast -> slightly more (capped).
    """
    if gray.size == 0:
        return 0.2, 0.6
    g = gray.astype(np.float32)
    p5, p95 = np.percentile(g, [5, 95])
    contrast_range = (p95 - p5) / 255.0
    contrast_range = np.clip(contrast_range, 0.05, 1.0)
    std = np.std(g)
    std_n = std / 255.0
    # High contrast / high std -> less blend and weaker CLAHE (adaptive)
    blend = float(np.clip(0.35 - 0.25 * std_n - 0.2 * contrast_range, 0.12, 0.28))
    clahe_clip = float(np.clip(1.0 - 0.5 * std_n, 0.5, 0.85))
    return blend, clahe_clip


def _triple_membrane_structural_enhance(
    img: np.ndarray,
    shrink_strength: float = 0.7,
    center_sharpen: float = 0.8,
) -> np.ndarray:
    """
    EM-aware membrane-radius contraction for app.py preprocessing.

    Design goal:
    - preprocess only for *membrane radius contraction* and gap recovery;
    - never darken membranes globally;
    - treat dark + edge-rich regions as membrane candidates;
    - use a small inverse point-spread step to pull back smeared dark halos so
      the double-membrane bright gap can reappear.
    """
    if img is None or img.size == 0:
        return img
    if float(shrink_strength) <= 0.0:
        return img.astype(np.uint8)

    img_u8 = img.astype(np.uint8)
    is_color = img_u8.ndim == 3

    def _to_luma_u8(x: np.ndarray):
        if x.ndim != 3:
            return x.astype(np.uint8), None, None
        if x.shape[2] == 3:
            lab = cv2.cvtColor(x, cv2.COLOR_BGR2LAB)
        else:
            bgr = cv2.cvtColor(x, cv2.COLOR_BGRA2BGR)
            lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        return l, a, b

    def _from_luma_u8(l: np.ndarray, a, b, src: np.ndarray) -> np.ndarray:
        if a is None or b is None or src.ndim != 3:
            return l.astype(np.uint8)
        lab = cv2.merge((l.astype(np.uint8), a, b))
        bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        if src.shape[2] == 3:
            return bgr
        return np.dstack([bgr, src[:, :, 3]])

    def _robust_norm(x: np.ndarray, lo_p: float, hi_p: float) -> np.ndarray:
        lo, hi = np.percentile(x, [lo_p, hi_p])
        if hi <= lo + 1e-6:
            return np.zeros_like(x, dtype=np.float32)
        return np.clip((x - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)

    l, a, b = _to_luma_u8(img_u8)
    work = l if is_color else img_u8
    base = work.astype(np.float32)

    smooth = cv2.GaussianBlur(work, (0, 0), 0.8)
    bg_med = cv2.medianBlur(work, 5).astype(np.float32)
    bg_open = cv2.morphologyEx(work, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))).astype(np.float32)

    gx = cv2.Scharr(smooth, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(smooth, cv2.CV_32F, 0, 1)
    edge_mag = cv2.magnitude(gx, gy)
    dark_rel = np.clip(np.maximum(bg_med, bg_open) - base, 0.0, 255.0)

    # dark + edge-rich = membrane; bright center between two dark rims = likely gap
    edge_n = _robust_norm(edge_mag, 55, 99.5)
    dark_n = _robust_norm(dark_rel, 45, 99.2)
    likelihood = 0.58 * edge_n + 0.42 * dark_n
    likelihood_u8 = np.clip(likelihood * 255.0, 0, 255).astype(np.uint8)
    _, mem_bin = cv2.threshold(likelihood_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mem_bin = cv2.morphologyEx(mem_bin, cv2.MORPH_OPEN, k3, iterations=1)

    # gap candidates: negative Laplacian inside membrane neighborhood => bright lane between dark edges
    lap = cv2.Laplacian(smooth.astype(np.float32), cv2.CV_32F, ksize=3)
    lap_n = _robust_norm(-lap, 60, 99.5)
    gap_mask = ((lap_n > 0.52) & (cv2.dilate(mem_bin, k3, iterations=1) > 0)).astype(np.uint8) * 255
    gap_mask = cv2.morphologyEx(gap_mask, cv2.MORPH_OPEN, k3, iterations=1)

    # core membrane = neighborhood excluding bright gap; this avoids collapsing the inter-membrane space
    mem_core = cv2.subtract(mem_bin, cv2.dilate(gap_mask, k3, iterations=1))
    mem_core = cv2.morphologyEx(mem_core, cv2.MORPH_OPEN, k3, iterations=1)

    # ring / halo to contract: slightly expanded membrane minus membrane core
    expand_iter = int(np.clip(1 + 2 * float(np.clip(shrink_strength, 0.0, 1.0)), 1, 3))
    ring = cv2.subtract(cv2.dilate(mem_core, k3, iterations=expand_iter), mem_core)
    ring_mask = ring > 0

    out = base.copy()

    # inverse PSF step on ring: pull dark smear back toward local background, but never overshoot bright gap
    if np.any(ring_mask):
        local_bg = np.maximum(bg_med, bg_open)
        alpha = float(np.clip(0.16 + 0.18 * shrink_strength, 0.12, 0.30))
        out[ring_mask] = (1.0 - alpha) * out[ring_mask] + alpha * local_bg[ring_mask]

    # mild backward diffusion only on membrane core: shrink radius by lightening the dark halo, not by darkening the ridge
    if cv2.countNonZero(mem_core) > 0:
        lap2 = cv2.Laplacian(out, cv2.CV_32F, ksize=3)
        beta = float(np.clip(0.03 + 0.06 * shrink_strength, 0.025, 0.08))
        candidate = np.clip(out - beta * lap2, 0, 255)
        core_mask = mem_core > 0
        # never darker than original membrane core; only tighten / slightly brighten
        candidate[core_mask] = np.maximum(candidate[core_mask], base[core_mask])
        out[core_mask] = candidate[core_mask]

        # Slightly deepen only the thinnest membrane centerline; do not darken the full membrane body.
        ridge_center = cv2.erode(mem_core, k3, iterations=1) > 0
        if np.any(ridge_center):
            ridge_boost = np.clip(1.2 + 3.0 * shrink_strength, 0.8, 4.2)
            out[ridge_center] = np.clip(out[ridge_center] - ridge_boost, 0, 255)

    # preserve / reveal bright inter-membrane gap
    if cv2.countNonZero(gap_mask) > 0:
        gm = gap_mask > 0
        bg = np.maximum(bg_med, base)
        out[gm] = np.clip(0.82 * bg[gm] + 0.18 * out[gm], 0, 255)

    # very light local sharpen on membrane neighborhood, but forbid global darkening
    if center_sharpen > 0:
        blur = cv2.GaussianBlur(out, (0, 0), 0.9)
        k = float(np.clip(0.05 + 0.10 * center_sharpen, 0.03, 0.10))
        sharpen = np.clip(out + k * (out - blur), 0, 255)
        roi = cv2.dilate(mem_core, k3, iterations=1) > 0
        sharpen[roi] = np.maximum(sharpen[roi], out[roi])
        out[roi] = sharpen[roi]

    out_u8 = np.clip(out, 0, 255).astype(np.uint8)
    return _from_luma_u8(out_u8, a, b, img_u8) if is_color else out_u8

def prepare_input_for_diffusion(
    input_folder: str,
    mild_clahe: bool = True,
    clip_percentile: Tuple[float, float] = (1.0, 99.0),
    clahe_clip_max: float = 0.85,
    auxiliary_blend_max: float = 0.25,
    deg: Optional[str] = None,
    process_degree: Optional[float] = None,
) -> Tuple[List[Tuple[str, np.ndarray]], List[Tuple[int, int]]]:
    """
    Light, task-aware normalization for Diffusion input. Model does restoration; preprocessing is minimal.
    - deno_em: percentile clip only, no CLAHE (blend=0) so input is near-raw.
    - deblur_em / sr2 / isotropic_em / inp_em: very light CLAHE blend (max 0.12), lower clip.
    """
    valid_extensions = [".png"]
    result: List[Tuple[str, np.ndarray]] = []
    original_sizes: List[Tuple[int, int]] = []
    # Task-aware caps. Every task gets membrane contraction assistance; only the strength differs.
    if deg == "deno_em":
        blend_cap = min(auxiliary_blend_max, 0.012)
        clip_cap = min(clahe_clip_max, 0.06)
        shrink_strength = 0.32
    elif deg == "deblur_em":
        blend_cap = min(auxiliary_blend_max, 0.015)
        clip_cap = min(clahe_clip_max, 0.06)
        shrink_strength = 0.34
    elif deg == "adaptive":
        blend_cap = min(auxiliary_blend_max, 0.015)
        clip_cap = min(clahe_clip_max, 0.06)
        shrink_strength = 0.32
    elif deg == "isotropic_em":
        blend_cap = min(auxiliary_blend_max, 0.012)
        clip_cap = min(clahe_clip_max, 0.06)
        shrink_strength = 0.30
    elif deg == "inp_em":
        # Keep input near-raw, but allow a *very* light local contrast lift so weak membranes
        # are still visible to the diffusion branch. This is intentionally much milder than a
        # restoration pipeline and avoids the old heavy preprocessing artifacts.
        blend_cap = min(auxiliary_blend_max, 0.015)
        clip_cap = min(clahe_clip_max, 0.06)
        shrink_strength = 0.0
    elif deg and deg[:2] == "sr":
        blend_cap = 0.0
        clip_cap = 0.0
        shrink_strength = 0.0
    else:
        blend_cap = min(auxiliary_blend_max, 0.018)
        clip_cap = min(clahe_clip_max, 0.07)
        shrink_strength = 0.30
    for filename in os.listdir(input_folder):
        if not any(filename.lower().endswith(ext) for ext in valid_extensions):
            continue
        img_path = os.path.join(input_folder, filename)
        try:
            img, _scale_factor, original_size = resize_image_if_needed(img_path)
        except Exception as e:
            logger.warning(f"Failed to load {filename}: {e}")
            continue
        if img is None:
            continue
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        # 1) Very mild percentile normalization.
        # For deblur/adaptive we deliberately keep the input close to raw observations.
        if deg == "deblur_em":
            lo, hi = np.percentile(gray, [0.2, 99.8])
        elif deg == "adaptive":
            lo, hi = np.percentile(gray, [0.5, 99.5])
        else:
            lo, hi = np.percentile(gray, [clip_percentile[0], clip_percentile[1]])
        base_norm = _tone_safe_input_normalize(img, lo, hi, deg=deg)
        cryo_like_input = _is_cryoet_like_gray(gray)
        if cryo_like_input and deg in ('inp_em', 'deblur_em', 'adaptive', 'deno_em', 'isotropic_em'):
            # For cryo-ET, normalization should not materially change the tone.
            # Keep only a tiny anchored correction instead of a visible remap.
            base_norm = np.clip(0.96 * img.astype(np.float32) + 0.04 * base_norm.astype(np.float32), 0, 255).astype(np.uint8)

        # 2) Membrane-radius contraction is mandatory for all EM tasks.
        # We shrink dark smeared membrane halos and protect / widen the bright bilayer gap.
        img_struct = _triple_membrane_structural_enhance(
            base_norm,
            shrink_strength=shrink_strength,
            center_sharpen=0.05 if deg in ("deblur_em", "adaptive") else 0.04,
        )
        img_norm = img_struct.copy()

        # 3) CLAHE remains auxiliary only; it should never dominate membrane shaping.
        if mild_clahe and clip_cap > 0:
            gray_norm_for_adapt = cv2.cvtColor(img_norm, cv2.COLOR_BGR2GRAY) if len(img_norm.shape) == 3 else img_norm
            cryo_like = _is_cryoet_like_gray(gray_norm_for_adapt)
            blend, clahe_clip = _adaptive_preprocess_strength(gray_norm_for_adapt)
            if cryo_like:
                # CLAHE on cryo-ET should be barely perceptible.
                blend *= 0.12
                clahe_clip *= 0.18
            blend = min(blend, blend_cap)
            clahe_clip = min(clahe_clip, clip_cap)
            if clahe_clip > 0:
                if len(img_norm.shape) == 2:
                    clahe = cv2.createCLAHE(clipLimit=float(clahe_clip), tileGridSize=(8, 8))
                    enhanced = clahe.apply(img_norm)
                    img_norm = np.clip(
                        (1 - blend) * img_norm.astype(np.float32) + blend * enhanced.astype(np.float32),
                        0, 255
                    ).astype(np.uint8)
                else:
                    lab = cv2.cvtColor(img_norm, cv2.COLOR_BGR2LAB)
                    l_ch, a_ch, b_ch = cv2.split(lab)
                    clahe = cv2.createCLAHE(clipLimit=float(clahe_clip), tileGridSize=(8, 8))
                    l_enh = clahe.apply(l_ch)
                    l_ch = np.clip(
                        (1 - blend) * l_ch.astype(np.float32) + blend * l_enh.astype(np.float32),
                        0, 255
                    ).astype(np.uint8)
                    img_norm = cv2.cvtColor(cv2.merge((l_ch, a_ch, b_ch)), cv2.COLOR_LAB2BGR)

        if len(img_norm.shape) == 3:
            img_norm = cv2.cvtColor(img_norm, cv2.COLOR_BGR2RGB)
        result.append((filename, img_norm))
        original_sizes.append(original_size)
    if not result:
        raise ValueError("No valid images found for Diffusion input")
    return result, original_sizes

def light_display_enhancement(
    image_path_or_array,
    clip_percentile: Tuple[float, float] = (2.0, 98.0),
    mild_clahe: bool = True,
    clahe_clip: float = 0.0,
    auxiliary_blend_max: float = 0.12,
) -> np.ndarray:
    """
    Light, auxiliary, adaptive display enhancement: 轻微的、辅助的、自适应的.
    Percentile norm + optional very mild CLAHE, blended with original so enhancement is auxiliary.
    """
    if isinstance(image_path_or_array, str):
        image, _sf, _size = resize_image_if_needed(image_path_or_array)
        if image is None:
            raise ValueError(f"Unable to load image at {image_path_or_array}")
    else:
        image = np.asarray(image_path_or_array)
        if image.size == 0:
            raise ValueError("Empty image in light_display_enhancement")
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    blend, adaptive_clip = _adaptive_preprocess_strength(gray)
    blend = min(blend, auxiliary_blend_max)
    clip_use = adaptive_clip if clahe_clip <= 0 else min(float(clahe_clip), 0.85)
    lo, hi = np.percentile(gray, [clip_percentile[0], clip_percentile[1]])
    img_float = np.clip((image.astype(np.float32) - lo) / (hi - lo + 1e-8), 0, 1)
    out = (img_float * 255).astype(np.uint8)
    if mild_clahe and clip_use > 0:
        if len(out.shape) == 2:
            clahe = cv2.createCLAHE(clipLimit=float(clip_use), tileGridSize=(8, 8))
            enhanced = clahe.apply(out)
            out = np.clip(
                (1 - blend) * out.astype(np.float32) + blend * enhanced.astype(np.float32),
                0, 255
            ).astype(np.uint8)
        else:
            lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
            l_ch, a_ch, b_ch = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=float(clip_use), tileGridSize=(8, 8))
            l_enh = clahe.apply(l_ch)
            l_ch = np.clip((1 - blend) * l_ch.astype(np.float32) + blend * l_enh.astype(np.float32), 0, 255).astype(np.uint8)
            out = cv2.cvtColor(cv2.merge((l_ch, a_ch, b_ch)), cv2.COLOR_LAB2BGR)
    return out

def convert_to_png(input_folder: str) -> None:
    """Convert all supported images in the input folder to PNG format."""
    try:
        if not os.path.exists(input_folder):
            logger.error(f"Folder {input_folder} does not exist")
            raise FileNotFoundError(f"Folder {input_folder} does not exist")

        valid_extensions = ['.tif', '.tiff', '.jpg', '.jpeg']
        files = os.listdir(input_folder)
        logger.info(f"Found files in {input_folder}: {files}")
        if not files:
            logger.error(f"No files found in {input_folder}")
            raise FileNotFoundError(f"No files found in {input_folder}")

        for filename in files:
            if any(filename.lower().endswith(ext) for ext in valid_extensions):
                src_path = os.path.join(input_folder, filename)
                name_without_ext = os.path.splitext(filename)[0]
                dst_path = os.path.join(input_folder, f"{name_without_ext}.png")
                try:
                    with Image.open(src_path) as img:
                        img.save(dst_path, 'PNG')
                    os.remove(src_path)
                    logger.info(f"Converted {filename} to {name_without_ext}.png")
                except Exception as e:
                    logger.warning(f"Failed to convert {filename}: {str(e)}")
            elif filename.lower().endswith('.png'):
                logger.info(f"Keeping existing PNG file: {filename}")
            else:
                logger.info(f"Skipping file {filename}: not a valid image extension")
    except Exception as e:
        logger.error(f"Error in convert_to_png: {str(e)}")
        raise

def setup_dataset_and_list(input_folder: str, enhanced_images: Optional[List[Tuple[str, np.ndarray]]]=None) -> Tuple[str, str]:
    """Set up dataset directory and image list file."""
    try:
        input_folder = os.path.normpath(input_folder)
        logger.info(f"Setting up dataset for folder: {input_folder}")
        
        dataset_dir = os.path.join(input_folder, "datasets", "MitEM", "MitEM")
        os.makedirs(dataset_dir, exist_ok=True)
        logger.info(f"Created dataset directory: {dataset_dir}")

        valid_extensions = ['.png']
        if enhanced_images is None:
            convert_to_png(input_folder)
            valid_files = [
                f for f in os.listdir(input_folder)
                if any(f.lower().endswith(ext) for ext in valid_extensions)
            ]
            logger.info(f"Found valid files: {valid_files}")
            
            for filename in valid_files:
                src_path = os.path.join(input_folder, filename)
                dst_path = os.path.join(dataset_dir, filename)
                try:
                    if not os.path.exists(dst_path):
                        shutil.copy2(src_path, dst_path)
                        logger.info(f"Copied {src_path} to {dst_path}")
                    else:
                        logger.info(f"File {dst_path} already exists, skipping copy")
                except Exception as e:
                    logger.warning(f"Failed to copy {filename}: {str(e)}")
        else:
            valid_files = []
            for filename, img in enhanced_images:
                if img is None or img.size == 0:
                    logger.warning(f"Skipping invalid image: {filename}")
                    continue
                dst_path = os.path.join(dataset_dir, filename)
                try:
                    cv2.imwrite(dst_path, img)
                    logger.info(f"Saved enhanced image {filename} to {dst_path}")
                    valid_files.append(filename)
                except Exception as e:
                    logger.warning(f"Failed to save enhanced image {filename}: {str(e)}")

        if not valid_files:
            raise ValueError(f"No valid images found in {input_folder}")

        txt_path = os.path.join(input_folder, "MitEM_val_1k.txt")
        sorted_files = natsorted(valid_files)
        with open(txt_path, 'w') as f:
            for filename in sorted_files:
                name_without_extension = os.path.splitext(filename)[0]
                f.write(f"{name_without_extension} 1\n")
        logger.info(f"Created image list file: {txt_path}")

        return txt_path, dataset_dir
    except Exception as e:
        logger.error(f"Error in setup_dataset_and_list: {str(e)}")
        raise

def dict2namespace(config: Dict) -> argparse.Namespace:
    """Convert dictionary to namespace recursively."""
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            setattr(namespace, key, dict2namespace(value))
        else:
            setattr(namespace, key, value)
    return namespace

def save_mrc_slices_as_images(mrc_path, output_folder):
    """Convert MRC file to image slices (preprocessing)."""
    try:
        os.makedirs(output_folder, exist_ok=True)
        image_list = []
        
        logger.info(f"Processing MRC file: {mrc_path}")
        with mrcfile.open(mrc_path, permissive=True) as mrc:
            data = mrc.data
            
            if data is None or data.size == 0:
                logger.warning(f"The MRC file {mrc_path} contains empty data.")
                return []
                
            if data.ndim == 3:
                num_slices = data.shape[0]
            else:
                num_slices = 1
                data = data[np.newaxis,...]
            
            logger.info(f"Processing {num_slices} slices...")
            for i in range(num_slices):
                try:
                    slice_data = data[i].copy()
                
                    if np.all(slice_data == 0):
                        logger.warning(f"Slice {i+1} contains all zero values.")
                        continue

                    slice_data = cv2.normalize(slice_data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    
                    output_path = os.path.join(output_folder, f'slice_{i+1:04d}.png')
                    cv2.imwrite(output_path, slice_data)
                    
                    if os.path.exists(output_path):
                        image_list.append(output_path)
                        logger.info(f"Slice {i+1}/{num_slices} saved successfully.")
                    else:
                        logger.warning(f"Failed to save slice {i+1}.")
                        
                except Exception as e:
                    logger.error(f"Error processing slice {i+1}: {e}")
                    continue
                    
        logger.info(f"Successfully saved {len(image_list)} slices.")
        return image_list
        
    except Exception as e:
        logger.error(f"Error processing MRC file {mrc_path}: {e}")
        return []

def read_mrc_header(file_path):
    """Read MRC file header information."""
    with mrcfile.open(file_path, 'r') as mrc:
        header = mrc.header
        logger.info(f"MRC header: nx={header.nx} (type={type(header.nx)}), ny={header.ny} (type={type(header.ny)})")
        return header

def identify_bright_dark_layers(data, bright_percentile=80, dark_percentile=20, max_fraction=0.2):
    """
    Identify overly bright and dark layers.
    Parameters:
        data: 3D numpy array, shape (z, y, x)
        bright_percentile: Percentile threshold for bright layers
        dark_percentile: Percentile threshold for dark layers
        max_fraction: Maximum proportion of layers to adjust
    Returns:
        bright_layers: List of bright layer indices
        dark_layers: List of dark layer indices
        layer_means: Gray mean values per layer
        bright_threshold: Bright threshold
        dark_threshold: Dark threshold
    """
    layer_means = np.mean(data, axis=(1, 2))
    bright_threshold = np.percentile(layer_means, bright_percentile)
    dark_threshold = np.percentile(layer_means, dark_percentile)
    
    bright_layers = np.where(layer_means > bright_threshold)[0]
    dark_layers = np.where(layer_means < dark_threshold)[0]
    
    max_layers = int(len(layer_means) * max_fraction)
    if len(bright_layers) > max_layers:
        bright_layers = bright_layers[np.argsort(layer_means[bright_layers])[-max_layers:]]
    if len(dark_layers) > max_layers:
        dark_layers = dark_layers[np.argsort(layer_means[dark_layers])[:max_layers]]
    
    return bright_layers, dark_layers, layer_means, bright_threshold, dark_threshold

def adjust_layers_dynamic(data, bright_layers, dark_layers, global_mean, min_scale=0.7, max_scale=1.3):
    """
    Dynamically adjust brightness of overly bright/dark layers to approach global mean.
    Parameters:
        data: 3D numpy array, shape (z, y, x)
        bright_layers: List of bright layer indices
        dark_layers: List of dark layer indices
        global_mean: Global mean (target mean)
        min_scale: Minimum scaling factor
        max_scale: Maximum scaling factor
    Returns:
        Processed 3D array
    """
    processed_data = np.copy(data)
    layer_means = np.mean(processed_data, axis=(1, 2))
    
    for z in bright_layers:
        if layer_means[z] > 0:
            scale = global_mean / layer_means[z]
            scale = min(max(scale, min_scale), max_scale)
            processed_data[z] = processed_data[z] * scale
    
    for z in dark_layers:
        if layer_means[z] > 0:
            scale = global_mean / layer_means[z]
            scale = min(max(scale, min_scale), max_scale)
            processed_data[z] = processed_data[z] * scale
    
    return processed_data

def smooth_layers(data, sigma=2):
    """
    Apply Gaussian smoothing along z-axis.
    Parameters:
        data: 3D numpy array, shape (z, y, x)
        sigma: Standard deviation of Gaussian kernel
    Returns:
        Smoothed 3D array
    """
    smoothed_data = gaussian_filter1d(data, sigma=sigma, axis=0)
    return smoothed_data

def normalize_to_global_mean(data, target_mean, max_scale=1.2, min_scale=0.8):
    """
    Normalize each layer's grayscale to global mean, with separate limits for bright and dark layers.
    Parameters:
        data: 3D numpy array, shape (z, y, x)
        target_mean: Target mean
        max_scale: Maximum scaling factor for darkening bright layers
        min_scale: Minimum scaling factor for lightening dark layers (to avoid over-brightening)
    Returns:
        Normalized 3D array
    """
    processed_data = np.copy(data)
    layer_means = np.mean(processed_data, axis=(1, 2))
    for z in range(len(layer_means)):
        if layer_means[z] > 0:
            scale = target_mean / layer_means[z]
            if scale > 1.0:
                # For dark layers (scale > 1), limit amplification to avoid whitening
                scale = min(scale, max_scale)
            else:
                # For bright layers (scale < 1), allow more reduction
                scale = max(scale, min_scale)
            # Apply scaling and clip to the global min/max to preserve overall range
            processed_data[z] = np.clip(processed_data[z] * scale, data.min(), data.max())
    return processed_data

def get_dtype_from_mode(mode):
    if mode == 0:
        return np.int8
    elif mode == 1:
        return np.int16
    elif mode == 2:
        return np.float32
    elif mode == 6:
        return np.uint16
    else:
        logger.warning(f"Unsupported mode {mode}, defaulting to float32")
        return np.float32


def create_mrc_from_images(image_dir, output_mrc_path, template_mrc_path, original_sizes=None):
    """Create MRC file from images with template header, preserving original grayscale values."""
    try:
        # Read template MRC header
        with mrcfile.open(template_mrc_path, 'r') as template_mrc:
            template_header = template_mrc.header

        # Get sorted list of image files
        image_files = sorted(
            [f for f in os.listdir(image_dir) if f.endswith('_-1.png')],
            key=lambda x: int(x.split('_')[0]) if x.split('_')[0].isdigit() else 0
        )
        
        if not image_files:
            logger.error(f"No image files found in directory: {image_dir}")
            return False

        logger.info(f"Found {len(image_files)} image files: {image_files}")

        # Load first image to get dimensions
        first_image_path = os.path.join(image_dir, image_files[0])
        first_image = Image.open(first_image_path).convert('L')  # Convert to grayscale
        img_array = np.array(first_image, dtype=np.float32)
        height, width = img_array.shape

        # Create empty 3D array with image dimensions
        num_slices = len(image_files)
        data = np.zeros((num_slices, height, width), dtype=np.float32)

        # Load all images into the data array
        for i, image_file in enumerate(image_files):
            image_path = os.path.join(image_dir, image_file)
            image = Image.open(image_path).convert('L')  # Convert to grayscale
            data[i, :, :] = np.array(image, dtype=np.float32)

        # Create MRC file
        with mrcfile.new(output_mrc_path, overwrite=True) as mrc:
            mrc.set_data(data)

            # Copy header information from template
            header = mrc.header
            header.nx = width  # Image width
            header.ny = height  # Image height
            header.nz = num_slices  # Number of slices
            header.mode = 2  # Mode 2: float32

            # Copy additional header fields
            header.nxstart = template_header.nxstart
            header.nystart = template_header.nystart
            header.nzstart = template_header.nzstart
            header.mx = template_header.mx
            header.my = template_header.my
            header.mz = template_header.mz
            
            header.cella.x = template_header.cella.x
            header.cella.y = template_header.cella.y
            header.cella.z = template_header.cella.z
            header.cellb.alpha = template_header.cellb.alpha
            header.cellb.beta = template_header.cellb.beta
            header.cellb.gamma = template_header.cellb.gamma
            
            header.mapc = template_header.mapc
            header.mapr = template_header.mapr
            header.maps = template_header.maps
            
            # Preserve template's min, max, mean
            header.dmin = template_header.dmin
            header.dmax = template_header.dmax
            header.dmean = template_header.dmean
            header.ispg = template_header.ispg
            header.origin.x = template_header.origin.x
            header.origin.y = template_header.origin.y
            header.origin.z = template_header.origin.z
            
            # Copy optional fields if they exist
            if hasattr(template_header, 'cmt'):
                header.cmt = template_header.cmt
            if hasattr(template_header, 'date'):
                header.date = template_header.date
            if hasattr(template_header, 'map'):
                header.map = template_header.map
            if hasattr(template_header, 'machst'):
                header.machst = template_header.machst
            if hasattr(template_header, 'rms'):
                header.rms = template_header.rms
            if hasattr(template_header, 'nlabl'):
                header.nlabl = template_header.nlabl
            if hasattr(template_header, 'label'):
                header.label = template_header.label

        logger.info(f"Created MRC file with {num_slices} slices at {output_mrc_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error creating MRC file: {str(e)}")
        return False


class ImageProcessor(QThread):
    finished = pyqtSignal(list)
    error = pyqtSignal(str)
    progress = pyqtSignal(int)
    status = pyqtSignal(str)

    def __init__(
        self,
        input_folder: str,
        deg: str,
        timesteps: int,
        sigma_0: float,
        model_path: str,
        use_membrane_enhancement: bool = False,
        top_percent: int = 80,
        dispersion_ratio: float = 0.9,
        denoise_strength: float = 0.005,
        apply_enhancement_to_output: bool = False,
    ):
        super().__init__()
        self.input_folder = input_folder
        self.deg = deg
        self.timesteps = timesteps
        self.sigma_0 = sigma_0
        self.model_path = model_path
        self.use_membrane_enhancement = use_membrane_enhancement
        self.top_percent = top_percent
        self.dispersion_ratio = dispersion_ratio
        self.denoise_strength = denoise_strength
        self.apply_enhancement_to_output = apply_enhancement_to_output
        self._running = True
        self.original_sizes = []
        self._zstack_manifest_path: Optional[str] = None

    def stop(self) -> None:
        """Stop the processing thread."""
        self._running = False

    def _run_zstack_pipeline(self, tiff_path: str) -> None:
        import tifffile

        self.status.emit("Z-stack TIFF detected; loading volume...")
        stack, _meta = load_z_stack(tiff_path)
        stack, _scale_factor, orig_hw = resize_z_stack_hw(stack)
        self.original_sizes = [orig_hw]
        stem = Path(tiff_path).stem
        z_depth = int(stack.shape[0])
        mid = z_depth // 2
        sidecar_in = os.path.join(self.input_folder, "_df5t_zstack_input_mid.png")
        cv2.imwrite(sidecar_in, stack[mid])
        self.progress.emit(6)
        if not self._running:
            return

        output_folder = os.path.join(self.input_folder, "output")
        os.makedirs(output_folder, exist_ok=True)
        dummy_txt = os.path.join(output_folder, "_zstack_unused_list.txt")
        with open(dummy_txt, "w", encoding="utf-8") as _f:
            _f.write("")

        def _vol_prog(cur: int, total: int) -> None:
            self.progress.emit(40 + int(38 * cur / max(total, 1)))

        args = argparse.Namespace(
            ni=True,
            config="DF5T_256.yml",
            doc="processed",
            timesteps=self.timesteps,
            deg=self.deg,
            sigma_0=self.sigma_0,
            seed=1234,
            exp=self.input_folder,
            comment="",
            verbose="info",
            sample=True,
            image_folder=output_folder,
            subset_start=-1,
            subset_end=-1,
            eta=0.85,
            etaB=1,
            model_path=self.model_path,
            fast_adaptive=True if self.deg == "adaptive" else False,
            apply_result_light_enhance=bool(self.apply_enhancement_to_output),
            status_cb=self.status.emit,
            volume_progress=_vol_prog,
        )

        config_path = resource_path(os.path.join("configs", args.config))
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file {args.config} not found")
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)
        config_dict["data"] = config_dict.get("data", {})
        config_dict["data"]["root"] = output_folder
        config_dict["data"]["txt"] = dummy_txt
        config = dict2namespace(config_dict)

        supported_degradations = ["deblur_em", "deno_em", "isotropic_em", "inp_em", "sr2", "adaptive"]
        if self.deg not in supported_degradations:
            raise ValueError(f"Degradation type '{self.deg}' not supported")

        self.progress.emit(35)
        if self.deg == "adaptive":
            self.status.emit(
                "Adaptive Z-stack: routing diagnostics on middle slice only; per-slice restoration for full volume."
            )
        else:
            self.status.emit(f"Processing Z-stack with degradation: {self.deg}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        runner = Diffusion(args, config, device)
        samples = runner.sample_z_stack(stack, stem)
        for s in samples:
            qm = s.debug.get("quality_metrics") if isinstance(getattr(s, "debug", None), dict) else None
            if qm:
                out = qm.get("output", {})
                logger.info(
                    "Quality metrics (z mid) [%s]: lap_var=%.0f grad=%.4f contrast=%.3f corr=%.3f Δlap=%.0f",
                    s.stem,
                    float(out.get("laplacian_variance", 0.0)),
                    float(out.get("gradient_mean", 0.0)),
                    float(out.get("percentile_contrast_5_95", 0.0)),
                    float(qm.get("global_correlation", 0.0)),
                    float(qm.get("delta_laplacian_variance", 0.0)),
                )
        if samples:
            q0 = samples[0].debug.get("quality_metrics", {}) if isinstance(samples[0].debug, dict) else {}
            if q0:
                o0 = q0.get("output", {})
                self.status.emit(
                    f"Restoration quality (middle slice): sharpness={float(o0.get('laplacian_variance', 0)):.0f} "
                    f"contrast={float(o0.get('percentile_contrast_5_95', 0)):.3f} corr={float(q0.get('global_correlation', 0)):.3f}"
                )

        if not self._running:
            return

        manifest = {
            "mode": "zstack",
            "z_depth": z_depth,
            "stem": stem,
            "input_tiff": os.path.abspath(tiff_path),
            "input_mid_png": os.path.abspath(sidecar_in),
            "output_mid_png": os.path.join(output_folder, f"{stem}_-1_mid.png"),
            "volumes": {
                "linear": f"{stem}_linear.tif",
                "svd_degraded": f"{stem}_svd_degraded.tif",
                "nonlinear": f"{stem}_nonlinear.tif",
                "final_gray": f"{stem}_final.tif",
                "final_restored": f"{stem}_-1.tif",
            },
        }
        man_path = os.path.join(output_folder, "zstack_manifest.json")
        with open(man_path, "w", encoding="utf-8") as _f:
            json.dump(manifest, _f, indent=2)
        self._zstack_manifest_path = man_path

        self.status.emit("Collecting Z-stack results (multi-page TIFF + middle-slice preview)...")
        restored_images = [os.path.join(output_folder, f"{stem}_-1_mid.png")]
        if not os.path.isfile(restored_images[0]):
            raise ValueError(f"No Z-stack preview output at {restored_images[0]}")

        if isinstance(self.deg, str) and self.deg.startswith("sr") and restored_images:
            self.status.emit("Applying x2 super-resolution to Z-stack TIFF...")
            sr_out_dir = os.path.join(output_folder, "sr_x2")
            os.makedirs(sr_out_dir, exist_ok=True)
            vpath = os.path.join(output_folder, f"{stem}_-1.tif")
            arr = tifffile.imread(vpath)
            out_sl = []
            for zi in range(arr.shape[0]):
                sl = arr[zi]
                if sl.ndim == 2:
                    sl3 = cv2.cvtColor(sl, cv2.COLOR_GRAY2BGR)
                else:
                    sl3 = sl[..., :3] if sl.shape[-1] >= 3 else cv2.cvtColor(sl[..., 0], cv2.COLOR_GRAY2BGR)
                sr_img = cv2.resize(sl3, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
                out_sl.append(sr_img)
            sr_stack = np.stack(out_sl, axis=0)
            save_z_stack_tiff(os.path.join(sr_out_dir, f"{stem}_-1.tif"), sr_stack)
            mid_sr_path = os.path.join(sr_out_dir, f"{stem}_-1_mid.png")
            cv2.imwrite(mid_sr_path, sr_stack[mid])
            restored_images = [mid_sr_path]

        self.progress.emit(100)
        self.finished.emit(restored_images)

    def run(self) -> None:
        try:
            zstack_path = detect_single_multipage_tiff(self.input_folder)
            if zstack_path is not None:
                self._run_zstack_pipeline(zstack_path)
                return

            self.status.emit("Converting images to PNG...")
            convert_to_png(self.input_folder)
            self.progress.emit(5)

            # Priority policy: keep restoration task effect dominant.
            # Skip app-side input preprocessing and output postprocessing by default.
            self.status.emit("Skipping app-side preprocessing (use near-raw input)...")
            valid_extensions = [".png"]
            enhanced_images = []
            self.original_sizes = []
            for f in os.listdir(self.input_folder):
                if f.lower().endswith(tuple(valid_extensions)):
                    img_path = os.path.join(self.input_folder, f)
                    img, _sf, original_size = resize_image_if_needed(img_path)
                    if len(img.shape) == 3:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    enhanced_images.append((f, img))
                    self.original_sizes.append(original_size)
            if not enhanced_images:
                raise ValueError("No valid images found in folder")
            self.progress.emit(10)

            self.status.emit("Preparing dataset...")
            txt_file, dataset_dir = setup_dataset_and_list(self.input_folder, enhanced_images)
            self.progress.emit(20)

            if not self._running:
                return

            self.status.emit("Setting up output directory...")
            output_folder = os.path.join(self.input_folder, "output")
            os.makedirs(output_folder, exist_ok=True)

            args = argparse.Namespace(
                ni=True,
                config="DF5T_256.yml",
                doc="processed",
                timesteps=self.timesteps,
                deg=self.deg,
                sigma_0=self.sigma_0,
                seed=1234,
                exp=self.input_folder,
                comment="",
                verbose="info",
                sample=True,
                image_folder=output_folder,
                subset_start=-1,
                subset_end=-1,
                eta=0.85,
                etaB=1,
                model_path=self.model_path,
                # Adaptive routing should be a fast SVD probe; model runs only for selected tasks.
                fast_adaptive=True if self.deg == "adaptive" else False,
                # Result postprocess should follow UI checkbox: "Result -> Light enhance".
                apply_result_light_enhance=bool(self.apply_enhancement_to_output),
                status_cb=self.status.emit,
            )

            config_path = resource_path(os.path.join("configs", args.config))
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Config file {config_path} not found")
            with open(config_path, "r") as f:
                config_dict = yaml.safe_load(f)
            config_dict['data'] = config_dict.get('data', {})
            config_dict['data']['root'] = dataset_dir
            config_dict['data']['txt'] = txt_file
            config = dict2namespace(config_dict)

            supported_degradations = ["deblur_em", "deno_em", "isotropic_em", "inp_em", "sr2", "adaptive"]
            if self.deg not in supported_degradations:
                raise ValueError(f"Degradation type '{self.deg}' not supported")

            self.progress.emit(40)
            if self.deg == "adaptive":
                self.status.emit("Adaptive routing: fast SVD probe first (no model), then run model only for selected tasks...")
            else:
                self.status.emit(f"Processing images with degradation: {self.deg}")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            runner = Diffusion(args, config, device)
            samples = runner.sample()
            for s in samples:
                qm = s.debug.get("quality_metrics") if isinstance(getattr(s, "debug", None), dict) else None
                if qm:
                    out = qm.get("output", {})
                    logger.info(
                        "Quality metrics [%s]: lap_var=%.0f grad=%.4f contrast=%.3f corr=%.3f Δlap=%.0f",
                        s.stem,
                        float(out.get("laplacian_variance", 0.0)),
                        float(out.get("gradient_mean", 0.0)),
                        float(out.get("percentile_contrast_5_95", 0.0)),
                        float(qm.get("global_correlation", 0.0)),
                        float(qm.get("delta_laplacian_variance", 0.0)),
                    )
            if samples:
                q0 = samples[0].debug.get("quality_metrics", {}) if isinstance(samples[0].debug, dict) else {}
                if q0:
                    o0 = q0.get("output", {})
                    self.status.emit(
                        f"Restoration quality (1/{len(samples)}): sharpness={float(o0.get('laplacian_variance', 0)):.0f} "
                        f"contrast={float(o0.get('percentile_contrast_5_95', 0)):.3f} corr={float(q0.get('global_correlation', 0)):.3f}"
                    )
            self.progress.emit(80)

            if not self._running:
                return

            self.status.emit("Collecting results...")
            restored_images = [
                os.path.join(output_folder, f) for f in os.listdir(output_folder)
                if f.endswith(".png") and "-1" in f
            ]
            if not restored_images:
                raise ValueError(f"No restored images found in {output_folder}")

            if False and self.apply_enhancement_to_output:
                self.status.emit("Applying light enhancement to restored output...")
                enhanced_out_dir = os.path.join(output_folder, "enhanced")
                os.makedirs(enhanced_out_dir, exist_ok=True)
                for path in restored_images:
                    try:
                        final_image = light_display_enhancement(
                            path,
                            clip_percentile=(2.0, 98.0),
                            mild_clahe=True,
                            clahe_clip=0.5,
                            auxiliary_blend_max=0.15,
                        )
                        out_path = os.path.join(enhanced_out_dir, os.path.basename(path))
                        cv2.imwrite(out_path, final_image)
                    except Exception as e:
                        logger.warning(f"Light enhancement failed for {path}: {e}")
                restored_images = [
                    os.path.join(enhanced_out_dir, f) for f in os.listdir(enhanced_out_dir)
                    if f.endswith(".png")
                ] if os.path.isdir(enhanced_out_dir) else restored_images

            if isinstance(self.deg, str) and self.deg.startswith("sr") and restored_images:
                self.status.emit("Applying x2 super-resolution output upscale...")
                sr_out_dir = os.path.join(output_folder, "sr_x2")
                os.makedirs(sr_out_dir, exist_ok=True)
                sr_paths = []
                for path in restored_images:
                    try:
                        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                        if img is None:
                            continue
                        sr_img = cv2.resize(img, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
                        out_path = os.path.join(sr_out_dir, os.path.basename(path))
                        cv2.imwrite(out_path, sr_img)
                        sr_paths.append(out_path)
                    except Exception as e:
                        logger.warning(f"SR output upscale failed for {path}: {e}")
                if sr_paths:
                    restored_images = sr_paths
            self.progress.emit(100)
            self.finished.emit(restored_images)
        except Exception as e:
            logger.exception("Error in ImageProcessor")
            self.error.emit(f"{type(e).__name__}: {e}")

class MRCPostProcessor(QThread):
    finished = pyqtSignal(str)
    error = pyqtSignal(str)
    progress = pyqtSignal(int)
    status = pyqtSignal(str)

    def __init__(self, template_mrc_path: str, output_image_dir: str, output_mrc_path: str, original_sizes: List[Tuple[int, int]] = None):
        super().__init__()
        self.template_mrc_path = template_mrc_path
        self.output_image_dir = output_image_dir
        self.output_mrc_path = output_mrc_path
        self.original_sizes = original_sizes
        self._running = True

    def stop(self) -> None:
        """Stop the processing thread."""
        self._running = False

    def run(self) -> None:
        try:
            self.status.emit("Reading template MRC header...")
            template_header = read_mrc_header(self.template_mrc_path)
            self.progress.emit(25)

            self.status.emit("Creating MRC from output images...")
            success = create_mrc_from_images(
                self.output_image_dir, 
                self.output_mrc_path, 
                self.template_mrc_path
            )
            self.progress.emit(75)

            if not self._running:
                return

            if success:
                self.status.emit("MRC file created successfully")
                self.progress.emit(100)
                self.finished.emit(self.output_mrc_path)
            else:
                raise Exception("Failed to create MRC file")
        except Exception as e:
            logger.error(f"Error in MRCPostProcessor: {str(e)}")
            self.error.emit(str(e))

class ComparisonWidget(QWidget):
    def __init__(self, original_path: str, generated_path: str, theme: str, parent=None):
        super().__init__(parent)
        self.theme = theme
        self.image_width = 700
        self.image_height = 700
        try:
            self.original_pixmap = QPixmap(original_path).scaled(
                self.image_width, self.image_height, Qt.AspectRatioMode.KeepAspectRatio
            )
            self.generated_pixmap = QPixmap(generated_path).scaled(
                self.image_width, self.image_height, Qt.AspectRatioMode.KeepAspectRatio
            )
        except Exception as e:
            logger.error(f"Error loading images for comparison: {str(e)}")
            raise

        self.split_position = 0
        self.setup_ui()

    def setup_ui(self) -> None:
        """Set up the comparison widget UI."""
        self.original_label = QLabel("Original")
        self.original_label.setFont(QFont("Segoe UI", 18, QFont.Weight.Bold))
        self.generated_label = QLabel("Generated")
        self.generated_label.setFont(QFont("Segoe UI", 18, QFont.Weight.Bold))

        self.slider = QSlider(Qt.Orientation.Horizontal, self)
        self.slider.setRange(0, self.image_width)
        self.slider.setValue(0)
        self.slider.valueChanged.connect(self.update_split)

        layout = QVBoxLayout()
        header_layout = QHBoxLayout()
        header_layout.addWidget(self.original_label)
        header_layout.addStretch()
        header_layout.addWidget(self.generated_label)

        layout.addLayout(header_layout)
        layout.addSpacing(25)
        layout.addStretch(1)
        layout.addWidget(self.slider)
        self.setLayout(layout)
        self.setMinimumSize(self.image_width, self.image_height + 150)
        self.update_style()

    def update_style(self) -> None:
        """Update widget style based on theme."""
        style = STYLES[self.theme]
        self.setStyleSheet(f"""
            background-color: {style['panel']};
            border: 1px solid {style['border']};
            border-radius: 12px;
            padding: 20px;
            box-shadow: {style['shadow']};
        """)
        for label in [self.original_label, self.generated_label]:
            label.setStyleSheet(f"color: {style['text']}; padding: 10px;")
        self.slider.setStyleSheet(f"""
            QSlider::groove:horizontal {{
                height: 16px;
                background: {style['border']};
                border-radius: 8px;
            }}
            QSlider::handle:horizontal {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, 
                                          stop:0 {style['button']}, 
                                          stop:1 {style['button_hover']});
                width: 32px;
                height: 32px;
                border-radius: 16px;
                margin: -8px 0;
                border: 2px solid {style['panel']};
            }}
            QSlider::handle:horizontal:hover {{
                background: {style['button_hover']};
            }}
        """)

    def update_split(self, value: int) -> None:
        """Update the split position for image comparison."""
        self.split_position = value
        self.update()

    def paintEvent(self, event) -> None:
        """Custom paint event for image comparison."""
        try:
            painter = QPainter(self)
            image_y = 60
            painter.drawPixmap(0, image_y, self.image_width, self.image_height, self.generated_pixmap)
            painter.setClipRect(self.split_position, image_y, self.image_width, self.image_height)
            painter.drawPixmap(0, image_y, self.image_width, self.image_height, self.original_pixmap)
            painter.setClipping(False)

            pen = QPen(QColor(STYLES[self.theme]['button']), 5, Qt.PenStyle.DashLine)
            pen.setDashPattern([6, 6])
            painter.setPen(pen)
            painter.drawLine(self.split_position, image_y, self.split_position, image_y + self.image_height)
        except Exception as e:
            logger.error(f"Error in paintEvent: {str(e)}")

class ComparisonDialog(QDialog):
    def __init__(self, original_path: str, generated_path: str, theme: str, parent=None):
        super().__init__(parent)
        self.theme = theme
        self.setWindowTitle("Image Comparison")
        self.setModal(False)
        self.setup_ui(original_path, generated_path)

    def setup_ui(self, original_path: str, generated_path: str) -> None:
        """Set up the comparison dialog UI."""
        try:
            main_layout = QVBoxLayout()
            main_layout.setContentsMargins(30, 30, 30, 30)
            main_layout.setSpacing(30)

            self.comparison_widget = ComparisonWidget(original_path, generated_path, self.theme)
            main_layout.addWidget(self.comparison_widget)

            self.close_btn = QPushButton("Close")
            self.close_btn.setFont(QFont("Segoe UI", 16))
            self.close_btn.clicked.connect(self.close)
            btn_layout = QHBoxLayout()
            btn_layout.addStretch()
            btn_layout.addWidget(self.close_btn)
            btn_layout.addStretch()
            main_layout.addLayout(btn_layout)

            self.setLayout(main_layout)
            self.update_style()
            self.resize(760, 860)
        except Exception as e:
            logger.error(f"Error setting up ComparisonDialog: {str(e)}")
            raise

    def update_style(self) -> None:
        """Update dialog style based on theme."""
        style = STYLES[self.theme]
        self.setStyleSheet(f"""
            QDialog {{
                background: {style['background']};
                border: 1px solid {style['border']};
                border-radius: 15px;
            }}
        """)
        self.close_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {style['button']};
                color: white;
                padding: 14px 35px;
                border-radius: 10px;
                font-weight: bold;
                border: none;
            }}
            QPushButton:hover:!pressed {{
                background-color: {style['button_hover']};
            }}
            QPushButton:pressed {{
                background-color: {style['accent']};
            }}
        """)


class CollapsibleSection(QWidget):
    """A collapsible container with a header button and a content area."""
    def __init__(self, title: str, icon: str = "", parent=None, start_collapsed: bool=False):
        super().__init__(parent)
        self._content = QWidget()
        self._content_layout = QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(0, 0, 0, 0)
        self._content_layout.setSpacing(10)

        self.toggle_btn = QToolButton(text=f" {title}" if icon else title, checkable=True, checked=not start_collapsed)
        self.toggle_btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.toggle_btn.setArrowType(Qt.ArrowType.DownArrow if not start_collapsed else Qt.ArrowType.RightArrow)
        self.toggle_btn.clicked.connect(self._on_toggled)
        
        # Add icon if provided
        if icon:
            self.toggle_btn.setIcon(QIcon.fromTheme(icon))
            self.toggle_btn.setIconSize(QSize(16, 16))

        self.anim = QPropertyAnimation(self._content, b"maximumHeight")
        self.anim.setDuration(200)
        self.anim.setEasingCurve(QEasingCurve.Type.InOutCubic)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        lay.addWidget(self.toggle_btn)
        lay.addWidget(self._content)

        if start_collapsed:
            self._content.setMaximumHeight(0)

    def _on_toggled(self, checked: bool):
        self.toggle_btn.setArrowType(Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow)
        start = self._content.maximumHeight()
        self._content.setMaximumHeight(16777215)  # expand to get sizeHint
        end = self._content.sizeHint().height() if checked else 0
        self._content.setMaximumHeight(start)
        self.anim.stop()
        self.anim.setStartValue(start)
        self.anim.setEndValue(end)
        self.anim.start()

    def content_layout(self) -> QVBoxLayout:
        return self._content_layout

class ImageLabel(QLabel):
    clicked = pyqtSignal(str)

    def __init__(self, image_path: str, theme: str):
        super().__init__()
        self.image_path = image_path
        self.theme = theme
        self.scale = 1.0
        self.setup_ui()

    def setup_ui(self) -> None:
        """Set up the image label UI."""
        try:
            self.setPixmap(QPixmap(self.image_path).scaled(
                220, 220, Qt.AspectRatioMode.KeepAspectRatio
            ))
            self.setStyleSheet(f"border: 1px solid {STYLES[self.theme]['border']}; border-radius: 6px; padding: 6px;")
            self.setCursor(Qt.CursorShape.PointingHandCursor)
            self.setToolTip(os.path.basename(self.image_path))
        except Exception as e:
            logger.error(f"Error setting up ImageLabel: {str(e)}")
            raise

    def mousePressEvent(self, event) -> None:
        """Handle mouse press event."""
        self.clicked.emit(self.image_path)

    def enterEvent(self, event) -> None:
        """Handle mouse enter event."""
        self.scale = 1.05
        self.update_pixmap()

    def leaveEvent(self, event) -> None:
        """Handle mouse leave event."""
        self.scale = 1.0
        self.update_pixmap()

    def update_pixmap(self) -> None:
        """Update the pixmap with current scale."""
        try:
            pixmap = QPixmap(self.image_path).scaled(
                int(220 * self.scale), int(220 * self.scale), Qt.AspectRatioMode.KeepAspectRatio
            )
            self.setPixmap(pixmap)
        except Exception as e:
            logger.error(f"Error updating pixmap: {str(e)}")

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.config = load_config()
        self.theme = self.config["theme"]
        self.processor = None
        self.mrc_processor = None
        self.original_images: List[str] = []
        self.generated_images: List[str] = []
        self.image_pairs: Dict[str, str] = {}
        self.model_path = resource_path(os.path.join("exp", "model", "MitEM", "model_2562.pt"))
        self.sidebar_collapsed = self.config["sidebar_collapsed"]
        self.input_type = self.config["input_type"]
        self.enable_postprocessing = self.config["enable_postprocessing"]
        self.original_mrc_path = ""
        self.original_sizes = []  
        self.output_folder = ""  
        self.template_mrc_path_manual = ""  
        self.output_image_dir_manual = ""   
        self.setup_ui()

    def setup_ui(self) -> None:
        """Set up the main window UI."""
        try:
            self.setWindowTitle("DF5T - Advanced Image Processor")
            self.setGeometry(100, 100, 1600, 1000)
            
            # Set window icon
            if os.path.exists("df5t_icon.png"):
                self.setWindowIcon(QIcon("df5t_icon.png"))

            self.main_widget = QWidget()
            self.setCentralWidget(self.main_widget)
            self.main_layout = QHBoxLayout(self.main_widget)
            self.main_layout.setContentsMargins(0, 0, 0, 0)
            self.main_layout.setSpacing(0)

            # Sidebar
            self.sidebar = QWidget()
            self.sidebar_layout = QVBoxLayout(self.sidebar)
            self.sidebar_layout.setContentsMargins(15, 15, 15, 15)
            self.sidebar_layout.setSpacing(15)
            self.sidebar.setMinimumWidth(350)
            self.sidebar.setMaximumWidth(350 if not self.sidebar_collapsed else 60)

            header_layout = QHBoxLayout()
            # Add DF5T icon to title
            self.title = QLabel("🔬 DF5T")
            self.title.setFont(QFont("Segoe UI", 24, QFont.Weight.Bold))
            header_layout.addWidget(self.title)
            self.collapse_btn = QPushButton("➖" if not self.sidebar_collapsed else "➕")
            self.collapse_btn.setFont(QFont("Segoe UI", 14))
            self.collapse_btn.clicked.connect(self.toggle_sidebar)
            header_layout.addStretch()
            header_layout.addWidget(self.collapse_btn)
            self.sidebar_layout.addLayout(header_layout)

            self.theme_combo = QComboBox()
            self.theme_combo.addItems(["Light", "Dark", "High Contrast"])
            self.theme_combo.setCurrentText(self.theme.capitalize())
            self.theme_combo.currentTextChanged.connect(self.change_theme)
            self.sidebar_layout.addWidget(self.theme_combo)

            # Input type selection
            input_type_group = CollapsibleSection("Input Type", "folder", start_collapsed=False)
            input_type_group.setObjectName("Input Type")
            input_type_layout = QVBoxLayout()
            input_type_layout.setSpacing(8)
            input_type_group.content_layout().addLayout(input_type_layout)
            
            self.image_radio = QRadioButton("📷 Images (TIF, PNG, etc.)")
            self.mrc_radio = QRadioButton("📁 MRC File")
            
            self.input_type_group = QButtonGroup()
            self.input_type_group.addButton(self.image_radio, 0)
            self.input_type_group.addButton(self.mrc_radio, 1)
            
            if self.input_type == "images":
                self.image_radio.setChecked(True)
            else:
                self.mrc_radio.setChecked(True)
                
            self.input_type_group.buttonClicked.connect(self.on_input_type_changed)
            
            input_type_layout.addWidget(self.image_radio)
            input_type_layout.addWidget(self.mrc_radio)
            self.sidebar_layout.addWidget(input_type_group)

            input_group = CollapsibleSection("Input", "document-open", start_collapsed=False)
            input_group.setObjectName("Input")
            input_layout = QVBoxLayout()
            input_layout.setSpacing(8)
            input_group.content_layout().addLayout(input_layout)
            
            # MRC file selection (only visible when MRC is selected)
            self.mrc_frame = QFrame()
            mrc_frame_layout = QHBoxLayout(self.mrc_frame)
            self.mrc_label = QLabel("📄 No MRC file selected")
            self.mrc_label.setFont(QFont("Segoe UI", 12))
            mrc_btn = QPushButton("📂 Select MRC")
            mrc_btn.setFont(QFont("Segoe UI", 12))
            mrc_btn.clicked.connect(self.select_mrc_file)
            mrc_frame_layout.addWidget(self.mrc_label)
            mrc_frame_layout.addWidget(mrc_btn)
            self.mrc_frame.setVisible(self.input_type == "mrc")
            
            # Folder selection
            folder_layout = QHBoxLayout()
            self.folder_label = QLabel(
                "📁 No folder selected" if not self.config["last_folder"]
                else f"📁 {os.path.basename(self.config['last_folder'])}"
            )
            self.folder_label.setFont(QFont("Segoe UI", 12))
            folder_layout.addWidget(self.folder_label)
            folder_btn = QPushButton("📂 Browse")
            folder_btn.setFont(QFont("Segoe UI", 12))
            folder_btn.clicked.connect(self.select_folder)
            folder_layout.addWidget(folder_btn)
            
            input_layout.addWidget(self.mrc_frame)
            input_layout.addLayout(folder_layout)
            self.sidebar_layout.addWidget(input_group)

            controls_group = CollapsibleSection("Parameters", "preferences-system", start_collapsed=False)
            controls_group.setObjectName("Parameters")
            controls_layout = QGridLayout()
            controls_layout.setVerticalSpacing(8)
            controls_layout.setHorizontalSpacing(8)
            controls_group.content_layout().addLayout(controls_layout)

            label_style = """
                QLabel {
                    font: bold 12px 'Arial';
                    color: %(text)s;
                    min-width: 100px;
                    padding-right: 10px;
                }
            """ % STYLES[self.theme]

            # Task
            task_label = QLabel("🎯 Task:")
            task_label.setStyleSheet(label_style)
            controls_layout.addWidget(task_label, 0, 0)
            self.deg_combo = QComboBox()
            self.deg_combo.setFont(QFont("Segoe UI", 12))
            # Grouped order in UI:
            # Image Enhancement: deblur/deno/sr2 + adaptive
            # Image Restoration: inp/isotropic
            self.deg_combo.addItem("Deblur (Image Enhancement)", "deblur_em")
            self.deg_combo.addItem("Denoise (Image Enhancement)", "deno_em")
            self.deg_combo.addItem("Super-Resolution x2 (Image Enhancement)", "sr2")
            self.deg_combo.addItem("Adaptive (Image Enhancement Auto-Routing)", "adaptive")
            self.deg_combo.insertSeparator(4)
            self.deg_combo.addItem("Inpainting (Image Restoration)", "inp_em")
            self.deg_combo.addItem("Isotropic Completion (Image Restoration)", "isotropic_em")
            # Enhancement tasks should default to adaptive routing first.
            self.deg_combo.setCurrentIndex(3)
            self.deg_combo.setToolTip(
                "Image Enhancement tasks: Deblur, Denoise, Super-Resolution x2, Adaptive.\n"
                "Image Restoration tasks: Inpainting, Isotropic Completion.\n"
                "Adaptive routes only within enhancement tasks (deblur/denoise/sr).\n"
                "Pipeline: linear enhancement + SVD nonlinear degradation branch + foundation-model nonlinear enhancement + organic fusion."
            )
            controls_layout.addWidget(self.deg_combo, 0, 1, 1, 2)

            # Timesteps
            time_label = QLabel("⏱️ Timesteps:")
            time_label.setStyleSheet(label_style)
            controls_layout.addWidget(time_label, 1, 0)
            self.time_label = QLabel("30")
            self.time_label.setFont(QFont("Segoe UI", 12))
            self.time_slider = QSlider(Qt.Orientation.Horizontal)
            self.time_slider.setRange(10, 100)
            self.time_slider.setValue(30)
            self.time_slider.valueChanged.connect(
                lambda: self.time_label.setText(str(self.time_slider.value()))
            )
            controls_layout.addWidget(self.time_slider, 1, 1)
            controls_layout.addWidget(self.time_label, 1, 2)

            # Processing degree (sigma_0): larger = stronger degradation, stronger enhancement
            sigma_label = QLabel("Processing degree:")
            sigma_label.setStyleSheet(label_style)
            sigma_label.setToolTip(
                "Higher value means stronger restoration. In Adaptive mode the slider is multiplied (~1.55×) "
                "then capped at 1.55 as processing_degree. It sets per-task strength floors before fusion. "
                "routing_softmax_weights ≈ task choice; global_weights are fusion scales (floor 1.0, max 2.0). "
                "See per_task_local_strength in routing_*.json."
            )
            controls_layout.addWidget(sigma_label, 2, 0)
            self.sigma_label = QLabel("0.82")
            self.sigma_label.setFont(QFont("Segoe UI", 12))
            self.sigma_slider = QSlider(Qt.Orientation.Horizontal)
            self.sigma_slider.setRange(1, 100)
            self.sigma_slider.setValue(82)
            self.sigma_slider.setToolTip(
                "Higher = stronger. Default 0.82. Adaptive: UI ×~1.55, cap 1.55 → processing_degree; each branch "
                "uses boosted local_strength (floor ~0.45–1). Fusion weights up to 2.0; anchor to input reduced. "
                "routing_*.json: routing_softmax_weights vs global_weights; per_task_local_strength ∈ [0,1]."
            )
            self.sigma_slider.valueChanged.connect(
                lambda: self.sigma_label.setText(f"{self.sigma_slider.value()/100:.2f}")
            )
            controls_layout.addWidget(self.sigma_slider, 2, 1)
            controls_layout.addWidget(self.sigma_label, 2, 2)

            # Input: Light normalization (for Diffusion input only; no heavy membrane pipeline)
            input_norm_label = QLabel("📥 Input norm:")
            input_norm_label.setStyleSheet(label_style)
            input_norm_label.setToolTip("Apply light normalization (percentile + mild CLAHE) to images before Diffusion. Recommended on. Uncheck to use raw images.")
            controls_layout.addWidget(input_norm_label, 3, 0)
            self.membrane_checkbox = QCheckBox("Enable")
            self.membrane_checkbox.setFont(QFont("Segoe UI", 12))
            self.membrane_checkbox.setChecked(False)
            self.membrane_checkbox.setToolTip("Light normalization for Diffusion input only.")
            self.membrane_checkbox.stateChanged.connect(self.toggle_membrane_params)
            controls_layout.addWidget(self.membrane_checkbox, 3, 1, 1, 2)

            # Apply light enhancement to result (grayscale-only; no coloring)
            result_enhance_label = QLabel("📤 Result:")
            result_enhance_label.setStyleSheet(label_style)
            result_enhance_label.setToolTip("Apply light grayscale enhancement to restored images. Off by default for best fusion with model output.")
            controls_layout.addWidget(result_enhance_label, 4, 0)
            self.apply_light_result_checkbox = QCheckBox("Light enhance")
            self.apply_light_result_checkbox.setFont(QFont("Segoe UI", 12))
            self.apply_light_result_checkbox.setChecked(False)
            self.apply_light_result_checkbox.setToolTip("Optional light enhancement on result (no coloring).")
            controls_layout.addWidget(self.apply_light_result_checkbox, 4, 1, 1, 2)

            # Top Percent (hidden when light norm is used; kept for compatibility)
            self.top_percent_label = QLabel("📊 Top Percent:")
            self.top_percent_label.setStyleSheet(label_style)
            controls_layout.addWidget(self.top_percent_label, 5, 0)
            self.top_percent_slider = QSlider(Qt.Orientation.Horizontal)
            self.top_percent_slider.setRange(1, 100)
            self.top_percent_slider.setValue(50)
            self.top_percent_value_label = QLabel("50")
            self.top_percent_value_label.setFont(QFont("Segoe UI", 12))
            self.top_percent_slider.valueChanged.connect(
                lambda: self.top_percent_value_label.setText(str(self.top_percent_slider.value()))
            )
            controls_layout.addWidget(self.top_percent_slider, 5, 1)
            controls_layout.addWidget(self.top_percent_value_label, 5, 2)

            # Dispersion Ratio
            self.dispersion_label = QLabel("🔍 Dispersion:")
            self.dispersion_label.setStyleSheet(label_style)
            controls_layout.addWidget(self.dispersion_label, 6, 0)
            self.dispersion_slider = QSlider(Qt.Orientation.Horizontal)
            self.dispersion_slider.setRange(0, 100)
            self.dispersion_slider.setValue(20)
            self.dispersion_value_label = QLabel("0.2")
            self.dispersion_value_label.setFont(QFont("Segoe UI", 12))
            self.dispersion_slider.valueChanged.connect(
                lambda: self.dispersion_value_label.setText(f"{self.dispersion_slider.value()/100:.1f}")
            )
            controls_layout.addWidget(self.dispersion_slider, 6, 1)
            controls_layout.addWidget(self.dispersion_value_label, 6, 2)

            # Denoise Strength
            self.denoise_label = QLabel("🔇 Denoise:")
            self.denoise_label.setStyleSheet(label_style)
            controls_layout.addWidget(self.denoise_label, 7, 0)
            self.denoise_slider = QSlider(Qt.Orientation.Horizontal)
            self.denoise_slider.setRange(0, 100)
            self.denoise_slider.setValue(50)
            self.denoise_value_label = QLabel("0.005")
            self.denoise_value_label.setFont(QFont("Segoe UI", 12))
            self.denoise_slider.valueChanged.connect(
                lambda: self.denoise_value_label.setText(f"{self.denoise_slider.value()/10000:.3f}")
            )
            controls_layout.addWidget(self.denoise_slider, 7, 1)
            controls_layout.addWidget(self.denoise_value_label, 7, 2)

            controls_layout.setColumnStretch(0, 1)
            controls_layout.setColumnStretch(1, 2)
            controls_layout.setColumnStretch(2, 1)

            self.sidebar_layout.addWidget(controls_group)

            # Post-processing options (only for MRC files)
            self.postprocess_group = CollapsibleSection("Post-processing (MRC only)", "document-save", start_collapsed=True)
            self.postprocess_group.setObjectName("Post-processing (MRC only)")
            postprocess_layout = QVBoxLayout()
            postprocess_layout.setSpacing(8)
            self.postprocess_group.content_layout().addLayout(postprocess_layout)
            
            self.postprocess_check = QCheckBox("🔄 Enable MRC reconstruction")
            self.postprocess_check.setChecked(self.enable_postprocessing)
            self.postprocess_check.stateChanged.connect(self.on_postprocess_changed)
            self.postprocess_check.setEnabled(self.input_type == "mrc")
            postprocess_layout.addWidget(self.postprocess_check)

            template_layout = QHBoxLayout()
            self.template_label = QLabel("📄 No template MRC selected")
            self.template_label.setFont(QFont("Segoe UI", 12))
            template_btn = QPushButton("📂 Select Template MRC")
            template_btn.setFont(QFont("Segoe UI", 12))
            template_btn.clicked.connect(self.select_template_mrc_manual)
            template_layout.addWidget(self.template_label)
            template_layout.addWidget(template_btn)
            postprocess_layout.addLayout(template_layout)

            output_dir_layout = QHBoxLayout()
            self.output_dir_label = QLabel("📁 No output folder selected")
            self.output_dir_label.setFont(QFont("Segoe UI", 12))
            output_dir_btn = QPushButton("📂 Select Output Folder")
            output_dir_btn.setFont(QFont("Segoe UI", 12))
            output_dir_btn.clicked.connect(self.select_output_dir_manual)
            output_dir_layout.addWidget(self.output_dir_label)
            output_dir_layout.addWidget(output_dir_btn)
            postprocess_layout.addLayout(output_dir_layout)
            
            self.reconstruct_btn = QPushButton("🔄 Reconstruct MRC")
            self.reconstruct_btn.setFont(QFont("Segoe UI", 12))
            self.reconstruct_btn.clicked.connect(self.manual_reconstruct_mrc)
            self.reconstruct_btn.setEnabled(self.input_type == "mrc" and bool(self.template_mrc_path_manual) and bool(self.output_image_dir_manual))
            postprocess_layout.addWidget(self.reconstruct_btn)
            
            self.sidebar_layout.addWidget(self.postprocess_group)

            btn_layout = QHBoxLayout()
            self.process_btn = QPushButton("🚀 Process")
            self.process_btn.setFont(QFont("Segoe UI", 14))
            self.process_btn.clicked.connect(self.process_images)
            self.process_btn.setEnabled(bool(self.config["last_folder"]))
            btn_layout.addWidget(self.process_btn)
            self.cancel_btn = QPushButton("❌ Cancel")
            self.cancel_btn.setFont(QFont("Segoe UI", 14))
            self.cancel_btn.clicked.connect(self.cancel_processing)
            self.cancel_btn.setEnabled(False)
            btn_layout.addWidget(self.cancel_btn)
            self.sidebar_layout.addLayout(btn_layout)

            self.sidebar_layout.addStretch()

            # Main Content
            self.content_widget = QWidget()
            self.content_layout = QVBoxLayout(self.content_widget)
            self.content_layout.setContentsMargins(20, 20, 20, 20)
            self.content_layout.setSpacing(15)

            split_widget = QWidget()
            split_layout = QHBoxLayout(split_widget)
            split_layout.setSpacing(15)

            preview_group = QGroupBox("📸 Preview")
            preview_layout = QVBoxLayout(preview_group)
            self.preview_scroll = QScrollArea()
            self.preview_widget = QWidget()
            self.preview_layout = QGridLayout(self.preview_widget)
            self.preview_layout.setSpacing(10)
            self.preview_scroll.setWidget(self.preview_widget)
            self.preview_scroll.setWidgetResizable(True)
            preview_layout.addWidget(self.preview_scroll)
            split_layout.addWidget(preview_group, 1)

            results_group = QGroupBox("📊 Results")
            results_layout = QVBoxLayout(results_group)
            self.results_scroll = QScrollArea()
            self.results_widget = QWidget()
            self.results_layout = QGridLayout(self.results_widget)
            self.results_layout.setSpacing(10)
            self.results_scroll.setWidget(self.results_widget)
            self.results_scroll.setWidgetResizable(True)
            results_layout.addWidget(self.results_scroll)
            split_layout.addWidget(results_group, 1)

            self.content_layout.addWidget(split_widget, stretch=1)

            progress_group = QGroupBox("📈 Progress")
            progress_layout = QVBoxLayout(progress_group)
            self.progress_bar = QProgressBar()
            self.progress_bar.setFont(QFont("Segoe UI", 10))
            progress_layout.addWidget(self.progress_bar)
            self.status_log = QTextEdit()
            self.status_log.setReadOnly(True)
            self.status_log.setFont(QFont("Segoe UI", 10))
            self.status_log.setMaximumHeight(80)
            progress_layout.addWidget(self.status_log)
            self.adaptive_info = QTextEdit()
            self.adaptive_info.setReadOnly(True)
            self.adaptive_info.setFont(QFont("Consolas", 10))
            self.adaptive_info.setMinimumHeight(140)
            self.adaptive_info.setPlaceholderText(
                "Adaptive task routing insights will appear here:\n"
                "- detection features (SVD response)\n"
                "- raw routing scores\n"
                "- routing_softmax_weights (≈ probabilities) vs global_weights (fusion, floor 1, max 2)\n"
                "- selected tasks; routing_*.json has per_task_local_strength (0–1 into each branch)"
            )
            progress_layout.addWidget(self.adaptive_info)
            self.content_layout.addWidget(progress_group)

            self.main_layout.addWidget(self.sidebar)
            self.main_layout.addWidget(self.content_widget, stretch=1)

            
            # Status bar: show CPU/GPU info
            status = QStatusBar()
            device = "GPU" if torch.cuda.is_available() else "CPU"
            cuda_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
            status.showMessage(f"💻 Device: {device} | 🎮 GPU: {cuda_name}")
            self.setStatusBar(status)
            self.update_theme()
            self.toggle_membrane_params()
            if self.config["last_folder"]:
                self.input_folder = self.config["last_folder"]
                self.display_preview(self.input_folder)
        except Exception as e:
            logger.error(f"Error setting up MainWindow: {str(e)}")
            raise

    def on_input_type_changed(self, button):
        """Handle input type change."""
        self.input_type = "mrc" if button == self.mrc_radio else "images"
        self.mrc_frame.setVisible(self.input_type == "mrc")
        self.postprocess_check.setEnabled(self.input_type == "mrc")
        self.reconstruct_btn.setEnabled(self.input_type == "mrc" and bool(self.template_mrc_path_manual) and bool(self.output_image_dir_manual)) 
        
        # Clear current preview when switching types
        self.folder_label.setText("📁 No folder selected")
        self.mrc_label.setText("📄 No MRC file selected")
        self.original_images = []
        self.clear_preview()
        
        self.config["input_type"] = self.input_type
        save_config(self.config)

    def on_postprocess_changed(self, state):
        """Handle post-processing checkbox change."""
        self.enable_postprocessing = state == Qt.CheckState.Checked.value
        self.config["enable_postprocessing"] = self.enable_postprocessing
        save_config(self.config)

    def select_mrc_file(self):
        """Select MRC file for processing."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select MRC File", "", "MRC Files (*.mrc)"
        )
        if file_path:
            self.original_mrc_path = file_path
            self.mrc_label.setText(f"📄 {os.path.basename(file_path)}")
            
            # Create a temporary folder for MRC slices
            mrc_folder = os.path.splitext(file_path)[0] + "_slices"
            os.makedirs(mrc_folder, exist_ok=True)
            
            # Convert MRC to images
            self.status_log.append(f"Converting MRC file to images...")
            image_list = save_mrc_slices_as_images(file_path, mrc_folder)
            
            if image_list:
                self.input_folder = mrc_folder
                self.folder_label.setText(f"📁 {os.path.basename(mrc_folder)}")
                self.display_preview(mrc_folder)
                self.process_btn.setEnabled(True)
                self.config["last_folder"] = mrc_folder
                save_config(self.config)
                self.status_log.append(f"Converted {len(image_list)} slices from MRC file.")
            else:
                self.status_log.append("Failed to convert MRC file to images.")

    def clear_preview(self):
        """Clear the preview area."""
        for i in reversed(range(self.preview_layout.count())):
            widget = self.preview_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)

    def toggle_membrane_params(self) -> None:
        """When Input norm is on, hide sliders (not used by light path). When off, show them."""
        light_norm_on = self.membrane_checkbox.isChecked()
        for w in (
            getattr(self, "top_percent_slider", None),
            getattr(self, "top_percent_value_label", None),
            getattr(self, "dispersion_slider", None),
            getattr(self, "dispersion_value_label", None),
            getattr(self, "denoise_slider", None),
            getattr(self, "denoise_value_label", None),
        ):
            if w is not None:
                w.setVisible(not light_norm_on)
        for w in (self.top_percent_label, self.dispersion_label, self.denoise_label):
            w.setVisible(not light_norm_on)

    def update_theme(self) -> None:
        """Update the UI theme."""
        try:
            style = STYLES[self.theme]
            self.main_widget.setStyleSheet(f"background: {style['background']};")
            self.sidebar.setStyleSheet(f"background: {style['panel']}; border-right: 1px solid {style['border']};")

            for group in [
                self.sidebar.findChild(QWidget, "Input Type"),
                self.sidebar.findChild(QWidget, "Input"),
                self.sidebar.findChild(QWidget, "Parameters"),
                self.sidebar.findChild(QWidget, "Post-processing (MRC only)"),
                self.content_widget.findChild(QGroupBox, "Preview"),
                self.content_widget.findChild(QGroupBox, "Results"),
                self.content_widget.findChild(QGroupBox, "Progress")
            ]:
                if group:
                    group.setStyleSheet(f"""
                        QWidget {{
                            background-color: {style['panel']};
                            border: 1px solid {style['border']};
                            border-radius: 8px;
                            padding: 10px;
                            margin-top: 8px;
                            color: {style['text']};
                        }}
                        QToolButton {{
                            font-weight: 600;
                            font-size: 14px;
                            color: {style['text']};
                            border: none;
                            text-align: left;
                            padding: 4px 2px;
                        }}
                        QToolButton:hover {{
                            background-color: rgba(0,0,0,0.04);
                            border-radius: 4px;
                        }}
                        QGroupBox::title {{
                            subcontrol-origin: margin;
                            left: 10px;
                            padding: 0 6px;
                            font-weight: bold;
                        }}
                    """)

            for btn in [self.process_btn, self.cancel_btn, self.collapse_btn, self.reconstruct_btn]:
                btn.setStyleSheet(f"""
                    QPushButton {{
                        background-color: {style['button']};
                        color: white;
                        padding: 10px 20px;
                        border-radius: 6px;
                        font-weight: bold;
                        border: none;
                    }}
                    QPushButton:hover:!pressed {{
                        background-color: {style['button_hover']};
                    }}
                    QPushButton:pressed {{
                        background-color: {style['accent']};
                    }}
                    QPushButton:disabled {{
                        background-color: #b2bec3;
                    }}
                """)

            self.theme_combo.setStyleSheet(f"""
                QComboBox {{
                    background-color: {style['panel']};
                    border: 1px solid {style['border']};
                    padding: 6px;
                    border-radius: 4px;
                    color: {style['text']};
                    font-size: 12px;
                }}
                QComboBox::drop-down {{
                    border-left: 1px solid {style['border']};
                    width: 25px;
                }}
                QComboBox QAbstractItemView {{
                    background-color: {style['panel']};
                    color: {style['text']};
                    selection-background-color: {style['button']};
                    border: 1px solid {style['border']};
                }}
            """)

            self.membrane_checkbox.setStyleSheet(f"""
                QCheckBox {{
                    color: {style['text']};
                    font-size: 12px;
                }}
                QCheckBox::indicator {{
                    width: 16px;
                    height: 16px;
                    border: 1px solid {style['border']};
                    border-radius: 3px;
                    background-color: {style['panel']};
                }}
                QCheckBox::indicator:checked {{
                    background-color: {style['button']};
                    border: 1px solid {style['button_hover']};
                }}
            """)

            self.postprocess_check.setStyleSheet(f"""
                QCheckBox {{
                    color: {style['text']};
                    font-size: 12px;
                }}
                QCheckBox::indicator {{
                    width: 16px;
                    height: 16px;
                    border: 1px solid {style['border']};
                    border-radius: 3px;
                    background-color: {style['panel']};
                }}
                QCheckBox::indicator:checked {{
                    background-color: {style['button']};
                    border: 1px solid {style['button_hover']};
                }}
            """)

            slider_style = f"""
                QSlider::groove:horizontal {{
                    height: 6px;
                    background: {style['border']};
                    border-radius: 3px;
                }}
                QSlider::handle:horizontal {{
                    background: {style['button']};
                    width: 16px;
                    height: 16px;
                    border-radius: 8px;
                    margin: -5px 0;
                }}
                QSlider::handle:horizontal:hover {{
                    background: {style['button_hover']};
                }}
            """
            for slider in [
                self.time_slider, self.sigma_slider,
                self.top_percent_slider, self.dispersion_slider,
                self.denoise_slider
            ]:
                slider.setStyleSheet(slider_style)

            self.progress_bar.setStyleSheet(f"""
                QProgressBar {{
                    border: 1px solid {style['border']};
                    border-radius: 4px;
                    background-color: {style['panel']};
                    text-align: center;
                    color: {style['text']};
                    font-size: 10px;
                }}
                QProgressBar::chunk {{
                    background-color: qlineargradient(x1:0, y1:0, x2:1, y2:1, 
                                                    stop:0 {style['button']}, 
                                                    stop:1 {style['button_hover']});
                    border-radius: 3px;
                }}
            """)

            self.status_log.setStyleSheet(f"""
            QTextEdit {{
                background-color: {style['panel']};
                border: 1px solid {style['border']};
                border-radius: 4px;
                color: {style['text']};
                padding: 4px;
            }}
            """)

            for scroll in [self.preview_scroll, self.results_scroll]:
                scroll.setStyleSheet(f"""
                    QScrollArea {{
                        background-color: {style['panel']};
                        border: none;
                    }}
                    QScrollBar:vertical, QScrollBar:horizontal {{
                        background: {style['panel']};
                        border: 1px solid {style['border']};
                        border-radius: 3px;
                    }}
                    QScrollBar::handle {{
                        background: {style['button']};
                        border-radius: 3px;
                    }}
                    QScrollBar::handle:hover {{
                        background: {style['button_hover']};
                    }}
                """)
        except Exception as e:
            logger.error(f"Error updating theme: {str(e)}")

    def toggle_sidebar(self) -> None:
        """Toggle sidebar visibility."""
        try:
            self.sidebar_collapsed = not self.sidebar_collapsed
            self.collapse_btn.setText("➖" if not self.sidebar_collapsed else "➕")
            animation = QPropertyAnimation(self.sidebar, b"maximumWidth")
            animation.setDuration(300)
            animation.setStartValue(self.sidebar.maximumWidth())
            animation.setEndValue(350 if not self.sidebar_collapsed else 60)
            animation.setEasingCurve(QEasingCurve.Type.InOutQuad)
            animation.start()
            self.config["sidebar_collapsed"] = self.sidebar_collapsed
            save_config(self.config)
        except Exception as e:
            logger.error(f"Error toggling sidebar: {str(e)}")

    def change_theme(self, theme_name: str) -> None:
        """Change the application theme."""
        try:
            self.theme = theme_name.lower()
            self.update_theme()
            self.config["theme"] = self.theme
            save_config(self.config)
            # Update image labels in preview and results
            self.display_preview(self.input_folder)
            self.display_images(self.generated_images)
        except Exception as e:
            logger.error(f"Error changing theme: {str(e)}")

    def select_folder(self) -> None:
        """Open folder selection dialog."""
        try:
            folder = QFileDialog.getExistingDirectory(self, "Select Image Folder")
            if folder:
                self.folder_label.setText(f"📁 {os.path.basename(folder)}")
                self.input_folder = folder
                self.process_btn.setEnabled(True)
                self.status_log.append(f"Folder selected: {folder}")
                self.display_preview(folder)
                self.config["last_folder"] = folder
                save_config(self.config)
        except Exception as e:
            logger.error(f"Error selecting folder: {str(e)}")
            self.status_log.append(f"Error selecting folder: {str(e)}")

    def display_preview(self, folder: str) -> None:
        """Display preview images."""
        try:
            self.clear_preview()

            valid_extensions = [".png"]
            self.original_images = [
                os.path.join(folder, f) for f in os.listdir(folder)
                if any(f.lower().endswith(ext) for ext in valid_extensions)
            ]
            images = natsorted(self.original_images[:8])
            captions: List[Tuple[str, str]] = [(img, os.path.splitext(os.path.basename(img))[0][:15]) for img in images]

            zstack_path = detect_single_multipage_tiff(folder)
            if not images and zstack_path:
                stack, _meta = load_z_stack(zstack_path)
                stack, _, _ = resize_z_stack_hw(stack)
                mid = int(stack.shape[0]) // 2
                preview_png = os.path.join(folder, "_df5t_zstack_preview_slice.png")
                cv2.imwrite(preview_png, stack[mid])
                self.original_images = [preview_png]
                nz = int(stack.shape[0])
                captions = [(preview_png, f"Z-mid ({nz})")]

            for idx, (img, caption_text) in enumerate(captions):
                label = ImageLabel(img, self.theme)
                caption = QLabel(caption_text)
                caption.setFont(QFont("Segoe UI", 10))
                caption.setStyleSheet(f"color: {STYLES[self.theme]['text']};")
                self.preview_layout.addWidget(label, idx // 4, idx % 4)
                self.preview_layout.addWidget(
                    caption, idx // 4 + 1, idx % 4, alignment=Qt.AlignmentFlag.AlignCenter
                )
        except Exception as e:
            logger.error(f"Error displaying preview: {str(e)}")
            self.status_log.append(f"Error displaying preview: {str(e)}")

    def process_images(self) -> None:
        """Start image processing."""
        try:
            self.process_btn.setEnabled(False)
            self.cancel_btn.setEnabled(True)
            self.adaptive_info.clear()
            selected_task = self.deg_combo.currentData()
            if not selected_task:
                selected_task = "deblur_em"
            processing_degree = self.sigma_slider.value() / 100
            if selected_task == "adaptive":
                processing_degree = float(np.clip(processing_degree * 1.55, 0.0, 1.55))
            self.processor = ImageProcessor(
                self.input_folder,
                selected_task,
                self.time_slider.value(),
                processing_degree,
                self.model_path,
                use_membrane_enhancement=self.membrane_checkbox.isChecked(),
                top_percent=self.top_percent_slider.value(),
                dispersion_ratio=self.dispersion_slider.value()/100,
                denoise_strength=self.denoise_slider.value()/10000,
                apply_enhancement_to_output=self.apply_light_result_checkbox.isChecked(),
            )
            self.processor.finished.connect(self.on_processing_finished)
            self.processor.error.connect(self.show_error)
            self.processor.progress.connect(self.update_progress)
            self.processor.status.connect(self.update_status)
            self.processor.finished.connect(self.processor.deleteLater)
            self.processor.start()
        except Exception as e:
            logger.error(f"Error starting image processing: {str(e)}")
            self.status_log.append(f"Error starting processing: {str(e)}")
            self.process_btn.setEnabled(True)
            self.cancel_btn.setEnabled(False)

    def on_processing_finished(self, image_paths: List[str]) -> None:

        self.display_images(image_paths)
        self._apply_zstack_manifest_pairing()
        self.output_folder = os.path.join(self.input_folder, "output")
        self._load_adaptive_routing_report()
        man_path = os.path.join(self.input_folder, "output", "zstack_manifest.json")
        if os.path.isfile(man_path):
            try:
                with open(man_path, "r", encoding="utf-8") as f:
                    man = json.load(f)
                if man.get("mode") == "zstack":
                    zd = int(man.get("z_depth", 0))
                    self.status_log.append(
                        f"Z-stack volume saved ({zd} slices): linear/svd_degraded/nonlinear/final and {man.get('stem')}_-1.tif in output."
                    )
            except Exception as e:
                logger.warning("Could not read zstack_manifest: %s", e)
        self.status_log.append("Image processing finished. You can now manually reconstruct MRC if needed.")

    def _apply_zstack_manifest_pairing(self) -> None:
        """Pair middle-slice outputs with middle-slice input preview for Z-stack runs."""
        man_path = os.path.join(self.input_folder, "output", "zstack_manifest.json")
        if not os.path.isfile(man_path):
            return
        try:
            with open(man_path, "r", encoding="utf-8") as f:
                man = json.load(f)
            if man.get("mode") != "zstack":
                return
            inp = man.get("input_mid_png")
            if isinstance(inp, str) and os.path.isfile(inp):
                for gen in list(self.generated_images):
                    self.image_pairs[gen] = inp
                self.original_images = [inp]
        except Exception as e:
            logger.warning("Z-stack manifest pairing failed: %s", e)

    def _load_adaptive_routing_report(self) -> None:
        """Load and show adaptive routing diagnostics if report exists."""
        try:
            report_path = os.path.join(self.input_folder, "output", "adaptive_routing_report.json")
            if not os.path.exists(report_path):
                # Newer adaptive path writes per-image routing_{idx}.json. Use routing_0.json as summary.
                fallback = os.path.join(self.input_folder, "output", "routing_0.json")
                zmid = os.path.join(self.input_folder, "output", "routing_vol_mid.json")
                if os.path.exists(fallback):
                    report_path = fallback
                elif os.path.exists(zmid):
                    report_path = zmid
                else:
                    if self.deg_combo.currentData() == "adaptive":
                        self.adaptive_info.setPlainText("Adaptive run finished, but no routing report was found.")
                    else:
                        self.adaptive_info.setPlainText("Routing report is shown for adaptive runs.")
                    return
            with open(report_path, "r", encoding="utf-8") as f:
                report = json.load(f)
            # Support both legacy aggregate report and per-image routing report.
            num_records = int(report.get("num_records", 1))
            sel_count = dict(report.get("task_selection_count", {}))
            avg_weights = dict(report.get("average_global_weights", {})) or dict(report.get("global_weights", {}))
            softmax_weights = dict(report.get("routing_softmax_weights", {}))
            fusion_weights = dict(report.get("global_weights", {})) or dict(avg_weights)
            restoration_diag = report.get("restoration_diag") if isinstance(report.get("restoration_diag"), dict) else {}
            records = list(report.get("records", []) or [])
            svd_scores = dict(report.get("svd_scores", {}))
            svd_response = dict(report.get("svd_response", {}))
            raw_scores = dict(report.get("raw_scores", {}))
            selected_tasks = list(report.get("selected_tasks", []) or [])
            metrics = dict(report.get("metrics", {}))
            timing = dict(report.get("timing", {}))
            if num_records <= 0:
                num_records = 1
            # Per-image routing_*.json has no aggregate selection counts.
            if not sel_count and selected_tasks:
                sel_count = {str(t): 1 for t in selected_tasks}

            all_tasks = set(sel_count.keys()) | set(avg_weights.keys()) | set(softmax_weights.keys()) | set(fusion_weights.keys())
            for rec in records:
                for task in rec.get("selected_tasks", []) or []:
                    all_tasks.add(str(task))
                for task in (rec.get("global_weights", {}) or {}).keys():
                    all_tasks.add(str(task))
            for task in svd_scores.keys():
                all_tasks.add(str(task))
            for task in svd_response.keys():
                all_tasks.add(str(task))
            for task in raw_scores.keys():
                all_tasks.add(str(task))
            for task in selected_tasks:
                all_tasks.add(str(task))
            sort_w = softmax_weights if softmax_weights else fusion_weights
            sorted_tasks = sorted(
                all_tasks,
                key=lambda t: (
                    float(sort_w.get(t, 0.0)),
                    int(sel_count.get(t, 0)),
                    t,
                ),
                reverse=True,
            )

            lines = []
            lines.append("Adaptive Routing Summary")
            lines.append("=" * 40)
            lines.append(f"Patch records: {num_records}")
            if timing:
                lines.append(
                    "Timing: "
                    + ", ".join(
                        [
                            f"{k}={float(v):.3f}s"
                            for k, v in timing.items()
                            if isinstance(v, (int, float))
                        ]
                    )
                )
            lines.append("")
            lines.append("Global metrics:")
            if metrics:
                lines.append(
                    "  - "
                    + ", ".join(
                        [f"{k}={float(v):.4f}" for k, v in sorted(metrics.items()) if isinstance(v, (int, float))]
                    )
                )
            else:
                lines.append("  - (no global metrics)")
            lines.append("")
            lines.append("1) Detection features (SVD response):")
            if svd_response:
                for task in sorted(svd_response.keys()):
                    resp = svd_response.get(task, {}) or {}
                    gap = float(resp.get("gap", 0.0))
                    tail = float(resp.get("tail", 0.0))
                    support_inv = float(resp.get("support_inv", 0.0))
                    rank_n = float(resp.get("rank_n", 0.0))
                    score = float(svd_scores.get(task, 0.0))
                    lines.append(
                        f"  - {task:<10} | gap={gap:.4f}, tail={tail:.4f}, support_inv={support_inv:.4f}, rank_n={rank_n:.4f} | svd_score={score:.4f}"
                    )
            elif svd_scores:
                for task, score in sorted(svd_scores.items(), key=lambda kv: kv[1], reverse=True):
                    lines.append(f"  - {task:<10} | svd_score={float(score):.4f}")
            else:
                lines.append("  - (no svd response records)")
            lines.append("")
            lines.append("2) Raw routing scores:")
            if raw_scores:
                for task, score in sorted(raw_scores.items(), key=lambda kv: kv[1], reverse=True):
                    lines.append(f"  - {task:<10} | raw_score={float(score):.4f}")
            else:
                lines.append("  - (no raw score records)")
            lines.append("")
            lines.append("3) routing_softmax_weights (selection, sum≈1):")
            if softmax_weights:
                for task, w in sorted(softmax_weights.items(), key=lambda kv: kv[1], reverse=True):
                    lines.append(f"  - {task:<10} | softmax={float(w):.4f}")
            else:
                lines.append("  - (not stored in this report; older runs only have global_weights)")
            lines.append("")
            lines.append("   global_weights (fusion scale, floor≈1.0, max 2.0):")
            if fusion_weights:
                for task, w in sorted(fusion_weights.items(), key=lambda kv: kv[1], reverse=True):
                    lines.append(f"  - {task:<10} | fusion_w={float(w):.4f}")
            else:
                lines.append("  - (no fusion weight records)")
            if restoration_diag:
                lines.append("")
                pd_pass = restoration_diag.get("processing_degree_passed")
                pts = restoration_diag.get("per_task_local_strength") or {}
                if pd_pass is not None:
                    lines.append(f"   processing_degree_passed={float(pd_pass):.4f}")
                if isinstance(pts, dict) and pts:
                    lines.append("   per_task_local_strength (into each branch, 0–1):")
                    for tk, v in sorted(pts.items(), key=lambda kv: str(kv[0])):
                        if isinstance(v, (int, float)):
                            lines.append(f"     - {tk:<10} | local_strength={float(v):.4f}")
            lines.append("")
            lines.append("4) Selected tasks and reasons:")
            if sorted_tasks:
                for task in sorted_tasks:
                    count = int(sel_count.get(task, 0))
                    ratio = (count / num_records * 100.0) if num_records > 0 else 0.0
                    smw = float(softmax_weights.get(task, 0.0)) if softmax_weights else 0.0
                    fw = float(fusion_weights.get(task, avg_weights.get(task, 0.0)))
                    thr_src = smw if softmax_weights else fw
                    selected_flag = "selected" if (task in selected_tasks or count > 0) else "not selected"
                    reason_parts = []
                    if task in selected_tasks:
                        reason_parts.append("in_current_selected")
                    if count > 0:
                        reason_parts.append("selected_in_patch_records")
                    if softmax_weights and thr_src >= 0.18:
                        reason_parts.append("softmax>=0.18")
                    elif (not softmax_weights) and thr_src >= 0.18:
                        reason_parts.append("weight>=0.18")
                    if not reason_parts:
                        reason_parts.append("below_threshold_or_not_ranked")
                    wtxt = f"softmax={smw:.4f} fusion={fw:.4f}" if softmax_weights else f"fusion_w={fw:.4f}"
                    lines.append(
                        f"  - {task:<10} | {selected_flag:<12} | count={count:>3}/{num_records:<3} ({ratio:5.1f}%) | {wtxt} | reason={'+'.join(reason_parts)}"
                    )
            else:
                lines.append("  - (no adaptive task records)")
            lines.append("")
            lines.append("Patch records (first 10):")
            for rec in records[:10]:
                pidx = int(rec.get("patch_idx", -1))
                mode = str(rec.get("routing_mode", "unknown"))
                selected = ",".join([str(x) for x in rec.get("selected_tasks", [])]) or "none"
                w = dict(rec.get("global_weights", {}))
                w_text = ", ".join([f"{k}:{float(v):.3f}" for k, v in sorted(w.items(), key=lambda kv: kv[1], reverse=True)])
                lines.append(f"  patch {pidx:>3} | mode={mode:<6} | selected=[{selected}] | weights=({w_text})")
            self.adaptive_info.setPlainText("\n".join(lines))
        except Exception as e:
            self.adaptive_info.setPlainText(f"Failed to load adaptive routing report: {e}")

    def select_template_mrc_manual(self):

        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Template MRC File", "", "MRC Files (*.mrc)"
        )
        if file_path:
            self.template_mrc_path_manual = file_path
            self.template_label.setText(f"📄 {os.path.basename(file_path)}")
            self.reconstruct_btn.setEnabled(bool(self.template_mrc_path_manual) and bool(self.output_image_dir_manual))

    def select_output_dir_manual(self):

        folder = QFileDialog.getExistingDirectory(self, "Select Output Images Folder")
        if folder:
            self.output_image_dir_manual = folder
            self.output_dir_label.setText(f"📁 {os.path.basename(folder)}")
            self.reconstruct_btn.setEnabled(bool(self.template_mrc_path_manual) and bool(self.output_image_dir_manual))

    def manual_reconstruct_mrc(self) -> None:
        try:
            if not self.template_mrc_path_manual or not self.output_image_dir_manual:
                self.status_log.append("Please select both template MRC and output folder first.")
                return
            

            output_mrc_path = os.path.join(os.path.dirname(self.output_image_dir_manual), "reconstructed.mrc")
            
            self.status_log.append(f"Starting manual MRC reconstruction...")
            self.mrc_processor = MRCPostProcessor(
                self.template_mrc_path_manual, 
                self.output_image_dir_manual, 
                output_mrc_path,
                self.original_sizes  
            )
            self.mrc_processor.finished.connect(self.on_mrc_postprocessing_finished)
            self.mrc_processor.error.connect(self.show_error)
            self.mrc_processor.progress.connect(self.update_progress)
            self.mrc_processor.status.connect(self.update_status)
            self.mrc_processor.start()
        except Exception as e:
            logger.error(f"Error in manual_reconstruct_mrc: {str(e)}")
            self.status_log.append(f"Error: {str(e)}")

    def on_mrc_postprocessing_finished(self, output_path: str) -> None:
        """Handle MRC post-processing finished event."""
        self.status_log.append(f"MRC file created: {output_path}")
        self.mrc_processor = None

    def cancel_processing(self) -> None:
        """Cancel ongoing processing."""
        try:
            if self.processor and self.processor.isRunning():
                self.processor.stop()
                self.processor.wait()
                self.status_log.append("Processing cancelled")
                self.process_btn.setEnabled(True)
                self.cancel_btn.setEnabled(False)
                self.processor = None
                
            if self.mrc_processor and self.mrc_processor.isRunning():
                self.mrc_processor.stop()
                self.mrc_processor.wait()
                self.status_log.append("MRC post-processing cancelled")
                self.mrc_processor = None
        except Exception as e:
            logger.error(f"Error cancelling processing: {str(e)}")
            self.status_log.append(f"Error cancelling processing: {str(e)}")

    def update_progress(self, value: int) -> None:
        """Update progress bar."""
        self.progress_bar.setValue(value)

    def update_status(self, message: str) -> None:
        """Update status log."""
        self.status_log.append(message)
        # For adaptive runs, mirror key routing/runtime signals into the adaptive panel live.
        try:
            if self.deg_combo.currentData() == "adaptive":
                if not hasattr(self, "_adaptive_live_lines"):
                    self._adaptive_live_lines = []
                live = self._adaptive_live_lines
                if any(
                    message.startswith(prefix)
                    for prefix in (
                        "Adaptive routing probe",
                        "Routing selected:",
                        "Loading diffusion backbone",
                        "Processing ",
                    )
                ):
                    live.append(message)
                    live[:] = live[-10:]
                    self.adaptive_info.setPlainText(
                        "Adaptive (live)\n"
                        + "=" * 40
                        + "\n"
                        + "\n".join([f"- {x}" for x in live])
                        + "\n\n(Report will populate here when finished.)"
                    )
        except Exception:
            pass

    def display_images(self, image_paths: List[str]) -> None:
        """Display processed images."""
        try:
            for i in reversed(range(self.results_layout.count())):
                widget = self.results_layout.itemAt(i).widget()
                if widget:
                    widget.setParent(None)

            self.generated_images = image_paths
            self.image_pairs.clear()

            sorted_originals = natsorted(self.original_images)
            sorted_generated = natsorted(image_paths)

            for idx, gen_path in enumerate(sorted_generated):
                if idx < len(sorted_originals):
                    self.image_pairs[gen_path] = sorted_originals[idx]

            for idx, img_path in enumerate(sorted_generated):
                label = ImageLabel(img_path, self.theme)
                label.clicked.connect(self.show_comparison)
                animation = QPropertyAnimation(label, b"pos")
                animation.setDuration(400)
                animation.setStartValue(QPoint(label.x(), label.y() - 30))
                animation.setEndValue(QPoint(label.x(), label.y()))
                animation.setEasingCurve(QEasingCurve.Type.OutBounce)
                animation.start()
                caption = QLabel("z-mid preview" if "_-1_mid" in img_path else os.path.basename(img_path)[:15])
                caption.setFont(QFont("Segoe UI", 10))
                caption.setStyleSheet(f"color: {STYLES[self.theme]['text']};")
                self.results_layout.addWidget(label, idx//4, idx%4)
                self.results_layout.addWidget(
                    caption, idx//4 + 1, idx%4, alignment=Qt.AlignmentFlag.AlignCenter
                )

            self.process_btn.setEnabled(True)
            self.cancel_btn.setEnabled(False)
        except Exception as e:
            logger.error(f"Error displaying images: {str(e)}")
            self.status_log.append(f"Error displaying images: {str(e)}")

    def show_comparison(self, generated_path: Optional[str] = None) -> None:
        """Show comparison dialog."""
        try:
            if not self.image_pairs:
                self.status_log.append("No images available for comparison")
                return

            if generated_path:
                original_path = self.image_pairs.get(generated_path)
                if not original_path:
                    self.status_log.append("No corresponding original image found")
                    return
            else:
                generated_path, original_path = next(iter(self.image_pairs.items()))

            dialog = ComparisonDialog(original_path, generated_path, self.theme, self)
            dialog.exec()
        except Exception as e:
            logger.error(f"Error showing comparison: {str(e)}")
            self.status_log.append(f"Error showing comparison: {str(e)}")

    def show_error(self, error_msg: str) -> None:
        """Display error message."""
        self.status_log.append(f"Error: {error_msg}")
        self.process_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)

def is_rgb_image(image: np.ndarray) -> bool:
    """Check if an image is RGB."""
    return len(image.shape) == 3 and image.shape[2] == 3

def enhance_contrast(image: np.ndarray, is_rgb: bool = False) -> np.ndarray:
    """Enhance image contrast using CLAHE, handling both RGB and grayscale."""
    try:
        if is_rgb:
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            lab = cv2.merge((l, a, b))
            return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        else:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            return clahe.apply(image)
    except Exception as e:
        logger.error(f"Error in enhance_contrast: {str(e)}")
        raise

def preprocess_image(image: np.ndarray, membrane_gray_min: int = 50, top_percent: int = 10, is_rgb: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Physics-respecting preprocessing for EM membrane analysis.

    This stage must not hallucinate deblurring. It only:
    - removes very low-frequency background bias;
    - estimates membrane support and bright-gap support;
    - returns a lightly normalized image plus a conservative membrane mask.
    """
    try:
        if image is None or image.size == 0:
            raise ValueError("Invalid image provided")

        src = image.copy()
        gray = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY) if is_rgb else src.copy()
        gray = gray.astype(np.uint8)

        bg = cv2.GaussianBlur(gray, (0, 0), 6.0)
        corrected = np.clip(gray.astype(np.float32) - 0.65 * (bg.astype(np.float32) - np.mean(bg)), 0, 255).astype(np.uint8)

        p1, p99 = np.percentile(corrected, [1.0, 99.0])
        if p99 <= p1 + 1:
            enhanced_gray = corrected
        else:
            enhanced_gray = np.clip((corrected.astype(np.float32) - p1) * (255.0 / (p99 - p1)), 0, 255).astype(np.uint8)

        gx = cv2.Scharr(enhanced_gray, cv2.CV_32F, 1, 0)
        gy = cv2.Scharr(enhanced_gray, cv2.CV_32F, 0, 1)
        edge_mag = cv2.magnitude(gx, gy)
        edge_mag = edge_mag / (edge_mag.max() + 1e-6)

        gray_f = enhanced_gray.astype(np.float32) / 255.0
        dark = 1.0 - gray_f
        lap = cv2.Laplacian(gray_f, cv2.CV_32F, ksize=3)
        left = np.pad(dark[:, :-1], ((0, 0), (1, 0)), mode='edge')
        right = np.pad(dark[:, 1:], ((0, 0), (0, 1)), mode='edge')
        up = np.pad(dark[:-1, :], ((1, 0), (0, 0)), mode='edge')
        down = np.pad(dark[1:, :], ((0, 1), (0, 0)), mode='edge')
        side_dark = np.maximum(np.minimum(left, right), np.minimum(up, down))
        local_bright = np.clip(gray_f - cv2.GaussianBlur(gray_f, (0, 0), 1.0), 0.0, 1.0)

        ridge_mask = (((dark > 0.22) & (edge_mag > 0.08)).astype(np.uint8) * 255)
        gap_mask = (((lap < -0.03) & (edge_mag > 0.05) & (side_dark > 0.20) & (local_bright > 0.01)).astype(np.uint8) * 255)

        k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        ridge_mask = cv2.morphologyEx(ridge_mask, cv2.MORPH_OPEN, k3, iterations=1)
        gap_mask = cv2.dilate(gap_mask, k3, iterations=1)
        membrane_mask = cv2.bitwise_or(ridge_mask, gap_mask)

        membrane_gray = cv2.bitwise_and(enhanced_gray, enhanced_gray, mask=membrane_mask)
        if is_rgb:
            enhanced = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2RGB)
        else:
            enhanced = enhanced_gray
        return enhanced, membrane_mask, membrane_gray
    except Exception as e:
        logger.error(f"Error in preprocess_image: {str(e)}")
        raise

def find_membranes_edges(membrane_mask: np.ndarray) -> np.ndarray:
    """Detect membrane edges."""
    try:
        edges = cv2.Canny(membrane_mask, 50, 80)
        return edges
    except Exception as e:
        logger.error(f"Error in find_membranes_edges: {str(e)}")
        raise

def enhance_membrane(
    image: np.ndarray,
    membrane_mask: np.ndarray,
    noise_reduction_level_1: float = 70,
    noise_enhance_level_2_3: float = 70,
    is_rgb: bool = False
) -> np.ndarray:
    """Enhance membrane regions, preserving RGB if needed."""
    try:
        if is_rgb:
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            membrane_pixels = l[membrane_mask > 0]
            if len(membrane_pixels) == 0:
                logger.warning("No membrane pixels found, returning original image")
                return image
            light_threshold = np.percentile(membrane_pixels, 90)
            dark_threshold = np.percentile(membrane_pixels, 10)

            light_pixels = l > light_threshold
            mid_dark_pixels = (l >= dark_threshold) & (l <= light_threshold)
            dark_pixels = l < dark_threshold

            enhanced_l = l.copy()
            enhanced_l[light_pixels] -= (enhanced_l[light_pixels] * noise_reduction_level_1 / 100)
            enhanced_l[mid_dark_pixels] += (255 - enhanced_l[mid_dark_pixels]) * noise_enhance_level_2_3 / 100
            enhanced_l[dark_pixels] += (255 - enhanced_l[dark_pixels]) * noise_enhance_level_2_3 / 100

            lab = cv2.merge((enhanced_l, a, b))
            return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        else:
            membrane_pixels = image[membrane_mask > 0]
            if len(membrane_pixels) == 0:
                logger.warning("No membrane pixels found, returning original image")
                return image
            light_threshold = np.percentile(membrane_pixels, 90)
            dark_threshold = np.percentile(membrane_pixels, 10)

            light_pixels = image > light_threshold
            mid_dark_pixels = (image >= dark_threshold) & (image <= light_threshold)
            dark_pixels = image < dark_threshold

            enhanced_image = image.copy()
            enhanced_image[light_pixels] -= (enhanced_image[light_pixels] * noise_reduction_level_1 / 100)
            enhanced_image[mid_dark_pixels] += (255 - enhanced_image[mid_dark_pixels]) * noise_enhance_level_2_3 / 100
            enhanced_image[dark_pixels] += (255 - enhanced_image[dark_pixels]) * noise_enhance_level_2_3 / 100

            return enhanced_image
    except Exception as e:
        logger.error(f"Error in enhance_membrane: {str(e)}")
        raise

def lighten_and_denoise(
    image: np.ndarray,
    mitochondria_mask: np.ndarray,
    denoise_strength: float = 0.005,
    is_rgb: bool = False
) -> np.ndarray:
    """Lighten background and apply denoising, preserving RGB if needed."""
    try:
        if not 0.0 <= denoise_strength <= 0.01:
            logger.warning("denoise_strength out of range, using 0.005")
            denoise_strength = 0.005

        if denoise_strength == 0:
            return image

        background_mask = cv2.bitwise_not(mitochondria_mask)
        if is_rgb:
            denoised = np.zeros_like(image)
            for c in range(3):
                channel = image[:, :, c]
                background = cv2.bitwise_and(channel, channel, mask=background_mask)
                denoised_channel = cv2.fastNlMeansDenoising(
                    background, None, h=10, templateWindowSize=7, searchWindowSize=21
                )
                lightened_channel = denoised_channel * (1 - denoise_strength) + 255 * denoise_strength
                denoised[:, :, c] = lightened_channel
            lightened_image = image.copy()
            lightened_image[background_mask > 0] = denoised[background_mask > 0]
            return lightened_image
        else:
            background = cv2.bitwise_and(image, image, mask=background_mask)
            denoised_background = cv2.fastNlMeansDenoising(
                background, None, h=10, templateWindowSize=7, searchWindowSize=21
                )
            lightened_background = denoised_background * (1 - denoise_strength) + 255 * denoise_strength
            lightened_image = image.copy()
            lightened_image[background_mask > 0] = lightened_background[background_mask > 0]
            return lightened_image
    except Exception as e:
        logger.error(f"Error in lighten_and_denoise: {str(e)}")
        raise

def process_mitochondria(
    image: np.ndarray,
    mitochondria_mask: np.ndarray,
    color_enhance_factor: float = 0.5,
    noise_compression_factor: float = 0.5,
    repair_gap_factor: float = 0.5,
    is_rgb: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    try:
        mitochondria_mask = (mitochondria_mask > 0).astype(np.uint8) * 255
        if is_rgb:
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            avg_gray = np.average(l[mitochondria_mask > 0]) if np.any(mitochondria_mask > 0) else np.average(l)
            logger.info(f"Average Gray Value (L channel): {avg_gray}")

            if avg_gray < 15:
                color_enhance_factor = 0.0005
            elif 15 <= avg_gray < 60:
                color_enhance_factor = 0.0004
            elif 60 <= avg_gray < 125:
                color_enhance_factor = 0.0003
            elif 125 <= avg_gray < 180:
                color_enhance_factor = 0.0002
            else:
                color_enhance_factor = 0.0001
            logger.info(f"Using color_enhance_factor: {color_enhance_factor}")

            enhanced_l = l.copy()
            enhanced_l[mitochondria_mask > 0] = np.clip(
                enhanced_l[mitochondria_mask > 0] - (enhanced_l[mitochondria_mask > 0] * color_enhance_factor),
                1, 254
            )
            lab = cv2.merge((enhanced_l, a, b))
            enhanced_image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            enhanced_mask = mitochondria_mask.copy()
            return enhanced_image, enhanced_mask
        else:
            avg_gray = np.average(mitochondria_mask)
            logger.info(f"Average Gray Value: {avg_gray}")

            if avg_gray < 15:
                color_enhance_factor = 0.0005
            elif 15 <= avg_gray < 60:
                color_enhance_factor = 0.0004
            elif 60 <= avg_gray < 125:
                color_enhance_factor = 0.0003
            elif 125 <= avg_gray < 180:
                color_enhance_factor = 0.0002
            else:
                color_enhance_factor = 0.0001
            logger.info(f"Using color_enhance_factor: {color_enhance_factor}")

            enhanced_mask = mitochondria_mask.copy()
            enhanced_mask[enhanced_mask > 0] = np.clip(
                enhanced_mask[enhanced_mask > 0] - (enhanced_mask[enhanced_mask > 0] * color_enhance_factor),
                1, 254
            )

            enhanced_image = image.copy()
            enhanced_image[enhanced_mask > 0] = np.clip(
                enhanced_image[enhanced_mask > 0] - (enhanced_image[enhanced_mask > 0] * color_enhance_factor),
                1, 254
            )

            enhanced_image = np.uint8(enhanced_image)
            return enhanced_image, enhanced_mask
    except Exception as e:
        logger.error(f"Error in process_mitochondria: {str(e)}")
        raise

def detect_membrane_regions_with_dense_noise(
    image: np.ndarray,
    membrane_mask: np.ndarray,
    window_size: int = 4,
    density_threshold: float = 0.5,
    dilation_iterations: int = 2,
    erosion_iterations: int = 2,
    min_cluster_size_ratio: float = 0.02,
    is_rgb: bool = False
) -> np.ndarray:
    """Detect dense membrane regions, using grayscale for processing."""
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if is_rgb else image
        height, width = membrane_mask.shape
        dense_mask = np.zeros_like(membrane_mask)
        dense_mask_before_morph = np.zeros_like(membrane_mask)
        membrane_mask_binary = (membrane_mask > 0).astype(int)
        noise_points = []
        window_area = (2 * window_size + 1) ** 2
        if window_area == 0:
            raise ValueError("Window size too small, causing division by zero")

        for y in range(height):
            for x in range(width):
                if membrane_mask_binary[y, x] > 0:
                    y_min = max(0, y - window_size)
                    y_max = min(height, y + window_size + 1)
                    x_min = max(0, x - window_size)
                    x_max = min(width, x + window_size + 1)
                    local_window = membrane_mask_binary[y_min:y_max, x_min:x_max]
                    local_density = np.sum(local_window)
                    density_ratio = local_density / window_area if window_area > 0 else 0
                    local_gray_value = gray[y, x]
                    dynamic_density_threshold = max(
                        density_threshold - (local_gray_value / 255.0) * 0.1, 0.3
                    )

                    if density_ratio > dynamic_density_threshold and local_gray_value > 0:
                        noise_points.append((y, x))

        for (y, x) in noise_points:
            dense_mask_before_morph[y, x] = 255

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
        dense_mask = cv2.dilate(dense_mask_before_morph, kernel, iterations=dilation_iterations)
        dense_mask = cv2.erode(dense_mask, kernel, iterations=erosion_iterations)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(dense_mask, connectivity=8)
        min_cluster_size = int(height * width * min_cluster_size_ratio)

        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] < min_cluster_size:
                dense_mask[labels == i] = 0

        final_mask = cv2.bitwise_and(membrane_mask, dense_mask)
        return final_mask
    except Exception as e:
        logger.error(f"Error in detect_membrane_regions_with_dense_noise: {str(e)}")
        raise

def setup_dataset_and_folder(input_folder: str, enhanced_images: List[Tuple[str, np.ndarray]]) -> Tuple[str, str]:
    """Set up dataset and folder for enhanced images."""
    try:
        dataset_dir = os.path.join(input_folder, "datasets", "MitEM", "MitEM")
        os.makedirs(dataset_dir, exist_ok=True)
        valid_files = []

        for filename, img in enhanced_images:
            if img is None or img.size == 0:
                logger.warning(f"Skipping invalid image: {filename}")
                continue
            dst_path = os.path.join(dataset_dir, filename)
            cv2.imwrite(dst_path, img)
            logger.info(f"Saved enhanced image {filename} to {dst_path}")
            valid_files.append(filename)

        if not valid_files:
            raise ValueError("No valid enhanced images to process")

        txt_path = os.path.join(input_folder, "MitEM_val_1k.txt")
        sorted_files = natsorted(valid_files)
        with open(txt_path, 'w') as f:
            for filename in sorted_files:
                name_without_extension = os.path.splitext(filename)[0]
                f.write(f"{name_without_extension} 1\n")
        return txt_path, dataset_dir
    except Exception as e:
        logger.error(f"Error in setup_dataset_and_folder: {str(e)}")
        raise

def process_and_color_membrane(
    image_path: str,
    membrane_gray_min: int = 50,
    top_percent: int = 10,
    density_threshold: float = 0.35,
    dispersion_ratio: float = 0.1,
    denoise_strength: float = 0.003,
    color_enhance_factor: float = 0.0001,
    noise_compression_factor: float = 0.2,
    window_size: int = 10,
    use_membrane_enhancement: bool = True,
    light_mode: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
    """
    When light_mode=True: only light normalization (percentile + mild CLAHE), no coloring.
    When light_mode=False: full pipeline with reduced defaults to avoid severe coloring.
    """
    try:
        image, scale_factor, original_size = resize_image_if_needed(image_path)
        if image is None:
            raise ValueError(f"Unable to load image at {image_path}")
        if not use_membrane_enhancement:
            return image, np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8), original_size
        if light_mode:
            out = light_display_enhancement(
                image, clip_percentile=(2.0, 98.0), mild_clahe=True, clahe_clip=0
            )
            return out, np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8), original_size

        denoise_strength = min(denoise_strength, 0.003)
        is_rgb = is_rgb_image(image)
        enhanced_image, membrane_mask, membrane_gray = preprocess_image(
            image, membrane_gray_min, top_percent, is_rgb=is_rgb
        )
        dense_mask = detect_membrane_regions_with_dense_noise(
            enhanced_image, membrane_mask, window_size=window_size,
            density_threshold=density_threshold, is_rgb=is_rgb
        )
        lightened_image = lighten_and_denoise(
            enhanced_image, dense_mask, denoise_strength=denoise_strength, is_rgb=is_rgb
        )
        dense_region = cv2.bitwise_and(enhanced_image, enhanced_image, mask=dense_mask)
        enhanced_image, refined_mask = process_mitochondria(
            dense_region, dense_mask, color_enhance_factor,
            noise_compression_factor, is_rgb=is_rgb
        )
        refined_mask_non_black = np.where(refined_mask > 0, refined_mask, 0)
        refined_mask_non_black_float = refined_mask_non_black.astype(float) / 255
        if is_rgb:
            final_image = (
                lightened_image.astype(float) * (1 - refined_mask_non_black_float[:, :, np.newaxis]) +
                enhanced_image.astype(float) * refined_mask_non_black_float[:, :, np.newaxis]
            ).astype(np.uint8)
        else:
            final_image = (
                lightened_image.astype(float) * (1 - refined_mask_non_black_float) +
                enhanced_image.astype(float) * refined_mask_non_black_float
            ).astype(np.uint8)
        return final_image, refined_mask_non_black, original_size
    except Exception as e:
        logger.error(f"Error in process_and_color_membrane: {str(e)}")
        raise

def process_images_in_folder(
    folder_path: str,
    membrane_gray_min: int = 1,
    top_percent: int = 10,
    density_threshold: float = 0.35,
    dispersion_ratio: float = 0.1,
    denoise_strength: float = 0.005,
    color_enhance_factor: float = 0.2,
    noise_compression_factor: float = 0.2,
    window_size: int = 10
) -> Tuple[List[Tuple[str, np.ndarray]], List[Tuple[int, int]]]:
    """Process all images in a folder, preserving RGB if needed."""
    try:
        enhanced_images = []
        original_sizes = []
        valid_extensions = ['.png']
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(tuple(valid_extensions)):
                image_path = os.path.join(folder_path, filename)
                try:
                    final_image, _, original_size = process_and_color_membrane(
                        image_path, 
                        membrane_gray_min, 
                        top_percent, 
                        density_threshold,
                        dispersion_ratio, 
                        denoise_strength, 
                        color_enhance_factor,
                        noise_compression_factor, 
                        window_size,
                        use_membrane_enhancement=True  
                    )
                    enhanced_images.append((filename, final_image))
                    original_sizes.append(original_size)
                except Exception as e:
                    logger.warning(f"Failed to process {filename}: {str(e)}")
                    continue
        if not enhanced_images:
            raise ValueError("No images were processed successfully")
        return enhanced_images, original_sizes
    except Exception as e:
        logger.error(f"Error in process_images_in_folder: {str(e)}")
        raise

def detect_and_color_dense_noise_points(
    image: np.ndarray,
    membrane_mask: np.ndarray,
    window_size: int = 30,
    dispersion_ratio: float = 0.1,
    noise_compression_factor: float = 0.3,
    is_rgb: bool = False
) -> np.ndarray:
    """Detect and color dense noise points, using grayscale for processing."""
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if is_rgb else image
        height, width = membrane_mask.shape
        dense_mask = np.zeros_like(membrane_mask)
        window_area = (2 * window_size + 1) ** 2
        if window_area == 0:
            raise ValueError("Window size too small, causing division by zero")

        for y in range(height):
            for x in range(width):
                if membrane_mask[y, x] > 0:
                    y_min = max(0, y - window_size)
                    y_max = min(height, y + window_size + 1)
                    x_min = max(0, x - window_size)
                    x_max = min(width, x + window_size + 1)
                    window = membrane_mask[y_min:y_max, x_min:x_max]
                    mask_pixels_in_window = np.sum(window > 0)
                    density = mask_pixels_in_window / window_area if window_area > 0 else 0

                    if density >= dispersion_ratio:
                        dense_mask[y, x] = membrane_mask[y, x]

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
        dense_mask = cv2.morphologyEx(dense_mask, cv2.MORPH_OPEN, kernel)

        dense_mask[membrane_mask == 0] = 0

        if noise_compression_factor > 0:
            dilation_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
            dense_mask = cv2.dilate(
                dense_mask, dilation_kernel, iterations=int(noise_compression_factor * 2)
            )
            erosion_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
            dense_mask = cv2.erode(
                dense_mask, erosion_kernel, iterations=int(noise_compression_factor * 2)
            )

        return dense_mask
    except Exception as e:
        logger.error(f"Error in detect_and_color_dense_noise_points: {str(e)}")
        raise

def main():
    """Main application entry point."""
    try:
        app = QApplication(sys.argv)
        app.setStyle("Fusion")
        app.setFont(QFont("Segoe UI", 11))
        window = MainWindow()
        window.show()
        sys.exit(app.exec())
    except Exception as e:
        logger.error(f"Error starting application: {str(e)}")
        sys.exit(1)


# ===== v26 stable preprocessing overrides =====
_prev_v26_prepare_input_for_diffusion = prepare_input_for_diffusion
_prev_v26_preprocess_image = preprocess_image

def _v26_sanitize_percentile_pair(pair):
    try:
        lo, hi = float(pair[0]), float(pair[1])
    except Exception:
        return 1.0, 99.0
    if 0.0 <= lo <= 1.0 and 0.0 <= hi <= 1.0:
        lo, hi = lo * 100.0, hi * 100.0
    lo = max(0.0, min(100.0, lo))
    hi = max(0.0, min(100.0, hi))
    if hi <= lo:
        lo, hi = 1.0, 99.0
    return lo, hi

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
    lo_p, hi_p = _v26_sanitize_percentile_pair(clip_percentile)
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
        if img.ndim == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            lo, hi = np.percentile(gray, [lo_p, hi_p])
            if hi <= lo + 1e-3:
                out = img.copy()
            else:
                scale = 255.0 / (hi - lo + 1e-6)
                out = np.clip((img.astype(np.float32) - lo) * scale, 0, 255).astype(np.uint8)
        else:
            lo, hi = np.percentile(img, [lo_p, hi_p])
            if hi <= lo + 1e-3:
                out = img.copy()
            else:
                out = np.clip((img.astype(np.float32) - lo) * (255.0 / (hi - lo + 1e-6)), 0, 255).astype(np.uint8)
        result.append((filename, out))
        original_sizes.append(original_size)
    return result, original_sizes

def preprocess_image(image: np.ndarray, membrane_gray_min: int = 50, top_percent: int = 10, is_rgb: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if image is None or image.size == 0:
        raise ValueError('Invalid image provided')
    src = image.copy()
    if src.ndim == 3:
        gray = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY if is_rgb else cv2.COLOR_BGR2GRAY)
    else:
        gray = src.copy()
    gray = gray.astype(np.uint8)
    # near-raw preview: only mild robust normalization if dynamic range is collapsed
    p1, p99 = np.percentile(gray, [1.0, 99.0])
    if p99 > p1 + 2.0:
        enhanced_gray = np.clip((gray.astype(np.float32) - p1) * (255.0 / (p99 - p1)), 0, 255).astype(np.uint8)
    else:
        enhanced_gray = gray
    gx = cv2.Scharr(enhanced_gray, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(enhanced_gray, cv2.CV_32F, 0, 1)
    edge = cv2.magnitude(gx, gy)
    edge = edge / (np.percentile(edge, 99.0) + 1e-6)
    edge = np.clip(edge, 0.0, 1.0)
    g = enhanced_gray.astype(np.float32) / 255.0
    dark = 1.0 - g
    ridge = ((dark > 0.18) & (edge > 0.06)).astype(np.uint8) * 255
    gap = ((cv2.Laplacian(g, cv2.CV_32F, ksize=3) < -0.02) & (edge > 0.05)).astype(np.uint8) * 255
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    membrane_mask = cv2.morphologyEx(cv2.bitwise_or(ridge, gap), cv2.MORPH_OPEN, k3, iterations=1)
    membrane_gray = cv2.bitwise_and(enhanced_gray, enhanced_gray, mask=membrane_mask)
    enhanced = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2RGB) if is_rgb else enhanced_gray
    return enhanced, membrane_mask, membrane_gray


if __name__ == "__main__":
    main()


# ===== v27 photometry-preserving preprocessing overrides =====
_prev_v27_prepare_input_for_diffusion = prepare_input_for_diffusion
_prev_v27_preprocess_image = preprocess_image

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
        # Keep photometry as close as possible to the loaded image.
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        result.append((filename, img.copy()))
        original_sizes.append(original_size)
    return result, original_sizes


def preprocess_image(image: np.ndarray, membrane_gray_min: int = 50, top_percent: int = 10, is_rgb: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if image is None or image.size == 0:
        raise ValueError('Invalid image provided')
    src = image.copy()
    if src.ndim == 3:
        gray = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY if is_rgb else cv2.COLOR_BGR2GRAY)
    else:
        gray = src.copy()
    gray = gray.astype(np.uint8)
    gx = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(gray, cv2.CV_32F, 0, 1)
    edge = cv2.magnitude(gx, gy)
    edge = edge / (np.percentile(edge, 99.0) + 1e-6)
    edge = np.clip(edge, 0.0, 1.0)
    g = gray.astype(np.float32) / 255.0
    lap = cv2.Laplacian(g, cv2.CV_32F, ksize=3)
    dark = 1.0 - g
    ridge = np.clip(0.65 * edge + 0.35 * np.clip(dark - cv2.GaussianBlur(dark, (0, 0), 1.0), 0.0, 1.0), 0.0, 1.0)
    gap = ((lap < -0.02).astype(np.float32) * edge)
    membrane_mask = ((0.58 * ridge + 0.42 * gap) > 0.16).astype(np.uint8) * 255
    membrane_mask = cv2.morphologyEx(membrane_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
    membrane_gray = cv2.bitwise_and(gray, gray, mask=membrane_mask)
    enhanced = src.copy() if src.ndim == 3 else gray.copy()
    return enhanced, membrane_mask, membrane_gray

# ===== v28 strictly photometry-preserving display/preprocess overrides =====
_prev_v28_prepare_input_for_diffusion = prepare_input_for_diffusion
_prev_v28_preprocess_image = preprocess_image
_prev_v28_light_display_enhancement = light_display_enhancement


def light_display_enhancement(
    image_path_or_array,
    clip_percentile: Tuple[float, float] = (2.0, 98.0),
    mild_clahe: bool = True,
    clahe_clip: float = 0.0,
    auxiliary_blend_max: float = 0.12,
) -> np.ndarray:
    """Display path must preserve original EM photometry; do not enhance by default."""
    if isinstance(image_path_or_array, str):
        image, _sf, _size = resize_image_if_needed(image_path_or_array)
        if image is None:
            raise ValueError(f"Unable to load image at {image_path_or_array}")
    else:
        image = np.asarray(image_path_or_array)
    if image is None or image.size == 0:
        raise ValueError("Empty image in light_display_enhancement")
    return image.copy().astype(np.uint8) if image.dtype != np.uint8 else image.copy()



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
        # Preserve OpenCV-native layout and grayscale semantics exactly.
        result.append((filename, img.copy()))
        original_sizes.append(original_size)
    return result, original_sizes



def preprocess_image(image: np.ndarray, membrane_gray_min: int = 50, top_percent: int = 10, is_rgb: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if image is None or image.size == 0:
        raise ValueError('Invalid image provided')
    src = np.asarray(image).copy()
    if src.ndim == 3:
        gray = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY if is_rgb else cv2.COLOR_BGR2GRAY)
    else:
        gray = src.copy()
    gray = gray.astype(np.uint8)
    g = gray.astype(np.float32) / 255.0
    gx = cv2.Scharr(g, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(g, cv2.CV_32F, 0, 1)
    edge = cv2.magnitude(gx, gy)
    edge = edge / (np.percentile(edge, 99.0) + 1e-6)
    edge = np.clip(edge, 0.0, 1.0)
    dark = 1.0 - g
    lap = cv2.Laplacian(g, cv2.CV_32F, ksize=3)
    ridge = np.clip(0.50 * edge + 0.35 * dark + 0.15 * np.clip(-lap, 0.0, 1.0), 0.0, 1.0)
    thr = float(np.percentile(ridge, max(55, min(95, 100 - top_percent))))
    membrane_mask = (ridge > thr).astype(np.uint8) * 255
    membrane_mask = cv2.morphologyEx(membrane_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
    membrane_mask = cv2.morphologyEx(membrane_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
    membrane_gray = cv2.bitwise_and(gray, gray, mask=membrane_mask)
    enhanced = src.copy() if src.ndim == 3 else gray.copy()
    return enhanced, membrane_mask, membrane_gray

# ===== V29 ET-friendly photometry-preserving overrides =====
def light_display_enhancement(image_path_or_array):
    if isinstance(image_path_or_array, str):
        image = cv2.imread(image_path_or_array, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"Unable to load image at {image_path_or_array}")
    else:
        image = np.asarray(image_path_or_array)
    if image is None or image.size == 0:
        raise ValueError("Empty image in light_display_enhancement")
    if image.ndim == 2:
        return image.astype(np.uint8, copy=True) if image.dtype != np.uint8 else image.copy()
    # Preserve exact channel semantics for UI preview.
    return image.astype(np.uint8, copy=True) if image.dtype != np.uint8 else image.copy()


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
        # No histogram changes, no CLAHE, no hidden normalization.
        result.append((filename, img.copy()))
        original_sizes.append(original_size)
    return result, original_sizes


def preprocess_image(image: np.ndarray, membrane_gray_min: int = 50, top_percent: int = 10, is_rgb: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if image is None or image.size == 0:
        raise ValueError('Invalid image provided')
    src = np.asarray(image).copy()
    if src.ndim == 3:
        gray = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY if is_rgb else cv2.COLOR_BGR2GRAY)
    else:
        gray = src.copy()
    gray = gray.astype(np.uint8)
    g = gray.astype(np.float32) / 255.0
    gx = cv2.Scharr(g, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(g, cv2.CV_32F, 0, 1)
    edge = cv2.magnitude(gx, gy)
    edge = np.clip(edge / (np.percentile(edge, 99.0) + 1e-6), 0.0, 1.0)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))).astype(np.float32) / 255.0
    dark = 1.0 - g
    mem_score = np.clip(0.42 * edge + 0.33 * blackhat + 0.25 * dark, 0.0, 1.0)
    thr = float(np.percentile(mem_score, max(70, min(96, 100 - int(top_percent)))))
    membrane_mask = (mem_score > thr).astype(np.uint8) * 255
    membrane_mask = cv2.morphologyEx(membrane_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
    membrane_mask = cv2.morphologyEx(membrane_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
    # Keep preview photometry identical; expose original gray for the second pane instead of a transformed image.
    enhanced = src.copy() if src.ndim == 3 else gray.copy()
    membrane_gray = gray.copy()
    return enhanced, membrane_mask, membrane_gray

# ===== V30 final active overrides =====
_prev_v30_light_display_enhancement = light_display_enhancement
_prev_v30_prepare_input_for_diffusion = prepare_input_for_diffusion
_prev_v30_preprocess_image = preprocess_image

def light_display_enhancement(image_path_or_array, *args, **kwargs):
    if isinstance(image_path_or_array, str):
        image = cv2.imread(image_path_or_array, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"Unable to load image at {image_path_or_array}")
    else:
        image = np.asarray(image_path_or_array)
    if image is None or image.size == 0:
        raise ValueError("Empty image in light_display_enhancement")
    return image.astype(np.uint8, copy=True) if image.dtype != np.uint8 else image.copy()


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
        result.append((filename, img.copy()))
        original_sizes.append(original_size)
    return result, original_sizes


def preprocess_image(image: np.ndarray, membrane_gray_min: int = 50, top_percent: int = 10, is_rgb: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if image is None or image.size == 0:
        raise ValueError('Invalid image provided')
    src = np.asarray(image).copy()
    if src.ndim == 3:
        gray = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY if is_rgb else cv2.COLOR_BGR2GRAY)
    else:
        gray = src.copy()
    gray = gray.astype(np.uint8)
    g = gray.astype(np.float32) / 255.0
    gx = cv2.Scharr(g, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(g, cv2.CV_32F, 0, 1)
    edge = cv2.magnitude(gx, gy)
    edge = np.clip(edge / (np.percentile(edge, 99.0) + 1e-6), 0.0, 1.0)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))).astype(np.float32) / 255.0
    dark = 1.0 - g
    mem_score = np.clip(0.48 * edge + 0.30 * blackhat + 0.22 * dark, 0.0, 1.0)
    thr = float(np.percentile(mem_score, max(78, min(96, 100 - int(top_percent)))))
    membrane_mask = (mem_score > thr).astype(np.uint8) * 255
    membrane_mask = cv2.morphologyEx(membrane_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
    membrane_mask = cv2.morphologyEx(membrane_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
    enhanced = src.copy() if src.ndim == 3 else gray.copy()
    membrane_gray = gray.copy()
    return enhanced, membrane_mask, membrane_gray


# ===== V31 strong inp + fine-grain overrides =====
_prev_v31_prepare_input_for_diffusion = prepare_input_for_diffusion
_prev_v31_preprocess_image = preprocess_image


def _v31_clip01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)


def _v31_small_image_factor_hw(height: int, width: int) -> float:
    return float(np.clip((320.0 - float(min(height, width))) / 192.0, 0.0, 1.0))


def _v31_gray_maps(gray: np.ndarray):
    g = gray.astype(np.float32) / 255.0
    gx = cv2.Scharr(g, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(g, cv2.CV_32F, 0, 1)
    edge = cv2.magnitude(gx, gy)
    edge = _v31_clip01(edge / (np.percentile(edge, 99.0) + 1e-6))
    dark = 1.0 - g
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))).astype(np.float32) / 255.0
    lap = cv2.Laplacian(g, cv2.CV_32F, ksize=3)
    gap = _v31_clip01(np.clip(-lap, 0.0, None) / (np.percentile(np.clip(-lap, 0.0, None), 99.0) + 1e-6))
    ridge = _v31_clip01(0.40 * dark + 0.34 * edge + 0.18 * blackhat + 0.08 * _v31_clip01(np.clip(lap, 0.0, None)))
    return g, dark, edge, blackhat, gap, ridge


def _v31_make_line_kernels(k: int):
    k = int(max(3, k))
    if k % 2 == 0:
        k += 1
    center = k // 2
    kernels = []
    for p1, p2 in [((0, center), (k - 1, center)), ((center, 0), (center, k - 1)), ((0, 0), (k - 1, k - 1)), ((0, k - 1), (k - 1, 0))]:
        ker = np.zeros((k, k), dtype=np.uint8)
        cv2.line(ker, p1, p2, 1, 1)
        kernels.append(ker)
    return kernels


def _v31_micro_detail(gray: np.ndarray, amount: float = 0.35) -> np.ndarray:
    h, w = gray.shape[:2]
    small = _v31_small_image_factor_hw(h, w)
    g, dark, edge, blackhat, gap, ridge = _v31_gray_maps(gray)
    src = gray.astype(np.float32)
    up = cv2.resize(src, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
    up_hp = up - cv2.GaussianBlur(up, (0, 0), 0.70 if small > 0.45 else 0.95)
    hp_up = cv2.resize(up_hp, (w, h), interpolation=cv2.INTER_AREA)
    hp_local = cv2.GaussianBlur(src, (0, 0), 0.65 if small > 0.45 else 0.85) - cv2.GaussianBlur(src, (0, 0), 1.45 if small > 0.45 else 1.80)
    detail = 0.58 * hp_up + 0.42 * hp_local
    detail = np.clip(detail, -30.0, 30.0)
    gate = _v31_clip01((0.46 * ridge + 0.26 * edge + 0.16 * blackhat + 0.12 * dark) * (1.0 - 0.55 * gap))
    gain = float((0.14 + 0.42 * amount) * (0.68 + 0.90 * small))
    out = src + gain * gate * detail
    return np.clip(out, 0, 255).astype(np.uint8)



def _v39_internal_blockiness_score(gray: np.ndarray) -> float:
    """
    Estimate how much an image behaves like a low-resolution image that was later enlarged.
    If the image is close to a nearest-neighbour upsampled reconstruction at multiple scales,
    we treat it as blocky even when its current pixel dimensions are large.
    """
    gray = np.asarray(gray)
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    src = gray.astype(np.float32)
    h, w = src.shape[:2]
    if min(h, w) < 24:
        return 1.0

    scores: List[float] = []
    for s in (2, 3, 4):
        hh = max(12, h // s)
        ww = max(12, w // s)
        if hh >= h or ww >= w:
            continue
        low = cv2.resize(src, (ww, hh), interpolation=cv2.INTER_AREA)
        near = cv2.resize(low, (w, h), interpolation=cv2.INTER_NEAREST)
        cubic = cv2.resize(low, (w, h), interpolation=cv2.INTER_CUBIC)
        err_near = float(np.mean(np.abs(src - near)))
        err_cubic = float(np.mean(np.abs(src - cubic)))
        scores.append(max(0.0, (err_cubic - err_near) / (err_cubic + 1e-6)))

    src01 = src / 255.0
    parity_groups = [src01[::2, ::2], src01[1::2, ::2], src01[::2, 1::2], src01[1::2, 1::2]]
    parity_means = [float(p.mean()) for p in parity_groups if p.size > 0]
    checker = (max(parity_means) - min(parity_means)) if parity_means else 0.0
    return float(np.clip((max(scores) if scores else 0.0) * 0.90 + 2.0 * checker, 0.0, 1.0))



def _v32_subpixel_granular_gray(gray: np.ndarray, amount: float = 0.50) -> np.ndarray:
    """
    Add gentle sub-pixel granularity for small or internally blocky inputs.
    This specifically targets low-resolution images that were enlarged beforehand,
    so the current HxW may look large while the content is still coarse.
    """
    h, w = gray.shape[:2]
    small = _v31_small_image_factor_hw(h, w)
    block = _v39_internal_blockiness_score(gray)
    drive = float(max(small, block))
    if drive <= 0.08:
        return gray

    src = gray.astype(np.float32)
    scale = 3 if drive > 0.45 else 2
    up = cv2.resize(src, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
    up_nn = cv2.resize(src, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)

    def _shift(img: np.ndarray, dx: float, dy: float) -> np.ndarray:
        mat = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
        return cv2.warpAffine(
            img,
            mat,
            (img.shape[1], img.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT101,
        )

    phase = 0.60 + 0.35 * drive
    mix = (
        0.42 * up
        + 0.16 * _shift(up_nn, 0.34 * phase, 0.16 * phase)
        + 0.14 * _shift(up_nn, -0.20 * phase, 0.38 * phase)
        + 0.14 * _shift(up_nn, 0.28 * phase, -0.30 * phase)
        + 0.14 * _shift(up, -0.26 * phase, -0.12 * phase)
    )
    back = cv2.resize(mix, (w, h), interpolation=cv2.INTER_AREA)
    detail = back - cv2.GaussianBlur(back, (0, 0), 0.50 if drive > 0.45 else 0.65)

    _g, dark, edge, blackhat, gap, ridge = _v31_gray_maps(gray)
    support = cv2.GaussianBlur(np.maximum(edge, ridge).astype(np.float32), (0, 0), 0.85 if drive > 0.45 else 1.05)
    flat = 1.0 - _v31_clip01(edge * 1.55)
    gate = _v31_clip01((0.40 * ridge + 0.24 * edge + 0.16 * blackhat + 0.12 * dark + 0.08 * support) * (1.0 - 0.45 * gap))
    gate = _v31_clip01(np.maximum(gate, (0.16 + 0.14 * amount) * block * flat))

    gain = float((0.05 + 0.16 * amount) * (0.42 + 0.92 * drive))
    out = src + gain * gate * np.clip(detail, -18.0, 18.0)
    if block > 0.12:
        hp = src - cv2.GaussianBlur(src, (0, 0), 0.85)
        out = out + (0.015 + 0.035 * amount) * block * flat * np.clip(hp, -8.0, 8.0)
    return np.clip(out, 0, 255).astype(np.uint8)

def _v31_bridge_gray(gray: np.ndarray, strength: float = 0.85) -> np.ndarray:
    h, w = gray.shape[:2]
    small = _v31_small_image_factor_hw(h, w)
    _g, dark, edge, blackhat, gap, ridge = _v31_gray_maps(gray)
    membrane_score = _v31_clip01(0.42 * ridge + 0.22 * edge + 0.20 * blackhat + 0.16 * dark)
    thr = float(np.percentile(membrane_score, 76.0 if small > 0.55 else 80.0))
    membrane = ((membrane_score >= thr) & (dark > 0.16) & (edge > 0.04)).astype(np.uint8) * 255
    membrane = cv2.morphologyEx(membrane, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
    membrane = cv2.morphologyEx(membrane, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)

    inv = 255 - gray
    bridge_mask = np.zeros_like(gray, dtype=np.uint8)
    bridge = gray.astype(np.float32)
    lengths = [7, 11, 15] if small > 0.55 else [9, 13, 17]
    max_len = max(5, (min(h, w) // 2) | 1)
    for k in lengths:
        k = int(min(k, max_len))
        if k % 2 == 0:
            k += 1
        for ker in _v31_make_line_kernels(k):
            closed_mask = cv2.morphologyEx(membrane, cv2.MORPH_CLOSE, ker, iterations=1)
            added = cv2.subtract(closed_mask, membrane)
            bridge_mask = np.maximum(bridge_mask, added)
            closed_inv = cv2.morphologyEx(inv, cv2.MORPH_CLOSE, ker, iterations=1)
            bridge = np.minimum(bridge, (255 - closed_inv).astype(np.float32))

    bridge_mask = cv2.dilate(bridge_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
    bridge_soft = cv2.GaussianBlur((bridge_mask > 0).astype(np.float32), (0, 0), 0.9 if small > 0.55 else 1.25)
    local_dark = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 0.8 if small > 0.55 else 1.1)
    candidate = np.minimum(bridge, 0.78 * local_dark + 0.22 * gray.astype(np.float32))

    blur = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 0.75 if small > 0.55 else 1.0)
    unsharp = gray.astype(np.float32) + (0.14 + 0.26 * strength) * (gray.astype(np.float32) - blur)
    membrane_soft = cv2.GaussianBlur((membrane > 0).astype(np.float32), (0, 0), 0.9 if small > 0.55 else 1.2)
    support = _v31_clip01(np.maximum(bridge_soft, 0.22 * membrane_soft))

    out = gray.astype(np.float32) * (1.0 - 0.10 * support) + unsharp * (0.10 * support)
    w_bridge = np.clip((0.85 + 0.30 * strength) * bridge_soft, 0.0, 1.0)
    out = out * (1.0 - w_bridge) + candidate * w_bridge
    out = np.clip(out, 0, 255).astype(np.uint8)
    out = _v31_micro_detail(out, amount=0.36 + 0.40 * strength)
    return out


def _v31_prepare_inp_image(img: np.ndarray) -> np.ndarray:
    src = np.asarray(img)
    if src.ndim == 2:
        gray = src.astype(np.uint8)
        p1, p99 = np.percentile(gray, [0.6, 99.6])
        base = gray if p99 <= p1 + 1e-3 else np.clip((gray.astype(np.float32) - p1) * (255.0 / (p99 - p1 + 1e-6)), 0, 255).astype(np.uint8)
        strong = _v31_bridge_gray(base, strength=0.96)
        strong = _v32_subpixel_granular_gray(strong, amount=0.34)
        return strong
    work = src.astype(np.uint8)
    lab = cv2.cvtColor(work, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    p1, p99 = np.percentile(l, [0.6, 99.6])
    base_l = l if p99 <= p1 + 1e-3 else np.clip((l.astype(np.float32) - p1) * (255.0 / (p99 - p1 + 1e-6)), 0, 255).astype(np.uint8)
    strong_l = _v31_bridge_gray(base_l, strength=0.96)
    strong_l = _v32_subpixel_granular_gray(strong_l, amount=0.34)
    merged = cv2.merge((strong_l, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


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
        if deg == 'inp_em':
            out = _v31_prepare_inp_image(img)
        elif isinstance(deg, str) and deg.startswith('sr'):
            if img.ndim == 3:
                lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                l = _v31_micro_detail(l, amount=0.34)
                out = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
            else:
                out = _v31_micro_detail(img, amount=0.34)
        else:
            out = img.copy()
        if out.ndim == 3:
            out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
        result.append((filename, out))
        original_sizes.append(original_size)
    if not result:
        raise ValueError('No valid images found for Diffusion input')
    return result, original_sizes


def preprocess_image(image: np.ndarray, membrane_gray_min: int = 50, top_percent: int = 10, is_rgb: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if image is None or image.size == 0:
        raise ValueError('Invalid image provided')
    src = np.asarray(image).copy()
    if src.ndim == 3:
        gray = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY if is_rgb else cv2.COLOR_BGR2GRAY)
    else:
        gray = src.copy()
    gray = gray.astype(np.uint8)
    prepared = _v32_subpixel_granular_gray(_v31_bridge_gray(gray, strength=0.82), amount=0.26)
    _g, dark, edge, blackhat, gap, ridge = _v31_gray_maps(prepared)
    score = _v31_clip01(0.38 * ridge + 0.28 * edge + 0.20 * blackhat + 0.14 * dark)
    thr = float(np.percentile(score, max(60, min(94, 100 - int(top_percent)))))
    membrane_mask = (score > thr).astype(np.uint8) * 255
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    membrane_mask = cv2.morphologyEx(membrane_mask, cv2.MORPH_CLOSE, k3, iterations=1)
    membrane_mask = cv2.dilate(membrane_mask, k3, iterations=1)
    membrane_mask = cv2.bitwise_or(membrane_mask, ((gap > 0.32).astype(np.uint8) * 255))
    membrane_mask = cv2.morphologyEx(membrane_mask, cv2.MORPH_CLOSE, k5, iterations=1)
    membrane_gray = cv2.bitwise_and(prepared, prepared, mask=membrane_mask)
    enhanced = cv2.cvtColor(prepared, cv2.COLOR_GRAY2RGB) if is_rgb else prepared
    return enhanced, membrane_mask, membrane_gray


# ===== v33 dark-membrane-prior inp strengthening =====
def dark_membrane_maps(gray: np.ndarray):
    gray = gray.astype(np.uint8)
    g = gray.astype(np.float32) / 255.0
    gx = cv2.Scharr(g, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(g, cv2.CV_32F, 0, 1)
    edge = cv2.magnitude(gx, gy)
    edge = _v31_clip01(edge / (np.percentile(edge, 99.0) + 1e-6))
    dark = 1.0 - g
    dark_rel = _v31_clip01(np.clip(cv2.GaussianBlur(dark, (0, 0), 1.35) - cv2.GaussianBlur(dark, (0, 0), 3.2), 0.0, None) / 0.18)
    blackhat5 = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))).astype(np.float32) / 255.0
    blackhat9 = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))).astype(np.float32) / 255.0
    blackhat = _v31_clip01(0.60 * blackhat5 + 0.40 * blackhat9)
    lap = cv2.Laplacian(g, cv2.CV_32F, ksize=3)
    side_dark = _v32_side_dark_np(dark) if '_v32_side_dark_np' in globals() else np.maximum(np.minimum(np.pad(dark[:, :-1], ((0, 0), (1, 0)), mode='edge'), np.pad(dark[:, 1:], ((0, 0), (0, 1)), mode='edge')), np.minimum(np.pad(dark[:-1, :], ((1, 0), (0, 0)), mode='edge'), np.pad(dark[1:, :], ((0, 1), (0, 0)), mode='edge')))
    gap = _v31_clip01(((lap < -0.015).astype(np.float32) * (0.45 + 0.55 * (side_dark > 0.18).astype(np.float32)) * (0.35 + 0.65 * edge)))
    membrane = _v31_clip01(0.34 * dark + 0.24 * dark_rel + 0.24 * blackhat + 0.18 * edge)
    return dark, edge, dark_rel, blackhat, gap, membrane


def bridge_gray(gray: np.ndarray, strength: float = 1.0) -> np.ndarray:
    gray = gray.astype(np.uint8)
    h, w = gray.shape[:2]
    small = _v31_small_image_factor_hw(h, w)
    dark, edge, dark_rel, blackhat, gap, membrane = dark_membrane_maps(gray)

    thr = float(np.percentile(membrane, 76.0 - 8.0 * small))
    mem = ((membrane >= thr) & (dark > (0.15 - 0.03 * small))).astype(np.uint8) * 255
    mem = cv2.morphologyEx(mem, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)

    dark_u8 = np.clip(dark * 255.0, 0, 255).astype(np.uint8)
    bridge_dark = dark_u8.astype(np.float32)
    bridge_mask = np.zeros_like(gray, dtype=np.uint8)
    lengths = [7, 11, 15, 19] if small > 0.45 else [9, 13, 17, 23]
    max_len = max(5, ((min(h, w) // 2) | 1))
    for k in lengths:
        k = min(k, max_len)
        if k % 2 == 0:
            k += 1
        for ker in _v31_make_line_kernels(k):
            closed_dark = cv2.morphologyEx(dark_u8, cv2.MORPH_CLOSE, ker, iterations=1)
            added = cv2.subtract(closed_dark, dark_u8)
            bridge_mask = np.maximum(bridge_mask, cv2.bitwise_and(added, cv2.dilate(mem, ker, iterations=1)))
            bridge_dark = np.maximum(bridge_dark, closed_dark.astype(np.float32))

    bridge_soft = cv2.GaussianBlur((bridge_mask > 0).astype(np.float32), (0, 0), 0.9 if small > 0.45 else 1.25)
    support = _v31_clip01(0.58 * membrane + 0.22 * dark_rel + 0.12 * blackhat + 0.08 * edge)
    support = cv2.GaussianBlur(support.astype(np.float32), (0, 0), 0.8 if small > 0.45 else 1.1)
    # only strong dark membrane neighborhoods should move substantially
    soft = _v31_clip01(np.maximum(bridge_soft, (0.30 + 0.18 * strength) * support) * (0.78 + 0.35 * support))

    bridge_gray = 255.0 - bridge_dark
    local_floor = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 0.7 if small > 0.45 else 1.0)
    candidate = np.minimum(bridge_gray, 0.82 * local_floor + 0.18 * gray.astype(np.float32))

    # Reinforce deepest membrane centerlines and explicitly connect tiny breaks.
    center = cv2.erode(mem, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
    out = gray.astype(np.float32) * (1.0 - soft) + candidate * soft
    center_w = ((center > 0).astype(np.float32)) * (0.06 + 0.12 * strength)
    out = out - center_w * (12.0 * dark_rel + 8.0 * blackhat)

    # Keep bright lumen / vesicle interior from collapsing.
    bright = gray.astype(np.float32) / 255.0
    bright_local = np.clip(bright - cv2.GaussianBlur(bright, (0, 0), 1.6), 0.0, 1.0)
    lumen_guard = cv2.GaussianBlur(((bright_local > 0.03) & (gap > 0.05)).astype(np.float32), (0, 0), 1.0)
    out = np.maximum(out, gray.astype(np.float32) * lumen_guard + out * (1.0 - lumen_guard))

    out = np.clip(out, 0, 255).astype(np.uint8)
    out = _v32_subpixel_granular_gray(out, amount=0.24 + 0.26 * strength)
    return out


def prepare_inp_image(img: np.ndarray) -> np.ndarray:
    src = np.asarray(img)
    if src.ndim == 2:
        gray = src.astype(np.uint8)
        p1, p99 = np.percentile(gray, [0.4, 99.7])
        base = gray if p99 <= p1 + 1e-3 else np.clip((gray.astype(np.float32) - p1) * (255.0 / (p99 - p1 + 1e-6)), 0, 255).astype(np.uint8)
        drive = float(max(_v31_small_image_factor_hw(*base.shape[:2]), _v39_internal_blockiness_score(base)))
        base = _v32_subpixel_granular_gray(base, amount=0.18 + 0.18 * drive)
        strong = bridge_gray(base, strength=1.0 + 0.12 * drive)
        strong = _v32_subpixel_granular_gray(strong, amount=0.38 + 0.22 * drive)
        if drive > 0.08:
            strong = _v31_micro_detail(strong, amount=0.16 + 0.12 * drive)
        return strong
    work = src.astype(np.uint8)
    lab = cv2.cvtColor(work, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    p1, p99 = np.percentile(l, [0.4, 99.7])
    base_l = l if p99 <= p1 + 1e-3 else np.clip((l.astype(np.float32) - p1) * (255.0 / (p99 - p1 + 1e-6)), 0, 255).astype(np.uint8)
    drive = float(max(_v31_small_image_factor_hw(*base_l.shape[:2]), _v39_internal_blockiness_score(base_l)))
    base_l = _v32_subpixel_granular_gray(base_l, amount=0.18 + 0.18 * drive)
    strong_l = bridge_gray(base_l, strength=1.0 + 0.12 * drive)
    strong_l = _v32_subpixel_granular_gray(strong_l, amount=0.38 + 0.22 * drive)
    if drive > 0.08:
        strong_l = _v31_micro_detail(strong_l, amount=0.16 + 0.12 * drive)
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
            out = prepare_inp_image(img)
            if out.ndim == 3:
                lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                l = _v31_micro_detail(l, amount=0.06 + 0.10 * strength)
                out = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
            else:
                out = _v31_micro_detail(out, amount=0.06 + 0.10 * strength)
        elif isinstance(deg, str) and deg.startswith('sr'):
            if img.ndim == 3:
                lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                l = _v31_micro_detail(l, amount=0.14 + 0.12 * strength)
                out = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
            else:
                out = _v31_micro_detail(img, amount=0.14 + 0.12 * strength)
        else:
            out = img.copy()
        if out.ndim == 3:
            out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
        result.append((filename, out))
        original_sizes.append(original_size)
    if not result:
        raise ValueError('No valid images found for Diffusion input')
    return result, original_sizes


def preprocess_image(image: np.ndarray, membrane_gray_min: int = 50, top_percent: int = 10, is_rgb: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if image is None or image.size == 0:
        raise ValueError('Invalid image provided')
    src = np.asarray(image).copy()
    if src.ndim == 3:
        gray = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY if is_rgb else cv2.COLOR_BGR2GRAY)
    else:
        gray = src.copy()
    gray = gray.astype(np.uint8)
    drive = float(max(_v31_small_image_factor_hw(*gray.shape[:2]), _v39_internal_blockiness_score(gray)))
    prepared = bridge_gray(gray, strength=0.90 + 0.06 * drive)
    prepared = _v32_subpixel_granular_gray(prepared, amount=0.30 + 0.18 * drive)
    if drive > 0.08:
        prepared = _v31_micro_detail(prepared, amount=0.10 + 0.08 * drive)
    dark, edge, dark_rel, blackhat, gap, membrane = dark_membrane_maps(prepared)
    score = _v31_clip01(0.34 * membrane + 0.22 * dark_rel + 0.18 * blackhat + 0.14 * edge + 0.12 * gap)
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
