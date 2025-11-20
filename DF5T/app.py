import os
import sys
if getattr(sys, 'frozen', False):
    base_path = sys._MEIPASS
else:
    base_path = os.path.abspath(".")
def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('image_processor.log'),
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
            "enable_postprocessing": True
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
            scale_factor = 1
            new_width = int(original_width * scale_factor)
            new_height = int(original_height * scale_factor)
            img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
            logger.info(f"Resized image from {original_width}x{original_height} to {new_width}x{new_height} (scale: 1/4)")
        elif original_width > 1024 or original_height > 1024:
            scale_factor = 1
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
        use_membrane_enhancement: bool = True,
        top_percent: int = 80,
        dispersion_ratio: float = 0.9,
        denoise_strength: float = 0.005
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
        self._running = True
        self.original_sizes = []  # Store original image sizes for MRC reconstruction

    def stop(self) -> None:
        """Stop the processing thread."""
        self._running = False

    def run(self) -> None:
        try:
            self.status.emit("Converting images to PNG...")
            convert_to_png(self.input_folder)
            self.progress.emit(5)

            enhanced_images = None
            if self.use_membrane_enhancement:
                self.status.emit("Running membrane preprocessing...")
                enhanced_images, self.original_sizes = process_images_in_folder(
                    self.input_folder,
                    top_percent=self.top_percent,
                    dispersion_ratio=self.dispersion_ratio,
                    denoise_strength=self.denoise_strength,
                    color_enhance_factor=0.005,
                    window_size=1
                )
                if not enhanced_images:
                    raise ValueError("No images were processed successfully")
            else:
                self.status.emit("Skipping membrane preprocessing...")
                valid_extensions = ['.png']
                enhanced_images = []
                for f in os.listdir(self.input_folder):
                    if f.lower().endswith(tuple(valid_extensions)):
                        img_path = os.path.join(self.input_folder, f)
                        img, scale_factor, original_size = resize_image_if_needed(img_path)
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
                model_path=self.model_path
            )

            config_path = os.path.join("configs", args.config)
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Config file {config_path} not found")
            with open(config_path, "r") as f:
                config_dict = yaml.safe_load(f)
            config_dict['data'] = config_dict.get('data', {})
            config_dict['data']['root'] = dataset_dir
            config_dict['data']['txt'] = txt_file
            config = dict2namespace(config_dict)

            supported_degradations = ["deblur_em", "deno_em", "isotropic_em", "inp_em", "sr2"]
            if self.deg not in supported_degradations:
                raise ValueError(f"Degradation type '{self.deg}' not supported")

            self.progress.emit(40)
            self.status.emit(f"Processing images with degradation: {self.deg}")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            runner = Diffusion(args, config, device)
            runner.sample()
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
            self.progress.emit(100)
            self.finished.emit(restored_images)
        except Exception as e:
            logger.error(f"Error in ImageProcessor: {str(e)}")
            self.error.emit(str(e))

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
        self.model_path = r"exp\model\MitEM\model_y.pt"
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
            self.deg_combo.addItems(["deblur_em", "deno_em", "isotropic_em", "inp_em", "sr2"])
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

            # Sigma
            sigma_label = QLabel("σ Sigma:")
            sigma_label.setStyleSheet(label_style)
            controls_layout.addWidget(sigma_label, 2, 0)
            self.sigma_label = QLabel("0.10")
            self.sigma_label.setFont(QFont("Segoe UI", 12))
            self.sigma_slider = QSlider(Qt.Orientation.Horizontal)
            self.sigma_slider.setRange(0, 100)
            self.sigma_slider.setValue(10)
            self.sigma_slider.valueChanged.connect(
                lambda: self.sigma_label.setText(f"{self.sigma_slider.value()/100:.2f}")
            )
            controls_layout.addWidget(self.sigma_slider, 2, 1)
            controls_layout.addWidget(self.sigma_label, 2, 2)

            # Membrane Enhancement Checkbox
            membrane_label = QLabel("🧠 Membrane:")
            membrane_label.setStyleSheet(label_style)
            controls_layout.addWidget(membrane_label, 3, 0)
            self.membrane_checkbox = QCheckBox("Enable")
            self.membrane_checkbox.setFont(QFont("Segoe UI", 12))
            self.membrane_checkbox.setChecked(True)
            self.membrane_checkbox.stateChanged.connect(self.toggle_membrane_params)
            controls_layout.addWidget(self.membrane_checkbox, 3, 1, 1, 2)

            # Top Percent
            top_percent_label = QLabel("📊 Top Percent:")
            top_percent_label.setStyleSheet(label_style)
            controls_layout.addWidget(top_percent_label, 4, 0)
            self.top_percent_slider = QSlider(Qt.Orientation.Horizontal)
            self.top_percent_slider.setRange(1, 100)
            self.top_percent_slider.setValue(50)
            self.top_percent_value_label = QLabel("50")
            self.top_percent_value_label.setFont(QFont("Segoe UI", 12))
            self.top_percent_slider.valueChanged.connect(
                lambda: self.top_percent_value_label.setText(str(self.top_percent_slider.value()))
            )
            controls_layout.addWidget(self.top_percent_slider, 4, 1)
            controls_layout.addWidget(self.top_percent_value_label, 4, 2)

            # Dispersion Ratio
            dispersion_label = QLabel("🔍 Dispersion:")
            dispersion_label.setStyleSheet(label_style)
            controls_layout.addWidget(dispersion_label, 5, 0)
            self.dispersion_slider = QSlider(Qt.Orientation.Horizontal)
            self.dispersion_slider.setRange(0, 100)
            self.dispersion_slider.setValue(20)
            self.dispersion_value_label = QLabel("0.2")
            self.dispersion_value_label.setFont(QFont("Segoe UI", 12))
            self.dispersion_slider.valueChanged.connect(
                lambda: self.dispersion_value_label.setText(f"{self.dispersion_slider.value()/100:.1f}")
            )
            controls_layout.addWidget(self.dispersion_slider, 5, 1)
            controls_layout.addWidget(self.dispersion_value_label, 5, 2)

            # Denoise Strength
            denoise_label = QLabel("🔇 Denoise:")
            denoise_label.setStyleSheet(label_style)
            controls_layout.addWidget(denoise_label, 6, 0)
            self.denoise_slider = QSlider(Qt.Orientation.Horizontal)
            self.denoise_slider.setRange(0, 100)
            self.denoise_slider.setValue(50)
            self.denoise_value_label = QLabel("0.005")
            self.denoise_value_label.setFont(QFont("Segoe UI", 12))
            self.denoise_slider.valueChanged.connect(
                lambda: self.denoise_value_label.setText(f"{self.denoise_slider.value()/10000:.3f}")
            )
            controls_layout.addWidget(self.denoise_slider, 6, 1)
            controls_layout.addWidget(self.denoise_value_label, 6, 2)

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
        """Enable or disable membrane enhancement parameters."""
        enabled = self.membrane_checkbox.isChecked()
        self.top_percent_slider.setEnabled(enabled)
        self.top_percent_value_label.setEnabled(enabled)
        self.dispersion_slider.setEnabled(enabled)
        self.dispersion_value_label.setEnabled(enabled)
        self.denoise_slider.setEnabled(enabled)
        self.denoise_value_label.setEnabled(enabled)

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

            valid_extensions = ['.png']
            self.original_images = [
                os.path.join(folder, f) for f in os.listdir(folder)
                if any(f.lower().endswith(ext) for ext in valid_extensions)
            ]
            images = natsorted(self.original_images[:8])
            for idx, img in enumerate(images):
                label = ImageLabel(img, self.theme)
                caption = QLabel(os.path.splitext(os.path.basename(img))[0][:15])
                caption.setFont(QFont("Segoe UI", 10))
                caption.setStyleSheet(f"color: {STYLES[self.theme]['text']};")
                self.preview_layout.addWidget(label, idx//4, idx%4)
                self.preview_layout.addWidget(
                    caption, idx//4 + 1, idx%4, alignment=Qt.AlignmentFlag.AlignCenter
                )
        except Exception as e:
            logger.error(f"Error displaying preview: {str(e)}")
            self.status_log.append(f"Error displaying preview: {str(e)}")

    def process_images(self) -> None:
        """Start image processing."""
        try:
            self.process_btn.setEnabled(False)
            self.cancel_btn.setEnabled(True)
            self.processor = ImageProcessor(
                self.input_folder,
                self.deg_combo.currentText(),
                self.time_slider.value(),
                self.sigma_slider.value()/100,
                self.model_path,
                use_membrane_enhancement=self.membrane_checkbox.isChecked(),
                top_percent=self.top_percent_slider.value(),
                dispersion_ratio=self.dispersion_slider.value()/100,
                denoise_strength=self.denoise_slider.value()/10000
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
        self.output_folder = os.path.join(self.input_folder, "output") 
        self.status_log.append("Image processing finished. You can now manually reconstruct MRC if needed.")

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
                caption = QLabel(os.path.basename(img_path)[:15])
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
    """Preprocess image for membrane detection, preserving RGB if needed."""
    try:
        if image is None or image.size == 0:
            raise ValueError("Invalid image provided")

        if is_rgb:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            enhanced = enhance_contrast(image, is_rgb=True)
        else:
            gray = image
            enhanced = enhance_contrast(image, is_rgb=False)

        pixels = gray.flatten()
        if len(pixels) == 0:
            raise ValueError("Empty image after preprocessing")
        sorted_pixels = np.sort(pixels)
        if len(sorted_pixels) == 0:
            raise ValueError("No valid pixels for percentile calculation")
        membrane_gray_max = np.percentile(sorted_pixels, 100 - top_percent)
        if membrane_gray_max == 0:
            logger.warning("Membrane gray max is 0, adjusting to avoid division by zero")
            membrane_gray_max = 255
        membrane_mask = cv2.inRange(gray, membrane_gray_min, int(membrane_gray_max))
        membrane_gray = gray * (membrane_mask > 0)
        logger.info(f"Membrane max gray value: {membrane_gray_max}")
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
    denoise_strength: float = 0.005,
    color_enhance_factor: float = 0.2,
    noise_compression_factor: float = 0.2,
    window_size: int = 10,
    use_membrane_enhancement: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """Process and color membrane regions, preserving RGB if needed."""
    try:
        # Load image in appropriate mode
        image, scale_factor, original_size = resize_image_if_needed(image_path)
        if image is None:
            raise ValueError(f"Unable to load image at {image_path}")
        
        if not use_membrane_enhancement:
            return image, np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8), original_size

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

if __name__ == "__main__":
    main()

