import sys
import os
import shutil
import argparse
import yaml
import torch
import json
import logging
import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
from pathlib import Path
from natsort import natsorted
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton, QLabel,
    QFileDialog, QScrollArea, QGridLayout, QComboBox, QSlider, QHBoxLayout,
    QProgressBar, QGroupBox, QDialog, QTextEdit, QSizePolicy
)
from PyQt6.QtGui import QPixmap, QColor, QIcon, QFont, QPainter, QPen
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QPropertyAnimation, QEasingCurve, QPoint

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

# Theme styles
STYLES = {
    "light": {
        "background": "qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #f5f7fa, stop:1 #e9ecef)",
        "text": "#2d3436",
        "button": "#1e90ff",
        "button_hover": "#0984e3",
        "panel": "#ffffff",
        "shadow": "0 6px 20px rgba(0,0,0,0.08)",
        "border": "#dfe6e9",
        "accent": "#ff6b6b"
    },
    "dark": {
        "background": "qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #2d3436, stop:1 #636e72)",
        "text": "#dfe6e9",
        "button": "#1e90ff",
        "button_hover": "#0984e3",
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
        default_config = {"theme": "light", "last_folder": "", "sidebar_collapsed": False}
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

def setup_dataset_and_list(input_folder: str) -> Tuple[str, str]:
    """Set up dataset directory and image list file."""
    try:
        valid_extensions = ['.png', '.tif', '.jpg', '.jpeg', '.tiff']
        valid_files = [
            f for f in os.listdir(input_folder)
            if any(f.lower().endswith(ext) for ext in valid_extensions)
        ]
        if not valid_files:
            raise ValueError(f"No valid images found in {input_folder}")

        dataset_dir = os.path.join(input_folder, "datasets", "MitEM", "MitEM")
        os.makedirs(dataset_dir, exist_ok=True)

        for filename in valid_files:
            src_path = os.path.join(input_folder, filename)
            dst_path = os.path.join(dataset_dir, filename)
            if not os.path.exists(dst_path):
                shutil.copy2(src_path, dst_path)

        txt_path = os.path.join(input_folder, "MitEM_val_1k.txt")
        sorted_files = natsorted(valid_files)
        with open(txt_path, 'w') as f:
            for filename in sorted_files:
                name_without_extension = os.path.splitext(filename)[0]
                f.write(f"{name_without_extension} 1\n")
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
        top_percent: int = 80,
        dispersion_ratio: float = 0.9
    ):
        super().__init__()
        self.input_folder = input_folder
        self.deg = deg
        self.timesteps = timesteps
        self.sigma_0 = sigma_0
        self.model_path = model_path
        self.top_percent = top_percent
        self.dispersion_ratio = dispersion_ratio
        self._running = True

    def stop(self) -> None:
        """Stop the processing thread."""
        self._running = False

    def run(self) -> None:
        """Run the image processing pipeline."""
        try:
            self.status.emit("Running membrane preprocessing...")
            enhanced_images = process_images_in_folder(
                self.input_folder,
                top_percent=self.top_percent,
                dispersion_ratio=self.dispersion_ratio,
                denoise_strength=0.00000001,
                color_enhance_factor=0.005,
                window_size=1
            )
            self.progress.emit(10)

            self.status.emit("Preparing dataset...")
            txt_file, dataset_dir = setup_dataset_and_folder(self.input_folder, enhanced_images)
            self.progress.emit(20)

            if not self._running:
                return

            self.status.emit("Setting up output directory...")
            output_folder = os.path.join(self.input_folder, "output")
            os.makedirs(output_folder, exist_ok=True)

            args = argparse.Namespace(
                ni=True,
                config="DF5T_512.yml",
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
            config_dict['data']['txt_file'] = txt_file
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
        self.original_label.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        self.generated_label = QLabel("Generated")
        self.generated_label.setFont(QFont("Arial", 18, QFont.Weight.Bold))

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
            self.close_btn.setFont(QFont("Arial", 16))
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
        self.original_images: List[str] = []
        self.generated_images: List[str] = []
        self.image_pairs: Dict[str, str] = {}
        self.model_path = r"exp\model\MitEM\model_512.pt"
        self.sidebar_collapsed = self.config["sidebar_collapsed"]
        self.setup_ui()

    def setup_ui(self) -> None:
        """Set up the main window UI."""
        try:
            self.setWindowTitle("DF5T - Advanced Image Processor")
            self.setGeometry(100, 100, 1600, 1000)

            self.main_widget = QWidget()
            self.setCentralWidget(self.main_widget)
            self.main_layout = QHBoxLayout(self.main_widget)
            self.main_layout.setContentsMargins(0, 0, 0, 0)
            self.main_layout.setSpacing(0)

            # Sidebar
            self.sidebar = QWidget()
            self.sidebar_layout = QVBoxLayout(self.sidebar)
            self.sidebar_layout.setContentsMargins(20, 20, 20, 20)
            self.sidebar_layout.setSpacing(20)
            self.sidebar.setMinimumWidth(350)
            self.sidebar.setMaximumWidth(350 if not self.sidebar_collapsed else 60)

            header_layout = QHBoxLayout()
            self.title = QLabel("DF5T")
            self.title.setFont(QFont("Arial", 30, QFont.Weight.Bold))
            header_layout.addWidget(self.title)
            self.collapse_btn = QPushButton("◄" if not self.sidebar_collapsed else "►")
            self.collapse_btn.setFont(QFont("Arial", 14))
            self.collapse_btn.clicked.connect(self.toggle_sidebar)
            header_layout.addStretch()
            header_layout.addWidget(self.collapse_btn)
            self.sidebar_layout.addLayout(header_layout)

            self.theme_combo = QComboBox()
            self.theme_combo.addItems(["Light", "Dark", "High Contrast"])
            self.theme_combo.setCurrentText(self.theme.capitalize())
            self.theme_combo.currentTextChanged.connect(self.change_theme)
            self.sidebar_layout.addWidget(self.theme_combo)

            input_group = QGroupBox("Input")
            input_layout = QVBoxLayout(input_group)
            folder_layout = QHBoxLayout()
            self.folder_icon = QLabel()
            self.folder_icon.setPixmap(QIcon.fromTheme("folder").pixmap(28))
            folder_layout.addWidget(self.folder_icon)
            self.folder_label = QLabel(
                "No folder selected" if not self.config["last_folder"]
                else os.path.basename(self.config["last_folder"])
            )
            self.folder_label.setFont(QFont("Arial", 14))
            folder_layout.addWidget(self.folder_label)
            folder_btn = QPushButton("Browse")
            folder_btn.setFont(QFont("Arial", 14))
            folder_btn.clicked.connect(self.select_folder)
            folder_layout.addWidget(folder_btn)
            input_layout.addLayout(folder_layout)
            self.sidebar_layout.addWidget(input_group)

            controls_group = QGroupBox("Parameters")
            controls_layout = QGridLayout(controls_group)
            controls_layout.setVerticalSpacing(15)
            controls_layout.setHorizontalSpacing(10)

            label_style = """
                QLabel {
                    font: bold 14px 'Arial';
                    color: %(text)s;
                    min-width: 120px;
                    border-right: 2px solid %(border)s;
                    padding-right: 15px;
                }
            """ % STYLES[self.theme]

            # Task
            task_label = QLabel("Task:")
            task_label.setStyleSheet(label_style)
            controls_layout.addWidget(task_label, 0, 0)
            self.deg_combo = QComboBox()
            self.deg_combo.setFont(QFont("Arial", 14))
            self.deg_combo.addItems(["deblur_em", "deno_em", "isotropic_em", "inp_em", "sr2"])
            controls_layout.addWidget(self.deg_combo, 0, 1, 1, 2)

            # Timesteps
            time_label = QLabel("Timesteps:")
            time_label.setStyleSheet(label_style)
            controls_layout.addWidget(time_label, 1, 0)
            self.time_label = QLabel("50")
            self.time_label.setFont(QFont("Arial", 14))
            self.time_slider = QSlider(Qt.Orientation.Horizontal)
            self.time_slider.setRange(10, 100)
            self.time_slider.setValue(50)
            self.time_slider.valueChanged.connect(
                lambda: self.time_label.setText(str(self.time_slider.value()))
            )
            controls_layout.addWidget(self.time_slider, 1, 1)
            controls_layout.addWidget(self.time_label, 1, 2)

            # Sigma
            sigma_label = QLabel("Sigma:")
            sigma_label.setStyleSheet(label_style)
            controls_layout.addWidget(sigma_label, 2, 0)
            self.sigma_label = QLabel("0.06")
            self.sigma_label.setFont(QFont("Arial", 14))
            self.sigma_slider = QSlider(Qt.Orientation.Horizontal)
            self.sigma_slider.setRange(0, 100)
            self.sigma_slider.setValue(6)
            self.sigma_slider.valueChanged.connect(
                lambda: self.sigma_label.setText(f"{self.sigma_slider.value()/100:.2f}")
            )
            controls_layout.addWidget(self.sigma_slider, 2, 1)
            controls_layout.addWidget(self.sigma_label, 2, 2)

            # Top Percent
            top_percent_label = QLabel("Top Percent:")
            top_percent_label.setStyleSheet(label_style)
            controls_layout.addWidget(top_percent_label, 3, 0)
            self.top_percent_slider = QSlider(Qt.Orientation.Horizontal)
            self.top_percent_slider.setRange(1, 100)
            self.top_percent_slider.setValue(80)
            self.top_percent_value_label = QLabel("80")
            self.top_percent_value_label.setFont(QFont("Arial", 14))
            self.top_percent_slider.valueChanged.connect(
                lambda: self.top_percent_value_label.setText(str(self.top_percent_slider.value()))
            )
            controls_layout.addWidget(self.top_percent_slider, 3, 1)
            controls_layout.addWidget(self.top_percent_value_label, 3, 2)

            # Dispersion Ratio
            dispersion_label = QLabel("Dispersion Ratio:")
            dispersion_label.setStyleSheet(label_style)
            controls_layout.addWidget(dispersion_label, 4, 0)
            self.dispersion_slider = QSlider(Qt.Orientation.Horizontal)
            self.dispersion_slider.setRange(0, 100)
            self.dispersion_slider.setValue(90)
            self.dispersion_value_label = QLabel("0.9")
            self.dispersion_value_label.setFont(QFont("Arial", 14))
            self.dispersion_slider.valueChanged.connect(
                lambda: self.dispersion_value_label.setText(f"{self.dispersion_slider.value()/100:.1f}")
            )
            controls_layout.addWidget(self.dispersion_slider, 4, 1)
            controls_layout.addWidget(self.dispersion_value_label, 4, 2)

            controls_layout.setColumnStretch(0, 1)
            controls_layout.setColumnStretch(1, 3)
            controls_layout.setColumnStretch(2, 1)

            self.sidebar_layout.addWidget(controls_group)

            btn_layout = QHBoxLayout()
            self.process_btn = QPushButton("Process")
            self.process_btn.setFont(QFont("Arial", 14))
            self.process_btn.clicked.connect(self.process_images)
            self.process_btn.setEnabled(bool(self.config["last_folder"]))
            btn_layout.addWidget(self.process_btn)
            self.cancel_btn = QPushButton("Cancel")
            self.cancel_btn.setFont(QFont("Arial", 14))
            self.cancel_btn.clicked.connect(self.cancel_processing)
            self.cancel_btn.setEnabled(False)
            btn_layout.addWidget(self.cancel_btn)
            self.sidebar_layout.addLayout(btn_layout)

            self.sidebar_layout.addStretch()

            # Main Content
            self.content_widget = QWidget()
            self.content_layout = QVBoxLayout(self.content_widget)
            self.content_layout.setContentsMargins(30, 30, 30, 30)
            self.content_layout.setSpacing(20)

            split_widget = QWidget()
            split_layout = QHBoxLayout(split_widget)

            preview_group = QGroupBox("Preview")
            preview_layout = QVBoxLayout(preview_group)
            self.preview_scroll = QScrollArea()
            self.preview_widget = QWidget()
            self.preview_layout = QGridLayout(self.preview_widget)
            self.preview_scroll.setWidget(self.preview_widget)
            self.preview_scroll.setWidgetResizable(True)
            preview_layout.addWidget(self.preview_scroll)
            split_layout.addWidget(preview_group)

            results_group = QGroupBox("Results")
            results_layout = QVBoxLayout(results_group)
            self.results_scroll = QScrollArea()
            self.results_widget = QWidget()
            self.results_layout = QGridLayout(self.results_widget)
            self.results_scroll.setWidget(self.results_widget)
            self.results_scroll.setWidgetResizable(True)
            results_layout.addWidget(self.results_scroll)
            split_layout.addWidget(results_group)

            self.content_layout.addWidget(split_widget, stretch=1)

            progress_group = QGroupBox("Progress")
            progress_layout = QVBoxLayout(progress_group)
            self.progress_bar = QProgressBar()
            self.progress_bar.setFont(QFont("Arial", 12))
            progress_layout.addWidget(self.progress_bar)
            self.status_log = QTextEdit()
            self.status_log.setReadOnly(True)
            self.status_log.setFont(QFont("Arial", 12))
            self.status_log.setMaximumHeight(100)
            progress_layout.addWidget(self.status_log)
            self.content_layout.addWidget(progress_group)

            self.main_layout.addWidget(self.sidebar)
            self.main_layout.addWidget(self.content_widget, stretch=1)

            self.update_theme()
            if self.config["last_folder"]:
                self.input_folder = self.config["last_folder"]
                self.display_preview(self.input_folder)
        except Exception as e:
            logger.error(f"Error setting up MainWindow: {str(e)}")
            raise

    def update_theme(self) -> None:
        """Update the UI theme."""
        try:
            style = STYLES[self.theme]
            self.main_widget.setStyleSheet(f"background: {style['background']};")
            self.sidebar.setStyleSheet(f"background: {style['panel']}; border-right: 1px solid {style['border']};")

            for group in [
                self.sidebar.findChild(QGroupBox, "Input"),
                self.sidebar.findChild(QGroupBox, "Parameters"),
                self.content_widget.findChild(QGroupBox, "Preview"),
                self.content_widget.findChild(QGroupBox, "Results"),
                self.content_widget.findChild(QGroupBox, "Progress")
            ]:
                if group:
                    group.setStyleSheet(f"""
                        QGroupBox {{
                            background-color: {style['panel']};
                            border: 1px solid {style['border']};
                            border-radius: 10px;
                            padding: 20px;
                            margin-top: 15px;
                            font-weight: bold;
                            font-size: 16px;
                            color: {style['text']};
                            box-shadow: {style['shadow']};
                        }}
                        QGroupBox::title {{
                            subcontrol-origin: margin;
                            left: 15px;
                            padding: 0 8px;
                        }}
                    """)

            for btn in [self.process_btn, self.cancel_btn, self.collapse_btn]:
                btn.setStyleSheet(f"""
                    QPushButton {{
                        background-color: {style['button']};
                        color: white;
                        padding: 12px 25px;
                        border-radius: 8px;
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
                    padding: 8px;
                    border-radius: 6px;
                    color: {style['text']};
                    font-size: 14px;
                }}
                QComboBox::drop-down {{
                    border-left: 1px solid {style['border']};
                    width: 30px;
                }}
                QComboBox QAbstractItemView {{
                    background-color: {style['panel']};
                    color: {style['text']};
                    selection-background-color: {style['button']};
                    border: 1px solid {style['border']};
                }}
            """)

            slider_style = f"""
                QSlider::groove:horizontal {{
                    height: 8px;
                    background: {style['border']};
                    border-radius: 4px;
                }}
                QSlider::handle:horizontal {{
                    background: {style['button']};
                    width: 20px;
                    height: 20px;
                    border-radius: 10px;
                    margin: -6px 0;
                }}
                QSlider::handle:horizontal:hover {{
                    background: {style['button_hover']};
                }}
            """
            for slider in [
                self.time_slider, self.sigma_slider,
                self.top_percent_slider, self.dispersion_slider
            ]:
                slider.setStyleSheet(slider_style)

            self.progress_bar.setStyleSheet(f"""
                QProgressBar {{
                    border: 1px solid {style['border']};
                    border-radius: 6px;
                    background-color: {style['panel']};
                    text-align: center;
                    color: {style['text']};
                    font-size: 12px;
                }}
                QProgressBar::chunk {{
                    background-color: qlineargradient(x1:0, y1:0, x2:1, y2:1, 
                                                    stop:0 {style['button']}, 
                                                    stop:1 {style['button_hover']});
                    border-radius: 5px;
                }}
            """)

            self.status_log.setStyleSheet(f"""
                QTextEdit {{
                    background-color: {style['panel']};
                    border: 1px solid {style['border']};
                    border-radius: 6px;
                    color: {style['text']};
                    padding: 5px;
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
                        border-radius: 4px;
                    }}
                    QScrollBar::handle {{
                        background: {style['button']};
                        border-radius: 4px;
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
            self.collapse_btn.setText("◄" if not self.sidebar_collapsed else "►")
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
        except Exception as e:
            logger.error(f"Error changing theme: {str(e)}")

    def select_folder(self) -> None:
        """Open folder selection dialog."""
        try:
            folder = QFileDialog.getExistingDirectory(self, "Select Image Folder")
            if folder:
                self.folder_label.setText(os.path.basename(folder))
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
            for i in reversed(range(self.preview_layout.count())):
                widget = self.preview_layout.itemAt(i).widget()
                if widget:
                    widget.setParent(None)

            valid_extensions = ['.png', '.tif', '.jpg', '.jpeg', '.tiff']
            self.original_images = [
                os.path.join(folder, f) for f in os.listdir(folder)
                if any(f.lower().endswith(ext) for ext in valid_extensions)
            ]
            images = natsorted(self.original_images[:8])
            for idx, img in enumerate(images):
                label = ImageLabel(img, self.theme)
                caption = QLabel(os.path.splitext(os.path.basename(img))[0][:15])
                caption.setFont(QFont("Arial", 12))
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
                top_percent=self.top_percent_slider.value(),
                dispersion_ratio=self.dispersion_slider.value()/100
            )
            self.processor.finished.connect(self.display_images)
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
                caption.setFont(QFont("Arial", 12))
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

def enhance_contrast(image: np.ndarray) -> np.ndarray:
    """Enhance image contrast using CLAHE."""
    try:
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        return clahe.apply(image)
    except Exception as e:
        logger.error(f"Error in enhance_contrast: {str(e)}")
        raise

def preprocess_image(image: np.ndarray, membrane_gray_min: int = 50, top_percent: int = 10) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Preprocess image for membrane detection."""
    try:
        if image is None or image.size == 0:
            raise ValueError("Invalid image provided")

        gray = enhance_contrast(image)
        pixels = gray.flatten()
        sorted_pixels = np.sort(pixels)
        membrane_gray_max = np.percentile(sorted_pixels, 100 - top_percent)
        membrane_mask = cv2.inRange(gray, membrane_gray_min, int(membrane_gray_max))
        membrane_gray = gray * (membrane_mask > 0)
        logger.info(f"Membrane max gray value: {membrane_gray_max}")
        return gray, membrane_mask, membrane_gray
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
    gray: np.ndarray,
    membrane_mask: np.ndarray,
    noise_reduction_level_1: float = 70,
    noise_enhance_level_2_3: float = 70
) -> np.ndarray:
    """Enhance membrane regions."""
    try:
        membrane_pixels = gray[membrane_mask > 0]
        light_threshold = np.percentile(membrane_pixels, 90)
        dark_threshold = np.percentile(membrane_pixels, 10)

        light_pixels = gray > light_threshold
        mid_dark_pixels = (gray >= dark_threshold) & (gray <= light_threshold)
        dark_pixels = gray < dark_threshold

        enhanced_image = gray.copy()
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
    denoise_strength: float = 0.1
) -> np.ndarray:
    """Lighten background and apply denoising."""
    try:
        if not 0.0 <= denoise_strength <= 1.0:
            logger.warning("denoise_strength out of range, using 0.1")
            denoise_strength = 0.1

        if denoise_strength == 0:
            return image

        background_mask = cv2.bitwise_not(mitochondria_mask)
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
    repair_gap_factor: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """Enhance mitochondria regions."""
    try:
        mitochondria_mask = (mitochondria_mask > 0).astype(np.uint8) * 255
        avg_gray = np.average(mitochondria_mask)
        logger.info(f"Average Gray Value: {avg_gray}")

        if avg_gray < 15:
            color_enhance_factor = 0.15
        elif 15 <= avg_gray < 60:
            color_enhance_factor = 0.12
        elif 60 <= avg_gray < 125:
            color_enhance_factor = 0.09
        elif 125 <= avg_gray < 180:
            color_enhance_factor = 0.06
        else:
            color_enhance_factor = 0.03
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
    membrane_gray: np.ndarray,
    window_size: int = 4,
    density_threshold: float = 0.5,
    dilation_iterations: int = 2,
    erosion_iterations: int = 2,
    min_cluster_size_ratio: float = 0.02
) -> np.ndarray:
    """Detect dense membrane regions."""
    try:
        height, width = membrane_mask.shape
        dense_mask = np.zeros_like(membrane_mask)
        dense_mask_before_morph = np.zeros_like(membrane_mask)
        membrane_mask_binary = (membrane_mask > 0).astype(int)
        noise_points = []
        window_area = (2 * window_size + 1) ** 2

        for y in range(height):
            for x in range(width):
                if membrane_mask_binary[y, x] > 0:
                    y_min = max(0, y - window_size)
                    y_max = min(height, y + window_size + 1)
                    x_min = max(0, x - window_size)
                    x_max = min(width, x + window_size + 1)
                    local_window = membrane_mask_binary[y_min:y_max, x_min:x_max]
                    local_density = np.sum(local_window)
                    density_ratio = local_density / window_area
                    local_gray_value = membrane_gray[y, x]
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
            dst_path = os.path.join(dataset_dir, filename)
            cv2.imwrite(dst_path, img)
            valid_files.append(filename)

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
    denoise_strength: float = 0.1,
    color_enhance_factor: float = 0.2,
    noise_compression_factor: float = 0.2,
    window_size: int = 10
) -> Tuple[np.ndarray, np.ndarray]:
    """Process and color membrane regions."""
    try:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Unable to load image at {image_path}")

        gray, membrane_mask, membrane_gray = preprocess_image(image, membrane_gray_min, top_percent)
        dense_mask = detect_and_color_dense_noise_points(
            gray, membrane_mask, window_size=window_size, dispersion_ratio=dispersion_ratio
        )
        lightened_image = lighten_and_denoise(image, dense_mask, denoise_strength)
        dense_region = cv2.bitwise_and(image, image, mask=dense_mask)
        enhanced_image, refined_mask = process_mitochondria(
            dense_region, dense_mask, color_enhance_factor, noise_compression_factor
        )

        refined_mask_non_black = np.where(refined_mask > 0, refined_mask, 0)
        refined_mask_non_black_float = refined_mask_non_black.astype(float) / 255
        final_image = (
            lightened_image.astype(float) * (1 - refined_mask_non_black_float) +
            enhanced_image.astype(float) * refined_mask_non_black_float
        ).astype(np.uint8)

        return final_image, refined_mask_non_black
    except Exception as e:
        logger.error(f"Error in process_and_color_membrane: {str(e)}")
        raise

def process_images_in_folder(
    folder_path: str,
    membrane_gray_min: int = 1,
    top_percent: int = 10,
    density_threshold: float = 0.35,
    dispersion_ratio: float = 0.1,
    denoise_strength: float = 0.4,
    color_enhance_factor: float = 0.2,
    noise_compression_factor: float = 0.2,
    window_size: int = 10
) -> List[Tuple[str, np.ndarray]]:
    """Process all images in a folder."""
    try:
        enhanced_images = []
        for filename in os.listdir(folder_path):
            if filename.lower().endswith((".png", ".jpg", ".jpeg", ".tiff")):
                image_path = os.path.join(folder_path, filename)
                try:
                    final_image, _ = process_and_color_membrane(
                        image_path, membrane_gray_min, top_percent, density_threshold,
                        dispersion_ratio, denoise_strength, color_enhance_factor,
                        noise_compression_factor, window_size
                    )
                    enhanced_images.append((filename, final_image))
                except Exception as e:
                    logger.warning(f"Failed to process {filename}: {str(e)}")
                    continue
        return enhanced_images
    except Exception as e:
        logger.error(f"Error in process_images_in_folder: {str(e)}")
        raise

def detect_and_color_dense_noise_points(
    image: np.ndarray,
    membrane_mask: np.ndarray,
    window_size: int = 30,
    dispersion_ratio: float = 0.1,
    noise_compression_factor: float = 0.3
) -> np.ndarray:
    """Detect and color dense noise points."""
    try:
        height, width = membrane_mask.shape
        dense_mask = np.zeros_like(membrane_mask)
        window_area = (2 * window_size + 1) ** 2

        for y in range(height):
            for x in range(width):
                if membrane_mask[y, x] > 0:
                    y_min = max(0, y - window_size)
                    y_max = min(height, y + window_size + 1)
                    x_min = max(0, x - window_size)
                    x_max = min(width, x + window_size + 1)
                    window = membrane_mask[y_min:y_max, x_min:x_max]
                    mask_pixels_in_window = np.sum(window > 0)
                    density = mask_pixels_in_window / window_area

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
        window = MainWindow()
        window.show()
        sys.exit(app.exec())
    except Exception as e:
        logger.error(f"Error starting application: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()