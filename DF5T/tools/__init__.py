from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image


@dataclass
class Crop:
    x1: int
    x2: int
    y1: int
    y2: int

    def __call__(self, img):
        if isinstance(img, Image.Image):
            return img.crop((self.y1, self.x1, self.y2, self.x2))
        arr = np.asarray(img)
        return arr[self.x1:self.x2, self.y1:self.y2]


def center_crop_arr(pil_image, image_size: int = 256):
    arr = np.asarray(pil_image)
    crop_y = max(0, (arr.shape[0] - image_size) // 2)
    crop_x = max(0, (arr.shape[1] - image_size) // 2)
    crop_h = min(image_size, arr.shape[0] - crop_y)
    crop_w = min(image_size, arr.shape[1] - crop_x)
    return arr[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w]


class _SimpleImageDataset:
    def __init__(self, root: str, txt_file: Optional[str] = None, normalize: bool = False):
        self.root = Path(root)
        self.normalize = normalize
        if txt_file and Path(txt_file).is_file():
            files = []
            for line in Path(txt_file).read_text(encoding='utf-8').splitlines():
                if not line.strip():
                    continue
                stem = line.split()[0]
                for ext in ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'):
                    p = self.root / f'{stem}{ext}'
                    if p.exists():
                        files.append(p)
                        break
            self.files = files
        else:
            self.files = sorted([p for p in self.root.iterdir() if p.suffix.lower() in {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int):
        img = Image.open(self.files[idx]).convert('RGB')
        arr = np.asarray(img, dtype=np.float32) / 255.0
        t = torch.from_numpy(arr.transpose(2, 0, 1))
        if self.normalize:
            t = t * 2.0 - 1.0
        return t, 0


def get_dataset(args, config):
    root = os.path.join(args.exp, 'datasets', 'MitEM', 'MitEM')
    txt = os.path.join(args.exp, 'MitEM_val_1k.txt')
    ds = _SimpleImageDataset(root, txt_file=txt, normalize=False)
    return ds, ds


def logit_transform(image, lam: float = 1e-6):
    image = lam + (1 - 2 * lam) * image
    return torch.log(image) - torch.log1p(-image)


def data_transform(config, X):
    if getattr(config.data, 'uniform_dequantization', False):
        X = X / 256.0 * 255.0 + torch.rand_like(X) / 256.0
    if getattr(config.data, 'gaussian_dequantization', False):
        X = X + torch.randn_like(X) * 0.01
    if getattr(config.data, 'rescaled', True):
        X = 2 * X - 1.0
    elif getattr(config.data, 'logit_transform', False):
        X = logit_transform(X)
    if hasattr(config, 'image_mean'):
        return X - config.image_mean.to(X.device)[None, ...]
    return X


def inverse_data_transform(config, X):
    if hasattr(config, 'image_mean'):
        X = X + config.image_mean.to(X.device)[None, ...]
    if getattr(config.data, 'logit_transform', False):
        X = torch.sigmoid(X)
    elif getattr(config.data, 'rescaled', True):
        X = (X + 1.0) / 2.0
    return torch.clamp(X, 0.0, 1.0)
