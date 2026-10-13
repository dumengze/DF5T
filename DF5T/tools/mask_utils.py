"""Deprecated: membrane scoring lives in `tools.em_maps.membrane_map`."""
from __future__ import annotations

import numpy as np

from tools.em_maps import membrane_map


def compute_membrane_mask(image, ksize=3, threshold=0.05):
    del ksize  # Sobel ksize; membrane_map uses multi-cue fusion instead
    """
    image: grayscale float [0,1] or uint8
    returns float32 mask in [0,1]
    """
    g = np.asarray(image, dtype=np.float32)
    if g.max() > 1.5:
        g = g / 255.0
    m = membrane_map(np.clip(g, 0.0, 1.0))
    return (m > float(threshold)).astype(np.float32)
