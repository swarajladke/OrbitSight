"""Static source suppression map (stars and hot pixels) for space object detection."""

import math
from typing import Optional
import numpy as np

from src.common import WINDOW_US


def build_static_mask(
    events: np.ndarray,
    width: int,
    height: int,
    window_us: int = WINDOW_US,
    active_frac_thresh: float = 0.5,
) -> np.ndarray:
    """Return a bool mask (height, width) of pixels active in >= active_frac_thresh of all windows.

    These represent stationary sources: background stars and hot pixels.
    Vectorized O(E) implementation with exact linear window semantics.
    """
    if events.shape[0] == 0:
        return np.zeros((height, width), dtype=bool)

    t = events[:, 3]
    t_start = int(t[0])
    t_end = int(t[-1])
    num_windows = int(math.ceil((t_end - t_start + 1) / window_us))
    if num_windows <= 0:
        return np.zeros((height, width), dtype=bool)

    x = events[:, 0].astype(np.int64)
    y = events[:, 1].astype(np.int64)
    w_idx = (t.astype(np.int64) - t_start) // window_us

    valid = (x >= 0) & (x < width) & (y >= 0) & (y < height) & (w_idx >= 0) & (w_idx < num_windows)
    x = x[valid]
    y = y[valid]
    w_idx = w_idx[valid]

    if len(x) == 0:
        return np.zeros((height, width), dtype=bool)

    pixel_idx = y * width + x
    combined_key = w_idx * (width * height) + pixel_idx
    unique_keys = np.unique(combined_key)
    unique_pixels = unique_keys % (width * height)

    active_counts = np.bincount(unique_pixels, minlength=width * height)
    frac_active = active_counts.reshape(height, width).astype(np.float32) / float(num_windows)
    return frac_active >= active_frac_thresh
