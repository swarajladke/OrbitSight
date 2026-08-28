"""Static source suppression map (stars and hot pixels) for space object detection."""

import math
from typing import Optional
import numpy as np

from src.common import WINDOW_US


def build_continuous_static_map(
    events: np.ndarray,
    width: int,
    height: int,
    window_us: int = WINDOW_US,
) -> np.ndarray:
    """Return a continuous float32 map (height, width) of the fraction of windows each pixel is active.

    Fixed-size accumulator implementation over streaming windows.
    """
    if events.shape[0] == 0:
        return np.zeros((height, width), dtype=np.float32)

    t = events[:, 3]
    t_start = int(t[0])
    t_end = int(t[-1])
    num_windows = int(math.ceil((t_end - t_start + 1) / float(window_us)))
    if num_windows <= 0:
        return np.zeros((height, width), dtype=np.float32)

    from src.common import iter_windows

    counts = np.zeros(height * width, dtype=np.int32)
    for _, _, w_ev in iter_windows(events, window_us=window_us):
        if len(w_ev) == 0:
            continue
        x = w_ev[:, 0].astype(np.int64)
        y = w_ev[:, 1].astype(np.int64)
        valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
        if not np.any(valid):
            continue
        idx = y[valid] * width + x[valid]
        active = np.bincount(idx, minlength=height * width) > 0
        counts += active.astype(np.int32)

    frac_active = (counts.astype(np.float32) / float(num_windows)).reshape(height, width)
    return frac_active


def build_static_mask(
    events: np.ndarray,
    width: int,
    height: int,
    window_us: int = WINDOW_US,
    active_frac_thresh: float = 0.5,
) -> np.ndarray:
    """Return a bool mask (height, width) of pixels active in >= active_frac_thresh of all windows.

    These represent stationary sources: background stars and hot pixels.
    """
    frac_active = build_continuous_static_map(events, width, height, window_us=window_us)
    return frac_active >= active_frac_thresh
