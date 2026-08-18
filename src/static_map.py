"""Static source suppression map (stars and hot pixels) for space object detection."""

from typing import Optional
import numpy as np

from src.common import WINDOW_US, iter_windows


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
    if events.shape[0] == 0:
        return np.zeros((height, width), dtype=bool)

    window_count_map = np.zeros((height, width), dtype=np.int32)
    num_windows = 0

    for _, _, w_events in iter_windows(events, window_us=window_us):
        num_windows += 1
        if w_events.shape[0] == 0:
            continue

        x = w_events[:, 0].astype(np.int64)
        y = w_events[:, 1].astype(np.int64)
        valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
        if not np.any(valid):
            continue

        # Find unique active pixels in this window
        flat_idx = np.unique(y[valid] * width + x[valid])
        uy = flat_idx // width
        ux = flat_idx % width
        window_count_map[uy, ux] += 1

    if num_windows == 0:
        return np.zeros((height, width), dtype=bool)

    frac_active = window_count_map.astype(np.float32) / float(num_windows)
    mask = frac_active >= active_frac_thresh
    return mask
