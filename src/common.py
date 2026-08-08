"""Common utilities, data loading, and event processing functions."""

from pathlib import Path
from typing import Any, Dict, Generator, Optional, Tuple, Union
import numpy as np

WINDOW_US: int = 40_000
LARGE_FILE_THRESHOLD_BYTES: int = 500 * 1024 * 1024  # 500 MB


def resolve_effective_config(cfg: Dict[str, Any], sensor_name: str) -> Dict[str, Any]:
    """Resolve and merge global defaults with sensor-specific overrides."""
    sensor_cfg = cfg.get(sensor_name, {}) if isinstance(cfg.get(sensor_name), dict) else {}
    effective: Dict[str, Any] = {}
    for k, v in cfg.items():
        if k not in ("EVK4", "DVX", "DAVIS") and not isinstance(v, dict):
            effective[k] = v[0] if isinstance(v, list) else v
    for k, v in sensor_cfg.items():
        effective[k] = v[0] if isinstance(v, list) else v
    return effective


def print_effective_config(cfg: Dict[str, Any]) -> None:
    """Print fully resolved effective configuration for all sensors."""
    print("\n==================================================")
    print("  RESOLVED EFFECTIVE CONFIGURATION PER SENSOR")
    print("==================================================")
    for sensor in ["DAVIS", "DVX", "EVK4"]:
        eff = resolve_effective_config(cfg, sensor)
        items_str = ", ".join(f"{k}: {v}" for k, v in sorted(eff.items()))
        print(f"[{sensor} EFFECTIVE CONFIG]: {items_str}")
    print("==================================================\n")


def sequence_name_from_npy(path: Union[str, Path]) -> str:
    """Extract sequence name from file path by stripping _labeled_events.npy.

    Args:
        path: Path to event file.

    Returns:
        Sequence name identifier string.
    """
    filename = Path(path).name
    if filename.endswith("_labeled_events.npy"):
        return filename[:-19]
    return Path(path).stem


def infer_resolution(
    seq_name: str,
    x: Optional[np.ndarray] = None,
    y: Optional[np.ndarray] = None,
) -> Tuple[int, int]:
    """Infer sensor resolution (width, height) using sequence name rules or coordinate bounds.

    Args:
        seq_name: Sequence name string.
        x: Optional array of x-coordinates for fallback.
        y: Optional array of y-coordinates for fallback.

    Returns:
        Tuple of (width, height) integers.

    Raises:
        ValueError: If resolution cannot be determined.
    """
    seq_upper = seq_name.upper()
    if "DAVIS" in seq_upper:
        return 346, 260
    if "DVX" in seq_upper:
        return 640, 480
    if "EVK4" in seq_upper:
        return 1280, 720

    if x is not None and y is not None and len(x) > 0 and len(y) > 0:
        return int(x.max()) + 1, int(y.max()) + 1

    raise ValueError(f"Could not infer sensor resolution for sequence: '{seq_name}'")


def load_events(path: Union[str, Path]) -> np.ndarray:
    """Load event data from .npy file into contiguous RAM for instant window slicing.

    Args:
        path: Path to the .npy file.

    Returns:
        NumPy array of shape (N, 6) containing event data.
    """
    file_path = Path(path)
    return np.load(file_path, allow_pickle=False)


def iter_windows(
    events: np.ndarray, window_us: int = WINDOW_US
) -> Generator[Tuple[int, int, np.ndarray], None, None]:
    """Generator yielding 40ms event windows using vectorized binary search.

    Args:
        events: Event array sorted by timestamp (column 3).
        window_us: Window duration in microseconds.

    Yields:
        Tuples of (start_timestamp_us, end_timestamp_us, window_events).
    """
    if events.shape[0] == 0:
        return

    t = events[:, 3]
    t_start = int(t[0])
    t_end = int(t[-1])

    grid = np.arange(t_start, t_end + window_us, window_us, dtype=np.int64)
    idx = np.searchsorted(t, grid)

    for i in range(len(grid) - 1):
        w_start = int(grid[i])
        w_end = int(grid[i + 1])
        w_events = np.asarray(events[idx[i] : idx[i + 1]])
        yield w_start, w_end, w_events


def event_image(
    window: np.ndarray, width: int, height: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Accumulate event count, positive, and negative polarity images.

    Args:
        window: Window event array of shape (K, 6).
        width: Image width.
        height: Image height.

    Returns:
        Tuple of (count_img, pos_img, neg_img) float32 arrays of shape (height, width).
    """
    count_img = np.zeros((height, width), dtype=np.float32)
    pos_img = np.zeros((height, width), dtype=np.float32)
    neg_img = np.zeros((height, width), dtype=np.float32)

    if window.shape[0] == 0:
        return count_img, pos_img, neg_img

    x = window[:, 0].astype(np.int64)
    y = window[:, 1].astype(np.int64)
    pol = window[:, 2].astype(np.int64)

    valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    x = x[valid]
    y = y[valid]
    pol = pol[valid]

    if len(x) == 0:
        return count_img, pos_img, neg_img

    np.add.at(count_img, (y, x), 1.0)

    pos_mask = pol == 1
    if np.any(pos_mask):
        np.add.at(pos_img, (y[pos_mask], x[pos_mask]), 1.0)

    neg_mask = pol == 0
    if np.any(neg_mask):
        np.add.at(neg_img, (y[neg_mask], x[neg_mask]), 1.0)

    return count_img, pos_img, neg_img
