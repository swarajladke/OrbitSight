"""Unified end-to-end inference and evaluation pipeline for OrbitSight."""

import math
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from src.common import (
    WINDOW_US,
    event_image,
    iter_windows,
    resolve_effective_config,
)
from src.detector import detect_boxes


def compute_confidence(
    box: Dict[str, float],
    hits: int,
    cfg: Dict[str, Any],
) -> float:
    """Compute deterministic weighted confidence score clipped to [0.01, 1.0]."""
    weights = cfg.get(
        "confidence_weights",
        {"density": 0.25, "events": 0.35, "compactness": 0.20, "persistence": 0.20},
    )
    w_den = float(weights.get("density", 0.25))
    w_evt = float(weights.get("events", 0.35))
    w_cmp = float(weights.get("compactness", 0.20))
    w_per = float(weights.get("persistence", 0.20))

    sub_density = min(1.0, float(box.get("density", 0.0)))
    min_evt = float(cfg.get("min_events_in_box", 3))
    sub_events = min(1.0, float(box.get("events", 0.0)) / (min_evt * 5.0))
    sub_compactness = 1.0 / (1.0 + abs(float(box.get("aspect", 1.0)) - 1.0))
    sub_persistence = min(1.0, hits / 3.0)

    score = (
        w_den * sub_density
        + w_evt * sub_events
        + w_cmp * sub_compactness
        + w_per * sub_persistence
    )

    return float(np.clip(score, 0.01, 1.0))


def run_sequence(
    events: np.ndarray,
    width: int,
    height: int,
    cfg: Dict[str, Any],
    window_us: int = WINDOW_US,
    max_windows: float = float("inf"),
) -> List[Tuple[int, int, int, int, int, int, float]]:
    """Run full unified detection pipeline on event stream.

    Encapsulates: windowing -> event_image -> detect_boxes -> persistence/min_hits gate
    -> compute_confidence -> conf_min filter -> max_candidates_per_window Top-K -> rounding.

    Args:
        events: NumPy array of events [x, y, t, p, (label)].
        width: Sensor width in pixels.
        height: Sensor height in pixels.
        cfg: Configuration dictionary.
        window_us: Window duration in microseconds (default 40,000 us).
        max_windows: Maximum windows to process (for testing/smoke tests).

    Returns:
        List of prediction tuples: (w_start, w_end, center_x, center_y, width, height, confidence).
    """
    if width >= 1200:
        sensor_name = "EVK4"
    elif width >= 600:
        sensor_name = "DVX"
    else:
        sensor_name = "DAVIS"

    eff = resolve_effective_config(cfg, sensor_name)

    window_records: List[Tuple[int, int, List[Dict[str, float]]]] = []
    window_count = 0

    for w_start, w_end, w_events in iter_windows(events, window_us=window_us):
        count_img, _, _ = event_image(w_events, width, height)
        boxes = detect_boxes(count_img, width, height, cfg)
        window_records.append((w_start, w_end, boxes))
        window_count += 1
        if window_count >= max_windows:
            break

    num_windows = len(window_records)
    min_hits = int(eff.get("min_hits", 1))
    max_dist_frac = float(eff.get("max_dist_frac", 0.08))
    conf_min = float(eff.get("conf_min", 0.0))
    max_k_val = eff.get("max_candidates_per_window", None)
    if max_k_val is not None:
        try:
            max_k = int(max_k_val)
        except (TypeError, ValueError):
            max_k = None
    else:
        max_k = None

    diagonal = math.hypot(width, height)
    max_dist = max_dist_frac * diagonal

    predictions: List[Tuple[int, int, int, int, int, int, float]] = []

    for w_idx in range(num_windows):
        w_start, w_end, boxes = window_records[w_idx]
        if not boxes:
            continue

        prev_boxes = window_records[w_idx - 1][2] if w_idx > 0 else []
        next_boxes = (
            window_records[w_idx + 1][2] if w_idx < num_windows - 1 else []
        )

        window_cands: List[Tuple[int, int, int, int, int, int, float]] = []

        for box in boxes:
            hits = 1
            has_prev = any(
                math.hypot(
                    box["center_x"] - p["center_x"],
                    box["center_y"] - p["center_y"],
                )
                <= max_dist
                for p in prev_boxes
            )
            if has_prev:
                hits += 1

            has_next = any(
                math.hypot(
                    box["center_x"] - n["center_x"],
                    box["center_y"] - n["center_y"],
                )
                <= max_dist
                for n in next_boxes
            )
            if has_next:
                hits += 1

            if min_hits >= 2 and hits < min_hits:
                continue

            conf = compute_confidence(box, hits, eff)

            if conf < conf_min:
                continue

            cx = int(round(box["center_x"]))
            cy = int(round(box["center_y"]))
            bw = int(round(box["width"]))
            bh = int(round(box["height"]))
            conf_rounded = round(conf, 4)

            window_cands.append((w_start, w_end, cx, cy, bw, bh, conf_rounded))

        if max_k is not None and max_k > 0 and len(window_cands) > max_k:
            window_cands.sort(key=lambda r: r[6], reverse=True)
            window_cands = window_cands[:max_k]

        predictions.extend(window_cands)

    return predictions
