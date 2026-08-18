"""Unified end-to-end inference and evaluation pipeline for OrbitSight.

Execution Order per Window:
1. Windowing (`iter_windows`) & Fast Count Map Generation (`event_image` with need_polarity=False)
2. Component Detection (`detect_boxes` returning full untruncated component lists)
3. Vectorized Neighborhood Persistence Matching against full untruncated neighbor windows (`min_hits`)
4. Deterministic Multi-term Weighted Confidence Scoring (`compute_confidence`)
5. Pipeline-stage NMS on weighted confidence (`apply_nms` if `nms_stage == 'pipeline'`)
6. Confidence Threshold Gating (`conf_min`)
7. Top-K Window Candidate Truncation (`max_candidates_per_window`)
8. Coordinate & Confidence Rounding
"""

import math
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
from src.nms import apply_nms


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
    deadline_ts: Optional[float] = None,
) -> Tuple[List[Tuple[int, int, int, int, int, int, float]], int]:
    """Run full unified detection pipeline on event stream.

    Args:
        events: NumPy array of events [x, y, p, t, (label)].
        width: Sensor width in pixels.
        height: Sensor height in pixels.
        cfg: Configuration dictionary.
        window_us: Window duration in microseconds (default 40,000 us).
        max_windows: Maximum windows to process (for testing/smoke tests).
        deadline_ts: Optional monotonic deadline timestamp.

    Returns:
        Tuple of (prediction_rows, num_windows_processed).
        Each prediction row: (w_start, w_end, center_x, center_y, width, height, confidence).
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
        if deadline_ts is not None and time.monotonic() > deadline_ts:
            break
        count_img, _, _ = event_image(w_events, width, height, need_polarity=False)
        boxes = detect_boxes(count_img, width, height, cfg)
        window_records.append((w_start, w_end, boxes))
        window_count += 1
        if window_count >= max_windows:
            break

    num_windows = len(window_records)
    min_hits = int(eff.get("min_hits", 1))
    max_dist_frac = float(eff.get("max_dist_frac", 0.08))
    conf_min = float(eff.get("conf_min", 0.0))
    nms_stage = str(eff.get("nms_stage", "pipeline")).lower()
    nms_iou_val = eff.get("nms_iou", 0.3)

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
    max_dist_sq = max_dist * max_dist

    predictions: List[Tuple[int, int, int, int, int, int, float]] = []

    for w_idx in range(num_windows):
        w_start, w_end, boxes = window_records[w_idx]
        if not boxes:
            continue

        prev_boxes = window_records[w_idx - 1][2] if w_idx > 0 else []
        next_boxes = (
            window_records[w_idx + 1][2] if w_idx < num_windows - 1 else []
        )

        n_cur = len(boxes)
        cur_centers = np.array([[b["center_x"], b["center_y"]] for b in boxes], dtype=np.float32)

        has_prev = np.zeros(n_cur, dtype=bool)
        if prev_boxes:
            p_centers = np.array([[p["center_x"], p["center_y"]] for p in prev_boxes], dtype=np.float32)
            diff_p = cur_centers[:, None, :] - p_centers[None, :, :]
            dist_sq_p = np.sum(diff_p * diff_p, axis=2)
            has_prev = np.any(dist_sq_p <= max_dist_sq, axis=1)

        has_next = np.zeros(n_cur, dtype=bool)
        if next_boxes:
            n_centers = np.array([[n["center_x"], n["center_y"]] for n in next_boxes], dtype=np.float32)
            diff_n = cur_centers[:, None, :] - n_centers[None, :, :]
            dist_sq_n = np.sum(diff_n * diff_n, axis=2)
            has_next = np.any(dist_sq_n <= max_dist_sq, axis=1)

        scored_cands: List[Dict[str, float]] = []

        # 1. Vectorized persistence matching on full untruncated neighbor lists & weighted scoring
        for idx, box in enumerate(boxes):
            hits = 1 + int(has_prev[idx]) + int(has_next[idx])

            if min_hits >= 2 and hits < min_hits:
                continue

            conf = compute_confidence(box, hits, eff)
            box_copy = dict(box)
            box_copy["confidence"] = conf
            scored_cands.append(box_copy)

        if not scored_cands:
            continue

        # 2. Pipeline-stage NMS using weighted confidence score
        if nms_stage == "pipeline" and nms_iou_val is not None:
            try:
                nms_thresh = float(nms_iou_val)
                scored_cands = apply_nms(scored_cands, nms_thresh)
            except (TypeError, ValueError):
                pass

        # 3. Confidence threshold filtering
        if conf_min > 0.0:
            scored_cands = [b for b in scored_cands if b["confidence"] >= conf_min]

        if not scored_cands:
            continue

        # 4. Top-K window candidate truncation
        if max_k is not None and max_k > 0 and len(scored_cands) > max_k:
            scored_cands.sort(key=lambda b: b["confidence"], reverse=True)
            scored_cands = scored_cands[:max_k]

        # 5. Output coordinate and confidence rounding
        for b in scored_cands:
            cx = int(round(b["center_x"]))
            cy = int(round(b["center_y"]))
            bw = int(round(b["width"]))
            bh = int(round(b["height"]))
            conf_rounded = round(float(b["confidence"]), 4)
            predictions.append((w_start, w_end, cx, cy, bw, bh, conf_rounded))

    return predictions, num_windows
