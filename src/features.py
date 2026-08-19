"""Motion and candidate feature extraction for learned space object re-scoring."""

import argparse
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from src.common import WINDOW_US, infer_resolution, load_events, sequence_name_from_npy
from src.metrics import iou
from src.scoreboard import load_gt_file
from src.static_map import build_static_mask


FEATURE_NAMES = [
    "events",
    "density",
    "area",
    "extent_w",
    "extent_h",
    "aspect",
    "hits",
    "disp_prev",
    "disp_next",
    "speed",
    "dir_consistency",
    "static_frac",
    "local_bg",
]


def extract_local_bg(
    count_img: np.ndarray,
    cx: float,
    cy: float,
    bw: float,
    bh: float,
    pad: int = 4,
) -> float:
    """Compute mean event count in an outer rectangular ring around the box, excluding box interior."""
    h_img, w_img = count_img.shape
    x1 = max(0, int(math.floor(cx - bw / 2.0)))
    y1 = max(0, int(math.floor(cy - bh / 2.0)))
    x2 = min(w_img, int(math.ceil(cx + bw / 2.0)))
    y2 = min(h_img, int(math.ceil(cy + bh / 2.0)))

    ox1 = max(0, x1 - pad)
    oy1 = max(0, y1 - pad)
    ox2 = min(w_img, x2 + pad)
    oy2 = min(h_img, y2 + pad)

    outer_area = (ox2 - oy1) * (ox2 - ox1)
    inner_area = (y2 - y1) * (x2 - x1)
    ring_area = outer_area - inner_area

    if ring_area <= 0:
        return 0.0

    outer_sum = float(np.sum(count_img[oy1:oy2, ox1:ox2]))
    inner_sum = float(np.sum(count_img[y1:y2, x1:x2]))
    ring_sum = max(0.0, outer_sum - inner_sum)

    return ring_sum / float(ring_area)


def extract_candidate_features(
    candidate: Dict[str, Any],
    prev_candidates: Optional[List[Dict[str, Any]]],
    next_candidates: Optional[List[Dict[str, Any]]],
    count_img: np.ndarray,
    static_frac_map: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Extract motion, geometric, and static spatial features for a single candidate box."""
    cx = float(candidate.get("center_x", 0.0))
    cy = float(candidate.get("center_y", 0.0))
    bw = max(1.0, float(candidate.get("width", 1.0)))
    bh = max(1.0, float(candidate.get("height", 1.0)))
    area = bw * bh
    events = float(candidate.get("events", 1.0))
    density = events / area
    aspect = max(bw, bh) / max(min(bw, bh), 1.0)
    hits = float(candidate.get("hits", candidate.get("persistence_hits", 1.0)))

    # Compute displacement to nearest previous window candidate
    disp_prev = -1.0
    vec_prev: Optional[Tuple[float, float]] = None
    if prev_candidates:
        min_d = float("inf")
        best_p: Optional[Dict[str, Any]] = None
        for p in prev_candidates:
            d = math.hypot(cx - float(p["center_x"]), cy - float(p["center_y"]))
            if d < min_d:
                min_d = d
                best_p = p
        disp_prev = min_d
        if best_p:
            vec_prev = (cx - float(best_p["center_x"]), cy - float(best_p["center_y"]))

    # Compute displacement to nearest next window candidate
    disp_next = -1.0
    vec_next: Optional[Tuple[float, float]] = None
    if next_candidates:
        min_d = float("inf")
        best_n: Optional[Dict[str, Any]] = None
        for n in next_candidates:
            d = math.hypot(float(n["center_x"]) - cx, float(n["center_y"]) - cy)
            if d < min_d:
                min_d = d
                best_n = n
        disp_next = min_d
        if best_n:
            vec_next = (float(best_n["center_x"]) - cx, float(best_n["center_y"]) - cy)

    # Speed
    if disp_prev >= 0.0 and disp_next >= 0.0:
        speed = (disp_prev + disp_next) / 2.0
    elif disp_prev >= 0.0:
        speed = disp_prev
    elif disp_next >= 0.0:
        speed = disp_next
    else:
        speed = -1.0

    # Direction consistency: cosine between (prev->cur) and (cur->next)
    dir_consistency = -1.0
    if vec_prev is not None and vec_next is not None:
        mag_p = math.hypot(vec_prev[0], vec_prev[1])
        mag_n = math.hypot(vec_next[0], vec_next[1])
        if mag_p > 1e-4 and mag_n > 1e-4:
            dot = vec_prev[0] * vec_next[0] + vec_prev[1] * vec_next[1]
            cos_sim = dot / (mag_p * mag_n)
            dir_consistency = max(-1.0, min(1.0, cos_sim))

    # Static fraction
    static_frac = 0.0
    if static_frac_map is not None:
        h_map, w_map = static_frac_map.shape
        cy_r = int(round(cy))
        cx_r = int(round(cx))
        if 0 <= cy_r < h_map and 0 <= cx_r < w_map:
            static_frac = float(static_frac_map[cy_r, cx_r])

    # Local background
    local_bg = extract_local_bg(count_img, cx, cy, bw, bh, pad=4)

    return {
        "events": events,
        "density": density,
        "area": area,
        "extent_w": bw,
        "extent_h": bh,
        "aspect": aspect,
        "hits": hits,
        "disp_prev": disp_prev,
        "disp_next": disp_next,
        "speed": speed,
        "dir_consistency": dir_consistency,
        "static_frac": static_frac,
        "local_bg": local_bg,
    }


def compute_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute exact ROC-AUC score without scikit-learn dependency."""
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    n_pos = len(pos)
    n_neg = len(neg)
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    # Wilcoxon-Mann-Whitney U statistic
    r = np.argsort(np.argsort(np.concatenate([pos, neg])))
    r_pos = r[:n_pos] + 1
    u_pos = np.sum(r_pos) - n_pos * (n_pos + 1) / 2.0
    return float(u_pos / (n_pos * n_neg))


def main() -> None:
    """Report feature distributions and binary AUC separation against ground truth."""
    parser = argparse.ArgumentParser(description="Evaluate feature discriminative power")
    parser.add_argument("--dataset-dir", type=str, default="../OrbitSight_Dataset")
    parser.add_argument("--report", action="store_true", help="Generate feature AUC report")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir).resolve()
    print(f"Generating feature discriminative report across dataset: {dataset_dir}...")


if __name__ == "__main__":
    main()
