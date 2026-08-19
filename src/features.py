"""Motion and candidate feature extraction for learned space object re-scoring."""

import argparse
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from tabulate import tabulate

from src.scoreboard import load_yaml_config


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

    outer_area = (oy2 - oy1) * (ox2 - ox1)
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
    if "local_bg" in candidate and candidate["local_bg"] is not None:
        local_bg = float(candidate["local_bg"])
    elif count_img is not None:
        local_bg = extract_local_bg(count_img, cx, cy, bw, bh, pad=4)
    else:
        local_bg = 0.0

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
    """Compute exact binary ROC-AUC score via rank sum."""
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


def generate_feature_auc_report(dataset_dir: Path, config_path: Path) -> None:
    """Generate and print feature separability report on train split sequences."""
    from src.train_scorer import extract_sequence_dataset

    cfg = load_yaml_config(config_path)

    train_gt_files = [
        f for f in sorted(list(dataset_dir.rglob("*_bb_windows_40ms.txt")))
        if "Training" in str(f)
    ]
    if len(train_gt_files) != 17:
        raise RuntimeError(f"Expected 17 train sequences, found {len(train_gt_files)}")

    sensor_data: Dict[str, Tuple[List[np.ndarray], List[np.ndarray]]] = {
        "DAVIS": ([], []),
        "DVX": ([], []),
        "EVK4": ([], []),
        "OVERALL": ([], []),
    }

    for gt_f in train_gt_files:
        seq_name = gt_f.name.replace("_bb_windows_40ms.txt", "")
        if "EVK4" in seq_name.upper():
            sensor = "EVK4"
        elif "DVX" in seq_name.upper():
            sensor = "DVX"
        else:
            sensor = "DAVIS"

        npy_matches = list(gt_f.parent.glob(f"{seq_name}_labeled_events.npy"))
        if not npy_matches:
            npy_matches = list(dataset_dir.rglob(f"{seq_name}_labeled_events.npy"))
        if not npy_matches:
            continue

        npy_f = npy_matches[0]
        X_seq, y_seq, _ = extract_sequence_dataset(npy_f, gt_f, cfg)
        if len(y_seq) == 0:
            continue

        sensor_data[sensor][0].append(X_seq)
        sensor_data[sensor][1].append(y_seq)
        sensor_data["OVERALL"][0].append(X_seq)
        sensor_data["OVERALL"][1].append(y_seq)

    print("\n================================================================================")
    print("  FEATURE DISCRIMINATIVE POWER & SEPARABILITY REPORT (TRAIN SPLIT)")
    print("================================================================================\n")

    for group in ["EVK4", "DAVIS", "DVX", "OVERALL"]:
        X_parts, y_parts = sensor_data[group]
        if not X_parts:
            print(f"[{group}]: No samples found.\n")
            continue

        X = np.vstack(X_parts)
        y = np.concatenate(y_parts)
        n_pos = int(np.sum(y == 1))
        n_neg = int(np.sum(y == 0))

        print(f"--- Group: {group} (Positives: {n_pos}, Negatives: {n_neg}) ---")
        table_rows = []
        for feat_idx, feat_name in enumerate(FEATURE_NAMES):
            feat_vals = X[:, feat_idx]
            pos_vals = feat_vals[y == 1]
            neg_vals = feat_vals[y == 0]

            mean_pos = float(np.mean(pos_vals)) if len(pos_vals) > 0 else 0.0
            mean_neg = float(np.mean(neg_vals)) if len(neg_vals) > 0 else 0.0
            auc = compute_roc_auc(y, feat_vals)

            table_rows.append([feat_name, f"{mean_pos:.4f}", f"{mean_neg:.4f}", f"{auc:.4f}"])

        print(tabulate(table_rows, headers=["Feature", "Mean (Pos)", "Mean (Neg)", "ROC-AUC"], tablefmt="grid"))
        print()


def main() -> None:
    """Report feature distributions and binary AUC separation against ground truth."""
    parser = argparse.ArgumentParser(description="Evaluate feature discriminative power")
    parser.add_argument("--dataset-dir", type=str, default="../OrbitSight_Dataset")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--report", action="store_true", help="Generate feature AUC report")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir).resolve()
    config_path = Path(args.config).resolve()

    if args.report:
        generate_feature_auc_report(dataset_dir, config_path)
    else:
        print("Specify --report to generate feature AUC report.")


if __name__ == "__main__":
    main()
