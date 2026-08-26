"""Train post-hoc box resizing models (Arm 1 least-squares and Arm 2 HistGradientBoostingRegressor).

Fits on 15 training sequences and evaluates validation MAE on VAL_SEQS.
"""

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from tabulate import tabulate

from src.common import (
    WINDOW_US,
    event_image,
    infer_resolution,
    iter_windows,
    load_events,
    resolve_effective_config,
    sequence_name_from_npy,
)
from src.detector import detect_boxes
from src.features import FEATURE_NAMES, extract_window_features_batch
from src.metrics import windows_overlap
from src.scoreboard import load_yaml_config
from src.static_map import build_continuous_static_map


VAL_SEQS = {
    "DAVIS_SL8RB_2025-01-13-19-15-36",
    "DVX_Filtered_BlockDM_SLRB_32405_2025-01-20-19-57-17",
}


def load_gt_rows(gt_path: Path) -> List[Tuple[int, int, int, int, int, int]]:
    """Read GT rows from file."""
    rows: List[Tuple[int, int, int, int, int, int]] = []
    with open(gt_path, "r", encoding="utf-8") as f:
        rdr = csv.DictReader(f, delimiter="\t")
        for r in rdr:
            rows.append(
                (
                    int(r["window_start_timestamp_us"]),
                    int(r["window_end_timestamp_us"]),
                    int(r["center_x"]),
                    int(r["center_y"]),
                    int(r["width"]),
                    int(r["height"]),
                )
            )
    return rows


def extract_box_training_samples(
    npy_path: Path,
    gt_path: Path,
    cfg: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Extract candidate boxes along with their features and matching GT (w, h)."""
    seq_name = sequence_name_from_npy(npy_path)
    events = load_events(npy_path)
    width, height = infer_resolution(seq_name, events[:, 0], events[:, 1])

    if width >= 1200:
        sensor_name = "EVK4"
    elif width >= 600:
        sensor_name = "DVX"
    else:
        sensor_name = "DAVIS"

    eff = resolve_effective_config(cfg, sensor_name)
    min_hits = int(eff.get("min_hits", 1))
    max_dist_frac = float(eff.get("max_dist_frac", 0.08))
    diagonal = math.hypot(width, height)
    max_dist = max_dist_frac * diagonal
    max_dist_sq = max_dist * max_dist

    # Static map
    static_frac_map = None
    static_mask = None
    if bool(eff.get("use_static_map", False)):
        rate_thresh = float(eff.get("static_rate_thresh", 50.0))
        time_span_s = (events[-1, 3] - events[0, 3]) / 1e6 if len(events) > 1 else 1.0
        static_frac_map = build_continuous_static_map(
            events[:, 0], events[:, 1], width, height, time_span_s, rate_thresh=rate_thresh
        )
        static_mask = static_frac_map > 0.0

    # Pass 1: detect boxes
    from src.features import extract_local_bg

    window_records = []
    for w_start, w_end, w_events in iter_windows(events, WINDOW_US):
        count_img, _, _ = event_image(w_events, width, height, need_polarity=False)
        boxes = detect_boxes(count_img, width, height, cfg)
        if static_mask is not None and boxes:
            filtered_boxes: List[Dict[str, float]] = []
            for b in boxes:
                cy_r = int(round(b["center_y"]))
                cx_r = int(round(b["center_x"]))
                if 0 <= cy_r < height and 0 <= cx_r < width and static_mask[cy_r, cx_r]:
                    continue
                filtered_boxes.append(b)
            boxes = filtered_boxes
        for b in boxes:
            b["local_bg"] = extract_local_bg(
                count_img, float(b["center_x"]), float(b["center_y"]), float(b["width"]), float(b["height"])
            )
        window_records.append((w_start, w_end, boxes))

    gt_rows = load_gt_rows(gt_path)
    gt_by_window: Dict[int, List[Tuple[int, int, int, int, int, int]]] = {}
    for gt in gt_rows:
        gt_by_window.setdefault(gt[0], []).append(gt)

    num_windows = len(window_records)
    samples: List[Dict[str, Any]] = []

    for w_idx in range(num_windows):
        w_start, w_end, boxes = window_records[w_idx]
        if not boxes:
            continue

        prev_boxes = window_records[w_idx - 1][2] if w_idx > 0 else []
        next_boxes = window_records[w_idx + 1][2] if w_idx < num_windows - 1 else []

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

        cand_boxes_list: List[Dict[str, float]] = []
        for idx, box in enumerate(boxes):
            hits = 1 + int(has_prev[idx]) + int(has_next[idx])
            if min_hits >= 2 and hits < min_hits:
                continue
            box_copy = dict(box)
            box_copy["hits"] = hits
            cand_boxes_list.append(box_copy)

        if not cand_boxes_list:
            continue

        batch_feats = extract_window_features_batch(
            cand_boxes_list,
            prev_boxes,
            next_boxes,
            count_img=None,
            static_frac_map=static_frac_map,
        )

        matching_gts = gt_by_window.get(w_start, [])
        if not matching_gts:
            matching_gts = [gt for gt in gt_rows if windows_overlap(w_start, w_end, gt[0], gt[1])]

        for c_box, f_dict in zip(cand_boxes_list, batch_feats):
            cx = c_box["center_x"]
            cy = c_box["center_y"]
            ext_w = f_dict.get("extent_w", c_box.get("extent_w", c_box["width"]))
            ext_h = f_dict.get("extent_h", c_box.get("extent_h", c_box["height"]))

            matched_gt = None
            if matching_gts:
                # Find closest GT box within max_dist
                best_d = float("inf")
                for gt in matching_gts:
                    d = math.hypot(cx - gt[2], cy - gt[3])
                    if d <= max_dist and d < best_d:
                        best_d = d
                        matched_gt = gt

            feat_vec = [f_dict[name] for name in FEATURE_NAMES] + [float(width), float(height)]

            sample = {
                "seq_name": seq_name,
                "sensor": sensor_name,
                "width": width,
                "height": height,
                "center_x": cx,
                "center_y": cy,
                "extent_w": ext_w,
                "extent_h": ext_h,
                "features": feat_vec,
                "has_gt_match": matched_gt is not None,
                "gt_w": matched_gt[4] if matched_gt else None,
                "gt_h": matched_gt[5] if matched_gt else None,
            }
            samples.append(sample)

    return samples


def train_and_evaluate_box_regressors(
    dataset_dir: Path,
    cfg: Dict[str, Any],
    out_dir: Path = Path("models"),
) -> None:
    """Extract data, train Arm 1 (least squares) & Arm 2 (HGBR), and validate."""
    train_gt_files = sorted(list(dataset_dir.rglob("*_bb_windows_40ms.txt")))
    train_gt_files = [f for f in train_gt_files if "Training" in str(f)]

    all_samples: List[Dict[str, Any]] = []
    for gt_f in train_gt_files:
        seq_name = gt_f.name.replace("_bb_windows_40ms.txt", "")
        npy_matches = list(gt_f.parent.glob(f"{seq_name}_labeled_events.npy"))
        if not npy_matches:
            npy_matches = list(dataset_dir.rglob(f"{seq_name}_labeled_events.npy"))
        if not npy_matches:
            continue
        print(f"Extracting candidates from '{seq_name}'...", flush=True)
        seq_samples = extract_box_training_samples(npy_matches[0], gt_f, cfg)
        all_samples.extend(seq_samples)

    print(f"\nTotal extracted candidate samples: {len(all_samples)}")

    # Split train vs val
    train_samples = [s for s in all_samples if s["seq_name"] not in VAL_SEQS and s["has_gt_match"]]
    val_samples = [s for s in all_samples if s["seq_name"] in VAL_SEQS and s["has_gt_match"]]

    print(f"Matched training samples (15 seqs): {len(train_samples)}")
    print(f"Matched validation samples (2 seqs): {len(val_samples)}")

    # -------------------------------------------------------------
    # ARM 1: Per-sensor least squares fit on (extent_w, extent_h) -> (gt_w, gt_h)
    # y = slope * x + intercept (clamped to slope >= 0, intercept >= 0)
    # -------------------------------------------------------------
    arm1_models: Dict[str, Dict[str, Tuple[float, float]]] = {}
    for sensor in ["EVK4", "DVX", "DAVIS"]:
        s_train = [s for s in train_samples if s["sensor"] == sensor]
        if not s_train:
            arm1_models[sensor] = {"w": (1.0, 0.0), "h": (1.0, 0.0)}
            continue

        ext_w = np.array([s["extent_w"] for s in s_train], dtype=np.float64)
        ext_h = np.array([s["extent_h"] for s in s_train], dtype=np.float64)
        gt_w = np.array([s["gt_w"] for s in s_train], dtype=np.float64)
        gt_h = np.array([s["gt_h"] for s in s_train], dtype=np.float64)

        # Fit w: gt_w = a_w * ext_w + b_w
        A_w = np.vstack([ext_w, np.ones(len(ext_w))]).T
        slope_w, intercept_w = np.linalg.lstsq(A_w, gt_w, rcond=None)[0]

        # Fit h: gt_h = a_h * ext_h + b_h
        A_h = np.vstack([ext_h, np.ones(len(ext_h))]).T
        slope_h, intercept_h = np.linalg.lstsq(A_h, gt_h, rcond=None)[0]

        arm1_models[sensor] = {
            "w": (float(slope_w), float(intercept_w)),
            "h": (float(slope_h), float(intercept_h)),
        }
        print(f"[Arm 1 Least-Squares {sensor}] w = {slope_w:.4f}*extent_w + {intercept_w:.4f}, h = {slope_h:.4f}*extent_h + {intercept_h:.4f}")

    # -------------------------------------------------------------
    # ARM 2: HistGradientBoostingRegressor predicting log(w), log(h)
    # -------------------------------------------------------------
    X_train = np.array([s["features"] for s in train_samples], dtype=np.float32)
    y_train_w = np.array([math.log(max(1.0, s["gt_w"])) for s in train_samples], dtype=np.float32)
    y_train_h = np.array([math.log(max(1.0, s["gt_h"])) for s in train_samples], dtype=np.float32)

    reg_w = HistGradientBoostingRegressor(
        max_iter=150,
        learning_rate=0.08,
        min_samples_leaf=20,
        random_state=42,
    )
    reg_h = HistGradientBoostingRegressor(
        max_iter=150,
        learning_rate=0.08,
        min_samples_leaf=20,
        random_state=42,
    )

    print("\nFitting Arm 2 Regressors (log(w) and log(h))...", flush=True)
    reg_w.fit(X_train, y_train_w)
    reg_h.fit(X_train, y_train_h)

    arm2_model = {"reg_w": reg_w, "reg_h": reg_h, "feature_names": FEATURE_NAMES + ["img_width", "img_height"]}

    # Evaluate Validation MAE per sensor
    val_table = []
    for sensor in ["EVK4", "DVX", "DAVIS"]:
        s_val = [s for s in val_samples if s["sensor"] == sensor]
        if not s_val:
            val_table.append([sensor, "0", "-", "-", "-", "-"])
            continue

        gt_w_val = np.array([s["gt_w"] for s in s_val], dtype=np.float64)
        gt_h_val = np.array([s["gt_h"] for s in s_val], dtype=np.float64)

        # Arm 1 predictions
        slope_w, int_w = arm1_models[sensor]["w"]
        slope_h, int_h = arm1_models[sensor]["h"]
        pred_w_a1 = slope_w * np.array([s["extent_w"] for s in s_val]) + int_w
        pred_h_a1 = slope_h * np.array([s["extent_h"] for s in s_val]) + int_h

        # Arm 2 predictions
        X_val_s = np.array([s["features"] for s in s_val], dtype=np.float32)
        pred_w_a2 = np.exp(reg_w.predict(X_val_s))
        pred_h_a2 = np.exp(reg_h.predict(X_val_s))

        mae_w_a1 = mean_absolute_error(gt_w_val, pred_w_a1)
        mae_h_a1 = mean_absolute_error(gt_h_val, pred_h_a1)
        mae_w_a2 = mean_absolute_error(gt_w_val, pred_w_a2)
        mae_h_a2 = mean_absolute_error(gt_h_val, pred_h_a2)

        val_table.append([
            sensor,
            len(s_val),
            f"{mae_w_a1:.2f} px",
            f"{mae_h_a1:.2f} px",
            f"{mae_w_a2:.2f} px",
            f"{mae_h_a2:.2f} px",
        ])

    print("\n" + "=" * 80)
    print("  VALIDATION SET BOX-SIZE PREDICTION MAE (VAL_SEQS: DAVIS_SL8RB & DVX_BlockDM)")
    print("=" * 80)
    print(tabulate(val_table, headers=["Sensor", "N_val", "Arm 1 MAE(w)", "Arm 1 MAE(h)", "Arm 2 MAE(w)", "Arm 2 MAE(h)"], tablefmt="github"))
    print()

    # Save models
    out_dir.mkdir(parents=True, exist_ok=True)
    arm1_path = out_dir / "box_regressor_arm1.joblib"
    arm2_path = out_dir / "box_regressor_arm2.joblib"

    joblib.dump(arm1_models, arm1_path)
    joblib.dump(arm2_model, arm2_path)
    print(f"Saved Arm 1 model to {arm1_path}")
    print(f"Saved Arm 2 model to {arm2_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Box Regressors.")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="../OrbitSight_Dataset/Training_sets",
        help="Path to training dataset",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Master configuration YAML",
    )
    args = parser.parse_args()

    cfg = load_yaml_config(Path(args.config))
    train_and_evaluate_box_regressors(Path(args.dataset_dir), cfg)


if __name__ == "__main__":
    main()
