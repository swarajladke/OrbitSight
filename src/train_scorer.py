"""Train learned candidate re-scorer on training split sequences."""

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple
import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
import yaml

from src.common import WINDOW_US, event_image, infer_resolution, iter_windows, load_events, sequence_name_from_npy
from src.detector import detect_boxes
from src.features import FEATURE_NAMES, extract_candidate_features
from src.metrics import iou, windows_overlap
from src.scoreboard import load_gt_file, resolve_effective_config
from src.static_map import build_continuous_static_map


def extract_sequence_dataset(
    npy_path: Path,
    gt_path: Path,
    cfg: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, str]:
    """Extract candidate feature vectors and GT binary labels for a sequence."""
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

    gt_rows = load_gt_file(gt_path)

    # Build continuous static fraction map and discrete static mask
    static_frac_map = build_continuous_static_map(events, width, height, window_us=WINDOW_US)
    static_thresh = eff.get("static_thresh", None)
    if static_thresh is not None:
        static_mask = static_frac_map >= float(static_thresh)
    else:
        static_mask = None

    # Pre-collect all window candidate lists
    window_records: List[Tuple[int, int, List[Dict[str, Any]], np.ndarray]] = []
    for ws, we, w_events in iter_windows(events, window_us=WINDOW_US):
        count_img, _, _ = event_image(w_events, width, height, need_polarity=False)
        boxes = detect_boxes(count_img, width, height, cfg)
        if static_mask is not None and boxes:
            filtered = []
            for b in boxes:
                cy_r = int(round(b["center_y"]))
                cx_r = int(round(b["center_x"]))
                if 0 <= cy_r < height and 0 <= cx_r < width and static_mask[cy_r, cx_r]:
                    continue
                filtered.append(b)
            boxes = filtered
        window_records.append((ws, we, boxes, count_img))

    X_list: List[List[float]] = []
    y_list: List[int] = []

    num_w = len(window_records)
    for w_idx, (ws, we, boxes, count_img) in enumerate(window_records):
        if not boxes:
            continue

        prev_boxes = window_records[w_idx - 1][2] if w_idx > 0 else []
        next_boxes = (
            window_records[w_idx + 1][2] if w_idx < num_w - 1 else []
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

        gt_matches = [
            g for g in gt_rows
            if (g[0] == ws and g[1] == we) or windows_overlap(ws, we, g[0], g[1])
        ]

        for idx, box in enumerate(boxes):
            hits = 1 + int(has_prev[idx]) + int(has_next[idx])
            if min_hits >= 2 and hits < min_hits:
                continue

            box_copy = dict(box)
            box_copy["hits"] = hits

            feats = extract_candidate_features(
                box_copy,
                prev_boxes,
                next_boxes,
                count_img,
                static_frac_map=static_frac_map,
            )
            feat_vec = [feats[name] for name in FEATURE_NAMES]

            # Determine binary label via IoU >= 0.5
            cx = float(box_copy["center_x"])
            cy = float(box_copy["center_y"])
            bw = float(box_copy["width"])
            bh = float(box_copy["height"])
            box_tuple = (cx, cy, bw, bh)

            label = 0
            for g in gt_matches:
                g_tuple = (float(g[2]), float(g[3]), float(g[4]), float(g[5]))
                if iou(box_tuple, g_tuple) >= 0.5:
                    label = 1
                    break

            X_list.append(feat_vec)
            y_list.append(label)

    if not X_list:
        return np.zeros((0, len(FEATURE_NAMES)), dtype=np.float32), np.zeros(0, dtype=np.int32), seq_name

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int32), seq_name


def train_scorer(
    dataset_dir: Path,
    config_path: Path,
    models_dir: Path,
) -> None:
    """Train Gradient Boosting re-scorer strictly on training split sequences."""
    models_dir.mkdir(parents=True, exist_ok=True)

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Locate training ground truth files using scoreboard's split-resolution rule
    train_gt_files = [
        f for f in sorted(list(dataset_dir.rglob("*_bb_windows_40ms.txt")))
        if "Training" in str(f)
    ]
    if len(train_gt_files) != 17:
        raise RuntimeError(f"Expected exactly 17 train sequences, found {len(train_gt_files)}")

    print(f"Found {len(train_gt_files)} Training sequences for learned re-scorer training.")

    # Validation holdout (2 sequences)
    val_seq_names = {
        "DAVIS_SL8RB_2025-01-13-19-15-36",
        "DVX_Filtered_BlockDM_SLRB_32405_2025-01-20-19-57-17",
    }

    train_X_parts, train_y_parts = [], []
    val_X_parts, val_y_parts = [], []

    train_seqs_used = []
    val_seqs_used = []

    for gt_f in train_gt_files:
        seq_name = gt_f.name.replace("_bb_windows_40ms.txt", "")
        npy_matches = list(gt_f.parent.glob(f"{seq_name}_labeled_events.npy"))
        if not npy_matches:
            npy_matches = list(dataset_dir.rglob(f"{seq_name}_labeled_events.npy"))
        if not npy_matches:
            continue

        npy_f = npy_matches[0]
        print(f"Extracting features from '{seq_name}'...", flush=True)
        X_seq, y_seq, _ = extract_sequence_dataset(npy_f, gt_f, cfg)
        if len(y_seq) == 0:
            continue

        if seq_name in val_seq_names:
            val_X_parts.append(X_seq)
            val_y_parts.append(y_seq)
            val_seqs_used.append(seq_name)
        else:
            train_X_parts.append(X_seq)
            train_y_parts.append(y_seq)
            train_seqs_used.append(seq_name)

    if not train_X_parts:
        raise RuntimeError("Training set extraction produced 0 samples.")
    if not val_X_parts:
        raise RuntimeError("Validation holdout sequences produced 0 samples.")

    X_train = np.vstack(train_X_parts)
    y_train = np.concatenate(train_y_parts)

    X_val = np.vstack(val_X_parts)
    y_val = np.concatenate(val_y_parts)

    print(f"\nDataset Assembly Complete:")
    print(f"  Train: {X_train.shape[0]} samples ({int(np.sum(y_train))} positive, {X_train.shape[0] - int(np.sum(y_train))} negative)")
    print(f"  Val:   {X_val.shape[0]} samples ({int(np.sum(y_val))} positive, {X_val.shape[0] - int(np.sum(y_val))} negative)")

    # Model training with HistGradientBoostingClassifier
    clf = HistGradientBoostingClassifier(
        max_iter=150,
        max_depth=6,
        learning_rate=0.08,
        min_samples_leaf=20,
        class_weight="balanced",
        random_state=42,
    )

    clf.fit(X_train, y_train)

    train_probs = clf.predict_proba(X_train)[:, 1]
    val_probs = clf.predict_proba(X_val)[:, 1]

    train_auc = float(roc_auc_score(y_train, train_probs))
    val_auc = float(roc_auc_score(y_val, val_probs))

    print(f"\nModel Performance:")
    print(f"  Train ROC-AUC: {train_auc:.6f}")
    print(f"  Val ROC-AUC:   {val_auc:.6f}")

    weights_path = models_dir / "scorer.joblib"
    joblib.dump(clf, weights_path)
    print(f"\nSaved weights to: {weights_path}")

    structure_info = {
        "model_type": "HistGradientBoostingClassifier",
        "hyperparameters": {
            "max_iter": 150,
            "max_depth": 6,
            "learning_rate": 0.08,
            "min_samples_leaf": 20,
            "class_weight": "balanced",
            "random_state": 42,
        },
        "feature_names": FEATURE_NAMES,
        "train_sequences": train_seqs_used,
        "val_sequences": val_seqs_used,
        "metrics": {
            "train_roc_auc": train_auc,
            "val_roc_auc": val_auc,
        },
    }

    structure_path = models_dir / "model_structure.json"
    with open(structure_path, "w", encoding="utf-8") as f:
        json.dump(structure_info, f, indent=2)
    print(f"Saved structure to: {structure_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train learned re-scorer")
    parser.add_argument("--dataset-dir", type=str, default="../OrbitSight_Dataset")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--models-dir", type=str, default="models")
    args = parser.parse_args()

    train_scorer(
        Path(args.dataset_dir).resolve(),
        Path(args.config).resolve(),
        Path(args.models_dir).resolve(),
    )


if __name__ == "__main__":
    main()
