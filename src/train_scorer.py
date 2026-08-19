"""Train learned candidate re-scorer on training split sequences."""

import argparse
import json
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
from src.metrics import iou
from src.scoreboard import load_gt_file
from src.static_map import build_static_mask


def build_continuous_static_map(events: np.ndarray, width: int, height: int, window_us: int = WINDOW_US) -> np.ndarray:
    """Compute continuous window activity fraction map across the entire sequence."""
    if events.shape[0] == 0:
        return np.zeros((height, width), dtype=np.float32)

    t = events[:, 3]
    t_start = int(t[0])
    t_end = int(t[-1])
    num_windows = int(np.ceil((t_end - t_start + 1) / float(window_us)))
    if num_windows <= 0:
        return np.zeros((height, width), dtype=np.float32)

    x = events[:, 0].astype(np.int64)
    y = events[:, 1].astype(np.int64)
    w_idx = (t.astype(np.int64) - t_start) // window_us

    valid = (x >= 0) & (x < width) & (y >= 0) & (y < height) & (w_idx >= 0) & (w_idx < num_windows)
    x = x[valid]
    y = y[valid]
    w_idx = w_idx[valid]

    if len(x) == 0:
        return np.zeros((height, width), dtype=np.float32)

    pixel_idx = y * width + x
    combined_key = w_idx * (width * height) + pixel_idx
    unique_keys = np.unique(combined_key)
    unique_pixels = unique_keys % (width * height)

    active_counts = np.bincount(unique_pixels, minlength=width * height)
    return active_counts.reshape(height, width).astype(np.float32) / float(num_windows)


def extract_sequence_dataset(
    npy_path: Path,
    gt_path: Path,
    cfg: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, str]:
    """Extract candidate feature vectors and GT binary labels for a sequence."""
    seq_name = sequence_name_from_npy(npy_path)
    events = load_events(npy_path)
    width, height = infer_resolution(seq_name, events[:, 0], events[:, 1])

    gt_rows = load_gt_file(gt_path)

    # Build continuous static fraction map and discrete static mask
    static_frac_map = build_continuous_static_map(events, width, height)
    static_mask = static_frac_map >= float(cfg.get("static_thresh", 0.5))

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

    # Map GT boxes by start timestamp for O(1) matching
    gt_by_start: Dict[int, List[Tuple[int, int, int, int, int, int]]] = {}
    for g in gt_rows:
        gt_by_start.setdefault(g[0], []).append(g)

    num_w = len(window_records)
    for w_idx, (ws, we, boxes, count_img) in enumerate(window_records):
        if not boxes:
            continue

        prev_boxes = window_records[w_idx - 1][2] if w_idx > 0 else None
        next_boxes = window_records[w_idx + 1][2] if w_idx < num_w - 1 else None

        gt_matches = gt_by_start.get(ws, [])

        for b in boxes:
            feats = extract_candidate_features(
                b,
                prev_boxes,
                next_boxes,
                count_img,
                static_frac_map=static_frac_map,
            )
            feat_vec = [feats[name] for name in FEATURE_NAMES]

            # Determine binary label via IoU >= 0.5
            cx = float(b["center_x"])
            cy = float(b["center_y"])
            bw = float(b["width"])
            bh = float(b["height"])
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

    # Locate training ground truth files
    train_gt_files = sorted(list(dataset_dir.glob("Training/*/*_bb_windows_40ms.txt")))
    if not train_gt_files:
        train_gt_files = [f for f in sorted(list(dataset_dir.rglob("*_bb_windows_40ms.txt"))) if "Training" in str(f)]

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

    X_train = np.vstack(train_X_parts)
    y_train = np.concatenate(train_y_parts)

    X_val = np.vstack(val_X_parts) if val_X_parts else X_train[:100]
    y_val = np.concatenate(val_y_parts) if val_y_parts else y_train[:100]

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
