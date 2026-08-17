"""Theoretical box geometry ceiling diagnostic and fast in-memory box mode sweep suite."""

import argparse
import csv
from copy import deepcopy
import math
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Tuple
import cv2
import numpy as np
from tabulate import tabulate

from src.common import WINDOW_US, event_image, infer_resolution, iter_windows, load_events, resolve_effective_config
from src.detector import detect_boxes
from src.metrics import compute_prf1, evaluate_sequence, iou
from src.nms import apply_nms
from src.pipeline import compute_confidence, run_sequence


def load_all_gt_boxes(dataset_dir: Path, split: str = "train") -> Dict[str, List[Tuple[float, float]]]:
    """Load all GT box dimensions (width, height) grouped by sensor family."""
    gt_files = sorted(list(dataset_dir.rglob("*_bb_windows_40ms.txt")))
    sensor_boxes: Dict[str, List[Tuple[float, float]]] = {
        "DAVIS": [],
        "DVX": [],
        "EVK4": [],
    }

    for gt_f in gt_files:
        seq_name = gt_f.name.replace("_bb_windows_40ms.txt", "")
        if split == "train" and "Training" not in str(gt_f):
            continue
        if split == "test" and "Testing" not in str(gt_f):
            continue

        if "EVK4" in seq_name.upper():
            sensor = "EVK4"
        elif "DVX" in seq_name.upper():
            sensor = "DVX"
        else:
            sensor = "DAVIS"

        with open(gt_f, "r", encoding="utf-8") as f:
            rdr = csv.DictReader(f, delimiter="\t")
            for r in rdr:
                sensor_boxes[sensor].append((float(r["width"]), float(r["height"])))

    return sensor_boxes


def compute_geometric_ceiling(
    gt_boxes: List[Tuple[float, float]],
    box_w: float,
    box_h: float,
    iou_thresh: float = 0.5,
) -> Tuple[float, float, float]:
    """Calculate the theoretical maximum achievable recall for a given fixed box geometry using metrics.iou."""
    if not gt_boxes:
        return 0.0, 0.0, 0.0

    bw = float(int(round(box_w)))
    bh = float(int(round(box_h)))

    ious: List[float] = []
    for gw, gh in gt_boxes:
        gw_f, gh_f = float(gw), float(gh)
        best_cand_iou = 0.0
        for dx in [-1.0, 0.0, 1.0]:
            for dy in [-1.0, 0.0, 1.0]:
                score = iou((dx, dy, bw, bh), (0.0, 0.0, gw_f, gh_f))
                if score > best_cand_iou:
                    best_cand_iou = score
        ious.append(best_cand_iou)

    ious_arr = np.array(ious)
    max_recall = float(np.mean(ious_arr >= iou_thresh))
    mean_iou = float(np.mean(ious_arr))
    median_iou = float(np.median(ious_arr))

    return max_recall, mean_iou, median_iou


def run_dvx_box_mode_evaluation(
    dataset_dir: Path, base_cfg: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Run in-memory evaluation on 8 DVX training sequences across candidate box modes."""
    gt_files = sorted(list(dataset_dir.rglob("*_bb_windows_40ms.txt")))
    dvx_train_files = [f for f in gt_files if "DVX" in f.name.upper() and "Training" in str(f)]

    candidates = [
        {"name": "fixed_18x18", "box_mode": "fixed", "box_w": 18.0, "box_h": 18.0},
        {"name": "fixed_14x15", "box_mode": "fixed", "box_w": 14.0, "box_h": 15.0},
        {"name": "fixed_13x13", "box_mode": "fixed", "box_w": 13.0, "box_h": 13.0},
        {"name": "fixed_12x12", "box_mode": "fixed", "box_w": 12.0, "box_h": 12.0},
        # Extent variations
        {"name": "extent_s1.0_p1.0", "box_mode": "extent", "extent_scale": 1.0, "extent_pad": 1.0, "min_dim": 8.0, "max_dim": 50.0},
        {"name": "extent_s1.0_p2.0", "box_mode": "extent", "extent_scale": 1.0, "extent_pad": 2.0, "min_dim": 8.0, "max_dim": 50.0},
        {"name": "extent_s1.0_p4.0", "box_mode": "extent", "extent_scale": 1.0, "extent_pad": 4.0, "min_dim": 8.0, "max_dim": 50.0},
        {"name": "extent_s1.2_p1.0", "box_mode": "extent", "extent_scale": 1.2, "extent_pad": 1.0, "min_dim": 8.0, "max_dim": 50.0},
        {"name": "extent_s1.2_p2.0", "box_mode": "extent", "extent_scale": 1.2, "extent_pad": 2.0, "min_dim": 8.0, "max_dim": 50.0},
        {"name": "extent_s1.2_p4.0", "box_mode": "extent", "extent_scale": 1.2, "extent_pad": 4.0, "min_dim": 8.0, "max_dim": 50.0},
        {"name": "extent_s1.5_p1.0", "box_mode": "extent", "extent_scale": 1.5, "extent_pad": 1.0, "min_dim": 8.0, "max_dim": 50.0},
        {"name": "extent_s1.5_p2.0", "box_mode": "extent", "extent_scale": 1.5, "extent_pad": 2.0, "min_dim": 8.0, "max_dim": 50.0},
        {"name": "extent_s1.5_p4.0", "box_mode": "extent", "extent_scale": 1.5, "extent_pad": 4.0, "min_dim": 8.0, "max_dim": 50.0},
    ]

    results_by_cand: Dict[str, Dict[str, Any]] = {
        cand["name"]: {
            "config_name": cand["name"],
            "box_mode": cand["box_mode"],
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "preds_emitted": 0,
            "aps": [],
        }
        for cand in candidates
    }

    # Step 1: Pre-extract raw candidate components for all 8 DVX training sequences
    print(f"[INFO] Extracting raw components for {len(dvx_train_files)} DVX training sequences...")

    dvx_cfg = deepcopy(base_cfg)
    if "DVX" not in dvx_cfg:
        dvx_cfg["DVX"] = {}
    dvx_cfg["DVX"]["open_kernel"] = 1
    dvx_cfg["DVX"]["percentile"] = 85.0
    eff = resolve_effective_config(dvx_cfg, "DVX")

    for gt_f in dvx_train_files:
        seq_name = gt_f.name.replace("_bb_windows_40ms.txt", "")
        npy_matches = list(gt_f.parent.glob(f"{seq_name}_labeled_events.npy"))
        if not npy_matches:
            npy_matches = list(dataset_dir.rglob(f"{seq_name}_labeled_events.npy"))
        if not npy_matches:
            continue

        events = load_events(npy_matches[0])
        width, height = infer_resolution(seq_name, events[:, 0], events[:, 1])

        gt_rows: List[Tuple[int, int, int, int, int, int]] = []
        with open(gt_f, "r", encoding="utf-8") as f:
            rdr = csv.DictReader(f, delimiter="\t")
            for r in rdr:
                gt_rows.append(
                    (
                        int(r["window_start_timestamp_us"]),
                        int(r["window_end_timestamp_us"]),
                        int(r["center_x"]),
                        int(r["center_y"]),
                        int(r["width"]),
                        int(r["height"]),
                    )
                )

        # Extract window raw components once
        window_raw_boxes: List[Tuple[int, int, List[Dict[str, float]]]] = []
        for w_start, w_end, w_events in iter_windows(events, window_us=WINDOW_US):
            count_img, _, _ = event_image(w_events, width, height)
            boxes = detect_boxes(count_img, width, height, dvx_cfg)
            window_raw_boxes.append((w_start, w_end, boxes))

        num_windows = len(window_raw_boxes)
        min_hits = int(eff.get("min_hits", 1))
        max_dist_frac = float(eff.get("max_dist_frac", 0.08))
        conf_min = float(eff.get("conf_min", 0.0))
        nms_stage = str(eff.get("nms_stage", "pipeline")).lower()
        nms_iou_val = eff.get("nms_iou", 0.3)
        diagonal = math.hypot(width, height)
        max_dist = max_dist_frac * diagonal

        # Step 2: In-memory evaluation across all candidate box geometries
        for cand in candidates:
            b_mode = cand["box_mode"]
            if b_mode == "fixed":
                cw = cand["box_w"]
                ch = cand["box_h"]
            else:
                scale_val = cand["extent_scale"]
                pad_val = cand["extent_pad"]
                min_d = cand["min_dim"]
                max_d = cand["max_dim"]

            preds: List[Tuple[int, int, int, int, int, int, float]] = []

            for w_idx in range(num_windows):
                w_start, w_end, boxes = window_raw_boxes[w_idx]
                if not boxes:
                    continue

                prev_boxes = window_raw_boxes[w_idx - 1][2] if w_idx > 0 else []
                next_boxes = window_raw_boxes[w_idx + 1][2] if w_idx < num_windows - 1 else []

                scored_cands: List[Dict[str, float]] = []

                for box in boxes:
                    hits = 1
                    has_prev = any(
                        math.hypot(box["center_x"] - p["center_x"], box["center_y"] - p["center_y"]) <= max_dist
                        for p in prev_boxes
                    )
                    if has_prev:
                        hits += 1

                    has_next = any(
                        math.hypot(box["center_x"] - n["center_x"], box["center_y"] - n["center_y"]) <= max_dist
                        for n in next_boxes
                    )
                    if has_next:
                        hits += 1

                    if min_hits >= 2 and hits < min_hits:
                        continue

                    # Adjust geometry
                    if b_mode == "fixed":
                        new_w = cw
                        new_h = ch
                    else:
                        raw_w = box["width"] * scale_val + 2.0 * pad_val
                        raw_h = box["height"] * scale_val + 2.0 * pad_val
                        new_w = max(min_d, min(max_d, raw_w))
                        new_h = max(min_d, min(max_d, raw_h))

                    cx = box["center_x"]
                    cy = box["center_y"]
                    conf = compute_confidence(box, hits, eff)

                    scored_cands.append({
                        "center_x": cx,
                        "center_y": cy,
                        "width": new_w,
                        "height": new_h,
                        "confidence": conf,
                    })

                if not scored_cands:
                    continue

                if nms_stage == "pipeline" and nms_iou_val is not None:
                    try:
                        scored_cands = apply_nms(scored_cands, float(nms_iou_val))
                    except (TypeError, ValueError):
                        pass

                if conf_min > 0.0:
                    scored_cands = [b for b in scored_cands if b["confidence"] >= conf_min]

                for b in scored_cands:
                    preds.append((
                        w_start,
                        w_end,
                        int(round(b["center_x"])),
                        int(round(b["center_y"])),
                        int(round(b["width"])),
                        int(round(b["height"])),
                        round(float(b["confidence"]), 4),
                    ))

            eval_res = evaluate_sequence(gt_rows, preds)
            c_entry = results_by_cand[cand["name"]]
            c_entry["preds_emitted"] += len(preds)
            c_entry["tp"] += eval_res["tp"]
            c_entry["fp"] += eval_res["fp"]
            c_entry["fn"] += eval_res["fn"]
            if not np.isnan(eval_res["ap"]):
                c_entry["aps"].append(eval_res["ap"])

    results: List[Dict[str, Any]] = []
    for cand in candidates:
        c_entry = results_by_cand[cand["name"]]
        prec, rec, f1 = compute_prf1(c_entry["tp"], c_entry["fp"], c_entry["fn"])
        mAP = float(np.mean(c_entry["aps"])) if c_entry["aps"] else 0.0
        results.append(
            {
                "config_name": cand["name"],
                "box_mode": cand["box_mode"],
                "preds_emitted": c_entry["preds_emitted"],
                "tp": c_entry["tp"],
                "fp": c_entry["fp"],
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "mAP": mAP,
            }
        )

    return results


def main() -> None:
    """CLI entrypoint for box geometry ceiling diagnostic."""
    parser = argparse.ArgumentParser(
        description="OrbitSight Theoretical Box Geometry Ceiling Diagnostic"
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="../OrbitSight_Dataset",
        help="Path to dataset root",
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to config file"
    )
    parser.add_argument(
        "--sweep-dvx",
        action="store_true",
        help="Run fast in-memory DVX box geometry sweep on 8 training sequences",
    )

    args = parser.parse_args()
    dataset_dir = Path(args.dataset_dir).resolve()

    print("==================================================")
    print("  THEORETICAL BOX GEOMETRY CEILING (GT ALONE)")
    print("==================================================")

    gt_boxes = load_all_gt_boxes(dataset_dir, split="train")

    ceiling_table = []

    # 1. DVX Baseline 18x18 vs optimal fixed
    dvx_18x18_ceil, dvx_18_miou, _ = compute_geometric_ceiling(gt_boxes["DVX"], 18.0, 18.0)
    dvx_14x15_ceil, dvx_14_miou, _ = compute_geometric_ceiling(gt_boxes["DVX"], 14.0, 15.0)
    dvx_13x13_ceil, dvx_13_miou, _ = compute_geometric_ceiling(gt_boxes["DVX"], 13.0, 13.0)
    dvx_12x12_ceil, dvx_12_miou, _ = compute_geometric_ceiling(gt_boxes["DVX"], 12.0, 12.0)

    ceiling_table.append(["DVX", "fixed", "18x18 (Current Config)", f"{dvx_18x18_ceil*100.0:.2f}%", f"{dvx_18_miou:.4f}"])
    ceiling_table.append(["DVX", "fixed", "14x15 (P75 Mean)", f"{dvx_14x15_ceil*100.0:.2f}%", f"{dvx_14_miou:.4f}"])
    ceiling_table.append(["DVX", "fixed", "13x13 (Median Height)", f"{dvx_13x13_ceil*100.0:.2f}%", f"{dvx_13_miou:.4f}"])
    ceiling_table.append(["DVX", "fixed", "12x12 (Median Width)", f"{dvx_12x12_ceil*100.0:.2f}%", f"{dvx_12_miou:.4f}"])

    # 2. EVK4 Fixed 52x56
    evk4_52x56_ceil, evk4_miou, _ = compute_geometric_ceiling(gt_boxes["EVK4"], 52.0, 56.0)
    ceiling_table.append(["EVK4", "fixed", "52x56 (Current Config)", f"{evk4_52x56_ceil*100.0:.2f}%", f"{evk4_miou:.4f}"])

    # 3. DAVIS Extent Reference
    davis_10x12_ceil, davis_miou, _ = compute_geometric_ceiling(gt_boxes["DAVIS"], 10.0, 12.0)
    ceiling_table.append(["DAVIS", "extent / fixed ref", "10x12", f"{davis_10x12_ceil*100.0:.2f}%", f"{davis_miou:.4f}"])

    print(
        tabulate(
            ceiling_table,
            headers=["Sensor", "Box Mode", "Configured Dimensions", "Max Achievable Recall (IoU>=0.5)", "Mean Centered IoU"],
            tablefmt="github",
        )
    )

    if args.sweep_dvx:
        print("\n==================================================")
        print("  DVX BOX MODE SWEEP (8 Training Sequences)")
        print("==================================================")
        from src.scoreboard import load_yaml_config

        cfg = load_yaml_config(Path(args.config))
        sweep_rows = run_dvx_box_mode_evaluation(dataset_dir, cfg)

        sw_table = [
            [
                r["config_name"],
                r["box_mode"],
                r["preds_emitted"],
                r["tp"],
                r["fp"],
                f"{r['precision']:.4f}",
                f"{r['recall']:.4f}",
                f"{r['f1']:.4f}",
                f"{r['mAP']:.4f}",
            ]
            for r in sweep_rows
        ]
        print(
            tabulate(
                sw_table,
                headers=["Config Variant", "Box Mode", "Preds", "TP", "FP", "Precision", "Recall", "F1", "mAP@0.5"],
                tablefmt="github",
            )
        )


if __name__ == "__main__":
    main()
