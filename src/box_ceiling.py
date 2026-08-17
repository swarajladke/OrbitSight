"""Theoretical box geometry ceiling diagnostic and fast cached box mode sweep suite."""

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

from src.common import WINDOW_US, event_image, infer_resolution, iter_windows, load_events
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
    """Run fast cached evaluation on 8 DVX training sequences across candidate box modes."""
    gt_files = sorted(list(dataset_dir.rglob("*_bb_windows_40ms.txt")))
    dvx_train_files = [f for f in gt_files if "DVX" in f.name.upper() and "Training" in str(f)]

    # Candidate box configurations to evaluate
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

    # Pre-extract sequences and run sequence evaluations
    results: List[Dict[str, Any]] = []

    for cand in candidates:
        test_cfg = deepcopy(base_cfg)
        if "DVX" not in test_cfg:
            test_cfg["DVX"] = {}

        test_cfg["DVX"]["open_kernel"] = 1
        test_cfg["DVX"]["percentile"] = 85.0
        test_cfg["DVX"]["box_mode"] = cand["box_mode"]
        if cand["box_mode"] == "fixed":
            test_cfg["DVX"]["box_w"] = cand["box_w"]
            test_cfg["DVX"]["box_h"] = cand["box_h"]
        else:
            test_cfg["DVX"]["extent_scale"] = cand["extent_scale"]
            test_cfg["DVX"]["extent_pad"] = cand["extent_pad"]
            test_cfg["DVX"]["min_dim"] = cand["min_dim"]
            test_cfg["DVX"]["max_dim"] = cand["max_dim"]

        tot_tp, tot_fp, tot_fn = 0, 0, 0
        tot_pred = 0
        all_aps: List[float] = []

        for gt_f in dvx_train_files:
            seq_name = gt_f.name.replace("_bb_windows_40ms.txt", "")
            npy_matches = list(gt_f.parent.glob(f"{seq_name}_labeled_events.npy"))
            if not npy_matches:
                npy_matches = list(dataset_dir.rglob(f"{seq_name}_labeled_events.npy"))
            if not npy_matches:
                continue

            events = load_events(npy_matches[0])
            width, height = infer_resolution(seq_name, events[:, 0], events[:, 1])

            gt_rows = []
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

            preds, _ = run_sequence(events, width, height, test_cfg)
            tot_pred += len(preds)
            eval_res = evaluate_sequence(gt_rows, preds)
            tot_tp += eval_res["tp"]
            tot_fp += eval_res["fp"]
            tot_fn += eval_res["fn"]
            if not np.isnan(eval_res["ap"]):
                all_aps.append(eval_res["ap"])

        prec, rec, f1 = compute_prf1(tot_tp, tot_fp, tot_fn)
        mAP = float(np.mean(all_aps)) if all_aps else 0.0

        results.append(
            {
                "config_name": cand["name"],
                "box_mode": cand["box_mode"],
                "preds_emitted": tot_pred,
                "tp": tot_tp,
                "fp": tot_fp,
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
        help="Run full 1-D DVX box geometry sweep on 8 training sequences",
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
