"""Oracle bound on emitted-box resizing.

Measures the theoretical performance ceiling if all emitted bounding boxes
were resized to the exact width and height of the ground-truth box in the
same temporal window, leaving centroids, confidences, ranking, and NMS untouched.
"""

import argparse
import csv
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from tabulate import tabulate

from src.metrics import (
    IOU_THRESHOLD,
    compute_ap,
    compute_prf1,
    evaluate_sequence,
    iou,
    match_predictions,
    windows_overlap,
)
from src.scoreboard import load_pred_file


def load_gt_rows(gt_path: Path) -> List[Tuple[int, int, int, int, int, int]]:
    """Load ground truth bounding box rows from file."""
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


def oracle_resize_predictions(
    pred_rows: List[Tuple[int, int, int, int, int, int, float]],
    gt_rows: List[Tuple[int, int, int, int, int, int]],
) -> Tuple[List[Tuple[int, int, int, int, int, int, float]], int, int]:
    """Replace pred (width, height) with matched GT (width, height) in the same window.

    Returns:
        oracle_preds: modified prediction rows.
        n_upgraded: count of predictions moving from IoU < 0.5 to IoU >= 0.5.
        n_downgraded: count of predictions moving from IoU >= 0.5 to IoU < 0.5.
    """
    gt_by_window: Dict[int, List[Tuple[int, int, int, int, int, int]]] = {}
    for gt in gt_rows:
        gt_by_window.setdefault(gt[0], []).append(gt)

    oracle_preds: List[Tuple[int, int, int, int, int, int, float]] = []
    n_upgraded = 0
    n_downgraded = 0

    for pred in pred_rows:
        ws_p, we_p, cx_p, cy_p, w_p, h_p, conf_p = pred
        matching_gts = gt_by_window.get(ws_p, [])
        if not matching_gts:
            matching_gts = [gt for gt in gt_rows if windows_overlap(ws_p, we_p, gt[0], gt[1])]

        if matching_gts:
            # H1 proved at most 1 GT box per window
            target_gt = matching_gts[0]
            ws_g, we_g, cx_g, cy_g, w_g, h_g = target_gt

            iou_before = iou((float(cx_p), float(cy_p), float(w_p), float(h_p)),
                             (float(cx_g), float(cy_g), float(w_g), float(h_g)))
            iou_after = iou((float(cx_p), float(cy_p), float(w_g), float(h_g)),
                            (float(cx_g), float(cy_g), float(w_g), float(h_g)))

            if iou_before < IOU_THRESHOLD and iou_after >= IOU_THRESHOLD:
                n_upgraded += 1
            elif iou_before >= IOU_THRESHOLD and iou_after < IOU_THRESHOLD:
                n_downgraded += 1

            oracle_preds.append((ws_p, we_p, cx_p, cy_p, w_g, h_g, conf_p))
        else:
            # No GT in this window - prediction unmodified
            oracle_preds.append((ws_p, we_p, cx_p, cy_p, w_p, h_p, conf_p))

    return oracle_preds, n_upgraded, n_downgraded


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Oracle Box Resizing Bound.")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="../OrbitSight_Dataset/Training_sets",
        help="Path to OrbitSight training dataset directory",
    )
    parser.add_argument(
        "--pred-dir",
        type=str,
        default="predictions_o5",
        help="Path to directory containing frozen prediction files",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir).resolve()
    pred_dir = Path(args.pred_dir).resolve()

    gt_files = sorted(list(dataset_dir.glob("*_bb_windows_40ms.txt")))
    if not gt_files:
        gt_files = sorted(list(dataset_dir.rglob("*_bb_windows_40ms.txt")))
        gt_files = [f for f in gt_files if "Training" in str(f) or "train" in str(f).lower()]

    if not gt_files:
        print(f"Error: No GT files found in {dataset_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(gt_files)} training sequence ground truth files.")
    print(f"Evaluating oracle box resizing against prediction directory: {pred_dir}\n")

    baseline_seq_results = []
    oracle_seq_results = []

    total_upgraded = 0
    total_downgraded = 0

    sensor_stats = {
        "EVK4": {"base_tp": 0, "base_fp": 0, "base_fn": 0, "base_aps": [],
                 "orc_tp": 0, "orc_fp": 0, "orc_fn": 0, "orc_aps": [],
                 "upgraded": 0, "downgraded": 0, "preds": 0, "gt": 0},
        "DVX": {"base_tp": 0, "base_fp": 0, "base_fn": 0, "base_aps": [],
                "orc_tp": 0, "orc_fp": 0, "orc_fn": 0, "orc_aps": [],
                "upgraded": 0, "downgraded": 0, "preds": 0, "gt": 0},
        "DAVIS": {"base_tp": 0, "base_fp": 0, "base_fn": 0, "base_aps": [],
                  "orc_tp": 0, "orc_fp": 0, "orc_fn": 0, "orc_aps": [],
                  "upgraded": 0, "downgraded": 0, "preds": 0, "gt": 0},
    }

    for gt_f in gt_files:
        seq_name = gt_f.name.replace("_bb_windows_40ms.txt", "")
        sensor = "EVK4" if "EVK4" in seq_name else ("DVX" if "DVX" in seq_name else "DAVIS")

        pred_f = pred_dir / f"{seq_name}_pred.txt"
        if not pred_f.exists():
            pred_f = pred_dir / f"{seq_name}.txt"
        if not pred_f.exists():
            pred_f = pred_dir / f"{seq_name}_bb_windows_40ms.txt"
        if not pred_f.exists():
            print(f"[WARN] Prediction file for {seq_name} not found in {pred_dir}. Skipping.")
            continue

        gt_rows = load_gt_rows(gt_f)
        pred_rows = load_pred_file(pred_f)

        # Baseline evaluation
        base_eval = evaluate_sequence(gt_rows, pred_rows)
        baseline_seq_results.append((seq_name, sensor, base_eval))

        # Oracle resized evaluation
        orc_preds, n_up, n_down = oracle_resize_predictions(pred_rows, gt_rows)
        orc_eval = evaluate_sequence(gt_rows, orc_preds)
        oracle_seq_results.append((seq_name, sensor, orc_eval, n_up, n_down))

        total_upgraded += n_up
        total_downgraded += n_down

        # Accumulate per-sensor
        s_dict = sensor_stats[sensor]
        s_dict["gt"] += base_eval["n_gt"]
        s_dict["preds"] += base_eval["n_pred"]
        s_dict["base_tp"] += base_eval["tp"]
        s_dict["base_fp"] += base_eval["fp"]
        s_dict["base_fn"] += base_eval["fn"]
        if not np.isnan(base_eval["ap"]):
            s_dict["base_aps"].append(base_eval["ap"])

        s_dict["orc_tp"] += orc_eval["tp"]
        s_dict["orc_fp"] += orc_eval["fp"]
        s_dict["orc_fn"] += orc_eval["fn"]
        if not np.isnan(orc_eval["ap"]):
            s_dict["orc_aps"].append(orc_eval["ap"])

        s_dict["upgraded"] += n_up
        s_dict["downgraded"] += n_down

    # Compute overall baseline metrics
    b_tp = sum(r[2]["tp"] for r in baseline_seq_results)
    b_fp = sum(r[2]["fp"] for r in baseline_seq_results)
    b_fn = sum(r[2]["fn"] for r in baseline_seq_results)
    b_prec, b_rec, b_f1 = compute_prf1(b_tp, b_fp, b_fn)
    b_aps = [r[2]["ap"] for r in baseline_seq_results if not np.isnan(r[2]["ap"])]
    b_map = float(np.mean(b_aps)) if b_aps else 0.0

    # Compute overall oracle metrics
    o_tp = sum(r[2]["tp"] for r in oracle_seq_results)
    o_fp = sum(r[2]["fp"] for r in oracle_seq_results)
    o_fn = sum(r[2]["fn"] for r in oracle_seq_results)
    o_prec, o_rec, o_f1 = compute_prf1(o_tp, o_fp, o_fn)
    o_aps = [r[2]["ap"] for r in oracle_seq_results if not np.isnan(r[2]["ap"])]
    o_map = float(np.mean(o_aps)) if o_aps else 0.0

    # Print summary tables
    print("=" * 105)
    print("  ORACLE BOUND ON EMITTED-BOX RESIZING vs SHIPPED BASELINE (Train-17)")
    print("=" * 105)

    comp_table = [
        [
            "Shipped Baseline",
            f"{b_map:.6f}",
            f"{b_prec:.6f}",
            f"{b_rec:.6f}",
            f"{b_f1:.6f}",
            b_tp,
            b_fp,
            b_fn,
            "-",
            "-",
        ],
        [
            "Oracle Box-Resized",
            f"{o_map:.6f}",
            f"{o_prec:.6f}",
            f"{o_rec:.6f}",
            f"{o_f1:.6f}",
            o_tp,
            o_fp,
            o_fn,
            f"{total_upgraded}",
            f"{total_downgraded}",
        ],
        [
            "Delta (Oracle - Base)",
            f"{o_map - b_map:+.6f}",
            f"{o_prec - b_prec:+.6f}",
            f"{o_rec - b_rec:+.6f}",
            f"{o_f1 - b_f1:+.6f}",
            f"{o_tp - b_tp:+d}",
            f"{o_fp - b_fp:+d}",
            f"{o_fn - b_fn:+d}",
            f"{total_upgraded}",
            f"{total_downgraded}",
        ],
    ]

    print(
        tabulate(
            comp_table,
            headers=["Configuration", "mAP@0.5", "Precision", "Recall", "F1 Score", "TP", "FP", "FN", "IoU <0.5 -> >=0.5", "IoU >=0.5 -> <0.5"],
            tablefmt="github",
        )
    )
    print()

    # Per-sensor breakdown table
    print("=" * 105)
    print("  PER-SENSOR ORACLE BOX RESIZING BREAKDOWN")
    print("=" * 105)

    sensor_rows = []
    for s_name in ["EVK4", "DVX", "DAVIS"]:
        s_d = sensor_stats[s_name]
        sb_p, sb_r, sb_f1 = compute_prf1(s_d["base_tp"], s_d["base_fp"], s_d["base_fn"])
        so_p, so_r, so_f1 = compute_prf1(s_d["orc_tp"], s_d["orc_fp"], s_d["orc_fn"])
        sb_map = float(np.mean(s_d["base_aps"])) if s_d["base_aps"] else 0.0
        so_map = float(np.mean(s_d["orc_aps"])) if s_d["orc_aps"] else 0.0

        sensor_rows.append([
            f"{s_name} Baseline",
            s_d["preds"],
            s_d["gt"],
            f"{sb_map:.6f}",
            f"{sb_p:.6f}",
            f"{sb_r:.6f}",
            f"{sb_f1:.6f}",
            s_d["base_tp"],
            s_d["base_fp"],
            s_d["base_fn"],
            "-",
        ])
        sensor_rows.append([
            f"{s_name} Oracle",
            s_d["preds"],
            s_d["gt"],
            f"{so_map:.6f}",
            f"{so_p:.6f}",
            f"{so_r:.6f}",
            f"{so_f1:.6f}",
            s_d["orc_tp"],
            s_d["orc_fp"],
            s_d["orc_fn"],
            f"{s_d['upgraded']}",
        ])
        sensor_rows.append([
            f"{s_name} Delta",
            "-",
            "-",
            f"{so_map - sb_map:+.6f}",
            f"{so_p - sb_p:+.6f}",
            f"{so_rec - sb_r:+.6f}" if 'so_rec' in locals() else f"{so_r - sb_r:+.6f}",
            f"{so_f1 - sb_f1:+.6f}",
            f"{s_d['orc_tp'] - s_d['base_tp']:+d}",
            f"{s_d['orc_fp'] - s_d['base_fp']:+d}",
            f"{s_d['orc_fn'] - s_d['base_fn']:+d}",
            f"{s_d['upgraded']}",
        ])

    print(
        tabulate(
            sensor_rows,
            headers=["Sensor & Variant", "Preds", "GT", "mAP@0.5", "Precision", "Recall", "F1 Score", "TP", "FP", "FN", "Upgraded"],
            tablefmt="github",
        )
    )
    print()

    # Per-sequence breakdown
    print("=" * 105)
    print("  PER-SEQUENCE BREAKDOWN (AP@0.5 Baseline vs Oracle)")
    print("=" * 105)
    seq_table = []
    for i, (seq_name, sensor, base_eval) in enumerate(baseline_seq_results):
        _, _, orc_eval, n_up, n_down = oracle_seq_results[i]
        b_ap_val = base_eval["ap"]
        o_ap_val = orc_eval["ap"]
        delta_ap = o_ap_val - b_ap_val
        seq_table.append([
            seq_name,
            sensor,
            base_eval["n_gt"],
            base_eval["n_pred"],
            f"{b_ap_val:.6f}",
            f"{o_ap_val:.6f}",
            f"{delta_ap:+.6f}",
            f"{base_eval['tp']} -> {orc_eval['tp']}",
            n_up,
        ])

    print(
        tabulate(
            seq_table,
            headers=["Sequence Name", "Sensor", "GT", "Preds", "Base AP", "Oracle AP", "Delta AP", "TP Transition", "Upgraded"],
            tablefmt="github",
        )
    )
    print()

    # Decision Rule Assessment
    print("=" * 105)
    print("  DECISION RULE ASSESSMENT")
    print("=" * 105)
    print(f"Oracle Train mAP: {o_map:.6f} (Baseline: {b_map:.6f}, Delta: {o_map - b_map:+.6f})")
    if o_map >= 0.190:
        print("VERDICT: PROCEED TO STEP 1 (Learned regressor) — ceiling is >= 0.190.")
    elif 0.170 <= o_map < 0.190:
        print("VERDICT: MARGINAL (0.170 <= mAP < 0.190) — stopping for user judgement.")
    else:
        print("VERDICT: DEAD (mAP < 0.170) — box resizing alone cannot provide meaningful gains. Do not build.")
    print("=" * 105)


if __name__ == "__main__":
    main()
