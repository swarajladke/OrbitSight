"""Evaluate post-hoc box resizing arms (Arm 0, Arm 1, Arm 2) against shipped baseline and oracle."""

import argparse
import copy
import csv
import math
from pathlib import Path
import time
from typing import Any, Dict, List, Tuple

import numpy as np
from tabulate import tabulate

from src.common import infer_resolution, load_events, sequence_name_from_npy
from src.metrics import IOU_THRESHOLD, compute_ap, compute_prf1, evaluate_sequence, iou, windows_overlap
from src.pipeline import run_sequence
from src.scoreboard import load_pred_file, load_yaml_config


def load_gt_rows(gt_path: Path) -> List[Tuple[int, int, int, int, int, int]]:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Box Resizing Arms.")
    parser.add_argument("--dataset-dir", type=str, default="../OrbitSight_Dataset/Training_sets")
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir).resolve()
    base_cfg = load_yaml_config(Path(args.config))

    gt_files = sorted(list(dataset_dir.rglob("*_bb_windows_40ms.txt")))
    gt_files = [f for f in gt_files if "Training" in str(f)]

    arms = {
        "Arm 0 (Control)": {"box_regressor_mode": "none"},
        "Arm 1 (Least-Squares)": {"box_regressor_mode": "arm1", "box_regressor_arm1_path": "models/box_regressor_arm1.joblib"},
        "Arm 2 (Learned HGBR)": {"box_regressor_mode": "arm2", "box_regressor_arm2_path": "models/box_regressor_arm2.joblib"},
    }

    arm_predictions: Dict[str, Dict[str, List[Tuple[int, int, int, int, int, int, float]]]] = {a: {} for a in arms}
    arm_latencies: Dict[str, List[float]] = {a: [] for a in arms}
    total_windows_list: List[int] = []

    # Run inference for each arm across all 17 sequences
    for gt_f in gt_files:
        seq_name = gt_f.name.replace("_bb_windows_40ms.txt", "")
        npy_matches = list(gt_f.parent.glob(f"{seq_name}_labeled_events.npy"))
        if not npy_matches:
            npy_matches = list(dataset_dir.rglob(f"{seq_name}_labeled_events.npy"))
        if not npy_matches:
            continue
        npy_f = npy_matches[0]
        events = load_events(npy_f)

        width, height = infer_resolution(seq_name, events[:, 0], events[:, 1])

        for arm_name, arm_cfg_override in arms.items():
            cfg_copy = copy.deepcopy(base_cfg)
            for k, v in arm_cfg_override.items():
                cfg_copy[k] = v
                for s in ["EVK4", "DVX", "DAVIS"]:
                    if s in cfg_copy:
                        cfg_copy[s][k] = v

            t0 = time.perf_counter()
            preds, n_win = run_sequence(events, width, height, cfg_copy)
            t1 = time.perf_counter()

            arm_predictions[arm_name][seq_name] = preds
            arm_latencies[arm_name].append((t1 - t0) * 1000.0)
            if arm_name == "Arm 0 (Control)":
                total_windows_list.append(n_win)

    total_windows = sum(total_windows_list)

    # Compute baseline reference predictions for parity and upgrade/downgrade counting
    arm0_preds_dict = arm_predictions["Arm 0 (Control)"]

    arm_summary_results = []
    arm_sensor_breakdowns: Dict[str, Dict[str, Dict[str, Any]]] = {a: {"EVK4": [], "DVX": [], "DAVIS": []} for a in arms}

    for arm_name in arms:
        seq_results = []
        n_upgraded_total = 0
        n_downgraded_total = 0

        for gt_f in gt_files:
            seq_name = gt_f.name.replace("_bb_windows_40ms.txt", "")
            sensor = "EVK4" if "EVK4" in seq_name else ("DVX" if "DVX" in seq_name else "DAVIS")
            gt_rows = load_gt_rows(gt_f)
            preds = arm_predictions[arm_name][seq_name]
            base_preds = arm0_preds_dict[seq_name]

            # Upgrades / downgrades vs Arm 0
            gt_by_w = {}
            for gt in gt_rows:
                gt_by_w.setdefault(gt[0], []).append(gt)

            for p_curr, p_base in zip(preds, base_preds):
                ws_p, we_p, cx_p, cy_p, w_curr, h_curr, _ = p_curr
                _, _, _, _, w_base, h_base, _ = p_base
                m_gts = gt_by_w.get(ws_p, [])
                if not m_gts:
                    m_gts = [gt for gt in gt_rows if windows_overlap(ws_p, we_p, gt[0], gt[1])]
                if m_gts:
                    target_gt = m_gts[0]
                    ws_g, we_g, cx_g, cy_g, w_g, h_g = target_gt
                    iou_base = iou((cx_p, cy_p, w_base, h_base), (cx_g, cy_g, w_g, h_g))
                    iou_curr = iou((cx_p, cy_p, w_curr, h_curr), (cx_g, cy_g, w_g, h_g))
                    if iou_base < IOU_THRESHOLD and iou_curr >= IOU_THRESHOLD:
                        n_upgraded_total += 1
                    elif iou_base >= IOU_THRESHOLD and iou_curr < IOU_THRESHOLD:
                        n_downgraded_total += 1

            ev = evaluate_sequence(gt_rows, preds)
            seq_results.append(ev)
            arm_sensor_breakdowns[arm_name][sensor].append(ev)

        tot_tp = sum(r["tp"] for r in seq_results)
        tot_fp = sum(r["fp"] for r in seq_results)
        tot_fn = sum(r["fn"] for r in seq_results)
        p, r, f1 = compute_prf1(tot_tp, tot_fp, tot_fn)
        aps = [r["ap"] for r in seq_results if not np.isnan(r["ap"])]
        m_ap = float(np.mean(aps)) if aps else 0.0

        arm_summary_results.append({
            "name": arm_name,
            "mAP": m_ap,
            "P": p,
            "R": r,
            "F1": f1,
            "TP": tot_tp,
            "FP": tot_fp,
            "FN": tot_fn,
            "upgrades": n_upgraded_total,
            "downgrades": n_downgraded_total,
        })

    # Summary table
    print("=" * 115)
    print("  POST-HOC BOX SIZING: THREE ARMS vs ORACLE CEILING (Train-17)")
    print("=" * 115)
    table_rows = []
    for res in arm_summary_results:
        table_rows.append([
            res["name"],
            f"{res['mAP']:.6f}",
            f"{res['P']:.6f}",
            f"{res['R']:.6f}",
            f"{res['F1']:.6f}",
            res["TP"],
            res["FP"],
            res["FN"],
            res["upgrades"],
            res["downgrades"],
        ])
    # Add Oracle row
    table_rows.append([
        "Oracle ceiling",
        "0.318067",
        "0.703339",
        "0.551007",
        "0.617923",
        8426,
        3554,
        6866,
        3398,
        32,
    ])

    print(tabulate(table_rows, headers=["Arm", "train mAP", "P", "R", "F1", "TP", "FP", "FN", "upgrades", "downgrades"], tablefmt="github"))
    print()

    # Per-sensor breakdown for best arm
    best_arm_res = max(arm_summary_results, key=lambda x: x["mAP"])
    best_arm_name = best_arm_res["name"]

    print("=" * 115)
    print(f"  PER-SENSOR BREAKDOWN FOR BEST ARM: {best_arm_name}")
    print("=" * 115)
    s_table = []
    for s_name in ["EVK4", "DVX", "DAVIS"]:
        base_recs = arm_sensor_breakdowns["Arm 0 (Control)"][s_name]
        best_recs = arm_sensor_breakdowns[best_arm_name][s_name]

        b_tp = sum(r["tp"] for r in base_recs)
        b_fp = sum(r["fp"] for r in base_recs)
        b_fn = sum(r["fn"] for r in base_recs)
        b_p, b_r, b_f1 = compute_prf1(b_tp, b_fp, b_fn)
        b_aps = [r["ap"] for r in base_recs if not np.isnan(r["ap"])]
        b_map = float(np.mean(b_aps)) if b_aps else 0.0

        o_tp = sum(r["tp"] for r in best_recs)
        o_fp = sum(r["fp"] for r in best_recs)
        o_fn = sum(r["fn"] for r in best_recs)
        o_p, o_r, o_f1 = compute_prf1(o_tp, o_fp, o_fn)
        o_aps = [r["ap"] for r in best_recs if not np.isnan(r["ap"])]
        o_map = float(np.mean(o_aps)) if o_aps else 0.0

        s_table.append([f"{s_name} Baseline", f"{b_map:.6f}", f"{b_p:.6f}", f"{b_r:.6f}", f"{b_f1:.6f}", b_tp, b_fp, b_fn])
        s_table.append([f"{s_name} {best_arm_name}", f"{o_map:.6f}", f"{o_p:.6f}", f"{o_r:.6f}", f"{o_f1:.6f}", o_tp, o_fp, o_fn])
        s_table.append([f"{s_name} Delta", f"{o_map - b_map:+.6f}", f"{o_p - b_p:+.6f}", f"{o_r - b_r:+.6f}", f"{o_f1 - b_f1:+.6f}", o_tp - b_tp, o_fp - b_fp, o_fn - b_fn])

    print(tabulate(s_table, headers=["Sensor & Arm", "mAP@0.5", "Precision", "Recall", "F1 Score", "TP", "FP", "FN"], tablefmt="github"))
    print()

    # Verify column parity: confirm centroids and confidences are byte-identical
    parity_errors = 0
    total_preds_checked = 0
    for gt_f in gt_files:
        seq_name = gt_f.name.replace("_bb_windows_40ms.txt", "")
        p0 = arm_predictions["Arm 0 (Control)"][seq_name]
        pb = arm_predictions[best_arm_name][seq_name]
        if len(p0) != len(pb):
            parity_errors += 1
            continue
        for row0, rowb in zip(p0, pb):
            # ws, we, cx, cy, w, h, conf
            if row0[0] != rowb[0] or row0[1] != rowb[1]: # timestamps
                parity_errors += 1
            if row0[2] != rowb[2] or row0[3] != rowb[3]: # cx, cy
                parity_errors += 1
            if row0[6] != rowb[6]: # confidence
                parity_errors += 1
            total_preds_checked += 1

    print("=" * 115)
    print("  CENTROID & CONFIDENCE PARITY VERIFICATION")
    print("=" * 115)
    if parity_errors == 0:
        print(f"[CONFIRMED] Centroids, window timestamps, and confidences are BYTE-IDENTICAL across all {total_preds_checked} predictions between {best_arm_name} and Baseline!")
    else:
        print(f"[VIOLATION] Found {parity_errors} discrepancies in centroids or confidences!")
    print()

    # Latency evaluation (Step 1d)
    print("=" * 115)
    print("  MEASURED LATENCY OVERHEAD (STEP 1d)")
    print("=" * 115)
    arm0_total_ms = sum(arm_latencies["Arm 0 (Control)"])
    best_total_ms = sum(arm_latencies[best_arm_name])
    arm0_ms_per_win = arm0_total_ms / total_windows if total_windows > 0 else 0.0
    best_ms_per_win = best_total_ms / total_windows if total_windows > 0 else 0.0
    delta_ms_per_win = best_ms_per_win - arm0_ms_per_win

    print(f"Total Temporal Windows Evaluated: {total_windows}")
    print(f"Arm 0 (Control) Latency:          {arm0_ms_per_win:.4f} ms/window ({arm0_total_ms:.2f} ms total)")
    print(f"{best_arm_name} Latency:         {best_ms_per_win:.4f} ms/window ({best_total_ms:.2f} ms total)")
    print(f"Incremental Resizing Overhead:    {delta_ms_per_win:+.4f} ms/window")
    print("=" * 115)


if __name__ == "__main__":
    main()
