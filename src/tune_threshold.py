"""Confidence threshold tuning, Top-K window candidate optimization, and confidence ranking AUC analysis."""

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

from src.common import (
    WINDOW_US,
    event_image,
    infer_resolution,
    iter_windows,
    load_events,
    print_effective_config,
    resolve_effective_config,
)
from src.detector import detect_boxes
from src.metrics import compute_ap, compute_prf1, evaluate_sequence, iou


def load_yaml_config(path: Path) -> Dict[str, Any]:
    """Load configuration file with zero-dependency fallback."""
    if not path.exists():
        return {}
    try:
        import yaml

        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except ImportError:
        cfg: Dict[str, Any] = {}
        curr_sec = ""
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                raw = line.split("#")[0].rstrip("\r\n")
                if not raw.strip():
                    continue
                indent = len(raw) - len(raw.lstrip(" "))
                stripped = raw.strip()
                if ":" not in stripped:
                    continue
                parts = stripped.split(":", 1)
                k = parts[0].strip()
                v = parts[1].strip()
                if indent == 0:
                    if not v:
                        curr_sec = k
                        cfg[k] = {}
                    else:
                        curr_sec = ""
                        try:
                            cfg[k] = float(v) if "." in v else int(v)
                        except ValueError:
                            cfg[k] = (
                                v.strip("'\"")
                                if v.lower() not in ("true", "false")
                                else v.lower() == "true"
                            )
                elif indent > 0 and curr_sec:
                    if not isinstance(cfg.get(curr_sec), dict):
                        cfg[curr_sec] = {}
                    try:
                        cfg[curr_sec][k] = float(v) if "." in v else int(v)
                    except ValueError:
                        cfg[curr_sec][k] = (
                            v.strip("'\"")
                            if v.lower() not in ("true", "false")
                            else v.lower() == "true"
                        )
        return cfg


def compute_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Compute Area Under ROC Curve (AUC) using trapezoidal rule."""
    if len(labels) == 0 or len(np.unique(labels)) < 2:
        return 0.5

    desc_indices = np.argsort(-scores)
    labels_sorted = labels[desc_indices]

    n_pos = float(np.sum(labels_sorted == 1))
    n_neg = float(np.sum(labels_sorted == 0))

    if n_pos == 0 or n_neg == 0:
        return 0.5

    tps = np.cumsum(labels_sorted == 1)
    fps = np.cumsum(labels_sorted == 0)

    tpr = tps / n_pos
    fpr = fps / n_neg

    tpr = np.concatenate(([0.0], tpr))
    fpr = np.concatenate(([0.0], fpr))

    return float(np.trapz(tpr, fpr))


def run_conf_min_sweep(
    dataset_dir: Path, target_sensor: str, base_cfg: Dict[str, Any]
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Part 3 — Sweep conf_min over [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]."""
    conf_min_list = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

    gt_files = sorted(list(dataset_dir.rglob("*_bb_windows_40ms.txt")))
    sensor_files = [f for f in gt_files if target_sensor.upper() in f.name.upper()]

    sweep_rows: List[Dict[str, Any]] = []

    for c_min in conf_min_list:
        test_cfg = deepcopy(base_cfg)
        if target_sensor not in test_cfg:
            test_cfg[target_sensor] = {}
        test_cfg[target_sensor]["conf_min"] = c_min

        tot_tp, tot_fp, tot_fn = 0, 0, 0
        tot_pred = 0
        all_aps: List[float] = []

        for gt_f in sensor_files:
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

            pred_rows: List[Tuple[int, int, int, int, int, int, float]] = []
            for w_start, w_end, w_events in iter_windows(events, window_us=WINDOW_US):
                count_img, _, _ = event_image(w_events, width, height)
                boxes = detect_boxes(count_img, width, height, test_cfg)
                for b in boxes:
                    conf = b.get("confidence", 0.01)
                    if conf >= c_min:
                        pred_rows.append(
                            (
                                w_start,
                                w_end,
                                int(round(b["center_x"])),
                                int(round(b["center_y"])),
                                int(round(b["width"])),
                                int(round(b["height"])),
                                conf,
                            )
                        )

            tot_pred += len(pred_rows)
            eval_res = evaluate_sequence(gt_rows, pred_rows)
            tot_tp += eval_res["tp"]
            tot_fp += eval_res["fp"]
            tot_fn += eval_res["fn"]
            if not np.isnan(eval_res["ap"]):
                all_aps.append(eval_res["ap"])

        prec, rec, f1 = compute_prf1(tot_tp, tot_fp, tot_fn)
        mAP = float(np.mean(all_aps)) if all_aps else 0.0

        sweep_rows.append(
            {
                "conf_min": c_min,
                "preds_emitted": tot_pred,
                "tp": tot_tp,
                "fp": tot_fp,
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "mAP": mAP,
            }
        )

    # Optimal analysis
    best_f1_row = max(sweep_rows, key=lambda x: x["f1"])
    best_mAP_row = max(sweep_rows, key=lambda x: x["mAP"])

    mAP_at_best_f1 = best_f1_row["mAP"]
    mAP_sacrificed = best_mAP_row["mAP"] - mAP_at_best_f1

    # Check joint single value optimization
    joint_opt_found = False
    joint_conf_min = None
    max_f1 = best_f1_row["f1"]
    max_mAP = best_mAP_row["mAP"]

    for r in sweep_rows:
        if max_f1 > 0 and max_mAP > 0:
            if (max_f1 - r["f1"]) / max_f1 <= 0.10 and (max_mAP - r["mAP"]) / max_mAP <= 0.05:
                joint_opt_found = True
                joint_conf_min = r["conf_min"]
                break

    summary = {
        "best_f1_conf_min": best_f1_row["conf_min"],
        "best_f1_val": best_f1_row["f1"],
        "mAP_sacrificed": mAP_sacrificed,
        "best_mAP_conf_min": best_mAP_row["conf_min"],
        "best_mAP_val": best_mAP_row["mAP"],
        "joint_opt_found": joint_opt_found,
        "joint_conf_min": joint_conf_min,
    }

    return sweep_rows, summary


def run_topk_sweep(
    dataset_dir: Path, target_sensor: str, base_cfg: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Part 3 — Sweep max_candidates_per_window over [1, 2, 3, 5, 8, 15, None]."""
    k_list = [1, 2, 3, 5, 8, 15, None]

    gt_files = sorted(list(dataset_dir.rglob("*_bb_windows_40ms.txt")))
    sensor_files = [f for f in gt_files if target_sensor.upper() in f.name.upper()]

    sweep_rows: List[Dict[str, Any]] = []

    for k_val in k_list:
        test_cfg = deepcopy(base_cfg)
        if target_sensor not in test_cfg:
            test_cfg[target_sensor] = {}
        test_cfg[target_sensor]["max_candidates_per_window"] = k_val

        tot_tp, tot_fp, tot_fn = 0, 0, 0
        tot_pred = 0
        all_aps: List[float] = []

        for gt_f in sensor_files:
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

            pred_rows: List[Tuple[int, int, int, int, int, int, float]] = []
            for w_start, w_end, w_events in iter_windows(events, window_us=WINDOW_US):
                count_img, _, _ = event_image(w_events, width, height)
                boxes = detect_boxes(count_img, width, height, test_cfg)
                for b in boxes:
                    pred_rows.append(
                        (
                            w_start,
                            w_end,
                            int(round(b["center_x"])),
                            int(round(b["center_y"])),
                            int(round(b["width"])),
                            int(round(b["height"])),
                            float(b.get("confidence", 0.01)),
                        )
                    )

            tot_pred += len(pred_rows)
            eval_res = evaluate_sequence(gt_rows, pred_rows)
            tot_tp += eval_res["tp"]
            tot_fp += eval_res["fp"]
            tot_fn += eval_res["fn"]
            if not np.isnan(eval_res["ap"]):
                all_aps.append(eval_res["ap"])

        prec, rec, f1 = compute_prf1(tot_tp, tot_fp, tot_fn)
        mAP = float(np.mean(all_aps)) if all_aps else 0.0

        sweep_rows.append(
            {
                "max_k": "None" if k_val is None else str(k_val),
                "preds_emitted": tot_pred,
                "tp": tot_tp,
                "fp": tot_fp,
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "mAP": mAP,
            }
        )

    return sweep_rows


def run_confidence_auc_analysis(
    dataset_dir: Path, target_sensor: str, base_cfg: Dict[str, Any]
) -> Dict[str, Any]:
    """Part 4 — Measure TP vs FP confidence distributions, AUC, and feature ablations."""
    gt_files = sorted(list(dataset_dir.rglob("*_bb_windows_40ms.txt")))
    sensor_files = [f for f in gt_files if target_sensor.upper() in f.name.upper()]

    tp_confs: List[float] = []
    fp_confs: List[float] = []

    all_labels: List[int] = []
    all_full_scores: List[float] = []
    all_density_scores: List[float] = []
    all_event_scores: List[float] = []
    all_compactness_scores: List[float] = []

    for gt_f in sensor_files:
        seq_name = gt_f.name.replace("_bb_windows_40ms.txt", "")
        npy_matches = list(gt_f.parent.glob(f"{seq_name}_labeled_events.npy"))
        if not npy_matches:
            npy_matches = list(dataset_dir.rglob(f"{seq_name}_labeled_events.npy"))
        if not npy_matches:
            continue

        events = load_events(npy_matches[0])
        width, height = infer_resolution(seq_name, events[:, 0], events[:, 1])

        gt_by_window: Dict[int, List[Tuple[float, float, float, float]]] = {}
        with open(gt_f, "r", encoding="utf-8") as f:
            rdr = csv.DictReader(f, delimiter="\t")
            for r in rdr:
                ws = int(r["window_start_timestamp_us"])
                gt_by_window.setdefault(ws, []).append(
                    (
                        float(r["center_x"]),
                        float(r["center_y"]),
                        float(r["width"]),
                        float(r["height"]),
                    )
                )

        for w_start, w_end, w_events in iter_windows(events, window_us=WINDOW_US):
            count_img, _, _ = event_image(w_events, width, height)
            boxes = detect_boxes(count_img, width, height, base_cfg)

            gt_boxes = gt_by_window.get(w_start, [])

            for b in boxes:
                cand_box = (b["center_x"], b["center_y"], b["width"], b["height"])
                conf = float(b.get("confidence", b.get("density", 0.01)))
                density = float(b.get("density", 0.0))
                ev_cnt = float(b.get("events", 0.0))
                compactness = float(b.get("aspect", 1.0))

                is_tp = any(iou(cand_box, g) >= 0.5 for g in gt_boxes)
                label_val = 1 if is_tp else 0

                if is_tp:
                    tp_confs.append(conf)
                else:
                    fp_confs.append(conf)

                all_labels.append(label_val)
                all_full_scores.append(conf)
                all_density_scores.append(density)
                all_event_scores.append(ev_cnt)
                all_compactness_scores.append(1.0 / max(0.1, compactness))

    y_true = np.array(all_labels)
    full_auc = compute_auc(y_true, np.array(all_full_scores))
    density_auc = compute_auc(y_true, np.array(all_density_scores))
    event_auc = compute_auc(y_true, np.array(all_event_scores))
    compact_auc = compute_auc(y_true, np.array(all_compactness_scores))

    tp_arr = np.array(tp_confs)
    fp_arr = np.array(fp_confs)

    return {
        "sensor": target_sensor,
        "tp_conf": {
            "mean": float(tp_arr.mean()) if len(tp_arr) else 0.0,
            "median": float(np.median(tp_arr)) if len(tp_arr) else 0.0,
        },
        "fp_conf": {
            "mean": float(fp_arr.mean()) if len(fp_arr) else 0.0,
            "median": float(np.median(fp_arr)) if len(fp_arr) else 0.0,
        },
        "auc_full": full_auc,
        "auc_ablation": {
            "density_alone": density_auc,
            "events_alone": event_auc,
            "compactness_alone": compact_auc,
        },
    }


def main() -> None:
    """Main CLI entrypoint for threshold tuning and confidence AUC analysis."""
    parser = argparse.ArgumentParser(
        description="OrbitSight Confidence Threshold & AUC Tuning Suite"
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

    args = parser.parse_args()

    cfg_path = Path(args.config)
    cfg = load_yaml_config(cfg_path)
    dataset_dir = Path(args.dataset_dir).resolve()

    print_effective_config(cfg)

    for sensor in ["DAVIS", "DVX", "EVK4"]:
        print(f"\n==================================================")
        print(f"  PART 3 & 4 DIAGNOSTICS FOR SENSOR: {sensor}")
        print("==================================================")

        sweep_rows, summary = run_conf_min_sweep(dataset_dir, sensor, cfg)
        print("\nPART 3: CONFIDENCE THRESHOLD (conf_min) SWEEP:")
        c_table = [
            [
                r["conf_min"],
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
                c_table,
                headers=["conf_min", "Preds", "TP", "FP", "Precision", "Recall", "F1", "mAP@0.5"],
                tablefmt="github",
            )
        )

        print(f"\nF1-Optimal conf_min: {summary['best_f1_conf_min']} (F1 = {summary['best_f1_val']:.4f}, mAP Sacrificed = {summary['mAP_sacrificed']:.4f})")
        print(f"mAP-Optimal conf_min: {summary['best_mAP_conf_min']} (mAP = {summary['best_mAP_val']:.4f})")
        print(f"Joint Optimal conf_min (within 10% F1 & 5% mAP): {summary['joint_conf_min']} (Found: {summary['joint_opt_found']})")

        topk_rows = run_topk_sweep(dataset_dir, sensor, cfg)
        print("\nPART 3: TOP-K CANDIDATES PER WINDOW (max_candidates_per_window) SWEEP:")
        k_table = [
            [
                r["max_k"],
                r["preds_emitted"],
                r["tp"],
                r["fp"],
                f"{r['precision']:.4f}",
                f"{r['recall']:.4f}",
                f"{r['f1']:.4f}",
                f"{r['mAP']:.4f}",
            ]
            for r in topk_rows
        ]
        print(
            tabulate(
                k_table,
                headers=["Max K", "Preds", "TP", "FP", "Precision", "Recall", "F1", "mAP@0.5"],
                tablefmt="github",
            )
        )

        auc_res = run_confidence_auc_analysis(dataset_dir, sensor, cfg)
        print("\nPART 4: CONFIDENCE AUC & FEATURE ABLATION ANALYSIS:")
        print(f"TP Confidence: Mean={auc_res['tp_conf']['mean']:.4f}, Median={auc_res['tp_conf']['median']:.4f}")
        print(f"FP Confidence: Mean={auc_res['fp_conf']['mean']:.4f}, Median={auc_res['fp_conf']['median']:.4f}")
        print(f"Full Confidence Scorer AUC: {auc_res['auc_full']:.4f}")
        print(f"Ablation AUC (Density Alone):    {auc_res['auc_ablation']['density_alone']:.4f}")
        print(f"Ablation AUC (Events Alone):     {auc_res['auc_ablation']['events_alone']:.4f}")
        print(f"Ablation AUC (Compactness Alone): {auc_res['auc_ablation']['compactness_alone']:.4f}")


if __name__ == "__main__":
    main()
