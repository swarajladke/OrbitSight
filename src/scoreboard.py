"""Authoritative scoreboard and run history logger for OrbitSight."""

import argparse
import csv
import hashlib
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Tuple
import numpy as np
from tabulate import tabulate

from src.common import (
    WINDOW_US,
    event_image,
    infer_resolution,
    iter_windows,
    load_events,
    sequence_name_from_npy,
)
from src.detector import detect_boxes
from src.metrics import compute_ap, compute_prf1, evaluate_sequence


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


def get_config_hash(path: Path) -> str:
    """Compute SHA256 hash of configuration file."""
    if not path.exists():
        return "NO_CONFIG_FILE"
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:12]


def run_full_inference_on_dataset(
    dataset_dir: Path, cfg: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Run current configuration across all dataset sequences."""
    gt_files = sorted(list(dataset_dir.rglob("*_bb_windows_40ms.txt")))
    sequence_results: List[Dict[str, Any]] = []

    for gt_f in gt_files:
        seq_name = gt_f.name.replace("_bb_windows_40ms.txt", "")
        split = "train" if "Training" in str(gt_f) else "test"

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

        start_time = time.perf_counter()
        window_boxes: List[Tuple[int, int, List[Dict[str, float]]]] = []

        for w_start, w_end, w_events in iter_windows(events, window_us=WINDOW_US):
            count_img, _, _ = event_image(w_events, width, height)
            boxes = detect_boxes(count_img, width, height, cfg)
            window_boxes.append((w_start, w_end, boxes))

        num_windows = len(window_boxes)
        diagonal = np.hypot(width, height)
        sensor_cfg = cfg.get(sensor, {}) if isinstance(cfg.get(sensor), dict) else {}
        min_hits = int(sensor_cfg.get("min_hits", cfg.get("min_hits", 2)))
        max_dist = 0.05 * diagonal

        pred_rows: List[Tuple[int, int, int, int, int, int, float]] = []

        for w_idx in range(num_windows):
            w_start, w_end, boxes = window_boxes[w_idx]
            if not boxes:
                continue

            prev_boxes = window_boxes[w_idx - 1][2] if w_idx > 0 else []
            next_boxes = (
                window_boxes[w_idx + 1][2] if w_idx < num_windows - 1 else []
            )

            for box in boxes:
                hits = 1
                if any(
                    np.hypot(
                        box["center_x"] - p["center_x"],
                        box["center_y"] - p["center_y"],
                    )
                    <= max_dist
                    for p in prev_boxes
                ):
                    hits += 1
                if any(
                    np.hypot(
                        box["center_x"] - n["center_x"],
                        box["center_y"] - n["center_y"],
                    )
                    <= max_dist
                    for n in next_boxes
                ):
                    hits += 1

                if min_hits >= 2 and hits < min_hits:
                    continue

                cx = int(round(box["center_x"]))
                cy = int(round(box["center_y"]))
                bw = int(round(box["width"]))
                bh = int(round(box["height"]))
                conf = min(1.0, max(0.01, box["density"] * (hits / 3.0)))

                pred_rows.append((w_start, w_end, cx, cy, bw, bh, conf))

        eval_res = evaluate_sequence(gt_rows, pred_rows)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        sequence_results.append(
            {
                "sequence": seq_name,
                "sensor": sensor,
                "split": split,
                "gt_count": len(gt_rows),
                "pred_count": len(pred_rows),
                "tp": eval_res["tp"],
                "fp": eval_res["fp"],
                "fn": eval_res["fn"],
                "precision": eval_res["precision"] if eval_res["precision"] is not None else 0.0,
                "recall": eval_res["recall"] if eval_res["recall"] is not None else 0.0,
                "f1": eval_res["f1"] if eval_res["f1"] is not None else 0.0,
                "ap": eval_res["ap"],
                "ms_per_window": (
                    elapsed_ms / num_windows if num_windows > 0 else 0.0
                ),
            }
        )

    return sequence_results


def aggregate_group(
    res_list: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Aggregate total TP, FP, FN, precision, recall, F1, and mean AP for a group."""
    if not res_list:
        return {
            "mAP": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "gt_count": 0,
            "pred_count": 0,
        }

    tp = sum(r["tp"] for r in res_list)
    fp = sum(r["fp"] for r in res_list)
    fn = sum(r["fn"] for r in res_list)
    gt_c = sum(r["gt_count"] for r in res_list)
    pred_c = sum(r["pred_count"] for r in res_list)

    prec, rec, f1 = compute_prf1(tp, fp, fn)
    aps = [r["ap"] for r in res_list if r["ap"] is not None and not np.isnan(r["ap"])]
    mAP = float(np.mean(aps)) if aps else 0.0

    return {
        "mAP": mAP,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "gt_count": gt_c,
        "pred_count": pred_c,
    }


def main() -> None:
    """Main CLI entrypoint for scoreboard."""
    parser = argparse.ArgumentParser(
        description="OrbitSight Authoritative Scoreboard & History Logger"
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to config file"
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="../OrbitSight_Dataset",
        help="Path to dataset directory",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="experiments/scoreboard.csv",
        help="Output scoreboard CSV file path",
    )
    parser.add_argument(
        "--history",
        type=str,
        default="experiments/run_history.csv",
        help="Run history CSV log file path",
    )
    parser.add_argument(
        "--tag", type=str, default="baseline_run", help="Tag label for the run"
    )

    args = parser.parse_args()

    cfg_path = Path(args.config)
    dataset_dir = Path(args.dataset_dir).resolve()
    cfg = load_yaml_config(cfg_path)
    cfg_hash = get_config_hash(cfg_path)

    seq_results = run_full_inference_on_dataset(dataset_dir, cfg)
    total_gt = sum(r["gt_count"] for r in seq_results)

    # 1. Overall
    overall = aggregate_group(seq_results)

    # 2. By Split
    train_res = [r for r in seq_results if r["split"] == "train"]
    test_res = [r for r in seq_results if r["split"] == "test"]
    split_train = aggregate_group(train_res)
    split_test = aggregate_group(test_res)

    # 3. By Sensor & Split
    sensor_split_tables: List[List[Any]] = []
    sensor_contrib: List[List[Any]] = []

    for sensor in ["DAVIS", "DVX", "EVK4"]:
        sensor_gt = sum(r["gt_count"] for r in seq_results if r["sensor"] == sensor)
        share = (sensor_gt / total_gt * 100.0) if total_gt > 0 else 0.0
        sensor_contrib.append([sensor, sensor_gt, f"{share:.2f}%"])

        for split_name in ["train", "test"]:
            sub = [
                r
                for r in seq_results
                if r["sensor"] == sensor and r["split"] == split_name
            ]
            agg = aggregate_group(sub)
            sensor_split_tables.append(
                [
                    sensor,
                    split_name,
                    len(sub),
                    agg["gt_count"],
                    agg["pred_count"],
                    f"{agg['mAP']:.6f}",
                    f"{agg['precision']:.6f}",
                    f"{agg['recall']:.6f}",
                    f"{agg['f1']:.6f}",
                    agg["tp"],
                    agg["fp"],
                    agg["fn"],
                ]
            )

    # 4. Per Sequence Table (Sorted by AP Ascending)
    seq_results.sort(key=lambda r: (0.0 if np.isnan(r["ap"]) else r["ap"]))

    per_seq_rows: List[List[Any]] = []
    for r in seq_results:
        ap_val = f"{r['ap']:.6f}" if not np.isnan(r["ap"]) else "N/A"
        per_seq_rows.append(
            [
                r["sequence"],
                r["sensor"],
                r["split"],
                r["gt_count"],
                r["pred_count"],
                f"{r['precision']:.6f}",
                f"{r['recall']:.6f}",
                f"{r['f1']:.6f}",
                ap_val,
                f"{r['ms_per_window']:.2f}",
            ]
        )

    # Print Tables
    print("\n================================================================================")
    print(f"  AUTHORITATIVE SCOREBOARD — TAG: {args.tag} (Config Hash: {cfg_hash})")
    print("================================================================================")

    print("\nOVERALL METRICS (Across All 21 Sequences):")
    print(
        tabulate(
            [
                [
                    f"{overall['mAP']:.6f}",
                    f"{overall['precision']:.6f}",
                    f"{overall['recall']:.6f}",
                    f"{overall['f1']:.6f}",
                    overall["tp"],
                    overall["fp"],
                    overall["fn"],
                ]
            ],
            headers=["mAP@0.5", "Precision", "Recall", "F1", "TP", "FP", "FN"],
            tablefmt="github",
        )
    )

    print("\nMETRICS BY SPLIT:")
    print(
        tabulate(
            [
                [
                    "Train (17)",
                    f"{split_train['mAP']:.6f}",
                    f"{split_train['precision']:.6f}",
                    f"{split_train['recall']:.6f}",
                    f"{split_train['f1']:.6f}",
                    split_train["tp"],
                    split_train["fp"],
                    split_train["fn"],
                ],
                [
                    "Test (4)",
                    f"{split_test['mAP']:.6f}",
                    f"{split_test['precision']:.6f}",
                    f"{split_test['recall']:.6f}",
                    f"{split_test['f1']:.6f}",
                    split_test["tp"],
                    split_test["fp"],
                    split_test["fn"],
                ],
            ],
            headers=["Split", "mAP@0.5", "Precision", "Recall", "F1", "TP", "FP", "FN"],
            tablefmt="github",
        )
    )

    print("\nMETRICS BY SENSOR AND SPLIT:")
    print(
        tabulate(
            sensor_split_tables,
            headers=[
                "Sensor",
                "Split",
                "Seqs",
                "GT Count",
                "Pred Count",
                "mAP@0.5",
                "Precision",
                "Recall",
                "F1",
                "TP",
                "FP",
                "FN",
            ],
            tablefmt="github",
        )
    )

    print("\nGT BOX CONTRIBUTION BY SENSOR:")
    print(
        tabulate(
            sensor_contrib,
            headers=["Sensor", "Total GT Boxes", "Share of Total GT"],
            tablefmt="github",
        )
    )

    print("\nPER SEQUENCE RESULTS (Sorted by AP Ascending):")
    print(
        tabulate(
            per_seq_rows,
            headers=[
                "Sequence",
                "Sensor",
                "Split",
                "GT",
                "Pred",
                "Precision",
                "Recall",
                "F1",
                "AP@0.5",
                "ms/win",
            ],
            tablefmt="github",
        )
    )

    # Save Scoreboard CSV
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "sequence",
                "sensor",
                "split",
                "gt_count",
                "pred_count",
                "precision",
                "recall",
                "f1",
                "ap",
                "ms_per_window",
            ]
        )
        for r in seq_results:
            writer.writerow(
                [
                    r["sequence"],
                    r["sensor"],
                    r["split"],
                    r["gt_count"],
                    r["pred_count"],
                    f"{r['precision']:.6f}",
                    f"{r['recall']:.6f}",
                    f"{r['f1']:.6f}",
                    f"{r['ap']:.6f}" if not np.isnan(r["ap"]) else "",
                    f"{r['ms_per_window']:.2f}",
                ]
            )
    print(f"\n[INFO] Scoreboard CSV saved to: {out_path}")

    # Append to Run History CSV
    hist_path = Path(args.history)
    hist_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not hist_path.exists()

    with open(hist_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(
                [
                    "timestamp",
                    "tag",
                    "config_hash",
                    "mAP",
                    "precision",
                    "recall",
                    "f1",
                    "tp",
                    "fp",
                    "fn",
                ]
            )
        writer.writerow(
            [
                time.strftime("%Y-%m-%d %H:%M:%S"),
                args.tag,
                cfg_hash,
                f"{overall['mAP']:.6f}",
                f"{overall['precision']:.6f}",
                f"{overall['recall']:.6f}",
                f"{overall['f1']:.6f}",
                overall["tp"],
                overall["fp"],
                overall["fn"],
            ]
        )
    print(f"[INFO] Run appended to history log: {hist_path}\n")


if __name__ == "__main__":
    main()
