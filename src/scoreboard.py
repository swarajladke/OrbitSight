"""Authoritative scoreboard generation, prediction file scoring, and run history logging for OrbitSight."""

import argparse
import csv
import datetime
import hashlib
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Tuple
import numpy as np
from tabulate import tabulate

from src.common import (
    WINDOW_US,
    infer_resolution,
    load_events,
    print_effective_config,
)
from src.metrics import compute_prf1, evaluate_sequence
from src.pipeline import run_sequence


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


def compute_config_hash(cfg_path: Path) -> str:
    """Compute SHA256 hash of configuration file for tracking."""
    if not cfg_path.exists():
        return "UNKNOWN"
    sha = hashlib.sha256()
    with open(cfg_path, "rb") as f:
        sha.update(f.read())
    return sha.hexdigest()[:12]


def load_pred_file(pred_path: Path) -> List[Tuple[int, int, int, int, int, int, float]]:
    """Load predictions from written _pred.txt file."""
    if not pred_path.exists():
        return []
    rows: List[Tuple[int, int, int, int, int, int, float]] = []
    with open(pred_path, "r", encoding="utf-8") as f:
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
                    float(r["confidence"]),
                )
            )
    return rows


def evaluate_dataset_sequences(
    dataset_dir: Path,
    pred_dir: Path,
    cfg: Dict[str, Any],
    split_filter: str = "train",
    recompute: bool = False,
) -> List[Dict[str, Any]]:
    """Evaluate dataset sequences either by reading prediction files or recomputing."""
    gt_files = sorted(list(dataset_dir.rglob("*_bb_windows_40ms.txt")))
    sequence_results: List[Dict[str, Any]] = []

    for gt_f in gt_files:
        seq_name = gt_f.name.replace("_bb_windows_40ms.txt", "")
        split = "train" if "Training" in str(gt_f) else "test"

        if split_filter != "all" and split != split_filter:
            continue

        if "EVK4" in seq_name.upper():
            sensor = "EVK4"
        elif "DVX" in seq_name.upper():
            sensor = "DVX"
        else:
            sensor = "DAVIS"

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

        ms_per_win = 0.0

        if recompute:
            npy_matches = list(gt_f.parent.glob(f"{seq_name}_labeled_events.npy"))
            if not npy_matches:
                npy_matches = list(dataset_dir.rglob(f"{seq_name}_labeled_events.npy"))
            if not npy_matches:
                print(f"[WARN] Skipping {seq_name}: labeled events npy not found.", file=sys.stderr)
                continue

            npy_f = npy_matches[0]
            events = load_events(npy_f)
            width, height = infer_resolution(seq_name, events[:, 0], events[:, 1])

            start_t = time.perf_counter()
            pred_rows, num_windows = run_sequence(events, width, height, cfg, window_us=WINDOW_US)
            elapsed_ms = (time.perf_counter() - start_t) * 1000.0
            ms_per_win = elapsed_ms / num_windows if num_windows > 0 else 0.0
        else:
            pred_file = pred_dir / f"{seq_name}_pred.txt"
            if not pred_file.exists():
                print(f"[WARN] Prediction file not found: {pred_file}. Run infer.py first or use --recompute.", file=sys.stderr)
                pred_rows = []
            else:
                pred_rows = load_pred_file(pred_file)

        eval_res = evaluate_sequence(gt_rows, pred_rows)

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
                "precision": eval_res["precision"],
                "recall": eval_res["recall"],
                "f1": eval_res["f1"],
                "ap": eval_res["ap"],
                "ms_per_window": ms_per_win,
            }
        )

    return sequence_results


def generate_scoreboard_reports(
    seq_results: List[Dict[str, Any]], cfg_hash: str, tag: str
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Tuple[str, str]]]:
    """Generate structured metric tables and check zero-prediction assertion guard."""
    tot_gt = sum(r["gt_count"] for r in seq_results)
    tot_pred = sum(r["pred_count"] for r in seq_results)
    tot_tp = sum(r["tp"] for r in seq_results)
    tot_fp = sum(r["fp"] for r in seq_results)
    tot_fn = sum(r["fn"] for r in seq_results)

    valid_aps = [r["ap"] for r in seq_results if not np.isnan(r["ap"])]
    overall_mAP = float(np.mean(valid_aps)) if valid_aps else 0.0
    overall_prec, overall_rec, overall_f1 = compute_prf1(tot_tp, tot_fp, tot_fn)

    overall_metrics = {
        "mAP": overall_mAP,
        "precision": overall_prec,
        "recall": overall_rec,
        "f1": overall_f1,
        "tp": tot_tp,
        "fp": tot_fp,
        "fn": tot_fn,
    }

    # Zero-prediction Assertion Guard per sensor and split
    sensor_split_counts: Dict[Tuple[str, str], int] = {}
    for r in seq_results:
        key = (r["sensor"], r["split"])
        sensor_split_counts[key] = sensor_split_counts.get(key, 0) + r["pred_count"]

    zero_prediction_failures: List[Tuple[str, str]] = []
    for (sensor, split), count in sensor_split_counts.items():
        if count == 0:
            zero_prediction_failures.append((sensor, split))

    if zero_prediction_failures:
        print("\n==================================================")
        print("  [FAIL] ZERO PREDICTION ASSERTION GUARD TRIGGERED")
        print("==================================================")
        for s_name, s_split in zero_prediction_failures:
            print(f"  -> [FAIL] {s_name} ({s_split} split) emitted 0 predictions across all sequences!")
        print("==================================================\n")

    return overall_metrics, seq_results, zero_prediction_failures


def log_run_history(
    history_path: Path,
    tag: str,
    cfg_hash: str,
    split: str,
    metrics: Dict[str, Any],
) -> None:
    """Append structured run summary to run history CSV."""
    history_path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not history_path.exists()
    timestamp_str = datetime.datetime.now().isoformat()

    with open(history_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow([
                "timestamp",
                "tag",
                "config_hash",
                "split",
                "mAP",
                "precision",
                "recall",
                "f1",
                "tp",
                "fp",
                "fn",
            ])
        writer.writerow([
            timestamp_str,
            tag,
            cfg_hash,
            split,
            f"{metrics['mAP']:.6f}",
            f"{metrics['precision']:.6f}",
            f"{metrics['recall']:.6f}",
            f"{metrics['f1']:.6f}",
            metrics["tp"],
            metrics["fp"],
            metrics["fn"],
        ])


def main() -> None:
    """Main CLI entrypoint for authoritative scoreboard."""
    parser = argparse.ArgumentParser(
        description="OrbitSight Authoritative Scoreboard & Metrics Generator"
    )
    parser.add_argument(
        "--pred-dir",
        type=str,
        default="predictions",
        help="Path to predictions directory containing _pred.txt files",
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
        "--split",
        type=str,
        default="train",
        choices=["train", "test", "all"],
        help="Dataset split to score (default: train)",
    )
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Recompute predictions in-process instead of reading from --pred-dir",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="experiments/scoreboard.csv",
        help="Output scoreboard CSV path",
    )
    parser.add_argument(
        "--history",
        type=str,
        default="experiments/run_history.csv",
        help="Run history CSV path",
    )
    parser.add_argument(
        "--tag", type=str, default="untagged_run", help="Run tag or description"
    )

    args = parser.parse_args()

    cfg_path = Path(args.config)
    cfg = load_yaml_config(cfg_path)
    cfg_hash = compute_config_hash(cfg_path)
    dataset_dir = Path(args.dataset_dir).resolve()
    pred_dir = Path(args.pred_dir).resolve()

    print(f"\n================================================================================")
    print(f"  AUTHORITATIVE SCOREBOARD — TAG: {args.tag} (Split: {args.split.upper()}, Config Hash: {cfg_hash})")
    print(f"================================================================================")

    print_effective_config(cfg)

    seq_results = evaluate_dataset_sequences(
        dataset_dir, pred_dir, cfg, split_filter=args.split, recompute=args.recompute
    )
    overall_metrics, seq_results, zero_failures = generate_scoreboard_reports(seq_results, cfg_hash, args.tag)

    print(f"\nOVERALL METRICS ({args.split.upper()} Split — {len(seq_results)} Sequences):")
    overall_table = [
        [
            f"{overall_metrics['mAP']:.6f}",
            f"{overall_metrics['precision']:.6f}",
            f"{overall_metrics['recall']:.6f}",
            f"{overall_metrics['f1']:.6f}",
            overall_metrics["tp"],
            overall_metrics["fp"],
            overall_metrics["fn"],
        ]
    ]
    print(
        tabulate(
            overall_table,
            headers=["mAP@0.5", "Precision", "Recall", "F1", "TP", "FP", "FN"],
            tablefmt="github",
        )
    )

    # Per-sequence breakdown table
    breakdown_table = [
        [
            r["sequence"],
            r["sensor"],
            r["split"],
            r["gt_count"],
            r["pred_count"],
            f"{r['precision']:.4f}",
            f"{r['recall']:.4f}",
            f"{r['f1']:.4f}",
            f"{r['ap']:.4f}",
            f"{r['ms_per_window']:.2f}" if args.recompute else "N/A",
        ]
        for r in seq_results
    ]
    print("\nPER-SEQUENCE BREAKDOWN:")
    print(
        tabulate(
            breakdown_table,
            headers=["Sequence", "Sensor", "Split", "GT", "Preds", "Precision", "Recall", "F1", "AP@0.5", "ms/win"],
            tablefmt="github",
        )
    )

    # Export scoreboard CSV
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sequence", "sensor", "split", "gt_count", "pred_count", "precision", "recall", "f1", "ap", "ms_per_window"])
        for r in seq_results:
            writer.writerow([
                r["sequence"],
                r["sensor"],
                r["split"],
                r["gt_count"],
                r["pred_count"],
                f"{r['precision']:.6f}",
                f"{r['recall']:.6f}",
                f"{r['f1']:.6f}",
                f"{r['ap']:.6f}",
                f"{r['ms_per_window']:.2f}",
            ])

    print(f"\n[INFO] Scoreboard CSV saved to: {out_path}")

    # Log run history
    if args.history:
        history_path = Path(args.history)
        log_run_history(history_path, args.tag, cfg_hash, args.split, overall_metrics)
        print(f"[INFO] Run logged to history: {history_path}")

    if zero_failures:
        print(f"\n[ERROR] Exiting with code 1 due to zero predictions on sensors: {zero_failures}")
        sys.exit(1)


if __name__ == "__main__":
    main()
