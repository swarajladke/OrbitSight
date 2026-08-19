"""CLI wrapper to compute metrics across sequences and write Evaluation_Metrics.xlsx."""

import argparse
from pathlib import Path
import sys
from typing import Any, Dict, List
import numpy as np

from src.report_xlsx import write_metrics_xlsx
from src.scoreboard import evaluate_dataset_sequences


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Evaluation_Metrics.xlsx from predictions"
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="../OrbitSight_Dataset",
        help="Path to OrbitSight dataset directory",
    )
    parser.add_argument(
        "--pred-dir",
        type=str,
        default="predictions",
        help="Path to predictions directory",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="Evaluation_Metrics.xlsx",
        help="Output path for Excel report",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="all",
        choices=["all", "train", "test"],
        help="Split to evaluate (all, train, test)",
    )

    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir).resolve()
    pred_dir = Path(args.pred_dir).resolve()
    out_path = Path(args.out).resolve()

    print(f"Computing evaluation metrics from {pred_dir} against {dataset_dir}...", flush=True)
    seq_results = evaluate_dataset_sequences(
        dataset_dir,
        pred_dir,
        cfg={},
        split_filter=args.split,
        recompute=False,
    )

    if not seq_results:
        print("Error: No sequence results computed.", file=sys.stderr)
        sys.exit(1)

    tot_tp = sum(r["tp"] for r in seq_results)
    tot_fp = sum(r["fp"] for r in seq_results)
    tot_fn = sum(r["fn"] for r in seq_results)

    precision = tot_tp / (tot_tp + tot_fp) if (tot_tp + tot_fp) > 0 else 0.0
    recall = tot_tp / (tot_tp + tot_fn) if (tot_tp + tot_fn) > 0 else 0.0
    f1 = (2.0 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    valid_aps = [r["ap"] for r in seq_results if not np.isnan(r["ap"])]
    mAP = float(np.mean(valid_aps)) if valid_aps else 0.0

    overall_row = {
        "split": args.split,
        "tp": tot_tp,
        "fp": tot_fp,
        "fn": tot_fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mAP": mAP,
    }

    print(f"Overall Metrics ({len(seq_results)} sequences):", flush=True)
    print(f"  mAP@IoU0.5: {mAP:.6f}", flush=True)
    print(f"  Precision:  {precision:.6f}", flush=True)
    print(f"  Recall:     {recall:.6f}", flush=True)
    print(f"  F1:         {f1:.6f}", flush=True)
    print(f"  TP: {tot_tp}, FP: {tot_fp}, FN: {tot_fn}", flush=True)

    write_metrics_xlsx(seq_results, overall_row, out_path)


if __name__ == "__main__":
    main()
