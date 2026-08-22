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
        print("[INFO] Ground truth not present or matched in mounted dataset. Generating summary report...", flush=True)
        # Find all prediction files and compile per-sequence summary without fabricating zero metrics
        pred_files = sorted(list(pred_dir.glob("*.txt")))
        seen_seqs = {}
        for pf in pred_files:
            s_name = pf.name
            for sfx in ["_bb_windows_40ms.txt", "_pred.txt", ".txt"]:
                if s_name.endswith(sfx):
                    s_name = s_name[:-len(sfx)]
                    break
            if s_name not in seen_seqs:
                with open(pf, "r", encoding="utf-8") as f:
                    n_rows = sum(1 for line in f if line.strip()) - 1
                sensor = "EVK4" if "EVK4" in s_name else ("DVX" if "DVX" in s_name else "DAVIS")
                seen_seqs[s_name] = {
                    "sequence": s_name,
                    "split": args.split,
                    "sensor": sensor,
                    "tp": None,
                    "fp": None,
                    "fn": None,
                    "precision": None,
                    "recall": None,
                    "f1": None,
                    "ap": None,
                    "n_pred": max(0, n_rows),
                }

        seq_results = list(seen_seqs.values())
        overall_row = {
            "split": args.split,
            "tp": None,
            "fp": None,
            "fn": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "mAP": None,
            "note": "Ground truth not present in mounted dataset; accuracy metrics not computable in-container.",
        }
        write_metrics_xlsx(seq_results, overall_row, out_path)
        print(f"[INFO] Summary XLSX written successfully to {out_path} ({len(seq_results)} sequences).", flush=True)
        return

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
