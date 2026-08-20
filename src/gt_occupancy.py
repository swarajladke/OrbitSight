"""Measure GT occupancy per window, max achievable recall under k=1, and recall loss decomposition."""

import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from tabulate import tabulate

from src.metrics import evaluate_sequence, compute_prf1
from src.scoreboard import load_yaml_config, evaluate_dataset_sequences, generate_scoreboard_reports


def load_gt_rows(gt_path: Path) -> List[Tuple[int, int, int, int, int, int]]:
    rows = []
    with open(gt_path, "r", encoding="utf-8") as f:
        rdr = csv.DictReader(f, delimiter="\t")
        for r in rdr:
            rows.append((
                int(r["window_start_timestamp_us"]),
                int(r["window_end_timestamp_us"]),
                int(round(float(r["center_x"]))),
                int(round(float(r["center_y"]))),
                int(round(float(r["width"]))),
                int(round(float(r["height"]))),
            ))
    return rows


def load_pred_rows(pred_path: Path) -> List[Tuple[int, int, int, int, int, int, float]]:
    rows = []
    if not pred_path.exists():
        return rows
    with open(pred_path, "r", encoding="utf-8") as f:
        rdr = csv.DictReader(f, delimiter="\t")
        for r in rdr:
            rows.append((
                int(r["window_start_timestamp_us"]),
                int(r["window_end_timestamp_us"]),
                int(round(float(r["center_x"]))),
                int(round(float(r["center_y"]))),
                int(round(float(r["width"]))),
                int(round(float(r["height"]))),
                float(r["confidence"]),
            ))
    return rows


def analyze_sequence_occupancy(
    gt_path: Path, pred_path: Path, iou_thresh: float = 0.5
) -> Dict[str, Any]:
    # 1. Load GT rows
    gt_rows = load_gt_rows(gt_path)
    total_gt = len(gt_rows)
    if total_gt == 0:
        return {
            "total_gt": 0,
            "occupied_windows": 0,
            "counts": Counter(),
            "max_per_win": 0,
            "ceiling_recall": float("nan"),
            "actual_recall": float("nan"),
            "headroom": float("nan"),
        }

    # 2. Window occupancy
    by_win = Counter([r[0] for r in gt_rows])
    occupied_windows = len(by_win)
    box_counts = Counter(by_win.values())
    max_per_win = max(by_win.values()) if by_win else 0

    ceiling_recall = occupied_windows / total_gt

    # 3. Actual recall from predictions
    actual_recall = 0.0
    if pred_path.exists():
        preds = load_pred_rows(pred_path)
        m = evaluate_sequence(gt_rows, preds, iou_thr=iou_thresh)
        actual_recall = m["recall"]

    headroom = ceiling_recall - actual_recall

    return {
        "total_gt": total_gt,
        "occupied_windows": occupied_windows,
        "counts": box_counts,
        "max_per_win": max_per_win,
        "ceiling_recall": ceiling_recall,
        "actual_recall": actual_recall,
        "headroom": headroom,
    }


def main():
    parser = argparse.ArgumentParser(description="GT occupancy and k=1 recall ceiling measurement")
    parser.add_argument("--dataset-dir", type=Path, default=Path("../OrbitSight_Dataset"))
    parser.add_argument("--pred-dir", type=Path, default=Path("predictions_final2"))
    parser.add_argument("--iou", type=float, default=0.5)
    args = parser.parse_args()

    gt_files = sorted(list(args.dataset_dir.rglob("*_bb_windows_40ms.txt")))
    if not gt_files:
        raise FileNotFoundError(f"No GT files found in {args.dataset_dir}")

    per_seq_rows = []

    splits_data = {
        "train": {"gt": 0, "occ": 0, "counts": Counter()},
        "test": {"gt": 0, "occ": 0, "counts": Counter()},
        "all": {"gt": 0, "occ": 0, "counts": Counter()},
    }

    for gtf in gt_files:
        seq_name = gtf.name.replace("_bb_windows_40ms.txt", "")
        split = "train" if "Training" in str(gtf) else "test"
        pred_file = args.pred_dir / f"{seq_name}_pred.txt"

        stats = analyze_sequence_occupancy(gtf, pred_file, iou_thresh=args.iou)

        # Update aggregates
        splits_data[split]["gt"] += stats["total_gt"]
        splits_data[split]["occ"] += stats["occupied_windows"]
        splits_data[split]["counts"].update(stats["counts"])

        splits_data["all"]["gt"] += stats["total_gt"]
        splits_data["all"]["occ"] += stats["occupied_windows"]
        splits_data["all"]["counts"].update(stats["counts"])

        c = stats["counts"]
        c1 = c.get(1, 0)
        c2 = c.get(2, 0)
        c3 = c.get(3, 0)
        c4 = c.get(4, 0)
        c5plus = sum(cnt for k, cnt in c.items() if k >= 5)

        per_seq_rows.append([
            seq_name,
            split,
            stats["total_gt"],
            stats["occupied_windows"],
            c1,
            c2,
            c3,
            c4,
            c5plus,
            stats["max_per_win"],
            f"{stats['ceiling_recall']:.6f}",
            f"{stats['actual_recall']:.6f}",
            f"{stats['headroom']:.6f}",
        ])

    headers = [
        "Sequence", "Split", "GT Total", "Occupied Win",
        "1 Box", "2 Boxes", "3 Boxes", "4 Boxes", "5+ Boxes", "Max/Win",
        "Ceiling R (k=1)", "Actual R", "Headroom"
    ]

    print("\n" + "=" * 140)
    print("  PER-SEQUENCE GT OCCUPANCY & K=1 RECALL CEILING ANALYSIS")
    print("=" * 140)
    print(tabulate(per_seq_rows, headers=headers, tablefmt="github"))

    # Summary table by split
    agg_rows = []
    decomp_rows = []

    cfg = load_yaml_config(Path("config.yaml"))

    for s_name in ["train", "test", "all"]:
        seq_res = evaluate_dataset_sequences(args.dataset_dir, args.pred_dir, cfg, split_filter=s_name, recompute=False)
        overall, _, _ = generate_scoreboard_reports(seq_res, "dummy_hash", "occupancy_eval")
        
        tot_gt = splits_data[s_name]["gt"]
        tot_occ = splits_data[s_name]["occ"]
        ceil_r = tot_occ / tot_gt if tot_gt > 0 else 0.0
        act_r = overall["recall"]
        
        cap_loss = 1.0 - ceil_r
        other_loss = ceil_r - act_r
        headroom = ceil_r - act_r

        agg_rows.append([
            f"{s_name.upper()} ({len(seq_res)})",
            tot_gt,
            tot_occ,
            f"{ceil_r:.6f}",
            f"{act_r:.6f}",
            f"{headroom:.6f}",
        ])

        decomp_rows.append([
            f"{s_name.upper()}",
            f"{cap_loss:.6f}",
            f"{other_loss:.6f}",
            f"{(1.0 - act_r):.6f}",
        ])

    print("\n" + "=" * 90)
    print("  AGGREGATE SPLIT RECALL CEILINGS & HEADROOM")
    print("=" * 90)
    print(tabulate(agg_rows, headers=["Split", "Total GT", "Total Occupied Win", "Ceiling R (k=1)", "Actual Recall", "Headroom"], tablefmt="github"))

    print("\n" + "=" * 80)
    print("  RECALL LOSS DECOMPOSITION (Cap vs Suppression)")
    print("=" * 80)
    print(tabulate(decomp_rows, headers=["Split", "Cap Loss (1 - Ceiling)", "Suppression Loss (Ceil - Actual)", "Total Recall Deficit (1 - Actual)"], tablefmt="github"))
    print()


if __name__ == "__main__":
    main()
