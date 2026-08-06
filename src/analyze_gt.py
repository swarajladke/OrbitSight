"""Read-only ground-truth diagnostic tool for OrbitSight event-camera dataset.

Performs:
1. GT box size distribution per sequence & per sensor type.
2. Window grid alignment verification (t0 vs absolute grid).
3. Window occupancy and positive rate calculation.
4. Event-label signal-to-noise ratio and inside/outside box event distribution.
5. Prediction comparison against GT (size ratios & timestamp overlap).
"""

import argparse
from collections import Counter
import csv
import math
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from tabulate import tabulate

from src.common import infer_resolution, load_events, sequence_name_from_npy


def load_gt_rows(gt_path: Path) -> List[Tuple[int, int, int, int, int, int]]:
    """Load ground truth rows from _bb_windows_40ms.txt file."""
    rows: List[Tuple[int, int, int, int, int, int]] = []
    with open(gt_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
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


def load_pred_rows(
    pred_path: Path,
) -> List[Tuple[int, int, int, int, int, int, float]]:
    """Load prediction rows from _pred.txt or _bb_windows_40ms.txt file."""
    rows: List[Tuple[int, int, int, int, int, int, float]] = []
    with open(pred_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
            conf = (
                float(r["confidence"])
                if "confidence" in r and r["confidence"] is not None
                else 1.0
            )
            rows.append(
                (
                    int(r["window_start_timestamp_us"]),
                    int(r["window_end_timestamp_us"]),
                    int(r["center_x"]),
                    int(r["center_y"]),
                    int(r["width"]),
                    int(r["height"]),
                    conf,
                )
            )
    return rows


def compute_stats(data: np.ndarray) -> Dict[str, float]:
    """Compute summary statistics for a 1D float/int array."""
    if len(data) == 0:
        return {
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "p10": 0.0,
            "p50": 0.0,
            "p90": 0.0,
        }
    return {
        "mean": float(np.mean(data)),
        "std": float(np.std(data)),
        "min": float(np.min(data)),
        "max": float(np.max(data)),
        "p10": float(np.percentile(data, 10)),
        "p50": float(np.median(data)),
        "p90": float(np.percentile(data, 90)),
    }


def analyze_sequence(
    seq_name: str,
    split: str,
    gt_path: Path,
    npy_path: Optional[Path],
    pred_path: Optional[Path],
) -> Dict[str, Any]:
    """Analyze a single sequence across all 5 diagnostic categories."""
    gt_rows = load_gt_rows(gt_path)
    sensor_w, sensor_h = infer_resolution(seq_name)
    sensor_area = float(sensor_w * sensor_h)

    sensor_type = "UNKNOWN"
    seq_upper = seq_name.upper()
    if "DAVIS" in seq_upper:
        sensor_type = "DAVIS"
    elif "DVX" in seq_upper:
        sensor_type = "DVX"
    elif "EVK4" in seq_upper:
        sensor_type = "EVK4"

    # Analysis 1: GT Box Size Distribution
    n_gt_boxes = len(gt_rows)
    widths = np.array([r[4] for r in gt_rows], dtype=np.float64)
    heights = np.array([r[5] for r in gt_rows], dtype=np.float64)
    w_stats = compute_stats(widths)
    h_stats = compute_stats(heights)

    box_areas = widths * heights
    area_fracs = box_areas / sensor_area if sensor_area > 0 else np.zeros(0)
    mean_area_frac = float(np.mean(area_fracs)) if len(area_fracs) > 0 else 0.0

    # Analysis 2: Window Grid Alignment
    verdict = "N/A (no events)"
    modal_offset = 0
    num_distinct_offsets = 0
    modal_frac = 0.0
    abs_multiples = False
    duration_valid = True

    if npy_path and npy_path.exists():
        events = load_events(npy_path)
        if len(events) > 0 and n_gt_boxes > 0:
            t0 = int(events[0, 3])
            offsets = [(int(r[0]) - t0) % 40000 for r in gt_rows]
            offset_counts = Counter(offsets)
            modal_offset, modal_count = offset_counts.most_common(1)[0]
            num_distinct_offsets = len(offset_counts)
            modal_frac = modal_count / float(n_gt_boxes)

            abs_multiples = all(int(r[0]) % 40000 == 0 for r in gt_rows)
            t0_multiples = all(
                (int(r[0]) - t0) % 40000 == 0 for r in gt_rows
            )

            if t0_multiples:
                verdict = "ALIGNED_TO_T0"
            elif abs_multiples:
                verdict = "ALIGNED_TO_ABSOLUTE"
            else:
                verdict = f"MISALIGNED (offset={modal_offset})"

            duration_valid = all(int(r[1]) - int(r[0]) == 40000 for r in gt_rows)

    # Analysis 3: Window Occupancy & Class Balance
    total_spanned_windows = 0
    gt_windows_set = set(r[0] for r in gt_rows)
    pos_windows_count = len(gt_windows_set)
    pos_rate = 0.0
    max_boxes_in_window = 0
    box_count_per_window: Counter = Counter()

    if npy_path and npy_path.exists():
        events = load_events(npy_path)
        if len(events) > 0:
            t_start = int(events[0, 3])
            t_end = int(events[-1, 3])
            total_spanned_windows = math.ceil(
                (t_end - t_start + 1) / 40000.0
            )
            pos_rate = (
                pos_windows_count / float(total_spanned_windows)
                if total_spanned_windows > 0
                else 0.0
            )

    win_to_boxes: Dict[int, List[Tuple[int, int, int, int]]] = {}
    for r in gt_rows:
        win_start = r[0]
        box = (r[2], r[3], r[4], r[5])
        win_to_boxes.setdefault(win_start, []).append(box)

    if win_to_boxes:
        max_boxes_in_window = max(len(b_list) for b_list in win_to_boxes.values())
        for b_list in win_to_boxes.values():
            box_count_per_window[len(b_list)] += 1

    # Analysis 4: Event-Label Statistics
    l1_counts: List[int] = []
    l1_fracs: List[float] = []
    l1_inside_counts: List[int] = []
    l1_outside_counts: List[int] = []

    total_seq_events = 0
    total_seq_l1_events = 0
    seq_l1_frac = 0.0

    if npy_path and npy_path.exists():
        events = load_events(npy_path)
        total_seq_events = len(events)
        if total_seq_events > 0:
            total_seq_l1_events = int(np.sum(events[:, 4] == 1))
            seq_l1_frac = total_seq_l1_events / float(total_seq_events)

            # Process positive windows
            t_col = events[:, 3]
            for win_start, b_list in win_to_boxes.items():
                win_end = win_start + 40000
                idx_start = np.searchsorted(t_col, win_start)
                idx_end = np.searchsorted(t_col, win_end)
                win_events = events[idx_start:idx_end]

                if len(win_events) == 0:
                    continue

                l1_ev = win_events[win_events[:, 4] == 1]
                l1_cnt = len(l1_ev)
                l1_counts.append(l1_cnt)
                l1_fracs.append(l1_cnt / float(len(win_events)))

                if l1_cnt > 0:
                    inside_mask = np.zeros(l1_cnt, dtype=bool)
                    for cx, cy, bw, bh in b_list:
                        x1 = cx - (bw - 1) / 2.0
                        y1 = cy - (bh - 1) / 2.0
                        x2 = x1 + bw - 1
                        y2 = y1 + bh - 1
                        in_b = (
                            (l1_ev[:, 0] >= x1)
                            & (l1_ev[:, 0] <= x2)
                            & (l1_ev[:, 1] >= y1)
                            & (l1_ev[:, 1] <= y2)
                        )
                        inside_mask |= in_b

                    l1_inside_counts.append(int(np.sum(inside_mask)))
                    l1_outside_counts.append(int(np.sum(~inside_mask)))
                else:
                    l1_inside_counts.append(0)
                    l1_outside_counts.append(0)

    l1_stats = compute_stats(np.array(l1_counts, dtype=np.float64))
    mean_l1_frac = float(np.mean(l1_fracs)) if l1_fracs else 0.0
    mean_l1_inside = float(np.mean(l1_inside_counts)) if l1_inside_counts else 0.0
    mean_l1_outside = (
        float(np.mean(l1_outside_counts)) if l1_outside_counts else 0.0
    )

    # Analysis 5: Prediction Comparison
    has_pred = False
    n_pred_boxes = 0
    pred_w_stats = compute_stats(np.zeros(0))
    pred_h_stats = compute_stats(np.zeros(0))
    ratio_w = 0.0
    ratio_h = 0.0
    timestamp_match_frac = 0.0

    if pred_path and pred_path.exists():
        has_pred = True
        pred_rows = load_pred_rows(pred_path)
        n_pred_boxes = len(pred_rows)
        pred_widths = np.array([r[4] for r in pred_rows], dtype=np.float64)
        pred_heights = np.array([r[5] for r in pred_rows], dtype=np.float64)

        pred_w_stats = compute_stats(pred_widths)
        pred_h_stats = compute_stats(pred_heights)

        ratio_w = (
            pred_w_stats["mean"] / w_stats["mean"]
            if w_stats["mean"] > 0
            else 0.0
        )
        ratio_h = (
            pred_h_stats["mean"] / h_stats["mean"]
            if h_stats["mean"] > 0
            else 0.0
        )

        pred_win_starts = set(r[0] for r in pred_rows)
        matching_starts = pred_win_starts.intersection(gt_windows_set)
        timestamp_match_frac = (
            len(matching_starts) / float(len(pred_win_starts))
            if len(pred_win_starts) > 0
            else (1.0 if len(gt_windows_set) == 0 else 0.0)
        )

    return {
        "seq": seq_name,
        "split": split,
        "sensor": sensor_type,
        "resolution": f"{sensor_w}x{sensor_h}",
        "n_gt_boxes": n_gt_boxes,
        "gt_w_mean": w_stats["mean"],
        "gt_w_std": w_stats["std"],
        "gt_w_p50": w_stats["p50"],
        "gt_h_mean": h_stats["mean"],
        "gt_h_std": h_stats["std"],
        "gt_h_p50": h_stats["p50"],
        "mean_area_frac": mean_area_frac,
        "verdict": verdict,
        "modal_offset": modal_offset,
        "modal_frac": modal_frac,
        "distinct_offsets": num_distinct_offsets,
        "abs_multiples": abs_multiples,
        "duration_valid": duration_valid,
        "total_spanned_windows": total_spanned_windows,
        "pos_windows_count": pos_windows_count,
        "pos_rate": pos_rate,
        "max_boxes_in_window": max_boxes_in_window,
        "l1_mean_per_win": l1_stats["mean"],
        "l1_median_per_win": l1_stats["p50"],
        "mean_l1_frac_win": mean_l1_frac,
        "mean_l1_inside": mean_l1_inside,
        "mean_l1_outside": mean_l1_outside,
        "total_seq_l1": total_seq_l1_events,
        "seq_l1_frac": seq_l1_frac,
        "has_pred": has_pred,
        "n_pred_boxes": n_pred_boxes,
        "pred_w_mean": pred_w_stats["mean"],
        "pred_h_mean": pred_h_stats["mean"],
        "ratio_w": ratio_w,
        "ratio_h": ratio_h,
        "timestamp_match_frac": timestamp_match_frac,
    }


def print_summary_tables(
    results: List[Dict[str, Any]], pred_dir_given: bool
) -> None:
    """Print formatted tabulate section tables to stdout."""
    # Section 1: Box Size Distribution per Sensor
    print(
        "\n=========================================================================================================="
    )
    print("  SECTION 1: GT BOX SIZE DISTRIBUTION (AGGREGATED BY SENSOR)")
    print(
        "=========================================================================================================="
    )
    sensor_groups: Dict[str, List[Dict[str, Any]]] = {}
    for r in results:
        sensor_groups.setdefault(r["sensor"], []).append(r)

    sensor_rows = []
    for s_type, r_list in sensor_groups.items():
        total_gt = sum(r["n_gt_boxes"] for r in r_list)
        avg_w = np.mean([r["gt_w_mean"] for r in r_list]) if r_list else 0.0
        avg_h = np.mean([r["gt_h_mean"] for r in r_list]) if r_list else 0.0
        avg_area_frac = (
            np.mean([r["mean_area_frac"] for r in r_list]) if r_list else 0.0
        )
        sensor_rows.append(
            [
                s_type,
                len(r_list),
                total_gt,
                f"{avg_w:.2f}",
                f"{avg_h:.2f}",
                f"{avg_area_frac * 100:.4f}%",
            ]
        )

    print(
        tabulate(
            sensor_rows,
            headers=[
                "Sensor",
                "Sequences",
                "Total GT Boxes",
                "Mean GT W",
                "Mean GT H",
                "Mean Area %",
            ],
            tablefmt="grid",
        )
    )

    # Section 2: Window Grid Alignment
    print(
        "\n=========================================================================================================="
    )
    print("  SECTION 2: WINDOW GRID ALIGNMENT VERIFICATION")
    print(
        "=========================================================================================================="
    )
    align_rows = []
    for r in results:
        align_rows.append(
            [
                r["split"],
                r["seq"],
                r["verdict"],
                r["modal_offset"],
                f"{r['modal_frac'] * 100:.1f}%",
                r["distinct_offsets"],
                "YES" if r["duration_valid"] else "NO",
            ]
        )

    print(
        tabulate(
            align_rows,
            headers=[
                "Split",
                "Sequence",
                "Verdict",
                "Modal Offset (us)",
                "Modal Frac",
                "Distinct Offsets",
                "40ms Duration",
            ],
            tablefmt="grid",
        )
    )

    # Section 3: Occupancy & Class Balance
    print(
        "\n=========================================================================================================="
    )
    print("  SECTION 3: WINDOW OCCUPANCY AND POSITIVE RATE")
    print(
        "=========================================================================================================="
    )
    occ_rows = []
    for r in results:
        occ_rows.append(
            [
                r["seq"],
                r["total_spanned_windows"],
                r["pos_windows_count"],
                f"{r['pos_rate'] * 100:.2f}%",
                r["max_boxes_in_window"],
            ]
        )

    print(
        tabulate(
            occ_rows,
            headers=[
                "Sequence",
                "Spanned Windows",
                "Pos Windows",
                "Pos Rate",
                "Max Boxes/Win",
            ],
            tablefmt="grid",
        )
    )

    # Section 4: Event Label Signal Statistics
    print(
        "\n=========================================================================================================="
    )
    print("  SECTION 4: EVENT LABEL SIGNAL-TO-NOISE RATIO (RSO vs BACKGROUND)")
    print(
        "=========================================================================================================="
    )
    sig_rows = []
    for r in results:
        sig_rows.append(
            [
                r["seq"],
                f"{r['l1_mean_per_win']:.1f}",
                f"{r['mean_l1_frac_win'] * 100:.2f}%",
                f"{r['mean_l1_inside']:.1f}",
                f"{r['mean_l1_outside']:.1f}",
                f"{r['seq_l1_frac'] * 100:.3f}%",
            ]
        )

    print(
        tabulate(
            sig_rows,
            headers=[
                "Sequence",
                "Mean L1 / PosWin",
                "L1 Frac / PosWin",
                "Mean L1 Inside",
                "Mean L1 Outside",
                "Global L1 Frac",
            ],
            tablefmt="grid",
        )
    )

    # Section 5: Prediction Comparison (if pred_dir given)
    if pred_dir_given:
        print(
            "\n=========================================================================================================="
        )
        print("  SECTION 5: PREDICTION VS GROUND TRUTH COMPARISON")
        print(
            "=========================================================================================================="
        )
        pred_rows = []
        for r in results:
            if not r["has_pred"]:
                pred_rows.append([r["seq"], r["n_gt_boxes"], 0, "N/A", "N/A", "N/A", "N/A"])
                continue
            pred_rows.append(
                [
                    r["seq"],
                    r["n_gt_boxes"],
                    r["n_pred_boxes"],
                    f"{r['gt_w_mean']:.1f} x {r['gt_h_mean']:.1f}",
                    f"{r['pred_w_mean']:.1f} x {r['pred_h_mean']:.1f}",
                    f"W:{r['ratio_w']:.2f} | H:{r['ratio_h']:.2f}",
                    f"{r['timestamp_match_frac'] * 100:.1f}%",
                ]
            )

        print(
            tabulate(
                pred_rows,
                headers=[
                    "Sequence",
                    "GT Boxes",
                    "Pred Boxes",
                    "Mean GT (WxH)",
                    "Mean Pred (WxH)",
                    "Pred/GT Ratio",
                    "Win Match %",
                ],
                tablefmt="grid",
            )
        )


def print_diagnosis_block(
    results: List[Dict[str, Any]], pred_dir_given: bool
) -> None:
    """Print final summary DIAGNOSIS block flagging potential issues."""
    print(
        "\n=========================================================================================================="
    )
    print("  DIAGNOSIS SUMMARY")
    print(
        "=========================================================================================================="
    )

    misaligned = [
        r["seq"]
        for r in results
        if r["verdict"] not in ("ALIGNED_TO_T0", "ALIGNED_TO_ABSOLUTE")
    ]
    if misaligned:
        print(
            f" [WARNING] Misaligned window grids detected in {len(misaligned)} sequence(s): {', '.join(misaligned)}"
        )
    else:
        print(" [PASS] All sequence window grids align cleanly with event stream / absolute grid.")

    if pred_dir_given:
        zero_preds = [
            r["seq"] for r in results if r["has_pred"] and r["n_pred_boxes"] == 0
        ]
        if zero_preds:
            print(
                f" [WARNING] 0 predictions generated for {len(zero_preds)} sequence(s): {', '.join(zero_preds)}"
            )
        else:
            print(" [PASS] All evaluated sequences produced predictions.")

        bad_size_ratios = [
            r["seq"]
            for r in results
            if r["has_pred"]
            and r["n_pred_boxes"] > 0
            and (not (0.7 <= r["ratio_w"] <= 1.4) or not (0.7 <= r["ratio_h"] <= 1.4))
        ]
        if bad_size_ratios:
            print(
                f" [WARNING] Box size ratio (Pred/GT) outside [0.7, 1.4] range in {len(bad_size_ratios)} sequence(s): {', '.join(bad_size_ratios)}"
            )
        else:
            print(" [PASS] Bounding box scale ratios are well-calibrated within [0.7, 1.4].")

        bad_timestamp_match = [
            r["seq"]
            for r in results
            if r["has_pred"]
            and r["n_pred_boxes"] > 0
            and r["timestamp_match_frac"] < 0.9
        ]
        if bad_timestamp_match:
            print(
                f" [WARNING] Low window timestamp match fraction (< 90%) in {len(bad_timestamp_match)} sequence(s): {', '.join(bad_timestamp_match)}"
            )
        else:
            print(" [PASS] High window timestamp overlap (>= 90%) across predicted windows.")

    print(
        "==========================================================================================================\n"
    )


def save_csv_output(results: List[Dict[str, Any]], out_path: Path) -> None:
    """Save structured per-sequence results to CSV output file."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "seq",
        "split",
        "sensor",
        "resolution",
        "n_gt_boxes",
        "gt_w_mean",
        "gt_w_std",
        "gt_w_p50",
        "gt_h_mean",
        "gt_h_std",
        "gt_h_p50",
        "mean_area_frac",
        "verdict",
        "modal_offset",
        "modal_frac",
        "distinct_offsets",
        "duration_valid",
        "total_spanned_windows",
        "pos_windows_count",
        "pos_rate",
        "max_boxes_in_window",
        "l1_mean_per_win",
        "mean_l1_frac_win",
        "mean_l1_inside",
        "mean_l1_outside",
        "seq_l1_frac",
        "has_pred",
        "n_pred_boxes",
        "pred_w_mean",
        "pred_h_mean",
        "ratio_w",
        "ratio_h",
        "timestamp_match_frac",
    ]

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row_dict = {k: r.get(k, "") for k in fieldnames}
            writer.writerow(row_dict)

    print(f"[INFO] Detailed analysis exported to CSV: {out_path}")


def main() -> None:
    """CLI main entrypoint."""
    parser = argparse.ArgumentParser(
        description="OrbitSight Read-Only Ground-Truth Diagnostic Script"
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="../OrbitSight_Dataset",
        help="Path to dataset root directory",
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
        default="experiments/gt_analysis.csv",
        help="Path for output CSV report",
    )

    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir).resolve()
    print(f"Dataset directory: {dataset_dir}")

    if not dataset_dir.exists():
        print(f"Error: Dataset directory '{dataset_dir}' does not exist.", file=sys.stderr)
        sys.exit(1)

    gt_files = sorted(list(dataset_dir.rglob("*_bb_windows_40ms.txt")))
    if not gt_files:
        print(f"Error: No '*_bb_windows_40ms.txt' files found in {dataset_dir}", file=sys.stderr)
        sys.exit(1)

    pred_dir = Path(args.pred_dir) if args.pred_dir else None
    pred_dir_given = pred_dir is not None and pred_dir.exists()

    results: List[Dict[str, Any]] = []

    for gt_path in gt_files:
        fname = gt_path.name
        seq_name = fname.replace("_bb_windows_40ms.txt", "")

        split = "Testing" if "Testing" in str(gt_path) else "Training"

        npy_matches = list(gt_path.parent.glob(f"{seq_name}_labeled_events.npy"))
        if not npy_matches:
            npy_matches = list(dataset_dir.rglob(f"{seq_name}_labeled_events.npy"))

        npy_path = npy_matches[0] if npy_matches else None

        pred_path = None
        if pred_dir_given and pred_dir:
            p_alt1 = pred_dir / f"{seq_name}_pred.txt"
            p_alt2 = pred_dir / f"{seq_name}_bb_windows_40ms.txt"
            if p_alt1.exists():
                pred_path = p_alt1
            elif p_alt2.exists():
                pred_path = p_alt2

        res = analyze_sequence(seq_name, split, gt_path, npy_path, pred_path)
        results.append(res)

    print_summary_tables(results, pred_dir_given)
    print_diagnosis_block(results, pred_dir_given)

    out_path = Path(args.out)
    save_csv_output(results, out_path)


if __name__ == "__main__":
    main()
