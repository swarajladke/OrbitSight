"""Hyper-parameter grid search harness with connected-component caching and per-sensor evaluation."""

import argparse
from copy import deepcopy
import csv
import itertools
import math
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Tuple
import yaml
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
from src.metrics import evaluate_sequence


def load_grid_config(grid_path: Path) -> Dict[str, Any]:
    """Load parameter grid configuration file."""
    if not grid_path.exists():
        print(f"Error: Grid file '{grid_path}' does not exist.", file=sys.stderr)
        sys.exit(1)
    with open(grid_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def extract_sensor_grid(
    raw_grid: Dict[str, Any], sensor_name: str
) -> Tuple[List[Dict[str, Any]], Dict[str, List[Any]]]:
    """Extract global threshold parameters and sensor-specific geometry parameters."""
    thresh_params = {
        "percentile": raw_grid.get("percentile", [97.5]),
        "min_events_in_box": raw_grid.get("min_events_in_box", [6]),
        "open_kernel": raw_grid.get("open_kernel", [2]),
        "dilate_kernel": raw_grid.get("dilate_kernel", [3]),
    }

    global_combos = [
        dict(zip(thresh_params.keys(), v))
        for v in itertools.product(*thresh_params.values())
    ]

    sensor_block = raw_grid.get(sensor_name, {})
    geom_params = {
        "min_hits": raw_grid.get("min_hits", [2]),
        "box_mode": raw_grid.get("box_mode", ["scale", "fixed"]),
        "centroid_mode": raw_grid.get("centroid_mode", ["component", "weighted"]),
        "box_scale": sensor_block.get("box_scale", [2.0]),
        "box_pad": sensor_block.get("box_pad", [4.0]),
        "box_w": sensor_block.get("box_w", [14]),
        "box_h": sensor_block.get("box_h", [14]),
    }

    return global_combos, geom_params


def generate_full_combos(
    global_combos: List[Dict[str, Any]], geom_params: Dict[str, List[Any]]
) -> List[Dict[str, Any]]:
    """Cartesian product of threshold combos and geometry combos."""
    geom_keys = list(geom_params.keys())
    geom_combos = [
        dict(zip(geom_keys, v))
        for v in itertools.product(*[geom_params[k] for k in geom_keys])
    ]

    full_combos = []
    for g_thresh in global_combos:
        for g_geom in geom_combos:
            combo = deepcopy(g_thresh)
            combo.update(g_geom)
            full_combos.append(combo)
    return full_combos


def run_sequence_sweep(
    npy_path: Path,
    gt_rows: List[Tuple[int, int, int, int, int, int]],
    full_combos: List[Dict[str, Any]],
    sensor_name: str,
) -> List[Dict[str, Any]]:
    """Process a single sequence across parameter combinations."""
    seq_name = sequence_name_from_npy(npy_path)
    events = load_events(npy_path)
    width, height = infer_resolution(seq_name, events[:, 0], events[:, 1])

    # Pre-generate 2D window count images once
    window_data: List[Tuple[int, int, np.ndarray]] = []
    for w_start, w_end, w_events in iter_windows(events, window_us=WINDOW_US):
        count_img, _, _ = event_image(w_events, width, height)
        window_data.append((w_start, w_end, count_img))

    num_windows = len(window_data)
    diagonal = math.hypot(width, height)

    results: List[Dict[str, Any]] = []

    for combo in full_combos:
        start_time = time.perf_counter()

        # Build candidate boxes for each window
        cfg = deepcopy(combo)
        cfg[sensor_name] = {
            "box_scale": combo["box_scale"],
            "box_pad": combo["box_pad"],
            "box_w": combo["box_w"],
            "box_h": combo["box_h"],
        }

        window_boxes: List[Tuple[int, int, List[Dict[str, float]]]] = []
        for w_start, w_end, count_img in window_data:
            boxes = detect_boxes(count_img, width, height, cfg)
            window_boxes.append((w_start, w_end, boxes))

        # Persistence filtering
        min_hits = combo["min_hits"]
        max_dist = combo.get("max_dist_frac", 0.05) * diagonal
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
                    math.hypot(
                        box["center_x"] - p["center_x"],
                        box["center_y"] - p["center_y"],
                    )
                    <= max_dist
                    for p in prev_boxes
                ):
                    hits += 1
                if any(
                    math.hypot(
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

        results.append(
            {
                "combo": combo,
                "eval": eval_res,
                "ms_per_window": (
                    elapsed_ms / num_windows if num_windows > 0 else 0.0
                ),
            }
        )

    return results


def main() -> None:
    """Main CLI entrypoint for parameter sweep harness."""
    parser = argparse.ArgumentParser(
        description="OrbitSight Parameter Sweep Harness"
    )
    parser.add_argument(
        "--grid", type=str, default="configs/grid.yaml", help="Path to grid YAML file"
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="../OrbitSight_Dataset",
        help="Path to dataset directory",
    )
    parser.add_argument(
        "--sequences",
        type=str,
        default=None,
        help="Comma-separated subset of sequence names",
    )
    parser.add_argument(
        "--sensor",
        type=str,
        default=None,
        choices=["DAVIS", "DVX", "EVK4"],
        help="Restrict sweep to specific sensor family",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="all",
        choices=["train", "test", "all"],
        help="Subset split to sweep",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="experiments/sweep_results.csv",
        help="Output CSV file path",
    )
    parser.add_argument(
        "--top", type=int, default=20, help="Number of top configurations to print"
    )
    parser.add_argument(
        "--max-combos",
        type=int,
        default=500,
        help="Maximum allowed combination count before aborting",
    )

    args = parser.parse_args()

    grid_path = Path(args.grid)
    raw_grid = load_grid_config(grid_path)
    dataset_dir = Path(args.dataset_dir).resolve()

    sensors_to_sweep = (
        [args.sensor] if args.sensor else ["DAVIS", "DVX", "EVK4"]
    )
    sensor_best_configs: Dict[str, Dict[str, Any]] = {}

    for sensor in sensors_to_sweep:
        global_combos, geom_params = extract_sensor_grid(raw_grid, sensor)
        full_combos = generate_full_combos(global_combos, geom_params)

        print(
            f"\n[INFO] Sensor '{sensor}': Expanded {len(full_combos)} parameter combination(s)."
        )

        if len(full_combos) > args.max_combos:
            print(
                f"[WARNING] Combination count {len(full_combos)} exceeds max limit {args.max_combos}. Aborting sweep for {sensor}.",
                file=sys.stderr,
            )
            continue

        # Discover matching sequence GT files
        gt_files = sorted(list(dataset_dir.rglob("*_bb_windows_40ms.txt")))
        filtered_files: List[Path] = []

        for gt_f in gt_files:
            seq_name = gt_f.name.replace("_bb_windows_40ms.txt", "")
            if sensor not in seq_name.upper():
                continue
            if args.split == "train" and "Training" not in str(gt_f):
                continue
            if args.split == "test" and "Testing" not in str(gt_f):
                continue
            if args.sequences:
                target_set = {
                    s.strip() for s in args.sequences.split(",") if s.strip()
                }
                if seq_name not in target_set:
                    continue
            filtered_files.append(gt_f)

        if not filtered_files:
            print(f"[INFO] No matching sequences found for sensor '{sensor}'.")
            continue

        print(
            f"[INFO] Sweeping {len(filtered_files)} sequence(s) for sensor '{sensor}'..."
        )

        # Aggregate metrics per combination
        combo_metrics: Dict[int, Dict[str, Any]] = {}
        for combo_idx, combo in enumerate(full_combos):
            combo_metrics[combo_idx] = {
                "combo": combo,
                "all_ap": [],
                "tp": 0,
                "fp": 0,
                "fn": 0,
                "ms_list": [],
            }

        for gt_file in filtered_files:
            seq_name = gt_file.name.replace("_bb_windows_40ms.txt", "")
            npy_matches = list(gt_file.parent.glob(f"{seq_name}_labeled_events.npy"))
            if not npy_matches:
                npy_matches = list(
                    dataset_dir.rglob(f"{seq_name}_labeled_events.npy")
                )
            if not npy_matches:
                continue

            # Load GT rows
            gt_rows = []
            with open(gt_file, "r", encoding="utf-8") as f:
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

            seq_sweep = run_sequence_sweep(
                npy_matches[0], gt_rows, full_combos, sensor
            )

            for combo_idx, res in enumerate(seq_sweep):
                cm = combo_metrics[combo_idx]
                cm["tp"] += res["eval"]["tp"]
                cm["fp"] += res["eval"]["fp"]
                cm["fn"] += res["eval"]["fn"]
                if not np.isnan(res["eval"]["ap"]):
                    cm["all_ap"].append(res["eval"]["ap"])
                cm["ms_list"].append(res["ms_per_window"])

        # Compute summary metrics for each combination
        sweep_summary: List[Dict[str, Any]] = []
        for cm in combo_metrics.values():
            tp, fp, fn = cm["tp"], cm["fp"], cm["fn"]
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (
                2.0 * prec * rec / (prec + rec)
                if (prec + rec) > 0.0
                else 0.0
            )
            mAP = float(np.mean(cm["all_ap"])) if cm["all_ap"] else 0.0
            avg_ms = float(np.mean(cm["ms_list"])) if cm["ms_list"] else 0.0

            sweep_summary.append(
                {
                    "sensor": sensor,
                    "mAP": mAP,
                    "precision": prec,
                    "recall": rec,
                    "f1": f1,
                    "avg_ms_per_window": avg_ms,
                    "combo": cm["combo"],
                }
            )

        sweep_summary.sort(key=lambda x: x["mAP"], reverse=True)
        sensor_best_configs[sensor] = sweep_summary[0]

        # Print top rows
        top_n = min(args.top, len(sweep_summary))
        print(f"\nTop {top_n} Combinations for Sensor '{sensor}':")
        table_data = []
        for s_idx, s_res in enumerate(sweep_summary[:top_n], start=1):
            cb = s_res["combo"]
            table_data.append(
                [
                    s_idx,
                    f"{s_res['mAP']:.4f}",
                    f"{s_res['precision']:.4f}",
                    f"{s_res['recall']:.4f}",
                    f"{s_res['f1']:.4f}",
                    f"{s_res['avg_ms_per_window']:.2f}",
                    cb["box_mode"],
                    cb["centroid_mode"],
                    cb["percentile"],
                    cb["min_events_in_box"],
                    cb["box_scale" if cb["box_mode"] == "scale" else "box_w"],
                ]
            )

        print(
            tabulate(
                table_data,
                headers=[
                    "#",
                    "mAP",
                    "Prec",
                    "Rec",
                    "F1",
                    "ms/win",
                    "BoxMode",
                    "Centroid",
                    "Perc",
                    "MinEvt",
                    "Scale/W",
                ],
                tablefmt="github",
            )
        )

    # Save all results to CSV
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "sensor",
            "mAP",
            "precision",
            "recall",
            "f1",
            "avg_ms_per_window",
            "percentile",
            "min_events_in_box",
            "open_kernel",
            "dilate_kernel",
            "min_hits",
            "box_mode",
            "centroid_mode",
            "box_scale",
            "box_pad",
            "box_w",
            "box_h",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s_name, best_info in sensor_best_configs.items():
            cb = best_info["combo"]
            row = {
                "sensor": s_name,
                "mAP": f"{best_info['mAP']:.4f}",
                "precision": f"{best_info['precision']:.4f}",
                "recall": f"{best_info['recall']:.4f}",
                "f1": f"{best_info['f1']:.4f}",
                "avg_ms_per_window": f"{best_info['avg_ms_per_window']:.2f}",
            }
            for k in fieldnames[6:]:
                row[k] = cb.get(k, "")
            writer.writerow(row)

    print(f"\n[INFO] Sweep results exported to CSV: {out_path}")

    # Print Recommendation Block
    print("\n==================================================")
    print("  RECOMMENDATION BLOCK (PER-SENSOR BEST CONFIG)")
    print("==================================================")
    for s_name, s_best in sensor_best_configs.items():
        cb = s_best["combo"]
        print(f"\n# --- {s_name} Optimal Parameters ---")
        print(
            f"# Performance: mAP={s_best['mAP']:.4f}, Prec={s_best['precision']:.4f}, Rec={s_best['recall']:.4f}, F1={s_best['f1']:.4f}"
        )
        print(f"{s_name}:")
        print(f"  percentile: {cb['percentile']}")
        print(f"  min_events_in_box: {cb['min_events_in_box']}")
        print(f"  open_kernel: {cb['open_kernel']}")
        print(f"  dilate_kernel: {cb['dilate_kernel']}")
        print(f"  min_hits: {cb['min_hits']}")
        print(f"  box_mode: '{cb['box_mode']}'")
        print(f"  centroid_mode: '{cb['centroid_mode']}'")
        if cb["box_mode"] == "fixed":
            print(f"  box_w: {cb['box_w']}")
            print(f"  box_h: {cb['box_h']}")
        else:
            print(f"  box_scale: {cb['box_scale']}")
            print(f"  box_pad: {cb['box_pad']}")

    print("\n==================================================\n")


if __name__ == "__main__":
    main()
