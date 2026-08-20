"""Hyper-parameter grid search harness with memory-efficient component caching."""

import argparse
from copy import deepcopy
import csv
import itertools
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
    sequence_name_from_npy,
)
from src.metrics import evaluate_sequence


def _parse_yaml_scalar(val_str: str) -> Any:
    """Helper to parse scalar strings into int, float, or bool values."""
    val_clean = val_str.strip("'\"")
    if not val_clean:
        return ""
    try:
        if "." in val_clean or "e" in val_clean.lower():
            return float(val_clean)
        return int(val_clean)
    except ValueError:
        if val_clean.lower() in ("true", "yes"):
            return True
        if val_clean.lower() in ("false", "no"):
            return False
        return val_clean


def _parse_yaml_list_or_scalar(val_str: str) -> Any:
    """Parse inline YAML list [a, b, c] or scalar value."""
    val_clean = val_str.strip()
    if val_clean.startswith("[") and val_clean.endswith("]"):
        items = val_clean[1:-1].split(",")
        return [_parse_yaml_scalar(item.strip()) for item in items if item.strip()]
    return _parse_yaml_scalar(val_clean)


def load_grid_config(grid_path: Path) -> Dict[str, Any]:
    """Load parameter grid configuration file with zero-dependency fallback parser."""
    if not grid_path.exists():
        print(f"Error: Grid file '{grid_path}' does not exist.", file=sys.stderr)
        sys.exit(1)

    try:
        import yaml

        with open(grid_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except ImportError:
        cfg: Dict[str, Any] = {}
        current_section: str = ""

        with open(grid_path, "r", encoding="utf-8") as f:
            for line in f:
                raw = line.split("#")[0].rstrip("\r\n")
                if not raw.strip():
                    continue

                indent = len(raw) - len(raw.lstrip(" "))
                stripped = raw.strip()

                if ":" not in stripped:
                    continue

                parts = stripped.split(":", 1)
                key = parts[0].strip()
                val_str = parts[1].strip()

                if indent == 0:
                    if not val_str:
                        current_section = key
                        cfg[key] = {}
                    else:
                        current_section = ""
                        cfg[key] = _parse_yaml_list_or_scalar(val_str)
                elif indent > 0 and current_section:
                    if not isinstance(cfg.get(current_section), dict):
                        cfg[current_section] = {}
                    cfg[current_section][key] = _parse_yaml_list_or_scalar(val_str)

        return cfg


def extract_raw_components_fast(
    count_img: np.ndarray,
    thresh: float,
    open_k: int,
    dilate_k: int,
    width: int,
    height: int,
) -> List[Dict[str, float]]:
    """Extract raw connected component properties given pre-calculated threshold."""
    binary = (count_img >= thresh).astype(np.uint8)
    if cv2.countNonZero(binary) < 4:
        return []

    if open_k > 1:
        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (open_k, open_k))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open)
        if cv2.countNonZero(binary) < 4:
            return []

    if dilate_k > 1:
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (dilate_k, dilate_k))
        binary = cv2.dilate(binary, kernel_dilate)
        if cv2.countNonZero(binary) < 4:
            return []

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )

    max_area = 0.02 * width * height
    comps: List[Dict[str, float]] = []

    for label_idx in range(1, num_labels):
        comp_area = float(stats[label_idx, cv2.CC_STAT_AREA])
        if comp_area > max_area:
            continue

        x_box = int(stats[label_idx, cv2.CC_STAT_LEFT])
        y_box = int(stats[label_idx, cv2.CC_STAT_TOP])
        w_box = int(stats[label_idx, cv2.CC_STAT_WIDTH])
        h_box = int(stats[label_idx, cv2.CC_STAT_HEIGHT])

        sub_img = count_img[y_box : y_box + h_box, x_box : x_box + w_box]
        sub_labels = labels[y_box : y_box + h_box, x_box : x_box + w_box]
        sub_mask = sub_labels == label_idx

        comp_events = float(sub_img[sub_mask].sum())

        center_comp_x = float(centroids[label_idx, 0])
        center_comp_y = float(centroids[label_idx, 1])

        if comp_events > 0:
            ys, xs = np.where(sub_mask)
            weights = sub_img[sub_mask]
            center_w_x = float(x_box + (xs * weights).sum() / comp_events)
            center_w_y = float(y_box + (ys * weights).sum() / comp_events)
        else:
            center_w_x = center_comp_x
            center_w_y = center_comp_y

        comps.append(
            {
                "x_box": float(x_box),
                "y_box": float(y_box),
                "w_box": float(w_box),
                "h_box": float(h_box),
                "area": comp_area,
                "events": comp_events,
                "cx_comp": center_comp_x,
                "cy_comp": center_comp_y,
                "cx_weighted": center_w_x,
                "cy_weighted": center_w_y,
            }
        )

    return comps


def run_sequence_sweep_cached(
    npy_path: Path,
    gt_rows: List[Tuple[int, int, int, int, int, int]],
    raw_grid: Dict[str, Any],
    sensor_name: str,
    max_windows: float = float("inf"),
) -> List[Dict[str, Any]]:
    """Memory-efficient sequence processing with component caching."""
    seq_name = sequence_name_from_npy(npy_path)
    events = load_events(npy_path)
    width, height = infer_resolution(seq_name, events[:, 0], events[:, 1])

    percentile_list = raw_grid.get("percentile", [97.5])
    open_k_list = raw_grid.get("open_kernel", [2])
    dilate_k_list = raw_grid.get("dilate_kernel", [3])
    thresh_combos = list(
        itertools.product(percentile_list, open_k_list, dilate_k_list)
    )

    comp_cache: Dict[Tuple[float, int, int], List[Tuple[int, int, List[Dict[str, float]]]]] = {
        tc: [] for tc in thresh_combos
    }

    win_count = 0
    for w_start, w_end, w_events in iter_windows(events, window_us=WINDOW_US):
        count_img, _, _ = event_image(w_events, width, height)
        nonzero_vals = count_img[count_img > 0]
        num_nonzero = len(nonzero_vals)

        # Pre-compute thresholds once per percentile for this window
        thresh_map: Dict[float, float] = {}
        if num_nonzero >= 4:
            for perc in percentile_list:
                actual_perc = min(99.0, perc + (num_nonzero - 1000) / 500.0) if num_nonzero > 1000 else perc
                thresh_map[perc] = max(1.0, float(np.percentile(nonzero_vals, actual_perc)))

        for perc, open_k, dilate_k in thresh_combos:
            if perc not in thresh_map:
                c_list = []
            else:
                c_list = extract_raw_components_fast(
                    count_img, thresh_map[perc], open_k, dilate_k, width, height
                )
            comp_cache[(perc, open_k, dilate_k)].append((w_start, w_end, c_list))

        del count_img
        win_count += 1
        if win_count >= max_windows:
            break

    num_windows = len(comp_cache[thresh_combos[0]])
    diagonal = math.hypot(width, height)

    min_events_list = raw_grid.get("min_events_in_box", [6])
    min_hits_list = raw_grid.get("min_hits", [2])
    box_mode_list = raw_grid.get("box_mode", ["scale", "fixed"])
    centroid_mode_list = raw_grid.get("centroid_mode", ["component", "weighted"])

    sensor_block = raw_grid.get(sensor_name, {})
    box_scale_list = sensor_block.get("box_scale", [2.0])
    box_pad_list = sensor_block.get("box_pad", [4.0])
    box_w_list = sensor_block.get("box_w", [14])
    box_h_list = sensor_block.get("box_h", [14])

    results: List[Dict[str, Any]] = []

    for (perc, open_k, dilate_k) in thresh_combos:
        window_cached = comp_cache[(perc, open_k, dilate_k)]

        for min_evt, min_hits, c_mode, b_mode in itertools.product(
            min_events_list, min_hits_list, centroid_mode_list, box_mode_list
        ):
            if b_mode == "scale":
                geom_tuples = [
                    (s, p, box_w_list[0], box_h_list[0])
                    for s, p in itertools.product(box_scale_list, box_pad_list)
                ]
            else:
                geom_tuples = [
                    (box_scale_list[0], box_pad_list[0], bw, bh)
                    for bw, bh in itertools.product(box_w_list, box_h_list)
                ]

            for b_scale, b_pad, b_w, b_h in geom_tuples:
                start_time = time.perf_counter()

                combo = {
                    "percentile": perc,
                    "min_events_in_box": min_evt,
                    "open_kernel": open_k,
                    "dilate_kernel": dilate_k,
                    "min_hits": min_hits,
                    "box_mode": b_mode,
                    "centroid_mode": c_mode,
                    "box_scale": b_scale,
                    "box_pad": b_pad,
                    "box_w": b_w,
                    "box_h": b_h,
                }

                window_boxes: List[Tuple[int, int, List[Dict[str, float]]]] = []
                for w_idx in range(num_windows):
                    w_start, w_end, raw_comps = window_cached[w_idx]

                    boxes: List[Dict[str, float]] = []
                    for comp in raw_comps:
                        if comp["events"] < min_evt:
                            continue

                        cx = (
                            comp["cx_weighted"]
                            if c_mode == "weighted"
                            else comp["cx_comp"]
                        )
                        cy = (
                            comp["cy_weighted"]
                            if c_mode == "weighted"
                            else comp["cy_comp"]
                        )

                        if b_mode == "fixed":
                            nw, nh = float(b_w), float(b_h)
                        else:
                            nw = comp["w_box"] * b_scale + 2.0 * b_pad
                            nh = comp["h_box"] * b_scale + 2.0 * b_pad

                        x1 = max(0.0, min(float(width), cx - nw / 2.0))
                        y1 = max(0.0, min(float(height), cy - nh / 2.0))
                        x2 = max(0.0, min(float(width), cx + nw / 2.0))
                        y2 = max(0.0, min(float(height), cy + nh / 2.0))

                        cw = max(3.0, x2 - x1)
                        ch = max(3.0, y2 - y1)
                        ccx = (x1 + x2) / 2.0
                        ccy = (y1 + y2) / 2.0

                        area_calc = cw * ch
                        density = comp["events"] / area_calc if area_calc > 0 else 0.0

                        boxes.append(
                            {
                                "center_x": ccx,
                                "center_y": ccy,
                                "width": cw,
                                "height": ch,
                                "events": comp["events"],
                                "density": density,
                            }
                        )

                    window_boxes.append((w_start, w_end, boxes))

                max_dist = 0.05 * diagonal
                pred_rows: List[Tuple[int, int, int, int, int, int, float]] = []

                for w_idx in range(num_windows):
                    w_start, w_end, boxes = window_boxes[w_idx]
                    if not boxes:
                        continue

                    prev_b = window_boxes[w_idx - 1][2] if w_idx > 0 else []
                    next_b = (
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
                            for p in prev_b
                        ):
                            hits += 1
                        if any(
                            math.hypot(
                                box["center_x"] - n["center_x"],
                                box["center_y"] - n["center_y"],
                            )
                            <= max_dist
                            for n in next_b
                        ):
                            hits += 1

                        if min_hits >= 2 and hits < min_hits:
                            continue

                        rcx = int(round(box["center_x"]))
                        rcy = int(round(box["center_y"]))
                        rbw = int(round(box["width"]))
                        rbh = int(round(box["height"]))
                        conf = min(1.0, max(0.01, box["density"] * (hits / 3.0)))

                        pred_rows.append((w_start, w_end, rcx, rcy, rbw, rbh, conf))

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
        "--max-windows",
        type=int,
        default=None,
        help="Limit number of windows per sequence for fast sweep test",
    )

    args = parser.parse_args()

    grid_path = Path(args.grid)
    raw_grid = load_grid_config(grid_path)
    dataset_dir = Path(args.dataset_dir).resolve()

    sensors_to_sweep = (
        [args.sensor] if args.sensor else ["DAVIS", "DVX", "EVK4"]
    )
    sensor_best_configs: Dict[str, Dict[str, Any]] = {}
    max_w = float("inf") if not args.max_windows else float(args.max_windows)

    for sensor in sensors_to_sweep:
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
            print(
                f"[INFO] No matching sequences found for sensor '{sensor}'.",
                flush=True,
            )
            continue

        print(
            f"\n[INFO] Sensor '{sensor}': Sweeping {len(filtered_files)} sequence(s)...",
            flush=True,
        )

        combo_metrics: Dict[int, Dict[str, Any]] = {}

        for seq_idx, gt_file in enumerate(filtered_files, start=1):
            seq_name = gt_file.name.replace("_bb_windows_40ms.txt", "")
            npy_matches = list(gt_file.parent.glob(f"{seq_name}_labeled_events.npy"))
            if not npy_matches:
                npy_matches = list(
                    dataset_dir.rglob(f"{seq_name}_labeled_events.npy")
                )
            if not npy_matches:
                continue

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

            print(
                f"  -> [{seq_idx}/{len(filtered_files)}] Caching & sweeping '{seq_name}'...",
                flush=True,
            )
            seq_sweep = run_sequence_sweep_cached(
                npy_matches[0], gt_rows, raw_grid, sensor, max_windows=max_w
            )

            for combo_idx, res in enumerate(seq_sweep):
                if combo_idx not in combo_metrics:
                    combo_metrics[combo_idx] = {
                        "combo": res["combo"],
                        "all_ap": [],
                        "tp": 0,
                        "fp": 0,
                        "fn": 0,
                        "ms_list": [],
                    }
                cm = combo_metrics[combo_idx]
                cm["tp"] += res["eval"]["tp"]
                cm["fp"] += res["eval"]["fp"]
                cm["fn"] += res["eval"]["fn"]
                if not np.isnan(res["eval"]["ap"]):
                    cm["all_ap"].append(res["eval"]["ap"])
                cm["ms_list"].append(res["ms_per_window"])

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
        if sweep_summary:
            sensor_best_configs[sensor] = sweep_summary[0]

            top_n = min(args.top, len(sweep_summary))
            print(f"\nTop {top_n} Combinations for Sensor '{sensor}':", flush=True)
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
                ),
                flush=True,
            )

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

    print(f"\n[INFO] Sweep results exported to CSV: {out_path}", flush=True)

    print("\n==================================================", flush=True)
    print("  RECOMMENDATION BLOCK (PER-SENSOR BEST CONFIG)", flush=True)
    print("==================================================", flush=True)
    for s_name, s_best in sensor_best_configs.items():
        cb = s_best["combo"]
        print(f"\n# --- {s_name} Optimal Parameters ---", flush=True)
        print(
            f"# Performance: mAP={s_best['mAP']:.4f}, Prec={s_best['precision']:.4f}, Rec={s_best['recall']:.4f}, F1={s_best['f1']:.4f}",
            flush=True,
        )
        print(f"{s_name}:", flush=True)
        print(f"  percentile: {cb['percentile']}", flush=True)
        print(f"  min_events_in_box: {cb['min_events_in_box']}", flush=True)
        print(f"  open_kernel: {cb['open_kernel']}", flush=True)
        print(f"  dilate_kernel: {cb['dilate_kernel']}", flush=True)
        print(f"  min_hits: {cb['min_hits']}", flush=True)
        print(f"  box_mode: '{cb['box_mode']}'", flush=True)
        print(f"  centroid_mode: '{cb['centroid_mode']}'", flush=True)
        if cb["box_mode"] == "fixed":
            print(f"  box_w: {cb['box_w']}", flush=True)
            print(f"  box_h: {cb['box_h']}", flush=True)
        else:
            print(f"  box_scale: {cb['box_scale']}", flush=True)
            print(f"  box_pad: {cb['box_pad']}", flush=True)

    print("\n==================================================\n", flush=True)


if __name__ == "__main__":
    main()
