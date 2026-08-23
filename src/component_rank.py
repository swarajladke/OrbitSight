"""Component Rank Profiling Tool for OrbitSight Connected-Component Extraction.

Measures the rank of ground-truth matched bounding boxes among connected components
sorted by event count descending across all GT-occupied windows.
"""

import argparse
import csv
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.common import WINDOW_US, infer_resolution, iter_windows, load_events, sequence_name_from_npy
from src.detector import int_percentile
from src.infer import load_config
from src.metrics import iou
from src.static_map import build_continuous_static_map


def profile_sequence_component_ranks(
    npy_file: Path,
    gt_file: Path,
    cfg: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Profile component rank of ground truth matches for a single sequence."""
    seq_name = sequence_name_from_npy(npy_file)
    events = load_events(npy_file)
    width, height = infer_resolution(seq_name, events[:, 0], events[:, 1])

    # Load GT rows indexed by window_start_timestamp_us
    gt_by_ts: Dict[int, List[Tuple[int, int, int, int]]] = {}
    with open(gt_file, "r", encoding="utf-8") as f:
        rdr = csv.DictReader(f, delimiter="\t")
        for r in rdr:
            ws = int(r["window_start_timestamp_us"])
            cx, cy = int(r["center_x"]), int(r["center_y"])
            bw, bh = int(r["width"]), int(r["height"])
            if ws not in gt_by_ts:
                gt_by_ts[ws] = []
            gt_by_ts[ws].append((cx, cy, bw, bh))

    if not gt_by_ts:
        return []

    # Sensor-specific parameters
    sensor_name = "DAVIS"
    if "EVK4" in seq_name.upper():
        sensor_name = "EVK4"
    elif "DVX" in seq_name.upper():
        sensor_name = "DVX"

    eff = {**cfg, **cfg.get(sensor_name, {})}
    percentile = float(eff.get("percentile", 97.5))
    min_events = float(eff.get("min_events_in_box", 6))
    open_k = int(eff.get("open_kernel", 2))
    dilate_k = int(eff.get("dilate_kernel", 3))
    max_area_frac = float(eff.get("max_area_frac", 0.02))
    max_area_pixels = max_area_frac * (width * height)
    static_thresh = float(eff.get("static_thresh", 0.5))
    box_mode = str(eff.get("box_mode", "fixed")).lower()
    box_w_cfg = float(eff.get("box_w", 18))
    box_h_cfg = float(eff.get("box_h", 18))
    extent_scale = float(eff.get("extent_scale", 1.0))
    extent_pad = float(eff.get("extent_pad", 2.0))

    # Continuous static mask
    static_map = build_continuous_static_map(events, width, height, window_us=WINDOW_US)
    static_mask = static_map >= static_thresh

    results: List[Dict[str, Any]] = []

    for w_events in iter_windows(events, window_us=WINDOW_US):
        if len(w_events) == 0:
            continue
        ws = int(w_events[0, 3])
        if ws not in gt_by_ts:
            continue

        xs = w_events[:, 0].astype(np.int32)
        ys = w_events[:, 1].astype(np.int32)
        valid = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
        xs, ys = xs[valid], ys[valid]

        count_img = np.zeros((height, width), dtype=np.int32)
        np.add.at(count_img, (ys, xs), 1)
        count_img[static_mask] = 0

        nonzero = count_img[count_img > 0]
        if len(nonzero) < 4:
            continue

        p_val = float(int_percentile(nonzero, percentile))
        binary = (count_img >= p_val).astype(np.uint8)

        if open_k > 0:
            k_open = cv2.getStructuringElement(cv2.MORPH_RECT, (open_k, open_k))
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k_open)

        if dilate_k > 0:
            k_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (dilate_k, dilate_k))
            binary = cv2.dilate(binary, k_dilate)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )
        if num_labels <= 1:
            continue

        label_event_sums = np.bincount(labels.ravel(), weights=count_img.ravel(), minlength=num_labels)

        # Collect components
        candidate_components: List[Tuple[float, float, float, float, float]] = []
        for label_idx in range(1, num_labels):
            comp_area = float(stats[label_idx, cv2.CC_STAT_AREA])
            x_box = int(stats[label_idx, cv2.CC_STAT_LEFT])
            y_box = int(stats[label_idx, cv2.CC_STAT_TOP])
            w_box = int(stats[label_idx, cv2.CC_STAT_WIDTH])
            h_box = int(stats[label_idx, cv2.CC_STAT_HEIGHT])

            if comp_area > max_area_pixels:
                continue

            comp_events = float(label_event_sums[label_idx])
            if comp_events >= min_events:
                if box_mode == "extent":
                    bw = max(4.0, float(w_box) * extent_scale + extent_pad)
                    bh = max(4.0, float(h_box) * extent_scale + extent_pad)
                else:
                    bw, bh = box_w_cfg, box_h_cfg
                cx = float(x_box) + float(w_box) / 2.0
                cy = float(y_box) + float(h_box) / 2.0
                candidate_components.append((cx, cy, bw, bh, comp_events))

        total_comps = len(candidate_components)
        if total_comps == 0:
            for _ in gt_by_ts[ws]:
                results.append({
                    "sequence": seq_name,
                    "sensor": sensor_name,
                    "matched": False,
                    "rank": -1,
                    "total_components": 0,
                })
            continue

        # Sort by event count descending
        candidate_components.sort(key=lambda c: c[4], reverse=True)

        for g_cx, g_cy, g_bw, g_bh in gt_by_ts[ws]:
            best_iou = 0.0
            best_rank = -1

            for rank_idx, (c_cx, c_cy, c_bw, c_bh, _) in enumerate(candidate_components, start=1):
                cur_iou = iou((c_cx, c_cy, c_bw, c_bh), (float(g_cx), float(g_cy), float(g_bw), float(g_bh)))
                if cur_iou > best_iou:
                    best_iou = cur_iou
                    if cur_iou >= 0.5:
                        best_rank = rank_idx
                        break

            results.append({
                "sequence": seq_name,
                "sensor": sensor_name,
                "matched": (best_iou >= 0.5),
                "rank": best_rank,
                "total_components": total_comps,
            })

    return results


def summarize_ranks(records: List[Dict[str, Any]], label: str = "ALL") -> None:
    """Print summary statistics table for a set of rank records."""
    if not records:
        print(f"{label:12s} | No records.")
        return

    n_windows = len(records)
    matched = [r for r in records if r["matched"]]
    n_matched = len(matched)
    if n_matched == 0:
        print(f"{label:12s} | win: {n_windows:5d} | matched: 0 | p50: UNMEASURED | p95: UNMEASURED | p99: UNMEASURED | max: UNMEASURED | >64: 0")
        return

    ranks = np.array([r["rank"] for r in matched], dtype=np.int32)
    p50 = float(np.percentile(ranks, 50))
    p95 = float(np.percentile(ranks, 95))
    p99 = float(np.percentile(ranks, 99))
    max_rank = int(np.max(ranks))
    n_gt64 = int(np.sum(ranks > 64))

    print(
        f"{label:12s} | win: {n_windows:5d} | matched: {n_matched:5d} | "
        f"p50: {p50:4.1f} | p95: {p95:4.1f} | p99: {p99:4.1f} | "
        f"max: {max_rank:4d} | >64: {n_gt64:3d}"
    )


def main() -> None:
    """CLI entrypoint for component rank profiling."""
    parser = argparse.ArgumentParser(description="Profile Connected Component Ranks for OrbitSight")
    parser.add_argument("--dataset-dir", type=str, required=True, help="Path to Training_sets directory")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config YAML")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir).resolve()
    cfg = load_config(Path(args.config))

    gt_files = sorted(list(dataset_dir.rglob("*_bb_windows_40ms.txt")))
    all_records: List[Dict[str, Any]] = []

    print(f"Profiling component ranks across {len(gt_files)} sequences...\n")

    for gt_f in gt_files:
        seq_name = gt_f.name.replace("_bb_windows_40ms.txt", "")
        npy_matches = list(dataset_dir.rglob(f"{seq_name}_labeled_events.npy"))
        if not npy_matches:
            continue
        npy_f = npy_matches[0]
        recs = profile_sequence_component_ranks(npy_f, gt_f, cfg)
        all_records.extend(recs)

    print("=== Component Rank Distribution across All GT-Occupied Windows ===")
    summarize_ranks([r for r in all_records if r["sensor"] == "EVK4"], "EVK4")
    summarize_ranks([r for r in all_records if r["sensor"] == "DVX"], "DVX")
    summarize_ranks([r for r in all_records if r["sensor"] == "DAVIS"], "DAVIS")
    summarize_ranks(all_records, "OVERALL")

    # Noisiest decile analysis (top 10% total components per window)
    if all_records:
        comp_counts = np.array([r["total_components"] for r in all_records])
        noisy_thresh = float(np.percentile(comp_counts, 90))
        noisy_records = [r for r in all_records if r["total_components"] >= noisy_thresh]

        print(f"\n=== Component Rank Distribution (Noisiest Decile >= {noisy_thresh:.0f} components) ===")
        summarize_ranks([r for r in noisy_records if r["sensor"] == "EVK4"], "EVK4 (Noisy)")
        summarize_ranks([r for r in noisy_records if r["sensor"] == "DVX"], "DVX (Noisy)")
        summarize_ranks([r for r in noisy_records if r["sensor"] == "DAVIS"], "DAVIS (Noisy)")
        summarize_ranks(noisy_records, "OVERALL (Noisy)")


if __name__ == "__main__":
    main()
