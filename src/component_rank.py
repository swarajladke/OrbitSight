"""Component Rank Profiling Tool for OrbitSight Connected-Component Extraction.

Measures the rank of ground-truth matched bounding boxes among connected components
sorted by event count descending across all GT-occupied windows.
Uses exact detect_boxes implementation and single-source src.metrics.iou.
"""

import argparse
import csv
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.common import WINDOW_US, infer_resolution, iter_windows, load_events, sequence_name_from_npy
from src.detector import detect_boxes
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
    static_thresh = float(eff.get("static_thresh", 0.5))

    # Profile config: set max_components_per_window to unbounded to measure true full rank
    profile_cfg = {**eff, "max_components_per_window": 10000}

    # Continuous static mask
    static_map = build_continuous_static_map(events, width, height, window_us=WINDOW_US)
    static_mask = static_map >= static_thresh

    results: List[Dict[str, Any]] = []

    for ws, we, w_events in iter_windows(events, window_us=WINDOW_US):
        if ws not in gt_by_ts:
            continue
        if len(w_events) == 0:
            for _ in gt_by_ts[ws]:
                results.append({
                    "sequence": seq_name,
                    "sensor": sensor_name,
                    "matched": False,
                    "rank": -1,
                    "total_components": 0,
                })
            continue

        # Run exact detector connected-component extraction and box construction
        candidate_boxes = detect_boxes(
            w_events, width, height, static_mask, cfg=profile_cfg
        )

        total_comps = len(candidate_boxes)
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

        for g_cx, g_cy, g_bw, g_bh in gt_by_ts[ws]:
            best_iou = 0.0
            best_rank = -1
            gt_tuple = (float(g_cx), float(g_cy), float(g_bw), float(g_bh))

            for rank_idx, c_box in enumerate(candidate_boxes, start=1):
                c_tuple = (
                    float(c_box["center_x"]),
                    float(c_box["center_y"]),
                    float(c_box["width"]),
                    float(c_box["height"]),
                )
                cur_iou = iou(c_tuple, gt_tuple)
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
        print(f"{label:15s} | No records.")
        return

    n_windows = len(records)
    matched = [r for r in records if r["matched"]]
    n_matched = len(matched)
    if n_matched == 0:
        print(f"{label:15s} | win: {n_windows:5d} | matched: {n_matched:5d} | p50: UNMEASURED | p95: UNMEASURED | p99: UNMEASURED | max: UNMEASURED | >64: 0")
        return

    ranks = np.array([r["rank"] for r in matched], dtype=np.int32)
    p50 = float(np.percentile(ranks, 50))
    p95 = float(np.percentile(ranks, 95))
    p99 = float(np.percentile(ranks, 99))
    max_rank = int(np.max(ranks))
    n_gt64 = int(np.sum(ranks > 64))

    print(
        f"{label:15s} | win: {n_windows:5d} | matched: {n_matched:5d} | "
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

    print(f"Profiling component ranks across {len(gt_files)} sequences with exact detect_boxes...\n")

    for gt_f in gt_files:
        seq_name = gt_f.name.replace("_bb_windows_40ms.txt", "")
        npy_matches = list(dataset_dir.rglob(f"{seq_name}_labeled_events.npy"))
        if not npy_matches:
            continue
        npy_f = npy_matches[0]
        recs = profile_sequence_component_ranks(npy_f, gt_f, cfg)
        all_records.extend(recs)

    print("=== Component Rank Distribution across All GT-Occupied Windows ===")
    evk4_recs = [r for r in all_records if r["sensor"] == "EVK4"]
    dvx_recs = [r for r in all_records if r["sensor"] == "DVX"]
    davis_recs = [r for r in all_records if r["sensor"] == "DAVIS"]

    summarize_ranks(evk4_recs, "EVK4")
    summarize_ranks(dvx_recs, "DVX")
    summarize_ranks(davis_recs, "DAVIS")
    summarize_ranks(all_records, "OVERALL")

    # True per-sensor noisiest decile analysis (top 10% component count per sensor)
    print("\n=== Component Rank Distribution (Per-Sensor Noisiest Decile, Top 10%) ===")
    noisy_all: List[Dict[str, Any]] = []

    for s_name, s_recs in [("EVK4", evk4_recs), ("DVX", dvx_recs), ("DAVIS", davis_recs)]:
        if not s_recs:
            continue
        s_counts = np.array([r["total_components"] for r in s_recs])
        p90_thresh = float(np.percentile(s_counts, 90))
        s_noisy = [r for r in s_recs if r["total_components"] >= p90_thresh]
        noisy_all.extend(s_noisy)
        print(f"[{s_name}] p90 Threshold = {p90_thresh:.1f} components:")
        summarize_ranks(s_noisy, f"{s_name} (Noisy)")

    print("-" * 65)
    summarize_ranks(noisy_all, "OVERALL (Noisy)")


if __name__ == "__main__":
    main()
