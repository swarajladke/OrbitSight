"""Inference pipeline CLI for space-object detection on event-camera data."""

import argparse
import math
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Tuple
import numpy as np

from src.common import (
    WINDOW_US,
    event_image,
    infer_resolution,
    iter_windows,
    load_events,
    sequence_name_from_npy,
)
from src.detector import detect_boxes


def _parse_yaml_value(val_str: str) -> Any:
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


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load configuration dictionary, with zero-dependency fallback parser if pyyaml is missing."""
    if not config_path.exists():
        return {}

    try:
        import yaml

        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except ImportError:
        cfg: Dict[str, Any] = {}
        current_section: str = ""

        with open(config_path, "r", encoding="utf-8") as f:
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
                        cfg[key] = _parse_yaml_value(val_str)
                elif indent > 0 and current_section:
                    if not isinstance(cfg.get(current_section), dict):
                        cfg[current_section] = {}
                    cfg[current_section][key] = _parse_yaml_value(val_str)

        return cfg


def compute_confidence(
    box: Dict[str, float],
    hits: int,
    cfg: Dict[str, Any],
) -> float:
    """Compute deterministic weighted confidence score clipped to [0.01, 1.0]."""
    weights = cfg.get(
        "confidence_weights",
        {"density": 0.25, "events": 0.35, "compactness": 0.20, "persistence": 0.20},
    )
    w_den = float(weights.get("density", 0.25))
    w_evt = float(weights.get("events", 0.35))
    w_cmp = float(weights.get("compactness", 0.20))
    w_per = float(weights.get("persistence", 0.20))

    sub_density = min(1.0, box["density"])
    min_evt = float(cfg.get("min_events_in_box", 3))
    sub_events = min(1.0, box["events"] / (min_evt * 5.0))
    sub_compactness = 1.0 / (1.0 + abs(box["aspect"] - 1.0))
    sub_persistence = min(1.0, hits / 3.0)

    score = (
        w_den * sub_density
        + w_evt * sub_events
        + w_cmp * sub_compactness
        + w_per * sub_persistence
    )

    return float(np.clip(score, 0.01, 1.0))


def process_sequence(
    npy_path: Path,
    output_dir: Path,
    cfg: Dict[str, Any],
    max_windows: float = float("inf"),
) -> Tuple[float, int]:
    """Process single event sequence file, execute detection, apply persistence filter, write predictions."""
    seq_name = sequence_name_from_npy(npy_path)
    file_mb = npy_path.stat().st_size / (1024 * 1024)
    events = load_events(npy_path)

    print(
        f"Processing sequence '{seq_name}' ({file_mb:.2f} MB, {events.shape[0]} events)...",
        flush=True,
    )
    start_time = time.perf_counter()

    width, height = infer_resolution(seq_name, events[:, 0], events[:, 1])
    window_us = int(cfg.get("window_us", WINDOW_US))

    window_records: List[Tuple[int, int, List[Dict[str, float]]]] = []
    window_count = 0

    for w_start, w_end, w_events in iter_windows(events, window_us=window_us):
        count_img, _, _ = event_image(w_events, width, height)
        boxes = detect_boxes(count_img, width, height, cfg)
        window_records.append((w_start, w_end, boxes))
        window_count += 1
        if window_count >= max_windows:
            break

    num_windows = len(window_records)
    min_hits = int(cfg.get("min_hits", 1))
    max_dist_frac = float(cfg.get("max_dist_frac", 0.08))
    diagonal = math.hypot(width, height)
    max_dist = max_dist_frac * diagonal

    output_lines: List[str] = [
        "window_start_timestamp_us\twindow_end_timestamp_us\tcenter_x\tcenter_y\twidth\theight\tconfidence"
    ]

    for w_idx in range(num_windows):
        w_start, w_end, boxes = window_records[w_idx]
        if not boxes:
            continue

        prev_boxes = window_records[w_idx - 1][2] if w_idx > 0 else []
        next_boxes = (
            window_records[w_idx + 1][2] if w_idx < num_windows - 1 else []
        )

        for box in boxes:
            hits = 1
            has_prev = any(
                math.hypot(
                    box["center_x"] - p["center_x"],
                    box["center_y"] - p["center_y"],
                )
                <= max_dist
                for p in prev_boxes
            )
            if has_prev:
                hits += 1

            has_next = any(
                math.hypot(
                    box["center_x"] - n["center_x"],
                    box["center_y"] - n["center_y"],
                )
                <= max_dist
                for n in next_boxes
            )
            if has_next:
                hits += 1

            if min_hits >= 2 and hits < min_hits:
                continue

            conf = compute_confidence(box, hits, cfg)

            cx = int(round(box["center_x"]))
            cy = int(round(box["center_y"]))
            bw = int(round(box["width"]))
            bh = int(round(box["height"]))

            output_lines.append(
                f"{w_start}\t{w_end}\t{cx}\t{cy}\t{bw}\t{bh}\t{conf:.4f}"
            )

    output_path = output_dir / f"{seq_name}_pred.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines) + "\n")

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    return elapsed_ms, num_windows


def main() -> None:
    """CLI entrypoint for inference pipeline."""
    parser = argparse.ArgumentParser(
        description="OrbitSight Neuromorphic Baseline Detection Pipeline"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="../OrbitSight_Dataset",
        help="Path to input dataset directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="predictions",
        help="Path to output predictions directory",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config YAML file",
    )
    parser.add_argument(
        "--sequences",
        type=str,
        default=None,
        help="Comma-separated sequence names to process",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only first N sequences",
    )
    parser.add_argument(
        "--max-windows",
        type=int,
        default=None,
        help="Stop after N windows per sequence (smoke test)",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    print(f"Input directory resolved to: {input_dir}", flush=True)

    npy_files = sorted(list(input_dir.rglob("*_labeled_events.npy")))
    print(
        f"Found {len(npy_files)} sequence file(s) matching *_labeled_events.npy",
        flush=True,
    )

    if len(npy_files) == 0:
        print(f"Error: No sequence files found in {input_dir}", file=sys.stderr)
        sys.exit(1)

    if args.sequences:
        target_seqs = {s.strip() for s in args.sequences.split(",") if s.strip()}
        npy_files = [
            p for p in npy_files if sequence_name_from_npy(p) in target_seqs
        ]
        print(f"Filtered to {len(npy_files)} specified sequence(s).", flush=True)

    if args.limit and args.limit > 0:
        npy_files = npy_files[: args.limit]
        print(f"Limited processing to first {len(npy_files)} sequence(s).", flush=True)

    if not npy_files:
        print("No matching sequences to process after filtering.", flush=True)
        return

    config_path = Path(args.config)
    cfg = load_config(config_path)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    max_w = float("inf") if not args.max_windows else float(args.max_windows)

    total_time_ms = 0.0
    total_windows = 0

    for idx, npy_file in enumerate(npy_files, start=1):
        print(
            f"\n[{idx}/{len(npy_files)}] Starting sequence processing...",
            flush=True,
        )
        seq_ms, num_w = process_sequence(npy_file, output_dir, cfg, max_windows=max_w)
        ms_per_window = seq_ms / num_w if num_w > 0 else 0.0
        seq_name = sequence_name_from_npy(npy_file)
        print(
            f"Done '{seq_name}': {seq_ms:.2f} ms total, "
            f"{ms_per_window:.2f} ms/window across {num_w} windows.",
            flush=True,
        )
        total_time_ms += seq_ms
        total_windows += num_w

    avg_ms_per_window = total_time_ms / total_windows if total_windows > 0 else 0.0
    print("\n--------------------------------------------------", flush=True)
    print(f"Overall average: {avg_ms_per_window:.2f} ms/window", flush=True)

    if avg_ms_per_window < 40.0:
        print("[PASS] Real-time target MET: latency < 40.0 ms/window.", flush=True)
    else:
        print("[FAIL] Real-time target MISSED: latency >= 40.0 ms/window.", flush=True)


if __name__ == "__main__":
    main()
