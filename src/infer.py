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
    infer_resolution,
    load_events,
    sequence_name_from_npy,
)
from src.pipeline import compute_confidence, run_sequence


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


def process_sequence(
    npy_path: Path,
    output_dir: Path,
    cfg: Dict[str, Any],
    max_windows: float = float("inf"),
    time_budget_sec: Optional[float] = None,
) -> Tuple[float, int]:
    """Process single event sequence file via unified pipeline, write predictions."""
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

    deadline_ts = (
        time.monotonic() + time_budget_sec if time_budget_sec is not None else None
    )

    predictions, num_windows = run_sequence(
        events,
        width,
        height,
        cfg,
        window_us=window_us,
        max_windows=max_windows,
        deadline_ts=deadline_ts,
    )

    if deadline_ts is not None and time.monotonic() >= deadline_ts:
        print(
            f"[SKIPPED] {seq_name}: exceeded time budget at window {num_windows}",
            flush=True,
        )

    output_lines: List[str] = [
        "window_start_timestamp_us\twindow_end_timestamp_us\tcenter_x\tcenter_y\twidth\theight\tconfidence"
    ]

    for ws, we, cx, cy, bw, bh, conf in predictions:
        output_lines.append(f"{ws}\t{we}\t{cx}\t{cy}\t{bw}\t{bh}\t{conf:.4f}")

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
    parser.add_argument(
        "--time-budget-sec",
        type=float,
        default=None,
        help="Abort sequence if processing exceeds budget in seconds",
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
        seq_ms, num_w = process_sequence(
            npy_file,
            output_dir,
            cfg,
            max_windows=max_w,
            time_budget_sec=args.time_budget_sec,
        )
        ms_per_window = seq_ms / num_w if num_w > 0 else 0.0
        total_time_ms += seq_ms
        total_windows += num_w
        print(
            f"Done '{sequence_name_from_npy(npy_file)}': {seq_ms:.2f} ms total, {ms_per_window:.2f} ms/window across {num_w} windows.",
            flush=True,
        )

    avg_ms_per_window = total_time_ms / total_windows if total_windows > 0 else 0.0
    print("\n--------------------------------------------------", flush=True)
    print(f"Overall average: {avg_ms_per_window:.2f} ms/window", flush=True)

    if avg_ms_per_window < 40.0:
        print("[PASS] Real-time target MET: latency < 40.0 ms/window.", flush=True)
    else:
        print("[FAIL] Real-time target MISSED: latency >= 40.0 ms/window.", flush=True)


if __name__ == "__main__":
    main()
