"""Inference pipeline CLI for space-object detection on event-camera data."""

import argparse
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
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
) -> Tuple[float, int, List[float]]:
    """Process single event sequence file via unified pipeline, write predictions."""
    start_time = time.perf_counter()
    deadline_ts = (
        time.monotonic() + time_budget_sec if time_budget_sec is not None else None
    )

    seq_name = sequence_name_from_npy(npy_path)
    file_mb = npy_path.stat().st_size / (1024 * 1024)
    events = load_events(npy_path)

    print(
        f"Processing sequence '{seq_name}' ({file_mb:.2f} MB, {events.shape[0]} events)...",
        flush=True,
    )

    width, height = infer_resolution(seq_name, events[:, 0], events[:, 1])
    window_us = int(cfg.get("window_us", WINDOW_US))

    # Instrument per-window compute timing during unified pipeline execution
    import src.pipeline as pipeline_mod
    orig_iter = pipeline_mod.iter_windows
    pass1_times: List[float] = []

    def instrumented_iter(ev_arr, window_us=WINDOW_US):
        gen = orig_iter(ev_arr, window_us)
        for item in gen:
            t0 = time.perf_counter()
            yield item
            t1 = time.perf_counter()
            pass1_times.append((t1 - t0) * 1000.0)

    pipeline_mod.iter_windows = instrumented_iter
    try:
        predictions, num_windows = run_sequence(
            events,
            width,
            height,
            cfg,
            window_us=window_us,
            max_windows=max_windows,
            deadline_ts=deadline_ts,
        )
    finally:
        pipeline_mod.iter_windows = orig_iter

    total_elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    if len(pass1_times) > 0:
        win_times = pass1_times[:num_windows]
    else:
        avg_w = total_elapsed_ms / num_windows if num_windows > 0 else 0.0
        win_times = [avg_w] * num_windows

    if deadline_ts is not None and time.monotonic() >= deadline_ts:
        print(
            f"[SKIPPED] {seq_name}: exceeded time budget at window {num_windows}",
            flush=True,
        )

    header_line = "sequence_id\twindow_start_timestamp_us\twindow_end_timestamp_us\tcenter_x\tcenter_y\twidth\theight\tclass_id\tconfidence"
    output_lines: List[str] = [header_line]

    for ws, we, cx, cy, bw, bh, conf in predictions:
        output_lines.append(f"{seq_name}\t{ws}\t{we}\t{cx}\t{cy}\t{bw}\t{bh}\t0\t{conf:.4f}")

    content = "\n".join(output_lines) + "\n"

    output_dir.mkdir(parents=True, exist_ok=True)
    target_files = [
        output_dir / f"{seq_name}.txt",
        output_dir / f"{seq_name}_pred.txt",
        output_dir / f"{seq_name}_bb_windows_40ms.txt",
    ]
    for out_p in target_files:
        with open(out_p, "w", encoding="utf-8", newline="\n") as f:
            f.write(content)

    return total_elapsed_ms, num_windows, win_times


def main() -> None:
    """CLI entrypoint for inference pipeline."""
    parser = argparse.ArgumentParser(
        description="OrbitSight Neuromorphic Baseline Detection Pipeline"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default=None,
        help="Path to input dataset directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
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

    raw_input_dir = args.input_dir or os.environ.get("ORBITSIGHT_DATASET_DIR") or "../OrbitSight_Dataset"
    raw_output_dir = args.output_dir or os.environ.get("ORBITSIGHT_OUTPUT_DIR") or "predictions"

    input_dir = Path(raw_input_dir).resolve()
    output_dir = Path(raw_output_dir).resolve()

    print(f"Dataset directory resolved to: {input_dir}", flush=True)
    print(f"Output directory resolved to: {output_dir}", flush=True)

    npy_files = sorted(list(input_dir.rglob("*.npy")))
    print(
        f"Found {len(npy_files)} sequence file(s) matching *.npy",
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

    output_dir.mkdir(parents=True, exist_ok=True)

    max_w = float("inf") if not args.max_windows else float(args.max_windows)

    total_time_ms = 0.0
    total_windows = 0
    seq_latency_records = []

    for idx, npy_file in enumerate(npy_files, start=1):
        seq_name = sequence_name_from_npy(npy_file)
        print(
            f"\n[{idx}/{len(npy_files)}] Starting sequence processing: {seq_name}...",
            flush=True,
        )
        seq_ms, num_w, win_times = process_sequence(
            npy_file,
            output_dir,
            cfg,
            max_windows=max_w,
            time_budget_sec=args.time_budget_sec,
        )
        ms_per_window = seq_ms / num_w if num_w > 0 else 0.0
        total_time_ms += seq_ms
        total_windows += num_w
        
        if win_times:
            w_arr = np.array(win_times, dtype=np.float64)
            p50 = float(np.percentile(w_arr, 50))
            p95 = float(np.percentile(w_arr, 95))
            p99 = float(np.percentile(w_arr, 99))
            mx = float(np.max(w_arr))
        else:
            p50, p95, p99, mx = ms_per_window, ms_per_window, ms_per_window, ms_per_window

        seq_latency_records.append((seq_name, num_w, seq_ms, ms_per_window, p50, p95, p99, mx))
        print(
            f"Done '{seq_name}': {seq_ms:.2f} ms total, {ms_per_window:.2f} ms/window across {num_w} windows (pass1 p50: {p50:.2f}, p99: {p99:.2f} ms).",
            flush=True,
        )

    avg_ms_per_window = total_time_ms / total_windows if total_windows > 0 else 0.0

    print("\n" + "=" * 135, flush=True)
    print("  PER-SEQUENCE LATENCY SUMMARY (Pass 1 Percentiles & Amortized Total)", flush=True)
    print("=" * 135, flush=True)
    num_p99_pass = 0

    for s_name, n_win, s_ms, m_ms, p50, p95, p99, mx in seq_latency_records:
        if p99 < 40.0:
            status = "PASS (Pass-1 p99 < 40ms)"
            num_p99_pass += 1
        else:
            status = f"EXCEEDS (Pass-1 p99 = {p99:.2f} ms)"
        print(
            f"  {s_name:<42} | {n_win:>5} win | amortized_total: {m_ms:>5.2f} ms | pass1 p50: {p50:>5.2f} ms | p95: {p95:>5.2f} ms | p99: {p99:>5.2f} ms | max: {mx:>6.2f} ms | {status}",
            flush=True,
        )
    print("-" * 135, flush=True)
    print(f"Mean compute throughput: {avg_ms_per_window:.2f} ms/window (not a latency guarantee)", flush=True)
    print(
        f"Sequences meeting Pass-1 p99 < 40ms: {num_p99_pass}/{len(seq_latency_records)}  "
        "(Pass 1 only — NOT end-to-end latency; see src/latency_bench.py for full-pipeline p99)",
        flush=True,
    )
    print("Note: First sequence includes model loading and JIT warmup latency overhead.", flush=True)


if __name__ == "__main__":
    main()
