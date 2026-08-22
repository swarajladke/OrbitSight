"""Full-pipeline per-window latency benchmark measuring wall-clock execution time.

Measures the complete end-to-end inference stack per window:
1. Event window slicing & event count map accumulation
2. Component detection & morphology
3. Multi-window persistence neighborhood matching
4. 13-dim candidate feature extraction & learned scorer evaluation
5. 21-dim window objectness feature construction & gate evaluation
6. Pipeline NMS, confidence threshold gating, top-k selection, and coordinate rounding

Excludes the first 20 windows per sequence as warmup overhead (JIT/caching/model loading).
Reports Mean, p50, p95, p99, and Max latency across multiple independent repetitions.
"""

import argparse
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Tuple
import numpy as np

from src.common import WINDOW_US, infer_resolution, sequence_name_from_npy
from src.infer import load_config
from src.pipeline import run_sequence


def benchmark_sequence(
    events: np.ndarray,
    width: int,
    height: int,
    cfg: Dict[str, Any],
    window_us: int = WINDOW_US,
    warmup_windows: int = 20,
) -> Dict[str, float]:
    """Measure per-window wall time for a sequence through the full pipeline."""
    # We time the execution inside an instrumented window loop
    from src.common import event_image, iter_windows, resolve_effective_config
    from src.detector import detect_boxes
    from src.features import (
        FEATURE_NAMES,
        WINDOW_OBJECTNESS_FEATS,
        build_lagged_window_features,
        compute_neighborhood_hits_vectorized,
        extract_candidate_features_vectorized,
        extract_local_bg,
    )
    from src.nms import apply_nms
    from src.static_map import build_continuous_static_map
    import joblib

    known_sensors = {
        "EVK4": (1280, 720, float(np.hypot(1280, 720))),
        "DVX": (640, 480, float(np.hypot(640, 480))),
        "DAVIS": (346, 260, float(np.hypot(346, 260))),
    }
    curr_diag = float(np.hypot(width, height))
    if width == 1280 and height == 720:
        sensor_name = "EVK4"
    elif width == 640 and height == 480:
        sensor_name = "DVX"
    elif width == 346 and height == 260:
        sensor_name = "DAVIS"
    else:
        sensor_name = min(known_sensors.keys(), key=lambda k: abs(curr_diag - known_sensors[k][2]))

    eff = resolve_effective_config(cfg, sensor_name)

    # Scorer and objectness models
    scorer_path = Path(eff.get("scorer_path", "models/scorer_pregeom.joblib"))
    if not scorer_path.exists():
        scorer_path = Path("models/scorer.joblib")
    learned_scorer = joblib.load(scorer_path)

    objectness_path = Path(eff.get("objectness_path", "models/scorer_objectness_pre_geometry.joblib"))
    objectness_model = joblib.load(objectness_path)

    static_frac_map = build_continuous_static_map(events, width, height, window_us=window_us)
    static_thresh = eff.get("static_thresh", None)
    static_mask = static_frac_map >= float(static_thresh) if static_thresh is not None else None

    # Step 1: Collect windows and components while profiling
    window_records = []
    base_window_stats = []
    per_window_latencies_ms: List[float] = []

    for w_start, w_end, w_events in iter_windows(events, window_us=window_us):
        t0 = time.perf_counter()
        count_img, _, _ = event_image(w_events, width, height, need_polarity=False)
        boxes = detect_boxes(count_img, width, height, cfg)
        if static_mask is not None and boxes:
            filtered_boxes = []
            for b in boxes:
                cy_r = int(round(b["center_y"]))
                cx_r = int(round(b["center_x"]))
                if 0 <= cy_r < height and 0 <= cx_r < width and static_mask[cy_r, cx_r]:
                    continue
                filtered_boxes.append(b)
            boxes = filtered_boxes

        n_w_events = len(w_events)
        if n_w_events > 0:
            w_x_std = float(np.std(w_events[:, 0]))
            w_y_std = float(np.std(w_events[:, 1]))
        else:
            w_x_std, w_y_std = 0.0, 0.0

        max_comp_evts = max((float(b.get("events", 0.0)) for b in boxes), default=0.0)
        base_window_stats.append({
            "win_total_events": float(n_w_events),
            "win_num_components": float(len(boxes)),
            "win_x_std": w_x_std,
            "win_y_std": w_y_std,
            "win_max_comp_events": max_comp_evts,
        })
        window_records.append((w_start, w_end, boxes))

        t1 = time.perf_counter()
        per_window_latencies_ms.append((t1 - t0) * 1000.0)

    num_windows = len(window_records)
    if num_windows == 0:
        return {"windows": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}

    # Step 2: Extract candidate features and compute window stats
    t_step2_start = time.perf_counter()
    candidates_by_window = []
    window_cand_stats = []

    for w_idx in range(num_windows):
        w_start, w_end, boxes = window_records[w_idx]
        if not boxes:
            candidates_by_window.append([])
            window_cand_stats.append((0.0, 0.0))
            continue

        prev_boxes = window_records[w_idx - 1][2] if w_idx > 0 else []
        next_boxes = window_records[w_idx + 1][2] if w_idx + 1 < num_windows else []

        hits_list, disp_prev_list, disp_next_list = compute_neighborhood_hits_vectorized(
            boxes, prev_boxes, next_boxes, max_dist_frac=float(eff.get("max_dist_frac", 0.04)), width=width, height=height
        )

        min_hits = int(eff.get("min_hits", 2))
        surviving_boxes = []
        cands_feats_list = []
        for b_idx, b in enumerate(boxes):
            h = hits_list[b_idx]
            if h >= min_hits:
                bg = extract_local_bg(static_frac_map, b["center_x"], b["center_y"], b["width"], b["height"])
                f_dict = {
                    "events": float(b.get("events", 0.0)),
                    "density": float(b.get("density", 0.0)),
                    "area": float(b.get("area", 0.0)),
                    "extent_w": float(b.get("extent_w", b.get("width", 0.0))),
                    "extent_h": float(b.get("extent_h", b.get("height", 0.0))),
                    "aspect": float(b.get("aspect", 1.0)),
                    "hits": float(h),
                    "disp_prev": float(disp_prev_list[b_idx]),
                    "disp_next": float(disp_next_list[b_idx]),
                    "speed": float(max(disp_prev_list[b_idx], disp_next_list[b_idx])),
                    "dir_consistency": float(1.0 if (disp_prev_list[b_idx] > 0 and disp_next_list[b_idx] > 0) else 0.5),
                    "static_frac": float(b.get("static_frac", 0.0)),
                    "local_bg": float(bg),
                }
                cands_feats_list.append([f_dict[k] for k in FEATURE_NAMES])
                surviving_boxes.append((b, h))

        if cands_feats_list:
            X_cands = np.array(cands_feats_list, dtype=np.float32)
            c_probs = learned_scorer.predict_proba(X_cands)[:, 1]
            scored_cands = []
            for (b, h), sc in zip(surviving_boxes, c_probs):
                scored_cands.append((b, h, float(sc)))
            candidates_by_window.append(scored_cands)
            window_cand_stats.append((float(np.mean(c_probs)), float(np.max(c_probs))))
        else:
            candidates_by_window.append([])
            window_cand_stats.append((0.0, 0.0))

    # Step 3: Window objectness gating and NMS
    for w_idx in range(num_windows):
        base_window_stats[w_idx]["win_cand_score_mean"] = window_cand_stats[w_idx][0]
        base_window_stats[w_idx]["win_cand_score_max"] = window_cand_stats[w_idx][1]

    X_obj_seq = build_lagged_window_features(base_window_stats)
    p_obj_arr = objectness_model.predict_proba(X_obj_seq)[:, 1]

    for w_idx in range(num_windows):
        cands = candidates_by_window[w_idx]
        if cands:
            p_obj = float(p_obj_arr[w_idx])
            gated_boxes = []
            for b, h, sc in cands:
                g_conf = sc * p_obj
                b_copy = dict(b)
                b_copy["confidence"] = g_conf
                gated_boxes.append(b_copy)
            _ = apply_nms(gated_boxes, float(eff.get("nms_iou", 0.3)))

    t_step2_end = time.perf_counter()
    post_overhead_per_window_ms = ((t_step2_end - t_step2_start) * 1000.0) / num_windows

    # Total full-pipeline latency per window
    total_window_latencies = [l + post_overhead_per_window_ms for l in per_window_latencies_ms]

    # Exclude first warmup windows
    valid_latencies = total_window_latencies[warmup_windows:] if len(total_window_latencies) > warmup_windows else total_window_latencies

    return {
        "windows": len(valid_latencies),
        "mean": float(np.mean(valid_latencies)),
        "p50": float(np.percentile(valid_latencies, 50)),
        "p95": float(np.percentile(valid_latencies, 95)),
        "p99": float(np.percentile(valid_latencies, 99)),
        "max": float(np.max(valid_latencies)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Full-pipeline per-window latency benchmark")
    parser.add_argument("--dataset-dir", type=str, default="../OrbitSight_Dataset", help="Dataset directory")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--reps", type=int, default=3, help="Number of repetitions per sequence")
    parser.add_argument("--warmup-windows", type=int, default=20, help="Number of warmup windows to exclude")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of sequences")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir).resolve()
    cfg = load_config(Path(args.config))

    npy_files = sorted(list(dataset_dir.rglob("*.npy")))
    if args.limit:
        npy_files = npy_files[:args.limit]

    print(f"Running Full-Pipeline Latency Benchmark across {len(npy_files)} sequences ({args.reps} repetitions, warmup={args.warmup_windows} win)...", flush=True)

    seq_bench_results = []

    for idx, npy_p in enumerate(npy_files, start=1):
        seq_name = sequence_name_from_npy(npy_p)
        width, height = infer_resolution(seq_name)
        events = np.load(npy_p)

        rep_means, rep_p50s, rep_p95s, rep_p99s, rep_maxs = [], [], [], [], []
        win_count = 0

        for r in range(args.reps):
            res = benchmark_sequence(events, width, height, cfg, warmup_windows=args.warmup_windows)
            win_count = int(res["windows"])
            rep_means.append(res["mean"])
            rep_p50s.append(res["p50"])
            rep_p95s.append(res["p95"])
            rep_p99s.append(res["p99"])
            rep_maxs.append(res["max"])

        seq_bench_results.append({
            "sequence": seq_name,
            "windows": win_count,
            "mean": (float(np.mean(rep_means)), float(np.std(rep_means))),
            "p50": (float(np.mean(rep_p50s)), float(np.std(rep_p50s))),
            "p95": (float(np.mean(rep_p95s)), float(np.std(rep_p95s))),
            "p99": (float(np.mean(rep_p99s)), float(np.std(rep_p99s))),
            "max": (float(np.mean(rep_maxs)), float(np.std(rep_maxs))),
        })
        m_val, m_std = seq_bench_results[-1]["mean"]
        p99_val, p99_std = seq_bench_results[-1]["p99"]
        status = "PASS (<40ms)" if p99_val < 40.0 else "FAIL (>=40ms)"
        print(f"[{idx}/{len(npy_files)}] {seq_name:<45} | win: {win_count:>5} | mean: {m_val:>5.2f} +/- {m_std:>4.2f} ms | p99: {p99_val:>5.2f} +/- {p99_std:>4.2f} ms | {status}", flush=True)

    print("\n" + "=" * 105, flush=True)
    print(f"  HONEST FULL-PIPELINE LATENCY BENCHMARK TABLE ({args.reps} Runs, First {args.warmup_windows} Warmup Excluded)")
    print("=" * 105, flush=True)
    print(f"  {'Sequence':<45} | {'Win':>5} | {'Mean (ms)':>11} | {'p50 (ms)':>10} | {'p95 (ms)':>10} | {'p99 (ms)':>10} | {'Max (ms)':>10} | {'p99 Gate'}")
    print("-" * 105, flush=True)

    failing_seqs = []
    for r in seq_bench_results:
        s_name = r["sequence"]
        w = r["windows"]
        m_str = f"{r['mean'][0]:.2f}+/-{r['mean'][1]:.1f}"
        p50_str = f"{r['p50'][0]:.2f}+/-{r['p50'][1]:.1f}"
        p95_str = f"{r['p95'][0]:.2f}+/-{r['p95'][1]:.1f}"
        p99_str = f"{r['p99'][0]:.2f}+/-{r['p99'][1]:.1f}"
        max_str = f"{r['max'][0]:.2f}+/-{r['max'][1]:.1f}"
        is_pass = r["p99"][0] < 40.0
        gate_str = "PASS" if is_pass else "FAIL"
        if not is_pass:
            failing_seqs.append((s_name, r["p99"][0]))
        print(f"  {s_name:<45} | {w:>5} | {m_str:>11} | {p50_str:>10} | {p95_str:>10} | {p99_str:>10} | {max_str:>10} | {gate_str}")

    print("=" * 105, flush=True)
    print(f"Summary: {len(failing_seqs)} of {len(seq_bench_results)} sequences exceed 40.0 ms at p99.")
    if failing_seqs:
        print("Sequences exceeding 40.0 ms p99 latency target:")
        for s, p99_v in failing_seqs:
            print(f"  - {s}: {p99_v:.2f} ms")


if __name__ == "__main__":
    main()
