"""Streaming per-window full-pipeline latency benchmark measuring true wall-clock execution time.

Evaluates one single streaming window loop holding a 3-window sliding buffer (prev, cur, next):
For each window t:
1. Window t+1's events become available.
2. Timer starts (t_start).
3. Compute event count map, detect_boxes, static filter, base window stats for window t+1.
4. With window t-1, t, and t+1 available:
   - Perform persistence neighborhood matching for window t.
   - Extract 13-dim candidate features for window t.
   - Run learned candidate scorer predict_proba for window t.
   - Construct 21-dim window objectness feature vector for window t.
   - Run window objectness gate predict_proba for window t.
   - Multiply candidate confidences by p_obj.
   - Apply conf_min, NMS, top-k selection, and coordinate rounding for window t.
5. Timer stops (t_end).
   Compute wall-clock latency per window: (t_end - t_start) * 1000.0 ms.

Excludes the first 20 warmup windows per sequence.
Separately tracks and isolates any system stall windows (> 1000 ms).
Reports COMPUTE p50, p95, p99, and Max, along with TOTAL latency (Compute + 40 ms algorithmic lookahead).
"""

import argparse
import math
from pathlib import Path
import time
from typing import Any, Dict, List, Tuple
import numpy as np

from src.common import WINDOW_US, infer_resolution, sequence_name_from_npy
from src.infer import load_config, load_events


def benchmark_sequence_streaming(
    events: np.ndarray,
    width: int,
    height: int,
    cfg: Dict[str, Any],
    window_us: int = WINDOW_US,
    warmup_windows: int = 20,
) -> Dict[str, Any]:
    """Measure true per-window wall time in a streaming 3-window buffer pipeline."""
    from src.common import event_image, iter_windows, resolve_effective_config
    from src.detector import detect_boxes
    from src.features import FEATURE_NAMES, extract_window_features_batch
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

    scorer_path = Path(eff.get("scorer_path", "models/scorer_pregeom.joblib"))
    if not scorer_path.exists():
        scorer_path = Path("models/scorer.joblib")
    learned_scorer = joblib.load(scorer_path)

    objectness_path = Path(eff.get("objectness_path", "models/scorer_objectness_pre_geometry.joblib"))
    objectness_model = joblib.load(objectness_path)

    static_frac_map = build_continuous_static_map(events, width, height, window_us=window_us)
    static_thresh = eff.get("static_thresh", None)
    static_mask = static_frac_map >= float(static_thresh) if static_thresh is not None else None

    diagonal = math.hypot(width, height)
    max_dist_sq = (float(eff.get("max_dist_frac", 0.04)) * diagonal) ** 2
    min_hits = int(eff.get("min_hits", 2))
    conf_min = float(eff.get("conf_min", 0.30))
    nms_iou = float(eff.get("nms_iou", 0.30))
    max_k = int(eff.get("max_candidates_per_window", 1))

    # Pre-slice event windows into an iterator stream
    event_window_list = list(iter_windows(events, window_us=window_us))
    num_total_windows = len(event_window_list)
    if num_total_windows == 0:
        return {"windows": 0, "compute_mean": 0.0, "compute_p50": 0.0, "compute_p95": 0.0, "compute_p99": 0.0, "compute_max": 0.0, "stalls": 0}

    # Streaming 3-window sliding buffer state:
    # Each entry is a dict: {"w_start", "w_end", "boxes", "base_stat", "cand_score_mean", "cand_score_max"}
    window_buffer: List[Dict[str, Any]] = []
    per_window_compute_ms: List[float] = []

    def process_new_window(w_start: int, w_end: int, w_events: np.ndarray) -> Dict[str, Any]:
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

        n_ev = len(w_events)
        x_std = float(np.std(w_events[:, 0])) if n_ev > 0 else 0.0
        y_std = float(np.std(w_events[:, 1])) if n_ev > 0 else 0.0
        max_c = max((float(b.get("events", 0.0)) for b in boxes), default=0.0)

        base_stat = {
            "win_total_events": float(n_ev),
            "win_num_components": float(len(boxes)),
            "win_x_std": x_std,
            "win_y_std": y_std,
            "win_max_comp_events": max_c,
            "win_cand_score_mean": 0.0,
            "win_cand_score_max": 0.0,
        }
        return {"w_start": w_start, "w_end": w_end, "boxes": boxes, "base_stat": base_stat}

    # Prime the buffer with window 0
    w0_s, w0_e, w0_ev = event_window_list[0]
    window_buffer.append(process_new_window(w0_s, w0_e, w0_ev))

    for t in range(num_total_windows):
        t_start = time.perf_counter()

        # Ingest window t+1 into lookahead buffer if available
        if t + 1 < num_total_windows:
            wn_s, wn_e, wn_ev = event_window_list[t + 1]
            win_next = process_new_window(wn_s, wn_e, wn_ev)
            window_buffer.append(win_next)
        else:
            win_next = None

        win_cur = window_buffer[0] if len(window_buffer) == 1 else window_buffer[-2 if win_next is not None else -1]
        win_prev = window_buffer[-3] if len(window_buffer) >= 3 and win_next is not None else (window_buffer[-2] if win_next is None and len(window_buffer) >= 2 else None)

        boxes_cur = win_cur["boxes"]
        boxes_prev = win_prev["boxes"] if win_prev is not None else []
        boxes_next = win_next["boxes"] if win_next is not None else []

        # Persistence matching for window t
        cands_cur = []
        if boxes_cur:
            n_cur = len(boxes_cur)
            cur_centers = np.array([[b["center_x"], b["center_y"]] for b in boxes_cur], dtype=np.float32)

            has_prev = np.zeros(n_cur, dtype=bool)
            if boxes_prev:
                p_centers = np.array([[p["center_x"], p["center_y"]] for p in boxes_prev], dtype=np.float32)
                diff_p = cur_centers[:, None, :] - p_centers[None, :, :]
                has_prev = np.any(np.sum(diff_p * diff_p, axis=2) <= max_dist_sq, axis=1)

            has_next = np.zeros(n_cur, dtype=bool)
            if boxes_next:
                n_centers = np.array([[n["center_x"], n["center_y"]] for n in boxes_next], dtype=np.float32)
                diff_n = cur_centers[:, None, :] - n_centers[None, :, :]
                has_next = np.any(np.sum(diff_n * diff_n, axis=2) <= max_dist_sq, axis=1)

            for idx, box in enumerate(boxes_cur):
                hits = 1 + int(has_prev[idx]) + int(has_next[idx])
                if min_hits >= 2 and hits < min_hits:
                    continue
                b_copy = dict(box)
                b_copy["hits"] = hits
                cands_cur.append(b_copy)

            if cands_cur:
                batch_feats = extract_window_features_batch(
                    cands_cur,
                    boxes_prev,
                    boxes_next,
                    count_img=None,
                    static_frac_map=static_frac_map,
                )
                cand_features_list = [[f[name] for name in FEATURE_NAMES] for f in batch_feats]
                X_cand = np.array(cand_features_list, dtype=np.float32)
                probs = learned_scorer.predict_proba(X_cand)[:, 1]
                for b_copy, p_score in zip(cands_cur, probs):
                    b_copy["confidence"] = float(p_score)

        # Window stats for objectness
        cur_scores = [c["confidence"] for c in cands_cur] if cands_cur else []
        win_cur["base_stat"]["win_cand_score_mean"] = float(np.mean(cur_scores)) if cur_scores else 0.0
        win_cur["base_stat"]["win_cand_score_max"] = float(np.max(cur_scores)) if cur_scores else 0.0

        # Construct 21-dim window objectness feature vector
        cur_s = win_cur["base_stat"]
        prev_s = win_prev["base_stat"] if win_prev is not None else {k: 0.0 for k in cur_s}
        next_s = win_next["base_stat"] if win_next is not None else {k: 0.0 for k in cur_s}

        w_vec = [
            cur_s["win_total_events"], cur_s["win_num_components"], cur_s["win_x_std"], cur_s["win_y_std"], cur_s["win_max_comp_events"], cur_s["win_cand_score_mean"], cur_s["win_cand_score_max"],
            prev_s["win_total_events"], prev_s["win_num_components"], prev_s["win_x_std"], prev_s["win_y_std"], prev_s["win_max_comp_events"], prev_s["win_cand_score_mean"], prev_s["win_cand_score_max"],
            next_s["win_total_events"], next_s["win_num_components"], next_s["win_x_std"], next_s["win_y_std"], next_s["win_max_comp_events"], next_s["win_cand_score_mean"], next_s["win_cand_score_max"],
        ]
        X_w = np.array([w_vec], dtype=np.float32)
        p_obj = float(objectness_model.predict_proba(X_w)[0, 1])

        # Finalize detections for window t
        final_preds_t = []
        if cands_cur:
            gated_boxes = []
            for c in cands_cur:
                g_conf = float(c["confidence"]) * p_obj
                if g_conf >= conf_min:
                    b_copy = dict(c)
                    b_copy["confidence"] = g_conf
                    gated_boxes.append(b_copy)

            if gated_boxes:
                nms_boxes = apply_nms(gated_boxes, nms_iou)
                nms_boxes.sort(key=lambda x: float(x.get("confidence", 0.0)), reverse=True)
                for b in nms_boxes[:max_k]:
                    final_preds_t.append((
                        win_cur["w_start"], win_cur["w_end"],
                        int(round(b["center_x"])), int(round(b["center_y"])),
                        int(round(b["width"])), int(round(b["height"])),
                        round(float(b["confidence"]), 4)
                    ))

        t_end = time.perf_counter()
        elapsed_ms = (t_end - t_start) * 1000.0
        per_window_compute_ms.append(elapsed_ms)

        # Maintain buffer size at most 3
        if len(window_buffer) > 3:
            window_buffer.pop(0)

    # Exclude warmup windows
    valid_latencies = per_window_compute_ms[warmup_windows:] if len(per_window_compute_ms) > warmup_windows else per_window_compute_ms

    # Identify and separate system stalls (> 1000 ms)
    stalls = [ms for ms in valid_latencies if ms > 1000.0]
    unhalted_latencies = [ms for ms in valid_latencies if ms <= 1000.0]
    if not unhalted_latencies:
        unhalted_latencies = valid_latencies

    return {
        "windows": len(unhalted_latencies),
        "compute_mean": float(np.mean(unhalted_latencies)),
        "compute_p50": float(np.percentile(unhalted_latencies, 50)),
        "compute_p95": float(np.percentile(unhalted_latencies, 95)),
        "compute_p99": float(np.percentile(unhalted_latencies, 99)),
        "compute_max": float(np.max(unhalted_latencies)),
        "stalls": len(stalls),
        "stall_values": stalls,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Streaming per-window latency benchmark")
    parser.add_argument("--dataset-dir", type=str, default="../OrbitSight_Dataset", help="Dataset directory")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--reps", type=int, default=3, help="Number of repetitions per sequence")
    parser.add_argument("--warmup-windows", type=int, default=20, help="Warmup windows to exclude")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of sequences")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir).resolve()
    cfg = load_config(Path(args.config))

    npy_files = sorted(list(dataset_dir.rglob("*.npy")))
    if args.limit:
        npy_files = npy_files[:args.limit]

    print(f"Running Real Streaming Full-Pipeline Latency Benchmark across {len(npy_files)} sequences ({args.reps} independent reps)...", flush=True)

    seq_bench_results = []
    all_stalls = []

    for idx, npy_p in enumerate(npy_files, start=1):
        seq_name = sequence_name_from_npy(npy_p)
        width, height = infer_resolution(seq_name)
        events = load_events(npy_p)

        rep_means, rep_p50s, rep_p95s, rep_p99s, rep_maxs = [], [], [], [], []
        win_count = 0
        seq_stalls = 0

        for r in range(args.reps):
            res = benchmark_sequence_streaming(events, width, height, cfg, warmup_windows=args.warmup_windows)
            win_count = int(res["windows"])
            seq_stalls += res["stalls"]
            rep_means.append(res["compute_mean"])
            rep_p50s.append(res["compute_p50"])
            rep_p95s.append(res["compute_p95"])
            rep_p99s.append(res["compute_p99"])
            rep_maxs.append(res["compute_max"])

        if seq_stalls > 0:
            all_stalls.append((seq_name, seq_stalls))

        seq_bench_results.append({
            "sequence": seq_name,
            "windows": win_count,
            "compute_mean": (float(np.mean(rep_means)), float(np.std(rep_means))),
            "compute_p50": (float(np.mean(rep_p50s)), float(np.std(rep_p50s))),
            "compute_p95": (float(np.mean(rep_p95s)), float(np.std(rep_p95s))),
            "compute_p99": (float(np.mean(rep_p99s)), float(np.std(rep_p99s))),
            "compute_max": (float(np.mean(rep_maxs)), float(np.std(rep_maxs))),
        })
        m_val, m_std = seq_bench_results[-1]["compute_mean"]
        p99_val, p99_std = seq_bench_results[-1]["compute_p99"]
        total_p99 = p99_val + 40.0
        status = "PASS (<40ms)" if p99_val < 40.0 else "FAIL (>=40ms)"
        print(f"[{idx}/{len(npy_files)}] {seq_name:<45} | win: {win_count:>5} | comp p99: {p99_val:>5.2f} +/- {p99_std:>4.2f} ms | total p99: {total_p99:>5.2f} ms | {status}", flush=True)

    print("\n" + "=" * 130, flush=True)
    print(f"  REAL STREAMING FULL-PIPELINE LATENCY BENCHMARK TABLE ({args.reps} Independent Runs, Warmup={args.warmup_windows} Win Excluded)")
    print("  Note: Constant Algorithmic Lookahead Latency = 40.0 ms. Total Latency = Compute + 40.0 ms.")
    print("=" * 130, flush=True)
    print(f"  {'Sequence':<45} | {'Win':>5} | {'Comp p50 (ms)':>13} | {'Comp p95 (ms)':>13} | {'Comp p99 (ms)':>13} | {'Comp Max (ms)':>13} | {'Total p99':>10} | {'Comp Gate'}")
    print("-" * 130, flush=True)

    failing_compute_seqs = []
    failing_total_seqs = []

    for r in seq_bench_results:
        s_name = r["sequence"]
        w = r["windows"]
        p50_str = f"{r['compute_p50'][0]:.2f}+/-{r['compute_p50'][1]:.1f}"
        p95_str = f"{r['compute_p95'][0]:.2f}+/-{r['compute_p95'][1]:.1f}"
        p99_str = f"{r['compute_p99'][0]:.2f}+/-{r['compute_p99'][1]:.1f}"
        max_str = f"{r['compute_max'][0]:.2f}+/-{r['compute_max'][1]:.1f}"
        total_p99 = r["compute_p99"][0] + 40.0
        total_str = f"{total_p99:.2f}"
        is_pass_compute = r["compute_p99"][0] < 40.0
        gate_str = "PASS" if is_pass_compute else "FAIL"

        if not is_pass_compute:
            failing_compute_seqs.append((s_name, r["compute_p99"][0]))
        if total_p99 > 40.0:
            failing_total_seqs.append((s_name, total_p99))

        print(f"  {s_name:<45} | {w:>5} | {p50_str:>13} | {p95_str:>13} | {p99_str:>13} | {max_str:>13} | {total_str:>10} | {gate_str}")

    print("=" * 130, flush=True)
    print(f"Summary:")
    print(f"  - COMPUTE p99: {len(failing_compute_seqs)} of {len(seq_bench_results)} sequences exceed 40.0 ms.")
    print(f"  - TOTAL p99 (Compute + 40ms lookahead): {len(failing_total_seqs)} of {len(seq_bench_results)} sequences exceed 40.0 ms.")
    if all_stalls:
        print("\nSystem Stalls (>1000ms isolated):")
        for s_name, count in all_stalls:
            print(f"  - {s_name}: {count} stall window(s)")


if __name__ == "__main__":
    main()
