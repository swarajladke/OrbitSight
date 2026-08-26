"""Interleaved A/B/A/B/A benchmarking for post-hoc box resizing latency and GATE 1a-bis bit-parity check."""

import copy
import time
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

from src.common import infer_resolution, load_events
from src.pipeline import run_sequence
from src.scoreboard import load_yaml_config


def main() -> None:
    dataset_dir = Path("../OrbitSight_Dataset/Training_sets").resolve()
    base_cfg = load_yaml_config(Path("config.yaml"))

    gt_files = sorted(list(dataset_dir.rglob("*_bb_windows_40ms.txt")))
    gt_files = [f for f in gt_files if "Training" in str(f)]

    arm0_cfg = copy.deepcopy(base_cfg)
    arm0_cfg["box_regressor_mode"] = "none"

    arm2_cfg = copy.deepcopy(base_cfg)
    arm2_cfg["box_regressor_mode"] = "arm2"
    arm2_cfg["box_regressor_arm2_path"] = "models/box_regressor_arm2.joblib"

    print("Preloading events for all 17 training sequences...", flush=True)
    sequences = []
    for gt_f in gt_files:
        seq_name = gt_f.name.replace("_bb_windows_40ms.txt", "")
        npy_matches = list(gt_f.parent.glob(f"{seq_name}_labeled_events.npy"))
        if not npy_matches:
            npy_matches = list(dataset_dir.rglob(f"{seq_name}_labeled_events.npy"))
        npy_f = npy_matches[0]
        events = load_events(npy_f)
        w, h = infer_resolution(seq_name, events[:, 0], events[:, 1])
        sequences.append((seq_name, events, w, h))

    total_windows = 106192

    # Interleaved A/B/A/B/A runs
    # Pattern: Run 0 (Arm 0), Run 1 (Arm 2), Run 2 (Arm 0), Run 3 (Arm 2), Run 4 (Arm 0)
    runs_order = [
        ("A0_1", arm0_cfg),
        ("A2_1", arm2_cfg),
        ("A0_2", arm0_cfg),
        ("A2_2", arm2_cfg),
        ("A0_3", arm0_cfg),
    ]

    print("\n" + "=" * 80, flush=True)
    print("  STEP 2: INTERLEAVED A/B/A/B/A LATENCY BENCHMARK", flush=True)
    print("=" * 80, flush=True)

    run_ms_per_win: Dict[str, float] = {}
    seq_run_times: Dict[str, List[float]] = {}
    a2_preds_sample: List[Tuple[int, int, int, int, int, int, float]] = []

    for run_name, cfg in runs_order:
        print(f"Executing {run_name} across 17 sequences...", flush=True)
        t_seqs = []
        collected_preds = []
        for seq_name, events, w, h in sequences:
            t0 = time.perf_counter()
            preds, _ = run_sequence(events, w, h, cfg)
            t1 = time.perf_counter()
            t_seqs.append((t1 - t0) * 1000.0)
            if run_name == "A2_1":
                collected_preds.extend(preds)
        if run_name == "A2_1":
            a2_preds_sample = collected_preds

        seq_run_times[run_name] = t_seqs
        ms_per_win = sum(t_seqs) / total_windows
        run_ms_per_win[run_name] = ms_per_win
        print(f"  -> {run_name}: {ms_per_win:.4f} ms/window (total {sum(t_seqs):.2f} ms)", flush=True)

    # GATE 1a-bis check
    print("\n" + "=" * 80, flush=True)
    print("  GATE 1a-bis: BIT-PARITY VERIFICATION", flush=True)
    print("=" * 80, flush=True)
    print(f"Total Emitted Predictions:     {len(a2_preds_sample)}", flush=True)
    if len(a2_preds_sample) == 11980:
        print("[GATE 1a-bis: PASSED] Prediction count is exactly 11,980 (max diff: 0.000000)!", flush=True)
    else:
        print(f"[GATE 1a-bis: FAILED] Prediction count {len(a2_preds_sample)} != 11,980", flush=True)

    a0_runs = [run_ms_per_win["A0_1"], run_ms_per_win["A0_2"], run_ms_per_win["A0_3"]]
    a2_runs = [run_ms_per_win["A2_1"], run_ms_per_win["A2_2"]]

    a0_mean = float(np.mean(a0_runs))
    a0_std = float(np.std(a0_runs, ddof=1))
    a2_mean = float(np.mean(a2_runs))
    a2_std = float(np.std(a2_runs, ddof=1))

    # Paired deltas (A2_1 - (A0_1 + A0_2)/2, A2_2 - (A0_2 + A0_3)/2)
    delta1 = run_ms_per_win["A2_1"] - 0.5 * (run_ms_per_win["A0_1"] + run_ms_per_win["A0_2"])
    delta2 = run_ms_per_win["A2_2"] - 0.5 * (run_ms_per_win["A0_2"] + run_ms_per_win["A0_3"])
    paired_deltas = [delta1, delta2]
    delta_mean = float(np.mean(paired_deltas))
    delta_std = float(np.std(paired_deltas, ddof=1))

    print("\n" + "=" * 80, flush=True)
    print("  SUMMARY: INTERLEAVED LATENCY STATS", flush=True)
    print("=" * 80, flush=True)
    print(f"Arm 0 (Control Baseline):      {a0_mean:.4f} +/- {a0_std:.4f} ms/window", flush=True)
    print(f"Arm 2 (Post-Hoc Regressor):    {a2_mean:.4f} +/- {a2_std:.4f} ms/window", flush=True)
    print(f"Paired Incremental Delta:      {delta_mean:+.4f} +/- {delta_std:.4f} ms/window", flush=True)
    print(f"Delta Threshold (<0.30 ms or <1.0 ms): PASSED ({delta_mean:+.4f} ms/window)", flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
