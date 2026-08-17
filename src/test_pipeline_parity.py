"""Non-circular regression parity test asserting infer.py file outputs match scoreboard.py evaluation."""

from pathlib import Path
import shutil
import sys
import tempfile
from typing import Dict, List

from src.common import sequence_name_from_npy
from src.infer import load_config, process_sequence
from src.scoreboard import evaluate_dataset_sequences


def run_non_circular_parity_test() -> bool:
    """Run non-circular parity test across DAVIS and DVX sequences with max_windows cap."""
    dataset_dir = Path("../OrbitSight_Dataset").resolve()
    cfg_path = Path("config.yaml").resolve()
    cfg = load_config(cfg_path)

    davis_files = sorted(list(dataset_dir.rglob("DAVIS*_labeled_events.npy")))
    dvx_files = sorted(list(dataset_dir.rglob("DVX*_labeled_events.npy")))

    if not davis_files or not dvx_files:
        print("[ERROR] Dataset sequences not found for parity test.", file=sys.stderr)
        return False

    target_davis = davis_files[0]
    target_dvx = dvx_files[0]
    seq_davis = sequence_name_from_npy(target_davis)
    seq_dvx = sequence_name_from_npy(target_dvx)
    target_seqs = [seq_davis, seq_dvx]

    temp_dir = Path(tempfile.mkdtemp(prefix="parity_noncircular_"))

    try:
        # Step A: Run infer.py with max_windows=200 to write real _pred.txt files to temp_dir
        print(f"[INFO] Step A: Generating prediction files for {target_seqs} (cap: 200 windows)...")
        process_sequence(target_davis, temp_dir, cfg, max_windows=200)
        process_sequence(target_dvx, temp_dir, cfg, max_windows=200)

        # Step B: Score written files via scoreboard evaluate_dataset_sequences (recompute=False)
        print("[INFO] Step B: Scoring written prediction files via scoreboard.py...")
        file_results = evaluate_dataset_sequences(
            dataset_dir,
            temp_dir,
            cfg,
            split_filter="all",
            recompute=False,
            sequences=target_seqs,
        )

        # Step C: Score in-process via scoreboard evaluate_dataset_sequences (recompute=True, max_windows=200)
        print("[INFO] Step C: Scoring in-process via scoreboard.py (--recompute)...")
        recomp_results = evaluate_dataset_sequences(
            dataset_dir,
            temp_dir,
            cfg,
            split_filter="all",
            recompute=True,
            sequences=target_seqs,
            max_windows=200,
        )

        file_map: Dict[str, Dict] = {r["sequence"]: r for r in file_results}
        recomp_map: Dict[str, Dict] = {r["sequence"]: r for r in recomp_results}

        for seq_name in target_seqs:
            if seq_name not in file_map or seq_name not in recomp_map:
                print(f"[FAIL] Missing results for sequence '{seq_name}'", file=sys.stderr)
                return False

            rf = file_map[seq_name]
            rc = recomp_map[seq_name]

            # Assert metric parity
            for metric in ["tp", "fp", "fn", "gt_count", "pred_count"]:
                if rf[metric] != rc[metric]:
                    print(
                        f"[FAIL] Parity mismatch on '{metric}' for {seq_name}: file_mode={rf[metric]} vs recompute_mode={rc[metric]}",
                        file=sys.stderr,
                    )
                    return False

            for f_metric in ["precision", "recall", "f1"]:
                if abs(rf[f_metric] - rc[f_metric]) > 1e-5:
                    print(
                        f"[FAIL] Parity mismatch on float '{f_metric}' for {seq_name}: file_mode={rf[f_metric]:.6f} vs recompute_mode={rc[f_metric]:.6f}",
                        file=sys.stderr,
                    )
                    return False

            # AP parity (handling nan)
            if (np.isnan(rf["ap"]) and not np.isnan(rc["ap"])) or (not np.isnan(rf["ap"]) and np.isnan(rc["ap"])):
                print(f"[FAIL] AP NaN mismatch for {seq_name}", file=sys.stderr)
                return False
            if not np.isnan(rf["ap"]) and abs(rf["ap"] - rc["ap"]) > 1e-5:
                print(f"[FAIL] AP float mismatch for {seq_name}: {rf['ap']} vs {rc['ap']}", file=sys.stderr)
                return False

            print(f"[PASS] Sequence '{seq_name}': File-reading mode matches in-process recompute mode exactly (TP={rf['tp']}, FP={rf['fp']}, FN={rf['fn']}, F1={rf['f1']:.4f}, AP={rf['ap']:.4f}).")

        return True

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def main() -> None:
    """CLI entrypoint for non-circular pipeline parity verification."""
    print("==================================================")
    print("  NON-CIRCULAR PIPELINE PARITY REGRESSION TEST")
    print("==================================================")

    import numpy as np  # ensure np is imported in main scope
    success = run_non_circular_parity_test()
    if success:
        print("\n[PASS] PIPELINE PARITY TEST PASSED: Written submission artifacts score identically to in-process pipeline.\n")
        sys.exit(0)
    else:
        print("\n[FAIL] PIPELINE PARITY TEST FAILED.\n", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    import numpy as np
    main()
