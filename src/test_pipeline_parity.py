"""Non-circular regression parity test asserting infer.py file outputs match scoreboard.py evaluation."""

from pathlib import Path
import shutil
import sys
import tempfile
from typing import Dict, List

from src.infer import load_config, process_sequence
from src.scoreboard import evaluate_dataset_sequences


def run_non_circular_parity_test() -> bool:
    """Run non-circular parity test across DAVIS and DVX sequences."""
    dataset_dir = Path("../OrbitSight_Dataset").resolve()
    cfg_path = Path("config.yaml").resolve()
    cfg = load_config(cfg_path)

    davis_files = sorted(list(dataset_dir.rglob("DAVIS*_labeled_events.npy")))
    dvx_files = sorted(list(dataset_dir.rglob("DVX*_labeled_events.npy")))

    if not davis_files or not dvx_files:
        print("[ERROR] Dataset sequences not found for parity test.", file=sys.stderr)
        return False

    temp_dir = Path(tempfile.mkdtemp(prefix="parity_noncircular_"))

    try:
        # Step A: Run infer.py to write real _pred.txt files to temp_dir
        print("[INFO] Step A: Generating prediction files via infer.process_sequence...")
        process_sequence(davis_files[0], temp_dir, cfg)
        process_sequence(dvx_files[0], temp_dir, cfg)

        # Step B: Score written files via scoreboard evaluate_dataset_sequences (recompute=False)
        print("[INFO] Step B: Scoring written prediction files via scoreboard.py...")
        file_results = evaluate_dataset_sequences(
            dataset_dir, temp_dir, cfg, split_filter="all", recompute=False
        )

        # Step C: Score in-process via scoreboard evaluate_dataset_sequences (recompute=True)
        print("[INFO] Step C: Scoring in-process via scoreboard.py (--recompute)...")
        recomp_results = evaluate_dataset_sequences(
            dataset_dir, temp_dir, cfg, split_filter="all", recompute=True
        )

        file_map: Dict[str, Dict] = {r["sequence"]: r for r in file_results}
        recomp_map: Dict[str, Dict] = {r["sequence"]: r for r in recomp_results}

        for target_npy in [davis_files[0], dvx_files[0]]:
            seq_name = target_npy.name.replace("_labeled_events.npy", "")
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

            for f_metric in ["precision", "recall", "f1", "ap"]:
                if abs(rf[f_metric] - rc[f_metric]) > 1e-5:
                    print(
                        f"[FAIL] Parity mismatch on float '{f_metric}' for {seq_name}: file_mode={rf[f_metric]:.6f} vs recompute_mode={rc[f_metric]:.6f}",
                        file=sys.stderr,
                    )
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

    success = run_non_circular_parity_test()
    if success:
        print("\n[PASS] PIPELINE PARITY TEST PASSED: Written submission artifacts score identically to in-process pipeline.\n")
        sys.exit(0)
    else:
        print("\n[FAIL] PIPELINE PARITY TEST FAILED.\n", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
