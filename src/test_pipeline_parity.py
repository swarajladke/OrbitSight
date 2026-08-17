"""Regression test asserting byte-identical parity between infer.py and scoreboard.py pipelines."""

from pathlib import Path
import shutil
import sys
import tempfile
from typing import List, Tuple

from src.common import infer_resolution, load_events, sequence_name_from_npy
from src.infer import load_config, process_sequence
from src.pipeline import run_sequence


def test_parity_for_sequence(
    npy_path: Path, cfg_path: Path, temp_dir: Path
) -> bool:
    """Assert infer.py output file matches scoreboard.py run_sequence output identically."""
    seq_name = sequence_name_from_npy(npy_path)
    cfg = load_config(cfg_path)

    # 1. Output from infer.py process_sequence
    process_sequence(npy_path, temp_dir, cfg, max_windows=50)
    pred_file = temp_dir / f"{seq_name}_pred.txt"

    with open(pred_file, "r", encoding="utf-8") as f:
        infer_lines = [line.strip() for line in f.readlines() if line.strip()]

    # 2. Output from scoreboard.py / pipeline.py run_sequence
    events = load_events(npy_path)
    width, height = infer_resolution(seq_name, events[:, 0], events[:, 1])
    pipe_preds = run_sequence(events, width, height, cfg, max_windows=50)

    pipe_lines = [
        "window_start_timestamp_us\twindow_end_timestamp_us\tcenter_x\tcenter_y\twidth\theight\tconfidence"
    ]
    for ws, we, cx, cy, bw, bh, conf in pipe_preds:
        pipe_lines.append(f"{ws}\t{we}\t{cx}\t{cy}\t{bw}\t{bh}\t{conf:.4f}")

    if len(infer_lines) != len(pipe_lines):
        print(
            f"[FAIL] Line count mismatch for {seq_name}: infer.py={len(infer_lines)} vs pipeline={len(pipe_lines)}",
            file=sys.stderr,
        )
        return False

    for idx, (l_inf, l_pipe) in enumerate(zip(infer_lines, pipe_lines)):
        if l_inf != l_pipe:
            print(
                f"[FAIL] Discrepancy at row {idx} for {seq_name}:\n  infer:    '{l_inf}'\n  pipeline: '{l_pipe}'",
                file=sys.stderr,
            )
            return False

    print(f"[PASS] Parity verified for sequence '{seq_name}': {len(pipe_lines)} lines byte-identical.")
    return True


def main() -> None:
    """Run parity verification across DAVIS and DVX sequences."""
    dataset_dir = Path("../OrbitSight_Dataset").resolve()
    cfg_path = Path("config.yaml").resolve()

    davis_files = sorted(list(dataset_dir.rglob("DAVIS*_labeled_events.npy")))
    dvx_files = sorted(list(dataset_dir.rglob("DVX*_labeled_events.npy")))

    if not davis_files or not dvx_files:
        print("[ERROR] Dataset sequences not found for parity test.", file=sys.stderr)
        sys.exit(1)

    temp_dir = Path(tempfile.mkdtemp(prefix="parity_test_"))

    try:
        pass_davis = test_parity_for_sequence(davis_files[0], cfg_path, temp_dir)
        pass_dvx = test_parity_for_sequence(dvx_files[0], cfg_path, temp_dir)

        if pass_davis and pass_dvx:
            print("\n[PASS] PIPELINE PARITY TEST PASSED: infer.py and scoreboard.py are unified.\n")
            sys.exit(0)
        else:
            print("\n[FAIL] PIPELINE PARITY TEST FAILED.\n", file=sys.stderr)
            sys.exit(1)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
