"""Validation CLI for prediction output text files."""

import argparse
from pathlib import Path
import sys
from typing import List, Tuple

from src.common import infer_resolution


def validate_file(pred_file: Path) -> Tuple[bool, int, List[str]]:
    """Validate a single prediction file against structural and domain constraints.

    Args:
        pred_file: Path to the prediction file (*_pred.txt).

    Returns:
        Tuple of (is_valid boolean, row count integer, list of error message strings).
    """
    errors: List[str] = []
    seq_name = pred_file.name.replace("_pred.txt", "")

    try:
        width, height = infer_resolution(seq_name)
    except ValueError as e:
        errors.append(f"Sensor resolution error: {e}")
        return False, 0, errors

    with open(pred_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    if not lines:
        errors.append("File is empty.")
        return False, 0, errors

    expected_header = (
        "window_start_timestamp_us\twindow_end_timestamp_us\t"
        "center_x\tcenter_y\twidth\theight\tconfidence"
    )

    if lines[0] != expected_header:
        errors.append(
            f"Header mismatch. Expected '{expected_header}', got '{lines[0]}'"
        )

    data_lines = lines[1:]
    row_count = len(data_lines)

    for line_idx, line in enumerate(data_lines, start=2):
        fields = line.split("\t")
        if len(fields) != 7:
            errors.append(
                f"Line {line_idx}: expected 7 tab-separated fields, found {len(fields)}"
            )
            continue

        try:
            start_us = int(fields[0])
            end_us = int(fields[1])
            cx = int(fields[2])
            cy = int(fields[3])
            _w = int(fields[4])
            _h = int(fields[5])
        except ValueError:
            errors.append(f"Line {line_idx}: non-integer values in first 6 fields.")
            continue

        try:
            conf = float(fields[6])
        except ValueError:
            errors.append(f"Line {line_idx}: confidence '{fields[6]}' is not float.")
            continue

        if not (0.0 <= conf <= 1.0):
            errors.append(
                f"Line {line_idx}: confidence {conf} out of bounds [0.0, 1.0]."
            )

        if end_us - start_us != 40_000:
            errors.append(
                f"Line {line_idx}: window duration {end_us - start_us} us != 40000 us."
            )

        if not (0 <= cx < width):
            errors.append(
                f"Line {line_idx}: center_x {cx} out of sensor width bounds [0, {width})."
            )

        if not (0 <= cy < height):
            errors.append(
                f"Line {line_idx}: center_y {cy} out of sensor height bounds [0, {height})."
            )

    is_valid = len(errors) == 0
    return is_valid, row_count, errors


def main() -> None:
    """CLI entrypoint for prediction validator."""
    parser = argparse.ArgumentParser(
        description="Validate OrbitSight prediction files"
    )
    parser.add_argument(
        "--pred-dir",
        type=str,
        default="predictions",
        help="Directory containing prediction files",
    )

    args = parser.parse_args()
    pred_dir = Path(args.pred_dir)

    if not pred_dir.exists():
        print(f"Error: Prediction directory '{pred_dir}' does not exist.", file=sys.stderr)
        sys.exit(1)

    pred_files = sorted(list(pred_dir.glob("*_pred.txt")))
    if not pred_files:
        print(f"Warning: No '*_pred.txt' files found in '{pred_dir}'.")
        sys.exit(0)

    any_failed = False
    print(f"Validating {len(pred_files)} prediction file(s) in '{pred_dir}'...\n")

    for pred_file in pred_files:
        is_valid, row_count, errors = validate_file(pred_file)
        status = "PASS" if is_valid else "FAIL"

        if not is_valid:
            any_failed = True

        print(f"[{status}] {pred_file.name} ({row_count} detections)")
        for err in errors:
            print(f"       -> Error: {err}")

    print("\n--------------------------------------------------")
    if any_failed:
        print("Validation Result: FAILED (issues detected in one or more files).")
        sys.exit(1)
    else:
        print("Validation Result: PASSED (all files adhere to specifications).")
        sys.exit(0)


if __name__ == "__main__":
    main()
