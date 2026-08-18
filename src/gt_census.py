"""Ground-truth census across all 21 dataset sequences without pipeline dependencies."""

import argparse
import csv
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple
import numpy as np


def run_gt_census(dataset_dir: Path) -> None:
    """Read every ground-truth bounding box file and produce exhaustive census."""
    gt_files = sorted(list(dataset_dir.rglob("*_bb_windows_40ms.txt")))

    if not gt_files:
        print(f"Error: No GT files found in {dataset_dir}", file=sys.stderr)
        sys.exit(1)

    print("sequence_name\tsplit\tsensor\tn_gt_rows\tn_distinct_windows\tfirst_window_start_us\tlast_window_end_us\tmedian_width\tmedian_height")

    summary_counts: Dict[Tuple[str, str], int] = {}
    zero_gt_sequences: List[Tuple[str, str, str]] = []

    for gt_f in gt_files:
        seq_name = gt_f.name.replace("_bb_windows_40ms.txt", "")
        split = "train" if "Training" in str(gt_f) else "test"

        if "EVK4" in seq_name.upper():
            sensor = "EVK4"
        elif "DVX" in seq_name.upper():
            sensor = "DVX"
        else:
            sensor = "DAVIS"

        windows_seen = set()
        widths: List[float] = []
        heights: List[float] = []
        first_start = "N/A"
        last_end = "N/A"

        with open(gt_f, "r", encoding="utf-8") as f:
            rdr = csv.DictReader(f, delimiter="\t")
            for r in rdr:
                ws = int(r["window_start_timestamp_us"])
                we = int(r["window_end_timestamp_us"])
                w = float(r["width"])
                h = float(r["height"])

                windows_seen.add((ws, we))
                widths.append(w)
                heights.append(h)

                if first_start == "N/A" or ws < first_start:
                    first_start = ws
                if last_end == "N/A" or we > last_end:
                    last_end = we

        n_rows = len(widths)
        n_dist_win = len(windows_seen)
        med_w = f"{np.median(widths):.1f}" if widths else "N/A"
        med_h = f"{np.median(heights):.1f}" if heights else "N/A"

        print(f"{seq_name}\t{split}\t{sensor}\t{n_rows}\t{n_dist_win}\t{first_start}\t{last_end}\t{med_w}\t{med_h}")

        summary_counts[(sensor, split)] = summary_counts.get((sensor, split), 0) + n_rows

        if n_rows == 0:
            zero_gt_sequences.append((seq_name, split, sensor))

    print("\n" + "=" * 50)
    print("  SUMMARY: TOTAL GT ROWS PER SENSOR PER SPLIT")
    print("=" * 50)
    for (sensor, split), count in sorted(summary_counts.items()):
        print(f"  {sensor} ({split}): {count} GT rows")

    print("\n" + "=" * 50)
    print("  ZERO-GT SEQUENCES (n_gt_rows == 0)")
    print("=" * 50)
    if zero_gt_sequences:
        for s_name, s_split, s_sensor in zero_gt_sequences:
            print(f"  - {s_name} [{s_split}, {s_sensor}]")
    else:
        print("  NONE")


def main() -> None:
    """CLI entrypoint for GT census."""
    parser = argparse.ArgumentParser(description="OrbitSight Ground Truth Census")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="../OrbitSight_Dataset",
        help="Path to dataset root",
    )
    args = parser.parse_args()
    dataset_dir = Path(args.dataset_dir).resolve()
    run_gt_census(dataset_dir)


if __name__ == "__main__":
    main()
