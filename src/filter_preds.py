"""Post-hoc confidence / Top-K filter for existing prediction files."""

import argparse
import csv
from pathlib import Path
from typing import Dict, List

HEADER = [
    "window_start_timestamp_us",
    "window_end_timestamp_us",
    "center_x",
    "center_y",
    "width",
    "height",
    "confidence",
]


def filter_pred_file(src_path: Path, dst_path: Path, conf_min: float, top_k: int) -> int:
    """Filter one *_pred.txt by confidence and keep at most top_k rows per window."""
    by_window: Dict[int, List[List[str]]] = {}
    with open(src_path, "r", encoding="utf-8") as f:
        rdr = csv.DictReader(f, delimiter="\t")
        for r in rdr:
            if float(r["confidence"]) < conf_min:
                continue
            ws = int(r["window_start_timestamp_us"])
            by_window.setdefault(ws, []).append([r[h] for h in HEADER])

    kept = 0
    with open(dst_path, "w", encoding="utf-8", newline="") as f:
        f.write("\t".join(HEADER) + "\n")
        for ws in sorted(by_window.keys()):
            rows = sorted(by_window[ws], key=lambda r: -float(r[6]))
            if top_k > 0:
                rows = rows[:top_k]
            for r in rows:
                f.write("\t".join(r) + "\n")
                kept += 1
    return kept


def main() -> None:
    p = argparse.ArgumentParser(description="Post-hoc conf_min / Top-K filter")
    p.add_argument("--in-dir", type=str, required=True)
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--conf-min", type=float, required=True)
    p.add_argument("--top-k", type=int, required=True)
    args = p.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(in_dir.glob("*_pred.txt"))
    if not files:
        raise RuntimeError(f"No *_pred.txt files found in {in_dir}")

    total = 0
    for src in files:
        total += filter_pred_file(src, out_dir / src.name, args.conf_min, args.top_k)

    print(
        f"conf_min={args.conf_min} top_k={args.top_k}: "
        f"{len(files)} files, {total} predictions kept -> {out_dir}"
    )


if __name__ == "__main__":
    main()
