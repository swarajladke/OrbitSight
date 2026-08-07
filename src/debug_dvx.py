"""Instrumented DVX detector debugging tool to diagnose zero prediction cause."""

import argparse
import csv
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple
import cv2
import numpy as np
from tabulate import tabulate

from src.common import (
    WINDOW_US,
    event_image,
    infer_resolution,
    iter_windows,
    load_events,
)


def load_yaml_config(path: Path) -> Dict[str, Any]:
    """Load configuration file with zero-dependency fallback."""
    if not path.exists():
        return {}
    try:
        import yaml

        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except ImportError:
        cfg: Dict[str, Any] = {}
        curr_sec = ""
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                raw = line.split("#")[0].rstrip("\r\n")
                if not raw.strip():
                    continue
                indent = len(raw) - len(raw.lstrip(" "))
                stripped = raw.strip()
                if ":" not in stripped:
                    continue
                parts = stripped.split(":", 1)
                k = parts[0].strip()
                v = parts[1].strip()
                if indent == 0:
                    if not v:
                        curr_sec = k
                        cfg[k] = {}
                    else:
                        curr_sec = ""
                        try:
                            cfg[k] = float(v) if "." in v else int(v)
                        except ValueError:
                            cfg[k] = (
                                v.strip("'\"")
                                if v.lower() not in ("true", "false")
                                else v.lower() == "true"
                            )
                elif indent > 0 and curr_sec:
                    if not isinstance(cfg.get(curr_sec), dict):
                        cfg[curr_sec] = {}
                    try:
                        cfg[curr_sec][k] = float(v) if "." in v else int(v)
                    except ValueError:
                        cfg[curr_sec][k] = (
                            v.strip("'\"")
                            if v.lower() not in ("true", "false")
                            else v.lower() == "true"
                        )
        return cfg


def debug_sequence(
    npy_path: Path, gt_path: Path, cfg: Dict[str, Any], strict: bool = False
) -> Dict[str, Any]:
    """Instrument a single DVX sequence for stage-by-stage component diagnosis."""
    seq_name = npy_path.name.replace("_labeled_events.npy", "")
    events = load_events(npy_path)
    width, height = infer_resolution(seq_name, events[:, 0], events[:, 1])
    img_area = float(width * height)

    sensor_cfg = cfg.get("DVX", {}) if isinstance(cfg.get("DVX"), dict) else {}
    percentile = float(sensor_cfg.get("percentile", cfg.get("percentile", 97.5)))
    min_events = float(sensor_cfg.get("min_events_in_box", cfg.get("min_events_in_box", 6)))
    open_k = int(sensor_cfg.get("open_kernel", cfg.get("open_kernel", 2)))
    dilate_k = int(sensor_cfg.get("dilate_kernel", cfg.get("dilate_kernel", 3)))
    max_area_frac = float(sensor_cfg.get("max_area_frac", cfg.get("max_area_frac", 0.02)))
    max_area_pixels = max_area_frac * img_area

    window_event_counts: List[int] = []
    window_nonzero_counts: List[int] = []
    thresh_values: List[float] = []
    surviving_pixel_counts: List[int] = []

    comp_count_raw: List[int] = []
    comp_count_post_open: List[int] = []
    comp_count_post_dilate: List[int] = []

    areas_raw: List[float] = []
    areas_post_open: List[float] = []
    areas_post_dilate: List[float] = []

    rejection_tallies = {
        "AREA_TOO_LARGE": 0,
        "AREA_TOO_SMALL": 0,
        "BELOW_MIN_EVENTS": 0,
        "ACCEPTED": 0,
    }
    large_area_fracs: List[float] = []

    for w_start, w_end, w_events in iter_windows(events, window_us=WINDOW_US):
        try:
            n_ev = len(w_events)
            window_event_counts.append(n_ev)

            count_img, _, _ = event_image(w_events, width, height)
            nonzero_vals = count_img[count_img > 0]
            num_nonzero = len(nonzero_vals)
            window_nonzero_counts.append(num_nonzero)

            if num_nonzero < 4:
                thresh_values.append(0.0)
                surviving_pixel_counts.append(0)
                comp_count_raw.append(0)
                comp_count_post_open.append(0)
                comp_count_post_dilate.append(0)
                continue

            actual_perc = min(99.0, percentile + (num_nonzero - 1000) / 500.0) if num_nonzero > 1000 else percentile
            raw_thresh = float(np.percentile(nonzero_vals, actual_perc))
            thresh = max(1.0, raw_thresh)
            thresh_values.append(thresh)

            binary_raw = (count_img >= thresh).astype(np.uint8)
            surv_pixels = int(cv2.countNonZero(binary_raw))
            surviving_pixel_counts.append(surv_pixels)

            # Raw CC
            n_raw, _, s_raw, _ = cv2.connectedComponentsWithStats(binary_raw, connectivity=8)
            comp_count_raw.append(n_raw - 1)
            for i in range(1, n_raw):
                areas_raw.append(float(s_raw[i, cv2.CC_STAT_AREA]))

            # Open CC
            b_open = binary_raw.copy()
            if open_k > 1:
                k_open = cv2.getStructuringElement(cv2.MORPH_RECT, (open_k, open_k))
                b_open = cv2.morphologyEx(b_open, cv2.MORPH_OPEN, k_open)

            n_open, _, s_open, _ = cv2.connectedComponentsWithStats(b_open, connectivity=8)
            comp_count_post_open.append(n_open - 1)
            for i in range(1, n_open):
                areas_post_open.append(float(s_open[i, cv2.CC_STAT_AREA]))

            # Dilate CC
            b_dilate = b_open.copy()
            if dilate_k > 1:
                k_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (dilate_k, dilate_k))
                b_dilate = cv2.dilate(b_dilate, k_dilate)

            n_dilate, labels_d, s_dilate, centroids_d = cv2.connectedComponentsWithStats(b_dilate, connectivity=8)
            comp_count_post_dilate.append(n_dilate - 1)

            for i in range(1, n_dilate):
                c_area = float(s_dilate[i, cv2.CC_STAT_AREA])
                areas_post_dilate.append(c_area)

                if c_area > max_area_pixels:
                    rejection_tallies["AREA_TOO_LARGE"] += 1
                    large_area_fracs.append(c_area / img_area)
                    continue

                if c_area < 1.0:
                    rejection_tallies["AREA_TOO_SMALL"] += 1
                    continue

                comp_mask = labels_d == i
                comp_events = float(count_img[comp_mask].sum())

                if comp_events < min_events:
                    rejection_tallies["BELOW_MIN_EVENTS"] += 1
                else:
                    rejection_tallies["ACCEPTED"] += 1

        except Exception as e:
            if strict:
                raise e
            else:
                print(f"[WARN] Exception during window processing: {e}", file=sys.stderr)

    return {
        "sequence": seq_name,
        "total_windows": len(window_event_counts),
        "events_per_win": {
            "mean": float(np.mean(window_event_counts)) if window_event_counts else 0.0,
            "median": float(np.median(window_event_counts)) if window_event_counts else 0.0,
        },
        "nonzero_pixels_per_win": {
            "mean": float(np.mean(window_nonzero_counts)) if window_nonzero_counts else 0.0,
            "median": float(np.median(window_nonzero_counts)) if window_nonzero_counts else 0.0,
        },
        "threshold_val": {
            "mean": float(np.mean(thresh_values)) if thresh_values else 0.0,
            "median": float(np.median(thresh_values)) if thresh_values else 0.0,
        },
        "surviving_pixels": {
            "mean": float(np.mean(surviving_pixel_counts)) if surviving_pixel_counts else 0.0,
            "median": float(np.median(surviving_pixel_counts)) if surviving_pixel_counts else 0.0,
        },
        "cc_counts": {
            "raw": float(np.mean(comp_count_raw)) if comp_count_raw else 0.0,
            "post_open": float(np.mean(comp_count_post_open)) if comp_count_post_open else 0.0,
            "post_dilate": float(np.mean(comp_count_post_dilate)) if comp_count_post_dilate else 0.0,
        },
        "area_dists": {
            "raw_mean": float(np.mean(areas_raw)) if areas_raw else 0.0,
            "open_mean": float(np.mean(areas_post_open)) if areas_post_open else 0.0,
            "dilate_mean": float(np.mean(areas_post_dilate)) if areas_post_dilate else 0.0,
        },
        "rejection_tallies": rejection_tallies,
        "large_area_fracs": {
            "mean": float(np.mean(large_area_fracs)) if large_area_fracs else 0.0,
            "median": float(np.median(large_area_fracs)) if large_area_fracs else 0.0,
            "p10": float(np.percentile(large_area_fracs, 10)) if large_area_fracs else 0.0,
            "p90": float(np.percentile(large_area_fracs, 90)) if large_area_fracs else 0.0,
            "max": float(np.max(large_area_fracs)) if large_area_fracs else 0.0,
        },
    }


def main() -> None:
    """Main CLI entrypoint for DVX debugging tool."""
    parser = argparse.ArgumentParser(
        description="OrbitSight DVX Zero Prediction Debugging Tool"
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="../OrbitSight_Dataset",
        help="Path to dataset root",
    )
    parser.add_argument(
        "--sequence",
        type=str,
        default=None,
        help="Specific DVX sequence name to instrument",
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to config file"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Re-raise exceptions rather than skipping",
    )

    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir).resolve()
    cfg = load_yaml_config(Path(args.config))

    gt_files = sorted(list(dataset_dir.rglob("*_bb_windows_40ms.txt")))
    dvx_files = [f for f in gt_files if "DVX" in f.name.upper()]

    if not dvx_files:
        print("Error: No DVX sequences found in dataset directory.", file=sys.stderr)
        sys.exit(1)

    target_gt = dvx_files[0]
    if args.sequence:
        matches = [f for f in dvx_files if args.sequence in f.name]
        if matches:
            target_gt = matches[0]

    seq_name = target_gt.name.replace("_bb_windows_40ms.txt", "")
    npy_matches = list(target_gt.parent.glob(f"{seq_name}_labeled_events.npy"))
    if not npy_matches:
        npy_matches = list(dataset_dir.rglob(f"{seq_name}_labeled_events.npy"))

    print(f"\n==================================================")
    print(f"  INSTRUMENTED DVX DEBUGGER — SEQUENCE: {seq_name}")
    print("==================================================")

    res = debug_sequence(npy_matches[0], target_gt, cfg, strict=args.strict)

    print("\nPER-WINDOW EVENT & THRESHOLD SUMMARY:")
    print(
        tabulate(
            [
                ["Total Events", res["events_per_win"]["mean"], res["events_per_win"]["median"]],
                ["Nonzero Pixels", res["nonzero_pixels_per_win"]["mean"], res["nonzero_pixels_per_win"]["median"]],
                ["Threshold Value", res["threshold_val"]["mean"], res["threshold_val"]["median"]],
                ["Surviving Pixels", res["surviving_pixels"]["mean"], res["surviving_pixels"]["median"]],
            ],
            headers=["Metric", "Mean per Window", "Median per Window"],
            tablefmt="github",
        )
    )

    print("\nCOMPONENT COUNTS & AREA BEFORE/AFTER MORPHOLOGY:")
    print(
        tabulate(
            [
                ["Raw Binary (S1)", res["cc_counts"]["raw"], res["area_dists"]["raw_mean"]],
                ["Post-Open (Kernel=2)", res["cc_counts"]["post_open"], res["area_dists"]["open_mean"]],
                ["Post-Dilate (Kernel=3)", res["cc_counts"]["post_dilate"], res["area_dists"]["dilate_mean"]],
            ],
            headers=["Stage", "Mean Component Count", "Mean Component Area (px)"],
            tablefmt="github",
        )
    )

    print("\nREJECTION TALLY BY REASON:")
    total_rejections = sum(res["rejection_tallies"].values())
    rej_rows = [
        [k, v, f"{(v / total_rejections * 100.0) if total_rejections > 0 else 0.0:.2f}%"]
        for k, v in res["rejection_tallies"].items()
    ]
    print(
        tabulate(
            rej_rows,
            headers=["Reason", "Count", "Percentage"],
            tablefmt="github",
        )
    )

    if res["large_area_fracs"]["max"] > 0:
        print("\nAREA_TOO_LARGE DISTRIBUTION (Fraction of Image Area):")
        print(
            tabulate(
                [
                    [
                        res["large_area_fracs"]["mean"],
                        res["large_area_fracs"]["median"],
                        res["large_area_fracs"]["p10"],
                        res["large_area_fracs"]["p90"],
                        res["large_area_fracs"]["max"],
                    ]
                ],
                headers=["Mean", "Median", "P10", "P90", "Max"],
                tablefmt="github",
            )
        )

    print("\n--------------------------------------------------")
    print("STAGE RESPONSIBLE FOR ZERO PREDICTIONS:")
    if res["rejection_tallies"]["AREA_TOO_LARGE"] > 0:
        print("  -> AREA_TOO_LARGE Filter (Max Area Exceeded)")
        print("  -> Evidence: Dilate kernel merges dense background event clusters into giant components > 2% max area.")
    elif res["rejection_tallies"]["BELOW_MIN_EVENTS"] > 0:
        print("  -> BELOW_MIN_EVENTS Filter")
    elif res["surviving_pixels"]["mean"] == 0:
        print("  -> Image Thresholding (Percentile set too high)")
    else:
        print("  -> Downstream Persistence Filter")
    print("--------------------------------------------------\n")


if __name__ == "__main__":
    main()
