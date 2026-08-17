"""Upper-bound recall attribution, DVX de-confounding, and per-stage diagnostic tool."""

import argparse
import csv
from copy import deepcopy
import math
from pathlib import Path
import sys
import time
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
    print_effective_config,
    resolve_effective_config,
    sequence_name_from_npy,
)
from src.detector import detect_boxes
from src.metrics import compute_ap, compute_prf1, evaluate_sequence, iou
from src.nms import apply_nms
from src.pipeline import run_sequence


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


def filter_gt_files(
    dataset_dir: Path, target_sensor: str, split: str = "all"
) -> List[Path]:
    """Discover and filter GT sequence files by sensor and split."""
    gt_files = sorted(list(dataset_dir.rglob("*_bb_windows_40ms.txt")))
    filtered: List[Path] = []
    for gt_f in gt_files:
        seq_name = gt_f.name.replace("_bb_windows_40ms.txt", "")
        if target_sensor.upper() != "ALL" and target_sensor.upper() not in seq_name.upper():
            continue
        if split == "train" and "Training" not in str(gt_f):
            continue
        if split == "test" and "Testing" not in str(gt_f):
            continue
        filtered.append(gt_f)
    return filtered


def run_upper_bound_recall_attribution(
    dataset_dir: Path, target_sensor: str, split: str, cfg: Dict[str, Any]
) -> Dict[str, Any]:
    """Run stage-by-stage upper-bound recall attribution and best-candidate IoU histogram."""
    sensor_files = filter_gt_files(dataset_dir, target_sensor, split)

    gt_total = 0
    cnt_reachable = 0
    cnt_ub_pre_nms = 0
    cnt_ub_post_nms = 0
    cnt_ub_post_topk = 0
    cnt_final_matched = 0

    iou_hist = {
        "0.0": 0,
        "0.0-0.25": 0,
        "0.25-0.5": 0,
        ">=0.5": 0,
    }

    for gt_f in sensor_files:
        seq_name = gt_f.name.replace("_bb_windows_40ms.txt", "")
        npy_matches = list(gt_f.parent.glob(f"{seq_name}_labeled_events.npy"))
        if not npy_matches:
            npy_matches = list(dataset_dir.rglob(f"{seq_name}_labeled_events.npy"))
        if not npy_matches:
            continue

        events = load_events(npy_matches[0])
        width, height = infer_resolution(seq_name, events[:, 0], events[:, 1])

        if width >= 1200:
            sensor_name = "EVK4"
        elif width >= 600:
            sensor_name = "DVX"
        else:
            sensor_name = "DAVIS"

        eff = resolve_effective_config(cfg, sensor_name)
        percentile = float(eff.get("percentile", 97.5))
        nms_iou_val = eff.get("nms_iou", 0.3)
        max_k_val = eff.get("max_candidates_per_window", None)

        gt_by_window: Dict[int, List[Tuple[float, float, float, float]]] = {}
        with open(gt_f, "r", encoding="utf-8") as f:
            rdr = csv.DictReader(f, delimiter="\t")
            for r in rdr:
                ws = int(r["window_start_timestamp_us"])
                gt_by_window.setdefault(ws, []).append(
                    (
                        float(r["center_x"]),
                        float(r["center_y"]),
                        float(r["width"]),
                        float(r["height"]),
                    )
                )

        # 1. Full pipeline run for final predictions
        final_preds = run_sequence(events, width, height, cfg, window_us=WINDOW_US)
        final_by_window: Dict[int, List[Tuple[float, float, float, float, float]]] = {}
        for ws, we, cx, cy, bw, bh, conf in final_preds:
            final_by_window.setdefault(ws, []).append((cx, cy, bw, bh, conf))

        # 2. Window-by-window stage analysis
        # Config without NMS or Top-K for pre-NMS detection extraction
        raw_det_cfg = deepcopy(cfg)
        if sensor_name not in raw_det_cfg:
            raw_det_cfg[sensor_name] = {}
        raw_det_cfg[sensor_name]["nms_iou"] = None
        raw_det_cfg[sensor_name]["conf_min"] = 0.0
        raw_det_cfg[sensor_name]["max_candidates_per_window"] = None

        for w_start, w_end, w_events in iter_windows(events, window_us=WINDOW_US):
            gt_boxes = gt_by_window.get(w_start, [])
            if not gt_boxes:
                continue

            count_img, _, _ = event_image(w_events, width, height)
            nonzero_vals = count_img[count_img > 0]
            num_nonzero = len(nonzero_vals)

            # Raw components (post-threshold, pre-morphology)
            raw_centroids: List[Tuple[float, float]] = []
            if num_nonzero >= 4:
                actual_perc = min(99.0, percentile + (num_nonzero - 1000) / 500.0) if num_nonzero > 1000 else percentile
                thresh = max(1.0, float(np.percentile(nonzero_vals, actual_perc)))
                b_raw = (count_img >= thresh).astype(np.uint8)
                n_raw, _, _, c_raw = cv2.connectedComponentsWithStats(b_raw, connectivity=8)
                for ci in range(1, n_raw):
                    raw_centroids.append((float(c_raw[ci, 0]), float(c_raw[ci, 1])))

            # Candidate boxes pre-NMS
            pre_nms_boxes = detect_boxes(count_img, width, height, raw_det_cfg)
            pre_nms_tuples = [(b["center_x"], b["center_y"], b["width"], b["height"]) for b in pre_nms_boxes]

            # Candidate boxes post-NMS
            post_nms_boxes = apply_nms(pre_nms_boxes, float(nms_iou_val) if nms_iou_val is not None else 0.3)
            post_nms_tuples = [(b["center_x"], b["center_y"], b["width"], b["height"]) for b in post_nms_boxes]

            # Candidate boxes post-TopK
            if max_k_val is not None:
                try:
                    k_int = int(max_k_val)
                    post_topk_boxes = sorted(post_nms_boxes, key=lambda b: float(b.get("confidence", 0.0)), reverse=True)[:k_int]
                except (TypeError, ValueError):
                    post_topk_boxes = post_nms_boxes
            else:
                post_topk_boxes = post_nms_boxes
            post_topk_tuples = [(b["center_x"], b["center_y"], b["width"], b["height"]) for b in post_topk_boxes]

            final_tuples = [(b[0], b[1], b[2], b[3]) for b in final_by_window.get(w_start, [])]

            for gt_box in gt_boxes:
                gt_total += 1
                gt_cx, gt_cy, gt_w, gt_h = gt_box

                # Reachable check (within 6px)
                is_reachable = any(math.hypot(c[0] - gt_cx, c[1] - gt_cy) <= 6.0 for c in raw_centroids)
                if is_reachable:
                    cnt_reachable += 1

                # Upper-bound recall checks
                m_pre_nms = any(iou(b, gt_box) >= 0.5 for b in pre_nms_tuples)
                if m_pre_nms:
                    cnt_ub_pre_nms += 1

                m_post_nms = any(iou(b, gt_box) >= 0.5 for b in post_nms_tuples)
                if m_post_nms:
                    cnt_ub_post_nms += 1

                m_post_topk = any(iou(b, gt_box) >= 0.5 for b in post_topk_tuples)
                if m_post_topk:
                    cnt_ub_post_topk += 1

                m_final = any(iou(b, gt_box) >= 0.5 for b in final_tuples)
                if m_final:
                    cnt_final_matched += 1
                else:
                    # Missed GT box - calculate IoU of best single candidate in window
                    best_cand_iou = 0.0
                    for b in pre_nms_tuples:
                        sc = iou(b, gt_box)
                        if sc > best_cand_iou:
                            best_cand_iou = sc

                    if best_cand_iou >= 0.5:
                        iou_hist[">=0.5"] += 1
                    elif best_cand_iou >= 0.25:
                        iou_hist["0.25-0.5"] += 1
                    elif best_cand_iou > 0.0:
                        iou_hist["0.0-0.25"] += 1
                    else:
                        iou_hist["0.0"] += 1

    rec_reachable = (cnt_reachable / gt_total) if gt_total > 0 else 0.0
    rec_pre_nms = (cnt_ub_pre_nms / gt_total) if gt_total > 0 else 0.0
    rec_post_nms = (cnt_ub_post_nms / gt_total) if gt_total > 0 else 0.0
    rec_post_topk = (cnt_ub_post_topk / gt_total) if gt_total > 0 else 0.0
    rec_final = (cnt_final_matched / gt_total) if gt_total > 0 else 0.0

    return {
        "sensor": target_sensor,
        "split": split,
        "gt_total": gt_total,
        "reachable": cnt_reachable,
        "reachable_rate": rec_reachable,
        "ub_recall_preNMS": cnt_ub_pre_nms,
        "ub_recall_preNMS_rate": rec_pre_nms,
        "ub_recall_postNMS": cnt_ub_post_nms,
        "ub_recall_postNMS_rate": rec_post_nms,
        "ub_recall_postTopK": cnt_ub_post_topk,
        "ub_recall_postTopK_rate": rec_post_topk,
        "final_recall": cnt_final_matched,
        "final_recall_rate": rec_final,
        "missed_iou_hist": iou_hist,
    }


def run_dvx_deconfounding(
    dataset_dir: Path, base_cfg: Dict[str, Any]
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    """Part 3 — De-confound DVX: 1-D percentile sweep, predilation centroiding, and GT size statistics."""
    sensor_files = filter_gt_files(dataset_dir, "DVX", "train")

    # 1. 1-D Percentile sweep over [85, 90, 93, 95, 97, 98, 99] with open_kernel: 1
    percentiles = [85.0, 90.0, 93.0, 95.0, 97.0, 98.0, 99.0]
    sweep_results: List[Dict[str, Any]] = []

    for perc in percentiles:
        test_cfg = deepcopy(base_cfg)
        if "DVX" not in test_cfg:
            test_cfg["DVX"] = {}
        test_cfg["DVX"]["open_kernel"] = 1
        test_cfg["DVX"]["percentile"] = perc

        attr_res = run_upper_bound_recall_attribution(dataset_dir, "DVX", "train", test_cfg)

        # Run pipeline evaluate for precision and AP
        tot_tp, tot_fp, tot_fn = 0, 0, 0
        all_aps: List[float] = []

        for gt_f in sensor_files:
            seq_name = gt_f.name.replace("_bb_windows_40ms.txt", "")
            npy_matches = list(gt_f.parent.glob(f"{seq_name}_labeled_events.npy"))
            if not npy_matches:
                npy_matches = list(dataset_dir.rglob(f"{seq_name}_labeled_events.npy"))
            if not npy_matches:
                continue

            events = load_events(npy_matches[0])
            width, height = infer_resolution(seq_name, events[:, 0], events[:, 1])

            gt_rows = []
            with open(gt_f, "r", encoding="utf-8") as f:
                rdr = csv.DictReader(f, delimiter="\t")
                for r in rdr:
                    gt_rows.append(
                        (
                            int(r["window_start_timestamp_us"]),
                            int(r["window_end_timestamp_us"]),
                            int(r["center_x"]),
                            int(r["center_y"]),
                            int(r["width"]),
                            int(r["height"]),
                        )
                    )

            preds = run_sequence(events, width, height, test_cfg, window_us=WINDOW_US)
            eval_res = evaluate_sequence(gt_rows, preds)
            tot_tp += eval_res["tp"]
            tot_fp += eval_res["fp"]
            tot_fn += eval_res["fn"]
            if not np.isnan(eval_res["ap"]):
                all_aps.append(eval_res["ap"])

        prec, rec, f1 = compute_prf1(tot_tp, tot_fp, tot_fn)
        mAP = float(np.mean(all_aps)) if all_aps else 0.0

        sweep_results.append(
            {
                "percentile": perc,
                "ub_recall_preNMS": attr_res["ub_recall_preNMS_rate"],
                "final_recall": rec,
                "precision": prec,
                "mAP": mAP,
            }
        )

    # 2. Centroid on predilation mask: True vs False
    cfg_predil_off = deepcopy(base_cfg)
    if "DVX" not in cfg_predil_off:
        cfg_predil_off["DVX"] = {}
    cfg_predil_off["DVX"]["centroid_on_predilation_mask"] = False

    cfg_predil_on = deepcopy(base_cfg)
    if "DVX" not in cfg_predil_on:
        cfg_predil_on["DVX"] = {}
    cfg_predil_on["DVX"]["centroid_on_predilation_mask"] = True

    res_predil_off = run_upper_bound_recall_attribution(dataset_dir, "DVX", "train", cfg_predil_off)
    res_predil_on = run_upper_bound_recall_attribution(dataset_dir, "DVX", "train", cfg_predil_on)

    predil_comparison = {
        "predilation_off": {
            "ub_recall_preNMS": res_predil_off["ub_recall_preNMS_rate"],
            "final_recall": res_predil_off["final_recall_rate"],
        },
        "predilation_on": {
            "ub_recall_preNMS": res_predil_on["ub_recall_preNMS_rate"],
            "final_recall": res_predil_on["final_recall_rate"],
        },
    }

    # 3. GT Width/Height distribution per sensor (min/median/p95/max)
    gt_stats: Dict[str, Any] = {}
    for s_name in ["DAVIS", "DVX", "EVK4"]:
        files = filter_gt_files(dataset_dir, s_name, "train")
        ws: List[float] = []
        hs: List[float] = []
        for gf in files:
            with open(gf, "r", encoding="utf-8") as f:
                rdr = csv.DictReader(f, delimiter="\t")
                for r in rdr:
                    ws.append(float(r["width"]))
                    hs.append(float(r["height"]))

        w_arr = np.array(ws) if ws else np.array([0.0])
        h_arr = np.array(hs) if hs else np.array([0.0])

        gt_stats[s_name] = {
            "width": {
                "min": float(np.min(w_arr)),
                "median": float(np.median(w_arr)),
                "p95": float(np.percentile(w_arr, 95)),
                "max": float(np.max(w_arr)),
            },
            "height": {
                "min": float(np.min(h_arr)),
                "median": float(np.median(h_arr)),
                "p95": float(np.percentile(h_arr, 95)),
                "max": float(np.max(h_arr)),
            },
        }

    return sweep_results, predil_comparison, gt_stats


def main() -> None:
    """CLI entrypoint for upper-bound recall attribution and DVX deconfounding."""
    parser = argparse.ArgumentParser(
        description="OrbitSight Upper-Bound Recall Attribution and DVX Deconfounding"
    )
    parser.add_argument(
        "--sensor",
        type=str,
        default="all",
        choices=["DAVIS", "DVX", "EVK4", "all"],
        help="Sensor family to analyze",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "test", "all"],
        help="Dataset split to analyze",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="../OrbitSight_Dataset",
        help="Path to dataset root",
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to config file"
    )
    parser.add_argument(
        "--deconfound-dvx",
        action="store_true",
        help="Run Part 3 DVX deconfounding analysis",
    )

    args = parser.parse_args()

    cfg_path = Path(args.config)
    cfg = load_yaml_config(cfg_path)
    dataset_dir = Path(args.dataset_dir).resolve()

    print_effective_config(cfg)

    sensors_to_run = ["DAVIS", "DVX", "EVK4"] if args.sensor.lower() == "all" else [args.sensor.upper()]

    if args.deconfound_dvx:
        print("\n==================================================")
        print("  PART 3: DVX DE-CONFOUNDING ANALYSIS (TRAIN SPLIT)")
        print("==================================================")

        sweep_res, predil_res, gt_stats = run_dvx_deconfounding(dataset_dir, cfg)

        print("\n1. DVX 1-D PERCENTILE SWEEP (with open_kernel: 1 fixed):")
        sw_table = [
            [
                r["percentile"],
                f"{r['ub_recall_preNMS']:.4f}",
                f"{r['final_recall']:.4f}",
                f"{r['precision']:.4f}",
                f"{r['mAP']:.4f}",
            ]
            for r in sweep_res
        ]
        print(
            tabulate(
                sw_table,
                headers=["Percentile", "ub_recall_preNMS", "final_recall", "Precision", "mAP@0.5"],
                tablefmt="github",
            )
        )

        print("\n2. CENTROID ON PREDILATION MASK COMPARISON (DVX):")
        predil_table = [
            ["centroid_on_predilation_mask: false", f"{predil_res['predilation_off']['ub_recall_preNMS']:.4f}", f"{predil_res['predilation_off']['final_recall']:.4f}"],
            ["centroid_on_predilation_mask: true",  f"{predil_res['predilation_on']['ub_recall_preNMS']:.4f}",  f"{predil_res['predilation_on']['final_recall']:.4f}"],
        ]
        print(
            tabulate(
                predil_table,
                headers=["Variant", "ub_recall_preNMS", "final_recall"],
                tablefmt="github",
            )
        )

        print("\n3. GROUND TRUTH SIZE DISTRIBUTIONS (Train Split):")
        size_table = []
        for s, d in gt_stats.items():
            size_table.append([f"{s} Width", f"{d['width']['min']:.1f}", f"{d['width']['median']:.1f}", f"{d['width']['p95']:.1f}", f"{d['width']['max']:.1f}"])
            size_table.append([f"{s} Height", f"{d['height']['min']:.1f}", f"{d['height']['median']:.1f}", f"{d['height']['p95']:.1f}", f"{d['height']['max']:.1f}"])

        print(
            tabulate(
                size_table,
                headers=["Sensor Dimension", "Min", "Median", "P95", "Max"],
                tablefmt="github",
            )
        )
        return

    print("\n==================================================")
    print(f"  PART 2: UPPER-BOUND RECALL ATTRIBUTION ({args.split.upper()} SPLIT)")
    print("==================================================")

    all_attr_rows = []

    for sensor in sensors_to_run:
        res = run_upper_bound_recall_attribution(dataset_dir, sensor, args.split, cfg)
        all_attr_rows.append(res)

        out_csv = Path("experiments") / f"recall_attribution_{sensor.lower()}_{args.split.lower()}.csv"
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "sensor",
                "split",
                "gt_total",
                "reachable",
                "reachable_rate",
                "ub_recall_preNMS",
                "ub_recall_preNMS_rate",
                "ub_recall_postNMS",
                "ub_recall_postNMS_rate",
                "ub_recall_postTopK",
                "ub_recall_postTopK_rate",
                "final_recall",
                "final_recall_rate",
                "miss_iou_0.0",
                "miss_iou_0.0_0.25",
                "miss_iou_0.25_0.5",
                "miss_iou_ge_0.5",
            ])
            writer.writerow([
                sensor,
                args.split,
                res["gt_total"],
                res["reachable"],
                f"{res['reachable_rate']:.4f}",
                res["ub_recall_preNMS"],
                f"{res['ub_recall_preNMS_rate']:.4f}",
                res["ub_recall_postNMS"],
                f"{res['ub_recall_postNMS_rate']:.4f}",
                res["ub_recall_postTopK"],
                f"{res['ub_recall_postTopK_rate']:.4f}",
                res["final_recall"],
                f"{res['final_recall_rate']:.4f}",
                res["missed_iou_hist"]["0.0"],
                res["missed_iou_hist"]["0.0-0.25"],
                res["missed_iou_hist"]["0.25-0.5"],
                res["missed_iou_hist"][">=0.5"],
            ])
        print(f"[INFO] Exported {out_csv}")

    # Print summary table
    summary_table = [
        [
            r["sensor"],
            r["gt_total"],
            f"{r['reachable']} ({r['reachable_rate']*100.0:.1f}%)",
            f"{r['ub_recall_preNMS']} ({r['ub_recall_preNMS_rate']*100.0:.1f}%)",
            f"{r['ub_recall_postNMS']} ({r['ub_recall_postNMS_rate']*100.0:.1f}%)",
            f"{r['ub_recall_postTopK']} ({r['ub_recall_postTopK_rate']*100.0:.1f}%)",
            f"{r['final_recall']} ({r['final_recall_rate']*100.0:.1f}%)",
        ]
        for r in all_attr_rows
    ]
    print("\nSTAGE-BY-STAGE UPPER-BOUND RECALL ATTRIBUTION:")
    print(
        tabulate(
            summary_table,
            headers=["Sensor", "gt_total", "reachable", "ub_recall_preNMS", "ub_recall_postNMS", "ub_recall_postTopK", "final_recall"],
            tablefmt="github",
        )
    )

    print("\nMISSED GT BOXES BEST-CANDIDATE IoU HISTOGRAM:")
    hist_table = [
        [
            r["sensor"],
            r["missed_iou_hist"]["0.0"],
            r["missed_iou_hist"]["0.0-0.25"],
            r["missed_iou_hist"]["0.25-0.5"],
            r["missed_iou_hist"][">=0.5"],
        ]
        for r in all_attr_rows
    ]
    print(
        tabulate(
            hist_table,
            headers=["Sensor", "IoU = 0.0", "0.0 < IoU < 0.25", "0.25 <= IoU < 0.5", "IoU >= 0.5"],
            tablefmt="github",
        )
    )


if __name__ == "__main__":
    main()
