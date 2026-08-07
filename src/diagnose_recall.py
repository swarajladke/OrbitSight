"""Recall attribution, oracle analysis, GT variance, box-mode comparisons, and targeted geometry checks."""

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
    sequence_name_from_npy,
)
from src.detector import detect_boxes
from src.metrics import compute_ap, compute_prf1, evaluate_sequence, iou


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
        if target_sensor.upper() not in seq_name.upper():
            continue
        if split == "train" and "Training" not in str(gt_f):
            continue
        if split == "test" and "Testing" not in str(gt_f):
            continue
        filtered.append(gt_f)
    return filtered


def run_gt_variance_analysis(
    dataset_dir: Path, target_sensor: str
) -> Dict[str, Any]:
    """3a — Quantify GT box size variance and multi-box window distribution."""
    sensor_files = filter_gt_files(dataset_dir, target_sensor, "all")
    widths: List[float] = []
    heights: List[float] = []
    boxes_per_window: List[int] = []

    for gt_f in sensor_files:
        gt_by_window: Dict[int, List[Tuple[float, float]]] = {}
        with open(gt_f, "r", encoding="utf-8") as f:
            rdr = csv.DictReader(f, delimiter="\t")
            for r in rdr:
                ws = int(r["window_start_timestamp_us"])
                w_val = float(r["width"])
                h_val = float(r["height"])
                widths.append(w_val)
                heights.append(h_val)
                gt_by_window.setdefault(ws, []).append((w_val, h_val))

        for w_boxes in gt_by_window.values():
            boxes_per_window.append(len(w_boxes))

    w_arr = np.array(widths)
    h_arr = np.array(heights)
    b_arr = np.array(boxes_per_window)

    w_mean = float(w_arr.mean()) if len(w_arr) else 0.0
    w_std = float(w_arr.std()) if len(w_arr) else 0.0
    h_mean = float(h_arr.mean()) if len(h_arr) else 0.0
    h_std = float(h_arr.std()) if len(h_arr) else 0.0

    return {
        "sensor": target_sensor,
        "total_gt_boxes": len(widths),
        "width_cov": (w_std / w_mean) if w_mean > 0 else 0.0,
        "height_cov": (h_std / h_mean) if h_mean > 0 else 0.0,
        "width_percentiles": {
            "p10": float(np.percentile(w_arr, 10)) if len(w_arr) else 0.0,
            "p25": float(np.percentile(w_arr, 25)) if len(w_arr) else 0.0,
            "p50": float(np.percentile(w_arr, 50)) if len(w_arr) else 0.0,
            "p75": float(np.percentile(w_arr, 75)) if len(w_arr) else 0.0,
            "p90": float(np.percentile(w_arr, 90)) if len(w_arr) else 0.0,
        },
        "height_percentiles": {
            "p10": float(np.percentile(h_arr, 10)) if len(h_arr) else 0.0,
            "p25": float(np.percentile(h_arr, 25)) if len(h_arr) else 0.0,
            "p50": float(np.percentile(h_arr, 50)) if len(h_arr) else 0.0,
            "p75": float(np.percentile(h_arr, 75)) if len(h_arr) else 0.0,
            "p90": float(np.percentile(h_arr, 90)) if len(h_arr) else 0.0,
        },
        "max_boxes_per_window": int(b_arr.max()) if len(b_arr) else 0,
        "frac_multi_box_windows": float((b_arr > 1).mean()) if len(b_arr) else 0.0,
    }


def run_oracle_analysis(
    dataset_dir: Path, target_sensor: str, cfg: Dict[str, Any], split: str = "all"
) -> Dict[str, Any]:
    """Analysis A — Naive vs Clustered Oracle ceiling using label == 1 RSO events."""
    sensor_files = filter_gt_files(dataset_dir, target_sensor, split)

    sensor_cfg = (
        cfg.get(target_sensor, {})
        if isinstance(cfg.get(target_sensor), dict)
        else {}
    )

    if target_sensor == "EVK4":
        def_w, def_h = 52, 56
    elif target_sensor == "DVX":
        def_w, def_h = 18, 18
    else:
        def_w, def_h = 10, 12

    fixed_w = float(sensor_cfg.get("box_w", cfg.get("box_w", def_w)))
    fixed_h = float(sensor_cfg.get("box_h", cfg.get("box_h", def_h)))

    naive_ious: List[float] = []
    clustered_ious: List[float] = []
    unweighted_ious: List[float] = []
    weighted_ious: List[float] = []
    centroid_errors: List[float] = []
    gt_w_thirds: List[float] = []
    zero_label_windows = 0
    total_gt_windows = 0

    for gt_f in sensor_files:
        seq_name = gt_f.name.replace("_bb_windows_40ms.txt", "")
        npy_matches = list(gt_f.parent.glob(f"{seq_name}_labeled_events.npy"))
        if not npy_matches:
            npy_matches = list(dataset_dir.rglob(f"{seq_name}_labeled_events.npy"))
        if not npy_matches:
            continue

        events = load_events(npy_matches[0])
        width, height = infer_resolution(seq_name, events[:, 0], events[:, 1])

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

        for w_start, _, w_events in iter_windows(events, window_us=WINDOW_US):
            if w_start not in gt_by_window:
                continue

            gt_boxes = gt_by_window[w_start]
            label1_events = w_events[w_events[:, 4] == 1]

            for gt_cx, gt_cy, gt_w, gt_h in gt_boxes:
                total_gt_windows += 1
                if len(label1_events) == 0:
                    zero_label_windows += 1
                    naive_ious.append(0.0)
                    clustered_ious.append(0.0)
                    unweighted_ious.append(0.0)
                    weighted_ious.append(0.0)
                    continue

                xs = label1_events[:, 0].astype(float)
                ys = label1_events[:, 1].astype(float)

                # Naive Oracle: Single tight box of all label==1 events
                min_x, max_x = xs.min(), xs.max()
                min_y, max_y = ys.min(), ys.max()
                t_w = max(3.0, max_x - min_x + 1.0)
                t_h = max(3.0, max_y - min_y + 1.0)
                t_cx = (min_x + max_x) / 2.0
                t_cy = (min_y + max_y) / 2.0

                naive_score = iou(
                    (t_cx, t_cy, t_w, t_h), (gt_cx, gt_cy, gt_w, gt_h)
                )
                naive_ious.append(naive_score)

                # Clustered Oracle: Spatial connected components on label==1 events
                l1_img = np.zeros((height, width), dtype=np.uint8)
                ixs = np.clip(xs.astype(int), 0, width - 1)
                iys = np.clip(ys.astype(int), 0, height - 1)
                l1_img[iys, ixs] = 1

                n_cl, l_cl, s_cl, c_cl = cv2.connectedComponentsWithStats(l1_img, connectivity=8)

                best_cl_score = 0.0
                best_w_cx, best_w_cy = t_cx, t_cy

                for ci in range(1, n_cl):
                    cx_c = float(c_cl[ci, 0])
                    cy_c = float(c_cl[ci, 1])
                    cw_c = float(s_cl[ci, cv2.CC_STAT_WIDTH])
                    ch_c = float(s_cl[ci, cv2.CC_STAT_HEIGHT])

                    score = iou((cx_c, cy_c, cw_c, ch_c), (gt_cx, gt_cy, gt_w, gt_h))
                    if score > best_cl_score:
                        best_cl_score = score
                        best_w_cx = cx_c
                        best_w_cy = cy_c

                clustered_ious.append(best_cl_score)

                # Unweighted & Weighted Centroids
                u_cx = xs.mean()
                u_cy = ys.mean()
                unw_score = iou(
                    (u_cx, u_cy, fixed_w, fixed_h),
                    (gt_cx, gt_cy, gt_w, gt_h),
                )
                unweighted_ious.append(unw_score)

                w_score = iou(
                    (best_w_cx, best_w_cy, fixed_w, fixed_h),
                    (gt_cx, gt_cy, gt_w, gt_h),
                )
                weighted_ious.append(w_score)

                c_err = math.hypot(best_w_cx - gt_cx, best_w_cy - gt_cy)
                centroid_errors.append(c_err)
                gt_w_thirds.append(gt_w / 3.0)

    naive_arr = np.array(naive_ious)
    cl_arr = np.array(clustered_ious)
    unw_arr = np.array(unweighted_ious)
    w_arr = np.array(weighted_ious)
    err_arr = np.array(centroid_errors)
    third_arr = np.array(gt_w_thirds)

    return {
        "sensor": target_sensor,
        "split": split,
        "total_gt_windows": total_gt_windows,
        "zero_label_windows": zero_label_windows,
        "naive_tight": {
            "reach_iou50": float((naive_arr >= 0.5).mean()) if len(naive_arr) else 0.0,
            "mean_iou": float(naive_arr.mean()) if len(naive_arr) else 0.0,
            "median_iou": float(np.median(naive_arr)) if len(naive_arr) else 0.0,
        },
        "clustered_tight": {
            "reach_iou50": float((cl_arr >= 0.5).mean()) if len(cl_arr) else 0.0,
            "mean_iou": float(cl_arr.mean()) if len(cl_arr) else 0.0,
            "median_iou": float(np.median(cl_arr)) if len(cl_arr) else 0.0,
        },
        "unweighted": {
            "reach_iou50": float((unw_arr >= 0.5).mean()) if len(unw_arr) else 0.0,
            "mean_iou": float(unw_arr.mean()) if len(unw_arr) else 0.0,
            "median_iou": float(np.median(unw_arr)) if len(unw_arr) else 0.0,
        },
        "weighted": {
            "reach_iou50": float((w_arr >= 0.5).mean()) if len(w_arr) else 0.0,
            "mean_iou": float(w_arr.mean()) if len(w_arr) else 0.0,
            "median_iou": float(np.median(w_arr)) if len(w_arr) else 0.0,
            "c_err_mean": float(err_arr.mean()) if len(err_arr) else 0.0,
            "c_err_median": float(np.median(err_arr)) if len(err_arr) else 0.0,
            "c_err_p90": float(np.percentile(err_arr, 90)) if len(err_arr) else 0.0,
            "within_gt_third": float((err_arr <= third_arr).mean()) if len(err_arr) else 0.0,
        },
    }


def run_stagewise_and_miss_analysis(
    dataset_dir: Path, target_sensor: str, cfg: Dict[str, Any], split: str = "all"
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Analysis B & C — Memory-efficient Stage-wise recall and miss attribution."""
    sensor_files = filter_gt_files(dataset_dir, target_sensor, split)

    sensor_cfg = (
        cfg.get(target_sensor, {})
        if isinstance(cfg.get(target_sensor), dict)
        else {}
    )

    percentile = float(sensor_cfg.get("percentile", cfg.get("percentile", 97.5)))
    open_k = int(sensor_cfg.get("open_kernel", cfg.get("open_kernel", 2)))
    dilate_k = int(sensor_cfg.get("dilate_kernel", cfg.get("dilate_kernel", 3)))
    min_hits = int(sensor_cfg.get("min_hits", cfg.get("min_hits", 2)))

    s1_matched = 0
    s2_matched = 0
    s3_matched = 0
    s4_matched = 0
    total_gt = 0

    miss_buckets = {
        "NO_EVENTS": 0,
        "BELOW_THRESHOLD": 0,
        "LOST_TO_MORPHOLOGY": 0,
        "LOST_TO_FILTERS": 0,
        "LOST_TO_PERSISTENCE": 0,
        "IOU_TOO_LOW": 0,
        "OTHER": 0,
    }

    iou_too_low_scores: List[float] = []
    iou_too_low_offsets: List[float] = []

    for gt_f in sensor_files:
        seq_name = gt_f.name.replace("_bb_windows_40ms.txt", "")
        npy_matches = list(gt_f.parent.glob(f"{seq_name}_labeled_events.npy"))
        if not npy_matches:
            npy_matches = list(dataset_dir.rglob(f"{seq_name}_labeled_events.npy"))
        if not npy_matches:
            continue

        events = load_events(npy_matches[0])
        width, height = infer_resolution(seq_name, events[:, 0], events[:, 1])
        diagonal = math.hypot(width, height)
        max_dist = 0.05 * diagonal

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

        w_idx = 0
        w_s3_boxes_list: List[List[Tuple[float, float, float, float]]] = []
        w_s1_boxes_list: List[List[Tuple[float, float, float, float]]] = []
        w_s2_boxes_list: List[List[Tuple[float, float, float, float]]] = []
        w_meta: List[Tuple[int, int, int]] = []

        for w_start, w_end, w_events in iter_windows(events, window_us=WINDOW_US):
            count_img, _, _ = event_image(w_events, width, height)
            nonzero_vals = count_img[count_img > 0]

            if len(nonzero_vals) < 4:
                w_s1_boxes_list.append([])
                w_s2_boxes_list.append([])
                w_s3_boxes_list.append([])
                w_meta.append((w_start, w_end, len(w_events)))
                del count_img
                w_idx += 1
                continue

            raw_thresh = float(np.percentile(nonzero_vals, percentile))
            thresh = max(1.0, raw_thresh)

            b_s1 = (count_img >= thresh).astype(np.uint8)
            num_l1, labels1, stats1, centroids1 = cv2.connectedComponentsWithStats(
                b_s1, connectivity=8
            )
            s1_b = [
                (
                    float(centroids1[i, 0]),
                    float(centroids1[i, 1]),
                    float(stats1[i, cv2.CC_STAT_WIDTH]),
                    float(stats1[i, cv2.CC_STAT_HEIGHT]),
                )
                for i in range(1, num_l1)
            ]
            w_s1_boxes_list.append(s1_b)

            b_s2 = b_s1.copy()
            if open_k > 1:
                kernel_open = cv2.getStructuringElement(
                    cv2.MORPH_RECT, (open_k, open_k)
                )
                b_s2 = cv2.morphologyEx(b_s2, cv2.MORPH_OPEN, kernel_open)
            if dilate_k > 1:
                kernel_dilate = cv2.getStructuringElement(
                    cv2.MORPH_RECT, (dilate_k, dilate_k)
                )
                b_s2 = cv2.dilate(b_s2, kernel_dilate)

            num_l2, labels2, stats2, centroids2 = cv2.connectedComponentsWithStats(
                b_s2, connectivity=8
            )
            s2_b = [
                (
                    float(centroids2[i, 0]),
                    float(centroids2[i, 1]),
                    float(stats2[i, cv2.CC_STAT_WIDTH]),
                    float(stats2[i, cv2.CC_STAT_HEIGHT]),
                )
                for i in range(1, num_l2)
            ]
            w_s2_boxes_list.append(s2_b)

            s3_dets = detect_boxes(count_img, width, height, cfg)
            s3_b = [
                (
                    b["center_x"],
                    b["center_y"],
                    b["width"],
                    b["height"],
                )
                for b in s3_dets
            ]
            w_s3_boxes_list.append(s3_b)
            w_meta.append((w_start, w_end, len(w_events)))
            del count_img
            w_idx += 1

        num_windows = len(w_meta)

        for idx in range(num_windows):
            w_start, w_end, ev_cnt = w_meta[idx]
            if w_start not in gt_by_window:
                continue

            s3_b = w_s3_boxes_list[idx]
            prev_b = w_s3_boxes_list[idx - 1] if idx > 0 else []
            next_b = w_s3_boxes_list[idx + 1] if idx < num_windows - 1 else []

            s4_b = []
            for box in s3_b:
                hits = 1
                if any(
                    math.hypot(box[0] - p[0], box[1] - p[1]) <= max_dist
                    for p in prev_b
                ):
                    hits += 1
                if any(
                    math.hypot(box[0] - n[0], box[1] - n[1]) <= max_dist
                    for n in next_b
                ):
                    hits += 1
                if min_hits < 2 or hits >= min_hits:
                    s4_b.append(box)

            w_s1_b = w_s1_boxes_list[idx]
            w_s2_b = w_s2_boxes_list[idx]

            for gt_box in gt_by_window[w_start]:
                total_gt += 1
                gt_cx, gt_cy, gt_w, gt_h = gt_box

                m1 = any(iou(b, gt_box) >= 0.5 for b in w_s1_b)
                m2 = any(iou(b, gt_box) >= 0.5 for b in w_s2_b)
                m3 = any(iou(b, gt_box) >= 0.5 for b in s3_b)
                m4 = any(iou(b, gt_box) >= 0.5 for b in s4_b)

                if m1:
                    s1_matched += 1
                if m2:
                    s2_matched += 1
                if m3:
                    s3_matched += 1
                if m4:
                    s4_matched += 1

                if not m4:
                    if ev_cnt == 0:
                        miss_buckets["NO_EVENTS"] += 1
                    elif not m1:
                        miss_buckets["BELOW_THRESHOLD"] += 1
                    elif not m2:
                        miss_buckets["LOST_TO_MORPHOLOGY"] += 1
                    elif not m3:
                        miss_buckets["LOST_TO_FILTERS"] += 1
                    elif not m4:
                        miss_buckets["LOST_TO_PERSISTENCE"] += 1
                    else:
                        best_iou = 0.0
                        best_off = 0.0
                        for b in s4_b:
                            score = iou(b, gt_box)
                            if score > best_iou:
                                best_iou = score
                                best_off = math.hypot(b[0] - gt_cx, b[1] - gt_cy)

                        if best_iou > 0.0 and best_iou < 0.5:
                            miss_buckets["IOU_TOO_LOW"] += 1
                            iou_too_low_scores.append(best_iou)
                            iou_too_low_offsets.append(best_off)
                        else:
                            miss_buckets["OTHER"] += 1

    stage_res = {
        "sensor": target_sensor,
        "split": split,
        "total_gt": total_gt,
        "recalls": {
            "S1": s1_matched / total_gt if total_gt > 0 else 0.0,
            "S2": s2_matched / total_gt if total_gt > 0 else 0.0,
            "S3": s3_matched / total_gt if total_gt > 0 else 0.0,
            "S4": s4_matched / total_gt if total_gt > 0 else 0.0,
        },
        "drops": {
            "S1_to_S2": (s1_matched - s2_matched) / total_gt if total_gt > 0 else 0.0,
            "S2_to_S3": (s2_matched - s3_matched) / total_gt if total_gt > 0 else 0.0,
            "S3_to_S4": (s3_matched - s4_matched) / total_gt if total_gt > 0 else 0.0,
        },
    }

    miss_res = {
        "buckets": miss_buckets,
        "iou_too_low_stats": {
            "mean": float(np.mean(iou_too_low_scores)) if iou_too_low_scores else 0.0,
            "median": float(np.median(iou_too_low_scores)) if iou_too_low_scores else 0.0,
            "p10": float(np.percentile(iou_too_low_scores, 10)) if iou_too_low_scores else 0.0,
            "p90": float(np.percentile(iou_too_low_scores, 90)) if iou_too_low_scores else 0.0,
        },
        "offset_stats": {
            "mean": float(np.mean(iou_too_low_offsets)) if iou_too_low_offsets else 0.0,
            "median": float(np.median(iou_too_low_offsets)) if iou_too_low_offsets else 0.0,
            "p90": float(np.percentile(iou_too_low_offsets, 90)) if iou_too_low_offsets else 0.0,
        },
    }

    return stage_res, miss_res


def run_threshold_sensitivity(
    dataset_dir: Path, target_sensor: str, cfg: Dict[str, Any], split: str = "all"
) -> List[Dict[str, Any]]:
    """Analysis E — Threshold sensitivity sweep over percentiles [85, 90, 93, 95, 97, 97.5, 98, 99, 99.5]."""
    percentiles = [85.0, 90.0, 93.0, 95.0, 97.0, 97.5, 98.0, 99.0, 99.5]
    sensor_files = filter_gt_files(dataset_dir, target_sensor, split)

    sweep_results: List[Dict[str, Any]] = []

    for perc in percentiles:
        test_cfg = deepcopy(cfg)
        if target_sensor not in test_cfg:
            test_cfg[target_sensor] = {}
        test_cfg[target_sensor]["percentile"] = perc

        s2_hits = 0
        final_tp = 0
        final_fp = 0
        final_fn = 0
        total_gt = 0
        total_cand_count = 0
        total_windows = 0
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

            total_gt += len(gt_rows)
            window_boxes: List[Tuple[int, int, List[Dict[str, float]]]] = []

            for w_start, w_end, w_events in iter_windows(events, window_us=WINDOW_US):
                total_windows += 1
                count_img, _, _ = event_image(w_events, width, height)
                boxes = detect_boxes(count_img, width, height, test_cfg)
                total_cand_count += len(boxes)
                window_boxes.append((w_start, w_end, boxes))

                for gt in gt_rows:
                    if gt[0] == w_start:
                        if any(
                            iou((b["center_x"], b["center_y"], b["width"], b["height"]), (gt[2], gt[3], gt[4], gt[5])) >= 0.5
                            for b in boxes
                        ):
                            s2_hits += 1

            pred_rows: List[Tuple[int, int, int, int, int, int, float]] = []
            num_windows = len(window_boxes)
            diagonal = math.hypot(width, height)
            max_dist = 0.05 * diagonal

            for w_idx in range(num_windows):
                w_start, w_end, boxes = window_boxes[w_idx]
                if not boxes:
                    continue
                prev_b = window_boxes[w_idx - 1][2] if w_idx > 0 else []
                next_b = (
                    window_boxes[w_idx + 1][2] if w_idx < num_windows - 1 else []
                )

                for box in boxes:
                    hits = 1
                    if any(
                        math.hypot(box["center_x"] - p["center_x"], box["center_y"] - p["center_y"]) <= max_dist
                        for p in prev_b
                    ):
                        hits += 1
                    if any(
                        math.hypot(box["center_x"] - n["center_x"], box["center_y"] - n["center_y"]) <= max_dist
                        for n in next_b
                    ):
                        hits += 1

                    if hits >= 2:
                        conf = min(1.0, max(0.01, box["density"] * (hits / 3.0)))
                        pred_rows.append(
                            (
                                w_start,
                                w_end,
                                int(round(box["center_x"])),
                                int(round(box["center_y"])),
                                int(round(box["width"])),
                                int(round(box["height"])),
                                conf,
                            )
                        )

            eval_res = evaluate_sequence(gt_rows, pred_rows)
            final_tp += eval_res["tp"]
            final_fp += eval_res["fp"]
            final_fn += eval_res["fn"]
            if not np.isnan(eval_res["ap"]):
                all_aps.append(eval_res["ap"])

        prec, rec, f1 = compute_prf1(final_tp, final_fp, final_fn)
        mAP = float(np.mean(all_aps)) if all_aps else 0.0

        sweep_results.append(
            {
                "percentile": perc,
                "s2_candidate_recall": s2_hits / total_gt if total_gt > 0 else 0.0,
                "final_recall": rec,
                "precision": prec,
                "mAP": mAP,
                "mean_candidates_per_window": total_cand_count / total_windows if total_windows > 0 else 0.0,
            }
        )

    return sweep_results


def run_box_mode_comparison(
    dataset_dir: Path, cfg: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """3c — Compare scale, fixed, and extent box modes per sensor and split."""
    results: List[Dict[str, Any]] = []

    for sensor in ["DAVIS", "DVX", "EVK4"]:
        for split in ["train", "test"]:
            sensor_files = filter_gt_files(dataset_dir, sensor, split)
            if not sensor_files:
                continue

            for b_mode in ["scale", "fixed", "extent"]:
                test_cfg = deepcopy(cfg)
                if sensor not in test_cfg:
                    test_cfg[sensor] = {}
                test_cfg[sensor]["box_mode"] = b_mode

                tot_tp, tot_fp, tot_fn = 0, 0, 0
                all_aps: List[float] = []
                total_ms, total_windows = 0.0, 0

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

                    start_t = time.perf_counter()
                    window_boxes: List[Tuple[int, int, List[Dict[str, float]]]] = []
                    for w_start, w_end, w_events in iter_windows(events, window_us=WINDOW_US):
                        count_img, _, _ = event_image(w_events, width, height)
                        boxes = detect_boxes(count_img, width, height, test_cfg)
                        window_boxes.append((w_start, w_end, boxes))

                    num_w = len(window_boxes)
                    elapsed_ms = (time.perf_counter() - start_t) * 1000.0
                    total_ms += elapsed_ms
                    total_windows += num_w

                    diagonal = math.hypot(width, height)
                    max_dist = 0.05 * diagonal
                    pred_rows: List[Tuple[int, int, int, int, int, int, float]] = []

                    for w_idx in range(num_w):
                        w_start, w_end, boxes = window_boxes[w_idx]
                        if not boxes:
                            continue
                        prev_b = window_boxes[w_idx - 1][2] if w_idx > 0 else []
                        next_b = (
                            window_boxes[w_idx + 1][2] if w_idx < num_w - 1 else []
                        )

                        for box in boxes:
                            hits = 1
                            if any(
                                math.hypot(box["center_x"] - p["center_x"], box["center_y"] - p["center_y"]) <= max_dist
                                for p in prev_b
                            ):
                                hits += 1
                            if any(
                                math.hypot(box["center_x"] - n["center_x"], box["center_y"] - n["center_y"]) <= max_dist
                                for n in next_b
                            ):
                                hits += 1

                            if hits >= 2:
                                conf = min(1.0, max(0.01, box["density"] * (hits / 3.0)))
                                pred_rows.append(
                                    (
                                        w_start,
                                        w_end,
                                        int(round(box["center_x"])),
                                        int(round(box["center_y"])),
                                        int(round(box["width"])),
                                        int(round(box["height"])),
                                        conf,
                                    )
                                )

                    eval_res = evaluate_sequence(gt_rows, pred_rows)
                    tot_tp += eval_res["tp"]
                    tot_fp += eval_res["fp"]
                    tot_fn += eval_res["fn"]
                    if not np.isnan(eval_res["ap"]):
                        all_aps.append(eval_res["ap"])

                prec, rec, f1 = compute_prf1(tot_tp, tot_fp, tot_fn)
                mAP = float(np.mean(all_aps)) if all_aps else 0.0
                ms_win = total_ms / total_windows if total_windows > 0 else 0.0

                results.append(
                    {
                        "sensor": sensor,
                        "split": split,
                        "box_mode": b_mode,
                        "mAP": mAP,
                        "precision": prec,
                        "recall": rec,
                        "f1": f1,
                        "ms_per_window": ms_win,
                        "at_risk": ms_win > 30.0,
                    }
                )

    return results


def run_targeted_geometry_checks(
    dataset_dir: Path, target_sensor: str, base_cfg: Dict[str, Any], split: str = "all"
) -> List[Dict[str, Any]]:
    """Part 3 — Targeted geometry checks per sensor."""
    sensor_files = filter_gt_files(dataset_dir, target_sensor, split)

    if target_sensor == "EVK4":
        candidates = [
            (52, h, c_mode)
            for h in [44, 46, 48, 52, 56]
            for c_mode in ["component", "weighted"]
        ]
    elif target_sensor == "DAVIS":
        candidates = [
            (w, h, c_mode)
            for w, h in [(8, 10), (9, 11), (10, 12), (11, 13)]
            for c_mode in ["component", "weighted"]
        ]
    else:  # DVX
        candidates = [
            (w, h, c_mode)
            for w, h in [(14, 14), (16, 16), (18, 18), (20, 20), (24, 24)]
            for c_mode in ["component", "weighted"]
        ]

    check_results: List[Dict[str, Any]] = []

    for bw, bh, c_mode in candidates:
        test_cfg = deepcopy(base_cfg)
        if target_sensor not in test_cfg:
            test_cfg[target_sensor] = {}
        test_cfg[target_sensor]["box_mode"] = "fixed"
        test_cfg[target_sensor]["box_w"] = bw
        test_cfg[target_sensor]["box_h"] = bh
        test_cfg[target_sensor]["centroid_mode"] = c_mode

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

            window_boxes: List[Tuple[int, int, List[Dict[str, float]]]] = []
            for w_start, w_end, w_events in iter_windows(events, window_us=WINDOW_US):
                count_img, _, _ = event_image(w_events, width, height)
                boxes = detect_boxes(count_img, width, height, test_cfg)
                window_boxes.append((w_start, w_end, boxes))

            num_windows = len(window_boxes)
            diagonal = math.hypot(width, height)
            max_dist = 0.05 * diagonal
            pred_rows: List[Tuple[int, int, int, int, int, int, float]] = []

            for w_idx in range(num_windows):
                w_start, w_end, boxes = window_boxes[w_idx]
                if not boxes:
                    continue
                prev_b = window_boxes[w_idx - 1][2] if w_idx > 0 else []
                next_b = (
                    window_boxes[w_idx + 1][2] if w_idx < num_windows - 1 else []
                )

                for box in boxes:
                    hits = 1
                    if any(
                        math.hypot(box["center_x"] - p["center_x"], box["center_y"] - p["center_y"]) <= max_dist
                        for p in prev_b
                    ):
                        hits += 1
                    if any(
                        math.hypot(box["center_x"] - n["center_x"], box["center_y"] - n["center_y"]) <= max_dist
                        for n in next_b
                    ):
                        hits += 1

                    if hits >= 2:
                        conf = min(1.0, max(0.01, box["density"] * (hits / 3.0)))
                        pred_rows.append(
                            (
                                w_start,
                                w_end,
                                int(round(box["center_x"])),
                                int(round(box["center_y"])),
                                int(round(box["width"])),
                                int(round(box["height"])),
                                conf,
                            )
                        )

            eval_res = evaluate_sequence(gt_rows, pred_rows)
            if not np.isnan(eval_res["ap"]):
                all_aps.append(eval_res["ap"])

        mAP = float(np.mean(all_aps)) if all_aps else 0.0
        check_results.append(
            {
                "box_w": bw,
                "box_h": bh,
                "centroid_mode": c_mode,
                "mAP": mAP,
            }
        )

    return check_results


def main() -> None:
    """Main CLI entrypoint for recall diagnostics and geometry checks."""
    parser = argparse.ArgumentParser(
        description="OrbitSight Recall Attribution & Geometry Check Harness"
    )
    parser.add_argument(
        "--sensor",
        type=str,
        default="DAVIS",
        choices=["DAVIS", "DVX", "EVK4"],
        help="Target sensor family",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="all",
        choices=["train", "test", "all"],
        help="Subset split to analyze (train, test, or all)",
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
        "--check-geometry",
        action="store_true",
        help="Run Part 3 targeted geometry checks mode",
    )
    parser.add_argument(
        "--compare-box-modes",
        action="store_true",
        help="Run 3c box mode comparison across scale, fixed, and extent",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="experiments/recall_diagnostics.csv",
        help="Output CSV file path",
    )

    args = parser.parse_args()

    cfg_path = Path(args.config)
    cfg = load_yaml_config(cfg_path)
    dataset_dir = Path(args.dataset_dir).resolve()
    sensor = args.sensor
    split = args.split

    if args.compare_box_modes:
        print("\n[INFO] Running 3c Box Mode Comparison (Scale vs Fixed vs Extent)...")
        bm_res = run_box_mode_comparison(dataset_dir, cfg)
        bm_table = [
            [
                r["sensor"],
                r["split"],
                r["box_mode"],
                f"{r['mAP']:.6f}",
                f"{r['precision']:.6f}",
                f"{r['recall']:.6f}",
                f"{r['f1']:.6f}",
                f"{r['ms_per_window']:.2f}",
                "AT RISK" if r["at_risk"] else "PASS",
            ]
            for r in bm_res
        ]
        print(
            tabulate(
                bm_table,
                headers=[
                    "Sensor",
                    "Split",
                    "Box Mode",
                    "mAP@0.5",
                    "Precision",
                    "Recall",
                    "F1",
                    "ms/win",
                    "Latency Guard",
                ],
                tablefmt="github",
            )
        )
        return

    if args.check_geometry:
        print(f"\n[INFO] Running Targeted Geometry Checks for Sensor '{sensor}' (Split: {split})...")
        geom_res = run_targeted_geometry_checks(dataset_dir, sensor, cfg, split)
        geom_res.sort(key=lambda x: x["mAP"], reverse=True)

        rows = [
            [
                r["box_w"],
                r["box_h"],
                r["centroid_mode"],
                f"{r['mAP']:.6f}",
            ]
            for r in geom_res
        ]
        print(
            tabulate(
                rows,
                headers=["Box W", "Box H", "Centroid Mode", "mAP@0.5"],
                tablefmt="github",
            )
        )
        return

    print(f"\n[INFO] Running Recall Diagnostics for Sensor '{sensor}' (Split: {split})...")

    gt_var = run_gt_variance_analysis(dataset_dir, sensor)
    oracle_res = run_oracle_analysis(dataset_dir, sensor, cfg, split)
    stage_res, miss_res = run_stagewise_and_miss_analysis(dataset_dir, sensor, cfg, split)
    sens_res = run_threshold_sensitivity(dataset_dir, sensor, cfg, split)

    print("\n--- 3a: GT BOX SIZE VARIANCE ---")
    print(f"Total GT Boxes: {gt_var['total_gt_boxes']}")
    print(f"Width CoV (std/mean): {gt_var['width_cov']:.4f}, Height CoV: {gt_var['height_cov']:.4f}")
    print(f"Width Percentiles  [P10={gt_var['width_percentiles']['p10']:.1f}, P25={gt_var['width_percentiles']['p25']:.1f}, P50={gt_var['width_percentiles']['p50']:.1f}, P75={gt_var['width_percentiles']['p75']:.1f}, P90={gt_var['width_percentiles']['p90']:.1f}]")
    print(f"Height Percentiles [P10={gt_var['height_percentiles']['p10']:.1f}, P25={gt_var['height_percentiles']['p25']:.1f}, P50={gt_var['height_percentiles']['p50']:.1f}, P75={gt_var['height_percentiles']['p75']:.1f}, P90={gt_var['height_percentiles']['p90']:.1f}]")
    print(f"Max GT Boxes per Window: {gt_var['max_boxes_per_window']}, Fraction Multi-Box Windows: {gt_var['frac_multi_box_windows']:.4f}")

    print("\n--- ANALYSIS A: NAIVE VS CLUSTERED ORACLE CEILING ---")
    oracle_table = [
        [
            "Naive Tight Box (All label=1)",
            f"{oracle_res['naive_tight']['reach_iou50']:.4f}",
            f"{oracle_res['naive_tight']['mean_iou']:.4f}",
            f"{oracle_res['naive_tight']['median_iou']:.4f}",
        ],
        [
            "Clustered Tight Box (Per-object)",
            f"{oracle_res['clustered_tight']['reach_iou50']:.4f}",
            f"{oracle_res['clustered_tight']['mean_iou']:.4f}",
            f"{oracle_res['clustered_tight']['median_iou']:.4f}",
        ],
        [
            "Fixed Box (Unweighted Centroid)",
            f"{oracle_res['unweighted']['reach_iou50']:.4f}",
            f"{oracle_res['unweighted']['mean_iou']:.4f}",
            f"{oracle_res['unweighted']['median_iou']:.4f}",
        ],
        [
            "Fixed Box (Weighted Centroid)",
            f"{oracle_res['weighted']['reach_iou50']:.4f}",
            f"{oracle_res['weighted']['mean_iou']:.4f}",
            f"{oracle_res['weighted']['median_iou']:.4f}",
        ],
    ]
    print(
        tabulate(
            oracle_table,
            headers=["Variant", "Fraction IoU>=0.5", "Mean IoU", "Median IoU"],
            tablefmt="github",
        )
    )

    print("\n--- ANALYSIS B: STAGE-WISE RECALL ---")
    stage_table = [
        ["S1 (After Thresholding)", f"{stage_res['recalls']['S1']:.4f}", "-"],
        ["S2 (After Connected Components)", f"{stage_res['recalls']['S2']:.4f}", f"{stage_res['drops']['S1_to_S2']:.4f}"],
        ["S3 (After Area & Event Filters)", f"{stage_res['recalls']['S3']:.4f}", f"{stage_res['drops']['S2_to_S3']:.4f}"],
        ["S4 (After Persistence Filter)", f"{stage_res['recalls']['S4']:.4f}", f"{stage_res['drops']['S3_to_S4']:.4f}"],
    ]
    print(
        tabulate(
            stage_table,
            headers=["Pipeline Stage", "Recall", "Drop from Prev Stage"],
            tablefmt="github",
        )
    )

    print("\n--- ANALYSIS C: MISS ATTRIBUTION BUCKETS ---")
    miss_table = [
        [k, v, f"{(v / stage_res['total_gt'] * 100.0) if stage_res['total_gt'] > 0 else 0.0:.2f}%"]
        for k, v in miss_res["buckets"].items()
    ]
    print(
        tabulate(
            miss_table,
            headers=["Miss Bucket", "Count", "Percentage"],
            tablefmt="github",
        )
    )

    print("\n--- ANALYSIS E: THRESHOLD SENSITIVITY ---")
    sens_table = [
        [
            r["percentile"],
            f"{r['s2_candidate_recall']:.4f}",
            f"{r['final_recall']:.4f}",
            f"{r['precision']:.4f}",
            f"{r['mAP']:.4f}",
            f"{r['mean_candidates_per_window']:.2f}",
        ]
        for r in sens_res
    ]
    print(
        tabulate(
            sens_table,
            headers=["Percentile", "S2 Rec", "Final Rec", "Precision", "mAP", "Cand/Win"],
            tablefmt="github",
        )
    )

    print("\n==================================================")
    print("  FINDINGS BLOCK (MEASURED VALUES ONLY)")
    print("==================================================")
    print(f"Sensor: {sensor} (Split: {split})")
    print(f"Naive Oracle Ceiling: {oracle_res['naive_tight']['reach_iou50']:.4f}, Clustered Oracle Ceiling: {oracle_res['clustered_tight']['reach_iou50']:.4f}")
    print(f"Largest Recall Drop Stage: {max(stage_res['drops'].items(), key=lambda x: x[1])[0]}")
    print(f"Measured Centroid Error: Mean={oracle_res['weighted']['c_err_mean']:.2f}px, Median={oracle_res['weighted']['c_err_median']:.2f}px, P90={oracle_res['weighted']['c_err_p90']:.2f}px")


if __name__ == "__main__":
    main()
