"""Oracle recall ceiling measurement, stage funnel breakdown, and counterfactual analysis."""

import argparse
import csv
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from tabulate import tabulate

from src.common import (
    WINDOW_US,
    event_image,
    infer_resolution,
    iter_windows,
    load_events,
    resolve_effective_config,
)
from src.metrics import cx_cy_wh_to_xyxy, iou
from src.scoreboard import load_yaml_config
from src.static_map import build_continuous_static_map


def load_gt_rows(gt_path: Path) -> List[Tuple[int, int, int, int, int, int]]:
    rows = []
    with open(gt_path, "r", encoding="utf-8") as f:
        rdr = csv.DictReader(f, delimiter="\t")
        for r in rdr:
            rows.append((
                int(r["window_start_timestamp_us"]),
                int(r["window_end_timestamp_us"]),
                int(round(float(r["center_x"]))),
                int(round(float(r["center_y"]))),
                int(round(float(r["width"]))),
                int(round(float(r["height"]))),
            ))
    return rows


def get_sensor_family(seq_name: str) -> str:
    s = seq_name.upper()
    if "EVK4" in s:
        return "EVK4"
    elif "DVX" in s:
        return "DVX"
    else:
        return "DAVIS"


def detect_stage_funnel_and_candidates(
    count_img: np.ndarray,
    width: int,
    height: int,
    cfg: Dict[str, Any],
    static_mask: Optional[np.ndarray] = None,
    override_box_mode_extent: bool = False,
    disable_escalation: bool = False,
    disable_static_mask: bool = False,
) -> Tuple[Dict[str, bool], List[Dict[str, Any]]]:
    """Execute detection stages with fine-grained tracking of survival."""
    survives = {
        "S0": False,
        "S1": False,
        "S2": False,
        "S3": False,
        "S4": False,
        "S5": False,
    }

    nonzero_vals = count_img[count_img > 0]
    num_nonzero = len(nonzero_vals)
    if num_nonzero < 4:
        return survives, []
    survives["S0"] = True

    base_percentile = float(cfg.get("percentile", 97.5))
    if not disable_escalation and num_nonzero > 1000:
        actual_perc = min(99.0, base_percentile + (num_nonzero - 1000) / 500.0)
    else:
        actual_perc = base_percentile

    raw_thresh = float(np.percentile(nonzero_vals, actual_perc))
    thresh = max(1.0, raw_thresh)
    binary = (count_img >= thresh).astype(np.uint8)

    if cv2.countNonZero(binary) < 4:
        return survives, []
    survives["S1"] = True

    open_k = int(cfg.get("open_kernel", 1))
    if open_k > 1:
        k_open = cv2.getStructuringElement(cv2.MORPH_RECT, (open_k, open_k))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k_open)
        if cv2.countNonZero(binary) < 4:
            return survives, []

    dilate_k = int(cfg.get("dilate_kernel", 3))
    if dilate_k > 1:
        k_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (dilate_k, dilate_k))
        binary = cv2.dilate(binary, k_dilate)
        if cv2.countNonZero(binary) < 4:
            return survives, []
    survives["S2"] = True

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    if num_labels <= 1:
        return survives, []

    total_pixels = width * height
    max_area_pixels = int(total_pixels * float(cfg.get("max_area_frac", 0.05)))
    min_dim = int(cfg.get("min_dim", 4))
    max_dim = int(cfg.get("max_dim", 60))
    min_events = int(cfg.get("min_events_in_box", 4))
    centroid_mode = cfg.get("centroid_mode", "weighted")

    box_mode = "extent" if override_box_mode_extent else cfg.get("box_mode", "fixed")
    cfg_bw = float(cfg.get("box_w", 18))
    cfg_bh = float(cfg.get("box_h", 18))
    extent_scale = float(cfg.get("extent_scale", 1.0))
    extent_pad = float(cfg.get("extent_pad", 2.0))

    comps_s3 = []
    for label_idx in range(1, num_labels):
        area = int(stats[label_idx, cv2.CC_STAT_AREA])
        x_box = int(stats[label_idx, cv2.CC_STAT_LEFT])
        y_box = int(stats[label_idx, cv2.CC_STAT_TOP])
        w_box = int(stats[label_idx, cv2.CC_STAT_WIDTH])
        h_box = int(stats[label_idx, cv2.CC_STAT_HEIGHT])

        if area > max_area_pixels or w_box > max_dim or h_box > max_dim:
            # S3 area splitting
            sub_img = count_img[y_box : y_box + h_box, x_box : x_box + w_box]
            sub_labels = labels[y_box : y_box + h_box, x_box : x_box + w_box]
            sub_mask = sub_labels == label_idx
            sub_vals = sub_img[sub_mask]
            if len(sub_vals) >= 4:
                sub_thresh = max(thresh + 1.0, float(np.percentile(sub_vals, 75.0)))
                sub_bin = ((sub_img >= sub_thresh) & sub_mask).astype(np.uint8)
                if cv2.countNonZero(sub_bin) >= 4:
                    n_sub, _, sub_stats, sub_cents = cv2.connectedComponentsWithStats(
                        sub_bin, connectivity=8
                    )
                    for s_idx in range(1, n_sub):
                        s_area = int(sub_stats[s_idx, cv2.CC_STAT_AREA])
                        s_w = int(sub_stats[s_idx, cv2.CC_STAT_WIDTH])
                        s_h = int(sub_stats[s_idx, cv2.CC_STAT_HEIGHT])
                        if s_area > max_area_pixels or s_w > max_dim or s_h > max_dim:
                            continue
                        comps_s3.append((
                            x_box + int(sub_stats[s_idx, cv2.CC_STAT_LEFT]),
                            y_box + int(sub_stats[s_idx, cv2.CC_STAT_TOP]),
                            s_w,
                            s_h,
                            x_box + float(sub_cents[s_idx, 0]),
                            y_box + float(sub_cents[s_idx, 1]),
                            label_idx,
                        ))
        else:
            comps_s3.append((
                x_box,
                y_box,
                w_box,
                h_box,
                float(centroids[label_idx, 0]),
                float(centroids[label_idx, 1]),
                label_idx,
            ))

    if not comps_s3:
        return survives, []
    survives["S3"] = True

    comps_s4 = []
    for x_b, y_b, w_b, h_b, c_x, c_y, l_idx in comps_s3:
        if box_mode == "fixed":
            bw = cfg_bw
            bh = cfg_bh
        else:
            bw = max(float(min_dim), min(float(max_dim), w_b * extent_scale + extent_pad))
            bh = max(float(min_dim), min(float(max_dim), h_b * extent_scale + extent_pad))

        # Intensity weighting
        if centroid_mode == "weighted":
            sub_c = count_img[y_b : y_b + h_b, x_b : x_b + w_b]
            sub_l = labels[y_b : y_b + h_b, x_b : x_b + w_b]
            mask_c = (sub_l == l_idx) & (sub_c > 0)
            if np.any(mask_c):
                tot_w = float(np.sum(sub_c[mask_c]))
                if tot_w > 0:
                    yy, xx = np.indices(sub_c.shape)
                    c_x = x_b + float(np.sum(xx[mask_c] * sub_c[mask_c])) / tot_w
                    c_y = y_b + float(np.sum(yy[mask_c] * sub_c[mask_c])) / tot_w

        # Slice box and compute event count
        x1_box = max(0, min(width - 1, int(round(c_x - (bw - 1.0) / 2.0))))
        y1_box = max(0, min(height - 1, int(round(c_y - (bh - 1.0) / 2.0))))
        x2_box = max(0, min(width - 1, int(round(x1_box + bw - 1.0))))
        y2_box = max(0, min(height - 1, int(round(y1_box + bh - 1.0))))

        box_crop = count_img[y1_box : y2_box + 1, x1_box : x2_box + 1]
        ev_in_box = int(np.sum(box_crop))
        if ev_in_box < min_events:
            continue

        comps_s4.append({
            "center_x": c_x,
            "center_y": c_y,
            "width": bw,
            "height": bh,
            "events": ev_in_box,
        })

    if not comps_s4:
        return survives, []
    survives["S4"] = True

    # S5: Static mask suppression
    comps_s5 = []
    active_static_mask = None if disable_static_mask else static_mask
    for c in comps_s4:
        cx_i = int(round(c["center_x"]))
        cy_i = int(round(c["center_y"]))
        if (
            active_static_mask is not None
            and 0 <= cy_i < height
            and 0 <= cx_i < width
            and active_static_mask[cy_i, cx_i]
        ):
            continue
        comps_s5.append(c)

    if not comps_s5:
        return survives, []
    survives["S5"] = True

    return survives, comps_s5


def main():
    parser = argparse.ArgumentParser(description="Oracle recall ceiling and stage funnel analysis")
    parser.add_argument("--dataset-dir", type=Path, default=Path("../OrbitSight_Dataset"))
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--log", type=Path, default=Path("experiments/infer_final2.txt"))
    args = parser.parse_args()

    cfg_all = load_yaml_config(args.config)
    gt_files = sorted(list(args.dataset_dir.rglob("*_bb_windows_40ms.txt")))
    if not gt_files:
        raise FileNotFoundError(f"No GT files found in {args.dataset_dir}")

    # Parse static mask log lines from infer log
    log_suppression = {}
    if args.log.exists():
        try:
            txt = args.log.read_text(encoding="utf-16")
        except Exception:
            txt = args.log.read_text(encoding="utf-8", errors="ignore")

        patt = r"Processing sequence '([^']+)'[^\n]*\nstatic_mask:\s*(\d+)\s*pixels suppressed\s*\(([0-9.]+)%\s*of frame\)"
        for m in re.finditer(patt, txt):
            log_suppression[m.group(1)] = (int(m.group(2)), float(m.group(3)))

    part_a_rows = []
    part_c_rows = []
    part_d_rows = []

    # Sensor family data containers
    sensor_funnel = {
        "EVK4": {"total_gt": 0, "S0": 0, "S1": 0, "S2": 0, "S3": 0, "S4": 0, "S5": 0, "S6": 0, "S7": 0, "S_oracle": 0},
        "DVX": {"total_gt": 0, "S0": 0, "S1": 0, "S2": 0, "S3": 0, "S4": 0, "S5": 0, "S6": 0, "S7": 0, "S_oracle": 0},
        "DAVIS": {"total_gt": 0, "S0": 0, "S1": 0, "S2": 0, "S3": 0, "S4": 0, "S5": 0, "S6": 0, "S7": 0, "S_oracle": 0},
        "ALL": {"total_gt": 0, "S0": 0, "S1": 0, "S2": 0, "S3": 0, "S4": 0, "S5": 0, "S6": 0, "S7": 0, "S_oracle": 0},
    }

    split_funnel = {
        "train": {"total_gt": 0, "S0": 0, "S1": 0, "S2": 0, "S3": 0, "S4": 0, "S5": 0, "S6": 0, "S7": 0, "C1": 0, "C2": 0, "C3": 0},
        "test": {"total_gt": 0, "S0": 0, "S1": 0, "S2": 0, "S3": 0, "S4": 0, "S5": 0, "S6": 0, "S7": 0, "C1": 0, "C2": 0, "C3": 0},
        "all": {"total_gt": 0, "S0": 0, "S1": 0, "S2": 0, "S3": 0, "S4": 0, "S5": 0, "S6": 0, "S7": 0, "C1": 0, "C2": 0, "C3": 0},
    }

    gt_box_dims = {"EVK4": {"w": [], "h": []}, "DVX": {"w": [], "h": []}, "DAVIS": {"w": [], "h": []}}
    gt_perfect_iou_match = {"EVK4": [], "DVX": [], "DAVIS": []}

    for gtf in gt_files:
        seq_name = gtf.name.replace("_bb_windows_40ms.txt", "")
        split = "train" if "Training" in str(gtf) else "test"
        sensor = get_sensor_family(seq_name)

        npy_matches = list(gtf.parent.glob(f"{seq_name}_labeled_events.npy"))
        if not npy_matches:
            npy_matches = list(args.dataset_dir.rglob(f"{seq_name}_labeled_events.npy"))
        npy_path = npy_matches[0]

        events = load_events(npy_path)
        width, height = infer_resolution(seq_name, events[:, 0], events[:, 1])
        eff_cfg = resolve_effective_config(cfg_all, sensor)

        static_map = build_continuous_static_map(events, width, height, window_us=WINDOW_US)
        static_mask = static_map >= float(eff_cfg.get("static_thresh", 0.5))

        gt_rows = load_gt_rows(gtf)
        total_gt = len(gt_rows)

        # Index GT by window_start_timestamp_us
        gt_dict = {r[0]: (r[2], r[3], r[4], r[5]) for r in gt_rows}

        # Track GT box dimensions
        for r in gt_rows:
            w_val = r[4]
            h_val = r[5]
            gt_box_dims[sensor]["w"].append(w_val)
            gt_box_dims[sensor]["h"].append(h_val)

            # Check IoU with perfectly centred configured box
            cfg_bw = float(eff_cfg.get("box_w", 18))
            cfg_bh = float(eff_cfg.get("box_h", 18))
            perfect_iou = iou((0.0, 0.0, cfg_bw, cfg_bh), (0.0, 0.0, w_val, h_val))
            gt_perfect_iou_match[sensor].append(perfect_iou >= 0.5)

        # Count GT killed by static mask
        killed_by_static_count = 0
        for r in gt_rows:
            gx = int(round(r[2]))
            gy = int(round(r[3]))
            if 0 <= gy < height and 0 <= gx < width and static_mask[gy, gx]:
                killed_by_static_count += 1
        gt_killed_by_static_frac = killed_by_static_count / total_gt if total_gt > 0 else 0.0

        # Stage counters for this sequence
        s_cnt = {"S0": 0, "S1": 0, "S2": 0, "S3": 0, "S4": 0, "S5": 0, "S6": 0, "S7": 0, "S_oracle": 0}
        c_cnt = {"C1": 0, "C2": 0, "C3": 0}

        for ws, we, w_events in iter_windows(events, window_us=WINDOW_US):
            if ws not in gt_dict:
                continue

            gt_cx, gt_cy, gt_w, gt_h = gt_dict[ws]
            gt_box = (gt_cx, gt_cy, gt_w, gt_h)

            count_img, _, _ = event_image(w_events, width, height, need_polarity=False)

            # Baseline funnel
            surv, cands_s5 = detect_stage_funnel_and_candidates(
                count_img, width, height, eff_cfg, static_mask=static_mask
            )

            if surv["S0"]: s_cnt["S0"] += 1
            if surv["S1"]: s_cnt["S1"] += 1
            if surv["S2"]: s_cnt["S2"] += 1
            if surv["S3"]: s_cnt["S3"] += 1
            if surv["S4"]: s_cnt["S4"] += 1
            if surv["S5"]:
                s_cnt["S5"] += 1

                # S6: IoU >= 0.5 with GT
                best_iou = 0.0
                best_dist = float("inf")
                best_oracle_iou = 0.0

                for cand in cands_s5:
                    c_box = (cand["center_x"], cand["center_y"], cand["width"], cand["height"])
                    sc = iou(c_box, gt_box)
                    if sc > best_iou:
                        best_iou = sc

                    dist = math.hypot(cand["center_x"] - gt_cx, cand["center_y"] - gt_cy)
                    if dist < best_dist:
                        best_dist = dist

                    # Oracle box size (candidate center + GT width/height)
                    oracle_box = (cand["center_x"], cand["center_y"], gt_w, gt_h)
                    sc_oracle = iou(oracle_box, gt_box)
                    if sc_oracle > best_oracle_iou:
                        best_oracle_iou = sc_oracle

                if best_iou >= 0.5:
                    s_cnt["S6"] += 1
                if best_dist <= 3.0:
                    s_cnt["S7"] += 1
                if best_oracle_iou >= 0.5:
                    s_cnt["S_oracle"] += 1

            # Counterfactual C1: static_thresh disabled
            _, cands_c1 = detect_stage_funnel_and_candidates(
                count_img, width, height, eff_cfg, static_mask=None, disable_static_mask=True
            )
            if any(iou((c["center_x"], c["center_y"], c["width"], c["height"]), gt_box) >= 0.5 for c in cands_c1):
                c_cnt["C1"] += 1

            # Counterfactual C2: escalation disabled
            _, cands_c2 = detect_stage_funnel_and_candidates(
                count_img, width, height, eff_cfg, static_mask=static_mask, disable_escalation=True
            )
            if any(iou((c["center_x"], c["center_y"], c["width"], c["height"]), gt_box) >= 0.5 for c in cands_c2):
                c_cnt["C2"] += 1

            # Counterfactual C3: box_mode = "extent"
            _, cands_c3 = detect_stage_funnel_and_candidates(
                count_img, width, height, eff_cfg, static_mask=static_mask, override_box_mode_extent=True
            )
            if any(iou((c["center_x"], c["center_y"], c["width"], c["height"]), gt_box) >= 0.5 for c in cands_c3):
                c_cnt["C3"] += 1

        # Accumulate sensor funnel
        sensor_funnel[sensor]["total_gt"] += total_gt
        for k in s_cnt:
            sensor_funnel[sensor][k] += s_cnt[k]
            sensor_funnel["ALL"][k] += s_cnt[k]
        sensor_funnel["ALL"]["total_gt"] += total_gt

        # Accumulate split funnel
        split_funnel[split]["total_gt"] += total_gt
        split_funnel["all"]["total_gt"] += total_gt
        for k in s_cnt:
            if k in split_funnel[split]:
                split_funnel[split][k] += s_cnt[k]
                split_funnel["all"][k] += s_cnt[k]
        for k in c_cnt:
            split_funnel[split][k] += c_cnt[k]
            split_funnel["all"][k] += c_cnt[k]

        r_candgen = s_cnt["S6"] / total_gt if total_gt > 0 else 0.0
        r_local = s_cnt["S7"] / total_gt if total_gt > 0 else 0.0
        r_oracle = s_cnt["S_oracle"] / total_gt if total_gt > 0 else 0.0
        r_c1 = c_cnt["C1"] / total_gt if total_gt > 0 else 0.0
        r_c2 = c_cnt["C2"] / total_gt if total_gt > 0 else 0.0
        r_c3 = c_cnt["C3"] / total_gt if total_gt > 0 else 0.0

        part_a_rows.append([
            seq_name,
            sensor,
            split,
            total_gt,
            s_cnt["S0"],
            s_cnt["S1"],
            s_cnt["S2"],
            s_cnt["S3"],
            s_cnt["S4"],
            s_cnt["S5"],
            s_cnt["S6"],
            s_cnt["S7"],
            f"{r_candgen:.4f}",
            f"{r_local:.4f}",
        ])

        part_c_rows.append([
            seq_name,
            sensor,
            split,
            f"{r_candgen:.4f}",
            f"{r_c1:.4f} ({(r_c1 - r_candgen):+.4f})",
            f"{r_c2:.4f} ({(r_c2 - r_candgen):+.4f})",
            f"{r_c3:.4f} ({(r_c3 - r_candgen):+.4f})",
        ])

        log_px, log_pct = log_suppression.get(seq_name, (0, 0.0))
        part_d_rows.append([
            seq_name,
            sensor,
            split,
            f"{log_px} ({log_pct:.2f}%)",
            f"{killed_by_static_count}/{total_gt} ({gt_killed_by_static_frac:.4f})",
            f"{r_candgen:.4f}",
        ])

    # ── Print Part A ─────────────────────────────────────────────────────────
    print("\n" + "=" * 150)
    print("  PART A — STAGE SURVIVAL FUNNEL ACROSS GT-OCCUPIED WINDOWS")
    print("=" * 150)
    headers_a = [
        "Sequence", "Sensor", "Split", "GT Total",
        "S0 (>=4px)", "S1 (Thresh)", "S2 (Morph)", "S3 (Area)", "S4 (MinEv)", "S5 (Static)",
        "S6 (IoU>=0.5)", "S7 (Dist<=3px)", "Ceil CANDGEN", "Ceil LOCAL"
    ]
    print(tabulate(part_a_rows, headers=headers_a, tablefmt="github"))

    # Part A Sensor Family Funnel Summary & Largest Drop
    sensor_drop_summary = []
    for s_fam in ["EVK4", "DVX", "DAVIS", "ALL"]:
        tot = sensor_funnel[s_fam]["total_gt"]
        s0 = sensor_funnel[s_fam]["S0"]
        s1 = sensor_funnel[s_fam]["S1"]
        s2 = sensor_funnel[s_fam]["S2"]
        s3 = sensor_funnel[s_fam]["S3"]
        s4 = sensor_funnel[s_fam]["S4"]
        s5 = sensor_funnel[s_fam]["S5"]
        s6 = sensor_funnel[s_fam]["S6"]
        s7 = sensor_funnel[s_fam]["S7"]

        drops = {
            "S0_drop (Sparse <4px)": tot - s0,
            "S1_drop (Perc Thresh)": s0 - s1,
            "S2_drop (Morphology)": s1 - s2,
            "S3_drop (Max Area)": s2 - s3,
            "S4_drop (Min Events)": s3 - s4,
            "S5_drop (Static Mask)": s4 - s5,
            "S6_drop (Box/IoU <0.5)": s5 - s6,
        }
        max_stage = max(drops.items(), key=lambda x: x[1])

        sensor_drop_summary.append([
            s_fam,
            tot,
            f"{s6 / tot:.4f}",
            f"{s7 / tot:.4f}",
            f"{max_stage[0]} ({max_stage[1]} lost, {max_stage[1] / tot:.2%})",
        ])

    print("\n" + "=" * 110)
    print("  PART A — LARGEST ABSOLUTE RECALL LOSS STAGE PER SENSOR FAMILY")
    print("=" * 110)
    print(tabulate(sensor_drop_summary, headers=["Sensor Family", "Total GT", "Ceil CANDGEN (S6)", "Ceil LOCAL (S7)", "Largest Loss Stage"], tablefmt="github"))

    # ── Print Part B ─────────────────────────────────────────────────────────
    print("\n" + "=" * 110)
    print("  PART B — ORACLE BOX GEOMETRY & SIZING LOSS")
    print("=" * 110)
    oracle_summary = []
    for s_fam in ["EVK4", "DVX", "DAVIS", "ALL"]:
        tot = sensor_funnel[s_fam]["total_gt"]
        s6 = sensor_funnel[s_fam]["S6"]
        s_orc = sensor_funnel[s_fam]["S_oracle"]
        r_cand = s6 / tot
        r_orc = s_orc / tot
        delta_sizing = r_orc - r_cand

        oracle_summary.append([
            s_fam,
            tot,
            f"{r_cand:.4f}",
            f"{r_orc:.4f}",
            f"{delta_sizing:+.4f} ({(delta_sizing * tot):.0f} boxes)",
        ])
    print(tabulate(oracle_summary, headers=["Sensor Family", "Total GT", "Ceil CANDGEN (S6)", "Ceil ORACLE_BOX", "Recall Lost to Sizing"], tablefmt="github"))

    # Part B Box dimension percentiles
    dim_rows = []
    for s_fam in ["EVK4", "DVX", "DAVIS"]:
        w_arr = np.array(gt_box_dims[s_fam]["w"])
        h_arr = np.array(gt_box_dims[s_fam]["h"])
        perf_arr = np.array(gt_perfect_iou_match[s_fam])

        w_p = np.percentile(w_arr, [0, 5, 25, 50, 75, 95, 100])
        h_p = np.percentile(h_arr, [0, 5, 25, 50, 75, 95, 100])
        frac_match = np.mean(perf_arr)

        dim_rows.append([
            f"{s_fam} Width",
            f"{w_p[0]:.1f}", f"{w_p[1]:.1f}", f"{w_p[2]:.1f}", f"{w_p[3]:.1f}", f"{w_p[4]:.1f}", f"{w_p[5]:.1f}", f"{w_p[6]:.1f}",
            f"{frac_match:.4f} ({frac_match:.1%})"
        ])
        dim_rows.append([
            f"{s_fam} Height",
            f"{h_p[0]:.1f}", f"{h_p[1]:.1f}", f"{h_p[2]:.1f}", f"{h_p[3]:.1f}", f"{h_p[4]:.1f}", f"{h_p[5]:.1f}", f"{h_p[6]:.1f}",
            "—"
        ])

    print("\n" + "=" * 110)
    print("  PART B — GT BOX DIMENSION DISTRIBUTIONS & FIXED-BOX COMPATIBILITY")
    print("=" * 110)
    print(tabulate(dim_rows, headers=["Dimension", "Min", "P5", "P25", "P50 (Med)", "P75", "P95", "Max", "Fixed-Box IoU>=0.5 Frac"], tablefmt="github"))

    # ── Print Part C ─────────────────────────────────────────────────────────
    print("\n" + "=" * 110)
    print("  PART C — COUNTERFACTUAL RECALL ANALYSIS (CANDGEN / S6 RECALL)")
    print("=" * 110)
    print(tabulate(part_c_rows, headers=["Sequence", "Sensor", "Split", "Baseline S6", "C1 (No Static)", "C2 (No Escalation)", "C3 (Extent Mode)"], tablefmt="github"))

    # Split aggregate summary for Part C
    split_c_summary = []
    for sp in ["train", "test", "all"]:
        tot = split_funnel[sp]["total_gt"]
        s6 = split_funnel[sp]["S6"]
        c1 = split_funnel[sp]["C1"]
        c2 = split_funnel[sp]["C2"]
        c3 = split_funnel[sp]["C3"]

        r_base = s6 / tot
        r_c1 = c1 / tot
        r_c2 = c2 / tot
        r_c3 = c3 / tot

        split_c_summary.append([
            sp.upper(),
            tot,
            f"{r_base:.4f}",
            f"{r_c1:.4f} ({(r_c1 - r_base):+.4f})",
            f"{r_c2:.4f} ({(r_c2 - r_base):+.4f})",
            f"{r_c3:.4f} ({(r_c3 - r_base):+.4f})",
        ])
    print("\n" + "=" * 90)
    print("  PART C — SPLIT AGGREGATE COUNTERFACTUAL CEILINGS")
    print("=" * 90)
    print(tabulate(split_c_summary, headers=["Split", "Total GT", "Baseline S6", "C1 (No Static)", "C2 (No Escalation)", "C3 (Extent Mode)"], tablefmt="github"))

    # ── Print Part D ─────────────────────────────────────────────────────────
    print("\n" + "=" * 110)
    print("  PART D — STATIC SUPPRESSION AUDIT & GT KILLED BY STATIC MASK")
    print("=" * 110)
    print(tabulate(part_d_rows, headers=["Sequence", "Sensor", "Split", "Static Suppressed (Frame %)", "GT Killed by Static (Frac)", "Baseline S6 Ceil"], tablefmt="github"))
    print()


if __name__ == "__main__":
    main()
