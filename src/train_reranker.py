"""Training, feature extraction, and evaluation of Variant A (Track-based reranker) and Variant B (Window Objectness Gate)."""

import argparse
import math
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from tabulate import tabulate

import csv
from src.features import FEATURE_NAMES, extract_local_bg, extract_window_features_batch
from src.metrics import evaluate_sequence, match_predictions, compute_prf1, compute_ap
from src.pipeline import WINDOW_US, event_image, iter_windows, resolve_effective_config
from src.scoreboard import compute_config_hash, infer_resolution, load_events, load_yaml_config
from src.tracker import TRACK_FEATURE_NAMES, build_sequence_tracks


def load_gt_file(gt_file: Path) -> List[Tuple[int, int, int, int, int, int]]:
    gt_rows = []
    with open(gt_file, "r", encoding="utf-8") as f:
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
    return gt_rows


ALL_TRACK_RERANKER_FEATS = FEATURE_NAMES + TRACK_FEATURE_NAMES

WINDOW_OBJECTNESS_FEATS = [
    "win_total_events",
    "win_num_components",
    "win_x_std",
    "win_y_std",
    "win_max_comp_events",
    "win_cand_score_mean",
    "win_cand_score_max",
    "prev_win_total_events",
    "prev_win_num_components",
    "prev_win_x_std",
    "prev_win_y_std",
    "prev_win_max_comp_events",
    "prev_win_cand_score_mean",
    "prev_win_cand_score_max",
    "next_win_total_events",
    "next_win_num_components",
    "next_win_x_std",
    "next_win_y_std",
    "next_win_max_comp_events",
    "next_win_cand_score_mean",
    "next_win_cand_score_max",
]

VAL_SEQS = [
    "DAVIS_SL8RB_2025-01-13-19-15-36",
    "DVX_Filtered_BlockDM_SLRB_32405_2025-01-20-19-57-17",
]


def extract_raw_sequence_cands_and_windows(
    events: np.ndarray,
    width: int,
    height: int,
    cfg: Dict[str, Any],
    learned_scorer: Any,
    window_us: int = WINDOW_US,
) -> Tuple[List[List[Dict[str, Any]]], List[Dict[str, float]], List[Tuple[int, int]], List[float]]:
    """Extract candidate boxes, their 13 base features, window-level features, and per-window latency."""
    if width >= 1200:
        sensor_name = "EVK4"
    elif width >= 600:
        sensor_name = "DVX"
    else:
        sensor_name = "DAVIS"

    eff = resolve_effective_config(cfg, sensor_name)
    from src.static_map import build_continuous_static_map
    from src.detector import detect_boxes

    static_frac_map = build_continuous_static_map(events, width, height, window_us=window_us)
    static_thresh = eff.get("static_thresh", None)
    if static_thresh is not None:
        static_mask = static_frac_map >= float(static_thresh)
    else:
        static_mask = None

    raw_window_records = []
    window_times = []
    base_window_stats = []
    win_latencies: List[float] = []

    for w_start, w_end, w_events in iter_windows(events, window_us=window_us):
        t0_win = time.perf_counter()
        count_img, _, _ = event_image(w_events, width, height, need_polarity=False)
        boxes = detect_boxes(count_img, width, height, cfg)
        if static_mask is not None and boxes:
            boxes = [
                b for b in boxes
                if not (0 <= int(round(b["center_y"])) < height and 0 <= int(round(b["center_x"])) < width and static_mask[int(round(b["center_y"])), int(round(b["center_x"]))])
            ]
        for b in boxes:
            b["local_bg"] = extract_local_bg(
                count_img, float(b["center_x"]), float(b["center_y"]), float(b["width"]), float(b["height"])
            )

        # Window level stats
        tot_ev = float(len(w_events))
        n_comp = float(len(boxes))
        if tot_ev > 0:
            x_std = float(np.std(w_events[:, 0]))
            y_std = float(np.std(w_events[:, 1]))
        else:
            x_std, y_std = 0.0, 0.0
        max_comp_ev = float(max([b.get("events", 0.0) for b in boxes])) if boxes else 0.0

        t_elapsed_ms = (time.perf_counter() - t0_win) * 1000.0
        win_latencies.append(t_elapsed_ms)

        base_window_stats.append({
            "win_total_events": tot_ev,
            "win_num_components": n_comp,
            "win_x_std": x_std,
            "win_y_std": y_std,
            "win_max_comp_events": max_comp_ev,
        })
        raw_window_records.append((w_start, w_end, boxes))
        window_times.append((w_start, w_end))

    num_windows = len(raw_window_records)
    min_hits = int(eff.get("min_hits", 1))
    max_dist_frac = float(eff.get("max_dist_frac", 0.08))
    diagonal = math.hypot(width, height)
    max_dist = max_dist_frac * diagonal
    max_dist_sq = max_dist * max_dist

    window_candidates: List[List[Dict[str, Any]]] = []

    for w_idx in range(num_windows):
        w_start, w_end, boxes = raw_window_records[w_idx]
        if not boxes:
            window_candidates.append([])
            continue

        prev_boxes = raw_window_records[w_idx - 1][2] if w_idx > 0 else []
        next_boxes = raw_window_records[w_idx + 1][2] if w_idx < num_windows - 1 else []

        n_cur = len(boxes)
        cur_centers = np.array([[b["center_x"], b["center_y"]] for b in boxes], dtype=np.float32)

        has_prev = np.zeros(n_cur, dtype=bool)
        if prev_boxes:
            p_centers = np.array([[p["center_x"], p["center_y"]] for p in prev_boxes], dtype=np.float32)
            diff_p = cur_centers[:, None, :] - p_centers[None, :, :]
            dist_sq_p = np.sum(diff_p * diff_p, axis=2)
            has_prev = np.any(dist_sq_p <= max_dist_sq, axis=1)

        has_next = np.zeros(n_cur, dtype=bool)
        if next_boxes:
            n_centers = np.array([[n["center_x"], n["center_y"]] for n in next_boxes], dtype=np.float32)
            diff_n = cur_centers[:, None, :] - n_centers[None, :, :]
            dist_sq_n = np.sum(diff_n * diff_n, axis=2)
            has_next = np.any(dist_sq_n <= max_dist_sq, axis=1)

        cand_boxes_list = []
        for idx, box in enumerate(boxes):
            hits = 1 + int(has_prev[idx]) + int(has_next[idx])
            if min_hits >= 2 and hits < min_hits:
                continue
            box_copy = dict(box)
            box_copy["hits"] = hits
            box_copy["w_idx"] = w_idx
            cand_boxes_list.append(box_copy)

        if cand_boxes_list and learned_scorer is not None:
            batch_feats = extract_window_features_batch(
                cand_boxes_list,
                prev_boxes,
                next_boxes,
                count_img=None,
                static_frac_map=static_frac_map,
            )
            cand_feat_vecs = [
                [f[name] for name in FEATURE_NAMES] for f in batch_feats
            ]
            X_cand = np.array(cand_feat_vecs, dtype=np.float32)
            probs = learned_scorer.predict_proba(X_cand)[:, 1]
            for b_copy, f_dict, p_score in zip(cand_boxes_list, batch_feats, probs):
                b_copy["confidence"] = float(p_score)
                b_copy["features_13"] = [f_dict[name] for name in FEATURE_NAMES]
            window_candidates.append(cand_boxes_list)
        else:
            window_candidates.append([])

    # Compute complete 21-dim window features with candidate score aggregates
    window_features_21: List[Dict[str, float]] = []
    for w_idx in range(num_windows):
        cands = window_candidates[w_idx]
        scores = [c["confidence"] for c in cands] if cands else []
        c_mean = float(np.mean(scores)) if scores else 0.0
        c_max = float(np.max(scores)) if scores else 0.0

        base_stat = dict(base_window_stats[w_idx])
        base_stat["win_cand_score_mean"] = c_mean
        base_stat["win_cand_score_max"] = c_max
        window_features_21.append(base_stat)

    full_window_features: List[Dict[str, float]] = []
    for w_idx in range(num_windows):
        cur = window_features_21[w_idx]
        prev = window_features_21[w_idx - 1] if w_idx > 0 else {k: 0.0 for k in cur}
        nxt = window_features_21[w_idx + 1] if w_idx < num_windows - 1 else {k: 0.0 for k in cur}

        w_dict = {
            "win_total_events": cur["win_total_events"],
            "win_num_components": cur["win_num_components"],
            "win_x_std": cur["win_x_std"],
            "win_y_std": cur["win_y_std"],
            "win_max_comp_events": cur["win_max_comp_events"],
            "win_cand_score_mean": cur["win_cand_score_mean"],
            "win_cand_score_max": cur["win_cand_score_max"],
            "prev_win_total_events": prev["win_total_events"],
            "prev_win_num_components": prev["win_num_components"],
            "prev_win_x_std": prev["win_x_std"],
            "prev_win_y_std": prev["win_y_std"],
            "prev_win_max_comp_events": prev["win_max_comp_events"],
            "prev_win_cand_score_mean": prev["win_cand_score_mean"],
            "prev_win_cand_score_max": prev["win_cand_score_max"],
            "next_win_total_events": nxt["win_total_events"],
            "next_win_num_components": nxt["win_num_components"],
            "next_win_x_std": nxt["win_x_std"],
            "next_win_y_std": nxt["win_y_std"],
            "next_win_max_comp_events": nxt["win_max_comp_events"],
            "next_win_cand_score_mean": nxt["win_cand_score_mean"],
            "next_win_cand_score_max": nxt["win_cand_score_max"],
        }
        full_window_features.append(w_dict)

    return window_candidates, full_window_features, window_times, win_latencies


def compute_candidate_ground_truth_labels(
    window_candidates: List[List[Dict[str, Any]]],
    window_times: List[Tuple[int, int]],
    gt_rows: List[Tuple[int, int, int, int, int, int]],
    iou_thr: float = 0.5,
) -> Tuple[List[int], List[int]]:
    """Compute IoU >= 0.5 label for every candidate in sequence, and return GT-occupied window indicator."""
    from src.metrics import iou
    gt_dict = {}
    for g in gt_rows:
        w_st = g[0]
        gt_dict.setdefault(w_st, []).append(g)

    cand_labels = []
    win_occupied = []

    for w_idx, (w_st, w_end) in enumerate(window_times):
        cands = window_candidates[w_idx]
        gts = gt_dict.get(w_st, [])
        win_occupied.append(1 if len(gts) > 0 else 0)

        for c in cands:
            c_box = (c["center_x"], c["center_y"], c["width"], c["height"])
            best_iou = 0.0
            for g in gts:
                g_box = (g[2], g[3], g[4], g[5])
                val_iou = iou(c_box, g_box)
                if val_iou > best_iou:
                    best_iou = val_iou
            cand_labels.append(1 if best_iou >= iou_thr else 0)

    return cand_labels, win_occupied


def get_cand_track_feat(c: Dict[str, Any], w_idx: int, cand_track_map: Dict[Tuple[int, int], List[float]]) -> List[float]:
    cand_id = (w_idx, id(c))
    if cand_id in cand_track_map:
        return cand_track_map[cand_id]
    conf = float(c.get("confidence", 0.0))
    ev = float(c.get("events", c.get("event_count", 0.0)))
    return [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, conf, conf, conf, ev]


def assemble_track_dataset(
    seq_data_dict: Dict[str, Any],
    max_gap: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, List[float]]:
    """Build track-enhanced candidate feature matrices for train and validation splits."""
    X_train_list, y_train_list = [], []
    X_val_list, y_val_list = [], []
    singleton_count = 0
    singleton_scores = []

    for seq_name, data in seq_data_dict.items():
        is_val = seq_name in VAL_SEQS
        window_cands = data["window_cands"]
        gt_rows = data["gt_rows"]
        window_times = data["window_times"]

        # Run multi-window tracker on the sequence
        tracks = build_sequence_tracks(window_cands, max_gap=max_gap, max_speed_per_win=30.0)

        # Build candidate-to-track lookup
        cand_track_map = {}
        for trk in tracks:
            trk_feats = trk.compute_features()
            trk_feat_vec = [trk_feats[name] for name in TRACK_FEATURE_NAMES]
            if len(trk.history) == 1:
                singleton_count += 1
                singleton_scores.append(float(trk.history[0][1]["confidence"]))

            for w_idx, c in trk.history:
                cand_id = (w_idx, id(c))
                cand_track_map[cand_id] = trk_feat_vec

        # Extract labeled feature vectors
        from src.metrics import iou
        gt_dict = {}
        for g in gt_rows:
            gt_dict.setdefault(g[0], []).append(g)

        for w_idx, (w_st, w_end) in enumerate(window_times):
            cands = window_cands[w_idx]
            gts = gt_dict.get(w_st, [])

            for c in cands:
                trk_feat_vec = get_cand_track_feat(c, w_idx, cand_track_map)
                c_feat_vec = c["features_13"] + trk_feat_vec

                # Compute IoU label
                c_box = (c["center_x"], c["center_y"], c["width"], c["height"])
                best_iou = 0.0
                for g in gts:
                    g_box = (g[2], g[3], g[4], g[5])
                    val_iou = iou(c_box, g_box)
                    if val_iou > best_iou:
                        best_iou = val_iou
                label = 1 if best_iou >= 0.5 else 0

                if is_val:
                    X_val_list.append(c_feat_vec)
                    y_val_list.append(label)
                else:
                    X_train_list.append(c_feat_vec)
                    y_train_list.append(label)

    return (
        np.array(X_train_list, dtype=np.float32),
        np.array(y_train_list, dtype=np.int32),
        np.array(X_val_list, dtype=np.float32),
        np.array(y_val_list, dtype=np.int32),
        singleton_count,
        singleton_scores,
    )


def assemble_objectness_dataset(
    seq_data_dict: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build window-level objectness dataset."""
    X_train_list, y_train_list = [], []
    X_val_list, y_val_list = [], []

    for seq_name, data in seq_data_dict.items():
        is_val = seq_name in VAL_SEQS
        full_win_feats = data["full_win_feats"]
        gt_rows = data["gt_rows"]
        window_times = data["window_times"]

        gt_dict = {}
        for g in gt_rows:
            gt_dict.setdefault(g[0], []).append(g)

        for w_idx, (w_st, w_end) in enumerate(window_times):
            w_feats = full_win_feats[w_idx]
            w_vec = [w_feats[k] for k in WINDOW_OBJECTNESS_FEATS]
            label = 1 if len(gt_dict.get(w_st, [])) > 0 else 0

            if is_val:
                X_val_list.append(w_vec)
                y_val_list.append(label)
            else:
                X_train_list.append(w_vec)
                y_train_list.append(label)

    return (
        np.array(X_train_list, dtype=np.float32),
        np.array(y_train_list, dtype=np.int32),
        np.array(X_val_list, dtype=np.float32),
        np.array(y_val_list, dtype=np.int32),
    )


def evaluate_variant_on_sequence(
    window_cands: List[List[Dict[str, Any]]],
    full_win_feats: List[Dict[str, float]],
    window_times: List[Tuple[int, int]],
    gt_rows: List[Tuple[int, int, int, int, int, int]],
    mode: str,
    reranker_model: Optional[Any] = None,
    objectness_model: Optional[Any] = None,
    max_gap: int = 2,
    conf_min: float = 0.30,
    max_k: int = 1,
) -> Tuple[Dict[str, Any], float]:
    """Evaluate a specific variant (baseline, variant_a, variant_b, variant_ab) on a sequence with timing."""
    start_t = time.perf_counter()

    num_windows = len(window_times)

    if mode == "baseline":
        # Baseline uses raw c["confidence"]
        seq_preds = []
        for w_idx in range(num_windows):
            cands = window_cands[w_idx]
            if not cands:
                continue
            scored = [c for c in cands if c["confidence"] >= conf_min]
            if not scored:
                continue
            if max_k is not None and len(scored) > max_k:
                scored.sort(key=lambda b: b["confidence"], reverse=True)
                scored = scored[:max_k]
            w_st, w_end = window_times[w_idx]
            for b in scored:
                seq_preds.append((
                    w_st, w_end,
                    int(round(b["center_x"])),
                    int(round(b["center_y"])),
                    int(round(b["width"])),
                    int(round(b["height"])),
                    round(float(b["confidence"]), 4),
                ))

    elif mode == "variant_a":
        # Track-based reranker
        tracks = build_sequence_tracks(window_cands, max_gap=max_gap, max_speed_per_win=30.0)
        cand_track_map = {}
        for trk in tracks:
            trk_feats = trk.compute_features()
            trk_feat_vec = [trk_feats[name] for name in TRACK_FEATURE_NAMES]
            for w_idx, c in trk.history:
                cand_track_map[(w_idx, id(c))] = trk_feat_vec

        # Batch predict all candidates in the sequence
        all_cands_list = []
        for w_idx in range(num_windows):
            for c in window_cands[w_idx]:
                all_cands_list.append((w_idx, c))

        if all_cands_list:
            X_all = np.array([
                c["features_13"] + get_cand_track_feat(c, w_idx, cand_track_map)
                for w_idx, c in all_cands_list
            ], dtype=np.float32)
            all_scores = reranker_model.predict_proba(X_all)[:, 1]
        else:
            all_scores = []

        rescored_win_cands = [[] for _ in range(num_windows)]
        for idx, (w_idx, c) in enumerate(all_cands_list):
            s = float(all_scores[idx])
            if s >= conf_min:
                c_copy = dict(c)
                c_copy["confidence"] = s
                rescored_win_cands[w_idx].append(c_copy)

        seq_preds = []
        for w_idx in range(num_windows):
            scored = rescored_win_cands[w_idx]
            if not scored:
                continue
            if max_k is not None and len(scored) > max_k:
                scored.sort(key=lambda b: b["confidence"], reverse=True)
                scored = scored[:max_k]
            w_st, w_end = window_times[w_idx]
            for b in scored:
                seq_preds.append((
                    w_st, w_end,
                    int(round(b["center_x"])),
                    int(round(b["center_y"])),
                    int(round(b["width"])),
                    int(round(b["height"])),
                    round(float(b["confidence"]), 4),
                ))

    elif mode == "variant_b":
        # Window-level objectness gate
        X_win = np.array([
            [full_win_feats[w_idx][k] for k in WINDOW_OBJECTNESS_FEATS]
            for w_idx in range(num_windows)
        ], dtype=np.float32)
        win_probs = objectness_model.predict_proba(X_win)[:, 1]

        seq_preds = []
        for w_idx in range(num_windows):
            cands = window_cands[w_idx]
            if not cands:
                continue
            p_obj = float(win_probs[w_idx])
            scored = []
            for c in cands:
                gated_score = float(c["confidence"]) * p_obj
                if gated_score >= conf_min:
                    c_copy = dict(c)
                    c_copy["confidence"] = gated_score
                    scored.append(c_copy)
            if not scored:
                continue
            if max_k is not None and len(scored) > max_k:
                scored.sort(key=lambda b: b["confidence"], reverse=True)
                scored = scored[:max_k]
            w_st, w_end = window_times[w_idx]
            for b in scored:
                seq_preds.append((
                    w_st, w_end,
                    int(round(b["center_x"])),
                    int(round(b["center_y"])),
                    int(round(b["width"])),
                    int(round(b["height"])),
                    round(float(b["confidence"]), 4),
                ))

    elif mode == "variant_ab":
        # Combined Track Reranker x Window Objectness Gate
        tracks = build_sequence_tracks(window_cands, max_gap=max_gap, max_speed_per_win=30.0)
        cand_track_map = {}
        for trk in tracks:
            trk_feats = trk.compute_features()
            trk_feat_vec = [trk_feats[name] for name in TRACK_FEATURE_NAMES]
            for w_idx, c in trk.history:
                cand_track_map[(w_idx, id(c))] = trk_feat_vec

        X_win = np.array([
            [full_win_feats[w_idx][k] for k in WINDOW_OBJECTNESS_FEATS]
            for w_idx in range(num_windows)
        ], dtype=np.float32)
        win_probs = objectness_model.predict_proba(X_win)[:, 1]

        all_cands_list = []
        for w_idx in range(num_windows):
            for c in window_cands[w_idx]:
                all_cands_list.append((w_idx, c))

        if all_cands_list:
            X_all = np.array([
                c["features_13"] + get_cand_track_feat(c, w_idx, cand_track_map)
                for w_idx, c in all_cands_list
            ], dtype=np.float32)
            all_scores = reranker_model.predict_proba(X_all)[:, 1]
        else:
            all_scores = []

        rescored_win_cands = [[] for _ in range(num_windows)]
        for idx, (w_idx, c) in enumerate(all_cands_list):
            p_obj = float(win_probs[w_idx])
            s = float(all_scores[idx]) * p_obj
            if s >= conf_min:
                c_copy = dict(c)
                c_copy["confidence"] = s
                rescored_win_cands[w_idx].append(c_copy)

        seq_preds = []
        for w_idx in range(num_windows):
            scored = rescored_win_cands[w_idx]
            if not scored:
                continue
            if max_k is not None and len(scored) > max_k:
                scored.sort(key=lambda b: b["confidence"], reverse=True)
                scored = scored[:max_k]
            w_st, w_end = window_times[w_idx]
            for b in scored:
                seq_preds.append((
                    w_st, w_end,
                    int(round(b["center_x"])),
                    int(round(b["center_y"])),
                    int(round(b["width"])),
                    int(round(b["height"])),
                    round(float(b["confidence"]), 4),
                ))

    eval_res = evaluate_sequence(gt_rows, seq_preds)
    elapsed_ms = (time.perf_counter() - start_t) * 1000.0
    ms_per_win = elapsed_ms / num_windows if num_windows > 0 else 0.0

    eval_res["n_pred"] = len(seq_preds)
    eval_res["ms_per_win"] = ms_per_win
    eval_res["preds"] = seq_preds
    return eval_res, ms_per_win


def main():
    parser = argparse.ArgumentParser(description="Multi-window tracker and window objectness reranking.")
    parser.add_argument("--dataset-dir", type=str, default="../OrbitSight_Dataset", help="Path to dataset directory")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--scorer-model", type=str, default="", help="Path to base candidate scorer model")
    parser.add_argument("--save-preds-dir", type=str, default="", help="Directory to save train predictions of Variant B")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    cfg = load_yaml_config(cfg_path)
    dataset_dir = Path(args.dataset_dir).resolve()

    if args.scorer_model:
        scorer_path = Path(args.scorer_model)
    elif "pre_geom" in cfg_path.stem and Path("models/scorer_pregeom.joblib").exists():
        scorer_path = Path("models/scorer_pregeom.joblib")
    else:
        scorer_path = Path("models/scorer.joblib")

    learned_scorer = joblib.load(scorer_path)
    print(f"[INFO] Loaded baseline learned scorer: {scorer_path}", flush=True)

    # Find train GT files
    all_gt_files = sorted(list(dataset_dir.rglob("*_bb_windows_40ms.txt")))
    train_gt_files = [f for f in all_gt_files if "Training" in str(f)]
    print(f"[INFO] Found {len(train_gt_files)} training sequence ground truth files.")

    # 1. Extract raw candidates and window stats for all 17 train sequences
    cache_path = Path(f"models/train_seq_extracted_cache_{cfg_path.stem}.joblib")
    if not cache_path.exists() and cfg_path.stem == "config" and Path("models/train_seq_extracted_cache.joblib").exists():
        cache_path = Path("models/train_seq_extracted_cache.joblib")
    if cache_path.exists():
        print(f"[INFO] Loading cached train sequence features from {cache_path}...", flush=True)
        seq_data_dict = joblib.load(cache_path)
    else:
        seq_data_dict = {}
        total_extract_time = 0.0

        for idx, gt_f in enumerate(train_gt_files, 1):
            seq_name = gt_f.name.replace("_bb_windows_40ms.txt", "")
            gt_rows = load_gt_file(gt_f)
            npy_matches = list(gt_f.parent.glob(f"{seq_name}_labeled_events.npy"))
            if not npy_matches:
                npy_matches = list(dataset_dir.rglob(f"{seq_name}_labeled_events.npy"))
            npy_f = npy_matches[0]
            events = load_events(npy_f)
            width, height = infer_resolution(seq_name, events[:, 0], events[:, 1])

            t0 = time.perf_counter()
            w_cands, full_w_feats, w_times, w_lats = extract_raw_sequence_cands_and_windows(
                events, width, height, cfg, learned_scorer
            )
            dt = time.perf_counter() - t0
            total_extract_time += dt

            cand_count = sum(len(c) for c in w_cands)
            print(f"[{idx}/17] Extracted '{seq_name}' ({len(w_times)} win, {len(gt_rows)} GT, {cand_count} cands, {dt:.1f}s)", flush=True)

            seq_data_dict[seq_name] = {
                "window_cands": w_cands,
                "full_win_feats": full_w_feats,
                "window_times": w_times,
                "win_latencies": w_lats,
                "gt_rows": gt_rows,
                "width": width,
                "height": height,
                "events": events,
            }
        print(f"[INFO] Saving extracted features to {cache_path}...", flush=True)
        joblib.dump(seq_data_dict, cache_path)

    # =========================================================================
    # VARIANT A — G sweep on Train (G in {1, 2, 3, 5})
    # =========================================================================
    print(f"\n==========================================================================================")
    print(f"  VARIANT A: SWEEPING GAP TOLERANCE G in {{1, 2, 3, 5}} ON TRAIN SPLIT")
    print(f"==========================================================================================")
    best_g = None
    best_g_map = -1.0
    g_sweep_results = []
    trained_rerankers = {}

    for g_val in [1, 2, 3, 5]:
        X_tr, y_tr, X_val, y_val, n_singles, single_scores = assemble_track_dataset(seq_data_dict, max_gap=g_val)
        print(f"\n[G={g_val}] Assembled Track Dataset: Train={X_tr.shape[0]} ({np.sum(y_tr)} pos), Val={X_val.shape[0]} ({np.sum(y_val)} pos), Singletons={n_singles}")

        clf_a = HistGradientBoostingClassifier(
            max_iter=150,
            learning_rate=0.08,
            max_leaf_nodes=31,
            min_samples_leaf=20,
            random_state=42,
        )
        clf_a.fit(X_tr, y_tr)
        val_auc = roc_auc_score(y_val, clf_a.predict_proba(X_val)[:, 1])
        val_pr_auc = average_precision_score(y_val, clf_a.predict_proba(X_val)[:, 1])
        trained_rerankers[g_val] = clf_a

        # Evaluate across all 17 train sequences
        seq_aps = []
        sparse_aps = []
        dense_aps = []
        tot_tp, tot_fp, tot_fn = 0, 0, 0

        for seq_name, data in seq_data_dict.items():
            res, _ = evaluate_variant_on_sequence(
                data["window_cands"], data["full_win_feats"], data["window_times"], data["gt_rows"],
                mode="variant_a", reranker_model=clf_a, max_gap=g_val, conf_min=0.30, max_k=1
            )
            seq_aps.append(res["ap"])
            if len(data["gt_rows"]) <= 43:
                sparse_aps.append(res["ap"])
            else:
                dense_aps.append(res["ap"])
            tot_tp += res["tp"]
            tot_fp += res["fp"]
            tot_fn += res["fn"]

        mean_map = float(np.mean(seq_aps))
        sparse_map = float(np.mean(sparse_aps))
        dense_map = float(np.mean(dense_aps))
        p, r, f1 = compute_prf1(tot_tp, tot_fp, tot_fn)

        g_sweep_results.append({
            "G": g_val,
            "Val ROC-AUC": val_auc,
            "Val PR-AUC": val_pr_auc,
            "Train mAP": mean_map,
            "Sparse mAP": sparse_map,
            "Dense mAP": dense_map,
            "Precision": p,
            "Recall": r,
            "F1": f1,
            "TP": tot_tp,
            "FP": tot_fp,
            "FN": tot_fn,
            "Singletons": n_singles,
            "Mean Single Score": float(np.mean(single_scores)) if single_scores else 0.0,
        })

        if sparse_map > best_g_map:
            best_g_map = sparse_map
            best_g = g_val

    g_table = [
        [
            r["G"], f"{r['Val ROC-AUC']:.4f}", f"{r['Val PR-AUC']:.4f}",
            f"{r['Train mAP']:.6f}", f"{r['Sparse mAP']:.6f}", f"{r['Dense mAP']:.6f}",
            f"{r['Precision']:.4f}", f"{r['Recall']:.4f}", f"{r['F1']:.4f}",
            r["TP"], r["FP"], r["Singletons"], f"{r['Mean Single Score']:.4f}"
        ]
        for r in g_sweep_results
    ]
    print(tabulate(
        g_table,
        headers=["G", "Val AUC", "Val PR-AUC", "Train mAP", "Sparse mAP", "Dense mAP", "Prec", "Rec", "F1", "TP", "FP", "Singles", "Single Score"],
        tablefmt="github"
    ))

    print(f"\n[WINNER G] Selected best G={best_g} (Sparse mAP: {best_g_map:.6f})")
    best_reranker = trained_rerankers[best_g]
    joblib.dump(best_reranker, "models/reranker_track.joblib")
    print(f"[INFO] Saved winning track reranker to models/reranker_track.joblib")

    # =========================================================================
    # VARIANT B — Window Objectness Gate
    # =========================================================================
    print(f"\n==========================================================================================")
    print(f"  VARIANT B: TRAINING WINDOW OBJECTNESS CLASSIFIER ON TRAIN SPLIT")
    print(f"==========================================================================================")
    X_win_tr, y_win_tr, X_win_val, y_win_val = assemble_objectness_dataset(seq_data_dict)
    tr_pos = int(np.sum(y_win_tr))
    tr_tot = int(X_win_tr.shape[0])
    tr_base_rate = tr_pos / tr_tot if tr_tot > 0 else 0.0

    val_pos = int(np.sum(y_win_val))
    val_tot = int(X_win_val.shape[0])
    val_base_rate = val_pos / val_tot if val_tot > 0 else 0.0

    print(f"[INFO] Objectness Dataset: Train={tr_tot} ({tr_pos} pos / {tr_tot - tr_pos} neg, {tr_base_rate*100:.2f}% pos base rate)")
    print(f"[INFO] Objectness Dataset: Val={val_tot} ({val_pos} pos / {val_tot - val_pos} neg, {val_base_rate*100:.2f}% pos base rate)")

    clf_b = HistGradientBoostingClassifier(
        max_iter=150,
        learning_rate=0.08,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        random_state=42,
    )
    clf_b.fit(X_win_tr, y_win_tr)
    tr_obj_probs = clf_b.predict_proba(X_win_tr)[:, 1]
    tr_obj_roc = roc_auc_score(y_win_tr, tr_obj_probs)
    tr_obj_pr = average_precision_score(y_win_tr, tr_obj_probs)
    tr_pr_ratio = tr_obj_pr / tr_base_rate if tr_base_rate > 0 else 0.0

    val_obj_probs = clf_b.predict_proba(X_win_val)[:, 1]
    val_obj_roc = roc_auc_score(y_win_val, val_obj_probs)
    val_obj_pr = average_precision_score(y_win_val, val_obj_probs)
    val_pr_ratio = val_obj_pr / val_base_rate if val_base_rate > 0 else 0.0

    print(f"[METRIC] Train Objectness ROC-AUC: {tr_obj_roc:.6f} | PR-AUC: {tr_obj_pr:.6f} (Base Rate: {tr_base_rate:.4f}, Ratio: {tr_pr_ratio:.2f}x trivial)")
    print(f"[METRIC] Val Objectness   ROC-AUC: {val_obj_roc:.6f} | PR-AUC: {val_obj_pr:.6f} (Base Rate: {val_base_rate:.4f}, Ratio: {val_pr_ratio:.2f}x trivial)")

    joblib.dump(clf_b, f"models/scorer_objectness_{cfg_path.stem}.joblib")
    print(f"[INFO] Saved window objectness classifier to models/scorer_objectness_{cfg_path.stem}.joblib")

    # =========================================================================
    # COMPREHENSIVE COMPARISON: BASELINE vs VARIANT A vs VARIANT B vs VARIANT A+B
    # =========================================================================
    print(f"\n==========================================================================================")
    print(f"  COMPREHENSIVE VARIANT EVALUATION (17 TRAIN SEQUENCES)")
    print(f"==========================================================================================")

    variants = [
        ("Baseline (Locked)", "baseline", None, None),
        (f"Variant A (Track G={best_g})", "variant_a", best_reranker, None),
        ("Variant B (Window Objectness)", "variant_b", None, clf_b),
        (f"Variant A+B (Track G={best_g} + Obj)", "variant_ab", best_reranker, clf_b),
    ]

    variant_summary = []
    detailed_seq_reports = {}

    for var_title, mode, r_mod, o_mod in variants:
        seq_aps = []
        sparse_aps = []
        dense_aps = []
        tot_tp, tot_fp, tot_fn = 0, 0, 0
        seq_rows = []
        added_ms_list = []

        for seq_name, data in seq_data_dict.items():
            res, ms_per_win = evaluate_variant_on_sequence(
                data["window_cands"], data["full_win_feats"], data["window_times"], data["gt_rows"],
                mode=mode, reranker_model=r_mod, objectness_model=o_mod, max_gap=best_g, conf_min=0.30, max_k=1
            )
            seq_aps.append(res["ap"])
            is_sparse = len(data["gt_rows"]) <= 43
            if is_sparse:
                sparse_aps.append(res["ap"])
            else:
                dense_aps.append(res["ap"])
            tot_tp += res["tp"]
            tot_fp += res["fp"]
            tot_fn += res["fn"]
            added_ms_list.append(ms_per_win)

            seq_rows.append({
                "sequence": seq_name,
                "gt": len(data["gt_rows"]),
                "sparse": is_sparse,
                "preds": res["n_pred"],
                "precision": res["precision"],
                "recall": res["recall"],
                "f1": res["f1"],
                "ap": res["ap"],
                "ms_per_win": ms_per_win,
            })

        mean_map = float(np.mean(seq_aps))
        sparse_map = float(np.mean(sparse_aps))
        dense_map = float(np.mean(dense_aps))
        p, r, f1 = compute_prf1(tot_tp, tot_fp, tot_fn)

        detailed_seq_reports[var_title] = seq_rows
        variant_summary.append({
            "Variant": var_title,
            "Train mAP": mean_map,
            "Sparse mAP (10)": sparse_map,
            "Dense mAP (7)": dense_map,
            "Precision": p,
            "Recall": r,
            "F1": f1,
            "TP": tot_tp,
            "FP": tot_fp,
            "FN": tot_fn,
            "Added ms/win": float(np.mean(added_ms_list)),
        })

    summary_table = [
        [
            r["Variant"],
            f"{r['Train mAP']:.6f}",
            f"{r['Sparse mAP (10)']:.6f}",
            f"{r['Dense mAP (7)']:.6f}",
            f"{r['Precision']:.6f}",
            f"{r['Recall']:.6f}",
            f"{r['F1']:.6f}",
            r["TP"], r["FP"], r["FN"],
            f"{r['Added ms/win']:.3f}"
        ]
        for r in variant_summary
    ]
    print(tabulate(
        summary_table,
        headers=["Variant", "Train mAP", "Sparse mAP (10)", "Dense mAP (7)", "Precision", "Recall", "F1", "TP", "FP", "FN", "ms/win (Rerank)"],
        tablefmt="github"
    ))

    # Per-sequence breakdown comparison
    print(f"\n==========================================================================================")
    print(f"  PER-SEQUENCE AP@0.5 BREAKDOWN (17 TRAIN SEQUENCES)")
    print(f"==========================================================================================")
    per_seq_table = []
    for i, seq_name in enumerate(seq_data_dict.keys()):
        gt_cnt = len(seq_data_dict[seq_name]["gt_rows"])
        is_sp = "SPARSE" if gt_cnt <= 43 else "DENSE"
        b_ap = detailed_seq_reports["Baseline (Locked)"][i]["ap"]
        a_ap = detailed_seq_reports[f"Variant A (Track G={best_g})"][i]["ap"]
        b_var_ap = detailed_seq_reports["Variant B (Window Objectness)"][i]["ap"]
        ab_ap = detailed_seq_reports[f"Variant A+B (Track G={best_g} + Obj)"][i]["ap"]
        per_seq_table.append([
            seq_name, is_sp, gt_cnt,
            f"{b_ap:.4f}", f"{a_ap:.4f}", f"{b_var_ap:.4f}", f"{ab_ap:.4f}"
        ])
    print(tabulate(
        per_seq_table,
        headers=["Sequence", "Type", "GT", "Baseline AP", f"Var A (G={best_g})", "Var B (Obj)", "Var A+B"],
        tablefmt="github"
    ))

    # Save predictions of Variant B if requested
    if args.save_preds_dir:
        out_pdir = Path(args.save_preds_dir)
        out_pdir.mkdir(parents=True, exist_ok=True)
        print(f"\n[INFO] Saving Variant B predictions to {out_pdir}...", flush=True)
        for seq_name, data in seq_data_dict.items():
            res, _ = evaluate_variant_on_sequence(
                data["window_cands"], data["full_win_feats"], data["window_times"], data["gt_rows"],
                mode="variant_b", reranker_model=None, objectness_model=clf_b, max_gap=best_g, conf_min=0.30, max_k=1
            )
            out_file = out_pdir / f"{seq_name}_pred.txt"
            with open(out_file, "w") as f:
                for row in res["preds"]:
                    f.write(f"{row[0]}\t{row[1]}\t{row[2]}\t{row[3]}\t{row[4]}\t{row[5]}\t{row[6]:.4f}\n")
        print(f"[INFO] Successfully wrote {len(seq_data_dict)} prediction files to {out_pdir}", flush=True)

    # =========================================================================
    # LATENCY BENCHMARK PER SEQUENCE
    # =========================================================================
    print(f"\n==========================================================================================", flush=True)
    print(f"  LATENCY BENCHMARK (TOTAL PIPELINE + BEST VARIANT, PER SEQUENCE)", flush=True)
    print(f"==========================================================================================", flush=True)
    latency_table = []
    any_exceed_40 = False

    for seq_name, data in seq_data_dict.items():
        win_times = data["win_latencies"]
        mean_ms = float(np.mean(win_times))
        p99_ms = float(np.percentile(win_times, 99))
        max_ms = float(np.max(win_times))

        exceeds = mean_ms > 40.0 or p99_ms > 40.0
        if exceeds:
            any_exceed_40 = True

        latency_table.append([
            seq_name, len(win_times), f"{mean_ms:.2f}", f"{p99_ms:.2f}", f"{max_ms:.2f}", "FAIL (>40ms)" if exceeds else "PASS (<40ms)"
        ])

    print(tabulate(
        latency_table,
        headers=["Sequence", "Windows", "Mean (ms)", "p99 (ms)", "Max (ms)", "Real-time Compliance"],
        tablefmt="github"
    ), flush=True)
    print(f"\n[LATENCY VERDICT] Any sequence exceeds 40ms/win: {any_exceed_40}", flush=True)

    # =========================================================================
    # TEST SPLIT EVALUATION OF BEST VARIANT
    # =========================================================================
    # Pick winner by Sparse mAP
    best_variant_row = max(variant_summary, key=lambda x: x["Sparse mAP (10)"])
    winner_name = best_variant_row["Variant"]
    print(f"\n==========================================================================================", flush=True)
    print(f"  TEST SPLIT EVALUATION — WINNER: {winner_name}", flush=True)
    print(f"==========================================================================================", flush=True)

    test_cache_path = Path(f"models/test_seq_extracted_cache_{cfg_path.stem}.joblib")
    if not test_cache_path.exists() and cfg_path.stem == "config" and Path("models/test_seq_extracted_cache.joblib").exists():
        test_cache_path = Path("models/test_seq_extracted_cache.joblib")

    if test_cache_path.exists():
        print(f"[INFO] Loading cached test sequence features from {test_cache_path}...", flush=True)
        test_seq_data = joblib.load(test_cache_path)
    else:
        test_gt_files = [f for f in all_gt_files if "Training" not in str(f)]
        test_seq_data = {}
        for idx, gt_f in enumerate(test_gt_files, 1):
            seq_name = gt_f.name.replace("_bb_windows_40ms.txt", "")
            gt_rows = load_gt_file(gt_f)
            npy_matches = list(gt_f.parent.glob(f"{seq_name}_labeled_events.npy"))
            if not npy_matches:
                npy_matches = list(dataset_dir.rglob(f"{seq_name}_labeled_events.npy"))
            npy_f = npy_matches[0]
            events = load_events(npy_f)
            width, height = infer_resolution(seq_name, events[:, 0], events[:, 1])

            t0_test = time.perf_counter()
            w_cands, full_w_feats, w_times, w_lats = extract_raw_sequence_cands_and_windows(
                events, width, height, cfg, learned_scorer
            )
            dt_test = time.perf_counter() - t0_test
            print(f"[{idx}/4] Extracted Test '{seq_name}' ({len(w_times)} win, {len(gt_rows)} GT, {dt_test:.1f}s)", flush=True)

            test_seq_data[seq_name] = {
                "window_cands": w_cands,
                "full_win_feats": full_w_feats,
                "window_times": w_times,
                "win_latencies": w_lats,
                "gt_rows": gt_rows,
            }
        print(f"[INFO] Saving extracted test features to {test_cache_path}...", flush=True)
        joblib.dump(test_seq_data, test_cache_path)

    test_mode_map = {
        "Baseline (Locked)": ("baseline", None, None),
        f"Variant A (Track G={best_g})": ("variant_a", best_reranker, None),
        "Variant B (Window Objectness)": ("variant_b", None, clf_b),
        f"Variant A+B (Track G={best_g} + Obj)": ("variant_ab", best_reranker, clf_b),
    }

    t_mode, t_rmod, t_omod = test_mode_map[winner_name]
    test_aps = []
    tot_tp, tot_fp, tot_fn = 0, 0, 0
    test_per_seq = []

    for seq_name, data in test_seq_data.items():
        res, ms_per_win = evaluate_variant_on_sequence(
            data["window_cands"], data["full_win_feats"], data["window_times"], data["gt_rows"],
            mode=t_mode, reranker_model=t_rmod, objectness_model=t_omod, max_gap=best_g, conf_min=0.30, max_k=1
        )
        test_aps.append(res["ap"])
        tot_tp += res["tp"]
        tot_fp += res["fp"]
        tot_fn += res["fn"]
        test_per_seq.append([
            seq_name, len(data["gt_rows"]), res["n_pred"], f"{res['precision']:.4f}", f"{res['recall']:.4f}", f"{res['f1']:.4f}", f"{res['ap']:.4f}", f"{ms_per_win:.2f}"
        ])

    test_map = float(np.mean(test_aps))
    t_p, t_r, t_f1 = compute_prf1(tot_tp, tot_fp, tot_fn)

    print(f"\nTEST SPLIT OVERALL METRICS ({winner_name}):")
    test_summary_table = [[
        f"{test_map:.6f}", f"{t_p:.6f}", f"{t_r:.6f}", f"{t_f1:.6f}", tot_tp, tot_fp, tot_fn
    ]]
    print(tabulate(
        test_summary_table,
        headers=["Test mAP", "Precision", "Recall", "F1", "TP", "FP", "FN"],
        tablefmt="github"
    ))

    print(f"\nTEST SPLIT PER-SEQUENCE BREAKDOWN:")
    print(tabulate(
        test_per_seq,
        headers=["Sequence", "GT", "Preds", "Precision", "Recall", "F1", "AP@0.5", "ms/win"],
        tablefmt="github"
    ))


if __name__ == "__main__":
    main()
