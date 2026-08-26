"""Unified end-to-end inference and evaluation pipeline for OrbitSight.

Execution Order per Window:
1. Windowing (`iter_windows`) & Fast Count Map Generation (`event_image` with need_polarity=False)
2. Component Detection (`detect_boxes` returning full untruncated component lists)
3. Vectorized Neighborhood Persistence Matching against full untruncated neighbor windows (`min_hits`)
4. Deterministic Multi-term Weighted Confidence Scoring (`compute_confidence`)
5. Pipeline-stage NMS on weighted confidence (`apply_nms` if `nms_stage == 'pipeline'`)
6. Confidence Threshold Gating (`conf_min`)
7. Top-K Window Candidate Truncation (`max_candidates_per_window`)
8. Coordinate & Confidence Rounding
"""

import math
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from src.common import (
    WINDOW_US,
    event_image,
    iter_windows,
    resolve_effective_config,
)
from src.detector import detect_boxes
from src.nms import apply_nms


def compute_confidence(
    box: Dict[str, float],
    hits: int,
    cfg: Dict[str, Any],
) -> float:
    """Compute deterministic weighted confidence score clipped to [0.01, 1.0]."""
    weights = cfg.get(
        "confidence_weights",
        {"density": 0.25, "events": 0.35, "compactness": 0.20, "persistence": 0.20},
    )
    w_den = float(weights.get("density", 0.25))
    w_evt = float(weights.get("events", 0.35))
    w_cmp = float(weights.get("compactness", 0.20))
    w_per = float(weights.get("persistence", 0.20))

    sub_density = min(1.0, float(box.get("density", 0.0)))
    min_evt = float(cfg.get("min_events_in_box", 3))
    sub_events = min(1.0, float(box.get("events", 0.0)) / (min_evt * 5.0))
    sub_compactness = 1.0 / (1.0 + abs(float(box.get("aspect", 1.0)) - 1.0))
    sub_persistence = min(1.0, hits / 3.0)

    score = (
        w_den * sub_density
        + w_evt * sub_events
        + w_cmp * sub_compactness
        + w_per * sub_persistence
    )

    return float(np.clip(score, 0.01, 1.0))


def run_sequence(
    events: np.ndarray,
    width: int,
    height: int,
    cfg: Dict[str, Any],
    window_us: int = WINDOW_US,
    max_windows: float = float("inf"),
    deadline_ts: Optional[float] = None,
) -> Tuple[List[Tuple[int, int, int, int, int, int, float]], int]:
    """Run full unified detection pipeline on event stream.

    Args:
        events: NumPy array of events [x, y, p, t, (label)].
        width: Sensor width in pixels.
        height: Sensor height in pixels.
        cfg: Configuration dictionary.
        window_us: Window duration in microseconds (default 40,000 us).
        max_windows: Maximum windows to process (for testing/smoke tests).
        deadline_ts: Optional monotonic deadline timestamp.

    Returns:
        Tuple of (prediction_rows, num_windows_processed).
        Each prediction row: (w_start, w_end, center_x, center_y, width, height, confidence).
    """
    known_sensors = {
        "EVK4": (1280, 720, float(np.hypot(1280, 720))),
        "DVX": (640, 480, float(np.hypot(640, 480))),
        "DAVIS": (346, 260, float(np.hypot(346, 260))),
    }
    curr_diag = float(np.hypot(width, height))

    if width == 1280 and height == 720:
        sensor_name = "EVK4"
    elif width == 640 and height == 480:
        sensor_name = "DVX"
    elif width == 346 and height == 260:
        sensor_name = "DAVIS"
    else:
        sensor_name = min(known_sensors.keys(), key=lambda k: abs(curr_diag - known_sensors[k][2]))

    eff = resolve_effective_config(cfg, sensor_name)

    from src.static_map import build_continuous_static_map
    static_frac_map = build_continuous_static_map(events, width, height, window_us=window_us)
    static_thresh = eff.get("static_thresh", None)
    if static_thresh is not None:
        static_mask = static_frac_map >= float(static_thresh)
        n_supp = int(np.sum(static_mask))
        pct_supp = (n_supp / (width * height)) * 100.0
        print(f"static_mask: {n_supp} pixels suppressed ({pct_supp:.2f}% of frame)", flush=True)
    else:
        static_mask = None

    from src.features import extract_local_bg

    window_records: List[Tuple[int, int, List[Dict[str, float]]]] = []
    base_window_stats: List[Dict[str, float]] = []
    window_count = 0

    for w_start, w_end, w_events in iter_windows(events, window_us=window_us):
        if deadline_ts is not None and time.monotonic() > deadline_ts:
            break
        count_img, _, _ = event_image(w_events, width, height, need_polarity=False)
        boxes = detect_boxes(count_img, width, height, cfg)
        if static_mask is not None and boxes:
            filtered_boxes: List[Dict[str, float]] = []
            for b in boxes:
                cy_r = int(round(b["center_y"]))
                cx_r = int(round(b["center_x"]))
                if 0 <= cy_r < height and 0 <= cx_r < width and static_mask[cy_r, cx_r]:
                    continue
                filtered_boxes.append(b)
            boxes = filtered_boxes

        # Precompute candidate local_bg while count_img is live in CPU cache
        for b in boxes:
            b["local_bg"] = extract_local_bg(
                count_img, float(b["center_x"]), float(b["center_y"]), float(b["width"]), float(b["height"])
            )

        # Base window stats for objectness gate (computed from all w_events and filtered boxes)
        tot_ev = float(len(w_events))
        n_comp = float(len(boxes))
        if tot_ev > 0:
            x_std = float(np.std(w_events[:, 0]))
            y_std = float(np.std(w_events[:, 1]))
        else:
            x_std, y_std = 0.0, 0.0
        max_comp_ev = float(max([b.get("events", 0.0) for b in boxes])) if boxes else 0.0

        base_window_stats.append({
            "win_total_events": tot_ev,
            "win_num_components": n_comp,
            "win_x_std": x_std,
            "win_y_std": y_std,
            "win_max_comp_events": max_comp_ev,
        })

        window_records.append((w_start, w_end, boxes))
        window_count += 1
        if window_count >= max_windows:
            break

    num_windows = len(window_records)
    min_hits = int(eff.get("min_hits", 1))
    max_dist_frac = float(eff.get("max_dist_frac", 0.08))
    conf_min = float(eff.get("conf_min", 0.0))
    nms_stage = str(eff.get("nms_stage", "pipeline")).lower()
    nms_iou_val = eff.get("nms_iou", 0.3)

    max_k_val = eff.get("max_candidates_per_window", None)
    if max_k_val is not None:
        try:
            max_k = int(max_k_val)
        except (TypeError, ValueError):
            max_k = None
    else:
        max_k = None

    diagonal = math.hypot(width, height)
    max_dist = max_dist_frac * diagonal
    max_dist_sq = max_dist * max_dist

    # Load candidate scorer model (raise if missing when scorer_mode == "learned")
    scorer_mode = str(eff.get("scorer_mode", "weighted")).lower()
    learned_model = None
    if scorer_mode == "learned":
        from pathlib import Path
        import joblib
        scorer_path_str = eff.get("scorer_path", "models/scorer.joblib")
        scorer_path = Path(scorer_path_str)
        if not scorer_path.exists():
            raise FileNotFoundError(f"Learned candidate scorer model not found: {scorer_path}")
        learned_model = joblib.load(scorer_path)
        print(f"learned scorer loaded: {scorer_path}", flush=True)

    # Load window objectness model if objectness_mode == "gate"
    objectness_mode = str(eff.get("objectness_mode", "off")).lower()
    objectness_model = None
    if objectness_mode == "gate":
        from pathlib import Path
        import joblib
        obj_path_str = eff.get("objectness_path", "models/scorer_objectness_pre_geometry.joblib")
        obj_path = Path(obj_path_str)
        if not obj_path.exists():
            if Path("models/scorer_objectness.joblib").exists():
                obj_path = Path("models/scorer_objectness.joblib")
            else:
                raise FileNotFoundError(f"Window objectness model not found: {obj_path}")
        objectness_model = joblib.load(obj_path)
        print(f"window objectness model loaded: {obj_path}", flush=True)

    # Load post-hoc box resizing model if configured
    box_reg_mode = str(eff.get("box_regressor_mode", "none")).lower()
    box_reg_model = None
    if box_reg_mode in ("arm1", "least_squares"):
        from pathlib import Path
        import joblib
        p = Path(eff.get("box_regressor_arm1_path", "models/box_regressor_arm1.joblib"))
        if p.exists():
            box_reg_model = joblib.load(p)
    elif box_reg_mode in ("arm2", "learned"):
        from pathlib import Path
        import joblib
        p = Path(eff.get("box_regressor_arm2_path", "models/box_regressor_arm2.joblib"))
        if p.exists():
            box_reg_model = joblib.load(p)

    # Pass 2: Persistence matching and candidate scoring
    window_candidates: List[List[Dict[str, Any]]] = []

    for w_idx in range(num_windows):
        w_start, w_end, boxes = window_records[w_idx]
        if not boxes:
            window_candidates.append([])
            continue

        prev_boxes = window_records[w_idx - 1][2] if w_idx > 0 else []
        next_boxes = (
            window_records[w_idx + 1][2] if w_idx < num_windows - 1 else []
        )

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

        cand_boxes_list: List[Dict[str, float]] = []

        for idx, box in enumerate(boxes):
            hits = 1 + int(has_prev[idx]) + int(has_next[idx])
            if min_hits >= 2 and hits < min_hits:
                continue
            box_copy = dict(box)
            box_copy["hits"] = hits
            cand_boxes_list.append(box_copy)

        if cand_boxes_list:
            if learned_model is not None:
                from src.features import FEATURE_NAMES, extract_window_features_batch
                batch_feats = extract_window_features_batch(
                    cand_boxes_list,
                    prev_boxes,
                    next_boxes,
                    count_img=None,
                    static_frac_map=static_frac_map,
                )
                cand_features_list = [
                    [f[name] for name in FEATURE_NAMES] for f in batch_feats
                ]
                X_cand = np.array(cand_features_list, dtype=np.float32)
                probs = learned_model.predict_proba(X_cand)[:, 1]
                for b_copy, p_score, f_dict in zip(cand_boxes_list, probs, batch_feats):
                    b_copy["confidence"] = float(p_score)
                    b_copy["features"] = [f_dict[name] for name in FEATURE_NAMES] + [float(width), float(height)]
                    b_copy["extent_w"] = f_dict.get("extent_w", b_copy.get("extent_w", b_copy["width"]))
                    b_copy["extent_h"] = f_dict.get("extent_h", b_copy.get("extent_h", b_copy["height"]))
            else:
                for b_copy in cand_boxes_list:
                    b_copy["confidence"] = compute_confidence(b_copy, int(b_copy["hits"]), eff)

        window_candidates.append(cand_boxes_list)

    # Pass 3: Window Objectness Gating
    if objectness_mode == "gate" and objectness_model is not None:
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

        X_win_list = []
        for w_idx in range(num_windows):
            cur = window_features_21[w_idx]
            prev = window_features_21[w_idx - 1] if w_idx > 0 else {k: 0.0 for k in cur}
            nxt = window_features_21[w_idx + 1] if w_idx < num_windows - 1 else {k: 0.0 for k in cur}

            w_vec = [
                cur["win_total_events"], cur["win_num_components"], cur["win_x_std"], cur["win_y_std"], cur["win_max_comp_events"], cur["win_cand_score_mean"], cur["win_cand_score_max"],
                prev["win_total_events"], prev["win_num_components"], prev["win_x_std"], prev["win_y_std"], prev["win_max_comp_events"], prev["win_cand_score_mean"], prev["win_cand_score_max"],
                nxt["win_total_events"], nxt["win_num_components"], nxt["win_x_std"], nxt["win_y_std"], nxt["win_max_comp_events"], nxt["win_cand_score_mean"], nxt["win_cand_score_max"],
            ]
            X_win_list.append(w_vec)

        if X_win_list:
            X_win = np.array(X_win_list, dtype=np.float32)
            win_probs = objectness_model.predict_proba(X_win)[:, 1]
            for w_idx in range(num_windows):
                p_obj = float(win_probs[w_idx])
                for c in window_candidates[w_idx]:
                    c["confidence"] = float(c["confidence"]) * p_obj

    # Pass 4: NMS, confidence filtering, top-K, and prediction assembly
    predictions: List[Tuple[int, int, int, int, int, int, float]] = []

    for w_idx in range(num_windows):
        w_start, w_end, _ = window_records[w_idx]
        cands = window_candidates[w_idx]
        if not cands:
            continue

        # Pipeline NMS
        if nms_stage == "pipeline" and nms_iou_val is not None:
            try:
                nms_thresh = float(nms_iou_val)
                cands = apply_nms(cands, nms_thresh)
            except (TypeError, ValueError):
                pass

        # Confidence threshold filtering
        if conf_min > 0.0:
            cands = [b for b in cands if b["confidence"] >= conf_min]

        if not cands:
            continue

        # Top-K candidate truncation
        if max_k is not None and max_k > 0 and len(cands) > max_k:
            cands.sort(key=lambda b: b["confidence"], reverse=True)
            cands = cands[:max_k]

        # Rounding into final prediction format (with post-hoc box resizing if active)
        min_dim = float(eff.get("min_dim", 1.0))
        max_dim = float(eff.get("max_dim", 200.0))

        for b in cands:
            cx = int(round(b["center_x"]))
            cy = int(round(b["center_y"]))
            bw = float(b["width"])
            bh = float(b["height"])

            if box_reg_mode in ("arm1", "least_squares") and box_reg_model is not None:
                s_dict = box_reg_model.get(sensor_name, {"w": (1.0, 0.0), "h": (1.0, 0.0)})
                ext_w = float(b.get("extent_w", bw))
                ext_h = float(b.get("extent_h", bh))
                bw = s_dict["w"][0] * ext_w + s_dict["w"][1]
                bh = s_dict["h"][0] * ext_h + s_dict["h"][1]
            elif box_reg_mode in ("arm2", "learned") and box_reg_model is not None:
                reg_w = box_reg_model["reg_w"]
                reg_h = box_reg_model["reg_h"]
                f_vec = b.get("features", None)
                if f_vec is not None:
                    X_cand = np.array([f_vec], dtype=np.float32)
                    bw = float(np.exp(reg_w.predict(X_cand)[0]))
                    bh = float(np.exp(reg_h.predict(X_cand)[0]))

            bw = max(min_dim, min(max_dim, bw))
            bh = max(min_dim, min(max_dim, bh))

            bw_int = int(round(bw))
            bh_int = int(round(bh))
            conf_rounded = round(float(b["confidence"]), 4)
            predictions.append((w_start, w_end, cx, cy, bw_int, bh_int, conf_rounded))

    return predictions, num_windows
