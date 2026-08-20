"""Multi-window candidate association and track-based feature extraction for space object reranking."""

import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


TRACK_FEATURE_NAMES = [
    "track_len",
    "track_num_gaps",
    "track_total_disp",
    "track_speed_mean",
    "track_speed_std",
    "track_straightness",
    "track_score_mean",
    "track_score_max",
    "track_score_min",
    "track_events_mean",
]


def compute_straightness(ts: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> float:
    """Compute mean R^2 of linear fits x(t) and y(t). Returns 0.0 if insufficient points or variance."""
    if len(ts) < 3:
        return 0.0

    def calc_r2(t_arr: np.ndarray, val_arr: np.ndarray) -> float:
        var_tot = np.var(val_arr)
        if var_tot < 1e-9:
            return 1.0  # Stationary or constant coordinate is perfectly linear
        poly = np.polyfit(t_arr, val_arr, 1)
        fitted = np.polyval(poly, t_arr)
        ss_res = np.sum((val_arr - fitted) ** 2)
        ss_tot = np.sum((val_arr - np.mean(val_arr)) ** 2)
        if ss_tot < 1e-9:
            return 1.0
        r2 = 1.0 - (ss_res / ss_tot)
        return float(max(0.0, min(1.0, r2)))

    r2_x = calc_r2(ts, xs)
    r2_y = calc_r2(ts, ys)
    return float((r2_x + r2_y) / 2.0)


class Track:
    """Represents a sequence of associated candidate bounding boxes across windows."""

    def __init__(self, track_id: int, w_idx: int, cand: Dict[str, Any]):
        self.track_id = track_id
        self.history: List[Tuple[int, Dict[str, Any]]] = [(w_idx, cand)]
        self.last_w_idx: int = w_idx

    def add_detection(self, w_idx: int, cand: Dict[str, Any]):
        self.history.append((w_idx, cand))
        self.last_w_idx = w_idx

    def compute_features(self) -> Dict[str, float]:
        """Compute the 10 track-level features."""
        k = len(self.history)
        if k == 1:
            w0, c0 = self.history[0]
            conf = float(c0.get("confidence", 0.0))
            ev = float(c0.get("events", c0.get("event_count", 0.0)))
            return {
                "track_len": 1.0,
                "track_num_gaps": 0.0,
                "track_total_disp": 0.0,
                "track_speed_mean": 0.0,
                "track_speed_std": 0.0,
                "track_straightness": 0.0,
                "track_score_mean": conf,
                "track_score_max": conf,
                "track_score_min": conf,
                "track_events_mean": ev,
            }

        w_indices = np.array([h[0] for h in self.history], dtype=np.float32)
        xs = np.array([h[1]["center_x"] for h in self.history], dtype=np.float32)
        ys = np.array([h[1]["center_y"] for h in self.history], dtype=np.float32)
        scores = np.array([float(h[1].get("confidence", 0.0)) for h in self.history], dtype=np.float32)
        events = np.array([float(h[1].get("events", h[1].get("event_count", 0.0))) for h in self.history], dtype=np.float32)

        span_windows = int(w_indices[-1] - w_indices[0] + 1)
        num_gaps = max(0, span_windows - k)
        total_disp = float(math.hypot(xs[-1] - xs[0], ys[-1] - ys[0]))

        # Step speeds (distance / dt)
        dt = np.diff(w_indices)
        dx = np.diff(xs)
        dy = np.diff(ys)
        step_dist = np.hypot(dx, dy)
        step_speeds = step_dist / np.maximum(dt, 1.0)
        speed_mean = float(np.mean(step_speeds))
        speed_std = float(np.std(step_speeds)) if len(step_speeds) > 1 else 0.0

        straightness = compute_straightness(w_indices, xs, ys)

        return {
            "track_len": float(k),
            "track_num_gaps": float(num_gaps),
            "track_total_disp": total_disp,
            "track_speed_mean": speed_mean,
            "track_speed_std": speed_std,
            "track_straightness": straightness,
            "track_score_mean": float(np.mean(scores)),
            "track_score_max": float(np.max(scores)),
            "track_score_min": float(np.min(scores)),
            "track_events_mean": float(np.mean(events)),
        }


def build_sequence_tracks(
    window_cands: List[List[Dict[str, Any]]],
    max_gap: int = 2,
    max_speed_per_win: float = 30.0,
    min_cand_conf: float = 0.05,
) -> List[Track]:
    """Associate per-window candidates across time using greedy nearest-neighbour matching.

    Args:
        window_cands: List of length W, where each element is a list of candidate dictionaries in that window.
        max_gap: Maximum allowed gap in windows (G).
        max_speed_per_win: Velocity gate threshold (max pixels per window step).
        min_cand_conf: Minimum baseline candidate confidence to consider for track association.

    Returns:
        List of all constructed Track objects.
    """
    active_tracks: List[Track] = []
    finished_tracks: List[Track] = []
    track_counter = 0

    for w_idx, raw_cands in enumerate(window_cands):
        # 1. Prune dead tracks beyond max_gap
        surviving_active: List[Track] = []
        for trk in active_tracks:
            if w_idx - trk.last_w_idx > (max_gap + 1):
                finished_tracks.append(trk)
            else:
                surviving_active.append(trk)
        active_tracks = surviving_active

        cands = [c for c in raw_cands if float(c.get("confidence", 1.0)) >= min_cand_conf]
        if not cands:
            continue

        if not active_tracks:
            # Start a new track for each candidate
            for c in cands:
                track_counter += 1
                active_tracks.append(Track(track_counter, w_idx, c))
            continue

        # 2. Greedy association: compute pairwise distances between active tracks and current candidates
        n_tracks = len(active_tracks)
        n_cands = len(cands)
        cost_matrix = np.full((n_tracks, n_cands), fill_value=np.inf, dtype=np.float32)

        for i, trk in enumerate(active_tracks):
            dt = w_idx - trk.last_w_idx
            max_d = max_speed_per_win * dt
            last_cand = trk.history[-1][1]
            lx, ly = last_cand["center_x"], last_cand["center_y"]

            # Extrapolate position if track length >= 2
            if len(trk.history) >= 2:
                prev_w, prev_cand = trk.history[-2]
                dt_prev = trk.last_w_idx - prev_w
                vx = (lx - prev_cand["center_x"]) / max(1.0, dt_prev)
                vy = (ly - prev_cand["center_y"]) / max(1.0, dt_prev)
                pred_x = lx + vx * dt
                pred_y = ly + vy * dt
            else:
                pred_x, pred_y = lx, ly

            for j, c in enumerate(cands):
                cx, cy = c["center_x"], c["center_y"]
                raw_dist = math.hypot(cx - lx, cy - ly)
                if raw_dist <= max_d:
                    pred_dist = math.hypot(cx - pred_x, cy - pred_y)
                    cost_matrix[i, j] = pred_dist

        # Greedy matching
        matched_tracks = set()
        matched_cands = set()

        # Sort all valid pairs by cost ascending
        valid_pairs = []
        for i in range(n_tracks):
            for j in range(n_cands):
                if not np.isinf(cost_matrix[i, j]):
                    valid_pairs.append((cost_matrix[i, j], i, j))
        valid_pairs.sort(key=lambda x: x[0])

        for cost, i, j in valid_pairs:
            if i not in matched_tracks and j not in matched_cands:
                active_tracks[i].add_detection(w_idx, cands[j])
                matched_tracks.add(i)
                matched_cands.add(j)

        # Unmatched candidates start new tracks
        for j, c in enumerate(cands):
            if j not in matched_cands:
                track_counter += 1
                active_tracks.append(Track(track_counter, w_idx, c))

    finished_tracks.extend(active_tracks)
    return finished_tracks
