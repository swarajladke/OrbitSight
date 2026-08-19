"""Parity test: Verify extract_window_features_batch exactly matches extract_candidate_features."""

import math
from pathlib import Path
import numpy as np

from src.common import WINDOW_US, event_image, infer_resolution, iter_windows, load_events, sequence_name_from_npy
from src.detector import detect_boxes
from src.features import (
    FEATURE_NAMES,
    extract_candidate_features,
    extract_local_bg,
    extract_window_features_batch,
)
from src.scoreboard import load_yaml_config
from src.static_map import build_continuous_static_map


def test_feature_parity() -> None:
    dataset_dir = Path("../OrbitSight_Dataset").resolve()
    target_seq = "DAVIS_SL16RB_26070_2024-12-04-19-14-39"

    npy_matches = list(dataset_dir.rglob(f"{target_seq}_labeled_events.npy"))
    if not npy_matches:
        raise FileNotFoundError(f"Sequence {target_seq} not found in {dataset_dir}")

    npy_path = npy_matches[0]
    cfg = load_yaml_config(Path("config.yaml").resolve())

    events = load_events(npy_path)
    width, height = infer_resolution(target_seq, events[:, 0], events[:, 1])

    static_frac_map = build_continuous_static_map(events, width, height, window_us=WINDOW_US)
    static_mask = static_frac_map >= float(cfg.get("static_thresh", 0.5))

    window_records = []
    window_limit = 200

    for ws, we, w_events in iter_windows(events, window_us=WINDOW_US):
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
        window_records.append((ws, we, boxes))
        if len(window_records) >= window_limit:
            break

    num_windows = len(window_records)
    total_candidates_compared = 0

    for w_idx in range(num_windows):
        ws, we, boxes = window_records[w_idx]
        if not boxes:
            continue

        prev_boxes = window_records[w_idx - 1][2] if w_idx > 0 else []
        next_boxes = window_records[w_idx + 1][2] if w_idx < num_windows - 1 else []

        # Vectorized batch extraction
        batch_feats = extract_window_features_batch(
            boxes, prev_boxes, next_boxes, count_img=None, static_frac_map=static_frac_map
        )

        # Per-candidate scalar extraction
        scalar_feats = []
        for b in boxes:
            f = extract_candidate_features(
                b, prev_boxes, next_boxes, count_img=None, static_frac_map=static_frac_map
            )
            scalar_feats.append(f)

        assert len(batch_feats) == len(scalar_feats), f"Length mismatch: {len(batch_feats)} vs {len(scalar_feats)}"

        for c_idx in range(len(boxes)):
            b_dict = batch_feats[c_idx]
            s_dict = scalar_feats[c_idx]

            for feat_name in FEATURE_NAMES:
                val_b = b_dict[feat_name]
                val_s = s_dict[feat_name]

                # Check exact equality or strict float equivalence
                if math.isnan(val_b) and math.isnan(val_s):
                    continue

                if val_b != val_s:
                    raise AssertionError(
                        f"Feature mismatch on window {w_idx}, candidate {c_idx}, feature '{feat_name}': "
                        f"batch={val_b} vs scalar={val_s}"
                    )

            total_candidates_compared += 1

    print(f"[PASS] Feature parity test PASSED: {total_candidates_compared} candidates verified across {num_windows} windows.")


if __name__ == "__main__":
    test_feature_parity()
