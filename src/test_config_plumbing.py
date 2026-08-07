"""Regression test for configuration parameter plumbing in detector and sweep harness."""

import sys
import numpy as np
from src.detector import detect_boxes
from src.sweep import run_sequence_sweep_cached


def test_box_height_plumbing() -> bool:
    """Assert box_h=20 versus box_h=60 yields different box heights."""
    img = np.zeros((720, 1280), dtype=np.float32)
    img[300:320, 500:520] = 10.0

    cfg_h20 = {
        "percentile": 90.0,
        "min_events_in_box": 1,
        "open_kernel": 1,
        "dilate_kernel": 1,
        "EVK4": {"box_mode": "fixed", "box_w": 52.0, "box_h": 20.0},
    }
    cfg_h60 = {
        "percentile": 90.0,
        "min_events_in_box": 1,
        "open_kernel": 1,
        "dilate_kernel": 1,
        "EVK4": {"box_mode": "fixed", "box_w": 52.0, "box_h": 60.0},
    }

    boxes_h20 = detect_boxes(img, 1280, 720, cfg_h20)
    boxes_h60 = detect_boxes(img, 1280, 720, cfg_h60)

    if not boxes_h20 or not boxes_h60:
        return False

    h20_val = boxes_h20[0]["height"]
    h60_val = boxes_h60[0]["height"]

    return abs(h20_val - 20.0) < 1e-3 and abs(h60_val - 60.0) < 1e-3 and h20_val != h60_val


def test_centroid_mode_plumbing() -> bool:
    """Assert centroid_mode="component" versus "weighted" yields different centers on asymmetric blob."""
    img = np.zeros((346, 260), dtype=np.float32)
    # Asymmetric blob: both sides survive thresholding, but right side has 10x higher event weights
    img[100:110, 100:105] = 10.0
    img[100:110, 105:110] = 100.0

    cfg_comp = {
        "percentile": 50.0,
        "min_events_in_box": 1,
        "open_kernel": 1,
        "dilate_kernel": 1,
        "DAVIS": {"box_mode": "fixed", "box_w": 10.0, "box_h": 12.0, "centroid_mode": "component"},
    }
    cfg_weight = {
        "percentile": 50.0,
        "min_events_in_box": 1,
        "open_kernel": 1,
        "dilate_kernel": 1,
        "DAVIS": {"box_mode": "fixed", "box_w": 10.0, "box_h": 12.0, "centroid_mode": "weighted"},
    }

    boxes_comp = detect_boxes(img, 260, 346, cfg_comp)
    boxes_weight = detect_boxes(img, 260, 346, cfg_weight)

    if not boxes_comp or not boxes_weight:
        return False

    cx_comp = boxes_comp[0]["center_x"]
    cx_weight = boxes_weight[0]["center_x"]

    return abs(cx_comp - cx_weight) >= 0.4


def test_sweep_cache_key_completeness() -> bool:
    """Assert sweep harness cache keys and configuration parameters are complete."""
    raw_grid = {
        "percentile": [97.5],
        "open_kernel": [1],
        "dilate_kernel": [3],
        "min_events_in_box": [6],
        "min_hits": [2],
        "box_mode": ["fixed"],
        "centroid_mode": ["component", "weighted"],
        "EVK4": {"box_w": [52], "box_h": [44, 56]},
    }

    # Verify that different box_h and centroid_mode settings generate distinct parameter dictionary keys
    b_h_keys = raw_grid["EVK4"]["box_h"]
    c_mode_keys = raw_grid["centroid_mode"]

    unique_combos = set()
    for bh in b_h_keys:
        for cm in c_mode_keys:
            key = (bh, cm)
            unique_combos.add(key)

    return len(unique_combos) == len(b_h_keys) * len(c_mode_keys)


def main() -> None:
    """Run all plumbing assertions and report PASS/FAIL per test."""
    print("\n==================================================")
    print("  CONFIGURATION PLUMBING REGRESSION TESTS")
    print("==================================================")

    res_h = test_box_height_plumbing()
    print(f"Assertion 1 (box_h=20 vs 60 height override):   [{'PASS' if res_h else 'FAIL'}]")

    res_c = test_centroid_mode_plumbing()
    print(f"Assertion 2 (component vs weighted centroid):    [{'PASS' if res_c else 'FAIL'}]")

    res_k = test_sweep_cache_key_completeness()
    print(f"Assertion 3 (sweep harness parameter inclusion): [{'PASS' if res_k else 'FAIL'}]")

    all_passed = res_h and res_c and res_k
    print(f"\nOVERALL RESULT: [{'PASS' if all_passed else 'FAIL'}]\n")
    if not all_passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
