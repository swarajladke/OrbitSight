"""Object detector implementation for event count maps with box_mode and centroid_mode options."""

from typing import Any, Dict, List
import cv2
import numpy as np


def _get_val(cfg: Dict[str, Any], key: str, default: Any) -> Any:
    """Helper to extract parameter from config or list."""
    val = cfg.get(key, default)
    if isinstance(val, list):
        return val[0]
    return val


def detect_boxes(
    count_img: np.ndarray, width: int, height: int, cfg: Dict[str, Any]
) -> List[Dict[str, float]]:
    """Detect bounding boxes of space objects from an event count image.

    Args:
        count_img: 2D float32 image of event counts (height, width).
        width: Image width.
        height: Image height.
        cfg: Configuration parameters dictionary.

    Returns:
        List of detection dictionaries with box properties.
    """
    nonzero_vals = count_img[count_img > 0]
    num_nonzero = len(nonzero_vals)
    if num_nonzero < 4:
        return []

    base_percentile = float(_get_val(cfg, "percentile", 97.5))

    # Dynamically scale percentile for ultra-dense windows to keep latency < 10ms
    if num_nonzero > 1000:
        percentile = min(99.0, base_percentile + (num_nonzero - 1000) / 500.0)
    else:
        percentile = base_percentile

    raw_thresh = float(np.percentile(nonzero_vals, percentile))
    thresh = max(1.0, raw_thresh)

    binary = (count_img >= thresh).astype(np.uint8)
    if not np.any(binary):
        return []

    open_k = int(_get_val(cfg, "open_kernel", 2))
    if open_k > 1:
        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (open_k, open_k))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open)

    dilate_k = int(_get_val(cfg, "dilate_kernel", 3))
    if dilate_k > 1:
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (dilate_k, dilate_k))
        binary = cv2.dilate(binary, kernel_dilate)

    if not np.any(binary):
        return []

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )

    max_area = 0.02 * width * height
    min_events = float(_get_val(cfg, "min_events_in_box", 6))
    box_mode = str(_get_val(cfg, "box_mode", "scale")).lower()
    centroid_mode = str(_get_val(cfg, "centroid_mode", "component")).lower()

    # Determine sensor family
    if width >= 1200:
        sensor_name = "EVK4"
        def_scale, def_pad = 3.8, 10.0
        def_bw, def_bh = 52.0, 46.0
    elif width >= 600:
        sensor_name = "DVX"
        def_scale, def_pad = 1.5, 4.0
        def_bw, def_bh = 18.0, 18.0
    else:
        sensor_name = "DAVIS"
        def_scale, def_pad = 1.1, 1.5
        def_bw, def_bh = 10.0, 10.0

    sensor_cfg = cfg.get(sensor_name, {}) if isinstance(cfg.get(sensor_name), dict) else {}

    box_scale = float(_get_val(sensor_cfg, "box_scale", _get_val(cfg, f"box_scale_{sensor_name.lower()}", _get_val(cfg, "box_scale", def_scale))))
    box_pad = float(_get_val(sensor_cfg, "box_pad", _get_val(cfg, f"box_pad_{sensor_name.lower()}", _get_val(cfg, "box_pad", def_pad))))
    box_w_cfg = float(_get_val(sensor_cfg, "box_w", _get_val(cfg, f"box_w_{sensor_name.lower()}", _get_val(cfg, "box_w", def_bw))))
    box_h_cfg = float(_get_val(sensor_cfg, "box_h", _get_val(cfg, f"box_h_{sensor_name.lower()}", _get_val(cfg, "box_h", def_bh))))

    results: List[Dict[str, float]] = []

    for label_idx in range(1, num_labels):
        comp_area = float(stats[label_idx, cv2.CC_STAT_AREA])
        if comp_area > max_area:
            continue

        comp_mask = labels == label_idx
        comp_events = float(count_img[comp_mask].sum())
        if comp_events < min_events:
            continue

        x_box = float(stats[label_idx, cv2.CC_STAT_LEFT])
        y_box = float(stats[label_idx, cv2.CC_STAT_TOP])
        w_box = float(stats[label_idx, cv2.CC_STAT_WIDTH])
        h_box = float(stats[label_idx, cv2.CC_STAT_HEIGHT])

        if centroid_mode == "weighted" and comp_events > 0:
            ys, xs = np.where(comp_mask)
            weights = count_img[comp_mask]
            center_x = float((xs * weights).sum() / comp_events)
            center_y = float((ys * weights).sum() / comp_events)
        else:
            center_x = x_box + w_box / 2.0
            center_y = y_box + h_box / 2.0

        if box_mode == "fixed":
            new_w = box_w_cfg
            new_h = box_h_cfg
        else:
            new_w = w_box * box_scale + 2.0 * box_pad
            new_h = h_box * box_scale + 2.0 * box_pad

        x1 = center_x - new_w / 2.0
        y1 = center_y - new_h / 2.0
        x2 = center_x + new_w / 2.0
        y2 = center_y + new_h / 2.0

        x1_clamped = max(0.0, min(float(width), x1))
        y1_clamped = max(0.0, min(float(height), y1))
        x2_clamped = max(0.0, min(float(width), x2))
        y2_clamped = max(0.0, min(float(height), y2))

        clamped_w = max(3.0, x2_clamped - x1_clamped)
        clamped_h = max(3.0, y2_clamped - y1_clamped)

        clamped_cx = (x1_clamped + x2_clamped) / 2.0
        clamped_cy = (y1_clamped + y2_clamped) / 2.0

        box_area_calc = clamped_w * clamped_h
        density = comp_events / box_area_calc if box_area_calc > 0 else 0.0
        aspect = clamped_w / clamped_h if clamped_h > 0 else 1.0

        results.append(
            {
                "center_x": clamped_cx,
                "center_y": clamped_cy,
                "width": clamped_w,
                "height": clamped_h,
                "area": comp_area,
                "events": comp_events,
                "density": density,
                "aspect": aspect,
            }
        )

    return results
