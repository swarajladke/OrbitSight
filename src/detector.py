"""Object detector implementation for event count maps with component splitting, extent box mode, and configurable NMS."""

from typing import Any, Dict, List, Tuple
import cv2
import numpy as np

from src.common import int_percentile, resolve_effective_config
from src.nms import apply_nms


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
        List of detection dictionaries with box properties (all surviving components).
    """
    nonzero_vals = count_img[count_img > 0]
    num_nonzero = len(nonzero_vals)
    if num_nonzero < 4:
        return []

    # Known sensor reference diagonals (width x height)
    # EVK4: 1280x720, DVX: 640x480, DAVIS: 346x260
    known_sensors = {
        "EVK4": (1280, 720, float(np.hypot(1280, 720)), (52.0, 56.0), (20.0, 80.0), (3.8, 10.0)),
        "DVX": (640, 480, float(np.hypot(640, 480)), (18.0, 18.0), (12.0, 60.0), (1.5, 4.0)),
        "DAVIS": (346, 260, float(np.hypot(346, 260)), (10.0, 12.0), (4.0, 30.0), (1.1, 1.5)),
    }

    curr_diag = float(np.hypot(width, height))

    # Exact known resolution match check first
    if width == 1280 and height == 720:
        sensor_name = "EVK4"
        diag_scale = 1.0
    elif width == 640 and height == 480:
        sensor_name = "DVX"
        diag_scale = 1.0
    elif width == 346 and height == 260:
        sensor_name = "DAVIS"
        diag_scale = 1.0
    else:
        # Unknown sensor: choose nearest sensor by diagonal distance and scale box geometry
        sensor_name = min(known_sensors.keys(), key=lambda k: abs(curr_diag - known_sensors[k][2]))
        ref_diag = known_sensors[sensor_name][2]
        diag_scale = curr_diag / ref_diag

    _, _, _, (ref_bw, ref_bh), (ref_min_d, ref_max_d), (ref_sc, ref_pad) = known_sensors[sensor_name]
    def_bw = ref_bw * diag_scale
    def_bh = ref_bh * diag_scale
    def_min_dim = ref_min_d * diag_scale
    def_max_dim = ref_max_d * diag_scale
    def_scale, def_pad = ref_sc, ref_pad

    eff = resolve_effective_config(cfg, sensor_name)
    if diag_scale != 1.0:
        eff = eff.copy()
        if "box_w" in eff:
            eff["box_w"] = float(eff["box_w"]) * diag_scale
        else:
            eff["box_w"] = def_bw
        if "box_h" in eff:
            eff["box_h"] = float(eff["box_h"]) * diag_scale
        else:
            eff["box_h"] = def_bh
        if "min_dim" in eff:
            eff["min_dim"] = float(eff["min_dim"]) * diag_scale
        if "max_dim" in eff:
            eff["max_dim"] = float(eff["max_dim"]) * diag_scale

    base_percentile = float(eff.get("percentile", 97.5))

    if num_nonzero > 1000:
        percentile = min(99.0, base_percentile + (num_nonzero - 1000) / 500.0)
    else:
        percentile = base_percentile

    raw_thresh = float(int_percentile(nonzero_vals, percentile))
    thresh = max(1.0, raw_thresh)

    binary = (count_img >= thresh).astype(np.uint8)
    if cv2.countNonZero(binary) < 4:
        return []

    open_k = int(eff.get("open_kernel", 2))
    if open_k > 1:
        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (open_k, open_k))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open)
        if cv2.countNonZero(binary) < 4:
            return []

    b_predilate = binary.copy()

    dilate_k = int(eff.get("dilate_kernel", 3))
    if dilate_k > 1:
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (dilate_k, dilate_k))
        binary = cv2.dilate(binary, kernel_dilate)
        if cv2.countNonZero(binary) < 4:
            return []

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )

    max_area_frac = float(eff.get("max_area_frac", 0.02))
    max_area_pixels = max_area_frac * width * height
    min_events = float(eff.get("min_events_in_box", 6))

    box_mode = str(eff.get("box_mode", "scale")).lower()
    centroid_mode = str(eff.get("centroid_mode", "component")).lower()
    centroid_on_predilate = bool(eff.get("centroid_on_predilation_mask", False))

    box_scale = float(eff.get("box_scale", def_scale))
    box_pad = float(eff.get("box_pad", def_pad))
    box_w_cfg = float(eff.get("box_w", def_bw))
    box_h_cfg = float(eff.get("box_h", def_bh))

    min_dim = float(eff.get("min_dim", def_min_dim))
    max_dim = float(eff.get("max_dim", def_max_dim))
    extent_scale = float(eff.get("extent_scale", 1.0))
    extent_pad = float(eff.get("extent_pad", 2.0))

    nms_stage = str(eff.get("nms_stage", "pipeline")).lower()
    nms_iou_val = eff.get("nms_iou", 0.3)
    max_comp = int(eff.get("max_components_per_window", 2000))

    # Component list after component splitting
    final_components: List[Tuple[float, float, float, float, int]] = []

    for label_idx in range(1, num_labels):
        comp_area = float(stats[label_idx, cv2.CC_STAT_AREA])
        x_box = int(stats[label_idx, cv2.CC_STAT_LEFT])
        y_box = int(stats[label_idx, cv2.CC_STAT_TOP])
        w_box = int(stats[label_idx, cv2.CC_STAT_WIDTH])
        h_box = int(stats[label_idx, cv2.CC_STAT_HEIGHT])

        if comp_area > max_area_pixels:
            # Component Splitting Path: re-threshold sub-image bounding box at higher percentile
            sub_count = count_img[y_box : y_box + h_box, x_box : x_box + w_box]
            sub_nonzero = sub_count[sub_count > 0]

            if len(sub_nonzero) >= 4:
                split_perc = min(99.5, percentile + 1.5)
                split_thresh = float(int_percentile(sub_nonzero, split_perc))

                sub_binary = (sub_count >= split_thresh).astype(np.uint8)
                n_sub, l_sub, s_sub, _ = cv2.connectedComponentsWithStats(
                    sub_binary, connectivity=8
                )

                for sub_i in range(1, n_sub):
                    sub_area = float(s_sub[sub_i, cv2.CC_STAT_AREA])
                    if sub_area <= max_area_pixels:
                        sx = x_box + int(s_sub[sub_i, cv2.CC_STAT_LEFT])
                        sy = y_box + int(s_sub[sub_i, cv2.CC_STAT_TOP])
                        sw = int(s_sub[sub_i, cv2.CC_STAT_WIDTH])
                        sh = int(s_sub[sub_i, cv2.CC_STAT_HEIGHT])
                        final_components.append((float(sx), float(sy), float(sw), float(sh), sub_i))
            continue

        final_components.append((float(x_box), float(y_box), float(w_box), float(h_box), label_idx))

    results: List[Dict[str, float]] = []

    for x_box, y_box, w_box, h_box, label_idx in final_components:
        x_int, y_int = int(x_box), int(y_box)
        w_int, h_int = int(w_box), int(h_box)

        sub_img = count_img[y_int : y_int + h_int, x_int : x_int + w_int]
        sub_labels = labels[y_int : y_int + h_int, x_int : x_int + w_int]
        sub_mask = sub_labels == label_idx

        if sub_mask.any():
            comp_events = float(sub_img[sub_mask].sum())
        else:
            comp_events = float(sub_img.sum())

        if comp_events < min_events:
            continue

        center_comp_x = x_box + w_box / 2.0
        center_comp_y = y_box + h_box / 2.0

        if centroid_mode == "weighted" and comp_events > 0 and sub_mask.any():
            if centroid_on_predilate:
                sub_predil = b_predilate[y_int : y_int + h_int, x_int : x_int + w_int]
                predil_mask = sub_mask & (sub_predil > 0)
                if predil_mask.any():
                    ys, xs = np.where(predil_mask)
                    weights = sub_img[predil_mask]
                    w_sum = weights.sum()
                    if w_sum > 0:
                        center_x = x_box + float((xs * weights).sum() / w_sum)
                        center_y = y_box + float((ys * weights).sum() / w_sum)
                    else:
                        center_x, center_y = center_comp_x, center_comp_y
                else:
                    ys, xs = np.where(sub_mask)
                    weights = sub_img[sub_mask]
                    center_x = x_box + float((xs * weights).sum() / comp_events)
                    center_y = y_box + float((ys * weights).sum() / comp_events)
            else:
                ys, xs = np.where(sub_mask)
                weights = sub_img[sub_mask]
                center_x = x_box + float((xs * weights).sum() / comp_events)
                center_y = y_box + float((ys * weights).sum() / comp_events)
        else:
            center_x = center_comp_x
            center_y = center_comp_y

        if box_mode == "fixed":
            new_w = box_w_cfg
            new_h = box_h_cfg
        elif box_mode == "extent":
            raw_w = w_box * extent_scale + 2.0 * extent_pad
            raw_h = h_box * extent_scale + 2.0 * extent_pad
            new_w = max(min_dim, min(max_dim, raw_w))
            new_h = max(min_dim, min(max_dim, raw_h))
        else:  # scale mode
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

        conf = min(1.0, max(0.01, density * 1.5))

        results.append(
            {
                "center_x": clamped_cx,
                "center_y": clamped_cy,
                "width": clamped_w,
                "height": clamped_h,
                "area": w_box * h_box,
                "events": comp_events,
                "density": density,
                "aspect": aspect,
                "confidence": conf,
            }
        )

    # Step 4: Hang guard max_components_per_window
    if len(results) > max_comp:
        print(f"[WARN] max_components_per_window hit: kept {max_comp} of {len(results)} candidates")
        results.sort(key=lambda b: b["density"], reverse=True)
        results = results[:max_comp]

    # Optional detector-stage NMS
    if nms_stage == "detector" and nms_iou_val is not None:
        try:
            nms_thresh = float(nms_iou_val)
            results = apply_nms(results, nms_thresh)
        except (TypeError, ValueError):
            pass

    return results
