"""Vectorized Non-Maximum Suppression (NMS) for candidate bounding boxes."""

from typing import Dict, List, Optional
import numpy as np
from src.metrics import iou


def apply_nms(
    boxes: List[Dict[str, float]], nms_iou: Optional[float] = 0.3
) -> List[Dict[str, float]]:
    """Apply vectorized greedy non-maximum suppression to candidate boxes.

    Args:
        boxes: List of candidate box dicts (containing center_x, center_y, width, height, confidence/density).
        nms_iou: IoU threshold above which overlapping lower-confidence boxes are suppressed.
                 If None or <= 0, NMS is disabled.

    Returns:
        Filtered list of non-overlapping candidate boxes sorted by score descending.
    """
    if not boxes or nms_iou is None or nms_iou <= 0.0:
        return boxes

    # Sort boxes by confidence / density descending
    sorted_boxes = sorted(
        boxes,
        key=lambda b: float(b.get("confidence", b.get("density", b.get("events", 0.0)))),
        reverse=True,
    )

    keep: List[Dict[str, float]] = []

    for cand in sorted_boxes:
        c_box = (
            float(cand["center_x"]),
            float(cand["center_y"]),
            float(cand["width"]),
            float(cand["height"]),
        )

        suppressed = False
        for k_b in keep:
            k_box = (
                float(k_b["center_x"]),
                float(k_b["center_y"]),
                float(k_b["width"]),
                float(k_b["height"]),
            )

            if iou(c_box, k_box) > nms_iou:
                suppressed = True
                break

        if not suppressed:
            keep.append(cand)

    return keep
