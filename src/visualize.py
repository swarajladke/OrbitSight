"""Video Visualization Utility for Neuromorphic Event Streams, Predictions, and Ground Truth.

Renders per-window 2D event accumulation maps as video frames with predicted
and ground-truth bounding boxes overlaid. Operates strictly in headless mode.
"""

import argparse
import csv
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.common import WINDOW_US, infer_resolution, iter_windows, load_events, sequence_name_from_npy


def load_bounding_boxes(
    file_path: Path,
) -> Dict[int, List[Tuple[float, float, float, float, float, Optional[int]]]]:
    """Load bounding boxes indexed by window_start_timestamp_us.
    
    Returns mapping: window_start_ts -> list of (center_x, center_y, width, height, confidence, track_id)
    """
    boxes_by_ts: Dict[int, List[Tuple[float, float, float, float, float, Optional[int]]]] = {}
    if not file_path.exists():
        return boxes_by_ts

    with open(file_path, "r", encoding="utf-8") as f:
        rdr = csv.DictReader(f, delimiter="\t")
        for r in rdr:
            ws = int(r["window_start_timestamp_us"])
            cx = float(r["center_x"])
            cy = float(r["center_y"])
            w = float(r["width"])
            h = float(r["height"])
            conf = float(r.get("confidence", 1.0))
            track_id = int(r["track_id"]) if "track_id" in r and r["track_id"] != "" else None

            if ws not in boxes_by_ts:
                boxes_by_ts[ws] = []
            boxes_by_ts[ws].append((cx, cy, w, h, conf, track_id))

    return boxes_by_ts


def render_video(
    npy_path: Path,
    pred_path: Optional[Path],
    gt_path: Optional[Path],
    out_path: Path,
    fps: int = 25,
    max_windows: Optional[int] = None,
) -> int:
    """Render MP4 video of event stream with optional prediction and ground-truth overlays."""
    events = load_events(npy_path)
    seq_name = sequence_name_from_npy(npy_path)
    width, height = infer_resolution(seq_name, events[:, 0], events[:, 1])

    preds_by_ts = load_bounding_boxes(pred_path) if pred_path else {}
    gt_by_ts = load_bounding_boxes(gt_path) if gt_path else {}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, float(fps), (width, height), isColor=True)

    if not writer.isOpened():
        print(f"Error: Could not open VideoWriter for {out_path}", file=sys.stderr)
        return 0

    frame_count = 0

    for w_idx, (ws_ts, we_ts, w_events) in enumerate(iter_windows(events, window_us=WINDOW_US)):
        if max_windows is not None and w_idx >= max_windows:
            break

        count_img = np.zeros((height, width), dtype=np.uint8)
        if len(w_events) > 0:
            xs = w_events[:, 0].astype(np.int32)
            ys = w_events[:, 1].astype(np.int32)
            valid = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
            xs, ys = xs[valid], ys[valid]

            accum = np.zeros((height, width), dtype=np.int32)
            np.add.at(accum, (ys, xs), 1)

            # Log-scale normalize for visual clarity
            log_accum = np.log1p(accum.astype(np.float32))
            max_val = log_accum.max()
            if max_val > 0:
                count_img = np.clip((log_accum / max_val) * 255.0, 0, 255).astype(np.uint8)

        # Convert to 3-channel BGR image
        frame = cv2.cvtColor(count_img, cv2.COLOR_GRAY2BGR)

        # Draw Ground Truth Boxes in Green (BGR: 0, 255, 0)
        if ws_ts in gt_by_ts:
            for cx, cy, bw, bh, _, _ in gt_by_ts[ws_ts]:
                x1 = int(cx - bw / 2.0)
                y1 = int(cy - bh / 2.0)
                x2 = int(cx + bw / 2.0)
                y2 = int(cy + bh / 2.0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
                cv2.putText(frame, "GT", (x1, max(12, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)

        # Draw Predicted Boxes in Red/Cyan (BGR: 0, 165, 255 or 0, 0, 255)
        if ws_ts in preds_by_ts:
            for cx, cy, bw, bh, conf, track_id in preds_by_ts[ws_ts]:
                x1 = int(cx - bw / 2.0)
                y1 = int(cy - bh / 2.0)
                x2 = int(cx + bw / 2.0)
                y2 = int(cy + bh / 2.0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 140, 255), 1)
                label = f"{conf:.2f}"
                if track_id is not None:
                    label = f"ID:{track_id} {label}"
                cv2.putText(frame, label, (x1, min(height - 4, y2 + 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 140, 255), 1)

        # Add sequence and window overlay in top-left
        info_text = f"{seq_name} | win {w_idx}"
        cv2.putText(frame, info_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        writer.write(frame)
        frame_count += 1

    writer.release()
    print(f"Rendered {frame_count} frames to {out_path} ({width}x{height} @ {fps} fps).", flush=True)
    return frame_count


def main() -> None:
    """CLI entrypoint for visualization script."""
    parser = argparse.ArgumentParser(description="Render event stream video with detections and GT")
    parser.add_argument("--npy", type=str, required=True, help="Path to *_labeled_events.npy")
    parser.add_argument("--pred", type=str, default=None, help="Path to prediction TSV file (*_pred.txt)")
    parser.add_argument("--gt", type=str, default=None, help="Path to ground-truth TSV file (*_bb_windows_40ms.txt)")
    parser.add_argument("--out", type=str, required=True, help="Output MP4 video path")
    parser.add_argument("--fps", type=int, default=25, help="Video frames per second (default: 25)")
    parser.add_argument("--max-windows", type=int, default=None, help="Max windows to render")

    args = parser.parse_args()

    npy_path = Path(args.npy).resolve()
    pred_path = Path(args.pred).resolve() if args.pred else None
    gt_path = Path(args.gt).resolve() if args.gt else None
    out_path = Path(args.out).resolve()

    render_video(
        npy_path=npy_path,
        pred_path=pred_path,
        gt_path=gt_path,
        out_path=out_path,
        fps=args.fps,
        max_windows=args.max_windows,
    )


if __name__ == "__main__":
    main()
