"""In-process detection evaluation metrics matching official evaluate.py semantics.

AP Interpolation Method:
    This module implements ALL-POINT CONTINUOUS precision-recall curve integration
    (VOC/COCO style continuous P-R curve area integration).
    Recalls and precisions are extended with boundary conditions (0, 1) and (last_recall, 0),
    precisions are smoothed to be monotonically non-increasing from right to left,
    and exact trapezoidal/step areas under the curve are integrated. This matches
    `evaluate.py` (lines 158-177) exactly.
"""

import argparse
import csv
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple, Union
import numpy as np
from tabulate import tabulate

IOU_THRESHOLD: float = 0.5


def cx_cy_wh_to_xyxy(
    cx: float, cy: float, w: float, h: float
) -> Tuple[float, float, float, float]:
    """Convert (center_x, center_y, width, height) to 1-based pixel corner coordinates (x1, y1, x2, y2)."""
    x1 = cx - (w - 1.0) / 2.0
    y1 = cy - (h - 1.0) / 2.0
    x2 = x1 + w - 1.0
    y2 = y1 + h - 1.0
    return x1, y1, x2, y2


def iou(box_a: Tuple[float, float, float, float], box_b: Tuple[float, float, float, float]) -> float:
    """Compute Intersection-over-Union (IoU) between two boxes given as (cx, cy, w, h)."""
    ax1, ay1, ax2, ay2 = cx_cy_wh_to_xyxy(*box_a)
    bx1, by1, bx2, by2 = cx_cy_wh_to_xyxy(*box_b)

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)

    inter = max(0.0, ix2 - ix1 + 1.0) * max(0.0, iy2 - iy1 + 1.0)

    area_a = (ax2 - ax1 + 1.0) * (ay2 - ay1 + 1.0)
    area_b = (bx2 - bx1 + 1.0) * (by2 - by1 + 1.0)
    union = area_a + area_b - inter

    return float(inter / union) if union > 0 else 0.0


def windows_overlap(ws_a: int, we_a: int, ws_b: int, we_b: int) -> bool:
    """Check if two time windows overlap."""
    return ws_a < we_b and we_a > ws_b


def match_predictions(
    gt_list: List[Tuple[int, int, int, int, int, int]],
    pred_list: List[Tuple[int, int, int, int, int, int, float]],
    iou_thresh: float = IOU_THRESHOLD,
) -> Tuple[np.ndarray, np.ndarray]:
    """Match confidence-ranked predictions against GT boxes using window indexing.

    Args:
        gt_list: List of GT rows (start_us, end_us, cx, cy, w, h).
        pred_list: List of prediction rows (start_us, end_us, cx, cy, w, h, confidence).
        iou_thresh: Minimum IoU threshold to consider a match.

    Returns:
        Tuple of (tp_array, fp_array) binary arrays.
    """
    gt_matched = [False] * len(gt_list)
    tp: List[int] = []
    fp: List[int] = []

    # Fast O(1) window-indexed GT lookup
    gt_by_window: Dict[int, List[int]] = {}
    for idx, gt in enumerate(gt_list):
        gt_by_window.setdefault(gt[0], []).append(idx)

    # Ensure predictions are sorted by confidence descending
    sorted_preds = sorted(pred_list, key=lambda r: r[6], reverse=True)

    for pred in sorted_preds:
        ws_p, we_p, cx_p, cy_p, w_p, h_p, _ = pred
        best_iou = 0.0
        best_idx = -1

        candidate_indices = gt_by_window.get(ws_p, [])
        if not candidate_indices:
            # Fallback for slight boundary overlap
            candidate_indices = [
                j for j, gt in enumerate(gt_list)
                if not gt_matched[j] and windows_overlap(ws_p, we_p, gt[0], gt[1])
            ]

        for j in candidate_indices:
            if gt_matched[j]:
                continue
            ws_g, we_g, cx_g, cy_g, w_g, h_g = gt_list[j]
            score = iou((cx_p, cy_p, w_p, h_p), (cx_g, cy_g, w_g, h_g))
            if score > best_iou:
                best_iou = score
                best_idx = j

        if best_iou >= iou_thresh and best_idx >= 0:
            tp.append(1)
            fp.append(0)
            gt_matched[best_idx] = True
        else:
            tp.append(0)
            fp.append(1)

    return np.array(tp, dtype=int), np.array(fp, dtype=int)


def compute_ap(tp: np.ndarray, fp: np.ndarray, n_gt: int) -> float:
    """Compute Average Precision (AP) using all-point continuous P-R integration."""
    if n_gt == 0:
        return float("nan")

    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)

    recalls = cum_tp / float(n_gt)
    precisions = cum_tp / (cum_tp + cum_fp + 1e-9)

    recalls = np.concatenate([[0.0], recalls, [recalls[-1] if len(recalls) else 0.0]])
    precisions = np.concatenate([[1.0], precisions, [0.0]])

    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])

    idx = np.where(recalls[1:] != recalls[:-1])[0]
    ap = np.sum((recalls[idx + 1] - recalls[idx]) * precisions[idx + 1])
    return float(ap)


def compute_prf1(tp_total: int, fp_total: int, fn_total: int) -> Tuple[float, float, float]:
    """Compute Precision, Recall, and F1 score."""
    precision = (
        float(tp_total) / float(tp_total + fp_total)
        if (tp_total + fp_total) > 0
        else 0.0
    )
    recall = (
        float(tp_total) / float(tp_total + fn_total)
        if (tp_total + fn_total) > 0
        else 0.0
    )
    f1 = (
        (2.0 * precision * recall) / (precision + recall)
        if (precision + recall) > 0.0
        else 0.0
    )
    return precision, recall, f1


def evaluate_sequence(
    gt_rows: List[Tuple[int, int, int, int, int, int]],
    pred_rows: List[Tuple[int, int, int, int, int, int, float]],
    iou_thr: float = IOU_THRESHOLD,
) -> Dict[str, Any]:
    """Evaluate a single sequence given GT and Prediction rows."""
    n_gt = len(gt_rows)
    n_pred = len(pred_rows)

    if n_gt == 0 and n_pred == 0:
        return {
            "n_gt": 0,
            "n_pred": 0,
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "precision": None,
            "recall": None,
            "f1": None,
            "ap": float("nan"),
        }

    tp_arr, fp_arr = match_predictions(gt_rows, pred_rows, iou_thr)
    tp = int(tp_arr.sum())
    fp = int(fp_arr.sum())
    fn = n_gt - tp

    prec, rec, f1 = compute_prf1(tp, fp, fn)
    ap = compute_ap(tp_arr, fp_arr, n_gt)

    return {
        "n_gt": n_gt,
        "n_pred": n_pred,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "ap": ap,
    }


def evaluate_all(
    dataset_dir: Union[str, Path],
    pred_dir: Union[str, Path],
    iou_thr: float = IOU_THRESHOLD,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """Evaluate all sequences in dataset_dir against pred_dir."""
    dataset_path = Path(dataset_dir)
    pred_path = Path(pred_dir)

    gt_files = sorted(list(dataset_path.rglob("*_bb_windows_40ms.txt")))
    per_sequence: Dict[str, Dict[str, Any]] = {}

    all_tp = 0
    all_fp = 0
    all_fn = 0
    all_aps: List[float] = []

    for gt_file in gt_files:
        seq_name = gt_file.name.replace("_bb_windows_40ms.txt", "")
        p_alt1 = pred_path / f"{seq_name}_pred.txt"
        p_alt2 = pred_path / f"{seq_name}_bb_windows_40ms.txt"

        if p_alt1.exists():
            target_pred = p_alt1
        elif p_alt2.exists():
            target_pred = p_alt2
        else:
            continue

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

        pred_rows = []
        with open(target_pred, "r", encoding="utf-8") as f:
            rdr = csv.DictReader(f, delimiter="\t")
            for r in rdr:
                conf = (
                    float(r["confidence"])
                    if "confidence" in r and r["confidence"] is not None
                    else 1.0
                )
                pred_rows.append(
                    (
                        int(r["window_start_timestamp_us"]),
                        int(r["window_end_timestamp_us"]),
                        int(r["center_x"]),
                        int(r["center_y"]),
                        int(r["width"]),
                        int(r["height"]),
                        conf,
                    )
                )

        seq_res = evaluate_sequence(gt_rows, pred_rows, iou_thr=iou_thr)
        per_sequence[seq_name] = seq_res

        all_tp += seq_res["tp"]
        all_fp += seq_res["fp"]
        all_fn += seq_res["fn"]
        if seq_res["ap"] is not None and not np.isnan(seq_res["ap"]):
            all_aps.append(seq_res["ap"])

    overall_prec, overall_rec, overall_f1 = compute_prf1(all_tp, all_fp, all_fn)
    mAP = float(np.mean(all_aps)) if all_aps else float("nan")

    overall = {
        "mAP": mAP,
        "precision": overall_prec,
        "recall": overall_rec,
        "f1": overall_f1,
        "tp": all_tp,
        "fp": all_fp,
        "fn": all_fn,
        "evaluated_sequences": len(per_sequence),
    }

    return per_sequence, overall


def run_verification(dataset_dir: str, pred_dir: str, loader_dir: str) -> None:
    """Run in-process evaluation and compare side-by-side with official evaluate.py script."""
    print("[INFO] Running in-process evaluation metrics...", flush=True)
    _per_seq_inproc, overall_inproc = evaluate_all(dataset_dir, pred_dir)

    loader_path = Path(loader_dir).resolve()
    if loader_path.exists() and str(loader_path) not in sys.path:
        sys.path.insert(0, str(loader_path))

    try:
        import evaluate as official_eval

        print("[INFO] Executing official evaluate.py logic for verification...", flush=True)
        train_gt = Path(dataset_dir) / "Training_sets"
        test_gt = Path(dataset_dir) / "Testing_sets"

        official_results = []
        if train_gt.exists():
            official_results += official_eval.evaluate_dir(
                str(train_gt), pred_dir, IOU_THRESHOLD, "Training"
            )
        if test_gt.exists():
            official_results += official_eval.evaluate_dir(
                str(test_gt), pred_dir, IOU_THRESHOLD, "Testing"
            )
        if not official_results:
            official_results += official_eval.evaluate_dir(
                dataset_dir, pred_dir, IOU_THRESHOLD, ""
            )

        off_tp = sum(r["tp"] for r in official_results)
        off_fp = sum(r["fp"] for r in official_results)
        off_fn = sum(r["fn"] for r in official_results)
        off_aps = [
            r["ap"]
            for r in official_results
            if r["ap"] is not None and not np.isnan(r["ap"])
        ]

        off_prec, off_rec, off_f1 = official_eval.compute_prf1(
            off_tp, off_fp, off_fn
        )
        off_map = float(np.mean(off_aps)) if off_aps else float("nan")

        verification_rows = [
            [
                "mAP @ IoU 0.5",
                f"{off_map:.6f}",
                f"{overall_inproc['mAP']:.6f}",
                f"{abs(off_map - overall_inproc['mAP']):.6e}",
            ],
            [
                "Precision",
                f"{off_prec:.6f}",
                f"{overall_inproc['precision']:.6f}",
                f"{abs(off_prec - overall_inproc['precision']):.6e}",
            ],
            [
                "Recall",
                f"{off_rec:.6f}",
                f"{overall_inproc['recall']:.6f}",
                f"{abs(off_rec - overall_inproc['recall']):.6e}",
            ],
            [
                "F1 Score",
                f"{off_f1:.6f}",
                f"{overall_inproc['f1']:.6f}",
                f"{abs(off_f1 - overall_inproc['f1']):.6e}",
            ],
            [
                "Total TP",
                off_tp,
                overall_inproc["tp"],
                abs(off_tp - overall_inproc["tp"]),
            ],
            [
                "Total FP",
                off_fp,
                overall_inproc["fp"],
                abs(off_fp - overall_inproc["fp"]),
            ],
            [
                "Total FN",
                off_fn,
                overall_inproc["fn"],
                abs(off_fn - overall_inproc["fn"]),
            ],
        ]

        print("\n" + "=" * 80, flush=True)
        print("  METRICS VERIFICATION REPORT (Official evaluate.py vs src/metrics.py)", flush=True)
        print("=" * 80, flush=True)
        print(
            tabulate(
                verification_rows,
                headers=[
                    "Metric",
                    "Official evaluate.py",
                    "src/metrics.py",
                    "Abs Difference",
                ],
                tablefmt="grid",
            ),
            flush=True,
        )

        all_exact = all(
            float(r[3]) == 0.0 or float(r[3]) < 1e-6 for r in verification_rows
        )
        if all_exact:
            print("\n[PASS] Perfect agreement! In-process metrics match official evaluate.py exactly.", flush=True)
        else:
            print("\n[WARNING] Discrepancies detected between official and in-process metrics.", flush=True)

    except Exception as e:
        print(f"[ERROR] Could not run official evaluate.py verification: {e}", flush=True)


def main() -> None:
    """CLI main entrypoint."""
    parser = argparse.ArgumentParser(
        description="OrbitSight In-Process Metrics Engine & Verification Tool"
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="../OrbitSight_Dataset",
        help="Path to dataset root directory",
    )
    parser.add_argument(
        "--pred-dir",
        type=str,
        default="predictions",
        help="Path to predictions directory",
    )
    parser.add_argument(
        "--loader-dir",
        type=str,
        default="../OrbitSight_DataLoader",
        help="Path to OrbitSight_DataLoader directory for official script import",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run verification against official evaluate.py script",
    )

    args = parser.parse_args()

    if args.verify:
        run_verification(args.dataset_dir, args.pred_dir, args.loader_dir)
    else:
        per_seq, overall = evaluate_all(args.dataset_dir, args.pred_dir)
        print("\nIn-Process Metrics Summary:", flush=True)
        print(f"  mAP @ 0.5: {overall['mAP']:.4f}", flush=True)
        print(f"  Precision:  {overall['precision']:.4f}", flush=True)
        print(f"  Recall:     {overall['recall']:.4f}", flush=True)
        print(f"  F1 Score:   {overall['f1']:.4f}", flush=True)
        print(f"  Total TP:   {overall['tp']}", flush=True)
        print(f"  Total FP:   {overall['fp']}", flush=True)
        print(f"  Total FN:   {overall['fn']}", flush=True)


if __name__ == "__main__":
    main()
