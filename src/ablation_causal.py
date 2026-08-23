"""Causal-variant ablation script (Step 5):
Trains and evaluates causal candidate scorer and objectness gate with next-window lookahead features zeroed.
"""

from pathlib import Path
import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from src.features import FEATURE_NAMES
from src.metrics import compute_ap, iou, match_predictions
from src.nms import apply_nms
from src.train_reranker import compute_candidate_ground_truth_labels


def main() -> None:
    cache_path = Path("models/train_seq_extracted_cache_pre_geometry.joblib")
    if not cache_path.exists():
        cache_path = Path("models/train_seq_extracted_cache.joblib")
    if not cache_path.exists():
        print(f"Error: Cache not found")
        return

    seq_data_dict = joblib.load(cache_path)
    train_seqs = sorted(list(seq_data_dict.keys()))
    print(f"Loaded cache with {len(train_seqs)} sequences.")

    # 1. Prepare causal candidate training dataset (zero disp_next, hits = 1 + has_prev)
    cand_X_tr, cand_y_tr = [], []
    for s_name in train_seqs:
        seq_entry = seq_data_dict[s_name]
        w_cands = seq_entry["window_cands"]
        w_times = seq_entry["window_times"]
        gt_rows = seq_entry["gt_rows"]
        cand_labels, _ = compute_candidate_ground_truth_labels(w_cands, w_times, gt_rows, iou_thr=0.5)

        flat_cands = [c for wc in w_cands for c in wc]
        for c, lbl in zip(flat_cands, cand_labels):
            f = list(c["features_13"])
            has_p = f[7] > 0
            f[6] = 2.0 if has_p else 1.0  # hits
            f[8] = 0.0  # disp_next
            f[9] = f[7]  # speed = disp_prev
            f[10] = 1.0 if has_p else 0.5  # dir_consistency
            cand_X_tr.append(f)
            cand_y_tr.append(lbl)

    cand_X_tr = np.array(cand_X_tr, dtype=np.float32)
    cand_y_tr = np.array(cand_y_tr, dtype=int)
    print(f"Training causal candidate scorer on {len(cand_X_tr)} samples ({np.sum(cand_y_tr)} positives)...")
    causal_scorer = HistGradientBoostingClassifier(
        max_iter=150,
        max_depth=6,
        learning_rate=0.08,
        min_samples_leaf=20,
        class_weight="balanced",
        random_state=42,
    )
    causal_scorer.fit(cand_X_tr, cand_y_tr)

    # 2. Prepare causal window objectness dataset (zero 7 nxt_* features)
    win_X_tr, win_y_tr = [], []
    for s_name in train_seqs:
        seq_entry = seq_data_dict[s_name]
        w_cands = seq_entry["window_cands"]
        w_times = seq_entry["window_times"]
        gt_rows = seq_entry["gt_rows"]
        _, win_occupied = compute_candidate_ground_truth_labels(w_cands, w_times, gt_rows, iou_thr=0.5)

        # Re-score candidates with causal scorer in batch
        flat_seq_cands = [c for wc in w_cands for c in wc]
        if flat_seq_cands:
            feats_batch = []
            for c in flat_seq_cands:
                f = list(c["features_13"])
                has_p = f[7] > 0
                f[6] = 2.0 if has_p else 1.0
                f[8] = 0.0
                f[9] = f[7]
                f[10] = 1.0 if has_p else 0.5
                feats_batch.append(f)
            flat_scores = causal_scorer.predict_proba(np.array(feats_batch, dtype=np.float32))[:, 1]
        else:
            flat_scores = np.array([], dtype=np.float32)

        w_causal_scores = []
        sc_ptr = 0
        for wc in w_cands:
            sc_list = [float(s) for s in flat_scores[sc_ptr : sc_ptr + len(wc)]]
            sc_ptr += len(wc)
            w_causal_scores.append(sc_list)

        full_feats = seq_entry["full_win_feats"]
        nw = len(full_feats)
        for w_idx in range(nw):
            cur = full_feats[w_idx]
            prev = full_feats[w_idx - 1] if w_idx > 0 else {k: 0.0 for k in cur}

            c_scores = w_causal_scores[w_idx]
            c_mean = float(np.mean(c_scores)) if c_scores else 0.0
            c_max = float(np.max(c_scores)) if c_scores else 0.0

            p_scores = w_causal_scores[w_idx - 1] if w_idx > 0 else []
            p_mean = float(np.mean(p_scores)) if p_scores else 0.0
            p_max = float(np.max(p_scores)) if p_scores else 0.0

            w_vec = [
                cur["win_total_events"], cur["win_num_components"], cur["win_x_std"], cur["win_y_std"], cur["win_max_comp_events"], c_mean, c_max,
                prev["win_total_events"], prev["win_num_components"], prev["win_x_std"], prev["win_y_std"], prev["win_max_comp_events"], p_mean, p_max,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ]
            win_X_tr.append(w_vec)
            win_y_tr.append(win_occupied[w_idx])

    win_X_tr = np.array(win_X_tr, dtype=np.float32)
    win_y_tr = np.array(win_y_tr, dtype=int)
    print(f"Training causal window objectness gate on {len(win_X_tr)} windows ({np.sum(win_y_tr)} positives)...")
    causal_obj = HistGradientBoostingClassifier(
        max_iter=150,
        learning_rate=0.08,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        random_state=42,
    )
    causal_obj.fit(win_X_tr, win_y_tr)

    # 3. Evaluate causal variant across all 17 training sequences
    seq_aps = []
    tot_tp, tot_fp, tot_fn = 0, 0, 0
    sparse_aps, dense_aps = [], []

    for s_name in train_seqs:
        seq_entry = seq_data_dict[s_name]
        w_cands = seq_entry["window_cands"]
        w_times = seq_entry["window_times"]
        gt_rows = seq_entry["gt_rows"]
        full_feats = seq_entry["full_win_feats"]
        nw = len(w_times)

        # Re-score candidates in batch
        flat_seq_cands = [c for wc in w_cands for c in wc]
        if flat_seq_cands:
            feats_batch = []
            for c in flat_seq_cands:
                f = list(c["features_13"])
                has_p = f[7] > 0
                f[6] = 2.0 if has_p else 1.0
                f[8] = 0.0
                f[9] = f[7]
                f[10] = 1.0 if has_p else 0.5
                feats_batch.append(f)
            flat_scores = causal_scorer.predict_proba(np.array(feats_batch, dtype=np.float32))[:, 1]
        else:
            flat_scores = np.array([], dtype=np.float32)

        w_causal_cands = []
        w_causal_scores = []
        sc_ptr = 0
        for wc in w_cands:
            cand_tuples = []
            sc_list = []
            for c, sc in zip(wc, flat_scores[sc_ptr : sc_ptr + len(wc)]):
                cand_tuples.append((c, float(sc)))
                sc_list.append(float(sc))
            sc_ptr += len(wc)
            w_causal_cands.append(cand_tuples)
            w_causal_scores.append(sc_list)

        win_X = []
        for w_idx in range(nw):
            cur = full_feats[w_idx]
            prev = full_feats[w_idx - 1] if w_idx > 0 else {k: 0.0 for k in cur}

            c_scores = w_causal_scores[w_idx]
            c_mean = float(np.mean(c_scores)) if c_scores else 0.0
            c_max = float(np.max(c_scores)) if c_scores else 0.0

            p_scores = w_causal_scores[w_idx - 1] if w_idx > 0 else []
            p_mean = float(np.mean(p_scores)) if p_scores else 0.0
            p_max = float(np.max(p_scores)) if p_scores else 0.0

            w_vec = [
                cur["win_total_events"], cur["win_num_components"], cur["win_x_std"], cur["win_y_std"], cur["win_max_comp_events"], c_mean, c_max,
                prev["win_total_events"], prev["win_num_components"], prev["win_x_std"], prev["win_y_std"], prev["win_max_comp_events"], p_mean, p_max,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ]
            win_X.append(w_vec)

        p_objs = causal_obj.predict_proba(np.array(win_X, dtype=np.float32))[:, 1]

        preds = []
        for w_idx in range(nw):
            p_obj = float(p_objs[w_idx])
            boxes = []
            for c, sc in w_causal_cands[w_idx]:
                g_conf = sc * p_obj
                if g_conf >= 0.30:
                    b = dict(c)
                    b["confidence"] = g_conf
                    boxes.append(b)
            if boxes:
                nms_b = apply_nms(boxes, 0.30)
                nms_b.sort(key=lambda x: float(x.get("confidence", 0.0)), reverse=True)
                for b in nms_b[:1]:
                    w_start, w_end = w_times[w_idx]
                    preds.append((
                        w_start, w_end,
                        int(round(b["center_x"])), int(round(b["center_y"])),
                        int(round(b["width"])), int(round(b["height"])),
                        round(float(b["confidence"]), 4),
                    ))

        n_gt = len(gt_rows)
        if n_gt > 0:
            tp_arr, fp_arr = match_predictions(gt_rows, preds, iou_thresh=0.5)
            ap = compute_ap(tp_arr, fp_arr, n_gt)
            n_tp = int(tp_arr.sum()) if len(tp_arr) > 0 else 0
            n_fp = int(fp_arr.sum()) if len(fp_arr) > 0 else 0
            n_fn = n_gt - n_tp
        else:
            ap = 0.0
            n_tp = 0
            n_fp = len(preds)
            n_fn = 0

        seq_aps.append(ap)
        tot_tp += n_tp
        tot_fp += n_fp
        tot_fn += n_fn
        if n_gt <= 50:
            sparse_aps.append(ap)
        else:
            dense_aps.append(ap)

    prec = tot_tp / (tot_tp + tot_fp) if (tot_tp + tot_fp) > 0 else 0.0
    rec = tot_tp / (tot_tp + tot_fn) if (tot_tp + tot_fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    train_map = float(np.mean(seq_aps))
    sp_map = float(np.mean(sparse_aps))
    dn_map = float(np.mean(dense_aps))

    print("\n" + "=" * 65)
    print("  CAUSAL STREAMING ABLATION RESULTS (17 Train Sequences)")
    print("=" * 65)
    print(f"Train mAP:    {train_map:.6f}  (Baseline: 0.163628, Delta: {train_map - 0.163628:+.6f})")
    print(f"Sparse mAP:   {sp_map:.6f}  (Baseline: 0.098205, Delta: {sp_map - 0.098205:+.6f})")
    print(f"Dense mAP:    {dn_map:.6f}  (Baseline: 0.257091, Delta: {dn_map - 0.257091:+.6f})")
    print(f"Precision:    {prec:.6f}  (Baseline: 0.422441, Delta: {prec - 0.422441:+.6f})")
    print(f"Recall:       {rec:.6f}  (Baseline: 0.330892, Delta: {rec - 0.330892:+.6f})")
    print(f"F1 Score:     {f1:.6f}  (Baseline: 0.371104, Delta: {f1 - 0.371104:+.6f})")
    print(f"Counts:       TP = {tot_tp} (was 5060), FP = {tot_fp} (was 6918), FN = {tot_fn} (was 10232)")
    print("=" * 65)


if __name__ == "__main__":
    main()
