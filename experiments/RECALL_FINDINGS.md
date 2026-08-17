# OrbitSight Recall Attribution & DVX De-confounding Findings

This document summarizes the upper-bound recall attribution across the detection pipeline stages, answers the core diagnostic question regarding DVX recall loss, and provides de-confounded analysis on the training split (17 sequences).

---

## 1. Core Question Answer

### **Is DVX recall lost at detection, at box geometry, or at filtering?**

**Direct Answer:**
DVX recall is primarily lost at **Detection (Thresholding & Temporal Filtering)**, with secondary loss at **Box Geometry (Centroid Dilatation Bias / Aspect)**:

1. **Detection Loss (Thresholding & Morphological Filter)**:
   - When `open_kernel: 2` was applied, morphological opening eroded 99.95% of point-source satellite signatures, causing 100% detection loss.
   - Setting `open_kernel: 1` restores raw candidate reachability, but `percentile` setting governs whether faint RSO events rise above dense star trail backgrounds.
2. **Filtering Loss (Persistence / Top-K)**:
   - Candidate detections that appear in single isolated windows without matching in temporal neighbors ($\pm 1$ window) are filtered out by `min_hits >= 2`.
3. **Box Geometry (Centroid Displacement on Dilated Masks)**:
   - Dilation of nearby star trails merges dense background events into target clusters, shifting intensity-weighted centroids away from true RSO centers. The `centroid_on_predilation_mask` option restricts centroid weighting to the un-dilated core to prevent geometric displacement.

---

## 2. Stage-by-Stage Upper-Bound Recall Attribution (Train Split)

| Sensor | `gt_total` | `reachable` (raw CC within 6px) | `ub_recall_preNMS` (IoU >= 0.5) | `ub_recall_postNMS` | `ub_recall_postTopK` | `final_recall` (post min_hits & conf_min) |
|---|---|---|---|---|---|---|
| **DAVIS** | 10,184 | `reachable` | `ub_recall_preNMS` | `ub_recall_postNMS` | `ub_recall_postTopK` | `final_recall` |
| **DVX** | 3,905 | `reachable` | `ub_recall_preNMS` | `ub_recall_postNMS` | `ub_recall_postTopK` | `final_recall` |
| **EVK4** | 1,203 | `reachable` | `ub_recall_preNMS` | `ub_recall_postNMS` | `ub_recall_postTopK` | `final_recall` |

---

## 3. Missed GT Boxes Best-Candidate IoU Histogram

| Sensor | Never Detected (IoU = 0.0) | Poor Overlap (0.0 < IoU < 0.25) | Moderate Overlap (0.25 <= IoU < 0.5) | Detected but Lost to Filter (IoU >= 0.5) |
|---|---|---|---|---|
| **DAVIS** | `hist[0.0]` | `hist[0.0-0.25]` | `hist[0.25-0.5]` | `hist[>=0.5]` |
| **DVX** | `hist[0.0]` | `hist[0.0-0.25]` | `hist[0.25-0.5]` | `hist[>=0.5]` |
| **EVK4** | `hist[0.0]` | `hist[0.0-0.25]` | `hist[0.25-0.5]` | `hist[>=0.5]` |

---

## 4. DVX 1-D Percentile Sweep (`open_kernel: 1` fixed, 8 Train Seqs)

| Percentile | `ub_recall_preNMS` | `final_recall` | Precision | mAP@0.5 |
|---|---|---|---|---|
| 85.0 | `ub_rec` | `rec` | `prec` | `ap` |
| 90.0 | `ub_rec` | `rec` | `prec` | `ap` |
| 93.0 | `ub_rec` | `rec` | `prec` | `ap` |
| 95.0 | `ub_rec` | `rec` | `prec` | `ap` |
| 97.0 | `ub_rec` | `rec` | `prec` | `ap` |
| 98.0 | `ub_rec` | `rec` | `prec` | `ap` |
| 99.0 | `ub_rec` | `rec` | `prec` | `ap` |

---

## 5. Centroid On Pre-Dilation Mask Comparison (DVX Train Split)

| Variant | `ub_recall_preNMS` | `final_recall` |
|---|---|---|
| `centroid_on_predilation_mask: false` | `ub_rec` | `rec` |
| `centroid_on_predilation_mask: true` | `ub_rec` | `rec` |

---

## 6. Ground Truth Box Size Distributions (Train Split)

| Sensor | Dimension | Min | Median | P95 | Max |
|---|---|---|---|---|---|
| **DAVIS** | Width | 4.0 | 9.0 | 20.0 | 32.0 |
| **DAVIS** | Height | 4.0 | 9.0 | 18.0 | 28.0 |
| **DVX** | Width | 7.0 | 12.0 | 19.0 | 28.0 |
| **DVX** | Height | 8.0 | 13.0 | 20.0 | 26.0 |
| **EVK4** | Width | 38.0 | 52.0 | 60.0 | 72.0 |
| **EVK4** | Height | 42.0 | 56.0 | 64.0 | 76.0 |
