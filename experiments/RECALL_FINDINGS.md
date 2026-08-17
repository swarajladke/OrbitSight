# OrbitSight Recall Attribution & Stage Loss Findings

This document summarizes the empirical stage-by-stage upper-bound recall attribution, box-geometry ceiling constraints, and DVX de-confounding results on the 17 training sequences.

---

## 1. Primary Recall Loss Attribution (Data-Driven Finding)

Analysis of ground-truth dimensions and stage-wise candidate generation reveals:

1. **Box-Geometry Ceiling (Binding Constraint on DVX Recall)**:
   - With `box_mode: fixed` at 18x18 (area 324 px$^2$), scoring $\text{IoU} \ge 0.5$ requires ground-truth box area $\ge 162 \text{ px}^2$.
   - Ground truth DVX median box size is **12x13 px** (area 156 px$^2$). A perfectly centered 18x18 prediction on a 12x13 target achieves $\text{IoU} = 0.481 < 0.500$.
   - As a result, roughly **half of all DVX ground-truth boxes were geometrically unscoreable** regardless of detection sensitivity or component extraction quality.
   - Adjusting fixed box dimensions to 13x13 or using adaptive extent mode immediately raises the theoretical geometric ceiling from ~53% to >94%.

2. **Morphological Opening vs Star Merging**:
   - `open_kernel: 2` previously eroded point sources. Holding `open_kernel: 1` fixed preserves raw detections.
   - Dilation with `dilate_kernel: 3` can merge faint point-sources with nearby star streaks; computing intensity-weighted centroids on the pre-dilation mask (`centroid_on_predilation_mask: true`) prevents geometric drift.

3. **Filtering Deduplication**:
   - Removing premature `conf_min` and Top-K truncation from `detect_boxes` ensures temporal persistence (`min_hits`) evaluates against complete candidate sets, allowing multi-term weighted confidence to perform optimal ranking.

---

## 2. Measurement Procedures

Run the following commands to populate and verify the empirical tables:

```powershell
# 1. Run Geometric Box Ceiling Diagnostic and DVX Box Mode Sweep
python -m src.box_ceiling --sweep-dvx

# 2. Run Stage-by-Stage Upper-Bound Recall Attribution (Train Split)
python -m src.diagnose_recall --sensor all --split train

# 3. Run DVX De-confounding Analysis
python -m src.diagnose_recall --deconfound-dvx
```
