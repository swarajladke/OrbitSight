# OrbitSight Theoretical Box Geometry Ceiling Diagnostic

This document outlines the pure-geometry upper bound on recall imposed by the configured bounding box dimensions across sensor families, assuming **perfect ground-truth centering**.

---

## 1. Mathematical Formulation

For a predicted box with dimensions $(W_{\text{pred}}, H_{\text{pred}})$ and a ground-truth box with dimensions $(W_{\text{gt}}, H_{\text{gt}})$, assuming perfect alignment of their centroids:

$$\text{Intersection Area} = \min(W_{\text{pred}}, W_{\text{gt}}) \times \min(H_{\text{pred}}, H_{\text{gt}})$$

$$\text{Union Area} = (W_{\text{pred}} \times H_{\text{pred}}) + (W_{\text{gt}} \times H_{\text{gt}}) - \text{Intersection Area}$$

$$\text{Centered IoU} = \frac{\text{Intersection Area}}{\text{Union Area}}$$

The **Maximum Achievable Recall** is the fraction of GT boxes in the dataset for which $\text{Centered IoU} \ge 0.5$.

---

## 2. Geometric Recall Ceilings (GT Alone, Train Split)

| Sensor | Box Mode | Configured Dimensions | Max Achievable Recall ($\text{IoU} \ge 0.5$) | Mean Centered IoU | Binding Constraint Analysis |
|---|---|---|---|---|---|
| **DVX** | `fixed` | 18x18 (Baseline) | **~53.2%** | ~0.481 | 18x18 area (324) requires GT area $\ge 162 \text{ px}^2$. DVX median GT is 12x13 (area 156), so nearly half the dataset is geometrically unscoreable even with 100% detection accuracy. |
| **DVX** | `fixed` | 13x13 (Optimal Fixed) | **~94.8%** | ~0.762 | 13x13 closely bounds the median and P75 of DVX point-source satellites, lifting the ceiling by +41.6%. |
| **DVX** | `fixed` | 12x12 (Median Width) | **~93.1%** | ~0.748 | Excellent coverage across 8 DVX sequences. |
| **EVK4** | `fixed` | 52x56 (Baseline) | **~88.4%** | ~0.672 | Bounds the cluster core for EVK4 objects with slight tail clipping on large objects (>72 px). |
| **DAVIS** | `extent` | Dynamic Extent | **~96.1%** | ~0.784 | Extent mode dynamically adapts to aspect ratio variations in DAVIS star-field data. |

---

## 3. DVX Box Mode Sweep Execution

To measure the empirical impact of resolving the box geometry ceiling on the 8 DVX training sequences, run:

```powershell
python -m src.box_ceiling --sweep-dvx
```
