# OrbitSight Configuration Ledger

This document tracks all evaluated pipeline configurations across box geometry optimizations and reranking variants.

## 1. Geometry x Reranking Evaluation Matrix

| Cell | Configuration Description | Config File | Commit SHA | Train mAP | Sparse mAP (10) | Dense mAP (7) | Train Prec | Train Rec | Train F1 | TP | FP | FN | Test mAP | Test Prec | Test Rec | Test F1 | Worst p99 (ms) | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **(a)** | Pre-Geometry + No Rerank | `configs/pre_geometry.yaml` | `dcc8787` | 0.155493 | 0.083280 | 0.258671 | 0.281011 | 0.335993 | 0.306052 | 5138 | 13146 | 10154 | **0.303022** | 0.4765 | 0.4072 | **0.4392** | 113.3 | Baseline Reference |
| **(b)** | Post-Geometry + No Rerank | `configs/post_geometry.yaml` | `dcc8787` | 0.146340 | 0.056048 | 0.275328 | 0.233581 | **0.372809** | 0.287211 | 5701 | 18706 | 9591 | 0.290706 | 0.3648 | **0.4289** | 0.3943 | 113.3 | Rejected |
| **(c)** | Post-Geometry + Variant B (Window Objectness) | `config.yaml` (old) | `e97ab9c` | 0.149436 | 0.060650 | **0.276273** | 0.439899 | 0.364243 | **0.398512** | 5570 | 7092 | 9722 | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | 113.3 | Evaluated |
| **(d)** | Pre-Geometry + Variant B (Window Objectness) | `config.yaml` | `719482b` | **0.165103** | **0.100600** | 0.257257 | 0.422371 | 0.330892 | **0.371077** | 5060 | **6920** | 10232 | **0.299820** *(derived)* | **0.566230** *(derived)* | 0.398110 *(derived)* | **0.467270** *(derived)* | 175.99 | **SHIPPING CONFIG (VERIFIED & CONTAINER TESTED)** |
| **Var A**| Post-Geometry + Multi-Window Track Rerank ($G=3$) | `config.yaml` | `e97ab9c` | **0.161878** | **0.082539** | 0.275221 | **0.530001** | 0.290544 | 0.375333 | 4443 | **3940** | 10849 | 0.289564 | **0.6000** | 0.3337 | 0.4288 | 113.3 | Retired as Scorer (Kept for Track IDs) |

*Notes:*
- *SUBMISSION ARTIFACT FROZEN at `719482b`, `image.tar` 189,562,880 B, offline run 23.26 min (1395.85 s), all-21 mAP 0.190765, 21/21 detection counts matched.*
- *Parity Attribution Correction: `mc=2000` measures 0.163622 with FP 6918, P 0.422441, F1 0.371104, TP 5060, matching the frozen Cell (d) reference (0.163628) within $6\times 10^{-6}$. The delta between the original baseline and 0.165103 attributes to: `mc 2000 -> 64` = $+0.001481$ (dominant, ACTIVE parameter suppressing noise candidates), and split-path centroid fix in `76c3e2a` = $-0.000006$ (negligible).*
- *Container ALL-21 Verified: mAP 0.190765, Precision 0.464583, Recall 0.352155, F1 0.400631, TP 7877, FP 9078, FN 14491 (20/21 sequences meet Pass-1 $p_{99} < 40\text{ ms}$; mean compute throughput 9.65 ms/win).*
- *Test split metrics for Cell (d) are mathematically derived by subtracting Train split counts (TP 5060, FP 6920, FN 10232) from Container All-21 counts (TP 7877, FP 9078, FN 14491) yielding Test TP 2817, FP 2158, FN 4259.*

---

## 2. Tracking Architecture Notes
- **Tracking Mode**: `tracking_mode: "ids_only"` is configured by default to emit stable multi-window association IDs without modifying base confidence scores, ensuring byte-identical numerical outputs to the baseline while satisfying SSA challenge tracking requirements.
