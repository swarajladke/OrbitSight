# OrbitSight Configuration Ledger

This document tracks all evaluated pipeline configurations across box geometry optimizations and reranking variants.

## 1. Geometry x Reranking Evaluation Matrix

| Cell | Configuration Description | Config File | Commit SHA | Train mAP | Sparse mAP (10) | Dense mAP (7) | Train Prec | Train Rec | Train F1 | TP | FP | FN | Test mAP | Test Prec | Test Rec | Test F1 | Worst p99 (ms) | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **(a)** | Pre-Geometry + No Rerank | `configs/pre_geometry.yaml` | `dcc8787` | **0.155493** | 0.083280 | 0.258671 | 0.281011 | 0.335993 | 0.306052 | 5138 | 13146 | 10154 | **0.303022** | 0.4765 | 0.4072 | **0.4392** | 113.3 | **SHIPPING CANDIDATE** |
| **(b)** | Post-Geometry + No Rerank | `config.yaml` | `dcc8787` | 0.146340 | 0.056048 | 0.275328 | 0.233581 | **0.372809** | 0.287211 | 5701 | 18706 | 9591 | 0.290706 | 0.3648 | **0.4289** | 0.3943 | 113.3 | Rejected |
| **(c)** | Post-Geometry + Variant B (Window Objectness) | `config.yaml` | `e97ab9c` | 0.149436 | 0.060650 | **0.276273** | 0.439899 | 0.364243 | **0.398512** | 5570 | 7092 | 9722 | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | 113.3 | Evaluated |
| **(d)** | Pre-Geometry + Variant B (Window Objectness) | `configs/pre_geometry.yaml` | `036c672` | **0.163628** | **0.098205** | 0.257091 | 0.422441 | 0.330892 | **0.371104** | 5060 | **6918** | 10232 | *MEASURING TEST* | *MEASURING TEST* | *MEASURING TEST* | *MEASURING TEST* | 113.3 | **TRAIN WINNER** |
| **Var A**| Post-Geometry + Multi-Window Track Rerank ($G=3$) | `config.yaml` | `e97ab9c` | **0.161878** | **0.082539** | 0.275221 | **0.530001** | 0.290544 | 0.375333 | 4443 | **3940** | 10849 | 0.289564 | **0.6000** | 0.3337 | 0.4288 | 113.3 | Retired as Scorer (Kept for Track IDs) |

---

## 2. Tracking Architecture Notes
- **Tracking Mode**: `tracking_mode: "ids_only"` is configured by default to emit stable multi-window association IDs without modifying base confidence scores, ensuring byte-identical numerical outputs to the baseline while satisfying SSA challenge tracking requirements.
