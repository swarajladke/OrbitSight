# TII OrbitSight Challenge — Authoritative Requirements

**Status:** Reference document. Transcribed from the official challenge page.
**Transcribed:** 2026-08-20
**Source:** TII OrbitSight Challenge, challengeon.atrc.ae — Challenge Statement,
Technical & Solution Requirements, Implementation Requirements, Criteria for
Application, and Scoring Criteria sections.

> **Rule for agents:** Sections 1–10 are OFFICIAL requirements. Do not alter,
> reinterpret, or "improve" them. Section 11 onward is OUR OWN assessment and is
> not authoritative. If official text and our assessment conflict, the official
> text wins. Anything the official source does not state is marked `UNSPECIFIED`
> and must not be guessed.

---

## 1. Challenge Identity

| Field | Value |
|---|---|
| Challenge name | TII OrbitSight Challenge |
| Organizer | Technology Innovation Institute (TII) |
| Authoring unit | TII Propulsion and Space Research Center (PSRC) |
| Domain | Space Situational Awareness (SSA) / Space Domain Awareness |
| Sensor modality | Neuromorphic Vision Sensor (NVS) / event-based cameras |
| Open to | Startups, researchers, students, enterprises, worldwide (subject to eligibility rules) |
| Data capture source | NVS mounted on a 0.8 m diameter telescope at the Abu Dhabi Quantum Optical Ground Station (ADQOGS) |

---

## 2. Timeline

| Milestone | Date |
|---|---|
| Application phase opens | June 2026 |
| **Submission deadline** | **9 September 2026, 23:59 (GMT+4)** |
| Evaluation of submissions | September 2026 |
| Top 5 notified privately | September 2026 |
| Reproducibility package due | Within 7 calendar days of finalist notification |
| Finalist video pitches | Date `UNSPECIFIED` — "to be released soon" |
| Winner announcement | October 2026 |

All deliverables must be submitted in **English**.

---

## 3. Challenge Statement

Develop high-performance AI/ML algorithms/pipelines that:

1. Process raw neuromorphic vision sensor (NVS) data from input to resident
   space objects (RSO) **detection, tracking and visualization**.
2. Achieve **real-time** RSO detection and tracking in noisy, low-light NVS data feeds.
3. Deliver real-time AI inference for efficient object detection and tracking.
4. Support **broad compatibility** with TII-provided NVS datasets across variable
   neuromorphic sensor resolutions.
5. Provide **visualization tools** for interpreting and presenting detection results.

---

## 4. Background of the Problem

- SSA involves detecting, tracking, and forecasting the movement of objects in
  Earth's orbit; it is critical for protecting space assets and preventing collisions.
- Existing sensor systems can only track and catalog debris larger than ~10 cm.
- A single large-debris collision could produce thousands of trackable fragments
  and tens of thousands of untrackable smaller pieces, especially under low-light,
  noisy conditions.
- Traditional optical sensing is limited in low-light environments, with
  fast-moving objects, and where real-time response is required.
- NVS sensors react to change, detecting motion at microsecond resolution with
  significantly lower power consumption and latency than frame-based cameras.
- Raw NVS data is asynchronous, event-driven, and noisy. There are currently no
  widely adopted, high-performance algorithms that turn raw event data into
  reliable real-time detection and tracking of fast-moving objects.
- The winning solution will be further developed by TII to enhance national
  capabilities in space debris detection and monitoring.

---

## 5. Technical & Solution Requirements

Proposed solutions should, where applicable, enable:

- [ ] Detection of objects in noisy, low-light NVS data feeds
- [ ] Classification of space objects in NVS stream against background noise/artifacts
- [ ] Detection of RSOs across varying magnitude levels (dim to bright objects)
- [ ] Broad compatibility with TII-provided NVS datasets across different camera resolutions
- [ ] Visualization tool to display detection results
- [ ] A real-time pipeline from sensor input to RSO detection and visualization from raw NVS stream
- [ ] Real-time RSO detection and tracking in low-light, noisy conditions with NVS
- [ ] Real-time AI inference for efficient object detection and tracking
- [ ] Integrated visualization tools for interpreting and presenting detection results

**Eligibility constraint:** Solutions may range in TRL but must be practical and
clearly articulate how they can be used. **Purely conceptual or theoretical
solutions are not eligible.**

---

## 6. Submission Deliverables

To participate, participants must:

1. Register on the challenge platform.
2. Complete the participation form.
3. Submit a **Docker image** (Section 7).
4. Submit a **5-page technical proposal (.pdf)** (Section 8).

Deadline for all of the above: **9 September 2026, 23:59 (GMT+4)**.

---

## 7. Docker Image Specification

### 7.1 Packaging
- Export with `docker save`, e.g. `docker save yourimage:tag -o image.tar`
- May be gzip-compressed as `image.tar.gz`

### 7.2 Runtime behaviour
- Must run **non-interactively** and **finish on its own**. No manual input.
- Provide an automatic entrypoint, e.g. `CMD ["sh","run.sh"]`
- **Avoid interactive shells.**
- Containers run **OFFLINE — no internet access.**

### 7.3 Input
- **Read inputs from:** `/OrbitSight_dataset` (mounted **read-only**)
- Contains:
  - NVS event recordings (training and testing sequences) as `*.npy`
  - Ground-truth metadata as `*.txt`
  - A `/OrbitSight_dataloader` folder with instructions on how to check, load,
    and start with the data

### 7.4 Model files
Must be included **inside** the Docker image:
- AI model weights
- Model structure file
- Inference script

The inference script must accept the OrbitSight event recordings (training and
testing sequences in `*.npy`) as input and produce a `.txt` prediction results
file within the container.

### 7.5 Output
- **Write outputs to:** `/work/teamName/DDMMYYYY` (mounted **write-only**)
- The Portal collects this folder to obtain prediction results and scoring sheet.
- **All outputs must be saved inside this folder.**
- Prediction files named: `<sequencename>.txt`
- An `Evaluation_Metrics.xlsx` file must be saved in the **same folder**.

### 7.6 Prediction row format

**One detection per row**, with exactly these fields in this order:

    sequence_id, window_start_timestamp_us, window_end_timestamp_us,
    x_centre, y_centre, w, h, class_id, confidence

Field delimiter: `UNSPECIFIED` in the official text — verify against
`/OrbitSight_dataloader` before finalizing.
Header row required or forbidden: `UNSPECIFIED`.
Permitted `class_id` values: `UNSPECIFIED` — derive from the official
dataloader / ground-truth metadata, do not invent.

---

## 8. Technical Proposal Specification

Format: **PDF, 5 pages maximum, English.**

Required sections:

1. **Problem statement and proposed solution**
2. **Outcome metrics** — effectiveness measures used to evaluate the solution:
   mAP (mean Average Precision), precision, recall, F1 score, inference efficiency
3. **Value proposition and competitive positioning**
4. **Technical approach and solution architecture** — detailed methodology and
   expected outputs
5. **Team capacity** — participant background, capacity, and capability to
   develop the solution
6. **Prior work** — details on any proof of concept (POC), additional
   development, or existing applications of the solution

---

## 9. Evaluation Hardware & Environment

| Property | Specification |
|---|---|
| CPU | Intel Core i9-12900H (14 cores / 28 threads) |
| RAM | 32 GB |
| OS | Ubuntu 22.04 / 24.04 LTS |
| GPU | **None — CPU-only execution** |
| Network | **Offline, no internet access** |
| Rationale | Applied uniformly across all submissions for fairness and reproducibility |

**Real-time target:** latency **under 40 ms end-to-end**.

---

## 10. Scoring Criteria

All submissions are assessed on these five criteria.

### 10.1 Technical Innovation (AI Approach)
Innovation and methodological depth of the proposed AI model. Evaluates:
- Novelty of the model architecture — examples given: **SNNs, graph-based neural
  networks, hybrid event-frame models**
- Rationale behind design choices
- Suitability of the approach for processing **sparse event-based inputs**
- *"Strong submissions include an ablation analysis of alternative models to
  justify the optimal solution quantitatively and qualitatively."*

### 10.2 Detection Accuracy (on test data)
Detection performance on **unseen/new dataset**. Evaluated using:
- mAP (mean Average Precision)
- Precision
- Recall
- F1 score

### 10.3 Real-time Performance
Inference efficiency on the evaluation hardware. Evaluated using:
- Inference latency
- Throughput
- Total runtime
- *"Strong submissions achieve real-time performance (latency under 40 ms end-to-end)."*

### 10.4 Documentation / Visualization & Reporting
Quality of documentation and visual reporting. Evaluated based on:
- Proposal report
- README
- Supporting visualizations — examples given: **architecture diagrams, training
  curves, detection examples, failure cases**
- *"Strong submissions clearly explain methodology, justify design choices, and
  present results in a reproducible and accessible manner."*

### 10.5 Team Competency / Solution Articulation
- Relevance of team members' background to the problem
- Depth of technical understanding shown in the proposal and the pitch
- Ability to clearly articulate design choices, **limitations**, and results

**Criterion weights:** `UNSPECIFIED`. The official text does not state relative
weighting. Do not assume mAP dominates.

---

## 11. Reproducibility Package — Top 5 Finalists Only

Due **within 7 calendar days** of private notification:

1. The **same** Docker image and prediction result files submitted in Phase 1,
   including results obtained from the **new unseen dataset provided**.
2. Full **training source code** (zipped repository), including model definition,
   training script, and configuration files used to produce the submitted weights.
3. A `README.md` documenting:
   - Training environment: Python version, key dependencies and versions,
     GPU/CPU hardware used
   - The exact command(s) to reproduce training from scratch
4. **Training dataset specification:** which datasets were used, and any
   preprocessing or augmentation applied.

**Implication:** the submitted weights must be reproducible from the submitted
training code. Do not ship weights whose provenance cannot be reconstructed.

---

## 12. Finalist Pitches

- Short **live-video pitches**, each a **30-minute session**, all on the same day.
- Scored by evaluators and aggregated.
- In parallel, solutions submitted for validation are evaluated holistically by
  technical experts for strategic fit.

---

## 13. Implementation / IP Requirements

Participants agree to provide TII with:
- Solution architecture or presentation explaining the solution in detail
- Access to the source code as a **Docker Image**

TII requests technical documentation and source code as part of the competition.
The winning solution will be further developed by TII.

---

## 14. Dataset

Training and testing dataset is distributed via Google Drive by TII.
Folder ID: `1LuOxeJnJSOjZm1VoLtu_FYx88GFH_ngu`

Contents per the Docker spec: `*.npy` event recordings, `*.txt` ground-truth
metadata, and an `OrbitSight_dataloader` folder.

---

## 15. Master Compliance Checklist

Marked `PASS` / `FAIL` / `UNMEASURED` on 2026-09-05 against commit `583f4bf` and
the submitted image in release `v1.0-submission`. Where a checklist line contains
two clauses and the verdicts differ, both are given. No cell is left blank.

### Submission mechanics
| Item | Verdict | Evidence |
|---|---|---|
| Registered on challenge platform | PASS | Team `OrbitAI`, team ID 1149 |
| Participation form complete | FAIL | 58 percent complete on 2026-09-05. Docker image link, proposal upload, referral question and the four declaration checkboxes are outstanding. |
| Image exported via `docker save`, `.tar` or `.tar.gz` | PASS | `image.tar.gz`, 188,539,903 bytes gzipped, 582,660,096 bytes raw, SHA-256 `7bb765b2...c722` |
| Entrypoint non-interactive, self-terminating | PASS | `CMD ["sh","run.sh"]`. Unattended 21-sequence run completed in 48.08 minutes wall clock with no input. |
| Runs successfully with `--network none` | PASS | Full 21-sequence container run executed under `--network none` |
| Reads from `/OrbitSight_dataset` (read-only mount) | PASS | `DATASET_DIR="${ORBITSIGHT_DATASET_DIR:-/OrbitSight_dataset}"` in the `run.sh` baked into the image |
| Writes to `/work/teamName/DDMMYYYY` | PASS, with a declared deviation | Writes `/work/OrbitAI/DDMMYYYY`. `run.sh` also writes three case and spelling mirrors of the same output. The official text names one folder. See 16.2 item 6. |
| Prediction files named `<sequencename>.txt` | PASS | 63 `.txt` files for 21 sequences: the required `<seq>.txt` plus two additional variants that do not replace it |
| Rows contain all 9 required fields in the required order | PASS | Tab-separated header `sequence_id`, `window_start_timestamp_us`, `window_end_timestamp_us`, `center_x`, `center_y`, `width`, `height`, `class_id`, `confidence` |
| `sequence_id` emitted | PASS | Column 1 of every row, filename stem |
| `class_id` emitted, value justified from the official dataloader | PASS emitted / UNMEASURED justified | Emitted as `0`. The official text leaves permitted values `UNSPECIFIED` and the dataloader defines no class taxonomy, so `0` is a single-class convention rather than a derived value. |
| `Evaluation_Metrics.xlsx` written to the same output folder | PASS | 7,341 bytes, written by `src/report_xlsx.py` alongside the prediction files |
| Model weights, structure file, and inference script baked into the image | PASS | `models/scorer_pregeom.joblib` 359,304 B, `models/scorer_objectness_pre_geometry.joblib` 525,648 B, `models/box_regressor_arm2.joblib` 611,945 B, totalling 1,496,897 B, plus `models/model_structure.json` and `src/infer.py` |
| No runtime network calls of any kind | PASS | Eight pinned dependencies, no network client in the runtime path, verified by the offline container run |
| 5-page PDF proposal, all 6 required sections present | PASS | `PROPOSAL.pdf`, 417,753 bytes, 5 pages, sections 1 through 6 present |
| All deliverables in English | PASS | Proposal, README and portal fields are English throughout |

### Capability coverage
| Item | Verdict | Evidence |
|---|---|---|
| Detection in noisy, low-light feeds | PASS | mAP@0.5 0.284406 across all 21 sequences, all of which are low-light telescope captures |
| Classification against background noise/artifacts | PASS, binary only | A learned candidate scorer (ROC-AUC 0.930) and a window-objectness gate (ROC-AUC 0.889, PR-AUC 0.921) separate target from noise. No multi-class object taxonomy is produced; see 16.2 item 8. |
| Detection across varying magnitude levels (dim to bright) | PASS | Sequences span magnitude 5.2 to 7.3; per-sensor operating points are separately tuned |
| Tracking (not just per-window detection) | PASS, limited | `src/tracker.py` with `tracking_mode: "ids_only"` assigns persistent identities across adjacent 40 ms windows. No orbital state estimation or multi-frame trajectory fitting. |
| Multi-resolution sensor compatibility | PASS | EVK4 1280x720, DVX 640x480, DAVIS 346x260, resolved by exact resolution match with nearest-diagonal fallback |
| Visualization tool | PASS | `src/visualize.py`, 6,304 bytes, ships inside the image and produced the detection example in the proposal |
| End-to-end real-time pipeline, raw stream to detection to visualization | PASS detection / FAIL visualization in-path | Detection and tracking run inside the 40 ms window budget. Visualization is an offline post-hoc renderer and is not part of the real-time path. |

### Performance evidence
| Item | Verdict | Evidence |
|---|---|---|
| mAP measured and reported | PASS | 0.284406 all-21, 0.258616 train-17, 0.394014 derived test-4 |
| Precision, recall, F1 measured and reported | PASS | 0.623120 / 0.472327 / 0.537345 all-21, from 10,565 TP, 6,390 FP, 11,803 FN |
| Latency measured on CPU-only hardware comparable to i9-12900H | PASS CPU-only / UNMEASURED comparability | Measured CPU-only on the development machine. No benchmark has been run on an i9-12900H, so hardware equivalence is asserted nowhere. |
| p99 and maximum per-window latency under 40 ms (not just the mean) | FAIL | Compute p99 is under 40 ms on 18 of 21 sequences over 5 repetitions with 20 warmup windows. Maximum per-window latency has never been reported. Three sequences exceed the budget at p99. |
| Throughput and total runtime measured | PASS | 143,750 windows processed, 48.08 minutes wall clock, RSS 114.3 to 130.7 MB |
| Ablation analysis of alternative model architectures completed | FAIL | The shipped four-arm ablation compares box-sizing strategies inside one architecture. No SNN, CNN or graph neural network baseline has been measured against the CPU-only budget. This is the largest single scoring gap; see 16.2 item 1. |
| Generalization evidence that does not rely on tuning against the test split | PASS | All operating points were selected on the 17 training sequences only. The 4 test sequences were never used for selection. |

### Documentation
| Item | Verdict | Evidence |
|---|---|---|
| Architecture diagram | PASS | Figure 2 in the proposal, plus a flowchart in `README.md` |
| Training curves | FAIL | Not produced and not producible in the usual sense: all three shipped models are gradient-boosted trees, so no epoch-wise loss curve exists. ROC-AUC and PR-AUC figures are reported instead. Recorded as a gap rather than substituted silently. |
| Detection example visualizations | PASS | Figure 1 in the proposal, rendered by `src/visualize.py` |
| Failure case visualizations | PASS documented / FAIL rendered | Four failure modes are characterised with measured evidence in proposal section 4.5. None is rendered as an image. |
| README covering training environment and exact reproduction commands | PASS | `README.md` rewritten at commit `583f4bf`, including quick start, container build and the commands to regenerate per-sequence breakdowns |
| Limitations explicitly stated | PASS | Proposal section 4.5, the README disclosure and latency-semantics sections, and section 16.2 below |

---

## 16. INTERNAL ASSESSMENT - NOT OFFICIAL

Everything below this line is internal analysis. It is not part of the official
challenge text and does not override sections 1 through 14.

**Revised:** 2026-09-05, against commit `583f4bf`.

The 2026-08-20 revision of this section listed thirteen suspected
non-compliances, six of them marked blocking, inferred from reading the README
and the Dockerfile rather than from running anything. Eleven have since been
closed by measurement. This revision records what each gap was and what closed
it, so the closure is auditable instead of assumed, and states plainly what is
still open.

### 16.1 Gap ledger - recorded 2026-08-20, status 2026-09-05
| Gap as originally recorded | Status | What closed it |
|---|---|---|
| Input path read `/dataset` | CLOSED | `run.sh` inside the built image reads `/OrbitSight_dataset`; confirmed by reading the file out of the container |
| Output path wrote `/predictions` | CLOSED | Writes `/work/OrbitAI/DDMMYYYY`; verified by a full container run |
| Prediction filenames were `*_pred.txt` only | CLOSED | The required `<sequencename>.txt` is emitted for all 21 sequences |
| Rows carried 7 fields | CLOSED | All 9 official fields verified present in the official order |
| `sequence_id` undocumented | CLOSED | Emitted and documented |
| `class_id` undocumented | CLOSED as emitted | Emitted as `0`; the value justification remains UNMEASURED because the official text specifies no value set |
| `Evaluation_Metrics.xlsx` location unverified | CLOSED | 7,341 bytes, written into the same output folder |
| No visualization module existed | CLOSED | `src/visualize.py` ships in the image and produced the proposal detection figure |
| Tracking emitted no persistent identities | PARTIALLY CLOSED | `src/tracker.py` with `tracking_mode: "ids_only"` emits persistent cross-window identities. No orbital state estimation. |
| Classification was binary target/noise only | OPEN BY DESIGN | Unchanged, and appropriate: the official text specifies no class taxonomy and ground truth carries none |
| Latency was a single mean on unstated hardware | PARTIALLY CLOSED | Now p99 over 5 repetitions across all 21 sequences. Maximum per-window latency and evaluation-hardware equivalence remain open. |
| Offline operation unverified | CLOSED | Full run under `--network none` |
| README inaccurate, `LICENSE` apparently absent | CLOSED | `README.md` rewritten at `583f4bf`; `LICENSE` is present at the repository root |

### 16.2 Gaps still open, held open deliberately
1. **No architecture-level ablation.** Criterion 10.1 rewards ablation of alternative models and names SNNs, graph neural networks and hybrid event-frame models. The shipped four-arm ablation varies box sizing within one architecture, which is not the same claim. The honest position is that no neural baseline has been timed against the CPU-only 40 ms budget, so the "no neural network" decision is argued from first principles and from a 1,496,897-byte model footprint rather than from a measured head-to-head.
2. **Three sequences exceed the latency budget at p99:** 84.67 ms, 77.76 ms and 58.73 ms, on the densest and the two highest-resolution streams. Per-sensor decimation is the untested next lever.
3. **Maximum per-window latency is unreported.** Only p99 is measured.
4. **Evaluation-hardware equivalence is not established.** No measurement exists on an i9-12900H.
5. **Training curves do not exist** for gradient-boosted trees and no substitute has been claimed as one.
6. **Four output directories are written where the specification names one.** Retained on purpose: a team-name mismatch scores zero with no visible error, whereas duplicate folders are visible to a human reader. The risk accepted is that recursive ingestion could double-count detections and depress precision silently.
7. **The test-4 metrics are derived**, computed as the residual of all-21 against train-17, not from an independent evaluator run restricted to the test split.
8. **Only binary target/noise discrimination is performed.** No object-class output.
9. **Failure modes are documented in prose, not rendered as imagery.**

### 16.3 Open questions resolved since 2026-08-20
- **Field delimiter:** tab. The official `evaluate.py` parses with `csv.DictReader`.
- **Header row:** required, and the column names must be `center_x`, `center_y`, `width`, `height`. This differs from the prose in the challenge brief; the evaluator, not the prose, is authoritative. Renaming these columns causes a `KeyError` and a zero score.
- **`class_id`:** emitted as `0` under a single-class convention.
- **Team name in the `/work` path:** `OrbitAI`, the registered team name.
- **`sequence_id` form:** the filename stem is used.

Still unresolved: the required schema of `Evaluation_Metrics.xlsx`, and whether
evaluators run sequences in parallel, which would change the latency tail.