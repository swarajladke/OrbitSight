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

Mark each `PASS` / `FAIL` / `UNMEASURED`. No cell may be left blank or filled
with a plausible guess.

### Submission mechanics
- [ ] Registered on challenge platform
- [ ] Participation form complete
- [ ] Image exported via `docker save`, `.tar` or `.tar.gz`
- [ ] Entrypoint non-interactive, self-terminating
- [ ] Runs successfully with `--network none`
- [ ] Reads from `/OrbitSight_dataset` (read-only mount)
- [ ] Writes to `/work/teamName/DDMMYYYY`
- [ ] Prediction files named `<sequencename>.txt`
- [ ] Rows contain all 9 required fields in the required order
- [ ] `sequence_id` emitted
- [ ] `class_id` emitted, value justified from the official dataloader
- [ ] `Evaluation_Metrics.xlsx` written to the same output folder
- [ ] Model weights, structure file, and inference script baked into the image
- [ ] No runtime network calls of any kind
- [ ] 5-page PDF proposal, all 6 required sections present
- [ ] All deliverables in English

### Capability coverage
- [ ] Detection in noisy, low-light feeds
- [ ] Classification against background noise/artifacts
- [ ] Detection across varying magnitude levels (dim → bright)
- [ ] **Tracking** (not just per-window detection)
- [ ] Multi-resolution sensor compatibility
- [ ] **Visualization tool**
- [ ] End-to-end real-time pipeline, raw stream → detection → visualization

### Performance evidence
- [ ] mAP measured and reported
- [ ] Precision, recall, F1 measured and reported
- [ ] Latency measured on CPU-only hardware comparable to i9-12900H
- [ ] p99 and maximum per-window latency under 40 ms (not just the mean)
- [ ] Throughput and total runtime measured
- [ ] Ablation analysis of alternative model architectures completed
- [ ] Generalization evidence that does not rely on tuning against the test split

### Documentation
- [ ] Architecture diagram
- [ ] Training curves
- [ ] Detection example visualizations
- [ ] **Failure case** visualizations
- [ ] README covering training environment and exact reproduction commands
- [ ] Limitations explicitly stated

---

## 16. OUR ASSESSMENT — NOT OFFICIAL

Everything below this line is internal analysis dated 2026-08-20, derived from
reading `README.md`, `AGENTS.md`, `Dockerfile`, and the `src/` file listing.
It is not part of the official challenge text and may be wrong. Verify before
acting.

### 16.1 Suspected non-compliance (requires verification)
| Item | Official requirement | Apparent current state | Severity |
|---|---|---|---|
| Input path | `/OrbitSight_dataset` | `/dataset` per Dockerfile | Blocking |
| Output path | `/work/teamName/DDMMYYYY` | `/predictions` per Dockerfile | Blocking |
| Filenames | `<sequencename>.txt` | `*_pred.txt` per README | Blocking |
| Row fields | 9 fields | 7 documented in README | Blocking |
| `sequence_id` | Required | Not documented | Blocking |
| `class_id` | Required | Not documented | Blocking |
| `Evaluation_Metrics.xlsx` | Required in output folder | `src/report_xlsx.py` exists, output location unverified | High |
| Visualization tool | Required capability + scoring criterion | No visualization module in `src/` | High |
| Tracking | Required throughout statement | ±1-window association as a feature only; no track output | High |
| Classification | Required capability | Binary target/noise scoring only | Medium |
| Latency on target HW | <40 ms on i9-12900H, CPU-only | 15.30 ms/window on unstated hardware; tail latency unmeasured | Medium |
| Offline operation | No internet | Unverified | Medium |
| README accuracy | Part of scored documentation | Quick Start references `OrbitSight_Research/` subdir; actual repo root is flat. `LICENSE` linked but absent from root listing | Low |

### 16.2 Strategic notes
- mAP is **one of five** criteria with unstated weights. Optimizing it alone
  leaves entire criteria unaddressed.
- Technical Innovation explicitly names SNNs / graph NNs / hybrid event-frame
  models. A connected-components detector plus a gradient-boosting re-scorer is
  unlikely to score well on architectural novelty **unless** paired with a
  rigorous ablation showing the alternatives lose on the CPU-only latency budget.
  An honest, well-evidenced negative result is a valid answer to this criterion.
- Finalists are re-scored on a **new unseen dataset**. Operating points selected
  by grid-sweeping the 17 training sequences carry overfitting risk. Prefer
  leave-one-sequence-out grouped validation for any parameter selection.
- Criterion 10.5 explicitly rewards articulating **limitations**. Hiding weak
  results is scored against, not for.
- Visualization is currently the largest single scoring gap and is comparatively
  cheap to build.

### 16.3 Open questions to resolve
- [ ] Field delimiter for prediction rows — comma or tab?
- [ ] Header row in prediction files — required, optional, or forbidden?
- [ ] Valid `class_id` values and whether ground truth carries class labels
- [ ] Required schema/sheet layout of `Evaluation_Metrics.xlsx`
- [ ] Whether `sequence_id` is the directory name, filename stem, or an index
- [ ] Whether test-sequence predictions must be included alongside training ones
- [ ] Exact team name string to use in the `/work/teamName/` path
- [ ] Whether evaluators run sequences in parallel (affects latency tail)
