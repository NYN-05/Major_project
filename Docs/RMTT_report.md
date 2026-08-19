# Deepfake Detection in Low-Resolution KYC Videos Using rPPG and Hybrid Quantum Machine Learning

**RMTT Project Report**

---

## 1. Deliverables

| # | Deliverable | Purpose and Expected Contribution |
|---|---|---|
| 1 | Project documentation | Records objectives, scope, architecture, design decisions, and verification results so the project is auditable and reproducible. |
| 2 | Literature survey | Establishes the state of the art in deepfake detection, rPPG-based liveness, and quantum machine learning, and justifies the proposed combination. |
| 3 | System architecture | Defines the two-stage framework (rPPG physiological analysis → hybrid quantum-classical classification) and the interfaces between stages. |
| 4 | Dataset / preprocessing pipeline | A real-data-only feature-extraction pipeline (DFDC + FF++ → 10 rPPG features per clip) with leakage-fixed train/val/test grouping. |
| 5 | Face detection and alignment module | Locates and quality-gates faces (YOLOv8 + MediaPipe landmarks) so only usable frames feed the physiological analysis. |
| 6 | rPPG extraction module | Recovers pulse-related signals (POS/CHROM) from facial ROIs (forehead, left/right cheek) under low-resolution conditions. |
| 7 | Physiological feature-generation module | Converts pulse signals into 10 compact temporal features (HR, SNR, spectral entropy, MAD, SQI, ROI correlations, etc.). |
| 8 | Hybrid quantum-classical classification module | QAOA-based feature selection feeding a hybrid VQC (PennyLane + PyTorch) that outputs P(real). |
| 9 | Confidence-aware decision module | Maps P(real) into REAL / FAKE / UNCERTAIN bins with confidence, avoiding over-confident decisions in low-signal cases. |
| 10 | Testing and evaluation | Standard metrics (accuracy, precision, recall, specificity, F1, AUC-ROC, PR-AUC, ECE), grouped cross-validation, and classical baselines. |
| 11 | Gantt chart | Tracks the 8-phase project plan and schedule adherence. |
| 12 | Final presentation / report | Communicates methodology, results, and future work to examiners and reviewers. |

## 2. Methodology

### 2.1 Problem definition
KYC video onboarding is vulnerable to deepfake attacks, and the videos are typically low-resolution, compressed, and unstable. Visual-artifact detectors degrade under such conditions. This project instead uses **physiological evidence** — the pulse signal recovered remotely from the face — which deepfakes do not reproduce faithfully, and classifies it with a **hybrid quantum-classical model**.

### 2.2 Literature and requirement analysis
A survey of deepfake detection (CNN/LSTM, transformer), rPPG-based liveness, and quantum-hybrid classifiers identified (a) the rPPG gap for low-resolution KYC and (b) QAOA + VQC as a candidate for compact, robust feature conditioning.

### 2.3 KYC video acquisition
Videos are collected from public deepfake benchmarks — DFDC (real/fake face-swap clips) and FaceForensics++ (FF-real, YouTube-real, FF-synthesis) — and validated at upload (format, magic bytes, size) in the deployed system.

### 2.4 Frame extraction and quality assessment
Videos are sampled at their native 30 fps and quality-gated (face size, blur, pose, brightness). This is necessary because low-quality frames corrupt pulse extraction; the gate raises the signal-to-noise ratio of the downstream rPPG stage.

### 2.5 Face detection and alignment
YOLOv8 face detection localizes faces; MediaPipe Face Landmarker provides stable landmarks for region alignment. Stable landmarks are required because pulse extraction assumes the same skin regions across time.

### 2.6 Low-resolution preprocessing
Frames are kept at native resolution with quality-based filtering rather than upsampling, since rPPG methods operate on skin-color statistics that survive low resolution better than visual details. Explicit enhancement/decimation preprocessing remains future work (see §6).

### 2.7 Facial ROI selection
Three ROIs — forehead and left/right cheeks — are extracted per frame with skin-mask refinement (YCrCb). Cheeks and forehead maximize blood-perfusion signal and minimize non-skin contamination.

### 2.8 rPPG signal extraction
POS (default) or CHROM methods derive a pulse waveform from spatial skin-color changes over time. rPPG is useful for low-resolution deepfake detection because the pulse is a **temporal, physiological** cue: compression destroys pixel artifacts but not the color-fluctuation statistics, while synthesized faces lack a real pulse.

### 2.9 Temporal physiological feature generation
Ten features are computed from the pulse (heart rate, SNR, PRV variability, spectral entropy, MAD, signal-quality index, ROI correlations, HR half-difference, peak prominence) using a shared Welch PSD. These compact features are what the classifier consumes.

### 2.10 Feature conditioning
Features are standardized with a train-derived scaler, and a QAOA circuit selects the 3 most discriminative features (supervised discrimination weights; Hamiltonian ≡ classical cost, verified). Conditioning reduces noise and dimensionality before the quantum classifier.

### 2.11 Hybrid quantum-classical classification
A hybrid VQC (PennyLane QNode on `default.qubit`/backprop fused with a PyTorch ML head, focal loss) maps the selected features to P(real). The hybrid is investigated because quantum circuits offer expressive feature embeddings with few parameters — attractive for small, noisy physiological feature sets.

### 2.12 Confidence-aware decision generation
P(real) is binned: ≥ 0.7 REAL, ≤ 0.3 FAKE, else UNCERTAIN, with confidence `2·|P(real) − 0.5|`. The bin prevents false certainty when physiological evidence is weak — essential for KYC where a wrong decision is costly.

### 2.13 Model evaluation and comparison
The VQC is evaluated on a held-out test split (accuracy, precision, recall, specificity, F1, AUC-ROC, PR-AUC, ECE, confusion matrix, decision bins) with leakage-fixed grouping (FF++ source-subject; YouTube/DFDC per-clip) and StratifiedGroupKFold CV, and compared against RandomForest, MLP, Logistic Regression, LinearSVC, GaussianNB, and XGBoost.

## 3. Results

### 3.1 Technical / system-level results
A complete three-stage pipeline (frame sampling → rPPG → quantum verdict) is implemented and runs end-to-end (`WORKING/run_pipeline.py`), wrapped by a web frontend with SSE progress, upload validation (200 MB cap, magic bytes), concurrency limiting, and a dashboard (verdict gauge, signal canvas, frame thumbnails). A sample run processed 305 sampled frames → 191 accepted → 186 usable rPPG frames at 30 fps and produced a verdict of UNCERTAIN (P(real) = 0.589). **Quantitative latency/throughput benchmarks are pending formal measurement.**

### 3.2 rPPG-related results
The rPPG stage successfully recovers pulse waveforms and 10 physiological features per clip from low-resolution accepted frames; POS and CHROM methods are both implemented, and the pipeline handles `features=None` (insufficient usable frames → INCONCLUSIVE). **Physiological validity (e.g., HR plausibility across subjects) is recorded but not yet systematically validated against ground-truth pulse.**

### 3.3 Feature-generation results
A real-only feature table of **3444 clips** (1883 real / 1561 fake after an implausible-value filter) was built from DFDC + FF++ at native 30 fps. Per-feature discrimination is weak: |AUC − 0.5| ≤ ~0.06 for all 10 features. QAOA selected `peak_prominence`, `left_right_cheek_correlation`, `spectral_entropy` (cost −1.363).

### 3.4 Classification results
Reported from the frozen 2026-08-15 baseline (leakage-fixed splits), computed on the held-out test set:

| Metric | Hybrid VQC | Best classical (LR) |
|---|---|---|
| Accuracy | 0.547 | 0.557 |
| AUC-ROC | 0.503 | 0.569 |
| PR-AUC | 0.549 | 0.603 |
| Specificity | 0.000 | 0.513 |
| ECE | 0.065 | 0.047 |
| Decision bins | 100% UNCERTAIN | — |

Grouped 5-fold CV: VQC AUC 0.517 ± 0.023; classical AUCs 0.481–0.529. The VQC is a **majority-class predictor** (all test predictions REAL; confusion matrix [[0, 312], [0, 377]]).

### 3.5 Overall project results
The framework, architecture, data pipeline, and evaluation harness are fully implemented and reproducible; leakage control is in place. **However, no genuine real/fake discrimination has been achieved yet** — the frozen baseline confirms the rPPG features themselves carry almost no class signal, which is the central open problem.

### 3.6 Limitations
- The classifier currently predicts the majority class; decision bins are 100% UNCERTAIN, so the confidence thresholds cannot yet be exercised meaningfully.
- rPPG features carry near-zero class discrimination (|AUC − 0.5| ≤ ~0.06); an 8.2× increase in training data (421 → 3444 clips) did not lift the ceiling.
- The quantum model has not yet outperformed classical baselines (LR/GNB/XGB AUC 0.56–0.57); its role is therefore not yet justified empirically.
- No ground-truth pulse validation, low-resolution stress testing (severe compression, 10 fps decimation), or latency benchmarking has been completed.
- Phase 2 rPPG method/ROI probing is the planned gate to determine whether physiological signal recovery can be improved before any threshold tuning.

## 4. Proposed Outcome vs Actual Outcome

[Insert Table: Proposed Outcome vs Actual Outcome]

| Aspect | Proposed Outcome | Actual / Current Outcome | Status |
|---|---|---|---|
| Problem identification | Low-resolution KYC videos defeat visual deepfake detectors; use physiological evidence | Problem validated by literature; no published rPPG-based low-res KYC baseline outperformed here yet | Implemented |
| System architecture | Two-stage: rPPG analysis → hybrid quantum-classical classification | Two-stage pipeline implemented end-to-end (frame → rPPG → QAOA+VQC → verdict) | Implemented |
| Video processing | Frame sampling with quality gating at low resolution | 30 fps sampling, quality gate (face size, blur, pose), metadata tracking | Implemented |
| Face detection/alignment | Stable, aligned faces across frames | YOLOv8 detection + MediaPipe landmarks; per-frame ROI tracking | Implemented |
| Low-resolution preprocessing | Explicit low-res enhancement/decimation stage | Native-resolution processing with quality filtering; no explicit enhancement stage yet | Partially implemented |
| rPPG extraction | Robust POS/CHROM pulse recovery | POS + CHROM implemented; pulse recovered; physiological validity unvalidated | Implemented / validation pending |
| Physiological feature generation | 10 compact temporal features | 10 features generated for 3444 real clips (DFDC + FF++) | Implemented |
| Quantum-classical classification | QAOA selection + hybrid VQC beating classical models | QAOA + VQC implemented and evaluated; currently majority-class predictor (test AUC 0.503) | Implemented / not validated as superior |
| Confidence-aware decision | REAL / FAKE / UNCERTAIN with confidence | Binning implemented; 100% UNCERTAIN on test — thresholds untestable until features improve | Implemented / outcome pending |
| Testing/evaluation | Leakage-safe evaluation, baselines, metrics | Leakage-fixed grouping, grouped CV, 6 classical baselines, full metric suite | Implemented |
| Documentation | Audit trail, scenario, report | AGENTS.md, audit doc, scenario + matrices doc; this report | Implemented |
| Project planning | 8-phase schedule with milestones | Gantt chart (§5) maintained | In progress |

## 5. Gantt Chart

[Insert Figure: Gantt Chart] — 8 project phases (weeks), `●` = active work, `✓` = completed milestone.

| Activity | Ph1 | Ph2 | Ph3 | Ph4 | Ph5 | Ph6 | Ph7 | Ph8 |
|---|---|---|---|---|---|---|---|---|
| Problem identification | ● | ✓ | | | | | | |
| Literature survey | ● | ● | ✓ | | | | | |
| Requirement analysis | | ● | ✓ | | | | | |
| System architecture | | | ● | ✓ | | | | |
| Dataset preparation | | | ● | ● | ✓ | | | |
| Video preprocessing | | | | ● | ✓ | | | |
| Face detection/alignment | | | | ● | ● | ✓ | | |
| rPPG extraction | | | | | ● | ● | ✓ | |
| Feature engineering | | | | | | ● | ✓ | |
| Quantum-classical model development | | | | | | ● | ● | ✓ |
| Model integration | | | | | | | ● | ✓ |
| Testing and evaluation | | | | | | | ● | ● |
| Documentation | | | | | | | ● | ● |
| Final review/presentation | | | | | | | | ✓ |

## 6. Future Work

Priorities are implementation and empirical validation, not additional theory:

1. **Experimental dataset preparation** — finalize a balanced, leakage-free low-resolution test set (DFDC + FF++), including severe-compression and low-fps variants representative of real KYC feeds.
2. **Robust rPPG extraction** — Phase 2 probe of rPPG methods and ROI configurations (currently the explicit gate); validate recovered heart-rate plausibility against ground truth.
3. **Hybrid quantum-classical implementation** — retain QAOA selection and VQC once features carry signal; compare fairly against classical baselines on identical splits.
4. **Baseline comparison** — formalize LR/GNB/XGB and visual-only reference detectors as the comparison floor.
5. **Quantitative evaluation** — report accuracy, precision, recall, specificity, F1, AUC-ROC, PR-AUC, ECE, and latency/throughput on the final test set.
6. **Error analysis** — characterize failures by clip type (compression level, pose, illumination, manipulation method).
7. **Optimization for low-resolution videos** — low-res preprocessing (anti-aliased decimation), longer observation windows, frame-rate normalization.
8. **Final validation** — decision-threshold tuning (0.3/0.7 bins) only after genuine class signal is confirmed; end-to-end KYC scenario test through the web frontend.

## 7. Overall Conclusion

The project attempts to solve deepfake detection in low-resolution KYC videos, where compression and low quality defeat visual-artifact detectors. The proposed approach is different because it relies on **physiological evidence** — the rPPG pulse recovered from facial ROIs — and processes the resulting compact features with a **hybrid quantum-classical classifier** (QAOA feature selection + hybrid VQC), producing confidence-aware REAL / FAKE / UNCERTAIN decisions rather than hard binary labels.

What has been achieved: a complete, reproducible two-stage framework — video ingestion with quality gating, face detection and alignment, POS/CHROM rPPG extraction, 10 physiological features on a real-only 3444-clip table (DFDC + FF++), leakage-fixed evaluation, QAOA feature selection, hybrid VQC training, classical baselines, and a web frontend demonstrating the full flow. What remains to be experimentally validated: whether improved rPPG recovery yields genuine class discrimination, and whether the quantum-classical classifier then outperforms classical baselines — the frozen baseline shows both questions are still open, as the current features carry almost no real/fake signal.

The approach remains relevant to low-resolution KYC deepfake detection because pulse-based evidence is more robust to compression than pixel artifacts, and the three-way confidence-aware decision is well suited to high-stakes onboarding, provided the Phase 2 rPPG signal-recovery work succeeds.