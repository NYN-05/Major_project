# Deepfake Detection Accuracy Enhancement
## Full Implementation Report and Engineering Guidelines

**Project:** Deepfake Detection in Low-Resolution KYC Videos Using rPPG and Hybrid Quantum Machine Learning  
**Implementation Objective:** Improve genuine real/fake discrimination while preserving rPPG and hybrid quantum ML as the central contribution.  
**Primary Operating Objective:** KYC-risk-aware detection with `REAL`, `FAKE`, and `UNCERTAIN / MANUAL REVIEW` outcomes.

---

# 1. Executive Summary

## 1.1 Current System Diagnosis

The current system does not have a VQC tuning problem first; it has a representation and evaluation problem.

The existing VQC test accuracy of approximately `0.5472` matches the majority-class baseline exactly:

- Test real samples: `377`
- Test fake samples: `312`
- Majority-class accuracy: `377 / 689 = 0.5472`
- Balanced accuracy: `0.50`
- Real recall: `1.00`

Therefore, the current VQC is effectively predicting `REAL` for every sample.

The existing ten physiological features also provide almost no useful class separation. Their observed individual AUC values are approximately within `0.45–0.56`, while classical models produce balanced accuracy close to chance. The current mean rPPG SNR is approximately `-4 dB` for both classes, and the 4.93-second DFDC clips provide only 148 frames at 30 FPS, producing a Welch frequency resolution of approximately `0.203 Hz`, or `12.2 BPM`.

The correct response is therefore **not** to increase VQC depth, train longer, or tune thresholds against the test set. The implementation must proceed in this order:

1. Correct evaluation leakage.
2. Establish a clean baseline.
3. Diagnose and improve rPPG extraction.
4. Expand the representation with deepfake-relevant temporal, behavioral, and rendering features.
5. Perform feature ablation and grouped validation.
6. Establish strong classical controls.
7. Only then optimize QAOA and VQC.
8. Calibrate the final three-way KYC decision system.
9. Evaluate once on an untouched final test set.

---

# 2. Confirmed Project Context

## 2.1 Datasets

### DFDC

- Location: `WORKING/RPPG/archive/DFDC_Dataset/`
- Approximately `1566 Fake + 1727 Real` MP4 clips before the HR-plausibility filtering stage.
- Face-cropped.
- Approximately `112×112`.
- Approximately 30 FPS.
- Approximately 148 frames / 4.93 seconds per clip.
- Face swaps.
- No reliable subject IDs are currently exposed by the project.

### FaceForensics++

- Location: repository root `FF++/`
- Approximately 800 MP4 files in the current project.
- Contains FF-real, YouTube-real, and FF-synthesis.
- FF-synthesis contains multiple manipulation families such as Deepfakes, Face2Face, FaceShifter, and NeuralTextures.
- Videos are full-frame rather than consistently face-cropped.
- Some videos can contain multiple people.
- Real/synthetic files can be paired by filename stem.

## 2.2 Current Feature Table

One row represents one complete video.

The current feature vector contains exactly ten physiological features:

1. `heart_rate_bpm`
2. `snr_db`
3. `prv_std_ms`
4. `spectral_entropy`
5. `mad`
6. `signal_quality_index`
7. `cheek_forehead_correlation`
8. `left_right_cheek_correlation`
9. `hr_half_diff`
10. `peak_prominence`

## 2.3 Current Model

```text
Video
  ↓
Face / ROI extraction
  ↓
POS or CHROM rPPG
  ↓
10 physiological features
  ↓
Train-fitted z-score scaling
  ↓
QAOA feature selection
  ↓
3 selected features
  ↓
3-qubit AngleEmbedding
  ↓
3-layer StronglyEntanglingLayers
  ↓
Quantum expectation values
  ↓
3 → 8 → 1 classical head
  ↓
Sigmoid
  ↓
REAL / FAKE / UNCERTAIN
```

Current QAOA-selected features:

- `left_right_cheek_correlation`
- `spectral_entropy`
- `snr_db`

---

# 3. Core Engineering Principles

## 3.1 Evidence Before Complexity

Do not add architecture complexity unless an experiment demonstrates that the additional component contributes useful information.

## 3.2 Evaluation Integrity Before Accuracy

A lower but trustworthy score is preferable to a higher score obtained through leakage.

## 3.3 Preserve the Research Contribution

Do not remove rPPG or the hybrid quantum layer merely because the current features are weak. Instead, reposition rPPG as one evidence branch in a stronger multimodal feature representation.

## 3.4 Optimize for KYC Risk, Not Raw Accuracy

The final system should support:

- `REAL`
- `FAKE`
- `UNCERTAIN / MANUAL REVIEW`

False acceptance of a fake should be treated as more serious than sending an uncertain legitimate sample for review.

## 3.5 No Fabricated Improvement

Never report an accuracy increase unless it is measured using a valid, leakage-controlled evaluation.

---

# 4. Phase 0 — Evaluation Integrity

## 4.1 Fix FF++ Pair Leakage

### Problem

An FF++ real video and its manipulated counterpart can be separated into different splits even though they share the same source identity/content.

### Required Change

Modify the FF++ split logic so that the filename stem is the group ID.

Example:

```text
id0_0000.mp4
        ↓
group_id = id0_0000
```

All files sharing the same source stem must remain in one split.

### Acceptance Criterion

No source stem may appear in more than one of:

- Train
- Validation
- Test

Create an automated assertion that fails if this condition is violated.

---

## 4.2 DFDC Subject Leakage

DFDC subject-level separation cannot currently be guaranteed because subject identities are not exposed by the current project metadata.

### Required Action

Investigate available DFDC metadata for recoverable identity/source grouping.

If no reliable identity mapping exists:

- Do not invent one.
- Document the limitation.
- Keep DFDC evaluation limitations explicit in the final report.

---

## 4.3 Final Test Isolation

The test set must not be used for:

- Feature selection.
- QAOA target-feature selection.
- Threshold tuning.
- Hyperparameter selection.
- Architecture selection.
- Repeated model comparison.

Use:

```text
Train → model fitting
Validation → development decisions
Final Test → one final evaluation
```

If extensive experimentation has already consumed the current test set, regenerate the evaluation protocol with a newly held-out final test partition.

---

# 5. Phase 1 — Baseline Reproduction

## 5.1 Establish a Frozen Baseline

Before modifying the model, run the current pipeline under the corrected split protocol.

Record:

- Accuracy
- Balanced accuracy
- Precision
- Recall
- Specificity
- F1
- ROC-AUC
- PR-AUC
- ECE
- Confusion matrix

Also save:

- Dataset split manifest.
- Feature configuration.
- Model configuration.
- Random seeds.
- Checkpoint.
- Evaluation configuration.

## 5.2 Baseline Requirement

The baseline is not a target to beat by manipulating the evaluation.

It is the reference against which every subsequent change is compared.

---

# 6. Phase 2 — rPPG Signal Recovery

## 6.1 Objective

Determine whether the physiological signal itself can be made sufficiently reliable before building additional physiological features.

## 6.2 Methods to Compare

Run controlled experiments using:

1. Current POS.
2. CHROM.
3. Green-channel baseline.

Do not assume POS is superior simply because it is theoretically appropriate.

## 6.3 ROI Configurations

Evaluate:

- Left cheek.
- Right cheek.
- Forehead.
- Whole available face region.
- Left + right cheek.
- Left + right cheek + forehead.

Evaluate each ROI configuration with each signal extraction method.

## 6.4 Signal Metrics

For each configuration calculate:

- Mean SNR.
- Median SNR.
- SNR distribution.
- Dominant frequency.
- Spectral concentration.
- Peak prominence.
- Signal quality index.
- HR stability.
- Percentage of plausible HR estimates.

Do not select a method based on fake-vs-real AUC alone. First determine whether the signal is physiologically coherent.

---

# 7. Phase 3 — Address the Short-Clip Limitation

## 7.1 Problem

DFDC clips are approximately 4.93 seconds long.

At 30 FPS:

```text
148 frames
```

The current Welch configuration uses all available samples, producing approximately:

```text
30 / 148 = 0.203 Hz
≈ 12.2 BPM
```

frequency resolution.

This explains the observed coarse HR grid.

## 7.2 Required Investigation

Where longer source videos are available, evaluate whether multiple temporal windows improve signal quality.

For short DFDC clips, do not fabricate additional temporal information through interpolation.

Possible approaches to evaluate:

- Longer available clips where permitted.
- Multiple valid temporal windows for longer FF++ videos.
- Window-level aggregation.
- Robust median/mean aggregation across valid windows.

Do not allow windows from one video to become independent train/test samples.

The video must remain the grouping unit.

---

# 8. Phase 4 — Deepfake-Relevant Feature Expansion

## 8.1 Objective

Add features that target manipulation artifacts rather than relying exclusively on physiological statistics.

The feature groups should remain compact and interpretable.

---

## 8.2 Behavioral Features

### Eye Features

Implement, where landmarks are reliable:

- Eye Aspect Ratio.
- Blink count.
- Blink duration.
- Blink interval.
- Left/right eye asymmetry.
- Temporal variance of eye measurements.

### Mouth Features

Implement:

- Mouth Aspect Ratio.
- Mouth opening frequency.
- Mouth-motion variance.
- Temporal mouth consistency.
- Left/right mouth symmetry.

### Head-Pose Features

Implement:

- Pitch.
- Yaw.
- Roll.
- Angular velocity.
- Angular acceleration.
- Temporal smoothness.

These features should describe temporal behavior, not simply one-frame measurements.

---

# 9. Phase 5 — Temporal Inconsistency Features

## 9.1 Objective

Deepfake manipulation is a temporal phenomenon. Measure whether facial structure and appearance evolve consistently from frame to frame.

## 9.2 Candidate Features

Evaluate:

- Landmark displacement.
- Landmark velocity.
- Landmark acceleration.
- Landmark trajectory variance.
- Optical-flow magnitude.
- Optical-flow variance.
- Temporal motion smoothness.
- Frame-to-frame residual energy.
- Temporal facial-region consistency.

## 9.3 Design Rule

Do not add a feature because it sounds relevant.

Every feature must be measured and tested through grouped validation.

---

# 10. Phase 6 — Rendering and Compression Features

## 10.1 Objective

Target artifacts introduced by face rendering, blending, re-encoding, and manipulation pipelines.

## 10.2 Candidate Features

Evaluate:

- Mean RGB statistics.
- RGB standard deviation.
- RGB channel ratios.
- Temporal RGB variation.
- High-frequency energy.
- Laplacian variance.
- Edge statistics.
- Local texture statistics.
- Compression/residual statistics.
- Temporal residual energy.
- Face-boundary consistency.

## 10.3 Low-Resolution Constraint

Do not assume high-resolution artifact detectors will work at 112×112.

All candidate features must be evaluated under the actual low-resolution operating condition.

---

# 11. Phase 7 — Feature Quality Screening

## 11.1 Individual Screening

For each candidate feature calculate on training/CV data:

- ROC-AUC.
- Balanced accuracy.
- Mutual information where appropriate.
- Missing-value rate.
- Variance.
- Stability across folds.

A useful initial screening heuristic is:

```text
abs(AUC - 0.5) >= 0.10
```

but this must not be treated as an absolute elimination rule.

A weak individual feature can still provide complementary information.

## 11.2 Redundancy Analysis

Calculate feature correlations and identify:

- Highly redundant features.
- Nearly constant features.
- Unstable features.
- Features strongly dependent on dataset source.

Do not allow an enormous feature set into the VQC.

---

# 12. Phase 8 — Ablation Experiments

Run the following controlled experiments.

## A — rPPG Only

```text
rPPG features
      ↓
Classifier
```

## B — Behavioral Only

```text
Blink + mouth + head pose
      ↓
Classifier
```

## C — Rendering Only

```text
RGB + texture + residual features
      ↓
Classifier
```

## D — rPPG + Behavioral

## E — rPPG + Rendering

## F — Behavioral + Rendering

## G — Complete Feature Set

```text
rPPG
+
Behavioral
+
Temporal
+
Rendering
```

All experiments must use identical leakage-safe splits.

---

# 13. Phase 9 — Classical Control Models

## 13.1 Required Models

Run at minimum:

- Logistic Regression.
- SVM.
- Random Forest.
- Gradient Boosting.
- Small MLP.

If available and justified, additional tree-based boosting models may be tested.

## 13.2 Purpose

These models are not replacements for the quantum classifier.

They answer a critical research question:

> Does the feature representation contain genuine discriminative information?

If every classical model remains near chance, do not blame the VQC.

---

# 14. Phase 10 — QAOA Feature Selection

## 14.1 Current Problem

QAOA currently selects three features that are themselves weak:

```text
left_right_cheek_correlation
spectral_entropy
snr_db
```

This selection should not be treated as inherently meaningful simply because QAOA produced it.

## 14.2 New Procedure

Once the expanded feature pool exists, compare:

1. All selected candidate features.
2. Classical top-k selection.
3. Mutual-information selection.
4. AUC-based selection.
5. QAOA selection.

Evaluate all selections under the same grouped validation protocol.

## 14.3 Research Value

If QAOA produces a compact feature subset that performs competitively or better than classical selection, that becomes a meaningful experimental result.

If it does not, report the result honestly.

---

# 15. Phase 11 — Hybrid VQC Optimization

## 15.1 Only Start This Phase After Signal Exists

Do not optimize the VQC while the classical controls are still at chance.

## 15.2 Variables to Test

Evaluate systematically:

- Number of selected features.
- Number of qubits.
- Feature-to-qubit mapping.
- Angle encoding configuration.
- Ansatz depth.
- Entanglement pattern.
- Optimizer.
- Learning rate.
- Weight decay.
- Dropout.
- Loss configuration.
- Training epochs.
- Early stopping.

## 15.3 Experimental Rule

Change a small number of variables per experiment.

Keep a reproducible experiment log.

Do not select the final configuration based on the final test set.

---

# 16. Phase 12 — KYC Decision Calibration

## 16.1 Three-Way Output

The final decision system should support:

```text
REAL
FAKE
UNCERTAIN / MANUAL REVIEW
```

## 16.2 Threshold Selection

The current `0.3 / 0.7` thresholds should be treated as initial configuration only.

Choose final thresholds on the validation set.

For example:

```text
P(REAL) >= upper threshold
    → REAL

P(REAL) <= lower threshold
    → FAKE

otherwise
    → UNCERTAIN
```

The exact values must be determined empirically.

## 16.3 KYC Risk Objective

Prefer a threshold policy that reduces false acceptance of manipulated videos while maintaining an acceptable legitimate-user review rate.

Do not optimize raw accuracy alone.

---

# 17. Phase 13 — Dataset-Specific Evaluation

Always report:

## DFDC

- Balanced accuracy.
- ROC-AUC.
- PR-AUC.
- Recall.
- Specificity.
- F1.
- Confusion matrix.

## FF++

Report the same metrics.

## Combined Dataset

Report the same metrics.

This prevents a strong result on one dataset from hiding poor performance on the other.

---

# 18. Phase 14 — Robustness Testing

Test the final candidate system against:

- Different resolutions.
- Compression.
- Illumination changes.
- Shorter valid clips.
- Noisy frames.
- Missing frames.
- Multiple-face situations in FF++.
- Different manipulation families.
- Dataset-source shifts.

The purpose is to determine whether the system has learned deepfake characteristics or simply learned dataset-specific shortcuts.

---

# 19. Required Project Architecture

The recommended architecture is:

```text
LOW-RES KYC VIDEO
        │
        ▼
Frame Quality Assessment
        │
        ▼
Face Detection / Tracking
        │
        ├───────────────────────┐
        ▼                       ▼
rPPG Branch              Behavioral Branch
        │                       │
POS / CHROM             Blink / Mouth / Pose
        │                       │
Physiological Features  Temporal Features
        │                       │
        └──────────────┬────────┘
                       ▼
             Rendering Branch
                       │
              RGB / Texture /
             Residual Features
                       │
                       ▼
                 Feature Fusion
                       │
                       ▼
              Feature Normalization
                       │
             ┌─────────┴─────────┐
             ▼                   ▼
     Classical Controls        QAOA
                                 │
                                 ▼
                          Feature Selection
                                 │
                                 ▼
                              VQC
                                 │
                                 ▼
                       Confidence Calibration
                                 │
                                 ▼
                  REAL / FAKE / UNCERTAIN
```

---

# 20. Recommended Code Organization

Do not create one giant feature-extraction file.

Prefer modular components such as:

```text
WORKING/
├── RPPG/
│   └── rppg/
│       ├── signal_extraction.py
│       ├── preprocessing.py
│       ├── face_roi.py
│       └── pipeline.py
│
├── features/
│   ├── rppg_features.py
│   ├── behavioral_features.py
│   ├── temporal_features.py
│   ├── rendering_features.py
│   ├── feature_quality.py
│   └── feature_fusion.py
│
├── quantum/
│   ├── data.py
│   ├── scaling.py
│   ├── qaoa.py
│   ├── vqc.py
│   ├── baselines.py
│   ├── evaluation.py
│   └── calibration.py
│
├── experiments/
│   ├── baseline/
│   ├── rppg/
│   ├── behavioral/
│   ├── rendering/
│   ├── ablation/
│   └── quantum/
│
└── configs/
    ├── data.yaml
    ├── features.yaml
    ├── model.yaml
    └── evaluation.yaml
```

Adapt this structure to the existing repository instead of blindly restructuring the project.

---

# 21. Implementation Rules for the Coding Agent

## Rule 1 — Inspect Before Editing

Read the relevant existing implementation before changing it.

## Rule 2 — Preserve Existing APIs Where Possible

Avoid unnecessary breaking changes.

## Rule 3 — Add Tests

Every important new component should have tests for:

- Shape.
- NaN handling.
- Empty input.
- Invalid input.
- Expected numerical ranges.
- Deterministic behavior where required.

## Rule 4 — No Silent Failure

If a feature cannot be calculated, explicitly record its failure or missing status.

Do not silently substitute arbitrary values.

## Rule 5 — No Data Leakage

Any fitted scaler, selector, threshold, or model must be fitted only on training/development data.

## Rule 6 — Keep Video as the Grouping Unit

Multiple windows from one video must never be treated as independent train/test samples.

## Rule 7 — Reproducibility

Record:

- Random seed.
- Dataset manifest.
- Git commit/version.
- Feature configuration.
- Model configuration.
- Split manifest.
- Training parameters.

## Rule 8 — Do Not Fabricate Results

If an experiment cannot be run, report it as not run.

---

# 22. Experiment Tracking

Create a machine-readable experiment registry.

Suggested columns:

```text
experiment_id
date
git_commit
dataset_version
split_version
feature_groups
feature_count
selection_method
model
qubits
ansatz_depth
learning_rate
epochs
accuracy
balanced_accuracy
precision
recall
specificity
f1
roc_auc
pr_auc
ece
notes
```

Each experiment must be reproducible from the recorded configuration.

---

# 23. Acceptance Criteria

## Evaluation

The implementation is not complete until:

- FF++ pair leakage is fixed.
- Final test data is isolated.
- Grouping logic is validated.
- PR-AUC and specificity are reported.
- Dataset-specific metrics are available.

## Feature Pipeline

The implementation must:

- Compare POS and CHROM.
- Include a green-channel baseline.
- Compare ROI strategies.
- Measure rPPG quality.
- Add selected behavioral features.
- Add selected temporal/rendering features.
- Perform feature screening and ablation.

## Model

The implementation must:

- Establish classical baselines.
- Compare classical and quantum models using identical feature inputs.
- Evaluate QAOA against classical feature selection.
- Optimize VQC only after useful signal is demonstrated.

## Decision System

The implementation must:

- Support REAL / FAKE / UNCERTAIN.
- Calibrate thresholds using validation data.
- Keep final test data untouched.

---

# 24. Definition of Success

Do not define success as:

> "Accuracy increased from 0.547 to 0.60."

That could still represent a majority-class predictor.

Success means:

1. Balanced accuracy materially exceeds `0.50`.
2. ROC-AUC materially exceeds `0.50`.
3. Fake recall is meaningfully above zero.
4. Specificity remains acceptable.
5. Results survive leakage-safe grouped validation.
6. Improvements appear on both relevant datasets where expected.
7. The improvement is reproducible across folds/seeds.
8. The final test set confirms the improvement.
9. The VQC can be fairly compared against strong classical controls.
10. The final KYC decision system provides a useful uncertainty/review region.

---

# 25. Recommended Execution Sequence

The coding agent should execute the following sequence without skipping stages:

```text
STEP 1
Audit current repository
        ↓
STEP 2
Fix FF++ grouping leakage
        ↓
STEP 3
Create clean train/validation/final-test protocol
        ↓
STEP 4
Reproduce baseline
        ↓
STEP 5
Benchmark POS / CHROM / Green
        ↓
STEP 6
Benchmark ROI configurations
        ↓
STEP 7
Diagnose rPPG quality
        ↓
STEP 8
Add behavioral features
        ↓
STEP 9
Add temporal features
        ↓
STEP 10
Add rendering/compression features
        ↓
STEP 11
Feature quality screening
        ↓
STEP 12
Ablation experiments
        ↓
STEP 13
Classical baseline comparison
        ↓
STEP 14
QAOA vs classical feature selection
        ↓
STEP 15
VQC optimization
        ↓
STEP 16
Probability calibration
        ↓
STEP 17
KYC threshold optimization
        ↓
STEP 18
Final untouched test evaluation
        ↓
STEP 19
Robustness analysis
        ↓
STEP 20
Final report
```

---

# 26. Final Implementation Instruction

The implementation agent must treat this document as an **engineering and experimental protocol**, not merely a list of suggestions.

For every phase:

1. Inspect the current implementation.
2. Implement the required change.
3. Run the relevant experiment.
4. Record the result.
5. Compare against the baseline.
6. Keep the change only if justified.
7. Move to the next phase.

Do not jump directly to VQC tuning.

Do not optimize against the final test set.

Do not fabricate missing metrics.

Do not claim that rPPG is useful unless the experiments demonstrate useful signal.

Do not claim that quantum ML improves performance unless the VQC beats or meaningfully complements the classical controls under the same leakage-safe evaluation protocol.

The final objective is a **credible, reproducible, low-resolution KYC deepfake detection system that combines physiological evidence, facial/temporal evidence, and hybrid quantum-classical classification**, with the evidence required to determine which components genuinely contribute to detection performance.
