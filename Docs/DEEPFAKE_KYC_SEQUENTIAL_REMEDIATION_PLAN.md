# Sequential Remediation Plan
## Deepfake Detection in Low-Resolution KYC Videos Using rPPG and Hybrid Quantum Machine Learning

> **Purpose:** This document is the implementation and evaluation roadmap for repairing the current deepfake/KYC detector in strict order of decreasing severity. It is written against the uploaded project repository and the supplied quantum-hybrid reference documents. Each phase has exactly one primary problem. A phase may contain multiple implementation steps, but it is not allowed to solve a different problem at the same time.

---

# 0. Project Baseline and Source Basis

## 0.1 Current Project Architecture

The current repository implements:

```text
Low-resolution KYC video
        |
        v
Stage 1: frame sampling + quality assessment
        |
        v
Stage 2: MediaPipe face ROIs -> POS/CHROM rPPG
        |
        v
20 physiological features
        |
        v
QAOA feature selection
        |
        v
Hybrid VQC
        |
        v
P(real)
        |
        +-----------------------------+
        |                             |
        v                             v
      REAL                         FAKE / UNCERTAIN
```

The active quantum layer uses the fixed feature contract in:

```text
WORKING/quantum/config.py
WORKING/RPPG/rppg/features.py
```

The repository documents the current pipeline as:

```text
frames -> rPPG -> quantum -> REAL / FAKE / UNCERTAIN
```

The rPPG layer is intended to remain the physiological evidence layer rather than being removed from the project.

---

## 0.2 Current Failure State

The latest metrics supplied with the task indicate:

| Failure | Current observation |
|---|---|
| Fake detection | 0/310 fake test videos detected |
| Specificity | 0.000 |
| Balanced accuracy | 0.499 |
| ROC-AUC | 0.535 |
| Accuracy | 0.552 |
| Decision coverage | 694/694 UNCERTAIN |
| Fake predictions | 0 |
| Classical baseline ceiling | weak; best reported baseline AUC around 0.58 |
| QAOA stability | restart costs vary substantially |
| Quantum advantage | not demonstrated |
| rPPG feature signal | individual feature AUC deviations are small |

The repository's latest documented GPU-first rerun is consistent with the same diagnosis: VQC test AUC about 0.543, grouped CV AUC about 0.529, best classical AUC about 0.570, and 100% UNCERTAIN at the current 0.3/0.7 policy.

These values must be treated as the failure baseline, not as evidence of successful deepfake detection.

---

## 0.3 Why the Order Matters

The most dangerous mistake at this point would be to begin by increasing VQC depth, adding more qubits, or lowering the uncertainty thresholds.

The correct order is:

```text
Prediction correctness
    ->
Evaluation integrity
    ->
Non-zero fake discrimination
    ->
rPPG signal quality
    ->
Feature information content
    ->
Complementary temporal/forensic evidence
    ->
Classical performance ceiling
    ->
QAOA stability and validity
    ->
Quantum attention / VQC contribution
    ->
Calibration and KYC operating point
    ->
Robustness / interpretability / deployment
```

If an earlier phase fails, later phases must not be used to hide that failure.

---

# 1. Methodology Imported From the Supplied Reference Documents

## 1.1 Quantum-Hybrid QAOA Framework

The supplied 2026 quantum-hybrid paper structures its detector into seven stages:

1. Data curation
2. Preprocessing
3. Classical feature extraction
4. QAOA-based feature selection
5. Quantum-inspired attention weighting
6. Classification
7. Interpretability

It combines relevance-based feature selection with a correlation redundancy penalty, uses QAOA/COBYLA for subset selection, applies a quantum-inspired attention mechanism using complex-valued amplitudes and Born-rule probabilities, and explicitly includes class balancing, label smoothing, confidence calibration, and ablation studies.

For this project, these ideas are adapted to the low-resolution KYC/rPPG setting rather than copied literally from the high-dimensional image setting.

Reference basis:
- `QAOA_QHybrid_Attn_2026.pdf`, especially pp. 2-7.
- `Quantumcomparison.docx`, pp. 1-2.

---

## 1.2 Xception + ViT + QNN Reference

The supplied 2025 hybrid paper combines:

```text
Xception local features
+
ViT global-context features
+
4-qubit / 2-layer QNN
```

with classical fusion before the quantum representation and a final binary classifier. It also discusses focal loss, mixup, augmentation, early stopping, and held-out evaluation.

For this project, the architecture is **not** to be transplanted literally because the target task is low-resolution KYC video and the current contribution is rPPG-based physiological evidence. The useful transferable idea is the separation of complementary evidence sources:

```text
local / fine detail
+
global / temporal context
+
quantum representation
```

This can be adapted to compact low-resolution temporal features later in the roadmap.

Reference basis:
- `QHybrid_XceptionViT_QNN_2025(2).pdf`, pp. 3-5.
- `Quantumcomparison.docx`, pp. 3-4.

---

## 1.3 Important Source Limitation

The supplied reference papers report very high performance on image-oriented datasets, including values around 98% accuracy and approximately 0.995-0.992 ROC-AUC. Those numbers are **not targets that can be directly transferred to this project** because the present system solves a different problem:

- low-resolution KYC videos;
- rPPG physiological evidence;
- temporal information;
- video-level grouping;
- REAL / FAKE / UNCERTAIN output.

The reference papers are therefore used for architecture and methodology ideas, not as evidence that the present system should attain the same scores.

---

# 2. Non-Negotiable Engineering Rules

## Rule 1 - No Test-Set Tuning

The final test set cannot be used for:

- feature selection;
- QAOA target feature count selection;
- VQC hyperparameter tuning;
- threshold tuning;
- calibration fitting;
- architecture selection.

Use:

```text
TRAIN -> learning
VALIDATION -> decisions
FINAL TEST -> one final evaluation
```

---

## Rule 2 - Video Is the Grouping Unit

If a video is divided into temporal windows, all windows from that video remain in the same split.

Never do:

```text
video -> windows -> random train/test split
```

Correct:

```text
video -> grouped split -> windows -> aggregation
```

---

## Rule 3 - No Synthetic Bridge Layer

The current repository explicitly avoids synthetic data and intermediate bridge representations.

Do not introduce:

- generated rPPG samples;
- synthetic physiological labels;
- synthetic feature vectors;
- artificial mappings designed only to improve the quantum classifier.

---

## Rule 4 - Preserve the Feature Contract

The lists in:

```text
WORKING/quantum/config.py
WORKING/RPPG/rppg/features.py
```

must remain identical in:

- names;
- order;
- semantics.

Any expansion must update the contract, tests, data serialization, scaler, selector, model, and inference path together.

---

## Rule 5 - No Silent Failure

A missing or invalid feature must not silently become a plausible numeric value.

Use explicit:

```text
valid
invalid
missing
inconclusive
```

states.

---

## Rule 6 - Do Not Claim Quantum Advantage Without Matched Controls

A quantum model must be compared against a strong classical model using:

- the same features;
- the same split;
- the same preprocessing;
- the same validation protocol;
- the same target metric.

---

# 3. Severity-Ordered Phase Map

| Phase | Primary problem | Severity | Main outcome |
|---|---|---|---|
| 1A | Possible label/probability mapping error | Critical | Prove class semantics are correct |
| 1B | Single-class REAL collapse | Critical | Restore non-zero FAKE capability |
| 1C | Misleading threshold-independent interpretation | Critical | Establish discrimination independent of decision bins |
| 2 | Test/split contamination risk | Critical | Produce immutable leakage-safe splits |
| 3 | Majority-class accuracy masking failure | Critical | Freeze honest evaluation baseline |
| 4 | rPPG extraction weakness | Critical | Identify best pulse-recovery configuration |
| 5 | Short-clip temporal limitation | High | Improve temporal evidence without leakage |
| 6 | Weak 20-feature representation | High | Quantify and improve feature signal |
| 7 | Missing complementary facial behavior | High | Add temporal facial evidence |
| 8 | Missing low-resolution forensic evidence | High | Add rendering/compression evidence |
| 9 | Feature redundancy / instability | High | Build a compact robust feature pool |
| 10 | Weak classical ceiling | High | Establish strongest classical reference |
| 11 | QAOA instability and subset validity | High | Make feature selection reproducible |
| 12 | Missing quantum-inspired attention | Medium-High | Test dynamic feature weighting |
| 13 | VQC optimization under stronger features | Medium-High | Build a valid hybrid quantum classifier |
| 14 | Class-imbalance / loss design | Medium-High | Improve difficult/fake sample learning |
| 15 | 100% UNCERTAIN decision policy | Medium-High | Produce useful safe coverage |
| 16 | Probability calibration | Medium | Make confidence operationally meaningful |
| 17 | Error-mode ignorance | Medium | Identify why remaining samples fail |
| 18 | Dataset/manipulation fragility | Medium | Measure cross-dataset robustness |
| 19 | Statistical instability | Medium | Establish repeated-split confidence |
| 20 | Training/inference mismatch | Medium | Make deployment path identical to evaluation |
| 21 | Interpretability gap | Low-Medium | Produce forensic evidence outputs |
| 22 | Security/artifact risk | Low-Medium | Protect model artifacts and inputs |
| 23 | Runtime/latency optimization | Low | Optimize only after correctness |
| 24 | Final freeze and independent evaluation | Final gate | Produce defensible final result |

---

# 4. Phase 1A - Resolve the Label and Probability Mapping Risk

## 4.1 Primary Problem

The current project uses different label conventions across stages:

```text
rPPG CSV:
1 = FAKE
0 = REAL

quantum:
LABEL_REAL = 1
LABEL_FAKE = 0

quantum/data.py:
y = 1 - csv_label

rPPG RandomForest cross-check:
1 = DEEPFAKE
```

Because the final model produces almost exclusively REAL predictions, a mapping error must be ruled out before changing the model.

---

## 4.2 Files

Inspect:

```text
WORKING/quantum/config.py
WORKING/quantum/data.py
WORKING/quantum/evaluation.py
WORKING/quantum/pipeline.py
WORKING/quantum/vqc.py
WORKING/run_pipeline.py
```

---

## 4.3 Required Implementation

Create one explicit label contract:

```python
LABEL_REAL = 1
LABEL_FAKE = 0
```

and expose conversion functions:

```python
def csv_to_quantum_label(csv_label): ...
def quantum_to_display_label(label): ...
```

Do not rely on undocumented arithmetic such as `1 - csv_label` outside one tested function.

Create a probability contract:

```text
prob_real = P(REAL)
prob_fake = 1 - prob_real
```

Every metric and threshold must consume the same semantic definition.

---

## 4.4 Required Tests

Test known examples:

```text
CSV 0 -> REAL
CSV 1 -> FAKE

P(real)=0.95 -> REAL
P(real)=0.05 -> FAKE
P(real)=0.50 -> threshold-dependent / UNCERTAIN
```

Verify that:

- confusion-matrix rows are actual labels;
- confusion-matrix columns are predicted labels;
- ROC uses the correct target;
- PR-AUC uses the intended target;
- displayed UI probability says the same class as the backend.

---

## 4.5 Acceptance Gate

PASS only if:

- all class semantics are documented;
- all conversion tests pass;
- probability direction is verified;
- the current zero-FAKE prediction behavior remains when run on the same artifact.

If the behavior changes materially after correcting mapping, return to Phase 3 and re-freeze the baseline.

---

# 5. Phase 1B - Eliminate the Single-Class REAL Collapse

## 5.1 Primary Problem

The model currently behaves like a majority-class predictor.

The target is not "higher accuracy". The target is:

```text
non-zero FAKE recall
+
balanced discrimination
```

---

## 5.2 Diagnostic Order

Before changing architecture, inspect:

1. distribution of `prob_real`;
2. logits;
3. training loss by class;
4. validation loss by class;
5. predicted class counts;
6. class weights;
7. target encoding;
8. gradient statistics;
9. selected-feature distributions.

---

## 5.3 Training Fixes

Run one controlled intervention at a time:

### Intervention A - Balanced class weighting

Use class-balanced loss or per-class weighting.

### Intervention B - Balanced focal loss

The 2026 QAOA-hybrid paper explicitly uses focal weighting and positive-class emphasis together with label smoothing and confidence control.

Map this to the present problem by giving the FAKE class explicit priority in the development objective.

### Intervention C - Label smoothing

Current code uses low label smoothing. Compare:

```text
0.00
0.03
0.05
0.10
```

without changing other parameters.

### Intervention D - Confidence penalty

Compare the existing confidence penalty with higher values only after checking whether it is suppressing useful score separation.

---

## 5.4 Acceptance Gate

A candidate passes only if the validation set shows:

```text
FAKE recall > 0
balanced accuracy > 0.50
```

and the improvement is reproducible under at least three independent seeds.

Do not move to the next phase merely because the model makes more FAKE predictions.

---

# 6. Phase 1C - Separate Discrimination Failure From Threshold Failure

## 6.1 Primary Problem

The system has:

```text
0.535 ROC-AUC
0.582 PR-AUC
100% UNCERTAIN
```

The 100% uncertainty result is partly a threshold issue, but a threshold cannot repair poor ranking.

---

## 6.2 Required Analysis

For validation data, plot:

```text
score distribution REAL
score distribution FAKE
ROC curve
precision-recall curve
threshold -> recall
threshold -> specificity
threshold -> coverage
```

Do not change the final thresholds yet.

---

## 6.3 Acceptance Gate

The project must now have a clear diagnosis:

### Case A

Scores separate the classes but 0.3/0.7 is too conservative.

Proceed to Phase 15 later.

### Case B

Scores do not separate the classes.

Proceed upstream to rPPG and feature improvement.

This phase prevents the team from "fixing" weak discrimination by blindly moving thresholds.

---

# 7. Phase 2 - Eliminate Split and Leakage Risk

## 7.1 Primary Problem

The project previously had an FF++ grouping issue. The current code groups FF++ source-related clips using the first `id` token and protects against cross-split group reuse.

This must be turned into a permanent audited data protocol.

---

## 7.2 Required Changes

Create:

```text
Scrape/split_manifest.csv
Scrape/split_audit.json
```

with at least:

```text
video_path
dataset
group_id
label
split
```

For FF++:

- pair/source grouping must remain together.

For YouTube-real:

- each unpaired clip can remain its own group.

For DFDC:

- investigate available metadata for subject/source grouping.
- if reliable subject identity cannot be reconstructed from the available project data, document that limitation instead of inventing an identity key.

---

## 7.3 Automated Assertions

The build must fail if:

```text
group_id appears in > 1 split
```

or if exact duplicate feature rows are detected across splits.

Add these tests to:

```text
WORKING/quantum/tests.py
```

---

## 7.4 Acceptance Gate

Only a leakage-audited split can be used by later phases.

---

# 8. Phase 3 - Freeze an Honest Evaluation Baseline

## 8.1 Primary Problem

Raw accuracy currently hides the fact that the model is essentially predicting REAL.

The frozen evaluation must make this impossible.

---

## 8.2 Required Metrics

Report all of:

```text
Accuracy
Balanced accuracy
REAL precision
REAL recall
FAKE precision
FAKE recall
Specificity
Macro-F1
FAKE-F1
ROC-AUC
PR-AUC
ECE
Confusion matrix
REAL coverage
FAKE coverage
UNCERTAIN coverage
```

For KYC, also report:

```text
False acceptance rate for FAKE -> REAL
False rejection rate for REAL -> FAKE
Manual-review rate
Automated coverage
```

---

## 8.3 Majority Baseline

Add:

```text
predict REAL for every sample
```

as a formal baseline.

The current 0.55-range accuracy must be compared against it.

---

## 8.4 Required Artifact

Create:

```text
WORKING/output/quantum/baseline_<timestamp>/
```

containing:

```text
data_manifest.json
config.json
metrics_quantum.json
metrics_baselines.json
confusion_matrix.png
roc_curve.png
calibration_curve.png
```

---

## 8.5 Acceptance Gate

The project now has one frozen reference. Every later phase must report:

```text
change from baseline
```

rather than only an absolute score.

---

# 9. Phase 4 - Repair the rPPG Signal Recovery Layer

## 9.1 Primary Problem

Classical models also perform poorly on the current 20 physiological features, strongly indicating that the upstream physiological representation is weak.

---

## 9.2 Methods to Compare

Using exactly the same split, compare:

```text
POS
CHROM
Green-channel baseline
```

Do not change the classifier during this phase.

---

## 9.3 ROI Configurations

Compare:

```text
left cheek
right cheek
forehead
left + right cheek
left + right cheek + forehead
```

For every configuration record:

- valid ROI rate;
- usable frame rate;
- SNR;
- SQI;
- HR plausibility;
- HR stability;
- inter-ROI correlation;
- feature extraction failure rate.

---

## 9.4 Signal Diagnostics

Add diagnostic outputs from:

```text
WORKING/RPPG/rppg/signal_extraction.py
WORKING/RPPG/rppg/face_roi.py
WORKING/RPPG/rppg/pipeline.py
```

Store representative traces for:

- high-quality REAL;
- low-quality REAL;
- high-quality FAKE;
- low-quality FAKE.

The goal is to discover whether the recovered pulse is genuinely physiological or simply a noisy spectral signal.

---

## 9.5 Acceptance Gate

Select the extraction configuration that produces the best combination of:

```text
signal validity
signal stability
and
validation class separation
```

Do not select based on one metric alone.

---

# 10. Phase 5 - Address the Short-Clip Temporal Limitation

## 10.1 Primary Problem

The project frequently operates on very short clips. Short clips produce coarse frequency resolution and weak temporal statistics.

The repository documentation notes approximately 4.9 seconds and roughly 148 frames for DFDC-like clips.

---

## 10.2 Required Strategy

Do not fabricate temporal duration.

For longer videos:

```text
whole clip
+
multiple valid temporal windows
```

For short videos:

```text
use all available temporal evidence
```

but do not pretend multiple windows are independent videos.

---

## 10.3 Temporal Window Experiments

Compare where sufficient data exists:

```text
2 s windows
3 s windows
4 s windows
full clip
overlapping windows
non-overlapping windows
```

Aggregate at the video level using:

```text
median
trimmed mean
quality-weighted mean
```

---

## 10.4 Acceptance Gate

Select the temporal policy that:

- improves feature stability;
- does not increase leakage;
- improves validation discrimination;
- does not create a large increase in missing windows.

---

# 11. Phase 6 - Determine Whether the Current 20 Features Contain Signal

## 11.1 Primary Problem

The present feature set contains:

```text
HR
SNR
PRV
spectral entropy
MAD
SQI
cross-ROI correlation
temporal HR difference
peak prominence
morphology
phase lag
motion ratio
stability features
```

but current per-feature AUC deviations are small.

---

## 11.2 Per-Feature Audit

For every feature calculate on training/development data:

```text
ROC-AUC
PR-AUC
missing rate
variance
class effect
fold-to-fold stability
dataset dependence
```

Also calculate:

```text
correlation matrix
```

to identify redundant features.

---

## 11.3 Important Interpretation Rule

A feature with:

```text
AUC ~ 0.50
```

is not automatically useless in a multivariate model.

A feature can still be complementary.

Therefore:

```text
individual AUC = screening evidence
not
final elimination rule
```

---

## 11.4 Acceptance Gate

The output of this phase must classify each feature as:

```text
strong
weak but complementary
redundant
unstable
invalid
```

and produce:

```text
Scrape/feature_audit.csv
```

---

# 12. Phase 7 - Add Temporal Facial-Behavior Evidence

## 12.1 Primary Problem

rPPG alone may not contain enough information at low resolution.

The supplied literature explicitly discusses temporal motion patterns, motion continuity, head-pose inconsistency, recurrent temporal modeling, and frame-to-frame behavior as useful deepfake evidence.

---

## 12.2 Feature Branch

Add compact, interpretable features for:

### Eyes

```text
Eye Aspect Ratio
blink count
blink duration
blink interval
eye asymmetry
eye-motion variance
```

### Mouth

```text
Mouth Aspect Ratio
mouth-motion frequency
mouth-motion variance
mouth symmetry
temporal mouth consistency
```

### Head Pose

```text
pitch
yaw
roll
angular velocity
angular acceleration
pose smoothness
```

---

## 12.3 Implementation Principle

Do not add a large RNN yet.

First test whether simple temporal statistics contain information.

Suggested module:

```text
WORKING/features/behavioral_features.py
```

or the closest modular location that preserves the current repository organization.

---

## 12.4 Acceptance Gate

Compare:

```text
rPPG only
behavior only
rPPG + behavior
```

on identical grouped validation splits.

Keep the branch only if it adds independent validation signal.

---

# 13. Phase 8 - Add Low-Resolution Rendering and Compression Evidence

## 13.1 Primary Problem

Deepfake detection should not rely only on physiological evidence.

The 2026 reference framework explicitly motivates manipulation artifacts, compression inconsistencies, color irregularities, and spatial/temporal inconsistencies as forensic evidence.

---

## 13.2 Candidate Features

At the actual project resolution, test:

```text
RGB mean
RGB variance
channel ratios
temporal color variation
Laplacian variance
edge density
high-frequency energy
local texture statistics
temporal residual energy
face-boundary consistency
```

Avoid features that require high-frequency detail not reliably present in low-resolution input.

---

## 13.3 Optional Compact Visual Backbone

Only if handcrafted low-resolution features remain weak, evaluate a compact backbone branch inspired by the Xception/ViT literature:

```text
low-resolution face sequence
        |
        +--> compact CNN embedding
        |
        +--> compact temporal/global embedding
        |
        +--> feature fusion
```

Do not immediately replace the rPPG pipeline with a high-resolution image model.

---

## 13.4 Acceptance Gate

The new branch must demonstrate complementary information over:

```text
rPPG + behavioral
```

before it becomes part of the final model.

---

# 14. Phase 9 - Build a Clean Feature Pool

## 14.1 Primary Problem

Adding features indefinitely creates a new high-dimensional, redundant, and unstable representation.

---

## 14.2 Required Feature Processing

Create a train-fitted preprocessing chain:

```text
raw features
   ->
non-finite rejection
   ->
domain-range validation
   ->
robust outlier handling
   ->
train-only scaling
   ->
optional correlation pruning
```

Do not fit any transformation on the final test set.

---

## 14.3 Feature Contract

The feature manifest must contain:

```text
name
group
source module
units
expected range
missing policy
normalization policy
```

---

## 14.4 Acceptance Gate

A clean feature matrix must be reproducible from the same input video and configuration.

---

# 15. Phase 10 - Ablate the Evidence Branches

## 15.1 Primary Problem

Without ablation, there is no evidence about which branch actually helps.

---

## 15.2 Required Experiments

Run:

```text
A: rPPG only
B: behavioral only
C: rendering only
D: rPPG + behavioral
E: rPPG + rendering
F: behavioral + rendering
G: all branches
```

For each record:

```text
balanced accuracy
FAKE recall
specificity
ROC-AUC
PR-AUC
Macro-F1
ECE
```

---

## 15.3 Required Interpretation

The complete system should not be credited with gains that come from a branch the research contribution does not actually justify.

The ablation table must explicitly identify:

```text
incremental contribution
```

of each branch.

---

# 16. Phase 11 - Establish the Classical Performance Ceiling

## 16.1 Primary Problem

The current classical baselines are weak, but the exact ceiling of the improved feature pool is unknown.

---

## 16.2 Models

Use matched feature inputs:

```text
Logistic Regression
Linear SVM
Gaussian NB
Random Forest
Gradient Boosting / XGBoost
Small MLP
Calibrated classifier
```

Where class imbalance exists, use appropriate class weighting only on training folds.

---

## 16.3 Primary Ranking

Rank by:

1. FAKE recall
2. Balanced accuracy
3. PR-AUC
4. ROC-AUC
5. Macro-F1
6. KYC false-acceptance rate
7. Coverage at controlled risk

Do not rank by accuracy alone.

---

## 16.4 Acceptance Gate

At least one classical model must show meaningful separation before investing heavily in quantum optimization.

If all classical models remain near chance, return to Phases 6-10.

This prevents the project from using the quantum layer to conceal weak upstream evidence.

---

# 17. Phase 12 - Stabilize and Validate QAOA Feature Selection

## 17.1 Primary Problem

The current QAOA restart costs vary widely and the selected features overlap poorly with the classical reference.

The current selection therefore cannot be treated as a stable scientific result.

---

## 17.2 QAOA Objective

Retain the current strong principle:

```text
selection cost
=
relevance
+
redundancy penalty
+
cardinality constraint
```

The supplied 2026 reference paper also uses relevance and correlation penalties in the Hamiltonian.

For the present project, keep the supervised discrimination weighting already implemented in:

```text
qaoa._discrimination_weights
```

but compare it against:

```text
mutual information
top-k AUC
classical greedy selection
```

---

## 17.3 Stability Experiments

Across repeated seeds, record:

```text
selected subset
objective value
optimizer status
restart spread
feature selection frequency
Jaccard similarity
```

---

## 17.4 Cardinality Sweep

Do not assume `k = 3`.

Evaluate:

```text
k = 3
k = 5
k = 7
k = 10
```

only after the expanded feature pool justifies the larger values.

---

## 17.5 High-Dimensional Adaptation

The 2026 reference paper uses block-wise QAOA because its original feature space is huge.

The current project has only 20 features, so block-wise decomposition is unnecessary now.

If the feature pool grows substantially, introduce:

```text
feature pool
   ->
blocks
   ->
local QAOA selection
   ->
global selection
```

only when the direct Hamiltonian becomes computationally or statistically impractical.

---

## 17.6 Acceptance Gate

QAOA passes only if:

- the Hamiltonian reproduces the classical cost;
- beta remains active;
- repeated runs are reasonably stable;
- the selected subset is competitive with classical selection;
- validation performance improves or matches the best baseline.

---

# 18. Phase 13 - Add Quantum-Inspired Attention

## 18.1 Primary Problem

QAOA selects a subset, but it does not dynamically model interactions among those selected features.

The supplied 2026 paper addresses this with complex-valued feature amplitudes and Born-rule probabilities.

---

## 18.2 Adapted Architecture

For selected features:

```text
x_i
 |
 v
learnable amplitude ψ_i = a_i exp(j θ_i)
 |
 v
normalize
 |
 v
attention weight p_i = |ψ_i|^2
 |
 v
weighted feature representation
 |
 v
classical attention / residual block
 |
 v
VQC
```

The feature dimensionality should remain small enough for stable training.

---

## 18.3 Do Not Copy Image-Specific Dimensions

Do not implement:

```text
512 -> 256
```

just because the reference paper does.

This project is not using 512-dimensional InceptionResNetV1 image embeddings.

Use the actual selected physiological/behavioral/forensic feature count.

---

## 18.4 Acceptance Gate

Compare:

```text
QAOA + VQC
QAOA + quantum-inspired attention + VQC
```

under identical splits and seeds.

The attention module is accepted only if it improves validation evidence.

---

# 19. Phase 14 - Optimize the VQC Architecture

## 19.1 Primary Problem

The current VQC is too weak to interpret as a meaningful quantum advantage when the input representation itself is weak.

Only begin this phase after Phase 11 has produced useful classical discrimination.

---

## 19.2 Baseline Quantum Model

Retain a minimal reference:

```text
selected normalized features
      ->
angle encoding
      ->
StronglyEntanglingLayers
      ->
expectation values
      ->
small classical head
      ->
P(real)
```

---

## 19.3 Controlled Architecture Sweep

Sweep one family at a time:

### Feature count

```text
3
5
7
10
```

### Quantum depth

```text
1
2
3
4
```

### Entanglement

```text
ring
all-to-all where computationally practical
```

### Classical head

```text
linear
small MLP
residual MLP
```

### Encoding

Compare only theoretically justified encodings, for example:

```text
RY angle encoding
RY/RZ mixed encoding
```

---

## 19.4 Acceptance Gate

The final quantum architecture must beat or meaningfully complement the strongest classical baseline on development data.

If it does not:

```text
do not claim quantum advantage
```

and retain the strongest scientifically defensible interpretation.

---

# 20. Phase 15 - Repair the Loss and Class-Balance Strategy

## 20.1 Primary Problem

The current model has historically collapsed to REAL, which makes loss design a first-class concern.

---

## 20.2 Reference Method

The supplied 2026 framework combines:

```text
focal weighting
+
class balancing
+
label smoothing
+
confidence penalty
```

The current VQC code already contains these concepts, so this phase is a controlled re-tuning rather than a wholesale rewrite.

---

## 20.3 Controlled Sweep

Evaluate:

```text
gamma: 1, 2, 3
alpha: class-balanced alternatives around current 0.75
label smoothing: 0.00, 0.03, 0.05, 0.10
confidence penalty: 0, current, moderately higher
```

Do not sweep all values simultaneously.

---

## 20.4 Acceptance Gate

The selected loss must improve:

```text
FAKE recall
balanced accuracy
```

without causing unacceptable calibration or specificity collapse.

---

# 21. Phase 16 - Fix the 100% UNCERTAIN Decision Policy

## 21.1 Primary Problem

The current:

```text
P(real) >= 0.7 -> REAL
P(real) <= 0.3 -> FAKE
otherwise -> UNCERTAIN
```

produces 100% UNCERTAIN.

---

## 21.2 Important Rule

Do not simply lower both thresholds until the number of confident predictions looks reasonable.

Thresholds must be chosen from the validation score distribution.

---

## 21.3 KYC-First Objective

Define:

```text
False acceptance:
FAKE -> REAL
```

as the primary security risk.

Define:

```text
False rejection:
REAL -> FAKE
```

as the secondary user-friction risk.

Define:

```text
UNCERTAIN:
manual review / escalation
```

as the safety valve.

---

## 21.4 Required Curves

Generate:

```text
coverage vs false acceptance
coverage vs fake recall
coverage vs manual review rate
threshold vs specificity
threshold vs FAKE recall
```

Select thresholds on validation data only.

---

## 21.5 Acceptance Gate

The final policy must achieve:

- non-zero useful automated coverage;
- non-zero FAKE detection;
- controlled false acceptance;
- a meaningful UNCERTAIN band.

---

# 22. Phase 17 - Calibrate Probabilities

## 22.1 Primary Problem

A model can have useful ranking but poor probability calibration.

ECE = 0.06-0.07 is not sufficient evidence of a useful KYC confidence system when discrimination is weak.

---

## 22.2 Calibration Methods

After freezing the classifier:

```text
temperature scaling
Platt-style calibration
isotonic regression
```

may be compared on validation data.

---

## 22.3 Required Metrics

Report:

```text
ECE
Brier score
reliability diagram
probability histogram
class-conditional confidence
```

---

## 22.4 Acceptance Gate

Calibration must improve confidence reliability without destroying discrimination.

---

# 23. Phase 18 - Perform Structured Error Analysis

## 23.1 Primary Problem

Aggregate metrics do not reveal why the model fails.

---

## 23.2 Required Error Categories

For every validation/test error, where metadata exists, record:

```text
frame quality
face confidence
usable-frame ratio
rPPG SQI
SNR
motion contamination
clip duration
resolution
compression
dataset
manipulation family
multiple-face condition
```

---

## 23.3 Required Artifacts

Create:

```text
Scrape/error_analysis.csv
Scrape/error_examples/
```

with representative cases:

```text
false REAL
false FAKE
UNCERTAIN
correct REAL
correct FAKE
```

---

## 23.4 Acceptance Gate

Identify the top recurring failure causes.

Only then decide whether another feature branch is justified.

---

# 24. Phase 19 - Evaluate Dataset and Manipulation Robustness

## 24.1 Primary Problem

A detector can learn dataset-specific shortcuts.

The final system must therefore be evaluated by source.

---

## 24.2 Required Slices

At minimum:

```text
DFDC
FF++ REAL / synthesis
FF++ manipulation family where available
combined
```

Where metadata permits, further slice by:

```text
resolution
compression
clip length
quality
```

---

## 24.3 Metrics Per Slice

Report:

```text
balanced accuracy
FAKE recall
specificity
ROC-AUC
PR-AUC
Macro-F1
ECE
UNCERTAIN rate
```

---

## 24.4 Acceptance Gate

Do not use an aggregate improvement to claim robustness if one dataset collapses.

---

# 25. Phase 20 - Establish Repeated Statistical Stability

## 25.1 Primary Problem

One split and one seed can create false confidence.

---

## 25.2 Protocol

After the architecture and feature pipeline are frozen:

```text
3-5 repeated grouped validation runs
```

or a suitable repeated grouped cross-validation protocol.

Report:

```text
mean +/- std
```

for:

- balanced accuracy;
- FAKE recall;
- specificity;
- ROC-AUC;
- PR-AUC;
- Macro-F1;
- ECE.

---

## 25.3 Acceptance Gate

A claimed improvement must be:

```text
repeatable
directionally consistent
not dependent on one seed
```

---

# 26. Phase 21 - Enforce Training/Inference Equivalence

## 26.1 Primary Problem

Offline evaluation and end-to-end inference can diverge.

Current deployment flow is:

```text
WORKING/run_pipeline.py
    ->
frame stage
    ->
RPPG
    ->
quantum.pipeline.predict_features
```

The feature contract, scaler, selected indices, checkpoint, probability direction, and thresholds must be identical.

---

## 26.2 Required Test

For one saved feature vector:

```text
offline preprocessing
==
inference preprocessing
```

For one saved probability:

```text
evaluation probability
==
deployment probability
```

within a small floating-point tolerance.

---

## 26.3 Acceptance Gate

No deployment release if:

```text
training path != inference path
```

or if the UI displays a different class probability than the backend.

---

# 27. Phase 22 - Add Forensic Interpretability

## 27.1 Primary Problem

The model currently provides a verdict, probability, and rPPG evidence, but the forensic story can be made more explicit.

The supplied 2026 framework emphasizes interpretability through:

```text
feature importance
quantum probabilities
attention patterns
Grad-CAM
```

---

## 27.2 Adaptation to This Project

For rPPG features, provide:

```text
selected feature names
QAOA selection probability
feature contribution / importance
ROI contribution
temporal window contribution
signal-quality evidence
```

If a visual CNN/ViT branch is introduced, add:

```text
Grad-CAM or equivalent spatial explanation
```

for that branch only.

Do not claim that Grad-CAM explains the rPPG quantum layer.

---

## 27.3 Acceptance Gate

A final result must show enough evidence for an examiner to answer:

```text
Why did the system call this video REAL, FAKE, or UNCERTAIN?
```

without relying only on one scalar probability.

---

# 28. Phase 23 - Protect Model and Dataset Artifacts

## 28.1 Primary Problem

The repository loads:

```text
rppg_classifier.pkl
```

using `pickle.load()`.

An untrusted pickle can execute arbitrary code if replaced.

---

## 28.2 Required Controls

Protect:

```text
output/rppg/
output/quantum/
model checkpoints
scaler files
QAOA selection
```

from arbitrary upload overwrite.

Ensure:

```text
uploaded video
!= model artifact
```

and that the frontend cannot replace model files.

---

## 28.3 Acceptance Gate

A user upload must never be able to modify model artifacts.

---

# 29. Phase 24 - Optimize Runtime Only After Correctness

## 29.1 Primary Problem

The project already contains meaningful runtime work:

- vectorized POS;
- shared Welch PSD;
- cached transforms;
- cached VQC models;
- lazy imports;
- GPU-aware classical head;
- batch quantum simulation.

Do not destabilize these optimizations while detection performance is unresolved.

---

## 29.2 Only After Performance Is Frozen

Measure:

```text
frame extraction latency
face detection latency
rPPG latency
feature latency
QAOA build-time cost
VQC inference latency
total end-to-end latency
```

---

## 29.3 Acceptance Gate

Any speed optimization is rejected if it causes a material degradation in:

```text
FAKE recall
balanced accuracy
false acceptance
```

---

# 30. Phase 25 - Final Model Freeze

## 30.1 Primary Problem

Continuous tuning can overfit the evaluation process.

---

## 30.2 Freeze

Freeze:

```text
dataset manifest
grouping rules
feature list
ROI configuration
rPPG method
temporal policy
feature scaler
feature selection method
QAOA configuration
QAOA selected subset
attention configuration
VQC architecture
loss configuration
calibration method
thresholds
```

---

## 30.3 Acceptance Gate

No further development decisions are allowed using the final test set after this point.

---

# 31. Phase 26 - Independent Final Test

## 31.1 Primary Problem

The final score must be produced once under a frozen configuration.

---

## 31.2 Required Outputs

Produce:

```text
final_metrics.json
final_confusion_matrix.png
final_roc.png
final_pr_curve.png
final_calibration.png
final_error_analysis.csv
final_decision_policy.json
final_experiment_manifest.json
```

---

## 31.3 Required Table

| Model | Balanced Acc. | FAKE Recall | Specificity | ROC-AUC | PR-AUC | Macro-F1 | ECE | UNCERTAIN Rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Majority baseline | | | | | | | | |
| Logistic Regression | | | | | | | | |
| Best classical model | | | | | | | | |
| Classical + selected features | | | | | | | | |
| QAOA + classical head | | | | | | | | |
| QAOA + quantum attention + VQC | | | | | | | | |

Use measured values only.

---

# 32. Final Success Definition

## 32.1 Minimum Technical Success

The project is considered repaired only if all are true:

1. Label/probability semantics are verified.
2. No train/test leakage remains under the declared grouping protocol.
3. The model detects a non-zero number of fake videos.
4. Balanced accuracy is meaningfully above 0.50.
5. ROC-AUC is meaningfully above 0.50.
6. FAKE recall is meaningfully above 0.
7. The result is better than the majority-class baseline.
8. The feature representation demonstrates measurable class information.
9. The 100% UNCERTAIN failure is eliminated or justified by an evidence-based KYC policy.
10. Results are stable across repeated validation.
11. The VQC is fairly compared with classical controls.
12. The final test set remains untouched until the architecture is frozen.

---

## 32.2 Strong Research Success

A stronger outcome would additionally show:

```text
rPPG signal
    +
behavioral evidence
    +
low-resolution forensic evidence
```

provide complementary information;

```text
QAOA
```

selects a stable compact subset;

```text
quantum-inspired attention
```

improves feature weighting;

```text
VQC
```

matches or improves upon strong classical controls;

and the final confidence-aware KYC policy gives useful automated coverage at a controlled fake-acceptance risk.

---

# 33. Decision Rules Between Phases

## If Phase 1A Fails

Fix label semantics before any model change.

---

## If Phase 1B Fails

Do not proceed to quantum architecture tuning.

Investigate the training signal, class weighting, scaling, and selected-feature distributions.

---

## If Phase 4 Fails

Change the rPPG extraction method/ROI policy before adding a deeper classifier.

---

## If Phase 6 Finds No Signal

Do not increase VQC complexity.

Proceed to complementary behavioral and rendering features.

---

## If Phase 11 Classical Models Are Still Near Chance

Do not claim quantum failure.

The feature representation is still the primary bottleneck.

Return upstream.

---

## If Phase 12 QAOA Is Unstable

Do not use the selected subset as scientific evidence.

Increase restart analysis, objective validation, and classical comparison.

---

## If Phase 13 Quantum Attention Does Not Help

Remove it from the production path but retain the ablation result.

A failed experimental module is acceptable. An unexplained module is not.

---

## If Phase 14 VQC Does Not Beat the Classical Control

Do not claim quantum advantage.

Report:

```text
classical ceiling
quantum result
difference
```

honestly.

---

## If Phase 16 Still Produces Near-100% UNCERTAIN

Do not lower thresholds blindly.

Return to Phase 1C and determine whether the problem is calibration or discrimination.

---

# 34. Required Experiment Registry

Create a machine-readable registry such as:

```text
Scrape/experiment_registry.csv
```

Columns:

```text
experiment_id
phase
date
git_commit
dataset_version
split_version
seed
rppg_method
roi_policy
temporal_policy
feature_groups
feature_count
selection_method
qaoa_k
qaoa_seed
quantum_layers
entanglement
loss
alpha
gamma
label_smoothing
confidence_penalty
learning_rate
weight_decay
accuracy
balanced_accuracy
fake_precision
fake_recall
specificity
macro_f1
roc_auc
pr_auc
ece
uncertain_rate
notes
decision
```

Every retained result must be reproducible from this registry.

---

# 35. Required Per-Phase Completion Template

For every phase, create a short report in:

```text
Scrape/phases/
```

using:

```text
PhaseXX_<problem_name>.md
```

Each report must contain:

```markdown
# Phase XX - <Problem>

## Problem
<one primary problem>

## Baseline
<previous phase result>

## Change
<exact implementation change>

## Experiment
<dataset / split / seed / metrics>

## Result
<measured result>

## Decision
KEEP / REJECT / REPEAT

## Evidence
<artifact paths>

## Next Gate
<what must be true before Phase XX+1>
```

---

# 36. Recommended End-to-End Execution Order

The coding agent should execute strictly in this order:

```text
1A  Label / probability contract
  |
1B  Single-class collapse
  |
1C  Threshold-independent discrimination diagnosis
  |
2   Split / leakage audit
  |
3   Frozen honest baseline
  |
4   POS / CHROM / Green + ROI probe
  |
5   Short-clip temporal strategy
  |
6   Existing-feature information audit
  |
7   Behavioral temporal features
  |
8   Low-resolution forensic features
  |
9   Feature quality + normalization contract
  |
10  Evidence-branch ablation
  |
11  Strong classical ceiling
  |
12  QAOA stability + classical selection comparison
  |
13  Quantum-inspired attention
  |
14  VQC architecture optimization
  |
15  Loss / class-balance optimization
  |
16  KYC three-way threshold selection
  |
17  Probability calibration
  |
18  Structured error analysis
  |
19  Dataset/manipulation robustness
  |
20  Repeated statistical validation
  |
21  Training/inference equivalence
  |
22  Forensic interpretability
  |
23  Security / artifact protection
  |
24  Runtime optimization
  |
25  Final freeze
  |
26  Independent final test
```

---

# 37. What Must Not Be Done

## Do Not Do This

```text
Increase VQC depth first
```

because a complex classifier cannot create discriminative information absent from the input.

---

## Do Not Do This

```text
Lower uncertainty thresholds until coverage looks good
```

because this can increase false acceptance of deepfakes.

---

## Do Not Do This

```text
Use test AUC to choose QAOA k
```

because the test set then becomes part of model selection.

---

## Do Not Do This

```text
Treat F1 = 0.71 as evidence of a good detector
```

while the model predicts only the dominant class.

---

## Do Not Do This

```text
Claim quantum advantage because QAOA or VQC is present
```

without matched classical controls.

---

## Do Not Do This

```text
Copy 98% accuracy from the supplied papers as the project target
```

because the supplied papers evaluate different image-oriented tasks and datasets.

---

# 38. Final Architecture After Successful Remediation

The target architecture is:

```text
LOW-RES KYC VIDEO
        |
        v
Frame sampling + quality assessment
        |
        v
Face detection / tracking
        |
        +-----------------------------+
        |                             |
        v                             v
   rPPG branch                  Behavior branch
        |                             |
 POS / CHROM / Green         Eyes / Mouth / Pose
        |                             |
 physiological features       temporal features
        |                             |
        +--------------+--------------+
                       |
                       v
             Low-res forensic branch
                       |
             color / texture /
             temporal residuals
                       |
                       v
                 Feature fusion
                       |
                       v
                Train-only scaling
                       |
             +---------+---------+
             |                   |
             v                   v
      Classical controls       QAOA
                                 |
                                 v
                         stable selected subset
                                 |
                                 v
                    Quantum-inspired attention
                                 |
                                 v
                                VQC
                                 |
                                 v
                        calibrated P(REAL)
                                 |
                                 v
                    KYC decision policy
                                 |
                    +------------+------------+
                    |            |            |
                    v            v            v
                  REAL         FAKE       UNCERTAIN
                                            /
                                      MANUAL REVIEW
```

This architecture preserves the original project contribution:

```text
rPPG physiological evidence
+
hybrid quantum-classical ML
```

while adding complementary evidence only where experiments prove that the original representation is insufficient.

---

# 39. Source Integration Matrix

| Source | Method integrated into project | Where used |
|---|---|---|
| `QAOA_QHybrid_Attn_2026.pdf` | relevance + redundancy QAOA objective | Phase 12 |
| `QAOA_QHybrid_Attn_2026.pdf` | quantum-inspired attention with complex amplitudes / Born-rule weighting | Phase 13 |
| `QAOA_QHybrid_Attn_2026.pdf` | balanced focal loss, label smoothing, confidence penalty | Phase 15 |
| `QAOA_QHybrid_Attn_2026.pdf` | interpretability outputs and ablation methodology | Phases 10, 22 |
| `QHybrid_XceptionViT_QNN_2025(2).pdf` | complementary local/global evidence fusion concept | Phase 8 |
| `QHybrid_XceptionViT_QNN_2025(2).pdf` | compact quantum feature-processing architecture | Phase 14 |
| `QHybrid_XceptionViT_QNN_2025(2).pdf` | focal-loss / augmentation / early-stopping concepts | Phase 15 |
| `Quantumcomparison.docx` | QAOA + attention + interpretability architecture summary | Phases 12-13, 22 |
| `Quantumcomparison.docx` | classical-vs-hybrid comparative evaluation | Phase 11 and final report |
| Project ZIP | existing rPPG, frame, QAOA, VQC and decision pipeline | All phases |
| Project ZIP `Docs/deepfake_accuracy_enhancement_implementation_guidelines.md` | existing phase logic, feature families, KYC risk objective | Phases 4-26 |
| Project ZIP `AGENTS.md` | label contract, no-synthetic-data rule, existing QAOA guards and deployment constraints | Phases 1A-26 |

---

# 40. Final Instruction to the Implementation Agent

This plan is an **ordered experimental protocol**, not a suggestion list.

For every phase:

1. Inspect the relevant current code.
2. Change only the component needed for that phase.
3. Run the required tests.
4. Run the phase experiment.
5. Save the metrics and artifacts.
6. Compare against the previous accepted phase.
7. Mark the phase:
   - `PASS`
   - `REPEAT`
   - `REJECT`
8. Move forward only when the gate passes.

The most important principle is:

> **Do not use a more complicated downstream model to compensate for a broken upstream representation or an invalid evaluation protocol.**

The project should only move toward more sophisticated quantum components after it has demonstrated that the low-resolution KYC input actually contains exploitable real-vs-fake information.

---

# 41. Final Deliverable Checklist

Before calling the project "fixed", confirm:

```text
[ ] Label mapping verified
[ ] Probability direction verified
[ ] No class inversion
[ ] No split leakage
[ ] Final test isolated
[ ] Majority-class baseline reported
[ ] Per-class metrics reported
[ ] POS vs CHROM vs Green evaluated
[ ] ROI strategies evaluated
[ ] rPPG quality measured
[ ] Temporal strategy validated
[ ] Current features audited
[ ] Behavioral features evaluated
[ ] Low-resolution forensic features evaluated
[ ] Feature branches ablated
[ ] Classical ceiling established
[ ] QAOA compared against classical selection
[ ] QAOA stability demonstrated
[ ] Quantum-inspired attention ablated
[ ] VQC optimized only after signal exists
[ ] FAKE recall is non-zero and useful
[ ] Balanced accuracy exceeds chance
[ ] ROC-AUC exceeds chance
[ ] 100% UNCERTAIN failure resolved or justified
[ ] Probability calibration validated
[ ] KYC risk operating point selected on validation data
[ ] Dataset-specific robustness reported
[ ] Repeated validation reported
[ ] Train/inference equivalence verified
[ ] Interpretability evidence available
[ ] Model artifacts protected
[ ] Runtime measured only after model freeze
[ ] Final test run once after freeze
```

---

# 42. Source References Used for This Plan

1. **QAOA_QHybrid_Attn_2026.pdf**  
   *A Quantum-Hybrid Framework for Enhanced Deepfake Detection: Integrating QAOA-Based Feature Selection With Quantum-Inspired Attention Mechanisms.*  
   Key sections used: motivation/research gaps, system architecture, QAOA selection, quantum-inspired attention, balanced focal loss, interpretability, and comprehensive ablation/evaluation.

2. **QHybrid_XceptionViT_QNN_2025(2).pdf**  
   *Quantum-Enhanced Deepfake Detection: A Hybrid Classical-Quantum Deep Learning Approach.*  
   Key sections used: Xception + ViT + QNN architecture, feature fusion, 4-qubit/2-layer QNN, training strategy, focal-loss motivation, and held-out evaluation.

3. **Quantumcomparison.docx**  
   Comparison summary of the above two quantum-hybrid approaches, including the seven-stage QAOA/attention framework and the Xception/ViT/QNN hybrid design.

4. **Uploaded project archive: `Major_project-main(2).zip`**  
   Primary engineering source for the current repository structure, implementation constraints, current pipeline, feature contract, QAOA/VQC implementation, tests, baseline status, frontend decision path, and existing accuracy-enhancement guidance.

