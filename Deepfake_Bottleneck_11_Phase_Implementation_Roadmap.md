# Bottleneck-Driven Improvement Roadmap

## Deepfake Detection Using rPPG + Hybrid Quantum Machine Learning

> **Dataset constraint:** This roadmap assumes that the existing dataset is not changed. No new videos, external samples, or additional physiological ground-truth data are introduced. The objective is to improve the processing, feature representation, experimentation, and decision-making pipeline using the existing videos.

---

# Phase 1 — Treat rPPG as One Evidence Source, Not the Entire Classifier

## 1. What the Idea Is About

The first phase changes the role of rPPG in the overall system. Instead of treating rPPG as the sole or dominant source of evidence for deciding whether a video is real or fake, rPPG should be treated as one information branch among multiple available sources.

The bottleneck analysis shows that the existing videos are approximately 4.9 seconds long, contain limited cardiac cycles, and produce weak physiological signals. Therefore, expecting rPPG alone to provide reliable deepfake classification is unrealistic under the current dataset constraints.

The proposed approach is to retain the existing rPPG pipeline while creating a parallel visual-information branch. The two branches can then be combined before classification.

A conceptual structure is:

```text
                         INPUT VIDEO
                              |
              +---------------+---------------+
              |                               |
              v                               v
        VISUAL BRANCH                  PHYSIOLOGICAL BRANCH
              |                               |
       Visual features                  POS / CHROM rPPG
              |                               |
              |                         rPPG features
              |                               |
              +---------------+---------------+
                              |
                       Feature Fusion
                              |
                 +------------+------------+
                 |                         |
          Classical ML                Quantum ML
                 |                         |
                 +------------+------------+
                              |
                    Final Decision
```

The purpose is not to assume that visual features will definitely solve the problem. The purpose is to determine whether physiological information provides complementary information when combined with visual evidence.

## 2. How to Implement It

### Step 1 — Preserve the Existing rPPG Pipeline

Keep the current POS and CHROM extraction methods unchanged initially. Store the existing rPPG features for each video.

Potential existing features include:

- Dominant frequency
- Spectral entropy
- SNR
- Inter-ROI correlation
- Phase lag
- Temporal stability features
- Other deterministic rPPG features already produced by the pipeline

### Step 2 — Create a Visual Feature Branch

Extract compact visual representations from the same frames already present in the dataset.

The visual branch should use the same face detection/alignment stage where possible so that the experiment does not introduce an unrelated preprocessing pipeline.

### Step 3 — Create Separate Feature Tables

Maintain separate representations:

```text
video_id | rPPG features...
video_id | visual features...
video_id | label
```

Then join them using the existing video identifier.

### Step 4 — Create a Fused Representation

Construct:

```text
Fused Features =
rPPG features + visual features
```

Normalize/standardize features where required before classification.

### Step 5 — Compare Feature Configurations

Run at least:

1. rPPG only
2. Visual only
3. rPPG + visual

Then pass comparable feature sets through classical and quantum classifiers.

## 3. Things to Keep in Mind

### Do Not Claim That Visual Features Are Automatically Better

The purpose of this phase is experimental comparison. The existing bottleneck analysis does not prove that a visual branch will improve performance.

### Prevent Data Leakage

All normalization, feature selection, and dimensionality reduction that learns parameters must be fitted inside the training fold and then applied to the validation fold.

### Keep the Same Splitting Strategy

The existing analysis uses subject-grouped StratifiedKFold. Preserve subject-level separation so that frames or clips belonging to the same person do not leak between training and validation.

### Maintain Feature Provenance

Record which features came from:

- Visual analysis
- POS
- CHROM
- Signal quality
- Temporal analysis

This will be essential for the ablation study later.

---

# Phase 2 — Quality-Weighted rPPG Instead of Aggressive Frame Rejection

## 1. What the Idea Is About

The current bottleneck analysis reports that quality rejection can exceed 90% of frames for many fake videos. This means that a large amount of the available temporal information may be discarded.

Instead of using a strict binary decision:

```text
Good frame → use
Bad frame → discard
```

this phase introduces a continuous quality weight.

For example:

```text
High-quality frame   → high weight
Medium-quality frame → medium weight
Low-quality frame    → low weight
Extremely poor frame → near-zero weight
```

The objective is to make better use of the existing frames without pretending that poor-quality frames contain reliable physiological information.

## 2. How to Implement It

### Step 1 — Define a Frame Quality Score

Construct a quality score from information already available in the preprocessing pipeline.

Possible components include:

- Face size
- Landmark stability
- ROI size
- ROI validity
- Skin-pixel availability where applicable
- Signal amplitude
- Noise level
- Existing rPPG quality indicators

Do not invent arbitrary physiological meaning for a quality metric.

### Step 2 — Convert Quality to Weights

Normalize the quality score to a consistent range such as:

\[
0 \leq w_i \leq 1
\]

where \(w_i\) represents the contribution of frame \(i\).

### Step 3 — Use Weighted Aggregation

For a feature or signal \(x_i\):

\[
X_{weighted} =
\frac{\sum_{i=1}^{N} w_i x_i}
{\sum_{i=1}^{N} w_i}
\]

The exact aggregation should depend on the feature. Do not apply a weighted mean blindly to quantities where it is mathematically inappropriate.

### Step 4 — Compare Against the Existing Pipeline

Create two experimental configurations:

```text
Baseline:
quality rejection → rPPG → features

Proposed:
quality score → weighted rPPG/features
```

Compare their performance and signal statistics.

## 3. Things to Keep in Mind

### Do Not Treat Quality as Deepfake Evidence

A low-quality frame does not mean the frame is fake.

The quality score should describe the reliability of the measurement, not the class label.

### Avoid Weight Leakage

If quality thresholds or weighting parameters are optimized using the complete dataset, the validation results can become optimistic. Tune learned thresholds within training folds.

### Do Not Force Bad Frames Into the Signal

Quality weighting is not an argument for using every pixel regardless of reliability. Extremely corrupted frames can still be excluded.

### Report Frame Utilization

For every experiment, report:

- Percentage of accepted frames
- Average quality
- Number of usable frames
- Signal SNR

This makes the improvement measurable.

---

# Phase 3 — Short Temporal Window Analysis

## 1. What the Idea Is About

The existing clips are short, approximately 4.9 seconds, and the bottleneck analysis states that this duration provides only around 3–5 cardiac cycles.

Instead of treating the complete clip as one homogeneous signal, divide the same video into several overlapping short temporal windows.

For example:

```text
Full clip
|--------------------------------------------|

Window 1
|----------------|

Window 2
       |----------------|

Window 3
              |----------------|

Window 4
                     |----------------|
```

The objective is to measure whether the physiological estimates remain consistent across portions of the same video.

This does not create additional information. It reorganizes the existing temporal information.

## 2. How to Implement It

### Step 1 — Define Window Parameters

Choose:

- Window duration
- Window overlap
- Minimum usable frames

The parameters should be selected based on the existing frame rate and available frames.

### Step 2 — Extract rPPG for Each Window

For every window:

```text
Window → POS/CHROM → signal → features
```

Calculate the same relevant features used in the existing pipeline.

### Step 3 — Generate Window-Level Features

For each feature, calculate statistics such as:

\[
Mean
\]

\[
Standard\ Deviation
\]

\[
Maximum-Minimum
\]

\[
Coefficient\ of\ Variation
\]

where mathematically appropriate.

### Step 4 — Construct Video-Level Representation

For example:

```text
Video
 ├── Window 1 features
 ├── Window 2 features
 ├── Window 3 features
 └── Window 4 features
        ↓
Video-level aggregation
```

### Step 5 — Compare With Whole-Clip Features

Run:

```text
Whole-clip features
vs.
Window-based features
```

using the same cross-validation protocol.

## 3. Things to Keep in Mind

### Do Not Claim Reliable HRV From Very Short Windows

The purpose is consistency analysis, not clinical-grade heart-rate variability estimation.

### Avoid Overlapping-Window Leakage

Windows from the same original video must always remain in the same train/validation fold.

### Keep Window Counts Consistent

If some videos produce different numbers of usable windows, use a well-defined aggregation strategy rather than silently dropping videos.

### Record Failed Windows

The number of windows that fail quality checks is itself useful information and can become part of the quality representation.

---

# Phase 4 — Cross-ROI Physiological Consistency

## 1. What the Idea Is About

Instead of asking only whether an individual ROI contains a pulse-like signal, analyze whether different facial regions behave consistently.

For example:

```text
Forehead rPPG
      \
       \ 
        → Cross-ROI consistency
       /
      /
Cheek rPPG
```

The underlying idea is that physiological behavior should have some temporal relationship across multiple facial regions.

This phase should be treated as an experimental hypothesis. The current bottleneck analysis does not establish that cross-ROI consistency will reliably separate real and fake videos.

## 2. How to Implement It

### Step 1 — Extract Signals From Existing ROIs

Use the existing forehead, cheek, or other valid ROIs.

### Step 2 — Synchronize the Signals

Ensure that ROI signals correspond to the same frame indices and temporal windows.

### Step 3 — Calculate Pairwise Features

Potential features include:

- Pearson correlation
- Cross-correlation
- Phase difference
- Frequency agreement
- Spectral similarity
- Coherence where appropriately implemented

For two signals \(S_A\) and \(S_B\):

\[
\rho_{AB}=corr(S_A,S_B)
\]

### Step 4 — Aggregate Across ROI Pairs

If there are multiple ROIs:

```text
ROI A ↔ ROI B
ROI A ↔ ROI C
ROI B ↔ ROI C
```

Generate summary statistics across valid pairs.

### Step 5 — Add the Features to the Classifier

Compare:

```text
Existing rPPG features
vs.
Existing rPPG + cross-ROI features
```

## 3. Things to Keep in Mind

### ROI Quality Must Be Considered

A correlation between two noisy signals does not automatically represent physiological synchronization.

### Avoid Correlation Misinterpretation

High correlation does not necessarily mean a genuine cardiac signal exists.

### Handle Missing ROIs

Some videos may have invalid or very small ROIs. The implementation must define what happens when one or more ROIs are unavailable.

### Avoid Class-Based Feature Engineering

Do not select a cross-ROI metric because it happens to work better on the test fold. Feature selection must occur within training data.

---

# Phase 5 — Physiological Quality Score

## 1. What the Idea Is About

This phase separates two concepts that must not be confused:

\[
Weak\ rPPG \neq Fake
\]

A genuine video can have poor physiological signal because of compression, resolution, lighting, face size, motion, or ROI problems.

Therefore, the system should explicitly estimate the reliability of its physiological evidence.

The result is a Physiological Quality Score (PQS) that indicates whether the rPPG evidence is sufficiently reliable to influence the final decision.

## 2. How to Implement It

### Step 1 — Select Quality Indicators

Possible indicators include:

- rPPG SNR
- Number of usable frames
- ROI validity
- Cross-ROI consistency
- Frequency stability
- Signal amplitude
- Temporal consistency

### Step 2 — Normalize the Components

Bring heterogeneous measurements onto comparable scales.

For example:

\[
z_i = \frac{x_i-\mu_i}{\sigma_i}
\]

if standardization is appropriate.

### Step 3 — Construct the Score

A generic structure could be:

\[
PQS =
w_1Q_{SNR}
+w_2Q_{ROI}
+w_3Q_{temporal}
+w_4Q_{crossROI}
\]

The weights should be learned or justified through validation rather than arbitrarily selected.

### Step 4 — Use PQS in Decision Making

A possible structure is:

```text
High PQS
   ↓
Physiological evidence receives greater influence

Medium PQS
   ↓
Physiological evidence receives moderate influence

Low PQS
   ↓
Physiological evidence receives limited influence
```

### Step 5 — Validate PQS Independently

Study whether PQS actually correlates with measurable signal reliability.

## 3. Things to Keep in Mind

### PQS Is Not a Fake Probability

Do not interpret a low PQS as evidence that a video is fake.

### Keep Quality and Classification Separate

PQS should measure evidence reliability. The classifier should estimate the class.

### Avoid Circular Design

Do not define PQS using the final classification prediction itself.

### Report Its Distribution

Show PQS distributions across the dataset and analyze whether the score is concentrated near unreliable values.

---

# Phase 6 — Quantum/Classical Feature-Fusion Experiment

## 1. What the Idea Is About

The existing results show that the VQC and Logistic Regression performance is very close. Therefore, this phase should not be framed as an attempt to prove that quantum ML is inherently superior.

Instead, use the quantum model as one classifier in a controlled comparison.

The main question becomes:

> Can the same compact feature representation be effectively processed by both classical and hybrid quantum-classical classifiers?

## 2. How to Implement It

### Step 1 — Establish a Classical Baseline

Use an appropriate classical classifier such as Logistic Regression, consistent with the existing experiment.

### Step 2 — Establish the Quantum Baseline

Use the existing VQC pipeline with the same input representation.

### Step 3 — Keep the Feature Set Identical

For a fair comparison:

```text
Same data
Same features
Same train/validation split
Same preprocessing
Different classifier
```

### Step 4 — Compare Multiple Feature Sets

For example:

| Feature Set      | Classical | VQC |
| ---------------- | --------- | --- |
| rPPG             | ✓        | ✓  |
| rPPG + Quality   | ✓        | ✓  |
| rPPG + Cross-ROI | ✓        | ✓  |
| Fused Features   | ✓        | ✓  |

### Step 5 — Compare Multiple Metrics

Use:

- Balanced Accuracy
- ROC-AUC
- Sensitivity/Recall
- Specificity
- F1-score
- Confusion Matrix

## 3. Things to Keep in Mind

### Do Not Claim Quantum Advantage Without Evidence

A small difference is not sufficient to establish superiority.

### Keep the Comparison Fair

Do not give one model a richer feature set or different validation split.

### Consider Model Variance

Report mean and standard deviation across cross-validation folds.

### Avoid Overfitting the Quantum Circuit

The quantum model should not become so complex that it simply memorizes training patterns.

---

# Phase 7 — Ablation Study

## 1. What the Idea Is About

The ablation study determines which components of the proposed system actually contribute useful information.

This is especially important because your project contains multiple possible feature groups.

Instead of reporting only one final accuracy, demonstrate how performance changes when components are added or removed.

## 2. How to Implement It

Construct experiments such as:

```text
Experiment A → rPPG only

Experiment B → POS only

Experiment C → CHROM only

Experiment D → POS + CHROM

Experiment E → rPPG + Quality

Experiment F → rPPG + Cross-ROI

Experiment G → rPPG + Visual

Experiment H → rPPG + Visual + Quality

Experiment I → Full proposed representation
```

Keep the classifier and validation protocol controlled.

### Create an Experimental Table

| Experiment | Feature Groups          | Balanced Accuracy | ROC-AUC | Recall | Specificity |
| ---------- | ----------------------- | ----------------: | ------: | -----: | ----------: |
| A          | rPPG                    |                   |         |        |             |
| B          | POS                     |                   |         |        |             |
| C          | CHROM                   |                   |         |        |             |
| D          | POS + CHROM             |                   |         |        |             |
| E          | rPPG + Quality          |                   |         |        |             |
| F          | rPPG + Cross-ROI        |                   |         |        |             |
| G          | rPPG + Visual           |                   |         |        |             |
| H          | rPPG + Visual + Quality |                   |         |        |             |
| I          | Full                    |                   |         |        |             |

## 3. Things to Keep in Mind

### Do Not Cherry-Pick Results

Predefine the experiments before inspecting final validation performance.

### Keep the Dataset Fixed

Every experiment should use the same underlying videos unless there is a clearly documented reason for a subset analysis.

### Preserve Grouped Validation

Subject-level grouping must remain consistent.

### Interpret Negative Results Properly

If adding a feature group does not improve performance, that is still a valid experimental finding.

---

# Phase 8 — Ensemble Classification

## 1. What the Idea Is About

This phase combines outputs from multiple models instead of relying on a single classifier.

The existing analysis shows a strong specificity/recall tradeoff for the VQC. An ensemble can investigate whether different evidence sources provide complementary predictions.

A conceptual formulation is:

\[
P_
==

\alpha P_{visual}
+
\beta P_{rPPG}
+
\gamma P_{quantum}
\]

where:

\[
\alpha+\beta+\gamma=1
\]

The weights should be determined through a training/validation procedure.

## 2. How to Implement It

### Step 1 — Train Individual Models

Train:

- Visual model
- rPPG model
- Quantum model

using controlled folds.

### Step 2 — Generate Validation Probabilities

For each validation sample, obtain model probabilities rather than only hard labels.

Example:

```text
Video 001:
Visual P(fake)  = 0.72
rPPG P(fake)    = 0.54
Quantum P(fake) = 0.63
```

### Step 3 — Combine Probabilities

Use a predefined or learned fusion strategy.

Possible approaches:

- Weighted averaging
- Logistic stacking
- Meta-classifier

### Step 4 — Evaluate the Ensemble

Compare it with each individual model.

## 3. Things to Keep in Mind

### Do Not Tune Weights on the Test Set

Ensemble weights must be selected using training/validation data.

### Probability Calibration Matters

A model's output may not represent a well-calibrated probability. Consider calibration when interpreting probability values.

### Do Not Assume More Models Means Better Performance

If models make similar errors, an ensemble may provide little benefit.

### Track Individual Contributions

Always report individual model performance alongside ensemble performance.

---

# Phase 9 — Three-State Decision With Insufficient Evidence

## 1. What the Idea Is About

The current binary system forces every video into:

```text
REAL
or
FAKE
```

However, the bottleneck analysis demonstrates that some videos contain insufficient physiological information.

Therefore, introduce a third operational outcome:

```text
REAL
FAKE
INSUFFICIENT EVIDENCE / REVIEW REQUIRED
```

This does not modify the dataset. It modifies the decision policy.

## 2. How to Implement It

### Step 1 — Define a Quality Threshold

Use the physiological quality score and/or broader evidence-quality indicators.

### Step 2 — Define the Decision Logic

Conceptually:

```text
Video
  |
  v
Quality Assessment
  |
  +---- Low evidence quality ----> Review Required
  |
  +---- Sufficient quality ------> Classifier
                                      |
                                +-----+-----+
                                |           |
                              Real        Fake
```

### Step 3 — Select Thresholds Properly

Thresholds should be determined using training/validation data.

### Step 4 — Report Coverage

For example, report:

- Percentage classified as Real
- Percentage classified as Fake
- Percentage sent to Review Required

### Step 5 — Analyze Selective Performance

Measure classification performance among samples for which the system considers the evidence sufficient.

## 3. Things to Keep in Mind

### Do Not Use “Insufficient Evidence” to Hide Poor Results

The third class must have a transparent, predefined rule.

### Report Both Coverage and Accuracy

A system could obtain excellent accuracy simply by rejecting almost every sample. Therefore:

\[
Coverage =
\frac{Number\ of\ automatically\ classified\ samples}
{Total\ samples}
\]

must be reported alongside performance.

### Keep Operational Meaning Clear

“Review Required” means insufficient evidence for an automatic decision, not “probably fake.”

---

# Phase 10 — Compression and Quality-Robustness Analysis

## 1. What the Idea Is About

Compression and low resolution are central bottlenecks in the current dataset. Instead of treating this only as a problem, analyze its effect systematically.

The purpose is to determine how signal quality changes with video quality and whether the proposed features remain useful under different levels of degradation.

The analysis should use the existing videos and their naturally occurring quality variation.

## 2. How to Implement It

### Step 1 — Define Quality Indicators

Possible indicators include:

- Face resolution
- ROI dimensions
- Number of usable frames
- Frame rejection percentage
- rPPG SNR
- Landmark stability
- Compression-related measurements already available

### Step 2 — Create Quality Groups

For example:

```text
Higher-quality samples
Medium-quality samples
Lower-quality samples
```

The boundaries should be defined quantitatively rather than subjectively.

### Step 3 — Evaluate rPPG Quality

For each group, calculate:

- Mean SNR
- Usable-frame percentage
- Feature stability

### Step 4 — Evaluate Classification Performance

Compare:

```text
Quality group → rPPG quality → classification performance
```

### Step 5 — Identify Failure Regions

Determine whether the system becomes unreliable below a particular quality range.

## 3. Things to Keep in Mind

### Do Not Invent Causal Relationships

If low-quality videos perform worse, that demonstrates association in the experiment. It does not automatically prove that compression alone caused the reduction.

### Avoid Repeated Testing on the Same Holdout

Keep a proper final evaluation set untouched until the experimental design is finalized.

### Define Quality Bins Before Evaluation

Do not select bins after seeing which bins produce the most convenient result.

### Report Sample Counts

Each quality group should include its number of videos so that performance is interpreted in context.

---

# Phase 11 — Separate Physiological Signal Quality From Deepfake Evidence

## 1. What the Idea Is About

This phase is the conceptual foundation connecting the previous phases.

A weak rPPG signal can result from:

- Low resolution
- Compression
- Short video duration
- Poor ROI quality
- Motion
- Lighting
- Landmark instability
- Other measurement limitations

Therefore:

\[
Weak\ rPPG \not\Rightarrow Fake
\]

The system should distinguish:

```text
How reliable is the physiological measurement?
```

from:

```text
What class does the video belong to?
```

This creates a cleaner two-stage decision structure.

## 2. How to Implement It

### Step 1 — Build a Signal Reliability Representation

Use the outputs developed in previous phases:

- PQS
- SNR
- usable-frame percentage
- temporal consistency
- cross-ROI consistency
- ROI validity

### Step 2 — Keep Reliability Features Separate

Do not simply merge everything into one unexplained number.

Maintain separate feature groups:

```text
Physiological Signal Features
        +
Signal Reliability Features
        +
Visual Features
```

### Step 3 — Train the Classification Model

Allow the classifier to learn how physiological evidence should be interpreted in the context of signal quality.

Conceptually:

\[
P(Fake \mid
rPPG,
Quality,
Visual)
\]

rather than:

\[
P(Fake \mid weak\ rPPG)
\]

### Step 4 — Analyze Failure Cases

For incorrectly classified videos, inspect:

- PQS
- SNR
- usable frames
- visual evidence
- classifier confidence

Determine whether errors are associated with unreliable physiological measurements.

### Step 5 — Produce Explainable Outputs

A final system output could contain:

```text
Prediction: Fake
Confidence: 0.71

Physiological Evidence:
Low reliability

Visual Evidence:
Moderate reliability

Decision:
Requires additional verification
```

The exact wording should match the implemented decision logic.

## 3. Things to Keep in Mind

### Do Not Treat Quality as a Label

Quality describes evidence reliability; it is not the ground-truth class.

### Do Not Overinterpret Confidence

A classifier confidence value is not automatically a calibrated probability.

### Preserve the Ground-Truth Labels

The existing labels remain the evaluation target.

### Document Failure Modes

A major research contribution can be identifying the conditions under which physiological deepfake detection becomes unreliable.

---

# Final Implementation Sequence

The 11 phases can be organized into a practical development sequence:

```text
PHASE 1
Multi-source evidence architecture
        ↓
PHASE 2
Quality-weighted rPPG
        ↓
PHASE 3
Short temporal windows
        ↓
PHASE 4
Cross-ROI consistency
        ↓
PHASE 5
Physiological Quality Score
        ↓
PHASE 6
Classical vs Quantum feature-fusion experiments
        ↓
PHASE 7
Ablation study
        ↓
PHASE 8
Ensemble classification
        ↓
PHASE 9
Three-state decision system
        ↓
PHASE 10
Compression/quality robustness analysis
        ↓
PHASE 11
Separate signal reliability from deepfake evidence
```

# Recommended Experimental DisciplineKeep the Dataset Fixed

Do not add external videos, synthetic videos, additional DFDC samples, or new physiological ground-truth data if the project requirement is to keep the dataset unchanged.

## Keep Subject-Level Splitting

Use subject-grouped validation consistently so that information from the same subject cannot leak across folds.

## Establish a Baseline First

Before modifying the pipeline, preserve the current baseline results so every phase can be compared against the original system.

## Change One Major Component at a Time

Do not implement all 11 phases simultaneously and then claim that the final improvement came from the complete architecture. Build incrementally.

## Record Every Experiment

Maintain an experiment log containing:

- Phase
- Feature configuration
- Preprocessing configuration
- Classifier
- Cross-validation strategy
- Balanced accuracy
- ROC-AUC
- Recall
- Specificity
- F1-score
- Standard deviation
- Number of usable samples
- Notes on failure cases

## Do Not Optimize for Accuracy Alone

Because your dataset is close to balanced, balanced accuracy remains useful, but it should not be the only metric. The current bottleneck analysis already demonstrates that specificity and recall can behave very differently, particularly for the VQC.

## Treat Negative Results as Results

If a proposed phase does not improve performance, document it. A rigorous demonstration that a particular modification does not overcome the physiological information bottleneck is more valuable than artificially tuning the system until one favorable number appears.

# Overall Objective

The purpose of these 11 phases is **not to pretend that the existing dataset contains more physiological information than it actually does**. The purpose is to systematically determine how much useful evidence can be extracted from the existing short, low-resolution videos and how that evidence can be combined, evaluated, and communicated responsibly.

The most important principle is:

> **Improve information utilization and experimental methodology without claiming that preprocessing or model changes can create physiological information that is absent from the source videos.**

This keeps the project technically honest while giving the existing dataset substantially more experimental value.
