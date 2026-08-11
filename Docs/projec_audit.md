
# Codebase Scan — Project Understanding

## 1. Overall Objective

### Project Purpose

I scanned the uploaded `Major_project-main.zip` and the important implementation files. The project is a **low-resolution deepfake/KYC verification system that uses physiological rPPG signals as the primary evidence and a hybrid quantum-classical model as the final classification layer**.

The intended architecture is:

```text
Input KYC Video
      ↓
Frame Sampling + Quality Assessment
      ↓
Face Detection / ROI Extraction
      ↓
rPPG Signal Extraction
      ↓
Physiological Feature Generation
      ↓
QAOA Feature Selection
      ↓
Hybrid Quantum-Classical VQC
      ↓
P(real)
      ↓
REAL / FAKE / UNCERTAIN
```

This matches the project direction you have been developing: **low-resolution deepfake detection using rPPG + quantum machine learning**, rather than relying purely on visual CNN-based artifacts.

---

# 2. Repository Structure

## Major Components

The actual working implementation is primarily under `WORKING/`:

```text
WORKING/
├── frame/
│   ├── app/
│   ├── weights/
│   └── Docs/
│
├── RPPG/
│   ├── rppg/
│   ├── rppg-pipeline/
│   └── dataset_features.csv
│
├── quantum/
│   ├── config.py
│   ├── data.py
│   ├── selection.py
│   ├── vqc.py
│   ├── evaluate.py
│   ├── baselines.py
│   └── run.py
│
└── run_pipeline.py
```

There is also substantial documentation, research papers, reports, and project material under `Docs/`.

---

# 3. Frame Layer

## Purpose

The `frame` module is responsible for preparing video/frame information before the physiological analysis.

### Main Responsibilities

It handles:

- Video ingestion.
- Frame sampling.
- Face detection.
- Face cropping.
- Blur assessment.
- Brightness assessment.
- Face-size checks.
- Quality filtering.
- Metadata generation.

The project uses a YOLO-based face detector for this stage.

The conceptual role is:

```text
Video
 ↓
Sample Frames
 ↓
Detect Face
 ↓
Assess Quality
 ↓
Accept / Reject Frame
```

### Important Observation

There is a **potential architectural duplication** here.

The main pipeline calls the frame stage, but the rPPG pipeline independently opens the original video and performs its own face/ROI processing using `FaceROIExtractor`.

That means the current implementation is not strictly:

```text
frame output → rPPG input
```

Instead, it is closer to:

```text
                  ┌→ Frame pipeline
Input Video ──────┤
                  └→ RPPG pipeline
```

The frame stage therefore currently provides quality/statistical information but does not appear to directly feed its processed frames into rPPG.

That is something worth addressing later if your goal is a genuinely integrated architecture.

---

# 4. rPPG Layer

## Core Purpose

This is the **most important feature-generation stage** in the project.

The rPPG pipeline extracts physiological information from facial regions and converts it into a fixed-length numerical representation.

The main implementation is:

```text
WORKING/RPPG/rppg/
```

### Main Modules

| File                     | Responsibility                    |
| ------------------------ | --------------------------------- |
| `pipeline.py`          | Main end-to-end rPPG processing   |
| `face_roi.py`          | Facial ROI detection/extraction   |
| `signal_extraction.py` | POS/CHROM pulse reconstruction    |
| `preprocessing.py`     | Signal cleaning/filtering         |
| `features.py`          | Physiological feature calculation |
| `model_utils.py`       | Model/helper functionality        |

---

# 5. rPPG Data Flow

## From Video to Features

The actual flow is approximately:

```text
Video
 ↓
Read frames
 ↓
Face detection
 ↓
Extract:
   ├── Left cheek
   ├── Right cheek
   └── Forehead
 ↓
Mean RGB traces
 ↓
POS / CHROM
 ↓
Pulse signals
 ↓
Filtering / cleaning
 ↓
Combined physiological signal
 ↓
Feature extraction
 ↓
8-dimensional rPPG feature vector
```

This is exactly the type of data that should feed the quantum layer.

---

# 6. The Actual rPPG Feature Vector

## Eight Features

The `RPPGFeatures` class produces **8 numerical features**:

```text
1. heart_rate_bpm
2. snr_db
3. prv_std_ms
4. spectral_entropy
5. mad
6. signal_quality_index
7. cheek_forehead_correlation
8. left_right_cheek_correlation
```

These represent physiological characteristics rather than visual image characteristics.

### Feature Meaning

| Feature                    | Meaning                                    |
| -------------------------- | ------------------------------------------ |
| Heart Rate                 | Dominant pulse frequency                   |
| SNR                        | Quality of physiological signal            |
| PRV                        | Pulse-rate variability                     |
| Spectral Entropy           | Distribution of spectral energy            |
| MAD                        | Waveform variation                         |
| SQI                        | Pulse-signal quality                       |
| Cheek-Forehead Correlation | Cross-region physiological synchronization |
| Left-Right Correlation     | Synchronization between cheeks             |

This is the **correct interface between rPPG and quantum ML**.

---

# 7. Quantum Layer

## Current Architecture

The quantum implementation consists mainly of:

```text
quantum/
├── data.py
├── selection.py
├── vqc.py
├── evaluate.py
├── baselines.py
├── config.py
└── run.py
```

Its intended architecture is:

```text
rPPG Features
      ↓
QAOA Feature Selection
      ↓
Selected Features
      ↓
Angle Embedding
      ↓
Variational Quantum Circuit
      ↓
Classical Neural Network Head
      ↓
P(real)
```

This is a legitimate hybrid quantum-classical design.

---

# 8. Important Finding — Synthetic Data Is Already Removed

## Current `quantum/data.py`

I checked the actual implementation, and there is **no synthetic-data generator in the current `quantum/data.py`**.

It explicitly reads:

```text
RPPG/dataset_features.csv
```

and converts that into the quantum training dataset.

The data flow is:

```text
RPPG dataset_features.csv
          ↓
quantum/data.py
          ↓
X, y
          ↓
train / validation / test split
          ↓
data.npz
```

So the repository appears to have **already undergone part of the change you were asking for**.

The comment in `data.py` explicitly describes it as:

> Real rPPG dataset for the quantum layer (no synthetic data).

That is important because we should **not blindly rewrite the quantum data layer again**.

---

# 9. Current Quantum Feature Contract

## `quantum/config.py`

The quantum layer defines the exact same eight feature names as the rPPG layer.

This is good architecture:

```text
RPPGFeatures.feature_names()
            ↓
      FEATURE_NAMES
            ↓
     quantum/config.py
```

The feature ordering is explicitly maintained.

That means the quantum layer knows exactly what each input dimension represents.

---

# 10. QAOA Layer

## Feature Selection

`quantum/selection.py` implements a QAOA-based feature-selection mechanism.

It calculates:

```text
Mutual Information
        +
Feature Correlation
        +
Redundancy Penalty
        +
Cardinality Penalty
        ↓
QAOA Optimization
        ↓
Selected Feature Subset
```

The configured target is:

```text
8 original features
        ↓
6 selected features
```

So the quantum model does not necessarily consume all eight rPPG features.

It first selects a subset.

---

# 11. VQC Layer

## Actual Model

`quantum/vqc.py` implements:

```text
Selected rPPG features
        ↓
AngleEmbedding
        ↓
StronglyEntanglingLayers
        ↓
Quantum expectation values
        ↓
Classical neural network
        ↓
Binary output
```

The quantum circuit uses PennyLane's:

```text
AngleEmbedding
StronglyEntanglingLayers
```

and produces one expectation value per qubit.

The classical head then processes those outputs.

This is the actual **hybrid quantum-classical classifier**.

---

# 12. Main Pipeline

## `run_pipeline.py`

This is the most important integration file.

Its intended flow is:

```text
[1] FRAMES
       ↓
[2] RPPG
       ↓
[3] QUANTUM
       ↓
Final Verdict
```

The rPPG stage returns:

```python
result.to_feature_vector()
```

which produces the actual 8-feature vector.

Then the quantum stage receives the feature dictionary:

```text
rPPG feature dictionary
        ↓
FEATURE_NAMES
        ↓
selected indices
        ↓
HybridModel
```

So conceptually, your requested architecture is **already implemented**.

---

# 13. But I Found a Serious Code Problem

## `run_pipeline.py` Contains an Apparent Bug

Inside `quantum_inference()`, the returned dictionary contains:

```python
"features": bridged,
```

but from the code scanned, `bridged` is not defined inside that function.

The function actually creates:

```python
full = np.asarray([features[name] for name in QUANTUM_FEATURE_NAMES], dtype=np.float32)
x = full[indices].reshape(1, -1)
```

but then returns `bridged`.

That is a **runtime-breaking error** unless another piece of code not shown in the relevant function defines it.

This is much more important than randomly removing code.

---

# 14. Another Important Architectural Issue

## Training and Inference Are Slightly Different Concepts

There are two separate data flows:

### Training

```text
dataset videos
      ↓
rPPG feature extraction
      ↓
dataset_features.csv
      ↓
quantum/data.py
      ↓
QAOA
      ↓
VQC training
```

### Real Inference

```text
Input video
      ↓
rPPG pipeline
      ↓
8 features
      ↓
QAOA-selected indices
      ↓
trained VQC
      ↓
prediction
```

This is actually the architecture you want.

The quantum layer **does not need to generate synthetic features**.

It needs to consume the real rPPG feature vector.

---

# 15. What Should NOT Be Removed

## Avoid Blind Refactoring

Do **not** remove these just because they look like separate components:

- `quantum/data.py`
- `quantum/selection.py`
- `quantum/vqc.py`
- `quantum/evaluate.py`
- `quantum/baselines.py`

They have distinct responsibilities.

Likewise, the eight rPPG feature calculations are not redundant simply because they all live in `features.py`.

---

# 16. What Actually Looks Worth Cleaning

## Priority 1 — Broken / Stale References

The `bridged` reference in `run_pipeline.py` needs investigation/fixing.

## Priority 2 — Duplicate Processing

The frame layer and rPPG layer appear to independently process the same video rather than sharing the frame-quality output.

That may be intentional, but it creates duplicated work.

## Priority 3 — Unused Variables

There are signs of unnecessary variables in places such as the rPPG pipeline and helper functions.

For example, `kept_idx` appears to be incremented but does not contribute to the final result.

That is a legitimate cleanup candidate.

## Priority 4 — Obsolete Documentation

Some documentation still describes older workflows or standalone components. The code and documentation should eventually agree exactly.

## Priority 5 — Multiple Classification Paths

The main pipeline currently has:

```text
rPPG → RandomForest cross-check
```

and:

```text
rPPG → QAOA → VQC
```

The RandomForest cross-check may be useful for research comparison, but if your final system is specifically supposed to be:

```text
rPPG → Quantum
```

then it should remain clearly classified as an **optional baseline/cross-check**, not part of the core decision pipeline.

---

# 17. My Understanding of Your Actual Project

## Final System

Your project is not fundamentally:

> "Take a video and classify it using a quantum model."

It is more accurately:

> **Extract physiological evidence from a low-resolution face video using rPPG, convert that physiological evidence into compact numerical features, select the most informative features using QAOA, and classify the resulting physiological representation using a hybrid quantum-classical model to determine whether the video is genuine or manipulated.**

That distinction matters.

The **rPPG layer is the feature-generation/physiological-analysis layer**.

The **quantum layer is the feature-selection/classification layer**.

The quantum layer should therefore **never invent its own input data during normal operation**.

---

# 18. Recommended Target Architecture

## Clean Final Pipeline

I recommend making the implementation conform strictly to this:

```text
                    INPUT VIDEO
                         │
                         ▼
              Frame Sampling / Quality
                         │
                         ▼
                 Face / ROI Detection
                         │
                         ▼
                  rPPG Signal Extraction
                         │
                         ▼
                Physiological Features
                         │
                  8-dimensional vector
                         │
                         ▼
                ┌───────────────────┐
                │   QAOA Selection  │
                └───────────────────┘
                         │
                    6 features
                         │
                         ▼
                ┌───────────────────┐
                │    Quantum VQC    │
                └───────────────────┘
                         │
                         ▼
                    P(real)
                         │
                         ▼
              REAL / FAKE / UNCERTAIN
```

### Training

```text
Training videos
      ↓
rPPG extraction
      ↓
dataset_features.csv
      ↓
QAOA feature selection
      ↓
VQC training
      ↓
Saved quantum model
```

### Inference

```text
New KYC video
      ↓
rPPG extraction
      ↓
actual 8-feature vector
      ↓
same QAOA feature indices
      ↓
trained VQC
      ↓
prediction
```

---

# 19. Bottom Line

## Current State

The project is **much closer to your desired architecture than I initially expected**. The synthetic-data problem appears to have already been addressed in the current quantum data implementation.

The major work now should **not** be "rewrite the quantum layer."

It should be:

1. **Verify the rPPG → quantum interface.**
2. **Fix the apparent `bridged` bug in `run_pipeline.py`.**
3. **Remove genuinely dead/redundant code.**
4. **Ensure training and inference use exactly the same feature ordering and preprocessing.**
5. **Keep QAOA selection consistent between training and inference.**
6. **Separate optional RandomForest baseline logic from the core quantum decision path.**
7. **Avoid unnecessary architectural changes.**

Most importantly, I would **not let an AI coding agent aggressively refactor this repository yet**. The architecture is already reasonably coherent, and a broad "remove redundant code" instruction could easily destroy the experimental pipeline. The correct next step is a **targeted redundancy audit followed by minimal fixes**, especially around the rPPG → QAOA → VQC data contract.
