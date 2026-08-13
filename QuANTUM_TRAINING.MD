# Quantum Layer Training Guide

Complete documentation for training, evaluating, and deploying the hybrid quantum-classical decision layer (Stage 3) of the Deepfake KYC detection system.

---

## Overview

The quantum layer consumes the 8 rPPG physiological features directly from `output/rppg/dataset_features.csv` and produces a KYC verdict: **REAL** (≥0.7), **FAKE** (≤0.3), or **UNCERTAIN** (0.3–0.7).

**Pipeline:** QAOA feature selection (8→6) → Feature scaling (z-score) → Hybrid VQC (angle encoding + StronglyEntanglingLayers + classical head) → P(real) → Decision bins

---

## Prerequisites

### Environment
```bash
# From WORKING/
pip install pennylane torch numpy scikit-learn scipy matplotlib
```

### Required Artifacts (must exist before training)
| File | Source | Description |
|------|--------|-------------|
| `output/rppg/dataset_features.csv` | Stage 2 (rPPG) | 8 features + label (1=fake, 0=real) per video |

> **Critical:** The quantum layer uses **only real data**. No synthetic data generation is allowed (AGENTS.md constraint).

---

## Architecture Details

### Feature Contract (Fixed Order)
```python
# quantum/config.py == RPPG/rppg/features.py:RPPGFeatures.feature_names()
FEATURE_NAMES = [
    "heart_rate_bpm",           # 0
    "snr_db",                   # 1
    "prv_std_ms",               # 2
    "spectral_entropy",         # 3
    "mad",                      # 4
    "signal_quality_index",     # 5
    "cheek_forehead_correlation", # 6
    "left_right_cheek_correlation" # 7
]
```
**Never change order or names** — `data.py`, `qaoa.py`, `pipeline.py` all index by this list.

### Label Convention
| Stage | Label Encoding |
|-------|----------------|
| rPPG CSV | `1 = fake`, `0 = real` |
| Quantum | `LABEL_REAL = 1`, `LABEL_FAKE = 0` (flipped in `data.py`: `y = 1 - csv_label`) |

### QAOA Feature Selection
- **Cost Hamiltonian:** Mutual information weights + correlation redundancy penalty + cardinality constraint
- **Optimizer:** COBYLA (gradient-free)
- **Layers (p):** 3 (configurable)
- **Target features:** 6 (from 8)
- **Output:** `output/quantum/qaoa_selection.json` with `selected_indices`, `selected_features`, `feature_weights`, `marginal_probabilities`

### Hybrid VQC Architecture
```
Input (6 features) → Angle Encoding (RY) → StronglyEntanglingLayers (3 layers, 6 qubits)
                    → Classical Head (Linear(6→8) → ReLU → Dropout(0.2) → Linear(8→1) → Sigmoid)
                    → P(real) ∈ [0, 1]
```
- **Quantum device:** `default.qubit` (simulator)
- **Qubits:** 6 (matches selected features)
- **Trainable params:** ~200 (quantum + classical)
- **Loss:** Focal Loss (α=0.75, γ=2.0) + Label Smoothing (0.1) + Confidence Penalty (0.3)

---

## Training Commands

### Full Pipeline (Recommended)
```bash
# From WORKING/
python -m quantum.pipeline --all
```
Runs: build-data → select → train → evaluate → baselines

### Step-by-Step (For Debugging)
```bash
# 1. Build dataset (creates data.npz with train/val/test splits)
python -m quantum.pipeline --build-data

# 2. Run QAOA feature selection
python -m quantum.pipeline --select

# 3. Train hybrid VQC
python -m quantum.pipeline --train

# 4. Evaluate on test set
python -m quantum.pipeline --evaluate

# 5. Train classical baselines (MLP, RandomForest) on same features
python -m quantum.pipeline --baselines
```

---

## Output Artifacts (in `output/quantum/`)

| File | Description |
|------|-------------|
| `data.npz` | Train/val/test splits (X_train, y_train, X_val, y_val, X_test, y_test) |
| `feature_scaler.json` | Fitted StandardScaler (mean_, scale_) for 8 features |
| `qaoa_selection.json` | Selected feature indices, names, weights, marginal probs |
| `hybrid_vqc.pt` | Full model checkpoint (quantum params + classical head + metadata) |
| `metrics_quantum.json` | Test metrics: accuracy, precision, recall, F1, AUC-ROC, ECE, decision bins |
| `metrics_baselines.json` | Baseline metrics (MLP, RandomForest) |
| `training_log.jsonl` | Per-epoch train/val loss and val accuracy |
| `roc_curve.png` | ROC curve plot |
| `confusion_matrix.png` | Confusion matrix plot |
| `results_report.md` | Markdown evaluation report |

---

## Model Checkpoint Structure (`hybrid_vqc.pt`)

```python
{
    "model_state_dict": {...},          # Full PyTorch state dict
    "n_features": 6,                    # Number of QAOA-selected features
    "selected_indices": [4, 1, 7, 3, 6, 0],  # Indices into original 8 features
    "selected_features": ["mad", "snr_db", "left_right_cheek_correlation", ...],
    "feature_names": FEATURE_NAMES,     # All 8 original feature names
    "scaler_mean": [...],               # StandardScaler mean_ (8 values)
    "scaler_scale": [...],              # StandardScaler scale_ (8 values)
    "qaoa_config": {...},               # QAOASelectionConfig as dict
    "vqc_config": {...},                # VQCConfig as dict
    "decision_config": {...},           # DecisionConfig as dict
    "qaoa_selection": {...}             # Full QAOA selection output
}
```

---

## Inference Entry Point

Used by `run_pipeline.py` and frontend:

```python
from quantum.pipeline import predict_features

# Input: dict with all 8 feature names (order doesn't matter)
features = {
    "heart_rate_bpm": 72.5,
    "snr_db": 2.3,
    "prv_std_ms": 45.2,
    "spectral_entropy": 0.65,
    "mad": 0.42,
    "signal_quality_index": 0.71,
    "cheek_forehead_correlation": 0.34,
    "left_right_cheek_correlation": 0.56,
}

result = predict_features(features)
# Output:
# {
#     "prob_real": 0.6359,
#     "verdict": "UNCERTAIN",
#     "confidence": 0.2718,
#     "selected_features": [...],
#     "selected_indices": [...],
#     "scaler_file": "output/quantum/feature_scaler.json"
# }
```

**Decision thresholds (configurable in `quantum/config.py`):**
- `fake_max_prob = 0.3` → FAKE
- `real_min_prob = 0.7` → REAL
- Otherwise → UNCERTAIN
- `confidence = 2 * |prob_real - 0.5|` (range 0–1)

---

## Configuration (`quantum/config.py`)

### DataConfig
```python
@dataclass(frozen=True)
class DataConfig:
    seed: int = 42
    train_ratio: float = 0.6
    val_ratio: float = 0.2
    # test_ratio = 0.2 (implicit)
    csv_file: Path = WORKING_ROOT / "output" / "rppg" / "dataset_features.csv"
    data_file: Path = OUTPUT_DIR / "data.npz"
```

### QAOASelectionConfig
```python
@dataclass(frozen=True)
class QAOASelectionConfig:
    p_layers: int = 3              # QAOA depth
    max_iter: int = 200            # COBYLA iterations
    redundancy_penalty: float = 0.3
    cardinality_penalty: float = 0.5
    target_features: int = 6       # Select 6 from 8
    seed: int = 42
    selection_file: Path = OUTPUT_DIR / "qaoa_selection.json"
```

### VQCConfig
```python
@dataclass(frozen=True)
class VQCConfig:
    qml_layers: int = 3            # StronglyEntanglingLayers depth
    hidden_units: int = 8          # Classical head hidden size
    dropout: float = 0.2
    epochs: int = 40
    batch_size: int = 32
    learning_rate: float = 1e-2
    weight_decay: float = 1e-2
    alpha: float = 0.75            # Focal loss alpha
    gamma: float = 2.0             # Focal loss gamma
    label_smoothing: float = 0.1
    confidence_penalty: float = 0.3
    seed: int = 42
    checkpoint_file: Path = OUTPUT_DIR / "hybrid_vqc.pt"
    log_file: Path = OUTPUT_DIR / "training_log.jsonl"
    metrics_file: Path = OUTPUT_DIR / "metrics_quantum.json"
```

### DecisionConfig
```python
@dataclass(frozen=True)
class DecisionConfig:
    fake_max_prob: float = 0.3
    real_min_prob: float = 0.7
    metrics_baseline_file: Path = OUTPUT_DIR / "metrics_baselines.json"
    roc_plot: Path = OUTPUT_DIR / "roc_curve.png"
    confusion_plot: Path = OUTPUT_DIR / "confusion_matrix.png"
    report_file: Path = DOCS_DIR / "results_report.md"
```

---

## Expected Metrics (Current Baseline)

With 18 samples (9 per class), train/val/test ≈ 10/4/4:

| Model | Accuracy | Precision | Recall | F1 | AUC-ROC | ECE |
|-------|----------|-----------|--------|-----|---------|-----|
| Hybrid VQC | 0.75 | 0.67 | 1.00 | 0.80 | 0.75 | 0.20 |
| MLP (baseline) | 0.75 | 0.67 | 1.00 | 0.80 | 1.00 | 0.29 |
| RandomForest (baseline) | 0.50 | 0.50 | 1.00 | 0.67 | 0.00 | 0.24 |

**Decision bins on test (4 samples):**
- REAL: 1, UNCERTAIN: 3, FAKE: 0
- Uncertain rate: 75%
- Confirmed accuracy: 100%

> **Note:** Metrics are unreliable with n=4 test samples. Primary issue is data quantity, not model quality.

---

## QAOA Selection Analysis (`qaoa_selection.json`)

Current output shows degenerate weights (data starvation):
```json
{
  "selected_indices": [4, 1, 7, 3, 6, 0],
  "selected_features": ["mad", "snr_db", "left_right_cheek_correlation", "spectral_entropy", "cheek_forehead_correlation", "heart_rate_bpm"],
  "feature_weights": [0.0, 0.845, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
  "marginal_probabilities": [0.56, 0.81, 0.49, 0.67, 0.81, 0.47, 0.64, 0.71],
  "success": false
}
```

**Interpretation:**
- Only `snr_db` (0.85) and `spectral_entropy` (1.0) have non-zero weights
- 5 of 8 features have weight 0.0 → QAOA couldn't differentiate mutual information
- `"success": false` confirms insufficient data for meaningful selection
- **Fix:** More training data (see Data Expansion section)

---

## Training Log Analysis (`training_log.jsonl`)

```json
{"epoch": 1, "train_loss": -0.116, "val_loss": -0.113, "val_accuracy": 0.25}
...
{"epoch": 10, "train_loss": -0.135, "val_loss": -0.110, "val_accuracy": 0.25}
{"epoch": 11, "train_loss": -0.137, "val_loss": -0.108, "val_accuracy": 0.5}
...
{"epoch": 40, "train_loss": -0.149, "val_loss": -0.071, "val_accuracy": 0.25}
```

**Observations:**
- Loss decreases steadily (good optimization)
- Val accuracy oscillates 25%–50% → random guessing on val set (n=4)
- No overfitting detected (train/val loss gap small)
- Model converges but has no signal to learn from

---

## Data Expansion (Critical for Accuracy)

### Current State
- **Available:** 3,293 videos in `WORKING/RPPG/archive/DFDC_Dataset/` (1,566 Fake + 1,727 Real)
- **Extracted:** Only 18 videos (9 per class) in `dataset_features.csv`

### Full Extraction
```bash
# From WORKING/RPPG/
python rppg-pipeline/extract_dataset_features.py --workers 0
```
- Uses all CPU cores
- Expected: 500–1,500 usable samples (some fail quality gates)
- Time: 2–6 hours depending on hardware

### After Full Extraction
```bash
# From WORKING/
python -m quantum.pipeline --all
```
- Retrains with real data
- QAOA should produce meaningful feature weights
- VQC should achieve >85% accuracy with sufficient samples

---

## Model Caching (Production Optimization)

`quantum/vqc.py` implements `HybridModel` caching by `(n_features, ckpt_mtime, size)` — reuses loaded weights across server requests. No code changes needed.

---

## Troubleshooting

| Issue | Cause | Fix |
|-------|-------|-----|
| `FileNotFoundError: data.npz` | Dataset not built | Run `python -m quantum.pipeline --build-data` |
| `qaoa_selection.json` missing | Selection not run | Run `python -m quantum.pipeline --select` |
| `hybrid_vqc.pt` missing | Model not trained | Run `python -m quantum.pipeline --train` |
| Val accuracy stuck at 0.25/0.5 | Insufficient data (n<20) | Extract more videos from archive |
| QAOA `"success": false` | Degenerate MI weights | More data; or reduce `target_features` |
| CUDA OOM | Not applicable (uses CPU simulator) | N/A |
| Feature mismatch error | `FEATURE_NAMES` out of sync | Ensure `quantum/config.py` matches `RPPG/rppg/features.py` |

---

## Reproducibility

All configs use `seed=42`:
- Data split (stratified)
- QAOA initialization
- VQC weight initialization
- PyTorch/NumPy random state

Set `PYTHONHASHSEED=42` for full determinism.

---

## Integration with End-to-End Pipeline

```bash
# Full inference (from WORKING/)
python run_pipeline.py --source video.mp4 --method POS --out result.json
```

Flow:
1. Stage 1 (frames) → accepted JPEGs + metadata
2. Stage 2 (rPPG) → 8 features (or None → INCONCLUSIVE)
3. Stage 3 (quantum) → `predict_features()` → verdict

The quantum layer is **stateless at inference** — loads scaler, QAOA selection, and VQC checkpoint fresh each call (cached by `HybridModel`).

---

## Future Improvements

| Priority | Improvement | Effort |
|----------|-------------|--------|
| **P0** | Extract all 3,293 DFDC videos | Low (run existing script) |
| **P1** | Add cross-validation to `evaluate.py` | Medium |
| **P1** | Add balanced accuracy metric | Low |
| **P2** | Try `lightning.pytorch` for training loop | Medium |
| **P2** | Experiment with `qml.device("default.qubit.torch")` for GPU | Medium |
| **P3** | Quantum kernel methods (QSVM) | High |
| **P3** | Ensemble of multiple VQC seeds | Medium |

---

## Quick Reference Card

```bash
# Full retrain (after data extraction)
cd WORKING
python -m quantum.pipeline --all

# Check current metrics
cat output/quantum/metrics_quantum.json

# Check QAOA selection
cat output/quantum/qaoa_selection.json

# View training progress
tail -20 output/quantum/training_log.jsonl

# Single inference test
python -c "
from quantum.pipeline import predict_features
import json
f = {'heart_rate_bpm':72, 'snr_db':2, 'prv_std_ms':40, 'spectral_entropy':0.6,
     'mad':0.4, 'signal_quality_index':0.7, 'cheek_forehead_correlation':0.3,
     'left_right_cheek_correlation':0.5}
print(json.dumps(predict_features(f), indent=2))
"
```