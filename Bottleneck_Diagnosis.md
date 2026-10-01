# Honest Bottleneck Diagnosis — Deepfake Detection via rPPG + Hybrid Quantum ML

## 1. Root Cause: Insufficient Physiological Signal in Source Data (Impact: Critical)

**Evidence:**

- All DFDC clips are **~4.9 seconds** (148 frames @ 30fps) — confirmed by Phase 3 probe (0/3293 clips ≥8s)
- Frequency resolution: **~0.2 Hz native → ~7 BPM grid** even with zero-padding
- Per-feature |AUC−0.5| ≤ **0.0475** on 3,450 samples — far below the ~0.20 needed for 70%+ balanced accuracy
- Temporal stability features require ≥60 frames; most clips have <60 usable frames → degenerate (removed in P6)

**Why this kills classification:** rPPG needs cardiac cycles. 4.9s captures ~3-5 heartbeats — statistically insufficient to estimate HR, HRV, or morphology reliably. Deepfake generation artifacts don't create a stronger signal than the physiological noise floor.

---

## 2. rPPG Extraction Quality is Fundamentally Limited by Input (Impact: Critical)

**Evidence:**

- MediaPipe landmarks on 100-200px faces → ROIs of 10-50 pixels
- Skin mask (YCrCb) unreliable <200px — disabled for low-res frames (P7)
- Quality rejection rates: 90%+ frames rejected on many DFDC Fake videos
- Best SNR achieved: **~0.87 dB** (CHROM 30fps) — barely above noise

**Why this kills classification:** The "signal" extracted is dominated by compression artifacts, not blood volume pulse. No amount of feature engineering fixes SNR < 1 dB.

---

## 3. Labels May Be Misaligned with Physiological Ground Truth (Impact: High)

**Evidence:**

- DFDC Fake = GAN-synthesized faces; Real = original actors
- But: Real videos are also compressed, downsampled, re-encoded
- No ground-truth PPG exists for any clip
- Label convention flip (rPPG: 1=fake, Quantum: 1=real) adds confusion risk

**Why this hurts:** If "Real" clips have equally degraded rPPG (compression), the physiological difference between classes may be near-zero.

---

## 4. Quantum/Classical Model is a Symptom, Not Cause (Impact: Low)

**Evidence:**

- VQC CV balanced accuracy: 0.522 ± 0.012
- Best classical (Logistic Regression): 0.5315 ± 0.016
- **Gap: ~1% — within noise**
- VQC achieves highest specificity (0.888) but recall 0.109 — same ROC tradeoff as classical

**Conclusion:** Model choice is irrelevant. The feature ceiling is the bottleneck.

---

## 5. Class Imbalance is Not the Problem (Impact: None)

**Evidence:**

- Dataset: 1,887 real / 1,563 fake (54/46 split)
- Per-sample class weights, focal loss (α=0.45), balanced accuracy metric all used
- CV uses subject-grouped StratifiedKFold

---

## Ranked Fix Priority

| Priority    | Fix                                                              | Feasibility                  | Expected Impact                                                                                                          |
| ----------- | ---------------------------------------------------------------- | ---------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| **1** | **Acquire ≥10s KYC-style captures** (300+ frames @ 30fps) | Requires new data collection | **Only path to >70% balanced accuracy** — resolves spectral quantization, enables temporal features, improves SNR |
| **2** | **Higher resolution input** (≥480p faces)                 | Camera/hardware dependent    | Improves landmark accuracy, ROI pixel count, skin mask reliability                                                       |
| **3** | **Multi-modal sensing** (depth, thermal, NIR)              | New hardware                 | Bypasses visible-light rPPG limitations                                                                                  |
| **4** | **Synthetic data with known PPG** (controlled generation)  | Feasible but risky           | Only useful for pre-training; domain gap to real compression                                                             |

---

## What NOT to Waste Time On

- ❌ Hyperparameter tuning (QAOA, VQC, classical)
- ❌ Feature engineering (already at ceiling)
- ❌ Model architecture search (quantum vs classical)
- ❌ Threshold calibration (already optimal via Youden's J)
- ❌ More data from DFDC/FF++ (same 4.9s ceiling)

---

## Verdict

**The project cannot achieve reliable deepfake classification with the current data.** The 4.9-second, low-resolution, compressed DFDC clips contain insufficient physiological signal for rPPG to separate real from deepfake. **New data acquisition (≥10s, higher resolution) is the only fix that matters.**


#RPPG

The rPPG stage is purely mathematical algorithms — no training involved:
Signal extraction (signal_extraction.py:45-138): POS (Wang et al. 2017) and CHROM (de Haan & Jeanne 2013) are classical fixed-formula algorithms that project normalized RGB traces onto chrominance/orthogonal planes to recover pulse.
Feature computation (features.py): All 24 features are deterministic mathematical functions — Welch PSD, peak detection, spectral entropy, inter-ROI correlation, phase lag, SNR, etc. Zero learned parameters.
The only "model" in rPPG is an optional RandomForest cross-check (rppg-pipeline/train_classifier.py, model_utils.py) which the AGENTS.md notes is currently skipped — the final verdict comes exclusively from the quantum VQC stage.
The trained model is only in Stage 3 (quantum/): the hybrid VQC (hybrid_vqc.pt) trained via python -m quantum.pipeline --all.
