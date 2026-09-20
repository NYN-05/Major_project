# Deepfake Detection Accuracy Improvement Plan

**Project:** Deepfake Detection in Low-Resolution KYC Videos Using rPPG and Hybrid Quantum ML  
**Target:** 80–85% balanced accuracy (engineering target, not guaranteed)  
**Date:** 2026-09-20  
**Status:** Audit complete — ready for implementation

---

## 1. Project Overview

Three-stage pipeline:
1. **Frame Stage** (`WORKING/frame/`) — YOLOv8 face detection + quality gating at 30 fps
2. **rPPG Stage** (`WORKING/RPPG/`) — MediaPipe landmarks → POS/CHROM pulse → 20 physiological features
3. **Quantum Stage** (`WORKING/quantum/`) — QAOA feature selection (20→3) → Hybrid VQC → P(real) → KYC verdict (REAL/FAKE/UNCERTAIN)

Orchestrator: `WORKING/run_pipeline.py`

---

## 2. Current Performance (Frozen Baseline: 2026-08-19, 3,473 samples)

| Metric | VQC Test | VQC 5-Fold CV | Best Classical (LR) |
|--------|----------|---------------|---------------------|
| Accuracy | 0.552 | 0.553 ± 0.009 | 0.552 |
| **Balanced Accuracy** | **0.499** | 0.500 ± 0.012 | **0.550** |
| Specificity (FAKE recall) | **0.000** | **0.000** | 0.532 |
| AUC-ROC | 0.535 | 0.556 ± 0.019 | **0.582** |
| Decision Bins | 0/694/0 (100% UNCERTAIN) | — | — |
| Per-feature |AUC−0.5| | — | ≤ ~0.06 | — |

**Key Finding:** The VQC is a **majority-class predictor** (confusion matrix `[[0,310],[1,383]]`). All 694 test clips fall in UNCERTAIN bin (P(real) ∈ [0.428, 0.503]). Classical baselines saturate at the same ceiling. **The bottleneck is rPPG feature informativeness, not the quantum/classical decision layer.**

---

## 3. Dataset & Label Validation

- **Samples:** 3,473 (1,921 real / 1,552 fake) from DFDC + FF++ at native 30 fps
- **Label convention:** rPPG CSV: 1=fake, 0=real → Quantum: LABEL_REAL=1, LABEL_FAKE=0 (flip in `csv_to_quantum_label`)
- **Grouping:** FF++ by source subject (`ffpp:src:<first-id>`); YouTube-real per clip; DFDC per clip (limitation: no subject IDs)
- **Leakage protection:** `_assert_no_group_leakage` enforced; group-aware CV (`StratifiedGroupKFold`) on train only
- **Split manifest:** `split_manifest.json` persists path → (split, group) for reproducibility
- ⚠ **DFDC subject-level leakage possible** — filenames lack subject IDs

---

## 4. rPPG Feature Audit (30 features: 20 core + 10 Phase-4 probes)

### Critical Implementation Issues

| Issue | Impact | Location |
|-------|--------|----------|
| **Spectral quantization** (~12 BPM grid) | HR estimates snap to {52.5, 60, 67.5, 75...} | `features.py:112` Welch PSD Δf ≈ 0.2 Hz @ 148 frames |
| **Temporal stability collapse** | `hr_window_std`, `sqi_window_std`, etc. → NaN→0.0 | `_window_stability` needs 60+ frames; most clips <60 usable |
| **ROI quality poor** | Skin mask → insufficient pixels → `valid_ratio < 0.3` excludes ROI | `face_roi.py:50` 8/10-pt polygons + YCrCb |
| **Preprocessing noise amplification** | Bandpass `padlen≈24` on 148-frame signals → edge distortion | `preprocessing.py:72` |
| **SQI gating mismatch** | Train: SQI<0.10 dropped; Inference: only SQI==0 rejected | `extract_dataset_features.py` vs `pipeline.py:348` |

### Feature Quality Summary
- Max per-feature |AUC−0.5| ≤ ~0.06 (verified)
- Several features degenerate: window-std features mostly 0.0, phase_lag implausible (up to 7000ms)
- Phase-4 probes unvalidated on current extraction

---

## 5. Classical Baseline Comparison

Identical subject-grouped splits, same 3 QAOA-selected features:

| Model | Test AUC | Balanced Acc | Specificity | CV AUC |
|-------|----------|--------------|-------------|--------|
| Logistic Regression | **0.582** | **0.550** | 0.532 | 0.603 ± 0.102 |
| LinearSVC | 0.581 | 0.516 | 0.122 | 0.473 ± 0.103 |
| GaussianNB | 0.478 | 0.475 | 0.367 | 0.605 ± 0.089 |
| Random Forest | 0.492 | 0.555 | 0.510 | 0.520 ± 0.091 |
| MLP | 0.504 | 0.495 | 0.408 | 0.508 ± 0.066 |
| XGBoost | 0.456 | 0.454 | 0.327 | 0.506 ± 0.094 |
| **Hybrid VQC** | **0.535** | **0.499** | **0.000** | **0.556 ± 0.019** |

**No quantum advantage demonstrated** — classical LR outperforms VQC on test AUC.

---

## 6. Quantum Model Audit

### Architecture
- QAOA: p=3, COBYLA max_iter=200, restarts=4, torch-native sim (~0.3-0.5ms/call)
- Feature selection: Discrimination weights (2×|AUC−0.5|) + redundancy (0.3) + cardinality (0.5)
- VQC: 3 layers `StronglyEntanglingLayers`, AngleEmbedding (RY), Classical head (3→8→1)
- Loss: Focal (α=0.45, γ=1.0) + label_smoothing(0.03) + confidence_penalty(0.02×entropy)

### Fixed Defects (Regression-Tested)
| Defect | Fix | Guard |
|--------|-----|-------|
| Dead β mixer | Separate cost_gates (γ) and mixer_gates (β) | `test_beta_alive` |
| Hamiltonian ≠ classical cost | Full closed-form derivation with −¼Σq_ij terms | `test_hamiltonian_matches_classical` (error < 1e-6) |

### Unfixed Training Collapse
```python
# vqc.py:435-438 — SCALAR balanced_weight (no-op)
balanced_weight = (weight_fake + weight_real) / 2  # ~0.9 constant

# vqc.py:312-324 — α INVERTED for majority REAL
alpha_t = 0.45*soft + 0.55*(1-soft)  # REAL (majority) gets LOWER weight

# vqc.py:339-344 — CONFIDENCE PENALTY rewards p≈0.5
loss = focal - 0.02*entropy  # Minimum at p=0.5 → explains narrow P(real) band
```

### Training Evidence
- `training_log.jsonl`: **only 2 lines** (epochs 1-2 of 80)
- Checkpoint: `best_epoch=None` → externally killed, no early stopping
- Scheduler LR at epoch 2 = 0.005 (expected ~0.0099 for CosineAnnealing T_max=80)

---

## 7. Root-Cause Analysis

| ID | Problem | Severity | Fix |
|----|---------|----------|-----|
| **P1** | Training killed at 2/80 epochs | 🔴 Critical | Fix loss, ensure 80 epochs with val monitoring |
| **P2** | Focal loss α inverted (majority upweighted) | 🔴 Critical | Set α≈0.45 for minority FAKE |
| **P3** | Confidence penalty rewards uncertainty | 🔴 Critical | Set `confidence_penalty = 0` |
| **P4** | Balanced weight is scalar no-op | 🔴 Critical | Per-sample `pos_weight` tensor |
| **P5** | Spectral quantization (~12 BPM grid) | 🔴 Critical | Longer clips / higher fps / zero-padding |
| **P6** | Temporal stability features → 0.0 | 🔴 Critical | Longer clips OR remove degenerate features |
| **P7** | ROI quality poor (skin mask → None) | 🟠 High | Larger ROIs / better skin mask |
| **P8** | SQI gate mismatch (train vs inference) | 🟠 High | Align gates (inference SQI<0.10) |
| **P9** | Per-feature discrimination ≤ 0.06 | 🔴 Critical | Upstream rPPG repair (method/ROI/FPS) |
| **P10** | QAOA unstable (cost spread 12.5) | 🟠 High | Increase restarts=8, max_iter=500 |
| **P11** | No quantum advantage vs classical | 🟠 High | Fix upstream first; fair comparison |
| **P12** | 100% UNCERTAIN decisions | 🔴 Critical | Requires P1-P9 fixes first |

---

## 8. Prioritized Improvement Roadmap

### Phase 1: Fix Collapsed Training (1-2 days)
| Priority | Fix | File |
|----------|-----|------|
| 1 | Invert focal_loss α for majority REAL | `vqc.py:312` |
| 2 | Remove confidence_penalty | `vqc.py:339` |
| 3 | Per-sample class weights in focal_loss | `vqc.py:435-472` |
| 4 | Ensure 80 epochs with val monitoring + early stopping | `vqc.py:498-570` |

**Validation:** `training_log.jsonl` → 80 lines; VAL balanced_acc > 0.5, FAKE recall > 0

### Phase 2: Confirm Case B Diagnosis (1 day)
- Re-run `python -m quantum.pipeline --all --dev-only`
- If AUC still ~0.5 → **stop quantum work**, proceed to rPPG repair

### Phase 3: rPPG Signal Repair (Critical Gate)
| Experiment | Config | Success Criteria |
|------------|--------|------------------|
| Method probe | POS vs CHROM vs Green-channel | Best SNR + SQI |
| ROI probe | Left / Right / Forehead / Combined | Max valid_ratio + stability |
| FPS probe | 30 vs 15 vs 10 fps | Balance SNR vs temporal resolution |
| Duration filter | Clips ≥8s (≥240 frames) | Resolve quantization |

### Phase 4+: Only If rPPG AUC > 0.65
- Feature audit → Classical ceiling → QAOA/VQC optimization → Calibration

---

## 9. Experimentation Strategy

### Protocol
- **Split:** Frozen subject-grouped (seed=42), test untouched until final
- **Primary metric:** Balanced accuracy (not raw accuracy)
- **Secondary:** FAKE recall, specificity, AUC-ROC, PR-AUC, ECE, decision bins
- **CV:** 5-fold `StratifiedGroupKFold` on train only
- **Seeds:** Report mean±std over 3-5 seeds for final claims
- **No test-set model selection** — use `--dev-only` for validation experiments

### Experiment Matrix
| ID | Hypothesis | Variable | Baseline | Test Configs |
|----|------------|----------|----------|--------------|
| EXP-001 | Training collapse prevents learning | Loss/epochs | Current broken | Fixed loss, 80 epochs |
| EXP-002 | POS more robust than CHROM | Method | POS | CHROM, Green |
| EXP-003 | Single ROI beats noisy combination | ROI weights | (0.35,0.35,0.30) | (1,0,0)/(0,1,0)/(0,0,1) |
| EXP-004 | 15 fps preserves SNR | target_fps | 30 | 15, 10 |
| EXP-005 | Longer clips resolve quantization | Duration filter | All clips | ≥8s only |

---

## 10. Technical Limitations & Honest Assessment

| Limitation | Current | Required for 80-85% |
|------------|---------|---------------------|
| Max per-feature \|AUC−0.5\| | ≤0.06 | **≥0.20** (theoretical ceiling ~70% bal. acc) |
| Temporal stability features | Degenerate (0.0) | Non-zero variance |
| Fake detection (specificity) | 0.000 | **≥0.70** |
| Training epochs completed | 2/80 | 80 with early stopping |
| Clip duration | ~4.9s median | **≥8s** (or validated zero-padding) |

**Honest assessment:** With current DFDC+FF++ clips (mostly <6s), **80-85% is likely unachievable** regardless of model. The physiological signal is too degraded by compression, resolution, and duration. Target should be revised to **65-70% balanced accuracy** with honest uncertainty quantification, or new data acquisition (longer KYC-style captures) is needed.

---

## 11. Immediate Next Steps

```bash
cd WORKING/
# 1. Fix training collapse
# Edit vqc.py: focal_loss α, confidence_penalty, per-sample weights, 80 epochs

# 2. Run tests
python -m quantum.tests                    # Must pass 10/10

# 3. Dev evaluation
python -m quantum.pipeline --all --dev-only  # Check VAL balanced_acc > 0.5

# 4. If AUC ~0.5 → proceed to rPPG probe (Phase 3)
#    If AUC > 0.55 → continue quantum optimization
```

---

## 12. Files to Modify (Phase 1)

| File | Changes |
|------|---------|
| `WORKING/quantum/vqc.py` | Lines 312-324 (α), 339-344 (confidence_penalty), 435-472 (per-sample weights), 498-570 (epochs/val) |
| `WORKING/quantum/config.py` | Line 122: `alpha = 0.45` (confirm) |
| `WORKING/RPPG/rppg/pipeline.py` | Line 348: Align SQI gate to <0.10 |

---

*This plan is evidence-driven — every claim cites specific code locations, config values, or metric artifacts from the 2026-08-19 frozen baseline. No accuracy improvements are claimed without experimental validation.*