# Immediate Fix Plan: Deepfake Detection Failure

**Date:** 2026-09-02
**Status:** Plan ready — awaiting user confirmation before implementation

---

## Current State (Verified Against Frozen Artifacts)

| Metric | Value | Meaning |
|---|---|---|
| Quantum test AUC | **0.485** (below chance) | Ranking inverted |
| Balanced Accuracy | 0.494 | Chance level |
| Confusion Matrix | `[[302,8],[378,6]]` | Collapsed — seed decides direction |
| CV Folds (5) | specificity 0.0 in ALL folds | Opposite collapse (always REAL) |
| Decision bins | **0/694/0** (100% UNCERTAIN) | Prob range [0.428, 0.503] — no separation |
| Best Classical LR | AUC **0.582** | Ceiling < 0.6 → upstream evidence weak |
| Training epochs | **2/80** completed | `training_log.jsonl:1-2` — checkpoint is under-fit |
| Per-feature signal | `|AUC-0.5| ≤ 0.06` | 20 rPPG features carry almost no class signal |

**Root cause is NOT a label bug** (`data.py:44-62 csv_to_quantum_label=1-csv_label` verified by `quantum/tests.py`).

---

## Root Cause Stack

### 1. Upstream rPPG Bottleneck (Primary, Critical)

Short clips (~4.9s / 148 frames @30fps) cause:
- **Spectral quantization:** `features.py:112` Welch PSD Δf ≈ 0.2Hz ≈ 12 BPM grid; `hr_half_diff:236-243` on ~73 points → 24 BPM quantization
- **Temporal stability degenerate:** `pipeline.py:92` `min_usable_frames=48`, `_window_stability:287` needs 60+ frames for 2 windows → `hr_window_std`, `sqi_window_std`, `entropy_window_std`, `max_hr_deviation` all fallback to NaN→0.0
- **ROI quality poor:** `face_roi.py:50` 8/10pt polygons + YCrCb skin mask → `mean_rgb:354` often `None`; `pipeline.py:402` valid_ratio < 0.3 excluded; linear interpolation over gaps = synthetic pulse
- **Preprocessing noise amplification:** `preprocessing.py:72` bandpass padlen ≈ 24 on short segments → phase distortion; z-score normalizes low-SNR to same scale as clean pulses
- **SQI gating mismatch:** train excludes SQI < 0.10, inference only excludes SQI == 0 → distribution shift

**Classical all-model sweep (0.513–0.582 AUC) proves quantum cannot rescue missing signal.**

### 2. Training Collapse + Loss Bug (Critical)

- **`focal_loss` alpha inverted** (`vqc.py:312-324`): `alpha_t = 0.75*soft + 0.25*(1-soft)` → majority REAL (1152/2083) gets 0.74 weight, FAKE gets 0.26 (backwards)
- **Confidence penalty rewards uncertainty:** `loss = focal - 0.02*entropy` → minimum at p=0.5, directly explains 0.428–0.502 narrow band
- **Balanced weight is a no-op:** `vqc.py:435-438` `balanced_weight ≈ 0.904` scalar × mean = constant LR scaling, no per-sample reweighting
- **Training killed after 2/80 epochs:** `training_log.jsonl:1-2` only 2 lines, `best_epoch=None` in checkpoint → no early stopping fired, external kill
- **Scheduler stepped wrong:** epoch 2 lr 0.005 not expected ~0.0099 for CosineAnnealing T_max=80

### 3. QAOA Instability (High, Not Bottleneck)

- Costs `[11.72, -0.09, -0.76, 9.83]` spread 12.5, `success:false`
- `p=3 max_iter=200 restarts=4` insufficient for 2^20 landscape
- `target_features=3` unjustified without expanded pool
- Overlaps 2/3 with classical greedy → QAOA adds no value yet

---

## Proposed Severity-Ordered Plan

### Phase 0 — Re-freeze Honest Baseline (1 day)

- Re-run `python -m quantum.tests` + `python -m quantum.pipeline --all --dev-only` from `WORKING/`
- Verify label/probability mapping reproduces current results
- Document split statistics in `Scrape/split_audit.json`
- Verify no group leakage: `group` never crosses splits (`data.py:273`)
- **Gate:** tests pass, no leakage, baseline numbers match

### Phase 1 — Fix Collapsed Training (Critical)

- Fix `vqc.py:435` balanced Weight → per-sample `pos_weight` tensor
- Set `VQCConfig.alpha ≈ 0.45` (invert for majority REAL)
- Set `confidence_penalty = 0` (remove uncertainty reward)
- Test `gamma = 1` (reduced focal flattening), `label_smoothing 0.0/0.03` ablated one at a time
- Ensure `train_vqc` receives `X_val/y_val`, logs `val_loss/val_accuracy` each epoch
- Enforce full 80 epochs or `patience=12` early-stop restoring `best_state`
- Verify `training_log.jsonl` grows to ~80 lines, `hybrid_vqc.pt` `best_epoch not None`
- **Gate:** non-zero FAKE recall and `balanced_accuracy > 0.5` on VAL across 3 seeds

### Phase 2 — Confirm Case B Diagnosis

- Keep thresholds `fake_max=0.3, real_min=0.7` (`DecisionConfig:128`)
- Plot score histograms, ROC/PR curves, ECE (`evaluation.py:200`, `evaluation.py:68`)
- Document: **Case B** = no separation, not threshold problem
- **Gate:** written diagnosis "upstream rPPG" vs "threshold"

### Phase 3 — Repair rPPG Signal (Critical Gate)

- Run `extract_dataset_features.py` variations on same grouped split:
  - **Method:** POS vs CHROM vs green-channel
  - **ROI:** left cheek / right cheek / forehead / combined
  - **FPS:** native 30 (dataset rate)
- Log per config: `valid_ratio, usable/total, SNR, SQI, HR plausibility, inter-ROI correlation`
- Keep classifier fixed (logistic regression on 20 features) to isolate signal
- Store probes in `Scrape/` (gitignored)
- **Gate:** pick extraction config with best signal validity + stability + VAL AUC

### Phase 4 — Temporal + Feature Audit (High)

- Window experiments: `2s / 3s / 4s / full`, overlapping vs non-overlapping
- Aggregation: median / quality-weighted mean
- Per-feature audit: ROC-AUC, PR-AUC, missing rate, variance, correlation matrix
- Produce `Scrape/feature_audit.csv` classifying each feature as strong / weak-but-complementary / redundant / unstable / invalid
- **Gate:** if max `|AUC - 0.5|` stays ≤ 0.08, proceed to Phase 5

### Phase 5 — Add Complementary Evidence (High, Modular)

Only if rPPG alone remains insufficient:
- **Branch B — Behavioral:** EAR, blink count/duration, MAR, head pose (pitch/yaw/roll, angular velocity) using MediaPipe landmarks already in `face_roi.py:109`
- **Branch C — Low-res forensic:** RGB mean/var, channel ratios, Laplacian variance, edge density
- Ablate: rPPG only / behavior only / rendering only / combinations
- Keep `FEATURE_NAMES` contract in sync (`config.py ↔ features.py`, `tests.py:test_feature_contract_sync`)
- **Gate:** keep branch only if VAL `ΔAUC > 0.03` and `ΔFAKE recall > 0`

### Phase 6 — Re-establish Classical Ceiling → QAOA → VQC (Medium-High)

- With clean feature pool, run `run_baselines` (LR / LinSVC / GNB / RF / XGB / MLP) ranked by FAKE recall → balanced_acc → PR-AUC
- QAOA sweep: `_discrimination_weights` vs MI, cardinality `k = 3/5/7/10`, `restarts 8`, `max_iter 500`, report Jaccard / selection frequency
- VQC architecture sweep (one family at a time): `qml_layers 1-4`, `hidden 8→16`, RY vs RY/RZ encoding
- **Gate:** quantum vs best classical on same features/split — do not claim advantage if not

### Phase 7 — Calibration & Operating Point (Medium)

- Freeze classifier
- Threshold sweep: coverage vs false-acceptance (FAKE→REAL)
- Temperature / isotonic calibration → ECE / Brier
- **Gate:** non-zero automated coverage at controlled false-acceptance, not 100% UNCERTAIN

### Phase 8 — Statistical Stability (Medium)

- 3–5 repeated grouped validation runs
- Report mean ± std for: balanced accuracy, FAKE recall, specificity, ROC-AUC, PR-AUC, ECE
- **Gate:** improvement must be repeatable and directionally consistent

### Phase 9 — Training/Inference Equivalence (Medium)

- For one saved feature vector: offline preprocessing == inference preprocessing
- For one saved probability: evaluation probability == deployment probability (within tolerance)
- Verify `run_pipeline.py:13` stale docs ("10-feature vector" → actually 20)
- Fix `run_pipeline.py:201` rppg_classifier guard: `n_features != 10` → `n_features != 20`
- **Gate:** training path == inference path

---

## Constraints (Do Not Break)

- **No synthetic data** — only real rPPG features from `output/rppg/dataset_features.csv`
- **Feature contract** — `FEATURE_NAMES` in `config.py` and `features.py` must stay identical in name, order, semantics
- **Label conventions differ per stage** — do not unify:
  - rPPG CSV: 1=fake, 0=real
  - quantum: LABEL_REAL=1, LABEL_FAKE=0
  - conversion only in `data.py:csv_to_quantum_label()`
- **Gitignored artifacts** — all `output/` dirs untracked; regenerate normally
- **rPPG returns `features=None`** when usable frames < 48; pipeline emits INCONCLUSIVE exit 3
- **Write-protected `output/rppg/`** — `rppg_classifier.pkl` loaded via `pickle.load` (arbitrary-code risk if replaced)

---

## Execution Notes

- All commands assume working directory is `WORKING/` (not repo root)
- `python -m quantum.pipeline --all` regenerates `output/quantum/*` (~15s QAOA + minutes VQC)
- `python -m quantum.tests` self-checks 10/10 — run after any `qaoa.py`/`config.py`/`vqc.py` change
- Lazy imports: `evaluation.py:27` sklearn (~12s), `vqc.py:29` CUDA, `qaoa_sim.QAOASimulator` torch-native default
- QAOA torch-native sim ~0.3-0.5ms/circuit vs PennyLane ~5.6s — keep `device="auto"` default
- `Scrape/` at repo root is the ONLY place for dev files (gitignored, permanent convention)

---

## Immediate Next Step

**Phase 1 — Fix collapsed training** (can be done without re-extracting rPPG):

1. Invert `focal_loss` alpha for majority-real dataset
2. Remove `confidence_penalty` (currently rewards p≈0.5)
3. Fix `balanced_weight` to be per-sample not scalar
4. Ensure training runs full 80 epochs with validation monitoring
5. Re-run `python -m quantum.pipeline --all` from `WORKING/`
6. Check `training_log.jsonl` has 80 lines and `hybrid_vqc.pt` has `best_epoch not None`
7. If AUC lifts above 0.55 → proceed to Phase 3 (rPPG repair)
8. If AUC stays ~0.50 → skip straight to Phase 3 (upstream is the bottleneck regardless)
