# VQC + QAOA Accuracy Improvement Sweep Plan

**Created:** 2026-08-18
**Scope:** Broad accuracy sweep across QAOA selection and VQC architecture
**Frozen baseline (2026-08-15, 3445 rows):**
- VQC test AUC 0.503, CV(5) AUC 0.517, ECE 0.067
- Classical LR/GNB/XGB AUC 0.565–0.569
- QAOA selected: `[peak_prominence, left_right_cheek_correlation, spectral_entropy]`
- Decision bins: 100% UNCERTAIN at 0.3/0.7 thresholds
- Per-feature |AUC−0.5| ≤ 0.06 (weak signal ceiling)

---

## Phase 0 — Fresh Baseline on New Dataset

### Objective
Re-run the full quantum pipeline on the new `dataset_features.csv` (3473 rows, 1921 real / 1552 fake) to establish an accurate frozen baseline. All subsequent phases measure improvement against this number. Without this step, sweep results may reflect data drift, not model changes.

### What changes
- Input: `output/rppg/dataset_features.csv` (3473 rows, extracted today)
- Output: fresh `output/quantum/*` artifacts (data.npz, qaoa_selection.json, hybrid_vqc.pt, metrics_quantum.json, metrics_baselines.json)
- No code changes to `config.py` or `qaoa.py`; only canonical artifacts are overwritten

### Implementation
```bash
cd WORKING
python -m quantum.pipeline --all
```
Regenerates every artifact in `output/quantum/`. Takes ~10–15 min.

### Verification
- Record QAOA selected features + cost from stdout
- Record VQC test AUC, accuracy, precision, recall, F1, specificity, PR-AUC, ECE, confusion matrix
- Record grouped CV(5) AUC (mean ± std)
- Record classical baseline AUCs (LR, GNB, XGB, RF, MLP, LinearSVC)
- Record decision bin distribution
- Record Hamiltonian verification error (< 1e-6)
- Run `python -m quantum.tests` to confirm no regressions
- Save the full stdout + metrics JSONs to `Scrape/phase0_baseline_YYYYMMDD.txt` as the reference for all later phases

### Exit criteria
All metrics recorded; `python -m quantum.tests` passes; baseline saved in `Scrape/`.

---

## Phase 1 — QAOA Selection Sweep

### Objective
Test whether the current QAOA selection (3 features, p=3 layers) is optimal or if a different feature count or QAOA depth yields a stronger VQC subset. Sweep `target_features` and `p_layers` over a grid; use classical LR grouped CV as a fast proxy to identify the top-2 (k, p) combinations, then confirm each with a full VQC train+evaluate.

### Parameters
| Parameter | Values | Rationale |
|-----------|--------|-----------|
| `target_features` (k) | 3, 4, 5, 6 | 3 was evidence-backed on 421-row table; more data may support larger subsets |
| `p_layers` | 2, 3, 4 | Current p=3; test shallower and deeper QAOA |
| `restarts` | 4 (fixed) | Sufficient; more adds wall time without improving selection quality |
| `seed` | 42 (fixed) | Deterministic; different seeds only add noise |
| `n_jobs` | auto | Parallel restarts |

Total combinations: 4 × 3 = 12 QAOA runs. Each takes ~15 s (lightning.qubit). Fast.

### Implementation

#### Step 1 — Fast proxy: classical LR CV AUC on each selection

Write `Scrape/sweep_qaoa.py`:

1. Load `output/rppg/dataset_features.csv` via `quantum.data.build_dataset()` (same grouped split).
2. For each (k, p) in the grid:
   a. Build `QAOASelectionConfig(target_features=k, p_layers=p)`.
   b. Run `QAOASelector.select(X_train, y_train)`.
   c. Run `select_classical(X_train, y_train)` (greedy top-k by discrimination AUC) for comparison.
   d. On the selected features only, train `LogisticRegression(max_iter=2000, class_weight='balanced')` with `StratifiedGroupKFold(n_splits=5)` grouped CV → record AUC, accuracy, F1.
   e. Also train on the classical greedy selection for the same k → record.
3. Write results table to `Scrape/sweep_qaoa_proxy.json` (list of dicts with k, p, qaoa_features, classical_features, qaoa_lr_auc, classical_lr_auc, overlap).
4. Print a summary table sorted by `qaoa_lr_auc`.

**Why LR as proxy:** LR hit AUC 0.565–0.569 on the frozen baseline — it was better than the VQC (0.503). If a QAOA selection improves LR CV AUC, it is a better feature subset; VQC should follow. LR training + CV takes seconds.

#### Step 2 — Confirm top-2 with full VQC

From the sorted results, take the top-2 (k, p) by `qaoa_lr_auc` (break ties by `overlap_count` with classical selection). For each:

1. Set the selection in a temp config; run `quantum.pipeline` with the winning config → saves temp checkpoint to `Scrape/vqc_k{k}_p{p}/hybrid_vqc.pt`.
2. Evaluate: test AUC, PR-AUC, accuracy, specificity, ECE, decision bins, confusion matrix, grouped CV(5) AUC.
3. Record all metrics to `Scrape/phase1_vqc_confirm.json`.

#### Step 3 — Decision

- Pick the (k, p) that achieves the highest VQC grouped CV AUC (tie-break: highest test AUC, then lowest ECE).
- If the best selection differs from `[peak_prominence, left_right_cheek_correlation, spectral_entropy]`, note it.
- Write the winning selection to `Scrape/phase1_winner.json`.
- Update `QAOASelectionConfig` in a new `Scrape/config_phase1_winner.py` (copy of `config.py` with the winner params) — do NOT modify production `config.py` yet.

### Regression guards
- Run `python -m quantum.tests` after any `config.py` import is touched (even by sweep scripts, if they modify the class at runtime).
- Hamiltonian verification error must stay < 1e-6.

### Exit criteria
All 12 QAOA runs complete; top-2 confirmed with VQC; winner recorded; no test regressions.

---

## Phase 2 — VQC Architecture Sweep

### Objective
Sweep VQC hyperparameters on the winning QAOA selection from Phase 1 to find the best-performing hybrid quantum-classical architecture. Screen at 50 epochs (early stopping), then retrain the top-3 configs at full 80 epochs with full metrics.

### Parameters (screening grid)
| Parameter | Values | Rationale |
|-----------|--------|-----------|
| `qml_layers` | 2, 3, 4, 6 | Current 3; deeper = more entangling power; 6 is practical upper for 3-qubit circuit |
| `hidden_units` | 8, 16, 32 | Current 8; more capacity in classical head |
| `learning_rate` | 1e-2, 3e-3, 1e-3 | Cosine annealing; range tests convergence speed vs final loss |
| `dropout` | 0.1, 0.2 | Current 0.2; slight regularization reduction may help |
| `embedding` | AngleEmbedding (Y), IQPEmbedding | Angle is current; IQP adds feature correlations in the kernel; AmplitudeEmbedding excluded (needs 2^n qubits, messy for 3–6 features) |

Screening grid size: 4 × 3 × 3 × 2 × 2 = 144 configs. Too many for full training.

**Screening strategy:**
- Train each at 50 epochs with early stopping (patience 12) — takes ~3–6 min per config.
- Use validation AUC as the screening metric (faster than test; correlates well).
- Keep the top-10 configs by validation AUC.
- Add 10 random-grid configs from the same space for diversity (Latin hypercube sampling).
- Total screening: ~20 configs × ~5 min ≈ 100 min.

### Implementation

#### Step 1 — Screening script

Write `Scrape/sweep_vqc_screen.py`:

1. Load data via `quantum.data.build_dataset()`.
2. Load QAOA selection from Phase 1 winner.
3. For each config in the grid:
   a. Build a `VQCConfig` with the hyperparameters (override `epochs=50` for screening).
   b. Train VQC on train split, validate on val split.
   c. Record: `val_auc`, `val_loss`, `epochs_run`, `early_stopped`, `train_loss_final`.
   d. **Do not evaluate on test split** — that's for top-3 only.
4. For each config, also run grouped CV(5) on train+val using `quantum.evaluation.run_cv()` to get `cv_auc_mean` and `cv_auc_std`.
5. Write all results to `Scrape/phase2_screening.json`.
6. Print sorted table; highlight top-10 by `val_auc`, then by `cv_auc_mean`.

#### Step 2 — Full training of top-3

From screening results, pick the top-3 unique configs (no two within 1 cv_auc_std of each other). For each:

1. Set epochs back to 80.
2. Train full VQC on train split with val monitoring + early stopping.
3. Evaluate on held-out test split: full `classification_metrics()` + `decision_bins()`.
4. Run grouped CV(5) on train+val for `cv_auc_mean ± std`.
5. Save to `Scrape/phase2_top3_vqc_{rank}.json` with full metrics.
6. Save checkpoint to `Scrape/vqc_top3_{rank}/hybrid_vqc.pt`.

#### Step 3 — Embedding sub-sweep

For the top-3 configs, re-run each with `IQPEmbedding` (or `AmplitudeEmbedding` if n_features is a power of 2) and compare. Add results to the same JSON.

#### Step 4 — Decision

- Pick the config with the highest VQC grouped CV AUC (tie-break: test AUC, then ECE).
- Record all metrics in `Scrape/phase2_winner.json`.
- Create `Scrape/config_phase2_winner.py` (copy of `config.py` with both QAOA and VQC winner params).

### Regression guards
- Run `python -m quantum.tests` once after screening (before top-3 training) — no `qaoa.py` changes, but `vqc.py` changes should be harmless to tests (tests cover `qaoa.py` only).
- Verify no `config.py` or `FEATURE_NAMES` drift.

### Exit criteria
Top-3 configs fully evaluated; winner recorded; no regressions.

---

## Phase 3 — Ensemble + Threshold Calibration

### Objective
Combine the best quantum model with strong classical baselines to push overall accuracy and calibration, then calibrate decision thresholds so the system produces actionable REAL/FAKE verdicts instead of always returning UNCERTAIN.

### Part A: Ensemble

#### Implementation

Write `Scrape/sweep_ensemble.py`:

1. Load Phase 2 winner VQC model and its QAOA-selected features.
2. Train the top-3 classical baselines (LR, GNB, XGB) from `quantum.evaluation.run_baselines()` — these are already trained on the same selected features; record their test probabilities.
3. Ensemble strategies to evaluate (all from test-set probabilities):
   - **Logit averaging**: average of `logit(p_quantum)`, `logit(p_lr)`, `logit(p_gnb)`
   - **Rank averaging**: average of rank-normalized probabilities
   - **Weighted logit average**: weights = `[0.3, 0.4, 0.3]` (favor LR, which scored highest)
   - **Soft voting**: majority vote on thresholded predictions (p > 0.5)
4. For each ensemble strategy, compute: `classification_metrics()`, `decision_bins()`, `expected_calibration_error()`.
5. Write results to `Scrape/phase3_ensemble.json`.
6. Print comparison table vs standalone VQC, LR, GNB, XGB.

### Part B: Threshold calibration

#### Objective
Replace the fixed `fake_max_prob=0.3 / real_min_prob=0.7` thresholds with learned thresholds that maximize confirmed accuracy subject to a minimum coverage constraint.

#### Implementation

Write `Scrape/sweep_calibration.py`:

1. Use the best-performing model/ensemble from Part A.
2. Fit Platt scaling (logistic calibration) on validation-set probabilities:
   ```python
   from sklearn.linear_model import LogisticRegression
   calibrator = LogisticRegression(C=1.0)
   calibrator.fit(val_probs.reshape(-1, 1), y_val)
   calibrated_probs = calibrator.predict_proba(test_probs.reshape(-1, 1))[:, 1]
   ```
3. Also fit isotonic regression on validation set and compare.
4. Find optimal thresholds via grid search on validation set:
   - Search `fake_max_prob` in [0.2, 0.25, 0.3, 0.35, 0.4]
   - Search `real_min_prob` in [0.6, 0.65, 0.7, 0.75, 0.8]
   - Objective: maximize `confirmed_accuracy` (accuracy on samples where prob_real ≤ fake_max OR prob_real ≥ real_min) subject to `uncertain_rate ≤ 0.30` (at least 70% of samples get a definitive call).
5. Evaluate the chosen thresholds on test set: confirmed accuracy, coverage, decision bin distribution.
6. Record to `Scrape/phase3_calibration.json`.

### Decision

- Pick the ensemble + calibration combo that achieves the best confirmed accuracy with coverage ≥ 70%.
- If no combo achieves > 60% confirmed accuracy, acknowledge the feature-quality ceiling.
- Record in `Scrape/phase3_winner.json`.

### Exit criteria
Ensemble results recorded; thresholds calibrated; comparison table produced.

---

## Phase 4 — Pivot Gate + Feature-Side Probe

### Objective
Determine whether QAOA/VQC tuning has reached a ceiling (rPPG feature quality is the bottleneck) and, if so, probe one feature-side improvement to quantify the potential lift. This phase is conditional: run only after Phases 1–3 complete.

### Pivot gate

Compute a simple decision rule after Phase 3:

| Metric | Threshold | Action |
|--------|-----------|--------|
| Best VQC CV AUC | > 0.55 | Model tuning is helping; continue Phase 5 (not yet scoped) |
| Best VQC CV AUC | ≤ 0.55 | Feature ceiling confirmed; proceed to feature probe |

The ceiling is defined by per-feature |AUC−0.5| ≤ 0.06 on the full 20-feature set. If no VQC/ensemble config breaks 0.55 CV AUC, the bottleneck is definitively the rPPG signal, not the classifier.

### Feature probe: SQI gating

Test whether dropping low-SQI clips improves feature discrimination.

#### Implementation

Write `Scrape/probe_sqi_gating.py`:

1. Load `output/rppg/dataset_features.csv`.
2. Compute per-row `signal_quality_index` distribution.
3. For each SQI threshold in [0.05, 0.10, 0.15, 0.20]:
   a. Filter dataset to rows with SQI ≥ threshold.
   b. Compute per-feature |AUC−0.5| on the filtered set.
   c. Run classical LR grouped CV on the full 20-feature filtered set → AUC.
   d. Run QAOA selection (k=3, p=3) on the filtered set → selected features + LR CV AUC.
4. Record to `Scrape/phase4_sqi_gating.json`.
5. Compare: does higher SQI threshold improve per-feature AUC? Does LR CV AUC improve?

### Exit criteria
- Pivot gate decision documented.
- SQI gating results recorded.
- Recommendation written for next steps (either continue model tuning or pivot to feature engineering).

---

## Phase 5 (Optional) — Feature Engineering (Program A–D)

### Objective
If Phase 4 pivot gate confirms feature ceiling, implement one or more of the feature engineering programs outlined in `Scrape/baseline_20260818_pre_accuracy.txt` to expand the rPPG feature set with higher-signal features. This phase is a placeholder — scoped only if Phase 4 confirms the need.

### Candidate programs
| Program | Description | Expected effort |
|---------|-------------|----------------|
| A: SQI gating | Implement proper SQI-based clip filtering in `extract_dataset_features.py` | Low (1 day) |
| B: Window features | Add per-segment statistics (mean, std, skewness) for HR, SNR, entropy across 4-sec windows | Medium (2-3 days) |
| C: Adaptive band + motion comp | Replace fixed HR band with adaptive frequency tracking; add motion-artifact rejection | High (1 week) |
| D: Waveform CNN fusion | Train a 1D-CNN on raw rPPG waveforms; extract features from CNN latent space | High (1 week) |
| E: Threshold calibration | Already done in Phase 3 | Done |

### Implementation scope
Only A and B are in scope if the gate opens. C and D require significant architectural changes and should be a separate project.

---

## Guardrails (Apply to All Phases)

| Rule | Enforcement |
|------|-------------|
| Run `python -m quantum.tests` after every `qaoa.py`/`config.py` change | Must pass all 4 tests (beta-alive, Hamiltonian≡classical, feature contract, split determinism) |
| Hamiltonian verification error < 1e-6 | Hard-assert in `pipeline.py` — pipeline crashes if violated |
| Grouped/leakage-fixed evaluation | All CV uses `StratifiedGroupKFold` (or `GroupKFold` fallback); train/test splits subject-grouped |
| No synthetic data | Only real rPPG features from `output/rppg/dataset_features.csv` |
| `FEATURE_NAMES` contract | `quantum/config.py:FEATURE_NAMES` must match `RPPG/rppg/features.py:RPPGFeatures.feature_names()` exactly |
| Sweep artifacts in `Scrape/` | All sweep scripts, results, and temp checkpoints live in `Scrape/` — never clobber `output/quantum/*` until a winner is finalized |
| Production `config.py` changes | Only write back after a winner is confirmed across all phases |

---

## Summary Timeline

| Phase | Duration | Key Deliverable |
|-------|----------|-----------------|
| 0: Baseline | 15 min | Fresh baseline metrics in `Scrape/phase0_baseline_*.txt` |
| 1: QAOA sweep | 2–3 h | Winner (k, p) in `Scrape/phase1_winner.json` |
| 2: VQC sweep | 3–4 h | Winner config in `Scrape/phase2_winner.json` |
| 3: Ensemble + calib | 1–2 h | Winner ensemble + thresholds in `Scrape/phase3_winner.json` |
| 4: Pivot + SQI | 1 h | Decision + SQI probe results in `Scrape/phase4_*.json` |
| **Total** | **~8–10 h** | |

---

## Success Criteria

| Metric | Current | Minimum target | Stretch target |
|--------|---------|----------------|----------------|
| VQC grouped CV AUC | 0.517 | 0.53 | 0.56 |
| VQC test AUC | 0.503 | 0.52 | 0.56 |
| Ensemble test AUC | — | 0.57 | 0.60 |
| Decision bins | 100% UNCERTAIN | < 50% UNCERTAIN | < 20% UNCERTAIN |
| Confirmed accuracy (if <50% UNCERTAIN) | N/A | > 60% | > 70% |
| ECE | 0.067 | ≤ 0.10 | ≤ 0.05 |

If after Phase 3 the stretch targets are not met, Phase 4 pivot gate will confirm the feature-quality ceiling and recommend Program A/B feature engineering as the only path forward.
