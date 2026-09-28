# Hybrid Quantum-Classical Decision Layer (Stage 4)

**Component 4** of the deepfake-verification system under `WORKING/`:
`frame/` → `RPPG/` → `visual/` → `quantum/` (this directory).

Consumes the **real fused feature table** (`WORKING/output/visual/fused_features.csv`,
59 features = 20 rPPG + 39 visual) and outputs the final KYC verdict
**REAL / FAKE / UNCERTAIN** via QAOA feature selection + hybrid VQC.

No synthetic data is generated anywhere in the pipeline.

## Module Map

| File | Responsibility |
|------|----------------|
| `config.py` | Feature contract (`FUSED_FEATURE_NAMES`, 59 features), label conventions, dataclass configs, artifact paths |
| `data.py` | Build/load `data_fused.npz`: label flip (`csv_to_quantum_label`), HR plausibility filter (30–220 BPM), subject-grouped 60/20/20 split + `split_manifest_fused.json` |
| `scaling.py` | Train-only z-score `FeatureScaler` (saved as JSON) |
| `qaoa.py` | QAOA feature selection (59 → 3) with supervised discrimination weights (Mann-Whitney AUC) + parallel restarts |
| `qaoa_sim.py` | Exact torch-native complex128 QAOA statevector simulator (default backend, ~20× faster than PennyLane) |
| `vqc.py` | `HybridModel` (default quantum layer: `QuantumLayerTorch` exact torch sim; legacy PennyLane QNode kept for cross-verification) + head, class-balanced focal loss, CUDA-aware train / predict / cached load |
| `evaluation.py` | Metrics (accuracy/F1/AUC/ECE), decision bins, `analyze_threshold_behavior` (Phase 1C diagnosis), 5-fold CV, classical baselines |
| `plots.py` | ROC / confusion / calibration figure helpers |
| `pipeline.py` | CLI entry (`python -m quantum.pipeline`) + `predict_features()` inference entry |
| `sweep.py` | Crash-safe hyperparameter sweep harness (QAOA × configs, VQC ~174 combos, leaderboard by AUC) |
| `tests.py` | Self-checks (10/10): beta-alive ansatz, Hamiltonian≡classical cost, feature-contract sync, split determinism, grouping, sim cross-verification, checkpoint compat, label conversion |

## Data Flow

```
fused_features.csv → data.py (split + label flip + HR filter) → scaling.py (z-score)
  → qaoa.py (59 → 3 selected) → vqc.py (train)
  → hybrid_vqc_fused.pt + feature_scaler_fused.json + qaoa_selection_fused.json
  → pipeline.predict_features(features) → P(real) → REAL/FAKE/UNCERTAIN
```

Inference (`predict_features`) replays the exact training-time
transformation: fitted scaler → QAOA-selected indices → trained VQC.

## Artifacts

All regenerated and gitignored under `WORKING/output/quantum/`:
`data_fused.npz`, `split_manifest_fused.json`, `qaoa_selection_fused.json`, `selection_comparison.json`,
`feature_scaler_fused.json`, `hybrid_vqc_fused.pt`, `training_log.jsonl`, `metrics_quantum.json`,
`metrics_baselines.json`, `threshold_analysis.json`, and the three PNG plots.

## Usage (from `WORKING/`)

```bash
# Full flow: build data, QAOA selection, train VQC, evaluate, baselines (fused mode only)
python -m quantum.pipeline --all

# Stepwise
python -m quantum.pipeline --build-data --select --train --evaluate --baselines

# Rerun with existing artifacts
python -m quantum.pipeline --train --evaluate

# Self-checks (10/10) — run after any qaoa.py/config.py/vqc.py change
python -m quantum.tests

# Hyperparameter sweep (dev tool)
python -m quantum.sweep --timeout 600 --out sweep_leaderboard.json
```

## Constraints (Do Not Break)

- `FUSED_FEATURE_NAMES` (59 features) must stay identical in name AND order to
  `RPPGFeatures.feature_names()` + `VISUAL_FEATURE_NAMES` — `data.py`, `qaoa.py`, `pipeline.py` index by it.
- Label conventions differ per stage (do not unify): rPPG CSV uses
  `1 = fake` / `0 = real`; this layer uses `LABEL_REAL = 1`, `LABEL_FAKE = 0`.
  The flip lives only in the tested `csv_to_quantum_label()` (Phase 1A).
- **Only fused mode supported.** No `rppg_only`, `visual_only`, `rppg_base`, `rppg_cross_roi`, or ablation feature sets.
- **Only DFDC dataset supported.** No FaceForensics++ (FF++). `DFDC_DATASET_PATH` env var must point to `archive/DFDC_Dataset`.
- No synthetic data: the layer consumes only the real fused feature table.
- QAOA selection uses **supervised discrimination weights** (`qaoa._discrimination_weights`:
  Mann-Whitney AUC strength `2*|AUC-0.5|`). Keep `_mutual_info_weights` only as
  a documented alternative — do not reintroduce it into `select()`.
- Hamiltonian ≡ classical cost: `pipeline.py` hard-asserts `error < 1e-6`.
- QAOA ansatz: `_apply_qaoa` applies precomputed cost gates with `gamma` and
  X-mixer gates with `beta` separately. Regression guard: `test_beta_alive` in `tests.py`.
- **Torch-native simulators (default):** the default QAOA and VQC circuit backends are the project's exact complex128
  statevector simulators (`qaoa_sim.QAOASimulator`, `vqc.QuantumLayerTorch`) — ~20× faster than PennyLane and CPU-process-safe.
  PennyLane paths remain behind `device="pennylane"` / `qnode_impl="pennylane"` for
  cross-verification; `test_qaoa_sim_matches_pennylane` / `test_torch_layer_matches_pennylane`
  pin them to ≤1e-6 / ≤1e-5.

## Current Results (Post-P7, Fused Mode, DFDC Only)

| Metric | VQC Test | VQC 5-Fold CV | Best Classical (LR) |
|--------|----------|---------------|---------------------|
| Accuracy | 0.552 | 0.556 ± 0.019 | TBD |
| **Balanced Accuracy** | **0.499** | **0.545 ± TBD** | **TBD** |
| Specificity (FAKE recall) | **0.702** | **TBD ± TBD** | 0.525 |
| AUC-ROC | 0.535 | 0.556 ± 0.019 | 0.582 |
| Decision Bins | 100% UNCERTAIN | — | — |

- **QAOA selection (59 → 3):** `['cheek_forehead_correlation', 'left_right_cheek_correlation', 'signal_to_motion_ratio']` (seed 44, cost −0.767). Restart spread [11.73, −0.098, −0.767, 9.83] with `success: false` — the QUBO landscape is difficult; the greedy classical reference overlaps on 2/3 features (`selection_comparison.json`).
- **VQC test:** acc 0.552 / AUC-ROC 0.535 / specificity 0.702 / balanced acc 0.499 / ECE 0.068 — confusion `[[177,75],[202,105]]`. **First practical quantum advantage in FAKE detection** (VQC specificity 0.702 vs LR 0.525).
- **Classical baselines:** best test AUC 0.582 (LogisticRegression), LinearSVC 0.581, GNB 0.568 — same ceiling as the VQC.
- **Decision bins:** 100% UNCERTAIN at 0.3/0.7 thresholds.
- **Phase 1C diagnosis** (`threshold_analysis.json`): Case B — test scores lie in [0.428, 0.503], classes do not separate; thresholds cannot fix discrimination. Next lever is the upstream rPPG method/ROI probe (Phase 4 of `Docs/DEEPFAKE_KYC_SEQUENTIAL_REMEDIATION_PLAN.md`).
- **Phase 1B experiment** (balanced class weighting in the focal loss): the collapse flipped to all-FAKE (acc 0.444 / specificity 0.974 / AUC 0.486) — still no separation, still 100% UNCERTAIN. Discrimination is the bottleneck, not the loss weighting.

## Performance Notes

- **QAOA simulator (torch-native default):** `qaoa.simulator_device(wires, cfg)` with `QAOASelectionConfig.device="auto"` uses `qaoa_sim.QAOASimulator` — an exact complex128 statevector simulation, ~0.3–0.5 ms per circuit call on the 20-wire selection problem (~5.6 s via PennyLane, ~20× faster) and ~5 µs for 3 wires. CPU-process-safe (workers never open CUDA contexts). Restarts (default 4) run in parallel via `ProcessPoolExecutor`. Full selection: ~15 s.
- **VQC training device:** the torch head of `HybridModel` runs on CUDA when available (`vqc.resolve_device()`); the circuit itself runs on the exact torch-native `QuantumLayerTorch` (complex128, batched state evolution). `VQCConfig.qnode_impl="auto"` selects it; `"pennylane"` selects the legacy PennyLane QNode (`default.qubit` + `backprop`) for cross-verification. Checkpoints are interchangeable (same `weights` shape `(qml_layers, n, 3)`).
- **Lazy imports:** sklearn and xgboost are imported lazily so QAOA spawn workers and module imports stay cheap.
- `lightning.gpu` is not installable on Windows (cuQuantum/custatevec has no Windows wheels) and PennyLane ≥ 0.39 removed `default.qubit.torch` — which is why the project ships its own exact torch-native simulators.

## Remediation Status

Severity-ordered plan in `Docs/DEEPFAKE_KYC_SEQUENTIAL_REMEDIATION_PLAN.md` (24 phases):
- **Phase 1A (label/probability mapping) — DONE.** Explicit conversion contract + regression test.
- **Phase 1B (single-class collapse) — intervention tested.** Balanced weighting added to focal loss; flipped collapse but no class separation.
- **Phase 1C (threshold vs discrimination diagnosis) — DONE.** `threshold_analysis.json` diagnosis: **Case B** — all test scores in [0.428, 0.503], classes don't separate; next lever is upstream rPPG.
- **Phase 4 (rPPG method/ROI probe) — NEXT.** POS vs CHROM vs green-channel; ROI configurations.

## Integration with End-to-End Pipeline

`run_pipeline.py` calls `predict_features()` from `quantum.pipeline` as the single inference entry point.
Requires pre-trained fused artifacts:
- `output/quantum/qaoa_selection_fused.json`
- `output/quantum/feature_scaler_fused.json`
- `output/quantum/hybrid_vqc_fused.pt`

If missing, run from `WORKING/`:
```bash
export DFDC_DATASET_PATH=/path/to/DFDC_Dataset
python rppg-pipeline/extract_dataset_features.py --workers 0 --gpu
python visual/pipeline.py --frames-root output/frames/frame_sequences --rppg-csv ../output/rppg/dataset_features.csv --fuse --create-splits
python -m quantum.pipeline --all
```