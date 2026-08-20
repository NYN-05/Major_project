# quantum — hybrid quantum-classical decision layer (stage 3)

Consumes the real rPPG feature table (`WORKING/output/rppg/dataset_features.csv`,
3473 rows @30 fps: 1921 real / 1552 fake, produced by the RPPG layer) and outputs
the final KYC verdict REAL / FAKE / UNCERTAIN via QAOA feature selection + hybrid VQC.

## Module map

| File           | Responsibility                                                        |
|----------------|------------------------------------------------------------------------|
| `config.py`    | Feature contract (`FEATURE_NAMES`, 20 features), label conventions, dataclass configs, artifact paths |
| `data.py`      | Build/load `data.npz`: label flip (`csv_to_quantum_label`), plausibility filter, subject-grouped 60/20/20 split + `split_manifest.json` |
| `scaling.py`   | Train-only z-score `FeatureScaler` (saved as JSON)                    |
| `qaoa.py`      | QAOA feature selection (20 → 3) with supervised discrimination weights (Mann-Whitney AUC) + parallel restarts |
| `qaoa_sim.py`  | Exact torch-native complex128 QAOA statevector simulator (default backend, ~20× faster than PennyLane) |
| `vqc.py`       | `HybridModel` (default quantum layer: `QuantumLayerTorch` exact torch sim; legacy PennyLane QNode kept for cross-verification) + head, class-balanced focal loss, CUDA-aware train / predict / cached load |
| `evaluation.py`| Metrics (accuracy/F1/AUC/ECE), decision bins, `analyze_threshold_behavior` (Phase 1C diagnosis), 5-fold CV, classical baselines |
| `plots.py`     | ROC / confusion / calibration figure helpers                          |
| `pipeline.py`  | CLI entry (`python -m quantum.pipeline`) + `predict_features()` inference entry |
| `sweep.py`     | Crash-safe hyperparameter sweep harness (QAOA ×9 configs, VQC ~174 combos, leaderboard by AUC) |
| `tests.py`     | Self-checks (10/10): beta-alive ansatz, Hamiltonian≡classical cost, feature-contract sync, split determinism, grouping, sim cross-verification, checkpoint compat, label conversion |

## Data flow

```
dataset_features.csv → data.py (split + label flip) → scaling.py (z-score)
  → qaoa.py (20 → 3 selected) → vqc.py (train)
  → hybrid_vqc.pt + feature_scaler.json + qaoa_selection.json
  → pipeline.predict_features(features) → P(real) → REAL/FAKE/UNCERTAIN
```

Inference (`predict_features`) replays the exact training-time
transformation: fitted scaler → QAOA-selected indices → trained VQC.

## Artifacts

All regenerated and gitignored under `WORKING/output/quantum/`:
`data.npz`, `split_manifest.json`, `qaoa_selection.json`, `selection_comparison.json`,
`feature_scaler.json`, `hybrid_vqc.pt`, `training_log.jsonl`, `metrics_quantum.json`,
`metrics_baselines.json`, `threshold_analysis.json`, and the three PNG plots.

## Usage (from `WORKING/`)

```bash
python -m quantum.pipeline --all                    # full flow: build data, QAOA, train, evaluate, baselines
python -m quantum.pipeline --build-data --select --train --evaluate --baselines
python -m quantum.pipeline --train --evaluate       # rerun with existing artifacts
python -m quantum.tests                             # self-checks 10/10 (run after any qaoa.py/config.py change)
python -m quantum.sweep --timeout 600 --out sweep_leaderboard.json   # hyperparameter sweep (dev)
```

## Constraints

- `FEATURE_NAMES` (20 features) must stay identical in name AND order to
  `RPPGFeatures.feature_names()` in `RPPG/rppg/features.py` — `data.py`,
  `qaoa.py`, and `pipeline.py` index by it.
- Label conventions differ per stage (do not unify): rPPG CSV uses
  `1 = fake` / `0 = real`; this layer uses `LABEL_REAL = 1`, `LABEL_FAKE = 0`.
  The flip lives only in the tested `csv_to_quantum_label()` (Phase 1A).
- No synthetic data: the layer consumes only the real rPPG table.
- QAOA selection uses **supervised discrimination weights** (`qaoa._discrimination_weights`:
  Mann-Whitney AUC strength `2*|AUC-0.5|`). Keep `_mutual_info_weights` only as
  a documented alternative — do not reintroduce it into `select()`.
- Hamiltonian ≡ classical cost: `pipeline.py` hard-asserts `error < 1e-6`.
- QAOA ansatz: `_apply_qaoa` applies precomputed cost gates with `gamma` and
  X-mixer gates with `beta` separately. Regression guard: `test_beta_alive` in `tests.py`.
- Simulators: default circuit backends are the project's exact complex128
  torch-native sims (`qaoa_sim.QAOASimulator`, `vqc.QuantumLayerTorch`). PennyLane
  paths are legacy, kept behind `device="pennylane"` / `qnode_impl="pennylane"` for
  cross-verification; `test_qaoa_sim_matches_pennylane` / `test_torch_layer_matches_pennylane`
  pin them to ≤1e-6 / ≤1e-5.

## Current results (frozen baseline, 2026-08-19)

- **QAOA selection (20 → 3):** `['cheek_forehead_correlation',
  'left_right_cheek_correlation', 'signal_to_motion_ratio']` (seed 44, cost
  −0.767). Restart spread [11.73, −0.098, −0.767, 9.83] with `success: false` —
  the QUBO landscape is difficult; the greedy classical reference overlaps on
  2/3 features (`selection_comparison.json`).
- **VQC test:** acc 0.552 / AUC-ROC 0.535 / specificity 0.000 / balanced acc
  0.499 / ECE 0.068 — a majority-class (all-REAL) predictor, confusion
  `[[0, 310], [1, 383]]`. Grouped 5-fold CV AUC 0.556 ± 0.019.
- **Classical baselines:** best test AUC 0.582 (LogisticRegression), LinearSVC
  0.581, GNB 0.568 — same ceiling as the VQC.
- **Decision bins:** 100% UNCERTAIN (694/694) at 0.3/0.7.
- **Phase 1C diagnosis** (`threshold_analysis.json`): Case B — test scores lie
  in [0.428, 0.503], classes do not separate; thresholds cannot fix
  discrimination. Next lever is the upstream rPPG method/ROI probe (Phase 4 of
  `Docs/DEEPFAKE_KYC_SEQUENTIAL_REMEDIATION_PLAN.md`).
- **Phase 1B experiment** (balanced class weighting in the focal loss): the
  collapse flipped to all-FAKE (acc 0.444 / specificity 0.974 / AUC 0.486) —
  still no separation, still 100% UNCERTAIN. Discrimination is the bottleneck,
  not the loss weighting.

## Performance notes

- **QAOA simulator (torch-native default):** `qaoa.simulator_device(wires, cfg)`
  with `QAOASelectionConfig.device="auto"` uses `qaoa_sim.QAOASimulator` — an
  exact complex128 statevector simulation, ~0.3–0.5 ms per circuit call on the
  20-wire selection problem (~5.6 s via PennyLane, ~20× faster) and ~5 µs for
  3 wires. CPU-process-safe (workers never open CUDA contexts). Restarts
  (default 4) run in parallel via `ProcessPoolExecutor`. Full selection: ~15 s.
- **VQC training device:** the torch head of `HybridModel` runs on CUDA when
  available (`vqc.resolve_device()`); the circuit itself runs on the exact
  torch-native `QuantumLayerTorch` (complex128, batched state evolution).
  `VQCConfig.qnode_impl="auto"` selects it; `"pennylane"` selects the legacy
  PennyLane QNode (`default.qubit` + `backprop`) for cross-verification.
  Checkpoints are interchangeable (same `weights` shape `(qml_layers, n, 3)`).
  Measured CUDA-head gain: 15-epoch VQC train 12.44 s (CUDA head) vs 14.35 s
  (CPU-only head), min of 2 (`Scrape/bench_gpu_backends.py`). Inference:
  ~275 ms cold / 32 ms cached.
- **Lazy imports:** sklearn and xgboost are imported lazily so QAOA spawn
  workers and module imports stay cheap.
- `lightning.gpu` is not installable on Windows (cuQuantum/custatevec has
  no Windows wheels) and PennyLane ≥ 0.39 removed `default.qubit.torch` —
  which is why the project ships its own exact torch-native simulators.