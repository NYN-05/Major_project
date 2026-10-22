# Hybrid Quantum-Classical Decision Layer (Stage 4)

> **Stage 4 of 4:** `frame/` → `RPPG/` → `visual/` → `quantum/` (this directory, final verdict).

Consumes the **real fused feature table** (`WORKING/MAJ_OUTPUT_ROOT/visual/fused_features.csv`,
59 features = 20 rPPG + 39 visual) and outputs the final KYC verdict
**REAL / FAKE** via QAOA feature selection + hybrid VQC.

No synthetic data is generated anywhere in the pipeline.

## Contents

- [Module Map](#module-map)
- [Data Flow](#data-flow)
- [Artifacts](#artifacts)
- [Usage](#usage)
- [Constraints (Do Not Break)](#constraints-do-not-break)
- [Current Results (Full-Dataset Run 2026-10-21, Fused Mode, DFDC Only)](#current-results-full-dataset-run-2026-10-21-fused-mode-dfdc-only)
- [Performance Notes](#performance-notes)
- [Remediation Status and Open Work](#remediation-status-and-open-work)
- [Integration with End-to-End Pipeline](#integration-with-end-to-end-pipeline)

## Module Map

| File            | Responsibility |
| --------------- | -------------- |
| `config.py`     | Feature contract (`FUSED_FEATURE_NAMES`, 59 features), `PipelineStatus`, label conventions, sha256 helpers, dataclass configs, artifact paths |
| `data.py`       | Build/load `data_fused.npz`: label flip (`csv_to_quantum_label`), HR plausibility filter (30–220 BPM), subject-grouped 60/20/20 split + `split_manifest_fused.json` |
| `scaling.py`    | Train-only z-score `FeatureScaler` (saved as JSON), `validate_feature_schema()` / `transform_features()` |
| `qaoa.py`       | QAOA feature selection (59 → 3) with supervised discrimination weights (Mann-Whitney AUC) + parallel restarts |
| `qaoa_sim.py`   | Exact torch-native complex128 QAOA statevector simulator (default backend, ~20× faster than PennyLane) |
| `vqc.py`        | `HybridModel` (default quantum layer: `QuantumLayerTorch` exact torch sim; legacy PennyLane QNode kept for cross-verification) + head, class-balanced focal loss, CUDA-aware train / predict / cached load |
| `evaluation.py` | Metrics (accuracy/F1/AUC/ECE), decision bins, `analyze_threshold_behavior` (Phase 1C diagnosis), 5-fold CV, classical baselines, ROC/confusion/calibration figure helpers |
| `pipeline.py`   | CLI entry (`python -m quantum.pipeline`) + `predict_features()` inference entry |
| `explain.py`    | Evidence dossier builder (verdict explanation consumed by the API/frontend) |
| `operations.py` | Shared hashing/fingerprint/dataset utilities for artifacts |
| `tests.py`      | Self-checks (10/10): beta-alive ansatz, Hamiltonian≡classical cost, feature-contract sync, split determinism, grouping, sim cross-verification, checkpoint compat, label conversion |

## Data Flow

```text
fused_features.csv → data.py (split + label flip + HR filter) → scaling.py (z-score)
  → qaoa.py (59 → 3 selected) → vqc.py (train)
  → hybrid_vqc_fused.pt + feature_scaler_fused.json + qaoa_selection_fused.json
  → pipeline.predict_features(features) → P(real) → REAL/FAKE
```

Inference (`predict_features`) replays the exact training-time
transformation: fitted scaler → QAOA-selected indices → trained VQC.

## Artifacts

All regenerated and gitignored under `WORKING/MAJ_OUTPUT_ROOT/quantum/`:
`data_fused.npz`, `split_manifest_fused.json`, `qaoa_selection_fused.json`, `selection_comparison.json`,
`feature_scaler_fused.json`, `hybrid_vqc_fused.pt`, `training_log.jsonl`, `metrics_quantum.json`,
`metrics_baselines.json`, `threshold_analysis.json`, and the three PNG plots.

## Usage

From `WORKING/`:

```bash
# Full flow: build data, QAOA selection, train VQC, evaluate, baselines (fused mode only)
python -m quantum.pipeline --all

# Stepwise
python -m quantum.pipeline --build-data --select --train --evaluate --baselines

# Rerun with existing artifacts
python -m quantum.pipeline --train --evaluate

# Self-checks (10/10) — run after any qaoa.py/config.py/vqc.py change
python -m quantum.tests
```

## Constraints (Do Not Break)

> `FUSED_FEATURE_NAMES` (59 features) must stay identical in name AND order to
> `RPPGFeatures.feature_names()` + `VISUAL_FEATURE_NAMES` — `data.py`, `qaoa.py`, `pipeline.py` index by it.

> Label conventions differ per stage (do not unify): rPPG CSV uses
> `1 = fake` / `0 = real`; this layer uses `LABEL_REAL = 1`, `LABEL_FAKE = 0`.
> The flip lives only in the tested `csv_to_quantum_label()` (Phase 1A).

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

## Current Results (Full-Dataset Run 2026-10-21, Fused Mode, DFDC Only)

Data: 3,912 labelled rows (2,087 REAL / 1,827 FAKE) after the HR-plausibility
filter; seed-42 clip-grouped 60/20/20 split (train 2,348 / val 782 /
test 782). Metrics are read from `MAJ_OUTPUT_ROOT/quantum/metrics_quantum.json`
and `metrics_baselines.json`; headline metrics are computed at the 0.5 cut
(all six classical baselines share that cut), while the deployed decision
bins use the calibrated validation-only threshold τ* = 0.47942 persisted in
the checkpoint and consumed at inference (`pipeline.py:114`).

| Metric                            | Quantum VQC (test)                 | VQC 5-Fold CV  | Best Classical (MLP) |
| --------------------------------- | ---------------------------------- | -------------- | -------------------- |
| Accuracy (0.5 cut)                | 0.569 (Wilson 95% [0.534, 0.603])  | 0.589 ± 0.020  | 0.632                |
| **Balanced Accuracy**             | **0.582**                          | 0.596 ± 0.020  | 0.631                |
| Specificity (FAKE recall)         | 0.773                              | 0.709 ± 0.039  | 0.616                |
| AUC-ROC                           | 0.658                              | 0.645 ± 0.020  | 0.684                |
| PR-AUC (REAL-positive)            | 0.687                              | 0.668 ± 0.023  | 0.730                |
| ECE / Brier                       | 0.080 / 0.236                      | 0.053 (ECE)    | 0.027 / 0.222        |
| Confusion @0.5 [[TN,FP],[FN,TP]]  | [[282, 83], [254, 163]]             | —              | —                    |
| Decision Bins (τ*)                | 378 REAL / 404 FAKE                | —              | —                    |

- **QAOA selection (59 → 3):** `['lbp_uniform_hist_0', 'deep_feat_5', 'lbp_uniform_hist_9']` — three visual-side features (chosen seed 47, cost -2.4726, `success: false`, 8-restart spread [-2.423, -2.473], `selection_stability.mean_jaccard` 1.0 over n_runs 1).
- **VQC test:** acc 0.569 / AUC-ROC 0.658 / FAKE rejection (specificity) 0.773 / balanced acc 0.582 / ECE 0.080 — confusion `[[282,83],[254,163]]` (REAL-positive); FAR 0.227 (FAKE accepted as genuine), FRR 0.609.
- **Classical baselines (same split, same 0.5 cut):** best test AUC 0.684 and accuracy 0.632 (MLP); baseline range acc 0.598-0.632, AUC 0.634-0.684, PR-AUC 0.670-0.730. **Every classical baseline currently exceeds the VQC in accuracy, AUC and PR-AUC on this split — no quantum-advantage claim is supportable.**
- **CV:** grouped 5-fold over the 2,348 training rows: VQC acc 0.589 ± 0.020, AUC 0.645 ± 0.020, balanced acc 0.596 ± 0.020, ECE 0.053.
- **Single-seed caveat:** all figures are seed-42, single-split. Repeatability work (multi-seed, bootstrap CIs, at-τ* metrics, EER) is staged in `paper_experiments/` and tracked in `reviewer_findings_response.md`.
- **Historical Phase 1B/1C experiments** (balanced focal weighting collapse flip; `threshold_analysis.json` Case-B non-separation) ran on a small smoke split and are retained as process history only.

## Performance Notes

| Area | Detail |
| ---- | ------ |
| QAOA simulator (torch-native default) | `qaoa.simulator_device(wires, cfg)` with `QAOASelectionConfig.device="auto"` uses `qaoa_sim.QAOASimulator` — an exact complex128 statevector simulation, ~0.3–0.5 ms per circuit call on the 59-wire selection problem (~5.6 s via PennyLane, ~20× faster) and ~5 µs for 3 wires. CPU-process-safe (workers never open CUDA contexts). Restarts (default 8) run in parallel via `ProcessPoolExecutor`. Full selection: ~15 s. |
| VQC training device | The torch head of `HybridModel` runs on CUDA when available (`vqc.resolve_device()`); the circuit itself runs on the exact torch-native `QuantumLayerTorch` (complex128, batched state evolution). `VQCConfig.qnode_impl="auto"` selects it; `"pennylane"` selects the legacy PennyLane QNode (`default.qubit` + `backprop`) for cross-verification. Checkpoints are interchangeable (same `weights` shape `(qml_layers, n, 3)`). |
| Lazy imports | sklearn and xgboost are imported lazily so QAOA spawn workers and module imports stay cheap. |
| Simulator rationale | `lightning.gpu` is not installable on Windows (cuQuantum/custatevec has no Windows wheels) and PennyLane ≥ 0.39 removed `default.qubit.torch` — which is why the project ships its own exact torch-native simulators. |

## Remediation Status and Open Work

Severity-ordered plan in `Docs/maj_proj_report_sem_6/SEM6_report/` (24 phases, covered by semester report).

| Item | Status |
| ---- | ------ |
| **Phase 1A** (label/probability mapping) | **DONE.** Explicit conversion contract + regression test. |
| **Phase 1B** (single-class collapse) | **Intervention tested.** Balanced weighting added to focal loss; flipped collapse but no class separation. |
| **Phase 1C** (threshold vs discrimination diagnosis) | **DONE (historical, smoke-run scale).** `threshold_analysis.json` diagnosis: **Case B** — all test scores in [0.428, 0.503], classes did not separate. |
| Phase 4 probe features (spectral flatness/centroid/kurtosis/coherence) | **Integrated** in the 20-name rPPG contract (`RPPG_FEATURE_NAMES`). |
| **Full-dataset training** | **DONE 2026-10-21** — figures in the table above; artifacts re-verified. |
| Multi-seed repeatability + bootstrap CIs | **PENDING** — scripts in `paper_experiments/` (multi_seed_repeats.py, ci_from_recorded.py), not yet executed. |
| EER + at-τ* metric reporting | **PENDING** — small evaluator extension (reviewer item T6). |
| Ablation matrix / robustness grid / latency benchmarks | **READY (unblocked by the full run)** — protocols provided, execution awaits authorization. |

## Integration with End-to-End Pipeline

`run_pipeline.py` calls `predict_features()` from `quantum.pipeline` as the single inference entry point.
Requires pre-trained fused artifacts:

- `MAJ_OUTPUT_ROOT/quantum/qaoa_selection_fused.json`
- `MAJ_OUTPUT_ROOT/quantum/feature_scaler_fused.json`
- `MAJ_OUTPUT_ROOT/quantum/hybrid_vqc_fused.pt`

If missing, run from `WORKING/`:

```bash
export DFDC_DATASET_PATH=/path/to/DFDC_Dataset
python RPPG/extract_dataset_features.py --workers 0 --gpu
python visual/pipeline.py --frames-root MAJ_OUTPUT_ROOT/frames/frame_sequences --rppg-csv MAJ_OUTPUT_ROOT/rppg/dataset_features.csv --fuse --create-splits
python -m quantum.pipeline --all
```
