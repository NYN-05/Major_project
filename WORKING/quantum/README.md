# quantum — hybrid quantum-classical decision layer (stage 3)

Consumes the real rPPG feature table (`WORKING/output/rppg/dataset_features.csv`,
produced by the RPPG layer) and outputs the final KYC verdict
REAL / FAKE / UNCERTAIN via QAOA feature selection + a hybrid VQC.

## Module map

| File           | Responsibility                                                        |
|----------------|------------------------------------------------------------------------|
| `config.py`    | Feature contract (`FEATURE_NAMES`), label conventions, dataclass configs, artifact paths |
| `data.py`      | Build/load `data.npz`: label flip, plausibility filter, subject-grouped 60/20/20 split |
| `scaling.py`   | Train-only z-score `FeatureScaler` (saved as JSON)                    |
| `qaoa.py`      | QAOA feature selection (PennyLane, `lightning.qubit` + precomputed `PauliRot` gates, parallel restarts) + classical MI-greedy reference |
| `vqc.py`       | `HybridModel` (quantum layer + head), focal loss, CUDA-aware train / predict / cached load |
| `evaluation.py`| Metrics (accuracy/F1/AUC/ECE), decision bins, 5-fold CV, classical baselines |
| `plots.py`     | ROC / confusion / calibration figure helpers                          |
| `pipeline.py`  | CLI entry (`python -m quantum.pipeline`) + `predict_features()` inference entry |

## Data flow

```
dataset_features.csv -> data.py (split) -> scaling.py (z-score)
  -> qaoa.py (10 -> 6 selected) -> vqc.py (train)
  -> hybrid_vqc.pt + feature_scaler.json + qaoa_selection.json
  -> pipeline.predict_features(features) -> P(real) -> REAL/FAKE/UNCERTAIN
```

Inference (`predict_features`) replays the exact training-time
transformation: fitted scaler -> QAOA-selected indices -> trained VQC.

## Artifacts

All regenerated and gitignored under `WORKING/output/quantum/`:
`data.npz`, `qaoa_selection.json`, `feature_scaler.json`, `hybrid_vqc.pt`,
`training_log.jsonl`, `metrics_quantum.json`, `metrics_baselines.json`,
`selection_comparison.json`, and the three PNG plots.

## Usage (from `WORKING/`)

```bash
python -m quantum.pipeline --all                    # full flow
python -m quantum.pipeline --build-data --select --train --evaluate --baselines
python -m quantum.pipeline --train --evaluate       # rerun with existing artifacts
```

## Constraints

- `FEATURE_NAMES` must stay identical in name AND order to
  `RPPGFeatures.feature_names()` in `RPPG/rppg/features.py` — `data.py`,
  `qaoa.py`, and `pipeline.py` index by it.
- Label conventions differ per stage (do not unify): rPPG CSV uses
  1 = fake / 0 = real; this layer uses `LABEL_REAL = 1`, `LABEL_FAKE = 0`
  (`data.py` flips).
- No synthetic data: the layer consumes only the real rPPG table.

## Performance notes

- **QAOA device:** `qaoa.simulator_device()` prefers `lightning.qubit`
  (SIMD statevector; falls back to `default.qubit`). Cost circuits
  precompute `PauliRot` gates — ~18 ms/cost call vs 405 ms naive on
  `lightning.qubit`, ~22x faster overall. Restarts (default 4) run in
  parallel via `ProcessPoolExecutor` (`QAOASelectionConfig.restarts` /
  `n_jobs`; 0 = auto). Full selection: ~15 s vs ~37 s before.
- **Training device:** the torch head of `HybridModel` runs on CUDA when
  available; the PennyLane QNode stays on `default.qubit` + `backprop`
  (measured fastest). Inference: ~275 ms cold / 32 ms cached.
- **Lazy imports:** sklearn and xgboost are imported lazily (~12 s /
  ~13 s on this host) so QAOA spawn workers and module imports stay cheap
  (`import quantum.pipeline` ~6.5 s vs ~22 s).
- `lightning.gpu` is not installable on Windows (cuQuantum/custatevec has
  no Windows wheels) — CPU SIMD is the ceiling for circuit simulation.