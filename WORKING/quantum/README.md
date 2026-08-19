# quantum — hybrid quantum-classical decision layer (stage 3)

Consumes the real rPPG feature table (`WORKING/output/rppg/dataset_features.csv`,
3445 rows @30 fps: 1883 real / 1562 fake, produced by the RPPG layer) and outputs
the final KYC verdict REAL / FAKE / UNCERTAIN via QAOA feature selection + hybrid VQC.

## Module map

| File           | Responsibility                                                        |
|----------------|------------------------------------------------------------------------|
| `config.py`    | Feature contract (`FEATURE_NAMES`), label conventions, dataclass configs, artifact paths |
| `data.py`      | Build/load `data.npz`: label flip, plausibility filter, subject-grouped 60/20/20 split |
| `scaling.py`   | Train-only z-score `FeatureScaler` (saved as JSON)                    |
| `qaoa.py`      | QAOA feature selection (10 → 3) with supervised discrimination weights (Mann-Whitney AUC) + parallel restarts |
| `vqc.py`       | `HybridModel` (quantum layer + head), focal loss, CUDA-aware train / predict / cached load |
| `evaluation.py`| Metrics (accuracy/F1/AUC/ECE), decision bins, 5-fold CV, classical baselines |
| `plots.py`     | ROC / confusion / calibration figure helpers                          |
| `pipeline.py`  | CLI entry (`python -m quantum.pipeline`) + `predict_features()` inference entry |
| `tests.py`     | Self-checks: beta-alive ansatz, Hamiltonian≡classical cost, feature-contract sync |

## Data flow

```
dataset_features.csv → data.py (split) → scaling.py (z-score)
  → qaoa.py (10 → 3 selected) → vqc.py (train)
  → hybrid_vqc.pt + feature_scaler.json + qaoa_selection.json
  → pipeline.predict_features(features) → P(real) → REAL/FAKE/UNCERTAIN
```

Inference (`predict_features`) replays the exact training-time
transformation: fitted scaler → QAOA-selected indices → trained VQC.

## Artifacts

All regenerated and gitignored under `WORKING/output/quantum/`:
`data.npz`, `qaoa_selection.json`, `feature_scaler.json`, `hybrid_vqc.pt`,
`training_log.jsonl`, `metrics_quantum.json`, `metrics_baselines.json`, and the three PNG plots.

## Usage (from `WORKING/`)

```bash
python -m quantum.pipeline --all                    # full flow: build data, QAOA, train, evaluate, baselines
python -m quantum.pipeline --build-data --select --train --evaluate --baselines
python -m quantum.pipeline --train --evaluate       # rerun with existing artifacts
python -m quantum.tests                             # self-checks (run after any qaoa.py/config.py change)
```

## Constraints

- `FEATURE_NAMES` must stay identical in name AND order to
  `RPPGFeatures.feature_names()` in `RPPG/rppg/features.py` — `data.py`,
  `qaoa.py`, and `pipeline.py` index by it.
- Label conventions differ per stage (do not unify): rPPG CSV uses
  `1 = fake` / `0 = real`; this layer uses `LABEL_REAL = 1`, `LABEL_FAKE = 0`
  (`data.py` flips).
- No synthetic data: the layer consumes only the real rPPG table.
- QAOA selection uses **supervised discrimination weights** (`qaoa._discrimination_weights`:
  Mann-Whitney AUC strength `2*|AUC-0.5|`). Keep `_mutual_info_weights` only as
  a documented alternative — do not reintroduce it into `select()`.
- Hamiltonian ≡ classical cost: `pipeline.py` hard-asserts `error < 1e-6`.
- QAOA ansatz: `_apply_qaoa` applies precomputed cost gates with `gamma` and
  X-mixer gates with `beta` separately. Regression guard: `test_beta_alive` in `tests.py`.

## Performance notes

- **QAOA device (GPU-first):** `qaoa.simulator_device(wires, cfg)` returns
  `(dev, backend)` with backend in {torch, lightning, default}.
  `QAOASelectionConfig.device="auto"` (default) prefers the CUDA-capable
  torch backend (`default.qubit.torch`) gated by a one-time probe
  (`qaoa._probe_gpu_device`): float64 precision (Hamiltonian ≡ classical
  cost within 1e-6) AND per-call cost ≤ ~3× lightning.qubit. On PennyLane
  >= 0.39 the torch device no longer exists, so the probe fails and QAOA
  runs on `lightning.qubit` (SIMD C++ statevector; falls back to
  `default.qubit`). Cost circuits precompute `PauliRot` gates. Restarts
  (default 4) run in parallel via `ProcessPoolExecutor`. Full selection:
  ~15 s.
- **Training device (GPU-first):** the torch head of `HybridModel` runs
  on CUDA when available; the QNode simulator is chosen by
  `VQCConfig.qnode_backend` ("auto" prefers the CUDA torch backend,
  falling back to `default.qubit` + `backprop`, measured fastest among
  CPU backends). On PennyLane >= 0.39 the QNode is CPU. Measured
  CUDA-head gain: 15-epoch VQC train 12.44 s (CUDA head) vs 14.35 s
  (CPU-only head), min of 2 (`Scrape/bench_gpu_backends.py`). Inference:
  ~275 ms cold / 32 ms cached.
- **Lazy imports:** sklearn and xgboost are imported lazily so QAOA spawn
  workers and module imports stay cheap.
- `lightning.gpu` is not installable on Windows (cuQuantum/custatevec has
  no Windows wheels; building `cuquantum-python` requires the full CUDA
  Toolkit) and PennyLane >= 0.39 removed `default.qubit.torch` — CPU SIMD
  is the circuit-simulation ceiling here.