# Deepfake Video Detection for KYC (rPPG + Quantum ML)

Detection of deepfake videos for video KYC in financial systems using
**rPPG (remote photoplethysmography)** as the primary physiological evidence layer and
**hybrid quantum-classical ML** as the final decision layer, designed for low-resolution videos.

## Architecture (Three-Stage Pipeline)

```
Input KYC video
  → Stage 1: Frame sampling + quality filtering    (WORKING/frame/)
  → Stage 2: rPPG signal extraction + features     (WORKING/RPPG/)
  → Stage 3: QAOA selection → hybrid VQC verdict   (WORKING/quantum/)
  → Decision: REAL / FAKE / UNCERTAIN
```

End-to-end orchestrator (`WORKING/run_pipeline.py`):

```
frames → rPPG → quantum → verdict (REAL / FAKE / UNCERTAIN)
```

## Repository Layout

| Path                         | Description                                                                                                               |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| `WORKING/`                 | Active project root (all three pipeline stages + end-to-end runner)                                                       |
| `WORKING/frame/`           | Stage 1 - frame sampling, YOLO face detection, quality filtering (has its own`app/` + `requirements.txt`)             |
| `WORKING/RPPG/`            | Stage 2 - MediaPipe face ROIs -> POS/CHROM pulse -> 20 physiological features (+`rppg-pipeline/`, `requirements.txt`) |
| `WORKING/quantum/`         | Stage 3 - QAOA feature selection -> hybrid VQC -> P(real) -> verdict (run via`python -m quantum.pipeline`)              |
| `WORKING/run_pipeline.py`  | End-to-end orchestrator: frames -> rPPG -> quantum -> verdict                                                             |
| `WORKING/output/`          | Global regenerated-artifacts root (all`output/` dirs untracked)                                                         |
| `WORKING/output/frames/`   | Stage 1:`frame_sequences/`, `frame_extraction_*.json(l)`, `docs/`                                                   |
| `WORKING/output/rppg/`     | Stage 2:`dataset_features.csv`, `rppg_classifier.pkl` + metadata, plots                                               |
| `WORKING/output/quantum/`  | Stage 3:`data.npz`, `qaoa_selection.json`, `feature_scaler.json`, `hybrid_vqc.pt`, metrics, plots                 |
| `WORKING/output/pipeline/` | `run_pipeline.py` result JSON                                                                                           |
| `frontend/`                | Web UI: React + Vite frontend + stdlib-only API server (port 8000)                                                        |
| `Docs/`                    | Project docs: RMTT semester report, quantum layer guide, key findings, remediation plan, problem analysis                   |
| `FF++/`                    | FaceForensics++ dataset (gitignored)                                                                                      |
| `Scrape/`                  | Dev scratch: tests, temp scripts, debug artifacts (gitignored)                                                            |

## Components

### 1. Frame Sampling & Quality (`WORKING/frame/`)

Modular face pipeline for video sources: frame sampling at a controlled
FPS, face detection (YOLO), quality assessment (blur / dark / overexposed / no-face / face-too-small / extreme-pose rules).
Outputs accepted JPEGs + per-frame metadata JSONL for the rPPG stage.

```bash
# Standalone frame stage (from WORKING/frame/)
python app/pipeline.py --source test.mp4 --save-metadata
```

(`app/pipeline.py` is the single CLI entry point; legacy `app/main.py` / `app/extract_frames.py` no longer exist.)

### 2. rPPG Pipeline (`WORKING/RPPG/`)

POS/CHROM pulse reconstruction from facial ROIs (left cheek, right cheek, forehead)
and a 20-feature physiological vector per video: heart rate, SNR, PRV, spectral
entropy, MAD, signal quality index, inter-region correlations, pulse morphology
(peak width, dicrotic notch), inter-ROI phase lag / pulse-transit-time proxy,
motion contamination, and per-window stability statistics (full list in
`RPPGFeatures.feature_names()`). Features are
persisted with labels (1 = fake, 0 = real) in `WORKING/output/rppg/dataset_features.csv`,
which is the direct data source for the quantum layer.

```bash
# From WORKING/RPPG/ - rebuild output/rppg/dataset_features.csv
#   --workers 0          = all CPU cores
#   --include-ffpp       = include FaceForensics++ clips
python rppg-pipeline/extract_dataset_features.py --include-ffpp --workers 0

# Train rPPG classifier (side path, not used for final verdict)
python rppg-pipeline/train_classifier.py
```

Feature sources: DFDC (`archive/DFDC_Dataset/Fake|Real`) and
FaceForensics++ (`FF++/` train/val/test via `--include-ffpp`). The full
extraction (3473 rows @30 fps: 1921 real / 1552 fake) was rebuilt 2026-08-19
(3471 rows survive the quantum HR-plausibility filter at dataset-build time).
Extraction supports GPU-accelerated face detection (`--gpu`, YuNet ONNX Runtime
CUDA) and quality gates (`--min-sqi`, `--max-nan-features`).
The legacy `archive (1)` CSV layout is absent on disk.

### 3. Quantum Model (`WORKING/quantum/`)

Hybrid classical-quantum decision stage built with PennyLane + PyTorch. It consumes
the rPPG 20-feature vector **directly** (same names/order as `RPPGFeatures.feature_names()`);
no synthetic data is generated anywhere in the pipeline.

- **Data** - `data.py` builds `output/quantum/data.npz` from the real labeled rPPG feature
  table `output/rppg/dataset_features.csv` (3473 rows: 1921 real / 1552 fake, DFDC + FF++).
  rPPG label `1 = fake` is flipped via the explicit, tested conversion
  `csv_to_quantum_label()` to the quantum convention `LABEL_REAL = 1, LABEL_FAKE = 0`
  (Phase 1A of the remediation plan; no undocumented `1 - csv_label` arithmetic).
  HR-plausibility filter (30–220 BPM, non-finite rejection) drops implausible rows at
  build time. Subject-grouped random split (seeded, per-clip for DFDC, per-clip for FF++
  with first-id token grouping), persisted to `split_manifest.json`.
- **QAOA feature selection** - `qaoa.py` selects the 3 most informative rPPG features using
  a cost Hamiltonian with **supervised discrimination weights** (`qaoa._discrimination_weights`:
  sign-agnostic exact Mann-Whitney AUC strength `2*|AUC-0.5|`, deterministic, no sklearn)
  plus correlation redundancy penalty and cardinality constraint, optimized with COBYLA.
  4 parallel restarts via `ProcessPoolExecutor`. The circuit runs on the project's own
  **exact torch-native statevector simulator** (`qaoa_sim.QAOASimulator`, complex128;
  ~0.3–0.5 ms vs ~5.6 s per PennyLane call on the 20-wire problem) and is cross-verified
  against PennyLane by the test suite.
- **Hybrid VQC** - `vqc.py`: angle-encoded features into a parameterized quantum
  circuit (`StronglyEntanglingLayers`) feeding a small classical head. The default
  quantum layer is the exact torch-native `QuantumLayerTorch` simulator (complex128,
  batched, same `weights` shape as the PennyLane layer so checkpoints are
  interchangeable); the legacy PennyLane QNode path remains behind `qnode_impl="pennylane"`
  for cross-verification. Training uses class-balanced focal loss (Phase 1B), cosine-annealed
  LR, gradient clipping, and early stopping on validation loss with restore of the
  best-validation checkpoint. CUDA head when available.
- **Evaluation** - `evaluate.py`: accuracy / precision / recall / F1 / AUC-ROC / ECE,
  KYC-friendly decision bins (real / uncertain / fake), ROC, confusion-matrix and
  calibration-curve plots. **StratifiedKFold (5-fold) cross-validation** (mean ± std,
  balanced accuracy) for the VQC and every baseline. Classical baselines:
  RandomForest, MLP, LogisticRegression, calibrated LinearSVC, GaussianNB, XGBoost.
  Phase 1C adds `analyze_threshold_behavior()` — a threshold-sweep diagnosis that
  separates a discrimination failure from a threshold failure (artifact
  `output/quantum/threshold_analysis.json`).
- **Orchestration** - `pipeline.py` drives the full training flow and exposes
  `predict_features()` as the single inference entry point used by `run_pipeline.py`.
  Hard-asserts Hamiltonian ≡ classical cost (`error < 1e-6`).
- **Sweep harness** - `sweep.py` runs crash-safe hyperparameter sweeps (QAOA phase ×9
  configs, VQC phase ~174 combos) with a leaderboard by AUC.

```bash
# From WORKING/: build data, QAOA selection, train VQC, evaluate, baselines
python -m quantum.pipeline --all
# Self-checks (10/10): beta-alive ansatz, Hamiltonian≡classical cost,
# feature-contract sync, split determinism, grouping, sim cross-verification,
# checkpoint compat, label conversion
python -m quantum.tests
```

**Frozen baseline (2026-08-19, 3473 rows, torch-native simulators):**
QAOA selects `['cheek_forehead_correlation', 'left_right_cheek_correlation',
'signal_to_motion_ratio']` (seed 44, cost −0.767; restart spread
[11.73, −0.098, −0.767, 9.83] shows a difficult optimization landscape, and the
greedy classical reference overlaps on 2/3 features). Grouped 5-fold CV AUC
0.556 ± 0.019; test acc 0.552 / AUC 0.535 / specificity 0.000 / balanced acc 0.499
(majority-class collapse: confusion [[0, 310], [1, 383]]). Classical baselines:
best LR test AUC 0.582. Decision bins remain 100% UNCERTAIN at the 0.3/0.7
thresholds. Per-feature |AUC−0.5| ≤ ~0.06 across all 20 features confirms the
rPPG features themselves carry limited class signal.

**Remediation status (severity-ordered plan in
`Docs/DEEPFAKE_KYC_SEQUENTIAL_REMEDIATION_PLAN.md`, 24 phases):**
- **Phase 1A (label/probability mapping) — DONE.** Explicit conversion contract
  (`csv_to_quantum_label`, `quantum_to_display_label`) + regression test; mapping verified correct.
- **Phase 1B (single-class collapse) — intervention tested.** Balanced class weighting
  added to the focal loss. The experimental run flipped the collapse from REAL to FAKE
  (test acc 0.444 / specificity 0.974 / balanced acc 0.495 / AUC 0.486) — still no
  class separation, still 100% UNCERTAIN.
- **Phase 1C (threshold vs discrimination diagnosis) — DONE.** `threshold_analysis.json`
  diagnosis: **Case B** — all test scores lie in [0.428, 0.503], classes do not separate,
  so thresholds cannot fix discrimination. **Next lever: upstream rPPG method/ROI probe
  (Phase 4), not threshold or model tuning.**

## Install

```bash
# Stage 1 (frame) - CUDA PyTorch, YOLO
pip install -r WORKING/frame/requirements.txt

# Stage 2 (rPPG) - MediaPipe, OpenCV, scipy, etc.
pip install -r WORKING/RPPG/requirements.txt

# Stage 3 (quantum) - PennyLane, PyTorch, scikit-learn, scipy, matplotlib
pip install pennylane torch numpy scikit-learn scipy matplotlib

# Frontend (optional)
cd frontend && npm install
```

## Run End-to-End

```bash
# From WORKING/
python run_pipeline.py --source path/to/video.mp4 --method POS --out result.json
```

Requires pre-trained quantum artifacts (`output/quantum/qaoa_selection.json`, `hybrid_vqc.pt`, `feature_scaler.json`).
If missing, run once from `WORKING/`:

```bash
python -m quantum.pipeline --all
```

## Frontend (Web UI)

```bash
# From frontend/
python server.py          # Backend API on http://127.0.0.1:8000
npm run dev               # React dev server on http://localhost:5173 (proxies /api)
```

Upload a KYC video, watch the three-stage pipeline run live via SSE,
and read the verdict with its evidence dossier (accepted frames, physiological
features, quantum probabilities and plots, the pulse waveform).

### API Contract (`frontend/server.py`)

```
POST /api/detect              upload video (raw body + X-Filename header) → {job: <id>}
GET  /api/jobs/<id>           {done, error, video, signal, lines[-400:], result}
GET  /api/jobs/<id>/events    SSE: line / stage / result / signal / error
GET  /api/previous            last canonical pipeline result (with _signal)
GET  /api/health              {ok, platform, running, has_previous, sequences, artifacts}
GET  /api/artifacts?dir=      list files under output/<rel>
GET  /api/files?rel=          serve artifact file (path traversal protected)
```

Upload limits: 200 MB max; magic-byte validation (MP4/MOV ftyp, AVI RIFF, WebM EBML).
Concurrency: max 2 jobs (429 on overflow). Jobs TTL: 1 h; frame sequences: 24 h.
CORS restricted to localhost origins. 30-min hard timeout per pipeline run.

### Frontend State Machine

```
idle → selected (preview + metadata + Start)
     → running  (7-stage pipeline + live panel + creeping progress)
     → done     (Verdict gauge, Insights, Signal canvas, Quantum flow, FrameSamples)
     → error    (inline banner)
```

Theme toggle persists `rppgqc.theme` in localStorage; respects `prefers-reduced-motion`. Responsive: 1920 → 375 px, no overflow.

## Verified Constraints (do not break)

- **No synthetic data.** The quantum layer consumes only the real rPPG feature table `output/rppg/dataset_features.csv` (3473 rows @30 fps: 1921 real / 1552 fake — DFDC archive + FF++). Never reintroduce a generator or a transform/bridge layer.
- **Feature contract:** `FEATURE_NAMES` in `quantum/config.py` (20 features) must stay identical in name AND order to `RPPGFeatures.feature_names()` in `RPPG/rppg/features.py` (duplicated on purpose; `data.py`, `qaoa.py`, and `pipeline.py` all index by it). Keep the two lists in sync; `test_feature_contract_sync` guards it.
- **Label conventions differ per stage — do not unify:**
  - rPPG CSV: `1 = fake, 0 = real`
  - quantum: `LABEL_REAL = 1`, `LABEL_FAKE = 0`; `quantum/data.py` flips via the tested `csv_to_quantum_label()` (`y = 1 - csv_label` lives only inside that function)
  - rPPG RandomForest cross-check in `run_pipeline.py`: `1 = DEEPFAKE`
- **Gitignored artifacts:** `*.csv`, `*.json`, `*.pkl`, `*.mp4`, and all `output/` dirs are untracked — `dataset_features.csv`, `output/quantum/*`, and the trained models will not appear in `git status`. Regenerating them is normal.
- **rPPG returns `features=None`** when usable frames < `min_usable_frames` (48); `run_pipeline.py` then emits INCONCLUSIVE and exits 3. New code must handle `None`.
- **Stage 1 feeds stage 2.** `run_pipeline.py` hands the frame stage's accepted JPEGs (`output/frames/frame_sequences/<video>/frames/`) plus `frame_metadata.jsonl` to `RPPGPipeline.process_frames()` at the stage-1 sample rate (30 fps — matches the dataset extraction rate; 10 fps decimation had no anti-alias filter and performed below chance). rPPG no longer re-gates on blur/brightness (stage 1 did) but still runs MediaPipe per frame; `features=None` handling (INCONCLUSIVE, exit 3) is unchanged. If stage 1 fails or yields no frames, `run_pipeline.py` falls back to `RPPGPipeline.process_video()` (direct video read; `input_mode` in the result JSON records which path ran).
- **RandomForest cross-check** is an optional side path; the final verdict comes exclusively from the quantum stage. Note: as of 2026-08-19 the 20-feature pickle is incompatible with the 10-feature guard in `run_pipeline.py`, so the cross-check currently reports `skipped` (stale guard, not a functional check).
- **rPPG classifier trust:** `output/rppg/` must remain write-protected; `rppg_classifier.pkl` is `pickle.load`-ed by `run_pipeline.py` (arbitrary-code risk if replaced). Never move to shared hosting as-is.
- **QAOA selection weights:** the selection objective uses supervised discrimination weights (`qaoa._discrimination_weights`: sign-agnostic exact Mann-Whitney AUC strength `2*|AUC-0.5|`, deterministic, no sklearn). `QAOASelectionConfig.target_features` is 3. Keep `_mutual_info_weights` only as a documented alternative — do not reintroduce it into `select()`.
- **QAOA ansatz:** `_apply_qaoa` applies precomputed cost gates with `gamma` and X-mixer gates with `beta` separately. Regression guard: `test_beta_alive` in `quantum/tests.py`. Any refactor must keep beta alive and re-run `python -m quantum.tests` + `--all`.
- **Hamiltonian ≡ classical cost:** `_cost_terms` reproduces `_classical_cost` exactly (verified ~5.7e-14 on real data) and `pipeline.py` hard-asserts `error < 1e-6`.
- **Torch-native simulators (2026-08-19):** the default QAOA and VQC circuit backends are the project's exact complex128 statevector simulators (`qaoa_sim.QAOASimulator`, `vqc.QuantumLayerTorch`) — ~20× faster than PennyLane and CPU-process-safe. PennyLane paths remain behind `device="pennylane"` / `qnode_impl="pennylane"` for cross-verification; `test_qaoa_sim_matches_pennylane` and `test_torch_layer_matches_pennylane` pin them to ≤1e-6/1e-5. The torch head still runs on CUDA when available (`vqc.resolve_device()`).
- **rPPG needs MediaPipe Face Landmarker**; the model auto-downloads on first run (internet required). Extraction can alternatively use GPU-accelerated YuNet face detection (`--gpu`). In `frame/`, only `yolov8n-face-lindevs.pt` auto-downloads; missing other presets raise `FileNotFoundError`.

## Docs

- `Docs/RMTT_report.md` / `.pdf` — RMTT semester project report (architecture, methodology, deliverables)
- `Docs/quantum_layer_guide.md` — deep dive: QAOA Hamiltonian, simulators, VQC, evaluation, frozen-baseline results
- `Docs/Key_Findings_Contributions_Significance.md` — honest findings (incl. the negative result), contributions, significance
- `Docs/DEEPFAKE_KYC_SEQUENTIAL_REMEDIATION_PLAN.md` — 24-phase severity-ordered remediation roadmap (phases 1A–1C status in this README)
- `Docs/problems.md` — ranked problem analysis of the current evaluation

## Team

- **Jhashank** - rPPG-to-classifier pipeline, quantum model (Stage 3)
- **Sumit** - input-to-ROI pipeline (data, frames, face detection) (Stage 1)
- **Aswin** - preprocessing policy, evaluation, integration (Stage 2)
