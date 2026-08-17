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
| `WORKING/RPPG/`            | Stage 2 - MediaPipe face ROIs -> POS/CHROM pulse -> 10 physiological features (+`rppg-pipeline/`, `requirements.txt`) |
| `WORKING/quantum/`         | Stage 3 - QAOA feature selection -> hybrid VQC -> P(real) -> verdict (run via`python -m quantum.pipeline`)              |
| `WORKING/run_pipeline.py`  | End-to-end orchestrator: frames -> rPPG -> quantum -> verdict                                                             |
| `WORKING/output/`          | Global regenerated-artifacts root (all`output/` dirs untracked)                                                         |
| `WORKING/output/frames/`   | Stage 1:`frame_sequences/`, `frame_extraction_*.json(l)`, `docs/`                                                   |
| `WORKING/output/rppg/`     | Stage 2:`dataset_features.csv`, `rppg_classifier.pkl` + metadata, plots                                               |
| `WORKING/output/quantum/`  | Stage 3:`data.npz`, `qaoa_selection.json`, `feature_scaler.json`, `hybrid_vqc.pt`, metrics, plots                 |
| `WORKING/output/pipeline/` | `run_pipeline.py` result JSON                                                                                           |
| `frontend/`                | Web UI: React + Vite frontend + stdlib-only API server (port 8000)                                                        |
| `Docs/`                    | Project docs, report drafts, research papers                                                                              |
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
and a 10-feature physiological vector per video: heart rate, SNR, PRV, spectral
entropy, MAD, signal quality index, and inter-region correlations (full list in
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
extraction (3445 rows @30 fps: 1883 real / 1562 fake) was rebuilt 2026-08-15.
The legacy `archive (1)` CSV layout is absent on disk.

### 3. Quantum Model (`WORKING/quantum/`)

Hybrid classical-quantum decision stage built with PennyLane + PyTorch. It consumes
the rPPG 10-feature vector **directly** (same names/order as `RPPGFeatures.feature_names()`);
no synthetic data is generated anywhere in the pipeline.

- **Data** - `data.py` builds `output/quantum/data.npz` from the real labeled rPPG feature
  table `output/rppg/dataset_features.csv` (3445 rows: 1883 real / 1562 fake, DFDC + FF++).
  rPPG label `1 = fake` is flipped to quantum convention `LABEL_REAL = 1, LABEL_FAKE = 0`.
  HR-plausibility filter (30–220 BPM, non-finite rejection) drops implausible rows at
  build time. Subject-grouped random split (seeded, per-clip for DFDC, per-clip for FF++
  with first-id token grouping).
- **QAOA feature selection** - `qaoa.py` selects the 3 most informative rPPG features using
  a cost Hamiltonian with **supervised discrimination weights** (`qaoa._discrimination_weights`:
  sign-agnostic exact Mann-Whitney AUC strength `2*|AUC-0.5|`, deterministic, no sklearn)
  plus correlation redundancy penalty and cardinality constraint, optimized with COBYLA.
  4 parallel restarts via `ProcessPoolExecutor`.
- **Hybrid VQC** - `vqc.py`: angle-encoded features into a parameterized quantum
  circuit (`StronglyEntanglingLayers`) feeding a small classical head. Training uses
  cosine-annealed LR, gradient clipping, and early stopping on validation loss with
  restore of the best-validation checkpoint. QNode runs in **broadcast mode**
  (batched rows through PennyLane, ~28x training / ~156x inference vs per-row loop).
  CUDA when available; PennyLane QNode on `default.qubit` + `backprop`.
- **Evaluation** - `evaluate.py`: accuracy / precision / recall / F1 / AUC-ROC / ECE,
  KYC-friendly decision bins (real / uncertain / fake), ROC, confusion-matrix and
  calibration-curve plots. **StratifiedKFold (5-fold) cross-validation** (mean ± std,
  balanced accuracy) for the VQC and every baseline. Classical baselines:
  RandomForest, MLP, LogisticRegression, calibrated LinearSVC, GaussianNB, XGBoost.
- **Orchestration** - `pipeline.py` drives the full training flow and exposes
  `predict_features()` as the single inference entry point used by `run_pipeline.py`.
  Hard-asserts Hamiltonian ≡ classical cost (`error < 1e-6`).

```bash
# From WORKING/: build data, QAOA selection, train VQC, evaluate, baselines
python -m quantum.pipeline --all
# Self-checks: beta-alive ansatz, Hamiltonian≡classical cost, feature-contract sync
python -m quantum.tests
```

**Current baseline (post-leakage-fix, 2026-08-15, 3445 rows):**
QAOA selects `['peak_prominence', 'left_right_cheek_correlation', 'spectral_entropy']`
(seed 44, cost −1.363). Grouped 5-fold CV AUC 0.517; test acc 0.547 / AUC 0.503.
Classical baselines (LR/GNB/XGB) test AUC 0.565–0.569. Decision bins remain 100%
UNCERTAIN at the 0.3/0.7 thresholds. Per-feature |AUC−0.5| ≤ ~0.06 confirms the
rPPG features themselves carry limited class signal — Phase 2 rPPG method/ROI
probe is the next lever.

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

- **No synthetic data.** The quantum layer consumes only the real rPPG feature table `output/rppg/dataset_features.csv` (3445 rows @30 fps: 1883 real / 1562 fake — DFDC archive + FF++). Never reintroduce a generator or a transform/bridge layer.
- **Feature contract:** `FEATURE_NAMES` in `quantum/config.py` must stay identical in name AND order to `RPPGFeatures.feature_names()` in `RPPG/rppg/features.py` (duplicated on purpose; `data.py`, `qaoa.py`, and `pipeline.py` all index by it). Keep the two lists in sync.
- **Label conventions differ per stage — do not unify:**
  - rPPG CSV: `1 = fake, 0 = real`
  - quantum: `LABEL_REAL = 1`, `LABEL_FAKE = 0`; `quantum/data.py` flips (`y = 1 - csv_label`)
  - rPPG RandomForest cross-check in `run_pipeline.py`: `1 = DEEPFAKE`
- **Gitignored artifacts:** `*.csv`, `*.json`, `*.pkl`, `*.mp4`, and all `output/` dirs are untracked — `dataset_features.csv`, `output/quantum/*`, and the trained models will not appear in `git status`. Regenerating them is normal.
- **rPPG returns `features=None`** when usable frames < `min_usable_frames` (48); `run_pipeline.py` then emits INCONCLUSIVE and exits 3. New code must handle `None`.
- **Stage 1 feeds stage 2.** `run_pipeline.py` hands the frame stage's accepted JPEGs (`output/frames/frame_sequences/<video>/frames/`) plus `frame_metadata.jsonl` to `RPPGPipeline.process_frames()` at the stage-1 sample rate (30 fps — matches the dataset extraction rate; 10 fps decimation had no anti-alias filter and performed below chance). rPPG no longer re-gates on blur/brightness (stage 1 did) but still runs MediaPipe per frame; `features=None` handling (INCONCLUSIVE, exit 3) is unchanged. If stage 1 fails or yields no frames, `run_pipeline.py` falls back to `RPPGPipeline.process_video()` (direct video read; `input_mode` in the result JSON records which path ran).
- **RandomForest cross-check** is an optional side path; the final verdict comes exclusively from the quantum stage.
- **rPPG classifier trust:** `output/rppg/` must remain write-protected; `rppg_classifier.pkl` is `pickle.load`-ed by `run_pipeline.py` (arbitrary-code risk if replaced). Never move to shared hosting as-is.
- **QAOA selection weights (2026-08-15):** the selection objective uses supervised discrimination weights (`qaoa._discrimination_weights`: sign-agnostic exact Mann-Whitney AUC strength `2*|AUC-0.5|`, deterministic, no sklearn) instead of the old unsupervised mutual-information weights. `QAOASelectionConfig.target_features` is 3. Keep `_mutual_info_weights` only as a documented alternative — do not reintroduce it into `select()`.
- **QAOA ansatz:** `_apply_qaoa` applies precomputed cost gates with `gamma` and X-mixer gates with `beta` separately. Regression guard: `test_beta_alive` in `quantum/tests.py`. Any refactor must keep beta alive and re-run `python -m quantum.tests` + `--all`.
- **Hamiltonian ≡ classical cost:** `_cost_terms` reproduces `_classical_cost` exactly (verified 7.1e-15 on real data) and `pipeline.py` hard-asserts `error < 1e-6`.
- **rPPG needs MediaPipe Face Landmarker**; the model auto-downloads on first run (internet required). In `frame/`, only `yolov8n-face-lindevs.pt` auto-downloads; missing other presets raise `FileNotFoundError`.

## Team

- **Jhashank** - rPPG-to-classifier pipeline, quantum model (Stage 3)
- **Sumit** - input-to-ROI pipeline (data, frames, face detection) (Stage 1)
- **Aswin** - preprocessing policy, evaluation, integration (Stage 2)
