# DeepFake Video Detection System for Video KYC (rPPG + Quantum ML)

Detection of deepfake videos for video KYC in financial systems using
**rPPG (remote photoplethysmography)** as the primary physiological evidence layer and
**quantum machine learning** as the final decision layer, designed for low-resolution videos.

## Architecture (Three-Stage Pipeline)

```
Input KYC video
  -> Stage 1: Frame sampling + quality filtering (WORKING/frame)
  -> Stage 2: rPPG signal extraction + feature engineering (WORKING/RPPG)
  -> Stage 3: Hybrid quantum classifier (WORKING/quantum)
  -> Decision output: REAL / FAKE / UNCERTAIN
```

The end-to-end flow is orchestrated by `WORKING/run_pipeline.py`:

```
frames  ->  rPPG  ->  quantum  ->  final verdict (REAL / FAKE / UNCERTAIN)
```

## Repository Layout

| Path | Description |
|---|---|
| `WORKING/` | Active project root (all three pipeline stages + end-to-end runner) |
| `WORKING/frame/` | Stage 1 - frame sampling, YOLO face detection, quality filtering (has its own `app/` + `requirements.txt`) |
| `WORKING/RPPG/` | Stage 2 - MediaPipe face ROIs -> POS/CHROM pulse -> 8 physiological features (+ `rppg-pipeline/`, `requirements.txt`) |
| `WORKING/quantum/` | Stage 3 - QAOA feature selection -> hybrid VQC -> P(real) -> verdict (run via `python -m quantum.pipeline`) |
| `WORKING/run_pipeline.py` | End-to-end orchestrator: frames -> rPPG -> quantum -> verdict |
| `WORKING/output/` | Global regenerated-artifacts root (all `output/` dirs untracked) |
| `WORKING/output/frames/` | Stage 1: `frame_sequences/`, `frame_extraction_*.json(l)`, `docs/` |
| `WORKING/output/rppg/` | Stage 2: `dataset_features.csv`, `rppg_classifier.pkl` + metadata, plots |
| `WORKING/output/quantum/` | Stage 3: `data.npz`, `qaoa_selection.json`, `feature_scaler.json`, `hybrid_vqc.pt`, metrics, plots |
| `WORKING/output/pipeline/` | `run_pipeline.py` result JSON |
| `frontend/` | Web UI: React + Vite frontend + stdlib-only API server |
| `Docs/` | Project docs, report drafts, research papers |
| `FF++/` | FaceForensics++ dataset (gitignored) |

## Components

### 1. Frame Sampling & Quality (`WORKING/frame/`)

Modular face pipeline for video sources: frame sampling at a controlled
FPS, face detection (YOLO), quality assessment (blur / dark / overexposed / no-face / face-too-small / extreme-pose rules).
Outputs accepted JPEGs + per-frame metadata JSONL for the rPPG stage.

```bash
# Standalone frame stage
python app/main.py --source test.mp4 --save-metadata
# Or extraction-only mode
python app/extract_frames.py --source test.mp4 --sample-fps 10 --save-quality-examples
```

### 2. rPPG Pipeline (`WORKING/RPPG/`)

POS/CHROM pulse reconstruction from facial ROIs (left cheek, right cheek, forehead)
and an 8-feature physiological vector per video: heart rate, SNR, PRV, spectral
entropy, MAD, signal quality index, and inter-region correlations. Features are
persisted with labels (1 = fake, 0 = real) in `WORKING/output/rppg/dataset_features.csv`,
which is the direct data source for the quantum layer.

```bash
# From WORKING/RPPG/ - rebuild output/rppg/dataset_features.csv
python rppg-pipeline/extract_dataset_features.py

# Train rPPG classifier (side path, not used for final verdict)
python rppg-pipeline/train_classifier.py
```

### 3. Quantum Model (`WORKING/quantum/`)

Hybrid classical-quantum decision stage built with PennyLane + PyTorch. It consumes
the rPPG 8-feature vector **directly** (same names/order as `RPPGFeatures.feature_names()`);
no synthetic data is generated anywhere in the pipeline.

- **Data** - `data.py` builds `output/quantum/data.npz` from the real labeled rPPG feature
  table `output/rppg/dataset_features.csv` (rPPG label 1 = fake is flipped to the quantum
  convention LABEL_REAL = 1, LABEL_FAKE = 0) with a stratified train/val/test split
- **QAOA feature selection** - `qaoa.py` selects the most informative rPPG features
  using a cost Hamiltonian (mutual information weights + correlation redundancy
  penalty + cardinality constraint), optimized with COBYLA
- **Hybrid VQC** - `vqc.py`: angle-encoded features into a parameterized quantum
  circuit (`StronglyEntanglingLayers`) feeding a small classical head
- **Evaluation** - `evaluate.py`: accuracy / precision / recall / F1 / AUC-ROC / ECE,
  KYC-friendly decision bins (real / uncertain / fake), ROC and confusion-matrix
  plots, plus classical baselines (MLP, RandomForest) on the same features
- **Orchestration** - `pipeline.py` drives the full training flow and exposes
  `predict_features()` as the single inference entry point used by `run_pipeline.py`

```bash
# From WORKING/: build data, QAOA selection, train VQC, evaluate, baselines
python -m quantum.pipeline --all
```

## Install

```bash
# Stage 1 (frame) - CUDA PyTorch
pip install -r WORKING/frame/requirements.txt

# Stage 2 (rPPG) - MediaPipe, OpenCV, scipy, etc.
pip install -r WORKING/RPPG/requirements.txt

# Stage 3 (quantum) - PennyLane, PyTorch, scikit-learn
pip install pennylane torch numpy scikit-learn scipy matplotlib
```

## Run End-to-End

```bash
# From WORKING/
python run_pipeline.py --source path/to/video.mp4 --method POS --out result.json
```

Requires pre-trained quantum artifacts (`output/quantum/qaoa_selection.json`, `hybrid_vqc.pt`).
If missing, run once from `WORKING/`:
```bash
python -m quantum.pipeline --all
```

## Frontend (Web UI)

```bash
# From frontend/
python server.py          # Backend API on http://127.0.0.1:8000
npm install
npm run dev               # React dev server on http://localhost:5173
```

Upload a KYC video, watch the three-stage pipeline run live via SSE,
and read the verdict with its evidence dossier (accepted frames, physiological
features, quantum probabilities and plots, the pulse waveform).

## Verified Constraints (do not break)

- **No synthetic data.** The quantum layer consumes only the real rPPG feature table `output/rppg/dataset_features.csv` (currently 18 labeled rows). Never reintroduce a generator or a transform/bridge layer.
- **Feature contract:** `FEATURE_NAMES` in `quantum/config.py` must stay identical in name AND order to `RPPGFeatures.feature_names()` in `RPPG/rppg/features.py` (duplicated on purpose; `data.py`, `qaoa.py`, and `pipeline.py` all index by it). Keep the two lists in sync.
- **Label conventions differ per stage — do not unify:**
  - rPPG CSV: `1 = fake, 0 = real`
  - quantum: `LABEL_REAL = 1`, `LABEL_FAKE = 0`; `quantum/data.py` flips (`y = 1 - csv_label`)
  - rPPG RandomForest cross-check in `run_pipeline.py`: `1 = DEEPFAKE`
- **Gitignored artifacts:** `*.csv`, `*.json`, `*.pkl`, `*.mp4`, and all `output/` dirs are untracked — `dataset_features.csv`, `output/quantum/*`, and the trained models will not appear in `git status`. Regenerating them is normal.
- **rPPG returns `features=None`** when usable frames < `min_usable_frames` (48); `run_pipeline.py` then emits INCONCLUSIVE and exits 3. New code must handle `None`.
- **Stage 1 feeds stage 2.** `run_pipeline.py` hands the frame stage's accepted JPEGs (`output/frames/frame_sequences/<video>/frames/`) plus `frame_metadata.jsonl` to `RPPGPipeline.process_frames()` at the stage-1 sample rate (10 fps). rPPG no longer re-gates on blur/brightness (stage 1 did) but still runs MediaPipe per frame; `features=None` handling (INCONCLUSIVE, exit 3) is unchanged.

## Team

- **Jhashank** - rPPG-to-classifier pipeline, quantum model (Stage 3)
- **Sumit** - input-to-ROI pipeline (data, frames, face detection) (Stage 1)
- **Aswin** - preprocessing policy, evaluation, integration (Stage 2)