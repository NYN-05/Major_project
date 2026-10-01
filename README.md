# Deepfake Video Detection for KYC (rPPG + Visual + Quantum ML)

[![GitHubTree](https://img.shields.io/badge/GitHubTree-Major__project-blue?style=flat-square)](https://githubtree.mgks.dev/repo/NYN-05/Major_project/main/?ref=badge)

Detection of deepfake videos for video KYC in financial systems using
**rPPG (remote photoplethysmography)** as the physiological evidence layer,
**visual features** (ResNet50 + handcrafted) as the complementary appearance layer,
and **hybrid quantum-classical ML** as the final decision layer, designed for low-resolution videos.

## Architecture (Four-Stage Pipeline)

```
Input KYC video
  → Stage 1: Frame sampling + quality filtering    (WORKING/frame/)
  → Stage 2: rPPG signal extraction + features     (WORKING/RPPG/)
  → Stage 3: Visual feature extraction             (WORKING/visual/)
  → Stage 4: QAOA selection → hybrid VQC verdict   (WORKING/quantum/)
  → Decision: REAL / FAKE / UNCERTAIN
```

End-to-end orchestrator (`WORKING/run_pipeline.py`):

```
frames → rPPG → visual → quantum (fused) → verdict (REAL / FAKE / UNCERTAIN)
```

Only **fused mode** (rPPG + visual) is supported. No `rppg_only`, `visual_only`, or other feature set modes.
Only **DFDC dataset** is supported for training. No FaceForensics++ (FF++).

## Repository Layout

| Path                         | Description                                                                                                               |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| `WORKING/`                 | Active project root (all four pipeline stages + end-to-end runner)                                                       |
| `WORKING/frame/`           | Stage 1 - frame sampling, YOLO face detection, quality filtering (has its own `app/` + `requirements.txt`)             |
| `WORKING/RPPG/`            | Stage 2 - MediaPipe face ROIs → POS/CHROM pulse → 24 physiological features (+ `rppg-pipeline/`, `requirements.txt`) |
| `WORKING/visual/`          | Stage 3 - ResNet50 + handcrafted visual features from face crops (39 features)                                          |
| `WORKING/quantum/`         | Stage 4 - QAOA feature selection → hybrid VQC → P(real) → verdict (run via `python -m quantum.pipeline`)        |
| `WORKING/run_pipeline.py`  | End-to-end orchestrator: frames → rPPG → visual → quantum → verdict                                                     |
| `WORKING/output/`          | Global regenerated-artifacts root (all `output/` dirs untracked)                                                         |
| `WORKING/output/frames/`   | Stage 1: `frame_sequences/`, `frame_extraction_*.json(l)`, `docs/`                                                   |
| `WORKING/output/rppg/`     | Stage 2: `dataset_features.csv`, `rppg_classifier.pkl` + metadata, plots                                               |
| `WORKING/output/visual/`   | Stage 3: `visual_features.csv`, `fused_features.csv`                                                                  |
| `WORKING/output/quantum/`  | Stage 4: `data_fused.npz`, `qaoa_selection_fused.json`, `feature_scaler_fused.json`, `hybrid_vqc_fused.pt`, metrics, plots |
| `WORKING/output/pipeline/` | `run_pipeline.py` result JSON                                                                                           |
| `frontend/`                | Web UI: React + Vite frontend + stdlib-only API server (port 8000)                                                        |
| `Docs/`                    | Project docs: RMTT semester report, key findings, remediation plan, problem analysis, bottleneck diagnosis                   |
| `Scrape/`                  | Dev scratch: tests, temp scripts, debug artifacts (gitignored)                                                            |

## Current Performance (Fused Mode, DFDC Only)

| Metric | VQC Test | VQC 5-Fold CV | Best Classical (LR) |
|--------|----------|---------------|---------------------|
| Accuracy | TBD | TBD | TBD |
| **Balanced Accuracy** | **TBD** | **TBD ± TBD** | **TBD** |
| Specificity (FAKE recall) | **TBD** | **TBD ± TBD** | TBD |
| AUC-ROC | TBD | TBD ± TBD | TBD |
| Decision Bins | TBD | — | — |

*Artifacts not yet regenerated — run `python -m quantum.pipeline --all` from `WORKING/` after building fused features to produce current metrics.*

## Components

### 1. Frame Sampling & Quality (`WORKING/frame/`)

Modular face pipeline for video sources: frame sampling at a controlled
FPS, face detection (YOLO), quality assessment (blur / dark / overexposed / no-face / face-too-small / extreme-pose rules).
**Quality thresholds relaxed** for short/low-quality videos: blur 3.0, dark 30, bright 235, min face area 0.002, min seq len 32, conf 0.25.
Outputs accepted JPEGs + per-frame metadata JSONL for the rPPG stage.

```bash
# Standalone frame stage (from WORKING/frame/)
python app/pipeline.py --source test.mp4 --save-metadata
```

(`app/pipeline.py` is the single CLI entry point; legacy `app/main.py` / `app/extract_frames.py` no longer exist.)

### 2. rPPG Pipeline (`WORKING/RPPG/`)

POS/CHROM pulse reconstruction from facial ROIs (left cheek, right cheek, forehead)
and a **24-feature** physiological vector per video (20 base + 4 Phase 4 probe):
heart rate, SNR, PRV, spectral entropy, MAD, signal quality index,
inter-region correlations, pulse morphology (peak width, dicrotic notch),
inter-ROI phase lag / pulse-transit-time proxy, motion contamination,
spectral/statistical probes (spectral flatness, centroid, kurtosis, phase coherence).
**Quality thresholds relaxed** for short/low-quality videos: min usable frames 24, min SQI 0.05.
Features are persisted with labels (1 = fake, 0 = real) in
`WORKING/output/rppg/dataset_features.csv`, the direct data source for the quantum layer.

```bash
# From WORKING/RPPG/ - rebuild output/rppg/dataset_features.csv
#   --workers 0          = all CPU cores
#   --gpu                = GPU-accelerated face detection (YuNet ONNX CUDA)
python rppg-pipeline/extract_dataset_features.py --workers 0 --gpu

# Train rPPG classifier (side path, not used for final verdict)
python rppg-pipeline/train_classifier.py
```

Feature source: **DFDC only** (`archive/DFDC_Dataset` via `DFDC_DATASET_PATH` env var).
No FaceForensics++ (FF++) support.

### 3. Visual Pipeline (`WORKING/visual/`)

ResNet50 (ImageNet pre-trained) + handcrafted features from face crops saved by stage 1.
Extracts **39 features**: 16 deep (PCA-reduced ResNet50 GAP) + 10 LBP + 4 GLCM texture + 6 color + 3 DCT frequency.
Fuses with rPPG features to create the **fused feature set** (63 features) used for training.

```bash
# From WORKING/visual/ - extract visual features from stage-1 frames
python pipeline.py --frames-root output/frames/frame_sequences --rppg-csv ../output/rppg/dataset_features.csv --fuse --create-splits
```

### 4. Quantum Model (`WORKING/quantum/`)

Hybrid classical-quantum decision stage built with PennyLane + PyTorch. It consumes
the **fused (rPPG + visual) 63-feature vector directly** (same names/order as `RPPGFeatures.feature_names()` + `VISUAL_FEATURE_NAMES`);
no synthetic data is generated anywhere in the pipeline.

- **Data** - `data.py` builds `output/quantum/data_fused.npz` from the real labeled fused feature
  table `output/visual/fused_features.csv`. 
  rPPG label `1 = fake` is flipped via the explicit, tested conversion
  `csv_to_quantum_label()` to the quantum convention `LABEL_REAL = 1, LABEL_FAKE = 0`
  (Phase 1A of the remediation plan). HR-plausibility filter (30–220 BPM, non-finite rejection)
  drops implausible rows at build time. Subject-grouped random split (seeded, per-clip for DFDC),
  persisted to `split_manifest_fused.json`.
- **QAOA feature selection** - `qaoa.py` selects the most informative fused features using
  a cost Hamiltonian with **supervised discrimination weights** (`qaoa._discrimination_weights`:
  sign-agnostic exact Mann-Whitney AUC strength `2*|AUC-0.5|`, deterministic, no sklearn)
  plus correlation redundancy penalty and cardinality constraint, optimized with COBYLA.
  8 parallel restarts via `ProcessPoolExecutor`. The circuit runs on the project's own
  **exact torch-native statevector simulator** (`qaoa_sim.QAOASimulator`, complex128;
  ~0.3–0.5 ms vs ~5.6 s per PennyLane call) and is cross-verified
  against PennyLane by the test suite.
- **Hybrid VQC** - `vqc.py`: angle-encoded features into a parameterized quantum
  circuit (`StronglyEntanglingLayers`) feeding a small classical head. The default
  quantum layer is the exact torch-native `QuantumLayerTorch` simulator (complex128,
  batched, same `weights` shape as the PennyLane layer so checkpoints are
  interchangeable); the legacy PennyLane QNode path remains behind `qnode_impl="pennylane"`
  for cross-verification. Training uses class-balanced focal loss (Phase 1B), cosine-annealed
  LR, gradient clipping, and early stopping on validation loss with restore of the
  best-validation checkpoint. CUDA head when available.
- **Evaluation** - `evaluation.py`: accuracy / precision / recall / F1 / AUC-ROC / ECE,
  KYC-friendly decision bins (real / uncertain / fake), ROC, confusion-matrix and
  calibration-curve plots. **StratifiedKFold (5-fold) cross-validation** (mean ± std,
  balanced accuracy) for the VQC and every baseline. Classical baselines:
  RandomForest, MLP, LogisticRegression, calibrated LinearSVC, GaussianNB, XGBoost.
- **Orchestration** - `pipeline.py` drives the full training flow and exposes
  `predict_features()` as the single inference entry point used by `run_pipeline.py`.
  Hard-asserts Hamiltonian ≡ classical cost (`error < 1e-6`).
- **Sweep harness** - `sweep.py` runs crash-safe hyperparameter sweeps with a leaderboard by AUC.

```bash
# From WORKING/: build data, QAOA selection, train VQC, evaluate, baselines (fused mode only)
python -m quantum.pipeline --all
# Self-checks (10/10): beta-alive ansatz, Hamiltonian≡classical cost,
# feature-contract sync, split determinism, grouping, sim cross-verification,
# checkpoint compat, label conversion
python -m quantum.tests
```

## Install

```bash
# Stage 1 (frame) - CUDA PyTorch, YOLO
pip install -r WORKING/frame/requirements.txt

# Stage 2 (rPPG) - MediaPipe, OpenCV, scipy, etc.
pip install -r WORKING/RPPG/requirements.txt

# Stage 3 (visual) - torchvision, scikit-image for handcrafted features
pip install torchvision scikit-image

# Stage 4 (quantum) - PennyLane, PyTorch, scikit-learn, scipy, matplotlib
pip install pennylane torch numpy scikit-learn scipy matplotlib

# Frontend (optional)
cd frontend && npm install
```

## Run End-to-End (Inference)

```bash
# From WORKING/
python run_pipeline.py --source path/to/video.mp4 --method POS --out result.json
```

Requires pre-trained quantum artifacts for **fused mode** (`output/quantum/qaoa_selection_fused.json`, `hybrid_vqc_fused.pt`, `feature_scaler_fused.json`).
If missing, run once from `WORKING/`:

```bash
# Build fused training data first (requires DFDC_DATASET_PATH env var)
export DFDC_DATASET_PATH=/path/to/DFDC_Dataset
python rppg-pipeline/extract_dataset_features.py --workers 0 --gpu
python visual/pipeline.py --frames-root output/frames/frame_sequences --rppg-csv ../output/rppg/dataset_features.csv --fuse --create-splits
# Then train quantum model
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
idle → selected (preview + metadata + Start button)
     → running  (7-stage pipeline + live panel + creeping progress)
     → done     (Verdict gauge, Insights, Signal canvas, Quantum flow, FrameSamples)
     → error    (inline banner)
```

Theme toggle persists `rppgqc.theme` in localStorage; respects `prefers-reduced-motion`. Responsive: 1920 → 375 px, no overflow.

## Verified Constraints (do not break)

- **No synthetic data.** The quantum layer consumes only the real fused feature table `output/visual/fused_features.csv`. Never reintroduce a generator or a transform/bridge layer.
- **Feature contract:** `FUSED_FEATURE_NAMES` in `quantum/config.py` (63 features = 24 rPPG + 39 visual) must stay identical in name AND order to `RPPGFeatures.feature_names()` + `VISUAL_FEATURE_NAMES`. Keep the two lists in sync; `test_feature_contract_sync` guards it.
- **Only fused mode supported.** No `rppg_only`, `visual_only`, `rppg_base`, `rppg_cross_roi`, or ablation feature sets.
- **Only DFDC dataset supported.** No FaceForensics++ (FF++). `DFDC_DATASET_PATH` env var must point to `archive/DFDC_Dataset`.
- **Label conventions differ per stage — do not unify:**
  - rPPG CSV: `1 = fake, 0 = real`
  - quantum: `LABEL_REAL = 1`, `LABEL_FAKE = 0`; `quantum/data.py` flips via the tested `csv_to_quantum_label()` (`y = 1 - csv_label` lives only inside that function)
  - rPPG RandomForest cross-check in `run_pipeline.py`: `1 = DEEPFAKE`
- **Gitignored artifacts:** `*.csv`, `*.json`, `*.pkl`, `*.mp4`, and all `output/` dirs are untracked — `dataset_features.csv`, `output/visual/fused_features.csv`, `output/quantum/*`, and the trained models will not appear in `git status`. Regenerating them is normal.
- **rPPG returns `features=None`** when usable frames < `min_usable_frames` (24); `run_pipeline.py` then emits INCONCLUSIVE and exits 3. New code must handle `None`.
- **Stage 1 feeds stage 2 & 3.** `run_pipeline.py` hands the frame stage's accepted JPEGs (`output/frames/frame_sequences/<video>/frames/`) plus `frame_metadata.jsonl` to `RPPGPipeline.process_frames()` at the stage-1 sample rate (30 fps) and to `visual.extractor.compute_visual_features()` for face crops. rPPG no longer re-gates on blur/brightness (stage 1 did) but still runs MediaPipe per frame; `features=None` handling (INCONCLUSIVE, exit 3) is unchanged. If stage 1 fails or yields no frames, `run_pipeline.py` falls back to `RPPGPipeline.process_video()` (direct video read; `input_mode` in the result JSON records which path ran). Standalone RPPG scripts keep using `process_video`. `run_pipeline.py --signal-out <path>` writes the decimated stage-2 waveform JSON (same schema as `frontend/dump_signal.py`, which is retained only as a standalone utility).
- **RandomForest cross-check** is an optional side path; the final verdict comes exclusively from the quantum stage.
- **rPPG classifier trust:** `output/rppg/` must remain write-protected; `rppg_classifier.pkl` is `pickle.load`-ed by `run_pipeline.py` (arbitrary-code risk if replaced). Never move to shared hosting as-is.
- **QAOA selection weights:** the selection objective uses supervised discrimination weights (`qaoa._discrimination_weights`: sign-agnostic exact Mann-Whitney AUC strength `2*|AUC-0.5|`, deterministic, no sklearn). `QAOASelectionConfig.target_features` is 3. Keep `_mutual_info_weights` only as a documented alternative — do not reintroduce it into `select()`.
- **QAOA ansatz:** `_apply_qaoa` applies precomputed cost gates with `gamma` and X-mixer gates with `beta` separately. Regression guard: `test_beta_alive` in `quantum/tests.py`. Any refactor must keep beta alive and re-run `python -m quantum.tests` + `--all`.
- **Hamiltonian ≡ classical cost:** `_cost_terms` reproduces `_classical_cost` exactly (verified ~1e-14 on real data) and `pipeline.py` hard-asserts `error < 1e-6`.
- **Torch-native simulators (default):** the default QAOA and VQC circuit backends are the project's exact complex128 statevector simulators (`qaoa_sim.QAOASimulator`, `vqc.QuantumLayerTorch`) — ~20× faster than PennyLane and CPU-process-safe. PennyLane paths remain behind `device="pennylane"` / `qnode_impl="pennylane"` for cross-verification; `test_qaoa_sim_matches_pennylane` and `test_torch_layer_matches_pennylane` pin them to ≤1e-6/1e-5.
- **rPPG needs MediaPipe Face Landmarker**; the model auto-downloads on first run (internet required). Extraction can alternatively use GPU-accelerated YuNet face detection (`--gpu`). In `frame/`, only `yolov8n-face-lindevs.pt` auto-downloads; missing other presets raise `FileNotFoundError`.

## Docs

- `Docs/RMTT_report.md` / `.pdf` — RMTT semester project report (historical snapshot)
- `Docs/Key_Findings_Contributions_Significance.md` — honest findings (incl. negative result), contributions, significance
- `Docs/IMMEDIATE_FIX_PLAN.md` — severity-ordered remediation roadmap
- `Docs/problems.md` — ranked problem analysis
- `Docs/Bottleneck_Diagnosis.md` — root cause diagnosis
- `changes.md` — historical development log (Phases 1–10, superseded)

## Team

- **Jhashank** - rPPG-to-classifier pipeline, quantum model (Stage 3)
- **Sumit** - input-to-ROI pipeline (data, frames, face detection) (Stage 1)
- **Aswin** - preprocessing policy, evaluation, integration (Stage 2)