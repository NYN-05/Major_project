# Copilot Instructions

## Project Context

This repository detects deepfake KYC videos using a four-stage Python pipeline:

```text
video
  -> WORKING/frame       sampled frames, YOLO face detection, quality gates
  -> WORKING/RPPG        MediaPipe ROIs, POS/CHROM, 24 physiological features
  -> WORKING/visual      ResNet50 + handcrafted features, 39 visual features
  -> WORKING/quantum     fused features, QAOA selection, hybrid VQC, verdict
```

`WORKING/run_pipeline.py` is the end-to-end inference orchestrator. The
`frontend/` directory contains a React/Vite UI and a standard-library Python
API server that launches the existing pipeline as a subprocess.

## Source of Truth

- Read [`AGENTS.md`](../AGENTS.md) before changing repository behavior.
- Use [`README.md`](../README.md) for the supported end-to-end workflow.
- Use [`WORKING/quantum/README.md`](../WORKING/quantum/README.md) for the
  quantum data flow and artifact contract.
- Use [`frontend/README.md`](../frontend/README.md) for the web/API contract.
- Record every code change in [`changes.md`](../changes.md) with date, time,
  affected area, and a short description.

## Build, Test, and Lint Commands

Run Python pipeline commands from `WORKING/`, not the repository root.

```powershell
# Install the merged Python environment from the repository root
pip install -r requirements.txt

# Run the quantum regression/self-check suite
cd WORKING
python -m quantum.tests

# Run one quantum self-check directly
python -c "import quantum.tests as t; t.test_feature_contract_sync()"

# Build data, select features, train, evaluate, and run baselines
python -m quantum.pipeline --all

# Run end-to-end inference using existing fused artifacts
python run_pipeline.py --source path\to\video.mp4 --method POS --out result.json
```

`WORKING/quantum/tests.py` is the project's primary self-check runner; it is
not a pytest-discovered test module. The individual functions can be invoked
with the one-test pattern above. Some checks require `DFDC_DATASET_PATH` and
the generated fused artifacts.

For formatting and static checks, the configured tools are:

```powershell
# From the repository root or the relevant package directory
ruff check .
mypy WORKING
```

The repository also contains pytest configuration in `pyproject.toml`, but
the maintained quantum regression suite is run with `python -m quantum.tests`.

For the frontend:

```powershell
cd frontend
npm install
npm run build
npm run dev
npm run preview
```

`package.json` currently defines `dev`, `build`, `preview`, and `server`
scripts. There is no configured frontend test script; use the production build
as the baseline validation for UI changes.

## Running the Application

Start the API from `frontend/` with the Python environment that contains the
pipeline dependencies:

```powershell
python server.py
```

Start the Vite UI in another terminal:

```powershell
npm run dev
```

The API listens on `http://127.0.0.1:8000` and the UI on
`http://localhost:5173`. The Vite development server proxies `/api` requests
to the Python API.

## Architecture and Data Flow

The stages share files through `WORKING/output/`, which is regenerated and
gitignored:

1. Frame extraction writes accepted JPEGs and `frame_metadata.jsonl`.
2. The end-to-end runner passes those accepted frames to both the rPPG and
   visual stages. If extraction fails or produces no frames, it falls back to
   direct video processing for rPPG.
3. rPPG writes `output/rppg/dataset_features.csv`; visual extraction joins
   with it to write `output/visual/fused_features.csv`.
4. `quantum.data` filters and splits the real fused table, preserving a
   subject-grouped split manifest.
5. `quantum.scaling` fits the train-only scaler, `quantum.qaoa` selects three
   features, and `quantum.vqc` trains or loads the hybrid model.
6. `quantum.pipeline.predict_features()` replays the same scaler and selected
   feature indices during inference and returns `P(real)` plus the final
   decision bin.
7. The frontend server launches `WORKING/run_pipeline.py`, streams progress
   with SSE, and serves result JSON, waveform data, frames, and plots.

Do not bypass the fused table or create a separate transformation layer.
Inference must use the same feature ordering and preprocessing as training.

## Repository-Specific Contracts

- Only fused mode is supported: 24 rPPG features plus 39 visual features.
- Only the DFDC dataset is supported. Set `DFDC_DATASET_PATH` to
  `archive/DFDC_Dataset` when rebuilding training data.
- `FUSED_FEATURE_NAMES` in `WORKING/quantum/config.py` must remain identical
  in name and order to `RPPGFeatures.feature_names()` followed by
  `VISUAL_FEATURE_NAMES`.
- rPPG CSV labels are `1 = fake`, `0 = real`; quantum labels are
  `LABEL_REAL = 1`, `LABEL_FAKE = 0`. The conversion belongs only in
  `quantum.data.csv_to_quantum_label()`.
- The RandomForest result is an optional cross-check; the quantum stage owns
  the final verdict.
- Fewer than 24 usable rPPG frames produces `features=None`; the pipeline
  must return an inconclusive result and exit with status 3.
- The default QAOA and VQC circuit paths are the project’s exact torch-native
  complex128 simulators. PennyLane paths are retained for cross-verification.
- QAOA uses supervised Mann-Whitney-AUC discrimination weights and a target of
  three selected features. Do not restore the retired mutual-information
  selection path.
- `_apply_qaoa` must keep the beta mixer active, and the QAOA Hamiltonian must
  remain equivalent to the classical cost. Run `python -m quantum.tests` after
  changing either implementation.
- Keep `WORKING/output/rppg/` protected during inference. The
  `rppg_classifier.pkl` file is loaded with `pickle` and must be treated as
  untrusted serialized code.
- Preserve API upload validation, 200 MB limit, magic-byte checks, localhost
  CORS policy, two-job concurrency cap, path-traversal protection, TTL cleanup,
  and the 30-minute pipeline timeout.
- Store temporary scripts, probes, test artifacts, screenshots, and logs under
  [`Scrape/`](../Scrape/), not in the repository root or production folders.

## API Contract

The backend endpoints in `frontend/server.py` are:

```text
POST /api/detect
GET  /api/jobs/<id>
GET  /api/jobs/<id>/events
GET  /api/previous
GET  /api/health
GET  /api/artifacts?dir=<relative-output-directory>
GET  /api/files?rel=<relative-artifact-path>
```

The job response includes completion/error state, pipeline log lines, result
data, and the generated signal. Keep the frontend state flow compatible with
`idle -> selected -> running -> done/error`.

## Change Workflow

Before editing, identify the affected pipeline stage and read its README plus
the relevant entry point. After editing:

1. Run the smallest applicable validation command.
2. Run `python -m quantum.tests` for quantum, feature-contract, label, QAOA, or
   VQC changes.
3. Run `npm run build` for frontend changes.
4. Add the required dated entry to [`changes.md`](../changes.md).
5. Do not include generated datasets, model checkpoints, videos, or secrets in
   the change.
