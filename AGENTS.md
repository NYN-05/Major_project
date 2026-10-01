# Repository Guide

## Project

This repository implements deepfake-video detection for KYC using:

- rPPG physiological evidence
- visual features from ResNet50 and handcrafted descriptors
- hybrid quantum-classical machine learning with PennyLane and PyTorch
- three decision outcomes: `REAL`, `FAKE`, and `UNCERTAIN`

Active application code is located in [`WORKING/`](WORKING/) and
[`frontend/`](frontend/). The root contains documentation, PDFs, and
[`Scrape/`](Scrape/), the required location for development-only files.

## Mandatory Change Log

After every code change:

1. Add an entry to [`changes.md`](changes.md).
2. Include the date, time, affected area, and a short description.
3. Record the entry in the same work session; never defer it.

Use this format:

```text
| YYYY-MM-DD HH:mm IST | Area | Short description of the change. | Complete / Experimental / Retired |
```

## Repository Structure

```text
WORKING/
  frame/              Frame sampling, face detection, and quality filtering
  RPPG/               MediaPipe ROIs, POS/CHROM signals, and rPPG features
  visual/             ResNet50 and handcrafted visual features
  quantum/            QAOA selection, VQC training, and inference
  run_pipeline.py     End-to-end frame → rPPG → visual → quantum pipeline
  output/             Regenerated, gitignored artifacts

frontend/
  server.py           Standard-library API, uploads, jobs, SSE, and artifacts
  src/                React + Vite frontend
  dump_signal.py      Standalone waveform reconstruction utility

Scrape/               Required location for tests, probes, logs, screenshots,
                      temporary scripts, and other development artifacts
```

## Development Rules

- Work from `WORKING/` for pipeline and quantum commands.
- Work from `frontend/` for frontend commands.
- Do not run quantum commands from the repository root.
- Do not create temporary or test files outside [`Scrape/`](Scrape/).
- Do not commit generated artifacts, datasets, videos, models, credentials, or
  other secrets.
- Preserve existing behavior unless the requested change explicitly changes it.
- Prefer small, typed, testable changes that follow existing project patterns.
- Do not add synthetic data, bridge transforms, or unsupported datasets.

## Supported Architecture

- Inference supports **fused mode only**: rPPG plus visual features.
- The supported dataset is **DFDC only**.
- FaceForensics++ and retired feature-set modes must not be reintroduced.
- The fused feature contract is defined by `FUSED_FEATURE_NAMES` in
  `WORKING/quantum/config.py`.
- The feature names and order must match the combined rPPG and visual feature
  definitions. Run `python -m quantum.tests` after changing this contract.
- The final verdict comes from the quantum stage. The RandomForest result is an
  optional cross-check only.
- If rPPG returns `features=None` because too few frames are usable, the
  pipeline must return an inconclusive result and exit with status 3.
- Stage 1 accepted frames feed both the rPPG and visual stages. If frame
  extraction fails or produces no frames, the pipeline may fall back to direct
  video processing.

## Data and Label Contracts

- `DFDC_DATASET_PATH` must point to `archive/DFDC_Dataset`.
- rPPG CSV labels use `1 = fake` and `0 = real`.
- Quantum labels use `LABEL_REAL = 1` and `LABEL_FAKE = 0`.
- The label conversion belongs only in
  `WORKING/quantum/data.py:csv_to_quantum_label()`.
- The rPPG RandomForest cross-check uses `1 = DEEPFAKE`.
- Do not unify these conventions without updating all dependent tests and
  documentation.

## Security Constraints

- Treat `WORKING/output/rppg/rppg_classifier.pkl` as untrusted serialized code.
- Keep `WORKING/output/rppg/` write-protected during inference.
- Never move the pickle model to shared hosting without a replacement strategy.
- Keep upload validation, size limits, path-traversal protection, CORS
  restrictions, concurrency limits, and job timeouts intact.
- Never commit API keys, passwords, tokens, private datasets, or credentials.

## Commands

Run these from `WORKING/`:

```powershell
# Full fused quantum pipeline
python -m quantum.pipeline --all

# Regression and contract tests
python -m quantum.tests

# End-to-end inference
python run_pipeline.py --source path\to\video.mp4 --method POS --out result.json

# Rebuild rPPG training features
python rppg-pipeline\extract_dataset_features.py

# Extract visual features and create the fused table
python visual\pipeline.py --frames-root output\frames\frame_sequences `
  --rppg-csv output\rppg\dataset_features.csv --fuse --create-splits
```

Run these from `frontend/`:

```powershell
# Start the API
python server.py

# Start the development frontend
npm run dev

# Build the frontend
npm run build
```

## Validation

Use the smallest relevant validation command for each change:

- Quantum configuration, QAOA, VQC, labels, or feature changes:
  `python -m quantum.tests`
- Full quantum or artifact-generation changes:
  `python -m quantum.pipeline --all`
- Frontend changes: `npm run build`
- API or pipeline changes: run the relevant Python self-checks and an
  end-to-end inference test when representative input is available.

Keep tests and generated validation files in [`Scrape/`](Scrape/) unless the
project already defines a tracked test location.

## API Contract

The API in `frontend/server.py` provides:

```text
POST /api/detect
GET  /api/jobs/<id>
GET  /api/jobs/<id>/events
GET  /api/previous
GET  /api/health
GET  /api/artifacts?dir=<relative-output-directory>
GET  /api/files?rel=<relative-artifact-path>
```

Important behavior:

- Uploads are limited to 200 MB and validated by file signature.
- At most two pipeline jobs run concurrently.
- Job records expire after one hour; frame sequences expire after 24 hours.
- Artifact paths must remain traversal-protected.
- Pipeline progress and the generated signal are delivered through the job
  result/SSE flow.

## Documentation

- Record all changes in [`changes.md`](changes.md).
- Keep detailed technical findings in the relevant documentation under
  [`Docs/`](Docs/).
- Treat this file as the active repository rulebook; remove stale guidance
  when the architecture changes.
