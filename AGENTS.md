# AGENTS.md

Deepfake-video detection for KYC using rPPG (physiological evidence) + hybrid quantum-classical ML (PennyLane + PyTorch). Low-resolution input, decision bins REAL / FAKE / UNCERTAIN.

## Layout

All active code lives under `WORKING/` and `frontend/`; the repo root holds docs/PDFs and README. Single git repo (no nested repos). `FF++/` is the gitignored dataset.

```
WORKING/
  frame/    stage 1: frame sampling, YOLO face detection, quality filtering (has its own app/ + requirements.txt)
  RPPG/     stage 2: MediaPipe face ROIs -> POS/CHROM pulse -> 8 physiological features (+ rppg-pipeline/, requirements.txt)
  quantum/  stage 3: QAOA feature selection -> hybrid VQC -> P(real) -> verdict (run via `python -m quantum.pipeline`)
  run_pipeline.py  end-to-end orchestrator: frames -> rPPG -> quantum -> verdict
  output/   global regenerated-artifacts root (all `output/` dirs untracked)
    frames/     stage 1: frame_sequences/, frame_extraction_*.json(l), docs/
    rppg/       stage 2: dataset_features.csv, rppg_classifier.pkl + metadata, plots
    quantum/    stage 3: data.npz, qaoa_selection.json, feature_scaler.json, hybrid_vqc.pt, metrics, plots
    pipeline/   run_pipeline.py result JSON

frontend/
  server.py   stdlib-only API: upload, SSE progress, artifact serving, concurrency cap, validation
  src/        React + Vite frontend (components/, hooks.js, api.js, styles.css)
  dump_signal.py  reconstructs rPPG waveform from stage-1 frames for UI visualization
```

## Commands

### From `WORKING/`
- `python -m quantum.pipeline --all` — full quantum flow: build real dataset, QAOA selection (8→6), train VQC, evaluate, baselines. Regenerates `output/quantum/*` (`data.npz`, `qaoa_selection.json`, `feature_scaler.json`, `hybrid_vqc.pt`, metrics, plots).
- `python run_pipeline.py --source path/video.mp4 [--method POS|CHROM] [--out result.json]` — end-to-end inference. Requires the quantum artifacts above.
- Rebuild rPPG training data: `python rppg-pipeline/extract_dataset_features.py` from `WORKING/RPPG/` (writes `output/rppg/dataset_features.csv`), then rerun `python -m quantum.pipeline --all`.
- Standalone frame stage: `python app/main.py --source test.mp4 --save-metadata` from `WORKING/frame/`.

The `quantum.*` imports and the `sys.path` insertions in `run_pipeline.py` assume the working directory is `WORKING/`. Do not run from the repo root.

### From `frontend/`
- `python server.py` — starts API on `http://127.0.0.1:8000` (port via `FRONTEMD_PORT` env)
- `npm run dev` — starts React dev server on `http://localhost:5173` (proxies `/api` to backend)
- `npm run build` — production build to `dist/`

## API Contract (frontend/server.py)

```
POST /api/detect          upload video (raw body + X-Filename header) -> {job: <id>}
GET  /api/jobs/<id>       {done, error, video, signal, lines[-400:], result}
GET  /api/jobs/<id>/events  SSE: line / stage / result / signal / error
GET  /api/previous        last canonical pipeline result (with _signal)
GET  /api/health          {ok, platform, running, has_previous, sequences, artifacts}
GET  /api/artifacts?dir=  list files under output/<rel>
GET  /api/files?rel=      serve artifact file (path traversal protected)
```

Upload limits: 200 MB max; magic-byte validation (MP4/MOV ftyp, AVI RIFF, WebM EBML). Concurrency: max 2 jobs (429 on overflow). Jobs TTL: 1h; frame sequences: 24h.

## Verified Constraints (do not break)

- **No synthetic data.** The quantum layer consumes only the real rPPG feature table `output/rppg/dataset_features.csv` (currently 18 labeled rows). Never reintroduce a generator or a transform/bridge layer.
- **Feature contract:** `FEATURE_NAMES` in `quantum/config.py` must stay identical in name AND order to `RPPGFeatures.feature_names()` in `RPPG/rppg/features.py` (duplicated on purpose; `data.py`, `qaoa.py`, and `pipeline.py` all index by it). Keep the two lists in sync.
- **Label conventions differ per stage — do not unify:**
  - rPPG CSV: `1 = fake, 0 = real`
  - quantum: `LABEL_REAL = 1`, `LABEL_FAKE = 0`; `quantum/data.py` flips (`y = 1 - csv_label`)
  - rPPG RandomForest cross-check in `run_pipeline.py`: `1 = DEEPFAKE`
- **Gitignored artifacts:** `*.csv`, `*.json`, `*.pkl`, `*.mp4`, and all `output/` dirs are untracked — `dataset_features.csv`, `output/quantum/*`, and the trained models will not appear in `git status`. Regenerating them is normal.
- **rPPG returns `features=None`** when usable frames < `min_usable_frames` (48); `run_pipeline.py` then emits INCONCLUSIVE and exits 3. New code must handle `None`.
- **Stage 1 feeds stage 2.** `run_pipeline.py` hands the frame stage's accepted JPEGs (`output/frames/frame_sequences/<video>/frames/`) plus `frame_metadata.jsonl` to `RPPGPipeline.process_frames()` at the stage-1 sample rate (10 fps). rPPG no longer re-gates on blur/brightness (stage 1 did) but still runs MediaPipe per frame; `features=None` handling (INCONCLUSIVE, exit 3) is unchanged. If stage 1 fails or yields no frames, `run_pipeline.py` falls back to `RPPGPipeline.process_video()` (direct video read; `input_mode` in the result JSON records which path ran). Standalone RPPG scripts keep using `process_video`.
- RandomForest cross-check is an optional side path; the final verdict comes exclusively from the quantum stage.
- Small training set (10 train rows) makes QAOA mutual-information weights degenerate (`"success": false`, mostly-zero weights in `qaoa_selection.json`). Expected — data-quantity issue, not a code bug.
- rPPG needs MediaPipe Face Landmarker; the model auto-downloads on first run (internet required). In `frame/`, only `yolov8n-face-lindevs.pt` auto-downloads; missing other presets raise `FileNotFoundError`.

## Optimizations (completed)

- **POS rPPG vectorization** (`WORKING/RPPG/rppg/signal_extraction.py`): `sliding_window_view` + batched matmul + `np.add.at` overlap-add — bit-identical to original loop, removes Python-level window iteration.
- **Shared Welch PSD** (`WORKING/RPPG/rppg/features.py`): single periodogram for HR, SNR, entropy, SQI — 3 fewer `scipy.signal.welch` calls per video.
- **Cached filter/detrend** (`WORKING/RPPG/rppg/preprocessing.py`): `@lru_cache` on Butterworth coefficients + detrend sparse matrix.
- **Skin-mask hoist** (`WORKING/RPPG/rppg/face_roi.py`): compute YCrCb+inRange once/frame instead of 3×.
- **Skip discarded quality work** (`WORKING/RPPG/rppg/pipeline.py`): stage-1 path skips Laplacian/brightness (overwritten by `face.found` anyway).
- **Quantum model cache** (`WORKING/quantum/vqc.py`): `HybridModel` cached by `(n_features, ckpt_mtime, size)` — reuses loaded weights across server requests.
- **Video metadata probe** (`WORKING/run_pipeline.py`): single `cv2.VideoCapture` open emits `video_meta` (name, size, duration, fps, resolution, frame_count) — no fabricated values.
- **Server hardening** (`frontend/server.py`): size/magic validation, concurrency cap (2), result-first SSE + async `signal` event, TTL cleanup, sanitized inbox filenames (`{job8}_{stem}.ext`) for thumbnail consistency.

## Frontend State Machine

`idle` → `selected` (preview + metadata + Start) → `running` (7-stage pipeline + live panel + creeping progress + continuous sheen) → `done` (pipeline persists with 100% bar + result strip; dashboard: Verdict radial gauge, Insights 6 metrics, Signal canvas, Quantum flow, FileInfo, FrameSamples) → `error` (inline banner)

Theme toggle persists `rppgqc.theme` in localStorage; respects `prefers-reduced-motion`.

## Verification

- **Numerical equivalence:** POS vectorization bit-identical to original loop (max |diff| = 0.0 across 7 (T,fs) pairs). Full pipeline verdict: `prob_real=0.6359 → UNCERTAIN, confidence=0.2718` — matches historical result.
- **E2E:** `idle → selected → running → done` with signal canvas, frame thumbnails (5 frames), theme toggle, sequential rerun. Invalid file → 415 friendly error. Responsive: 1920→375px no overflow.
- **No JS errors.** Console 404s = missing frame thumbnails for stale runs (gitignored).

Run: `python -m quantum.pipeline --all` (or `--build-data --select --train`) and `python run_pipeline.py --source <video> --method POS` from `WORKING/`. From `frontend/`: `python server.py` + `npm run dev`.

`Docs/projec_audit.md` predates the real-data refactor; its "bridged bug" finding is already fixed in `run_pipeline.py` — ignore stale recommendations in it.