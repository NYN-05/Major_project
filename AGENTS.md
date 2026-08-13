# AGENTS.md

Deepfake-video detection for KYC using rPPG (physiological evidence) + hybrid quantum-classical ML (PennyLane + PyTorch). Low-resolution input, decision bins REAL / FAKE / UNCERTAIN.

## Layout

All active code lives under `WORKING/` and `frontend/`; the repo root holds docs/PDFs, README, and `Scrape/` (dev scratch). Single git repo (no nested repos). `FF++/` is the gitignored dataset.

```
WORKING/
  frame/    stage 1: frame sampling, YOLO face detection, quality filtering (has its own app/ + requirements.txt)
  RPPG/     stage 2: MediaPipe face ROIs -> POS/CHROM pulse -> 10 physiological features (+ rppg-pipeline/, requirements.txt)
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

## Dev Scratch (`Scrape/`) — Mandatory Home for Dev Files

`Scrape/` at the repo root is the **only** place for throwaway development files. This is a permanent project convention:

- **All tests, test artifacts, temporary scripts, debugging scripts, one-off probes, benchmark scripts, screenshots/screen captures, logs, and related development files MUST be created and stored inside `Scrape/`** — including everything produced by an agent (opencode/Claude) during a session.
- **Never create or leave such files in temp directories** (e.g. `%TEMP%\opencode`, `/tmp`), the repo root, or any other project directory. This keeps the project root, `WORKING/`, `frontend/`, and docs clean and `git status` meaningful.
- Preserve meaningful artifacts: move old temp-dir files into `Scrape/` rather than deleting them (done 2026-08-13: ~200 files migrated from `%TEMP%\opencode`).
- `Scrape/` is gitignored — treat it as scratch, not source. Production code still belongs in `WORKING/`/`frontend/`; only files with lasting value (e.g. `Scrape/test_*.py` harnesses) may be promoted into the repo deliberately, never by accident.

## Commands

### From `WORKING/`
- `python -m quantum.pipeline --all` — full quantum flow: build real dataset, QAOA selection (8→6), train VQC, evaluate, baselines. Regenerates `output/quantum/*` (`data.npz`, `qaoa_selection.json`, `feature_scaler.json`, `hybrid_vqc.pt`, metrics, plots).
- `python -m quantum.tests` — self-checks (no pytest): beta-alive ansatz, Hamiltonian≡classical cost, feature-contract sync, split determinism. Run after any `qaoa.py`/`config.py` change.
- `python run_pipeline.py --source path/video.mp4 [--method POS|CHROM] [--out result.json]` — end-to-end inference. Requires the quantum artifacts above.
- Rebuild rPPG training data: `python rppg-pipeline/extract_dataset_features.py` from `WORKING/RPPG/` (writes `output/rppg/dataset_features.csv`), then rerun `python -m quantum.pipeline --all`.
- Standalone frame stage: `python app/pipeline.py --source test.mp4 --save-metadata` from `WORKING/frame/` (`app/main.py` does not exist).

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

- **No synthetic data.** The quantum layer consumes only the real rPPG feature table `output/rppg/dataset_features.csv` (currently 16 labeled rows: 9 real, 7 fake — DFDC archive only; FF++ is NOT used despite README claims). Never reintroduce a generator or a transform/bridge layer.
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
- **QAOA simulator:** `qaoa.simulator_device()` uses `lightning.qubit` (falls back to `default.qubit` if `pennylane-lightning` is missing). Cost circuits precompute `PauliRot` gates; restarts (default 4) run in a `ProcessPoolExecutor` (`QAOASelectionConfig.restarts`/`n_jobs`). Expect ~15 s vs ~37 s sequential on `default.qubit`. `lightning.gpu` is NOT available on Windows (no `custatevec` wheels) — CPU SIMD is the ceiling there.
- **QAOA ansatz (fixed 2026-08-13):** `_apply_qaoa` applies precomputed cost gates with `gamma` and the X-mixer gates with `beta` separately (both lists come from `_precompute_gates`). Regression guard: `test_beta_alive` in `quantum/tests.py` asserts perturbing beta changes the cost. Any refactor must keep beta alive and re-run `python -m quantum.tests` + `--all`.
- **Hamiltonian ≡ classical cost (fixed 2026-08-13):** `_cost_terms` now reproduces `_classical_cost` exactly (verified 7.1e-15 on real data over all-zeros/all-ones/single-bit/random bitstrings) and `pipeline.py` hard-asserts `error < 1e-6`. The old `verify_hamiltonian` max err ≈ 1.94e+00 (print-only) is history — do not reintroduce a silent verification.
- **VQC training device:** torch side of `HybridModel` runs on CUDA when available (`vqc.resolve_device()`); the PennyLane QNode stays on `default.qubit` + `backprop` (lightning.qubit adjoint was slower: 15.3 s vs 13.2 s). Inference: ~275 ms cold / 32 ms cached.
- **Lazy heavy imports:** xgboost is imported only inside `run_baselines` (~13 s on this host); sklearn is still EAGER via `quantum/plots.py` (imported by `evaluation.py`, ~2.4 s) — move it into the plot functions to fully defer. `qaoa.py` imports sklearn lazily via `_mutual_info_weights`. Keeps QAOA spawn workers cheap — `import quantum.pipeline` ~6.5 s (warm) / ~9.5 s (cold) vs ~22 s before.
- rPPG needs MediaPipe Face Landmarker; the model auto-downloads on first run (internet required). In `frame/`, only `yolov8n-face-lindevs.pt` auto-downloads; missing other presets raise `FileNotFoundError`.

## Optimizations (completed)

- **POS rPPG vectorization** (`WORKING/RPPG/rppg/signal_extraction.py`): `sliding_window_view` + batched matmul + `np.add.at` overlap-add — bit-identical to original loop, removes Python-level window iteration.
- **Shared Welch PSD** (`WORKING/RPPG/rppg/features.py`): single periodogram for HR, SNR, entropy, SQI — 3 fewer `scipy.signal.welch` calls per video.
- **Cached filter/detrend** (`WORKING/RPPG/rppg/preprocessing.py`): `@lru_cache` on Butterworth coefficients + detrend sparse matrix.
- **Skin-mask hoist** (`WORKING/RPPG/rppg/face_roi.py`): compute YCrCb+inRange once/frame instead of 3×.
- **Discarded quality work** (`WORKING/RPPG/rppg/pipeline.py`): stage-1 path still computes Laplacian/brightness per frame (they are recorded in metadata) but the rPPG gate uses only `face.found`.
- **Quantum model cache** (`WORKING/quantum/vqc.py`): `HybridModel` cached by `(n_features, ckpt_mtime, size)` — reuses loaded weights across server requests.
- **Server hardening** (`frontend/server.py`): size/magic validation, concurrency cap (2), result-first SSE + async `signal` event, TTL cleanup, sanitized inbox filenames (`{job8}_{stem}.ext`) for thumbnail consistency, CORS restricted to localhost origins (POST from other origins → 403), uploads streamed to disk in 64 KB chunks (no 200 MB in-memory buffers), 30-min hard timeout per pipeline run (worker killed), `FRONTEND_PORT` env (deprecated `FRONTEMD_PORT` still honored).

See `PROJECT_AUDIT_REPORT.md` (repo root) for the full 2026-08-13 audit — two confirmed QAOA-layer defects (§6.1 beta-mixer regression, §6.2 Hamiltonian mismatch) since fixed and regression-guarded in `quantum/tests.py`, dataset-scale limits, and server security notes.

## Frontend State Machine

`idle` → `selected` (preview + metadata + Start) → `running` (7-stage pipeline + live panel + creeping progress + continuous sheen) → `done` (pipeline persists with 100% bar + result strip; dashboard: Verdict radial gauge, Insights 6 metrics, Signal canvas, Quantum flow, FileInfo, FrameSamples) → `error` (inline banner)

Theme toggle persists `rppgqc.theme` in localStorage; respects `prefers-reduced-motion`.

## Verification

- **Numerical equivalence:** POS vectorization bit-identical to original loop (max |diff| = 0.0 across 7 (T,fs) pairs). Post-fix pipeline verdict (2026-08-13): `prob_real=0.6155 → UNCERTAIN, confidence=0.2311`; QAOA selection `['hr_half_diff','cheek_forehead_correlation','left_right_cheek_correlation','mad','heart_rate_bpm','spectral_entropy']` (chosen restart seed 43, cost 0.3950), ECE 0.1052 (was 0.2478 under the buggy ansatz/Hamiltonian).
- **E2E:** `idle → selected → running → done` with signal canvas, frame thumbnails (5 frames), theme toggle, sequential rerun. Invalid file → 415 friendly error. Responsive: 1920→375px no overflow.
- **No JS errors.** Console 404s = missing frame thumbnails for stale runs (gitignored).

Run: `python -m quantum.pipeline --all` (or `--build-data --select --train`) and `python run_pipeline.py --source <video> --method POS` from `WORKING/`. From `frontend/`: `python server.py` + `npm run dev`.

`Docs/projec_audit.md` predates the real-data refactor; its "bridged bug" finding is already fixed in `run_pipeline.py` — ignore stale recommendations in it.