# AGENTS.md

Deepfake-video detection for KYC using rPPG (physiological evidence) + hybrid quantum-classical ML (PennyLane + PyTorch). Low-resolution input, decision bins REAL / FAKE / UNCERTAIN.

## Layout

All active code lives under `WORKING/` and `frontend/`; the repo root holds docs/PDFs, README, and `Scrape/` (dev scratch). Single git repo (no nested repos). `FF++/` is the gitignored dataset.

```
WORKING/
  frame/    stage 1: frame sampling, YOLO face detection, quality filtering (has its own app/ + requirements.txt)
  RPPG/     stage 2: MediaPipe face ROIs -> POS/CHROM pulse -> 20 physiological features (+ rppg-pipeline/, requirements.txt)
  quantum/  stage 3: QAOA feature selection -> hybrid VQC -> P(real) -> verdict (run via `python -m quantum.pipeline`)
  run_pipeline.py  end-to-end orchestrator: frames -> rPPG -> quantum -> verdict
  output/   global regenerated-artifacts root (all `output/` dirs untracked)
    frames/     stage 1: frame_sequences/, frame_extraction_*.json(l), docs/
    rppg/       stage 2: dataset_features.csv, rppg_classifier.pkl + metadata, plots
    quantum/    stage 3: data.npz, split_manifest.json, qaoa_selection.json, selection_comparison.json, feature_scaler.json, hybrid_vqc.pt, threshold_analysis.json, metrics, plots
    pipeline/   run_pipeline.py result JSON

frontend/
  server.py   stdlib-only API: upload, SSE progress, artifact serving, concurrency cap, validation
  src/        React + Vite frontend (components/, hooks.js, api.js, styles.css)
  dump_signal.py  standalone utility: reconstructs rPPG waveform from stage-1 frames (the pipeline itself writes it via run_pipeline.py --signal-out, served as result._signal)
```

## Dev Scratch (`Scrape/`) — Mandatory Home for Dev Files

`Scrape/` at the repo root is the **only** place for throwaway development files. This is a permanent project convention:

- **All tests, test artifacts, temporary scripts, debugging scripts, one-off probes, benchmark scripts, screenshots/screen captures, logs, and related development files MUST be created and stored inside `Scrape/`** — including everything produced by an agent (opencode/Claude) during a session.
- **Never create or leave such files in temp directories** (e.g. `%TEMP%\opencode`, `/tmp`), the repo root, or any other project directory. This keeps the project root, `WORKING/`, `frontend/`, and docs clean and `git status` meaningful.
- Preserve meaningful artifacts: move old temp-dir files into `Scrape/` rather than deleting them (done 2026-08-13: ~200 files migrated from `%TEMP%\opencode`).
- `Scrape/` is gitignored — treat it as scratch, not source. Production code still belongs in `WORKING/`/`frontend/`; only files with lasting value (e.g. `Scrape/test_*.py` harnesses) may be promoted into the repo deliberately, never by accident.

## Commands

### From `WORKING/`
- `python -m quantum.pipeline --all` — full quantum flow: build real dataset, QAOA selection (20→3), train VQC, evaluate, baselines. Regenerates `output/quantum/*` (`data.npz`, `split_manifest.json`, `qaoa_selection.json`, `selection_comparison.json`, `feature_scaler.json`, `hybrid_vqc.pt`, `threshold_analysis.json`, metrics, plots).
- `python -m quantum.tests` — self-checks 10/10 (no pytest): beta-alive ansatz, Hamiltonian≡classical cost, feature-contract sync, split determinism, subject grouping, no group leakage, torch-sim ≡ PennyLane (QAOA + VQC layer), checkpoint compat, label conversion. Run after any `qaoa.py`/`config.py`/`vqc.py` change.
- `python -m quantum.sweep --timeout 600 --out <file>.json` — crash-safe QAOA/VQC hyperparameter sweep (dev tool).
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

- **No synthetic data.** The quantum layer consumes only the real rPPG feature table `output/rppg/dataset_features.csv` (3473 rows @native 30 fps, full extraction: 1921 real / 1552 fake — DFDC archive (`archive/DFDC_Dataset`, 1566 fake + 1727 real clips on disk) AND FF++ (train/val/test: FF-real + YouTube-real reals, FF-synthesis fakes); legacy `archive (1)` CSV layout is absent on disk). Rebuilt 2026-08-19 via `extract_dataset_features.py --include-ffpp --workers 0` (3471 rows survive the quantum HR-plausibility filter; backup of the prior 421-row table in `Scrape/dataset_features_backup_421rows_20260815.csv`). Extraction supports `--gpu` (YuNet ONNX CUDA face detection) and quality gates (`--min-sqi`, `--max-nan-features`). Never reintroduce a generator or a transform/bridge layer.
- **Feature contract:** `FEATURE_NAMES` in `quantum/config.py` (20 features) must stay identical in name AND order to `RPPGFeatures.feature_names()` in `RPPG/rppg/features.py` (duplicated on purpose; `data.py`, `qaoa.py`, and `pipeline.py` all index by it). Keep the two lists in sync; `test_feature_contract_sync` asserts equality.
- **Label conventions differ per stage — do not unify:**
  - rPPG CSV: `1 = fake, 0 = real`
  - quantum: `LABEL_REAL = 1`, `LABEL_FAKE = 0`; the flip lives only in the tested `quantum/data.py: csv_to_quantum_label()` (Phase 1A of the remediation plan)
  - rPPG RandomForest cross-check in `run_pipeline.py`: `1 = DEEPFAKE`
- **Gitignored artifacts:** `*.csv`, `*.json`, `*.pkl`, `*.mp4`, and all `output/` dirs are untracked — `dataset_features.csv`, `output/quantum/*`, and the trained models will not appear in `git status`. Regenerating them is normal.
- **rPPG returns `features=None`** when usable frames < `min_usable_frames` (48); `run_pipeline.py` then emits INCONCLUSIVE and exits 3. New code must handle `None`.
- **Stage 1 feeds stage 2.** `run_pipeline.py` hands the frame stage's accepted JPEGs (`output/frames/frame_sequences/<video>/frames/`) plus `frame_metadata.jsonl` to `RPPGPipeline.process_frames()` at the stage-1 sample rate (30 fps — matches the dataset extraction rate; 10 fps decimation had no anti-alias filter and performed below chance, see Verification). rPPG no longer re-gates on blur/brightness (stage 1 did) but still runs MediaPipe per frame; `features=None` handling (INCONCLUSIVE, exit 3) is unchanged. If stage 1 fails or yields no frames, `run_pipeline.py` falls back to `RPPGPipeline.process_video()` (direct video read; `input_mode` in the result JSON records which path ran). Standalone RPPG scripts keep using `process_video`. `run_pipeline.py --signal-out <path>` writes the decimated stage-2 waveform JSON (same schema as `frontend/dump_signal.py`, which is retained only as a standalone utility).
- RandomForest cross-check is an optional side path; the final verdict comes exclusively from the quantum stage. As of 2026-08-19 it reports `skipped`: the 20-feature pickle is incompatible with the 10-feature guard in `run_pipeline.py` — a stale guard, not a functional check.
- **rPPG classifier trust:** `output/rppg/` must remain write-protected; `rppg_classifier.pkl` is `pickle.load`-ed by `run_pipeline.py` and the Streamlit demo (arbitrary-code risk if replaced). Never move to shared hosting as-is.
- **QAOA selection weights:** the selection objective uses supervised discrimination weights (`qaoa._discrimination_weights`: sign-agnostic exact Mann-Whitney AUC strength `2*|AUC-0.5|`, deterministic, no sklearn) instead of the old unsupervised mutual-information weights, which systematically excluded the two strongest features (HR, SNR). `QAOASelectionConfig.target_features` is 3 (evidence: HR+SNR+hr_half_diff VQC CV AUC 0.569 vs 0.484 for the old 6-feature MI pick). `select_classical` and `verify_hamiltonian` use the same weights for a fair reference. Keep `_mutual_info_weights` only as a documented alternative — do not reintroduce it into `select()`.
- **QAOA simulator (torch-native default, 2026-08-19):** `qaoa.simulator_device(wires, cfg)` returns `(dev, backend)`; `QAOASelectionConfig.device="auto"` (default) uses the project's exact complex128 statevector simulator `qaoa_sim.QAOASimulator` (~0.3–0.5 ms per circuit call on the 20-wire problem vs ~5.6 s via PennyLane — ~20× faster; ~5 µs for 3 wires), CPU-process-safe (workers never open CUDA contexts). `"pennylane"` (aliases `"lightning"`/`"default"`) selects the legacy PennyLane QNode path for cross-verification; `test_qaoa_sim_matches_pennylane` pins the sim to ≤1e-6. The GPU-first probe history is superseded: on this host PL ≥ 0.39 removed `default.qubit.torch` and `lightning.gpu` has no Windows wheels, which is why the torch-native sims were written. Cost circuits precompute `PauliRot` gates; restarts (default 4) run in a `ProcessPoolExecutor` (`QAOASelectionConfig.restarts`/`n_jobs`). Expect ~15 s.
- **QAOA ansatz (fixed 2026-08-13):** `_apply_qaoa` applies precomputed cost gates with `gamma` and the X-mixer gates with `beta` separately (both lists come from `_precompute_gates`). Regression guard: `test_beta_alive` in `quantum/tests.py` asserts perturbing beta changes the cost. Any refactor must keep beta alive and re-run `python -m quantum.tests` + `--all`.
- **Hamiltonian ≡ classical cost (fixed 2026-08-13):** `_cost_terms` now reproduces `_classical_cost` exactly (verified 7.1e-15 on real data over all-zeros/all-ones/single-bit/random bitstrings) and `pipeline.py` hard-asserts `error < 1e-6`. The old `verify_hamiltonian` max err ≈ 1.94e+00 (print-only) is history — do not reintroduce a silent verification.
- **VQC training device (GPU-first):** torch side of `HybridModel` (head, loss, optimizer) runs on CUDA when available (`vqc.resolve_device()`). The quantum layer is GPU-first in intent but the circuit runs on the exact torch-native `QuantumLayerTorch` simulator (complex128, batched state evolution; `VQCConfig.qnode_impl="auto"` default). `"pennylane"` selects the legacy PennyLane QNode (`default.qubit` + `backprop`) for cross-verification; `test_torch_layer_matches_pennylane` pins it to ≤1e-5 and checkpoints stay interchangeable (same `weights` shape `(qml_layers, n, 3)`). On PL 0.42 there is no GPU circuit backend, so the circuit is CPU today; measured CUDA-head gain: 15-epoch VQC train 12.44 s (CUDA head) vs 14.35 s (CPU-only head), min of 2 (`Scrape/bench_gpu_backends.py`). Inference: ~275 ms cold / 32 ms cached. On GPU hosts the 5-fold VQC CV in `evaluation.py` runs folds sequentially (`n_jobs=1`): parallel fold workers would each open a CUDA context on the shared 6 GB card (OOM risk); CPU hosts keep parallel folds.
- **Lazy heavy imports:** xgboost is imported only inside `run_baselines` (~13 s on this host); sklearn is deferred inside the plot functions (`plots._sklearn()`, ~2.4 s); matplotlib is deferred inside the plot functions (`plots._plt()`, ~0.76 s). Note: matplotlib is still loaded eagerly by `pennylane` itself (95 submodules incl. pyplot), so the deferral only keeps `quantum.plots` importable without it — the measurable `import quantum.pipeline` gain is small. `import quantum.pipeline` ~6.9 s (warm) vs ~7.1 s before; ~22 s before the original lazy-import work. `run_pipeline.py` additionally defers `from quantum.pipeline import predict_features` (torch+pennylane) into `quantum_inference()` so stage-1/2 progress lines stream to the SSE client before the heavy import stack loads. The new `qaoa._discrimination_weights` needs no sklearn (exact Mann-Whitney), so QAOA workers avoid it entirely.
- rPPG needs MediaPipe Face Landmarker; the model auto-downloads on first run (internet required). In `frame/`, only `yolov8n-face-lindevs.pt` auto-downloads; missing other presets raise `FileNotFoundError`.

## Optimizations (completed)

- **POS rPPG vectorization** (`WORKING/RPPG/rppg/signal_extraction.py`): `sliding_window_view` + batched matmul + `np.add.at` overlap-add — bit-identical to original loop, removes Python-level window iteration.
- **Shared Welch PSD** (`WORKING/RPPG/rppg/features.py`): single periodogram for HR, SNR, entropy, SQI — 3 fewer `scipy.signal.welch` calls per video.
- **Cached filter/detrend** (`WORKING/RPPG/rppg/preprocessing.py`): `@lru_cache` on Butterworth coefficients + detrend sparse matrix.
- **Skin-mask hoist** (`WORKING/RPPG/rppg/face_roi.py`): compute YCrCb+inRange once/frame instead of 3×.
- **Discarded quality work** (`WORKING/RPPG/rppg/pipeline.py`): stage-1 path still computes Laplacian/brightness per frame (they are recorded in metadata) but the rPPG gate uses only `face.found`.
- **Quantum model cache** (`WORKING/quantum/vqc.py`): `HybridModel` cached by `(n_features, ckpt_mtime, size)` — reuses loaded weights across server requests.
- **Server hardening** (`frontend/server.py`): size/magic validation, concurrency cap (2), synchronous result-first SSE (the waveform is produced by `run_pipeline.py --signal-out` itself, so the `signal` rel rides inside the result payload as `_signal` — no background dump subprocess, no lost signal event), TTL cleanup, sanitized inbox filenames (`{job8}_{stem}.ext`) for thumbnail consistency, CORS restricted to localhost origins (POST from other origins → 403), uploads streamed to disk in 64 KB chunks (no 200 MB in-memory buffers), 30-min hard timeout per pipeline run (worker killed), `FRONTEND_PORT` env (deprecated `FRONTEMD_PORT` still honored).
- **ROI mean via `cv2.mean`** (`WORKING/RPPG/rppg/face_roi.py`): `mean_rgb` uses `cv2.mean(frame, mask)` instead of `frame[mask > 0].mean(axis=0)` — no masked-pixel array allocation per ROI per frame.
- **Single-forward val metrics** (`WORKING/quantum/vqc.py`): `_val_metrics` computes loss + accuracy from one logits tensor instead of two model forwards per epoch.

The 2026-08-13 audit found two confirmed QAOA-layer defects (§6.1 beta-mixer regression, §6.2 Hamiltonian mismatch) — both fixed and regression-guarded in `quantum/tests.py` (see "QAOA ansatz" / "Hamiltonian ≡ classical cost" above).

## Frontend State Machine

`idle` → `selected` (preview + metadata + Start) → `running` (7-stage pipeline + live panel + creeping progress + continuous sheen) → `done` (pipeline persists with 100% bar + result strip; dashboard: Verdict radial gauge, Insights 6 metrics, Signal canvas, Quantum flow, FileInfo, FrameSamples) → `error` (inline banner)

Theme toggle persists `rppgqc.theme` in localStorage; respects `prefers-reduced-motion`.

## Verification

- **Numerical equivalence:** POS vectorization bit-identical to original loop (max |diff| = 0.0 across 7 (T,fs) pairs). Post-fix pipeline verdict (2026-08-13): `prob_real=0.6155 → UNCERTAIN, confidence=0.2311`; QAOA selection `['hr_half_diff','cheek_forehead_correlation','left_right_cheek_correlation','mad','heart_rate_bpm','spectral_entropy']` (chosen restart seed 43, cost 0.3950), ECE 0.1052 (was 0.2478 under the buggy ansatz/Hamiltonian). Post-selection-fix (2026-08-15): QAOA selects `['heart_rate_bpm','snr_db','mad']` (the old MI objective excluded HR/SNR), VQC test acc 0.512 / AUC 0.534, CV(5-fold) AUC 0.557 (was 0.500), ECE 0.084; decision bins remain 100% UNCERTAIN at 0.3/0.7 — thresholds are the next lever, and HR-only LR (AUC 0.595) beats the 10-feature VQC, so feature dilution still caps the ceiling. Post-full-extraction (2026-08-15, 3445 rows incl. FF++): QAOA selects `['left_right_cheek_correlation','spectral_entropy','snr_db']` (seed 44, cost −1.587), VQC test acc 0.547 / AUC 0.532, CV(5-fold) AUC 0.493 (below the 0.557 from the 421-row DFDC-only table), ECE 0.067; decision bins remain 100% UNCERTAIN — 8.2x more data did not lift the ceiling, confirming the rPPG features themselves carry almost no class signal (per-feature |AUC−0.5| ≤ 0.06). **Post-leakage-fix frozen baseline (2026-08-15):** with FF++ source-subject grouping (first-id token; youtube-real per-clip; DFDC per-clip) and grouped evaluation, QAOA selects `['peak_prominence','left_right_cheek_correlation','spectral_entropy']` (seed 44, cost −1.363), test acc 0.547 / AUC 0.503 / PR-AUC 0.549 / specificity 0.0 (confusion [[0,312],[0,377]] → majority-class predictor), grouped CV(5) AUC 0.517; classical LR/GNB/XGB test AUC 0.565–0.569. Snapshot + manifest at `output/quantum/baseline_20260815_221652/`. Still 100% UNCERTAIN bins; per-feature |AUC−0.5| ≤ ~0.06 → Phase 2 rPPG method/ROI probe (Scrape/probe_rppg_methods.py) is the gate. **GPU-first rerun (2026-08-18, device auto-detect; no behavior change — no GPU circuit backend exists on this host):** same code path (QNode CPU, torch head CUDA), data rebuilt at 3471 rows → QAOA selects `['signal_quality_index','peak_prominence','entropy_window_std']` (seed 44, cost −0.7072), test acc 0.553 / AUC 0.543 / ECE 0.060, CV(5) AUC 0.529 (sequential folds on GPU host); baselines GNB best AUC 0.570. Decision bins still 100% UNCERTAIN. Regression suite 7/7 PASS. **Frozen baseline (2026-08-19, 3473 rows, torch-native simulators):** QAOA selects `['cheek_forehead_correlation','left_right_cheek_correlation','signal_to_motion_ratio']` (seed 44, cost −0.767; restart spread [11.73, −0.098, −0.767, 9.83], `success: false` — difficult QUBO landscape; greedy classical reference overlaps 2/3). VQC test acc 0.552 / AUC 0.535 / specificity 0.000 / balanced acc 0.499 / ECE 0.068 (majority-class collapse, confusion [[0,310],[1,383]]); grouped 5-fold CV AUC 0.556 ± 0.019; best classical LR test AUC 0.582. Decision bins still 100% UNCERTAIN (694/694). Regression suite 10/10 PASS. **Remediation status (plan: `Docs/DEEPFAKE_KYC_SEQUENTIAL_REMEDIATION_PLAN.md`, 24 phases):** Phase 1A DONE (label contract `csv_to_quantum_label`/`quantum_to_display_label` + `test_label_conversion`); Phase 1B tested (balanced class weighting in `focal_loss`; experiment flipped the collapse to all-FAKE: acc 0.444 / specificity 0.974 / balanced acc 0.495 / AUC 0.486 — still 100% UNCERTAIN, no separation); Phase 1C DONE (`evaluation.analyze_threshold_behavior` → `threshold_analysis.json` diagnosis Case B: test scores ∈ [0.428, 0.503], classes do not separate). **Next lever: upstream rPPG method/ROI probe (Phase 4), not threshold or model tuning.**
- **Signal flow E2E:** server job now serves `result._signal` synchronously with the result (verified via `Scrape/e2e_signal_flow.py`: job dict + result `_signal` + `/api/files` waveform with fps=30.0 matching the verdict computation).
- **E2E:** `idle → selected → running → done` with signal canvas, frame thumbnails (5 frames), theme toggle, sequential rerun. Invalid file → 415 friendly error. Responsive: 1920→375px no overflow.
- **No JS errors.** Console 404s = missing frame thumbnails for stale runs (gitignored).

Run: `python -m quantum.pipeline --all` (or `--build-data --select --train`) and `python run_pipeline.py --source <video> --method POS` from `WORKING/`. From `frontend/`: `python server.py` + `npm run dev`.

`Docs/projec_audit.md` is gone (deleted with the other pre-2026-08 docs). The current remediation roadmap lives in `Docs/DEEPFAKE_KYC_SEQUENTIAL_REMEDIATION_PLAN.md` (24 severity-ordered phases; 1A–1C status in the Verification section); `Docs/Key_Findings_Contributions_Significance.md` is the honest findings/contributions write-up; `Docs/problems.md` is the ranked problem analysis.