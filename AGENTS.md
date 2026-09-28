# AGENTS.md

Deepfake-video detection for KYC using **rPPG (physiological evidence) + visual features (ResNet50 + handcrafted) + hybrid quantum-classical ML (PennyLane + PyTorch)**. Low-resolution input, decision bins REAL / FAKE / UNCERTAIN.

## Layout

All active code lives under `WORKING/` and `frontend/`; the repo root holds docs/PDFs, README, and `Scrape/` (dev scratch). Single git repo (no nested repos). FF++ dataset support removed.

```
WORKING/
  frame/    stage 1: frame sampling, YOLO face detection, quality filtering (has its own app/ + requirements.txt)
  RPPG/     stage 2: MediaPipe face ROIs -> POS/CHROM pulse -> 20 physiological features (+ rppg-pipeline/, requirements.txt)
  visual/   stage 3: ResNet50 + handcrafted features from face crops -> 39 visual features (+ requirements.txt)
  quantum/  stage 4: QAOA feature selection -> hybrid VQC -> P(real) -> verdict (run via `python -m quantum.pipeline`)
  run_pipeline.py  end-to-end orchestrator: frames -> rPPG -> visual -> quantum (fused) -> verdict
  output/   global regenerated-artifacts root (all `output/` dirs untracked)
    frames/     stage 1: frame_sequences/, frame_extraction_*.json(l), docs/
    rppg/       stage 2: dataset_features.csv, rppg_classifier.pkl + metadata, plots
    visual/     stage 3: visual_features.csv, fused_features.csv
    quantum/    stage 4: data_fused.npz, split_manifest_fused.json, qaoa_selection_fused.json, feature_scaler_fused.json, hybrid_vqc_fused.pt, threshold_analysis.json, metrics, plots
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
- `python -m quantum.pipeline --all` — full quantum flow (fused mode only): build real dataset, QAOA selection, train VQC, evaluate, baselines. Regenerates `output/quantum/*` (`data_fused.npz`, `split_manifest_fused.json`, `qaoa_selection_fused.json`, `feature_scaler_fused.json`, `hybrid_vqc_fused.pt`, `threshold_analysis.json`, metrics, plots).
- `python -m quantum.tests` — self-checks 10/10 (no pytest): beta-alive ansatz, Hamiltonian≡classical cost, feature-contract sync, split determinism, subject grouping, no group leakage, torch-sim ≡ PennyLane (QAOA + VQC layer), checkpoint compat, label conversion. Run after any `qaoa.py`/`config.py`/`vqc.py` change.
- `python -m quantum.sweep --timeout 600 --out <file>.json` — crash-safe QAOA/VQC hyperparameter sweep (dev tool).
- `python run_pipeline.py --source path/video.mp4 [--method POS|CHROM] [--out result.json]` — end-to-end inference (fused mode). Requires the fused quantum artifacts above.
- Rebuild rPPG training data (DFDC only): `python rppg-pipeline/extract_dataset_features.py` from `WORKING/RPPG/` (writes `output/rppg/dataset_features.csv`).
- Extract visual features & fuse: `python visual/pipeline.py --frames-root output/frames/frame_sequences --rppg-csv ../output/rppg/dataset_features.csv --fuse --create-splits` from `WORKING/`.
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

- **No synthetic data.** The quantum layer consumes only the real fused feature table `output/visual/fused_features.csv` (DFDC only). Never reintroduce a generator or a transform/bridge layer.
- **Only fused mode supported.** No `rppg_only`, `visual_only`, `rppg_base`, `rppg_cross_roi`, or ablation feature sets.
- **Only DFDC dataset supported.** No FaceForensics++ (FF++). `DFDC_DATASET_PATH` env var must point to `archive/DFDC_Dataset`.
- **Feature contract:** `FUSED_FEATURE_NAMES` in `quantum/config.py` (59 features = 20 rPPG + 39 visual) must stay identical in name AND order to `RPPGFeatures.feature_names()` + `VISUAL_FEATURE_NAMES`. Keep the two lists in sync; `test_feature_contract_sync` guards it.
- **Label conventions differ per stage — do not unify:**
  - rPPG CSV: `1 = fake, 0 = real`
  - quantum: `LABEL_REAL = 1`, `LABEL_FAKE = 0`; the flip lives only in the tested `quantum/data.py: csv_to_quantum_label()` (Phase 1A of the remediation plan)
  - rPPG RandomForest cross-check in `run_pipeline.py`: `1 = DEEPFAKE`
- **Gitignored artifacts:** `*.csv`, `*.json`, `*.pkl`, `*.mp4`, and all `output/` dirs are untracked — `dataset_features.csv`, `output/visual/fused_features.csv`, `output/quantum/*`, and the trained models will not appear in `git status`. Regenerating them is normal.
- **rPPG returns `features=None`** when usable frames < `min_usable_frames` (24); `run_pipeline.py` then emits INCONCLUSIVE and exits 3. New code must handle `None`.
- **Stage 1 feeds stage 2 & 3.** `run_pipeline.py` hands the frame stage's accepted JPEGs (`output/frames/frame_sequences/<video>/frames/`) plus `frame_metadata.jsonl` to `RPPGPipeline.process_frames()` at the stage-1 sample rate (30 fps) and to `visual.extractor.compute_visual_features()` for face crops. rPPG no longer re-gates on blur/brightness (stage 1 did) but still runs MediaPipe per frame; `features=None` handling (INCONCLUSIVE, exit 3) is unchanged. If stage 1 fails or yields no frames, `run_pipeline.py` falls back to `RPPGPipeline.process_video()` (direct video read; `input_mode` in the result JSON records which path ran). Standalone RPPG scripts keep using `process_video`. `run_pipeline.py --signal-out <path>` writes the decimated stage-2 waveform JSON (same schema as `frontend/dump_signal.py`, which is retained only as a standalone utility).
- RandomForest cross-check is an optional side path; the final verdict comes exclusively from the quantum stage.
- **rPPG classifier trust:** `output/rppg/` must remain write-protected; `rppg_classifier.pkl` is `pickle.load`-ed by `run_pipeline.py` (arbitrary-code risk if replaced). Never move to shared hosting as-is.
- **QAOA selection weights:** the selection objective uses supervised discrimination weights (`qaoa._discrimination_weights`: sign-agnostic exact Mann-Whitney AUC strength `2*|AUC-0.5|`, deterministic, no sklearn). `QAOASelectionConfig.target_features` is 3. Keep `_mutual_info_weights` only as a documented alternative — do not reintroduce it into `select()`.
- **QAOA simulator (torch-native default):** `qaoa.simulator_device(wires, cfg)` returns `(dev, backend)`; `QAOASelectionConfig.device="auto"` (default) uses the project's exact complex128 statevector simulator `qaoa_sim.QAOASimulator` (~0.3–0.5 ms per circuit call vs ~5.6 s via PennyLane — ~20× faster), CPU-process-safe (workers never open CUDA contexts). `"pennylane"` (aliases `"lightning"`/`"default"`) selects the legacy PennyLane QNode path for cross-verification; `test_qaoa_sim_matches_pennylane` pins the sim to ≤1e-6. Cost circuits precompute `PauliRot` gates; restarts (default 4) run in a `ProcessPoolExecutor` (`QAOASelectionConfig.restarts`/`n_jobs`). Expect ~15 s.
- **QAOA ansatz (fixed 2026-08-13):** `_apply_qaoa` applies precomputed cost gates with `gamma` and the X-mixer gates with `beta` separately (both lists come from `_precompute_gates`). Regression guard: `test_beta_alive` in `quantum/tests.py` asserts perturbing beta changes the cost. Any refactor must keep beta alive and re-run `python -m quantum.tests` + `--all`.
- **Hamiltonian ≡ classical cost (fixed 2026-08-13):** `_cost_terms` now reproduces `_classical_cost` exactly (verified 7.1e-15 on real data) and `pipeline.py` hard-asserts `error < 1e-6`.
- **VQC training device (GPU-first):** torch side of `HybridModel` (head, loss, optimizer) runs on CUDA when available (`vqc.resolve_device()`). The quantum layer runs on the exact torch-native `QuantumLayerTorch` simulator (complex128, batched state evolution; `VQCConfig.qnode_impl="auto"` default). `"pennylane"` selects the legacy PennyLane QNode (`default.qubit` + `backprop`) for cross-verification; `test_torch_layer_matches_pennylane` pins it to ≤1e-5 and checkpoints stay interchangeable (same `weights` shape `(qml_layers, n, 3)`). On GPU hosts the 5-fold VQC CV in `evaluation.py` runs folds sequentially (`n_jobs=1`): parallel fold workers would each open a CUDA context on the shared GPU (OOM risk); CPU hosts keep parallel folds.
- **Lazy heavy imports:** xgboost is imported only inside `run_baselines` (~13 s); sklearn is deferred inside the plot functions (`plots._sklearn()`, ~2.4 s); matplotlib is deferred inside the plot functions (`plots._plt()`, ~0.76 s). Note: matplotlib is still loaded eagerly by `pennylane` itself. `run_pipeline.py` additionally defers `from quantum.pipeline import predict_features` (torch+pennylane) into `quantum_inference()` so stage-1/2/3 progress lines stream to the SSE client before the heavy import stack loads. The `qaoa._discrimination_weights` needs no sklearn (exact Mann-Whitney), so QAOA workers avoid it entirely.
- rPPG needs MediaPipe Face Landmarker; the model auto-downloads on first run (internet required). In `frame/`, only `yolov8n-face-lindevs.pt` auto-downloads; missing other presets raise `FileNotFoundError`.
- **Quality thresholds relaxed** for short/low-quality videos: frame stage (blur 3.0, dark 30, bright 235, min face area 0.002, min seq len 32, conf 0.25); rPPG stage (min usable frames 24, min SQI 0.05).

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

## Agent Skills (OpenCode)

This project uses skills installed under `~/.config/opencode/skills/`.

### Core Rules
- If a task matches a skill, invoke it with the `skill` tool before acting.
- Skills are located in `~/.config/opencode/skills/<skill-name>/SKILL.md`.
- Follow the skill workflow strictly; do not partially apply it.
- Never skip required steps such as spec, plan, or test when a skill demands them.

### Intent → Skill Mapping
| User Intent | Primary Skill | Follow-up Skills |
|-------------|---------------|------------------|
| "Build a feature" / "Add X" | `spec-driven-development` | `incremental-implementation`, `test-driven-development` |
| "Plan this work" / "Break it down" | `planning-and-task-breakdown` | — |
| "Fix this bug" / "It's broken" | `debugging-and-error-recovery` | `test-driven-development` |
| "Review this PR" / "Check my code" | `code-review-and-quality` | — |
| "Simplify this" / "Refactor" | `code-simplification` | `test-driven-development` |
| "Design an API" / "Define interface" | `api-and-interface-design` | `spec-driven-development` |
| "Build UI" / "Fix the frontend" | `frontend-ui-engineering` | `test-driven-development` |
| "Improve performance" | `performance-optimization` | `observability-and-instrumentation` |
| "Secure this" / "Audit security" | `security-and-hardening` | — |
| "Set up CI/CD" | `ci-cd-and-automation` | `git-workflow-and-versioning` |
| "Write docs / ADR" | `documentation-and-adrs` | — |
| "Ship / deploy" | `shipping-and-launch` | — |

### Execution Model
For every request:
1. Determine if any skill applies (even a small chance).
2. Load the skill with `skill({ name: "<skill-name>" })`.
3. Follow the skill workflow exactly.
4. Only proceed to implementation once required steps are complete.

### Project-Specific Overrides
- **Test framework:** Python `unittest`/`assert` (no pytest), Vitest + React Testing Library for frontend
- **Lint:** Python `ruff` (if configured), eslint + prettier for frontend
- **Git:** trunk-based, conventional commits
- **Definition of Done:** See `~/.config/opencode/references/definition-of-done.md`