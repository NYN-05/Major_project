# Project Audit Report

Deepfake-video detection for KYC — rPPG physiological evidence + hybrid quantum-classical ML.

- Date: 2026-08-13
- Scope: all tracked code under `WORKING/`, `frontend/`, root docs; runtime artifacts under `WORKING/output/`; git history.
- Method: source review of every tracked module, numerical reproduction of suspected defects, artifact inspection, git-history diffs, measured benchmarks.
- Convention: **[confirmed]** = reproduced/verified directly; **[potential]** = inferred, needs verification; **[doc]** = documentation-only claim.

---

## Post-audit resolution (same day)

All P1 correctness items from §13 are **fixed, re-verified, and regression-guarded**:

| §13 item | Status |
|---|---|
| 4. QAOA beta mixer (§6.1) | Fixed in `qaoa.py` (`_precompute_gates` returns `(cost_gates, mixer_gates)`; mixer driven by `beta`). Guarded by `test_beta_alive`. |
| 5. Hamiltonian verification (§6.2) | Fixed in `_cost_terms` (linear terms now include the `-(1/4)Σ_{j≠i} q_ij` contribution from expanding `(1-Z_i)(1-Z_j)/4`); `verify_hamiltonian` checks 20+ bitstrings; `pipeline.py` hard-asserts `error < 1e-6`. Max error on real data: 7.1e-15. |
| 6. Test suite | Added `WORKING/quantum/tests.py` (5 checks, no pytest) — `python -m quantum.tests` from `WORKING/`, all pass. |
| 7. `face_roi.py` false-accept | Centered-bbox fallback now returns `found=False` (frame rejected → `features=None` → INCONCLUSIVE). |
| 9. `server.py` | CORS restricted to localhost origins (evil-origin POST → 403), uploads streamed to disk (64 KB chunks), 30-min job timeout with worker kill, `FRONTEND_PORT` honored (typo'd `FRONTEMD_PORT` kept as fallback). |
| 10. `plots.py` lazy sklearn | `sklearn.metrics` import moved inside the plot functions. |

**New canonical numbers (post-fix, seed 44, 2026-08-13):** QAOA selection `[hr_half_diff, cheek_forehead_correlation, left_right_cheek_correlation, mad, heart_rate_bpm, spectral_entropy]` (chosen restart seed 43, cost 0.3950); ECE 0.1052 (was 0.2478); acc/f1 unchanged at 0.6667/0.8000. E2E verdict `prob_real=0.6155 → UNCERTAIN, confidence=0.2311`. All audit "before" numbers in §6/§9/Appendix B remain as historical record.

Also verified: port 8000 was held by a stale pre-hardening server instance during the first smoke test — kill lingering `python server.py` processes before retesting.

---

## 1. Executive Summary

The project is a well-engineered three-stage research prototype: stage 1 samples frames and gates quality (YOLO), stage 2 derives 10 physiological features from facial regions via POS/CHROM rPPG (MediaPipe), stage 3 performs QAOA feature selection and a hybrid quantum-classical VQC verdict with REAL/FAKE/UNCERTAIN bins. The pipeline is deterministic, reproducible, and fast (~29 s end-to-end for the quantum layer; recent work cut it from ~81 s).

The audit found **two confirmed defects in the QAOA selection layer**, one of which is a regression introduced by the recent optimization refactor:

1. **[confirmed, regression]** The QAOA ansatz no longer uses the `beta` parameters at all — the X-mixer gates are driven by `gamma`. Beta slots are dead parameters (verified: perturbing `beta` leaves the circuit output bit-identical). This changed the feature-selection outcome vs. the pre-refactor run.
2. **[confirmed, pre-existing]** The encoded Hamiltonian does not match the classical cost function it claims to implement (`verify_hamiltonian` error ≈ 1.94, silently non-zero and not gated).

Beyond these, the dominant limitation is **data scale**: the entire training set is 16 labeled DFDC videos (9 real / 7 fake). Reported metrics (acc 0.67, f1 0.80, AUC 0.50, ECE 0.248; XGBoost baseline 1.0) are not statistically meaningful, and the README's claim of 469 FF++ train clips is false — FF++ data is not used anywhere.

Security is appropriate for a local demo tool (no auth, open artifact serving, 200 MB in-memory uploads) but would need real hardening before any multi-user KYC deployment.

**No project tests, CI, or linters exist** — the two quantum defects above would have been caught by ~10 lines of assertions.

---

## 2. Project Overview

### Architecture

```
Input KYC video
  → Stage 1 (WORKING/frame): sample 10 fps → YOLO face detect → blur/dark/bright/face-size/pose gates
  → Stage 2 (WORKING/RPPG): MediaPipe Face Landmarker ROIs (forehead, L/R cheek) → POS/CHROM pulse → 10 physiological features
  → Stage 3 (WORKING/quantum): QAOA selects 6 of 10 features → hybrid VQC (angle embedding, torch head) → P(real) → REAL / FAKE / UNCERTAIN
```

Orchestrated by `WORKING/run_pipeline.py`, exposed via `frontend/server.py` (stdlib HTTP + SSE) driving a React/Vite UI.

### Data flow

- `WORKING/output/rppg/dataset_features.csv` — **16 rows** (9 label `0` real, 7 label `1` fake), all from `archive/DFDC_Dataset`. This is the only labeled data consumed.
- `quantum/data.py` splits 16 rows → 10 train / 3 val / 3 test (stratified, seed 44), writes `output/quantum/data.npz`.
- QAOA selection on the train rows → `qaoa_selection.json` (current selection: `cheek_forehead_correlation`, `left_right_cheek_correlation`, `mad`, `heart_rate_bpm`, `prv_std_ms`, `spectral_entropy`).
- VQC trained on the 6 selected features → `hybrid_vqc.pt`; `run_pipeline.py` scores incoming videos through the same path.

### Repo inventory

- 150 tracked files; ~5,785 lines of tracked Python across `WORKING/` (frame ~2,000, RPPG ~2,000, quantum ~1,000, run_pipeline ~500) plus `frontend/server.py` (466) and ~1,500 lines of React.
- 13 commits on `main` (2026-08-12 → 2026-08-13). Working tree clean except `WORKING/quantum/evaluation.py` (ProcessPoolExecutor CV change, not yet committed).

---

## 3. Strengths

**[confirmed]**

- **Clean stage separation with a single orchestrator.** `run_pipeline.py` is the only entry point for inference; it records `input_mode` (frame-driven vs. direct video) and degrades gracefully: if stage 1 yields no frames it falls back to `RPPGPipeline.process_video()`. `features=None` (usable frames < 48) → INCONCLUSIVE + exit 3, documented and handled.
- **Deterministic, reproducible pipeline.** Fixed seeds (44), fixed hyperparameters, restart aggregation — full `--all` run reproduced bit-identical metrics across runs.
- **rPPG implementation quality.** POS/CHROM extraction is clean and numerically verified (vectorized POS bit-identical to the original loop, max |diff| = 0.0 across 7 (T, fs) pairs). Shared Welch PSD, cached filter coefficients, hoisted skin-mask — sensible micro-optimizations.
- **Recent performance work is real.** Full quantum flow 81.1 s → 28.8 s (−65%). QAOA: lightning.qubit + precomputed gates (18 ms/call), 4 parallel COBYLA restarts via `ProcessPoolExecutor` (~15 s vs ~37 s sequential). Imports 22 s → ~6.5–9.5 s. VQC trains on CUDA (15.2 s). Inference 32 ms cached.
- **"No synthetic data" discipline.** The quantum layer consumes only the real feature table; no generators/bridge layers exist. This is the right call for credibility and is enforced as a documented constraint.
- **Label-convention hazards are documented** (rPPG 1=fake vs quantum 1=real vs RF 1=DEEPFAKE) instead of silently unified — an intentional, documented design decision.
- **Frontend hardening.** Magic-byte upload validation, 200 MB cap, concurrency cap (2) with 429, job TTL (1 h) + frame-sequence TTL (24 h), sanitized inbox filenames, result-first SSE (client renders verdict before the heavy `signal` artifact is ready), path-traversal guard on `/api/files`.
- **Frontend code quality.** Well-structured React (12 small components, clean hooks), theme persistence with `prefers-reduced-motion` respect, responsive (verified 1920→375 px), no console errors in E2E.
- **Docs that are genuinely useful** (`WORKING/frame/README.md` output contract, `frontend/README.md` API contract, `AGENTS.md` verified constraints) — see §12 for the stale items.

---

## 4. Bottlenecks

1. **[confirmed] Data — the binding constraint.** 16 labeled rows. QAOA MI-weights degenerate (`"success": false`, mostly-zero weights — expected at this size), XGBoost baseline reaches acc = 1.0 (overfit), AUC = 0.50. No metric in the project generalizes. Everything else is secondary.
2. **[confirmed] rPPG stage latency.** MediaPipe Face Landmarker runs per frame at 10 fps sampling with no GPU path and no frame-pacing parallelism. For a 2-minute KYC video (~1,200 frames) this is the dominant wall-clock cost in `run_pipeline.py` (the ~29 s figure above is quantum-layer only).
3. **[confirmed] Serial bottleneck in inference.** `run_pipeline.py` is single-process; the frontend caps concurrency at 2 and each job holds a slot for the full run. No per-stage caching (e.g., frame extraction is redone on every run).
4. **[confirmed] Import cost.** `import quantum.pipeline` ≈ 6.5–9.5 s (torch 5.9 s, pennylane 2.3 s, sklearn ~2.4 s via `plots.py`). Felt on every server start and every QAOA spawn (mitigated by lazy xgboost + `ProcessPoolExecutor`).
5. **[confirmed] Feature-table regeneration requires manual orchestration.** Adding data = extract → train RF → rerun `--all`; no single command, no caching of the extract step (MediaPipe re-download/re-run per video).

---

## 5. Limitations

**[confirmed] unless noted**

- **Dataset:** 16 DFDC videos; no FF++ data despite README claims; no frame-level labels; no test-timer/compression/identity-swap variety beyond what DFDC provides; no provenance columns beyond `source` string.
- **Metrics meaningless at this scale:** acc 0.6667, f1 0.8000, AUC 0.5000, ECE 0.2478, and baselines (RF/XGBoost acc 1.0) — no split size is adequate to claim detection capability.
- **QAOA selection is non-deterministic in outcome space:** COBYLA restarts from random initial angles; identical seed gives identical results, but the selected feature set changed between refactors (see §6.1) and is sensitive to the ansatz implementation.
- **`features=None` gating:** videos with < 48 usable frames (short clips, heavy occlusion, small faces) are INCONCLUSIVE (exit 3) by design.
- **Face-landmark dependency:** MediaPipe Face Landmarker model auto-downloads from Google storage on first use (internet required, no pinned hash).
- **Local-only deployment model:** in-memory job store, no persistence, no auth (see §10–11).
- **`lightning.gpu` / cuQuantum unavailable on Windows** (no `custatevec` wheels) — CPU SIMD is the ceiling for the QAOA stage; only the torch head of the VQC uses the GPU.

---

## 6. Failure Points

### 6.1 QAOA `beta` mixer regression — [confirmed]

**Location:** `WORKING/quantum/qaoa.py`, `_precompute_gates()` / `_apply_qaoa()` (introduced by the precompute-gates optimization in commit `e74fbc8`).

**What happens:** `_apply_qaoa` unpacks `gamma, beta` per layer but only `gamma` is used. The X-mixer rotations that `_precompute_gates` adds (with coefficient c=1.0) are appended to the cost-gate list and therefore also driven by `gamma`. `beta` never reaches the circuit.

**Numerical proof** (run against the real code):

| experiment | original ansatz (git fc61a54) | current code |
|---|---|---|
| cost at fixed params | 11.5206 | 13.1548 |
| cost after perturbing **only beta slots** | 4.2043 (beta works) | **13.1548 (bit-identical — beta dead)** |

**Consequences:**
- Half the variational parameters are dead; the effective search space is halved and the ansatz is a non-standard evolution (two sequences of the same rotation type).
- The selection outcome changed vs. pre-refactor: was `[snr_db, signal_quality_index, peak_prominence, prv_std_ms, spectral_entropy, heart_rate_bpm]`; now `[cheek_forehead_correlation, left_right_cheek_correlation, mad, heart_rate_bpm, prv_std_ms, spectral_entropy]`. Both results are "plausible", which is why the regression went unnoticed — only reproducibility tests on the ansatz would have caught it.

**Fix:** in `_apply_qaoa`, apply the precomputed cost gates with `gamma` and a separately-precomputed X-mixer gate list with `beta` (still reusing the precompute speedup). Then re-run `--all` and record the new selection as canonical.

### 6.2 Quantum Hamiltonian ≠ classical cost — [confirmed, pre-existing]

**Location:** `WORKING/quantum/qaoa.py`, `_cost_terms()` vs `_classical_cost()`; `verify_hamiltonian` prints a number but never fails.

**Evidence:** bitstring all-ones → quantum 6.4092 vs classical 7.3493 (err 0.9401); all-zeros → 19.5599 vs 18.0000 (err 1.5599); pipeline printed max err ≈ 1.94e+00. The hand-derived (1−Z)/2 substitution in `_cost_terms` does not reproduce the classical objective.

**Consequences:** QAOA optimizes a *different* pseudo-boolean objective than documented. It is still structurally reasonable (MI term + redundancy penalty + cardinality penalty), so the selection is not garbage — but it is not the objective the paper/README describes, and the silent verify gate gives false confidence.

**Fix:** either correct the Hamiltonian derivation and make `verify_hamiltonian` a hard assertion over a batch of bitstrings (needs ≤ 2ⁿ ≤ 1024 evals for n=10; all-zero/all-ones alone catch this), or drop the Hamiltonian-verification claim and document that QAOA minimizes the encoded quadratic form directly.

### 6.3 Stale/false claims about the dataset — [confirmed, doc]

- `README.md` claims 469 FF++ train clips / 669 labeled clips — **no FF++ data is used**; the real table is 16 DFDC rows.
- `AGENTS.md` says "18 labeled rows" — actual: **16** (9 real, 7 fake).
- `WORKING/run_pipeline.py` docstring: "8-feature vector" — actual contract is **10 features** (both `quantum/config.py` and `RPPG/rppg/features.py`).
- `AGENTS.md` "Optimizations" lists a **`video_meta` probe in `run_pipeline.py` that does not exist** (no `VideoCapture` in the file at all — the claim is fabricated).

### 6.4 Frame-stage robustness — [confirmed]

- `WORKING/frame/app/pipeline.py` dispatches on CLI flags (`--source` vs `--input-dir` vs `--compare-rates`...) with no error path for ambiguous combinations; failures surface as confusing partial artifacts.
- `app/main.py` ("legacy entry") also triggers `extract_frames` — two entry points doing overlapping work.
- YOLO weights auto-download from GitHub without a pinned hash; other presets raise `FileNotFoundError` (documented).

### 6.5 rPPG feature extraction tooling — [confirmed]

- `WORKING/RPPG/rppg-pipeline/extract_dataset_features.py`: redirects stderr to `NUL` (crash diagnostics swallowed), uses `multiprocessing.Pool` which can hang permanently on a worker crash, and has hardcoded FF++-flavored path/source strings while actually reading DFDC (`--workers 0` is the documented workaround).
- `RPPG/rppg/features.py`: docstring says missing values are "median"-imputed; the code replaces them with hardcoded `FALLBACK_*` constants — the two are not the same and the fallbacks are silently used in training.

### 6.6 Face-ROI false-accept fallback — [confirmed]

`WORKING/RPPG/rppg/face_roi.py`: when MediaPipe finds no landmarks, a centered bounding-box ROI is used and `found=True, confidence=0.4` is returned — the pipeline proceeds to compute features on a guessed ROI instead of rejecting the frame. For adversarial/no-face KYC inputs this is a false-accept path. (The `min_usable_frames=48` gate is the only safety net, and it is per-video, not per-frame.)

### 6.7 CV evaluation worker crash — [confirmed, already fixed locally]

`WORKING/quantum/evaluation.py` previously used `multiprocessing.Pool` (deadlocks if a worker dies mid-fold). Replaced with `ProcessPoolExecutor` — verified 5 parallel folds in 6.1 s. Change is uncommitted.

---

## 7. Technical Risks

- **[potential]** **Feature-contract drift:** `FEATURE_NAMES` is duplicated between `quantum/config.py` and `RPPG/rppg/features.py` (documented as intentional). Any reorder/rename silently misaligns `data.npz` columns with the QAOA index and the VQC input size; nothing validates the two lists at runtime.
- **[potential]** **Pickle trust boundary:** `run_pipeline.py` and the RF cross-check `pickle.load` `rppg_classifier.pkl`. Acceptable for locally-generated artifacts, dangerous if the output directory is ever writable by an attacker (it is, via the frontend uploads).
- **[confirmed]** **Model cache keyed on (n_features, mtime, size):** `vqc.py` reuses loaded weights across requests; a tampered-but-same-size checkpoint would be served silently.
- **[confirmed]** **No runtime check that the scaler/checkpoint/feature-set are mutually consistent** in `run_pipeline.py` — stale artifacts (e.g., 8-feature checkpoint + 10-feature scaler) fail only at shape-mismatch time with an opaque torch error.
- **[potential]** **rPPG signals on low-resolution / compressed KYC feeds** (the stated target) degrade SNR; thresholds (min_usable_frames, quality gates) were tuned on a handful of clips.
- **[confirmed]** **Frontend 404 noise:** stale runs reference missing (gitignored) frame thumbnails — cosmetic, but signals a TTL/path coupling that could bite at higher concurrency.

---

## 8. Code & Architecture Observations

- `quantum/config.py` centralizes hyperparameters and documents each stage — good. All numbers that matter (seeds, layers, restarts, n_jobs) live there.
- `quantum/data.py` correctly flips labels (csv 1=fake → quantum 1=real) and records provenance in `data.npz`; split is stratified on the tiny set.
- `quantum/evaluation.py` reports ECE + calibration plot — a level of rigor above the project's scale.
- `quantum/plots.py` imports `sklearn.metrics` at module level and is imported eagerly by `evaluation.py` → the "sklearn is lazy" claim in `AGENTS.md` is only half true (xgboost is lazy; sklearn costs ~2.4 s on every `import quantum.pipeline`). Consider moving the import inside the three plot functions.
- `frontend/server.py` (466 lines, stdlib only) is well-factored: `_run_job` releases the semaphore in `finally`, TTL cleanup is periodic, filename sanitization `{job8}_{stem}.ext` is applied for thumbnail consistency. The `FRONTEMD_PORT` env name is a typo (documented in `AGENTS.md`, harmless but confusing).
- `App.jsx` restores the last canonical result on load (`/api/previous`) — nice touch for a demo tool.
- Legacy/unused surface: `rppg-pipeline/streamlit_app.py`, `run_live_webcam.py`, `batch_run.py`, `retrain_dfdc.py`, `debug_run.py`, `_make_test_video.py`, `_check_video.py`, `frame/app/main.py`, `quantum/scaling.py`'s `SCALER_FILE` duplication. None are wired into the main path; they should be pruned or clearly marked.
- `.gitignore` ignores all `*.json` (including `package.json` — negated explicitly) and all `output/` — correct for regenerable artifacts, but it means **every clone must regenerate `dataset_features.csv` + quantum artifacts** before anything runs; no bootstrap script exists.

---

## 9. Performance

**[confirmed]** Measurements on this host (Windows, RTX 4050 Laptop GPU, 16 logical CPUs, 24 GB RAM; Python 3.10; torch 2.5.1+cu121; PennyLane 0.42.3; pennylane-lightning 0.42.0):

| Item | Before | After | Notes |
|---|---|---|---|
| `python -m quantum.pipeline --all` | 81.1 s | **28.8 s** | identical metrics across runs |
| QAOA stage (4 restarts) | ~37 s (sequential, default.qubit) | **14.6 s** | lightning.qubit, 18 ms/gate-circ, ProcessPoolExecutor |
| `import quantum.pipeline` | ~22 s | **6.5–9.5 s** | xgboost fully lazy; sklearn still eager via plots.py (~2.4 s) |
| VQC train (10 rows × 6 feats) | — | **15.2 s** | torch head CUDA; QNode default.qubit + backprop (lightning adjoint was slower) |
| Inference (cached model) | — | **32 ms** | ~275 ms cold; cache keyed on (n_features, mtime, size) |
| Parallel CV (5 folds) | hung (mp.Pool) | **6.1 s** | ProcessPoolExecutor, uncommitted |

- Stage-1 (frame) YOLO runs on GPU via ultralytics; stage-2 (rPPG) is CPU-only (MediaPipe per-frame) and is the wall-clock dominator for long videos.
- Upload/artifact serving is stdlib; SSE is chunked and result-first.

---

## 10. Security

**[confirmed] posture:** local single-user tool. Findings, in severity order:

1. **No authentication** anywhere; any local process (or, with the frontend's CORS, any web page) can POST to `/api/detect` and read `/api/jobs/*`.
2. **CORS `*`** (`server.py`): a malicious website can initiate uploads to `127.0.0.1:8000` and read results (CSRF-style). Restricted to localhost origins would be trivially safer.
3. **Open artifact serving:** `/api/files?rel=` serves *any* file under `output/` (path-traversal guarded, but no authorization) — includes accepted-frame JPEGs (faces) and result JSONs (KYC PII) for every processed video.
4. **Uploads buffered fully in memory** (up to 200 MB × 2 concurrent jobs = 400 MB potential spike against 24 GB) — stream-to-disk would be safer and allow a hard disk quota.
5. **Subprocess runs untrusted-adjacent code** (`run_pipeline.py` with no timeout): a pathological input (e.g., huge resolution) can hold a job slot indefinitely; the semaphore is released in `finally`, but the worker process is not killed.
6. **Unsafe deserialization:** `pickle.load` of `rppg_classifier.pkl` (see §7); unauthenticated auto-downloads (YOLO weights, MediaPipe model) without pinned hashes.

For a local research demo these are acceptable; for any multi-tenant KYC deployment, items 1–5 are release-blockers.

---

## 11. Scalability

**[confirmed]**

- **Compute:** single node; max 2 concurrent pipeline runs; quantum inference is cached (32 ms) but rPPG per-frame CPU work is the serial bottleneck.
- **State:** jobs live in an in-memory dict + one JSON result file (`output/pipeline/`); no database, no job history beyond the last canonical result (`/api/previous`), no metrics/audit log.
- **Storage:** frame sequences retained 24 h, TTL sweep is periodic; 200 MB upload cap.
- **Multi-tenant:** not designed for it (see §10). The architecture (stages as subprocess, artifacts on disk, SSE) is *plausibly* shippable behind a real server with auth + queue + object storage, but the rPPG stage would need GPU/parallelism and the dataset problem (§5) dominates any scale discussion.

---

## 12. Documentation Issues

**[confirmed]**

| Doc | Issue |
|---|---|
| `README.md` | Claims 469 FF++ train clips / 669 labeled; actual: 16 DFDC rows, no FF++. Layout says "8 physiological features" (actual 10). |
| `AGENTS.md` | "18 labeled rows" (actual 16); "video_meta probe" optimization bullet does not exist in code; "sklearn lazy" is half-true; missing QAOA beta-mixer + Hamiltonian-mismatch warnings. |
| `WORKING/run_pipeline.py` docstring | "8-feature vector" (actual 10). |
| `Docs/projec_audit.md` | Pre-refactor snapshot (8 features, old module names `selection.py`/`evaluate.py`/`run.py`; claims frame stage doesn't feed rPPG — fixed since). Should be marked historical. |
| `WORKING/RPPG/rppg/features.py` docstring | "median imputation" vs actual hardcoded fallback constants. |
| `WORKING/frame/README.md`, `frontend/README.md` | Accurate (verified `rppg_checks` exists at `frame/app/pipeline.py:542`; API contract matches `server.py`). |

---

## 13. Prioritized Recommendations

**P0 — data credibility (project-level):**
1. Fix the dataset claims in `README.md` + `AGENTS.md` (16 DFDC rows; no FF++).
2. State plainly in the README that current metrics are indicative only at this scale; add a "Dataset" section with composition, provenance, and how to extend.
3. Add a bootstrap/regeneration script (`extract → train RF → --all`) so a fresh clone can reproduce artifacts.

**P1 — correctness (both confirmed bugs):**
4. Fix the QAOA `beta` mixer (§6.1); re-run `--all`; record the new canonical selection.
5. Fix or gate the Hamiltonian verification (§6.2) with a hard assertion over sampled bitstrings.
6. Add a minimal test suite (~10–15 assertions, no framework needed): beta affects cost; Hamiltonian ≈ classical on all-0/all-1/sampled bitstrings; `FEATURE_NAMES` (config) == `RPPGFeatures.feature_names()` order; data split sizes; determinism (same seed → same selection). This would have caught both bugs.

**P2 — robustness:**
7. `face_roi.py`: on no-landmark fallback, return `found=False` (route to `features=None` → INCONCLUSIVE) or drop confidence to a level the pipeline rejects.
8. `extract_dataset_features.py`: remove stderr→NUL, replace `mp.Pool` with `ProcessPoolExecutor`, parametrize source paths.
9. `server.py`: restrict CORS to `localhost`/`127.0.0.1` origins; stream uploads to disk; add per-job timeout + worker kill; consider a token for `/api/files`/`/api/detect` even on localhost.
10. `plots.py`: move `sklearn.metrics` import into functions (saves ~2.4 s off every process start).

**P3 — hygiene:**
11. Prune/flag legacy scripts (`streamlit_app.py`, webcam/batch/debug runs, `main.py` legacy path).
12. Fix stale docstrings (`run_pipeline.py` 8→10 features; `features.py` median→fallback).
13. Pin hashes for auto-downloaded models; add runtime shape/feature-count assertions in `run_pipeline.py`.
14. Mark `Docs/projec_audit.md` as historical (superseded by this report + `AGENTS.md`).

---

## 14. Overall Assessment

This is a **solid, well-documented research prototype** whose engineering quality (determinism, clean stage separation, careful numerical work, honest fallbacks, a genuinely nice frontend) is well above typical student-project level — but whose scientific claims are currently limited by two things: a 16-row dataset, and two confirmed QAOA-layer defects (one a refactor regression, one pre-existing and silently hidden by a non-failing verification).

The fixes are small and surgical (§13 P1 items 4–6, maybe 40 lines total + a test file). The dataset question is the real strategic decision: either grow the DFDC sample and re-baseline, or explicitly reposition the project as a framework/demonstrator rather than a trained detector. Either way, the report should be read as "two concrete bugs + one existential data question," not as a failing audit — the architecture and implementation are the strong foundation the project needs.

---

## Appendix A — Reproduction commands

```bash
# from WORKING/
python -m quantum.pipeline --all                       # full flow, 28.8 s, seed 44
python run_pipeline.py --source <video> --method POS    # end-to-end inference

# beta-mixer proof (see §6.1)
python -c "import numpy as np; from quantum.qaoa import _cost_terms, _precompute_gates, simulator_device, _classical_cost, _normalize_weights, _mutual_info_weights; from quantum.config import QAOASelectionConfig; ..."  # see §6.1 numbers

# Hamiltonian-vs-classical proof (see §6.2)
python -c "..."  # quantum expval vs _classical_cost on all-zeros / all-ones bitstrings

# dataset truth
python -c "import csv; rows=list(csv.DictReader(open('output/rppg/dataset_features.csv'))); print(len(rows), sum(r['label']=='0' for r in rows), sum(r['label']=='1' for r in rows))"
# → 16 9 7
```

## Appendix B — Reference numbers (single host, 2026-08-13)

- VQC metrics (test split, 3 samples): acc 0.6667, f1 0.8000, AUC 0.5000, ECE 0.2478.
- Baselines: RandomForest acc 1.0, XGBoost acc 1.0 (overfit).
- QAOA selection (current code): `cheek_forehead_correlation, left_right_cheek_correlation, mad, heart_rate_bpm, prv_std_ms, spectral_entropy`, seed 44, cost 1.3209.
- QAOA selection (pre-refactor, fc61a54): `snr_db, signal_quality_index, peak_prominence, prv_std_ms, spectral_entropy, heart_rate_bpm`.
- E2E inference check: `WIN_20260729_11_14_42_Pro.mp4`, POS → prob_real 0.5475 → UNCERTAIN.
