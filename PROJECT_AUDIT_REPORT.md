# Project Audit Report — Full-Project Review

Deepfake-video detection for KYC — rPPG physiological evidence + hybrid quantum-classical ML.

- Date: 2026-08-13 (second, exhaustive pass)
- Scope: every tracked file in `WORKING/` (frame stage, RPPG stage, quantum stage, `run_pipeline.py`), `frontend/` (server.py, React app), root docs (README, AGENTS.md, .gitignore, requirements, SKILL.md, result.json), `Docs/`, and runtime artifacts (`WORKING/output/`).
- Method: full source review of all 35 tracked `.py` modules, numerical verification of suspected defects, artifact inspection (CSV, JSON, pkl, checkpoints, plots), git inventory (`git ls-files`), measured timings.
- Conventions: **[confirmed]** = verified directly in code/artifacts; **[potential]** = plausible, needs further verification; **[doc]** = documentation claim only.
- Severity scale: **Critical** (data integrity / security / silent wrong output) · **High** (crash paths, misleading public claims) · **Medium** (fragility, dead config, drift) · **Low** (cosmetic / hygiene).

---

## Post-audit resolution (earlier same day — history, do not re-report as open)

The previous audit found two QAOA-layer defects and other issues; all were fixed, re-verified, and regression-guarded in commit `a11f06e`:

| Item | Status |
|---|---|
| QAOA beta-mixer regression (mixer driven by `gamma`) | Fixed: `_precompute_gates` returns `(cost_gates, mixer_gates)`; mixer driven by `beta`. Guarded by `test_beta_alive` in `quantum/tests.py`. |
| Hamiltonian ≠ classical cost (silent, err ≈ 1.94) | Fixed: `_cost_terms` now includes the `-(1/4)Σ_{j≠i} q_ij` linear term from `(1-Z_i)(1-Z_j)/4`; `verify_hamiltonian` checks all-zeros/all-ones/single-bit/random bitstrings; `pipeline.py` hard-asserts `error < 1e-6`. Max verified error: 7.1e-15. |
| No test suite | `WORKING/quantum/tests.py` added (5 checks, no pytest). `python -m quantum.tests` — all pass. |
| `face_roi.py` centered-bbox false-accept | Fallback now returns `found=False` (frame rejected → `features=None` → INCONCLUSIVE). |
| `server.py` hardening | CORS allowlist (localhost:5173/8000, 127.0.0.1:5173/8000); evil-origin POST → 403; uploads streamed in 64 KB chunks with 4 KB magic-check prefix; 30-min job timeout with `proc.kill()`; `FRONTEND_PORT` honored (`FRONTEMD_PORT` deprecated fallback). |
| `plots.py` eager sklearn | sklearn import moved inside plot functions (lazy). |

**Canonical post-fix numbers:** QAOA selection `[hr_half_diff, cheek_forehead_correlation, left_right_cheek_correlation, mad, heart_rate_bpm, spectral_entropy]` (chosen restart seed 43, cost 0.3950); ECE 0.1052 (was 0.2478); acc 0.6667 / F1 0.8000 / AUC 0.5000; E2E verdict `prob_real=0.6155 → UNCERTAIN, confidence=0.2311` (84/84 usable frames).

---

## 1. Executive Summary

The project is a cleanly separated, deterministic three-stage prototype whose documented engineering claims (structure, reproducibility, degradation behavior, no synthetic data) hold up under full review. The quantum-layer correctness fixes from the morning audit are verified in place with regression guards.

This second, exhaustive pass found **no new critical defects**, but it did confirm:

1. **A silent data-integrity flaw at inference** (High): degenerate rPPG signals are *fabricated* into plausible-looking feature vectors instead of being rejected (`features.py::_fill_nan_with_median`).
2. **An unguarded crash path in the rPPG cross-check** (High): a stale or single-class `rppg_classifier.pkl` crashes `run_pipeline.py` after the rPPG stage (`predict_proba[0][1]` outside any try/except), killing server jobs.
3. **Public documentation that describes a system that does not exist** (High): the root `README.md` documents an FF++-trained model (469 clips, acc 0.564, AUC 0.640) and calls the actual DFDC data "false-positive noise" — while the real trained model uses exactly 16 DFDC rows.
4. **Dataset-quality observations** (Medium): all 16 labeled rows have *negative* SNR; HR features are quantized to a coarse ~24.3 BPM grid; the train set is 10 rows. Metrics are indicative only.
5. **Widespread "8 features" → 10-feature documentation drift** across code, README, frontend UI (Medium/Low).
6. **Broken documented commands**: `app/main.py` and `app/extract_frames.py` do not exist; README and AGENTS.md point at them.
7. **Hygiene issues**: ~10 MB of binaries + 60 PDFs + duplicated docs committed to git; dead config (`DataConfig.train_ratio`); a dependency (`xgboost`) used unconditionally by `--all` but present in no requirements file.

Strengths confirmed: deterministic seeds and split logic, correct `features=None`/INCONCLUSIVE degradation, stage-1→stage-2 handoff with fallback, server hardening, no synthetic data, and effective numerics (vectorized POS, shared Welch PSD, cached filters, lazy heavy imports).

---

## 2. Repo Inventory (verified)

- 146 tracked files: 60 PDF, 35 `.py`, 15 `.jsx`, 14 `.md`, 7 PNG, 5 TXT, 4 `.js`, 3 DOCX, 2 `.pt`, 2 JSON, 1 CSS, 1 `.ps1`, 1 `.gitignore`, 1 HTML, 1 MP4.
- No untracked non-ignored files (`git status` clean after `a11f06e`).
- Python LOC (tracked): frame stage ~2,000; RPPG ~2,000; quantum ~1,200; `run_pipeline.py` 294; `frontend/server.py` 510; React ~1,700.
- Line counts of key modules: `frame/app/pipeline.py` 770, `RPPG/rppg/pipeline.py` ~330, `RPPG/rppg/face_roi.py` 411, `quantum/qaoa.py` 256, `quantum/evaluation.py` 236, `quantum/vqc.py` 233, `quantum/data.py` 261, `frontend/server.py` 510.
- Tracked binaries (bloat): `WORKING/frame/weights/yolov8n-face-lindevs.pt` (6,281,321 B), `yolov9t-face-lindevs.pt`, `WORKING/frame/test.mp4` (1,890,734 B); 60 PDFs under `Docs/maj_proj_report_sem_6/`; duplicated `implementation_work_division_checklist.{pdf,txt}` and `project_system_document.{pdf,txt}` at repo root AND `WORKING/frame/Docs/`.
- `frontend/package-lock.json` is tracked (deps pinned); `package.json` name is `frontemd` (typo).

---

## 3. Confirmed Findings — High

### H1. Degenerate signals are silently converted into "average human" features (silent wrong evidence)
- **Files:** `WORKING/RPPG/rppg/features.py:334-354` (`_fill_nan_with_median`), consumed by `RPPG/rppg/pipeline.py::_finalize` and `run_pipeline.py`.
- **Evidence:** The fallback table is hardcoded: NaN `heart_rate_bpm → 72.0`, `spectral_entropy → 0.5`, `peak_prominence → 1.0`, `snr_db → 0.0`, etc. A totally flat/degenerate pulse (e.g., no face ROI signal, static scene, heavily compressed face) therefore produces the feature vector of a *plausible healthy person* (HR 72 BPM, mid entropy) instead of being rejected. `signal_quality_index → 0.0` is the only honest "bad" feature, and nothing consumes it to reject the clip. The quantum layer's plausibility filter (`quantum/data.py:89-98`) only drops non-finite values, which can no longer occur after filling.
- **Consequence:** At inference, garbage input can score as "likely real" (or fake) on fabricated physiology — precisely the failure mode the system claims to detect. At training time it would inject fake-neutral rows.
- **Fix:** treat low-SQI/NaN-heavy signals as `features=None` (INCONCLUSIVE) at the pipeline level, or add a `degenerate_signal` flag carried through `run_pipeline.py`; at minimum, log a warning when >2 fallbacks fire.

### H2. rPPG cross-check crash path: stale/single-class classifier kills the pipeline
- **Files:** `WORKING/run_pipeline.py:183-198`; enabler `WORKING/RPPG/rppg-pipeline/retrain_dfdc.py:194-204`.
- **Evidence:** `pickle.load` is wrapped, but `clf.predict_proba(x)[0]` and `clf.predict(x)[0]` are not. `retrain_dfdc.py` writes `rppg_classifier.pkl` **incrementally** — a checkpoint is saved after every 10 processed videos (default), which can be trained on a single class (e.g., first 10 sorted clips all Fake). sklearn `RandomForestClassifier.predict_proba` on a single-class fit returns a 1-column array → `proba[1]` raises `IndexError`. A stale pkl with a different feature count raises `ValueError`. Either crashes `run_pipeline.py` after the rPPG stage → server job ends in error.
- **Consequence:** One bad checkpoint (or an old pkl left in `output/rppg/`) breaks every subsequent run. The cross-check is informational; it must never be able to kill the job.
- **Fix:** wrap the predict/proba calls in try/except and emit `{"skipped": ...}`; validate `clf.n_features_in_ == 10` and `clf.n_classes_ == 2` before use.

### H3. README documents a model that does not exist
- **Files:** `README.md:118-123` ("Current trained model": 469 FF++ clips, test acc 0.564, AUC 0.640, ECE 0.073, GaussianNB 0.621) and `README.md:166` ("currently an FF++-sourced table of 669 labeled clips; the DFDC row subset is false-positive noise and excluded").
- **Evidence:** The actual training table is `WORKING/output/rppg/dataset_features.csv` — 16 rows, all `archive/DFDC_Dataset` (9 real / 7 fake). FF++ is gitignored, not used (`--include-ffpp` defaults off), and no 469/669-clip artifacts exist in `output/quantum/` metrics. The README's "Verified Constraints" section also claims the official FF++ split path is active; `data.py::_split_by_source` is dead code for the current data.
- **Consequence:** Anyone evaluating the project from the README gets a completely wrong picture of the deployed model and its validity. The §H1/H2 issues compound this.
- **Fix:** rewrite the Data/Model/Constraints sections to the 16-row DFDC reality (AGENTS.md already reflects it; README was not updated).

---

## 4. Confirmed Findings — Medium

### M1. Broken documented commands
- `README.md:51-53` and `AGENTS.md` instruct `python app/main.py --source test.mp4 --save-metadata` and `python app/extract_frames.py ...`. Neither file exists (`WORKING/frame/app/` contains only `config.py, detector.py, pipeline.py, processing.py`; verified via `git ls-files`). Both commands fail with `ModuleNotFoundError`.
- The real entry point is `python app/pipeline.py --source ...` (`extract` mode via `--input-dir`/`--sample-fps` flags). Fix the docs (or add shim modules).

### M2. "8 features" → 10-feature drift (confirmed, 8 sites)
- The feature contract is 10 features (both `FEATURE_NAMES` and `RPPGFeatures.feature_names()` verified identical). Stale "8" claims remain in:
  - `WORKING/run_pipeline.py:13` (docstring), `WORKING/quantum/pipeline.py:50` ("keyed by the 8 rPPG feature names"),
  - `WORKING/RPPG/README.md:6`, `WORKING/RPPG/rppg-pipeline/retrain_dfdc.py:8` ("the 8 rPPG features"),
  - `frontend/src/lib.js:43` (pipeline step "Measuring 8 physiological features") and `lib.js:116` (QAOA glossary "from the 8 extracted ones"),
  - `frontend/src/components/QuantumFlow.jsx:7` ("8 features") and `:88` ("Six of the eight rPPG features").
- Additionally `frontend/src/lib.js:132-141` `FEATURE_LABELS` lists only **8 of 10** features — `hr_half_diff` and `peak_prominence` (both selected by QAOA and displayed nowhere) are missing from the Insights panel; `lib.js:123` describes MAD as "derivative magnitude approach" while `features.py:182` computes plain mean absolute deviation.
- Fix: sweep `8`→`10` and add the two missing labels (the QAOA "Selected by QAOA" tags in `QuantumFlow.jsx` already fall back to raw names for them).

### M3. Dead/misleading split configuration
- `quantum/config.py:43` declares `DataConfig.train_ratio = 0.6`, which is **never read**; `quantum/data.py:130-131` computes `test_c = round(per_class * val_ratio)` — the test ratio is the val ratio, there is no `test_ratio` field. Changing `train_ratio` silently does nothing; the split is train = remainder, val = test = 0.2.
- Fix: remove `train_ratio`, add `test_ratio` (or document that val and test share the ratio).

### M4. `predict_features` has no artifact-consistency validation
- `WORKING/quantum/pipeline.py::predict_features` applies saved QAOA indices, a saved scaler, and a saved checkpoint independently. If artifacts are rebuilt out of sync (e.g., new `--select` run + old checkpoint, or a checkpoint from a different feature count), inference fails with opaque shape errors deep inside torch/sklearn instead of a clear message. The stale "8 rPPG feature names" docstring (`pipeline.py:50`) makes this worse.
- Fix: at load time assert `scaler.mean_.shape == (len(FEATURE_NAMES),)`, `selection indices < 10`, and `checkpoint input dim == len(selection)`; raise a descriptive error.

### M5. Fallback docstring/code mismatch + `-inf` SNR edge
- `features.py:334-354` docstring claims "per-feature median of non-NaN values" — the code only uses hardcoded constants (median is never computed).
- `estimate_snr` returns `-inf` when signal power is zero (`features.py:134`); `np.isnan(-inf)` is `False`, so the fallback does not replace it — `-inf` propagates to the CSV (silently dropped by `data.py`'s isfinite filter) or, at inference, into the scaler → NaN probability. Confirm the scaler/`predict_features` behavior for non-finite inputs (see P3).

### M6. Timestamp-based frame sampling can silently collapse to 1 frame
- `WORKING/frame/app/processing.py:100-126`: when `sample_fps` is set, sampling keys off `CAP_PROP_POS_MSEC`. For containers/codecs where OpenCV reports `POS_MSEC == 0` for every frame, only the first frame passes (`0 >= next_target(0)`, then `next_target=100` never reached) → stage 1 yields 1 frame → rPPG `features=None` → INCONCLUSIVE, with the root cause invisible in logs.
- Fix: detect `all(timestamp == 0)` and fall back to stride-based sampling (`(source_frame_id-1) % step == 0`) with a warning.

### M7. Extraction workers hide crash diagnostics (and can hang the pool)
- `extract_dataset_features.py:40-58` and `probe_features.py:40-47` redirect worker `stderr` to NUL (`os.dup2`). Any worker crash prints nothing; with `mp.Pool.imap_unordered`, a hard worker death can hang the parent indefinitely (no timeout). `chunksize=1` maximizes pickle round-trips.
- Fix: send `traceback.format_exc()` back in the result dict; add a parent-side deadline (e.g., `pool.imap_unordered(...)` + `multiprocessing` timeout or a watchdog).

### M8. Dataset-cap selection is biased and CSV order is non-deterministic
- `extract_dataset_features.py:110-116`: `sorted(...)[:max_per_class]` takes the alphabetically-first N files — with DFDC's random 10-char IDs this is effectively an arbitrary but *reproducible* subset; with any named layout (FF++ `id0_id1_…`) it systematically selects the first actor pairs.
- `extract_dataset_features.py:234` uses `imap_unordered` → CSV row order varies between extraction runs; `data.py` group iteration order (dict insertion = first-seen order) then feeds `rng.shuffle(mixed)` — so **regenerating the CSV can change the train/val/test assignment even with the same seed** (same groups, different order). Within a fixed CSV, the split is deterministic.
- Fix: seed a random subset for caps; write the CSV sorted by `video_path`.

### M9. Dataset quality observations (verified from the CSV)
- All 16 rows have **negative SNR** (`-1.37 … -8.13 dB`) — the pulse is buried in noise across the entire training set; the signal is weak evidence.
- HR and `hr_half_diff` values lie on a coarse grid (multiples of ≈24.32 BPM; HR ∈ {48.6, 60.8, 72.97, 85.1, 121.6, 133.8, 145.9}) — coarse PSD binning on short clips limits physiological precision. `hr_half_diff` multiples of 24.32 also make HR-difference features nearly quantized constants.
- Training size: 10 train rows (5 real / 5 fake). Metrics (acc 0.6667, F1 0.8000, AUC 0.5000) are not statistically meaningful; the report should state this explicitly.

### M10. `--all` crashes on fresh environments (missing xgboost)
- `quantum/evaluation.py:222` imports `XGBClassifier` inside `run_baselines` **without a try/except**, and xgboost appears in **no** requirements file (root, frame, or RPPG). On a host without xgboost, `python -m quantum.pipeline --all` fails at the baselines step after all training work is done.
- Fix: add `xgboost>=2.0.0` to root requirements, or wrap in try/except and skip the XGBoost baseline.

### M11. `roc_auc_score` is unguarded for single-class splits
- `quantum/evaluation.py:82` and `quantum/plots.py:28` call `roc_auc_score(y_true, prob_real)` with no class-count check → `ValueError` if a test fold ends up single-class (today's 3-row test split is 2/1; any future small split can break `--all`).
- Fix: compute AUC only when both classes present (skip → `null`).

---

## 5. Confirmed Findings — Low

- **L1** `frontend/package.json:2` — `"name": "frontemd"` typo.
- **L2** `rppg-pipeline/run_on_video.py:7` — docstring says `python examples/run_on_video.py`; actual path is `rppg-pipeline/`.
- **L3** `rppg-pipeline/streamlit_app.py:119-128` — hardcodes `output/rppg/rppg_classifier.pkl` relative to 3 parents up; works only from repo layout; `min_frames` initial-value logic tied to stale `fps` widget state (cosmetic).
- **L4** `frame/app/pipeline.py:92-94` — silently remaps webcam source `0` → `1` with a warning; undocumented oddity.
- **L5** `RPPG/rppg/model_utils.py` — downloads (`face_landmarker.task`, Haar XML) without checksum pinning; the `latest` URL is a moving target (reproducibility drift); no timeout on `urlretrieve`.
- **L6** `AGENTS.md` optimization bullet claims stage-1 path "skips Laplacian/brightness" — **false**: `frame/app/pipeline.py:448` still computes and records them in metadata for every sampled frame (the *use* of the result is skipped in the rPPG stage, not the computation).
- **L7** `quantum/plots.py` `_sklearn` docstring says "~12 s" import; measured ~2.4 s (stale comment).
- **L8** Root `result.json` (untracked, gitignored) is a stale demo artifact referencing `FF++/train/FF-synthesis/id0_id1_0005.mp4` with an INCONCLUSIVE verdict — confusing if someone opens it.
- **L9** `SKILL.md` at repo root is an AI agent skill file (frontend-design guidance) — unrelated repo clutter; likely committed by accident.
- **L10** `quantum/config.py:73` `VQCConfig.batch_size = 32` with a 10-row train set → always a single batch; benign but misleading.
- **L11** `quantum/vqc.py:58` `QuantumLayer.forward` calls `self.weights.cpu()` on every forward; could cache on the torch device to avoid per-call copies.
- **L12** `WORKING/frame/requirements.txt:9` pins `opencv-python==4.6.0.66` while `WORKING/RPPG/requirements.txt:1` requires `>=4.8.0,<5.0.0` — installing per-stage requirements in the wrong order breaks stage 2; only the merged root `requirements.txt` resolves this. Foot-gun documented at root, still present.

---

## 6. Potential Risks / Assumptions (need verification)

- **P1 — DFDC subject leakage (unresolvable from current data).** DFDC archive filenames are random 10-char IDs (`aaagqkcdis.mp4`, …) that encode no subject pairing, so per-clip grouping (`data.py:30-41`) is the best available and *not demonstrably wrong*. However, DFDC contains multiple clips per subject; without the DFDC subject-metadata JSON, real/fake takes of the same person can land in different splits. Mitigation: import the DFDC metadata mapping for true subject IDs.
- **P2 — Non-finite features at inference.** If a vector contains `-inf`/NaN at inference (see M5), `predict_features` behavior is untested; the scaler will produce NaN → probability NaN → verdict handling unknown. Verify and guard.
- **P3 — GPU memory under concurrency.** Two simultaneous server jobs each load YOLO (CUDA) + MediaPipe + the VQC — the 4 GB RTX 4050 is near its ceiling; the concurrency cap of 2 is not enforced at the GPU level. Monitor during load tests.
- **P4 — `pickle.load` of `rppg_classifier.pkl`.** Arbitrary-code execution if the pkl is replaced by anyone with write access to `output/rppg/`; low risk locally, must not move to shared hosting as-is.
- **P5 — mediapipe version floor.** `mediapipe>=0.10.9` may resolve to a build whose Tasks `FaceLandmarker` API differs from what `face_roi.py` imports (docstring notes 0.10.30+ dropped the legacy API); the model bundle URL is unpinned (`latest`). Pin both for reproducibility.
- **P6 — `_grouped_train_val_test_split` balance fallback.** When group sizes exceed per-split targets (not the case today: all groups are size 1), `place()` assigns to the least-filled split, which can unbalance classes. Fine for the current data; re-check if DFDC metadata brings multi-clip groups.

---

## 7. Strengths (confirmed)

- **Deterministic and reproducible.** Fixed seeds, restart aggregation, and now regression-guarded: `tests.py` (5/5) pins the beta-mixer, Hamiltonian≡cost, feature-contract sync, and split determinism.
- **Correct degradation paths.** `features=None` → INCONCLUSIVE + exit 3; stage-1 failure → `process_video` fallback with `input_mode` recorded; server 30-min kill; TTL cleanup.
- **Server hardening holds.** CORS allowlist, streamed uploads, magic-byte validation, concurrency cap — all verified in smoke tests (evil-origin POST → 403, bad body → 415).
- **No synthetic data.** The quantum layer consumes only the real 16-row CSV (verified); no generator or bridge layer exists.
- **Efficient numerics.** Vectorized POS (bit-identical to the loop), shared Welch PSD, cached detrend/filter coefficients, lazy heavy imports (torch/pennyLane measured; sklearn + xgboost deferred).
- **Frontend quality.** Clean 7-stage state machine, SSE progress, verdict dossier, theme persistence with `prefers-reduced-motion`, no JS errors in E2E, responsive to 375 px.
- **Feature contract enforced** by `test_feature_contract_sync` (the two 10-feature lists are identical, verified).

---

## 8. Recommended Fix Plan (priority order)

| # | Severity | Fix | Effort |
|---|---|---|---|
| 1 | High (H1) | Reject degenerate signals (SQI≈0 / fallback-heavy) as `features=None`; add warning log | ~30 min |
| 2 | High (H2) | Guard cross-check predict/proba; validate `n_features_`/`n_classes_` | ~15 min |
| 3 | High (H3) | Rewrite README data/model sections to the 16-row DFDC reality | ~20 min |
| 4 | Medium (M1, M6, M10) | Fix doc commands; POS_MSEC==0 fallback; xgboost in requirements | ~45 min |
| 5 | Medium (M2) | 8→10 sweep + add `hr_half_diff`/`peak_prominence` to `FEATURE_LABELS` | ~20 min |
| 6 | Medium (M3, M4, M5) | `test_ratio` config; artifact-consistency asserts; docstring/`-inf` handling | ~40 min |
| 7 | Medium (M7, M8) | Worker crash reporting + parent timeout; seeded caps + sorted CSV | ~40 min |
| 8 | Medium (M11) | Guard AUC for single-class folds | ~10 min |
| 9 | Low (L1–L12) | Triage; at minimum fix L6 (AGENTS.md claim) and L8 (remove stale result.json) | ~30 min |
| 10 | Potential (P1–P6) | DFDC subject metadata, non-finite inference guard, GPU load test, pkl trust note, mediapipe pin | 1–2 days |

---

## Appendix A — Environment & Benchmarks (measured this session)

- Host: Windows 11, RTX 4050 Laptop (4 GB), 16 logical CPUs, 24 GB RAM; Python 3.10; torch 2.5.1+cu121; PennyLane 0.42.3; pennylane-lightning 0.42.0.
- `python -m quantum.pipeline --all`: ~29 s (QAOA ~14.6 s; VQC train ~15.2 s). `import quantum.pipeline` ~6.5 s warm. Inference ~275 ms cold / ~32 ms cached.
- QAOA restarts use `lightning.qubit` via `ProcessPoolExecutor` (4 workers); `lightning.gpu` has no Windows wheels — CPU SIMD is the ceiling.
- Verify command set: `python -m quantum.tests`; `python -m quantum.pipeline --all`; `python run_pipeline.py --source <video> --method POS` (all from `WORKING/`).

## Appendix B — Data & artifact ground truth (this session)

- `output/rppg/dataset_features.csv`: 16 rows (9 real `0` / 7 fake `1`), all `archive/DFDC_Dataset/Real|Fake`, 10 feature columns + label + video_path + source. Verified contents above (M9).
- `output/quantum/`: `data.npz` (10/3/3 split), `qaoa_selection.json` (6 features, restart seed 43, cost 0.3950), `feature_scaler.json`, `hybrid_vqc.pt`, `metrics_quantum.json` (ECE 0.1052, acc 0.6667, F1 0.8000), plots.
- `output/frames/`: `frame_sequences/`, `frame_extraction_log.jsonl`, `frame_extraction_summary.json`, `docs/` (sampling note, quality checklist, examples report) — all regenerated artifacts, untracked.
- `output/pipeline/`: `run_pipeline.py` result JSON (prob_real 0.6155 → UNCERTAIN).

## Appendix C — Historical note

`Docs/projec_audit.md` (699 lines) predates the real-data refactor: it references `quantum/selection.py` (renamed to `qaoa.py`), the "8 features" count, and an older bug finding already fixed in `run_pipeline.py` — treat as historical only.