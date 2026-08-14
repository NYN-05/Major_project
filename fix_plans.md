 

# Fix Plans — PROJECT_AUDIT_REPORT.md (2026-08-13)

Execution plan for every confirmed finding and potential risk in `PROJECT_AUDIT_REPORT.md`.
Each item follows: **Problem → Reason → Solution → Why this solution → Instructions → Verification**.

**Canonical baselines (do not regress):** QAOA selection `[hr_half_diff, cheek_forehead_correlation, left_right_cheek_correlation, mad, heart_rate_bpm, spectral_entropy]` (chosen restart seed 43, cost 0.3950); ECE 0.1052; acc 0.6667 / F1 0.8000 / AUC 0.5000; E2E verdict `prob_real=0.6155 → UNCERTAIN, confidence=0.2311`; `python -m quantum.tests` 5/5.

**Decisions locked in with the user:**

- H1: degenerate signals are rejected (INCONCLUSIVE) — behavior change approved.
- P1: DFDC subject metadata unavailable → document-only limitation (no code change).
- M8: fix code for future extractions only; the 16-row `dataset_features.csv` is NOT regenerated.

## Status (updated 2026-08-14)

**✅ DONE** — H1, H2, H3; M1–M11 (all Phase 1 + Phase 2 items, commits `d948d46` → `1173767`); L6 (already fixed at plan time), L7. Canonical baselines verified after the fixes: ECE 0.1052, `prob_real=0.6155 → UNCERTAIN`, `python -m quantum.tests` 5/5.

**⏳ OPEN (not started)** — L1, L2, L3, L4, L5, L8, L9, L10, L11, L12 (Phase 3 low-severity hygiene batch) and P1–P5 (Phase 4 documented risks: P1/P2/P4/P5 are doc-only or small guards, still pending; P6 awaits P1).

| Item | Status | Where |
|---|---|---|
| H1 degenerate-signal rejection | ✅ DONE (`d948d46`) | `RPPG/rppg/pipeline.py:305-308` |
| H2 cross-check guard | ✅ DONE (`3f84d00`) | `run_pipeline.py:194-206` |
| H3 README rewrite | ✅ DONE (`9da797f`) | `README.md:120-132,175` |
| M1 frame-stage docs | ✅ DONE (`ce1367c`) | `README.md:51-55`, `frame/README.md` |
| M2 8→10 sweep + UI labels | ✅ DONE (`ec18e8c`) | `lib.js:43,143-144`, `QuantumFlow.jsx:7` |
| M3 `test_ratio` config | ✅ DONE (`5930f93`) | `quantum/config.py:44`, `data.py:131` |
| M4 artifact-consistency asserts | ✅ DONE (`4ad2a6c`) | `quantum/pipeline.py:54-75` |
| M5 docstring + `-inf→nan` | ✅ DONE (`c1f4848`) | `RPPG/rppg/features.py` |
| M6 stride-sampling fallback | ✅ DONE (`ed51efb`) | `frame/app/processing.py:102-143` |
| M7 worker timeout + crash visibility | ✅ DONE (`8377760`) | `extract_dataset_features.py`, `probe_features.py` |
| M8 seeded caps + sorted CSV | ✅ DONE (`8377760`) | `extract_dataset_features.py` |
| M9 data-quality transparency | ✅ DONE (`8377760`, `9da797f`) | `README.md:128-132`, extraction docstrings |
| M10 xgboost optional | ✅ DONE (`1173767`) | `quantum/evaluation.py:288`, `requirements.txt` |
| M11 AUC guard | ✅ DONE (`1173767`) | `quantum/evaluation.py`, `plots.py` |
| L1–L5, L8–L12 | ⏳ OPEN | Phase 3 quick wins not yet applied |
| L6 AGENTS.md/RPPG README claim | ✅ DONE (pre-plan) | — |
| L7 plots.py docstring | ✅ DONE | `quantum/plots.py:15` |
| P1 README subject note | ⏳ OPEN | doc-only, sentence not yet added |
| P2 non-finite inference guard | ⏳ OPEN | not in `predict_features` |
| P3–P6 | ⏳ OPEN | P3 load test; P4 doc note; P5 mediapipe pin; P6 awaits P1 |

---

## Phase 0 — Preflight

1. Confirm clean tree: `git status` (expect clean after `a11f06e`).
2. Snapshot current artifacts: `WORKING/output/rppg/dataset_features.csv` (16 rows), `output/quantum/qaoa_selection.json`, `metrics_quantum.json`, `data.npz` — copy hashes (`Get-FileHash`) so regressions are detectable.
3. Record baseline numbers by running `python -m quantum.tests` and one `python run_pipeline.py --source <canonical video> --method POS` from `WORKING/` (expect `prob_real=0.6155 → UNCERTAIN`).

---

## Phase 1 — High severity

### H1 — Reject degenerate rPPG signals instead of fabricating neutral features — ✅ DONE (`d948d46`)

**Problem:** `_fill_nan_with_median` (`WORKING/RPPG/rppg/features.py:334-354`) replaces NaN features with hardcoded "average human" constants: `heart_rate_bpm→72.0`, `spectral_entropy→0.5`, `peak_prominence→1.0`, `snr_db→0.0`. A flat/dead pulse (static scene, no face signal, extreme compression) therefore produces the feature vector of a plausible healthy person and can be scored REAL or FAKE on fabricated physiology. `signal_quality_index→0.0` is the only honest marker, and nothing consumes it.

**Reason:** The fill exists so the CSV/classifier never sees NaN, but at inference it silently converts "cannot measure" into "measured: normal human". The quantum layer's plausibility filter (`quantum/data.py:89-98`) only drops non-finite values, which can no longer occur after filling.

**Solution:** Gate in `RPPGPipeline._finalize` (`WORKING/RPPG/rppg/pipeline.py`), BEFORE `_fill_nan_with_median` runs: count non-finite raw features from `compute_features(...)`; if `nan_count >= 2` or `features.signal_quality_index == 0.0`, return `features=None` (the existing INCONCLUSIVE path, exit 3) and append a warning to `result.warnings` naming the degenerate signal. `_fill_nan_with_median` remains for the rare single-NaN case.

**Why this solution:** Reuses the tested `features=None → INCONCLUSIVE` path (no new failure mode); zero schema changes (`to_dict`/CSV contract and `to_feature_vector` untouched); the honest signal (SQI) becomes the gate instead of being ignored; warnings surface the cause in the server log lines.

**Instructions:**

1. In `WORKING/RPPG/rppg/pipeline.py::_finalize`, immediately after the `compute_features(...)` call, add:
   ```python
   raw = features.to_vector()
   nan_count = int(np.isnan(raw).sum())
   degenerate = nan_count >= 2 or features.signal_quality_index == 0.0
   ```
2. If `degenerate`: return the `features=None` result (same shape as the `n_usable < min_usable_frames` branch) and add `"degenerate signal (NaN-heavy or zero-quality pulse); returning INCONCLUSIVE"` to `warnings`.
3. Do NOT change `_fill_nan_with_median` itself here (docstring fix is M5).
4. Verify the gate fires: run the pipeline on a static image converted to a short mp4 → expect `features=None` + warning; run on `WORKING/frame/test.mp4` → expected values unchanged.

**Verification:** `python -m quantum.tests` still 5/5; canonical clip still `0.6155 → UNCERTAIN`; degenerate input → INCONCLUSIVE with the warning visible in the result JSON `stages.rppg.warnings`.

---

### H2 — Guard the rPPG cross-check so a bad pkl cannot crash the pipeline — ✅ DONE (`3f84d00`)

**Problem:** `WORKING/run_pipeline.py:183-198` wraps only `pickle.load`; `clf.predict_proba(x)[0]` (line 194) and `clf.predict(x)[0]` (line 195) are unguarded. A single-class or stale `rppg_classifier.pkl` raises `IndexError`/`ValueError` after the rPPG stage, killing `run_pipeline.py` and the server job.

**Reason:** `retrain_dfdc.py:194-204` writes `rppg_classifier.pkl` incrementally — a checkpoint is saved after every 10 processed videos, which can be trained on a single class (e.g., first 10 sorted clips all Fake). sklearn `predict_proba` on a single-class fit returns a 1-column array → `proba[1]` → `IndexError`. A pkl trained on a different feature count → `ValueError`.

**Solution:** Wrap the whole predict block in `try/except Exception` → return `{"skipped": "rppg_classifier.pkl unusable: ..."}`; before predicting, validate `getattr(clf, "n_features_in_", 0) == 10` and `len(getattr(clf, "classes_", [])) == 2`.

**Why this solution:** The cross-check is informational by design (final verdict comes from the quantum stage); the fix is ~8 lines, requires no retraining, and preserves the happy path exactly.

**Instructions:**

1. In `rppg_classifier_crosscheck`, after the load guard, add:
   ```python
   if getattr(clf, "n_features_in_", 0) != 10 or len(getattr(clf, "classes_", [])) != 2:
       return {"skipped": f"rppg_classifier.pkl incompatible (features={getattr(clf, 'n_features_in_', '?')}, classes={getattr(clf, 'classes_', '?')})"}
   ```
2. Wrap `proba = clf.predict_proba(x)[0]; pred = int(clf.predict(x)[0])` in `try/except Exception` → `{"skipped": f"rppg_classifier.pkl prediction failed: {exc}"}`.

**Verification:** craft a single-class `RandomForestClassifier` pkl in a temp copy of the output dir (or monkey-patch the path) and run `run_pipeline.py` → pipeline completes with `rppg_crosscheck.skipped` set; with the real pkl (if present) the happy path still emits `verdict`/`probability`.

---

### H3 — Rewrite the README data/model sections to describe the real system — ✅ DONE (`9da797f`)

**Problem:** `README.md:118-123` ("Current trained model": 469 FF++ clips, test acc 0.564, AUC 0.640, ECE 0.073) and `README.md:166` ("FF++-sourced table of 669 labeled clips; the DFDC row subset is false-positive noise and excluded") describe a system that does not exist. The trained model uses exactly 16 DFDC rows (9 real / 7 fake).

**Reason:** The README predates the real-data refactor; `AGENTS.md` was corrected but the README was not.

**Solution:** Replace both blocks with the verified facts: 16-row DFDC table, split 10/3/3, selection seed 43 cost 0.3950, ECE 0.1052, acc 0.6667, E2E `prob_real 0.6155 → UNCERTAIN`, an explicit small-data disclaimer ("metrics are indicative only at this scale"), and a pointer to `PROJECT_AUDIT_REPORT.md`. Also fix "8 physiological features" → 10 at lines 29, 59, 85.

**Why this solution:** Documentation integrity — a reviewer must get the true system state from the entry-point doc; wording stays aligned with `AGENTS.md` so the two docs cannot contradict each other.

**Instructions:**

1. Edit the "Current trained model" paragraph (lines ~118-123) to the 16-row reality + disclaimer.
2. Edit the "Verified Constraints" bullet (line ~166) to "16 labeled rows (9 real / 7 fake — DFDC archive only; FF++ is NOT used)".
3. Replace `8-feature`/`8 physiological` strings with `10-feature`/`10 physiological` (lines 29, 59, 85).
4. Keep the architecture/stage descriptions intact.

**Verification:** `Select-String -Path README.md -Pattern "469|669|8 feature|8 physiological|FF++-sourced"` → no hits (except explicit FF++-not-used mentions).

---

## Phase 2 — Medium severity

### M1 — Fix broken frame-stage commands in the docs — ✅ DONE (`ce1367c`)

**Problem:** `README.md:51-53` and `WORKING/frame/README.md:42,101,110` reference `app/main.py` and `app/extract_frames.py`; neither exists (`app/` contains only `config.py, detector.py, pipeline.py, processing.py`). Commands fail with `ModuleNotFoundError`. (`AGENTS.md` already corrected this session.)

**Reason:** The legacy entry points were consolidated into `app/pipeline.py` (its module docstring says "was app/main.py") without updating the docs.

**Solution:** Point to `python app/pipeline.py --source test.mp4 --save-metadata` for the face pipeline and `python app/pipeline.py --source test.mp4 --sample-fps 10 --save-quality-examples` for extraction-only mode (flags `--input-dir`, `--sample-fps`, `--min-seq-len`, etc. trigger `extract_main`).

**Why this solution:** The commands must be runnable as documented; zero code changes.

**Instructions:** Update the three doc locations with the working commands; mention `app/main.py` no longer exists.

**Verification:** run `python app/pipeline.py --source test.mp4 --save-metadata` from `WORKING/frame/` (on the tracked `test.mp4`) → pipeline runs.

---

### M2 — Sweep the "8 features" → 10-feature drift and complete the UI feature list — ✅ DONE (`ec18e8c`)

**Problem:** 8 stale sites say "8 features" (contract is 10): `WORKING/run_pipeline.py:13`; `WORKING/quantum/pipeline.py:50`; `WORKING/RPPG/README.md:6`; `WORKING/RPPG/rppg-pipeline/retrain_dfdc.py:8`; `frontend/src/lib.js:43` and `:116`; `frontend/src/components/QuantumFlow.jsx:7` and `:88`. Additionally `frontend/src/lib.js:132-141` `FEATURE_LABELS` lists only 8 of 10 — `hr_half_diff` and `peak_prominence` (both QAOA-selected) never surface in the UI — and `lib.js:123` describes MAD as "derivative magnitude approach" (code uses plain mean absolute deviation, `features.py:182`).

**Reason:** The feature set grew from 8 to 10 during development; UI labels and prose were not updated; the two newest features were never added to the display list.

**Solution:** Sweep all sites to "10"; add `hr_half_diff` and `peak_prominence` to `FEATURE_LABELS` with labels + tips ("Heart-rate half-diff", "Spectral peak prominence"); fix the MAD glossary entry to "Mean absolute deviation of the pulse waveform".

**Why this solution:** Text must match the enforced feature contract (guarded by `test_feature_contract_sync`); the QAOA "Selected by QAOA" tags in `QuantumFlow.jsx` currently fall back to raw snake_case names for the two missing features — adding labels makes the Insights panel complete.

**Instructions:**

1. Text-only edits at the 8 sites.
2. In `lib.js` `FEATURE_LABELS`, append `{ key: "hr_half_diff", label: "HR half-diff", unit: "BPM", tip: ... }` and `{ key: "peak_prominence", label: "Peak prominence", unit: "", tip: ... }` (order must mirror `RPPGFeatures.feature_names()`).
3. Update `GLOSSARY.mad`.
4. Rebuild the frontend.

**Verification:** `npm run build` succeeds; grep `frontend/src` + `WORKING` for `8 feature|eight rPPG|from the 8` → zero hits; E2E dashboard shows all 10 features.

---

### M3 — Make the split configuration truthful — ✅ DONE (`5930f93`)

**Problem:** `quantum/config.py:43` declares `DataConfig.train_ratio = 0.6`, never read; `quantum/data.py:130-131` uses `val_ratio` for BOTH val and test (`test_c = round(per_class * val_ratio)`); there is no `test_ratio`. Changing `train_ratio` silently does nothing.

**Reason:** The split is computed as train = remainder, val = test = 0.2; the config field is leftover and misleading.

**Solution:** Remove `train_ratio`; add `test_ratio: float = 0.2`; use it in `_grouped_train_val_test_split`; extend `test_split_determinism` in `quantum/tests.py` to assert `train+val+test == total` per class and that ratios are honored.

**Why this solution:** Default `test_ratio=0.2` reproduces today's exact 10/3/3 split (verified by rebuild), so no model drift; config now reflects reality and supports asymmetric splits later.

**Instructions:**

1. Edit `config.py` DataConfig.
2. Edit `data.py` `_grouped_train_val_test_split`: `test_c = int(round(per_class[c] * cfg.test_ratio))`.
3. Extend the determinism test (assert totals + per-class ratio bounds).
4. Rebuild data: `python -m quantum.pipeline --build-data` from `WORKING/` → confirm 10/3/3 and same group assignment as baseline snapshot.

**Verification:** `python -m quantum.tests` 5+/5; `--build-data` output matches baseline split; canonical metrics unchanged after `--all`.

---

### M4 — Validate quantum artifact consistency in `predict_features` — ✅ DONE (`4ad2a6c`)

**Problem:** `quantum/pipeline.py::predict_features` applies the saved QAOA indices, scaler, and checkpoint independently. Out-of-sync artifacts (new `--select` run + old checkpoint, checkpoint from another feature count) fail with opaque shape errors deep in torch/sklearn. Docstring says "8 rPPG feature names" (line 50).

**Reason:** No cross-artifact validation at load; stale artifacts are indistinguishable from valid ones until a shape error surfaces far from the cause.

**Solution:** At load time assert: `scaler.mean_.shape[0] == len(FEATURE_NAMES)` (10); all selection indices `< 10`; checkpoint input dim == `len(selection)`. Raise `RuntimeError` naming the artifact paths and the remedy ("rerun `python -m quantum.pipeline --all` from WORKING/"). Fix the docstring to 10.

**Why this solution:** ~15 lines, zero behavior change on consistent artifacts, turns silent failures into actionable messages — critical for the server path where the model loads on every job.

**Instructions:**

1. In `predict_features`, after loading scaler/selection/checkpoint, add the three asserts.
2. Fix docstring.
3. Test the failure mode by temporarily pointing at a mismatched checkpoint.

**Verification:** happy-path run unchanged (`0.6155 → UNCERTAIN`); a deliberately mismatched artifact raises the descriptive error.

---

### M5 — Fix `_fill_nan_with_median` docstring and the `-inf` SNR edge — ✅ DONE (`c1f4848`)

**Problem:** `features.py:334-354` docstring claims "per-feature median of non-NaN values" — the code uses hardcoded constants only. And `estimate_snr` (`features.py:134`) returns `-inf` when signal power ≤ 0; `np.isnan(-inf)` is `False`, so the fallback never replaces it → `-inf` propagates to the CSV (silently dropped by `data.py`'s isfinite filter) or, at inference, into the scaler → NaN probability.

**Reason:** Stale docstring; `-inf` is a second, invisible "missing" convention that the NaN-based plumbing does not handle.

**Solution:** Rewrite the docstring to describe the actual hardcoded neutral fallbacks (now that H1 gates degenerate signals, the fallback only fires for minor single-NaN cases). Change `estimate_snr` to return `float("nan")` when `signal_power <= 0`.

**Why this solution:** NaN becomes the single missing-value convention; H1's `nan_count` gate then catches the case; data.py's isfinite filter and the scaler both behave consistently.

**Instructions:**

1. `features.py` docstring + `-inf → nan` change.
2. Confirm the 16-row CSV is unaffected (no row currently has `-inf`, verified: all rows finite).

**Verification:** `python -m quantum.tests`; `--build-data` filter stats unchanged (0 dropped_invalid); E2E smoke unchanged.

---

### M6 — FrameIngestor: fall back to stride sampling when timestamps are unusable — ✅ DONE (`ed51efb`)

**Problem:** `WORKING/frame/app/processing.py:100-126` keys sampling off `CAP_PROP_POS_MSEC` when `sample_fps` is set. Some containers/codecs report `POS_MSEC == 0` for every frame → only the first frame passes the `timestamp_ms >= next_target_ms` test → stage 1 yields 1 frame → rPPG `features=None` → INCONCLUSIVE, with no diagnostic.

**Reason:** Timestamp-based sampling trusts the container's timebase without verifying it exists.

**Solution:** Track sampled timestamps; if after `2 × sample_step` source frames all sampled timestamps are 0, disable timestamp sampling for the rest of the stream and switch to stride sampling `(source_frame_id - 1) % sample_step == 0`, logging a warning.

**Why this solution:** Preserves the 10 fps contract on any container; the fallback is automatic and visible; no config changes.

**Instructions:**

1. In `FrameIngestor.frames()`, add a `zero_ts_count`/`mode` switch per the above.
2. Emit the warning via `logging.getLogger("face_pipeline")`.
3. Test with a synthetic container whose POS_MSEC is 0 (e.g., an AVI written without timestamps) → extraction now yields ~10 fps frames instead of 1.

**Verification:** `python app/pipeline.py --input-dir <test dir> --sample-fps 10` on the synthetic file → `sampled_frames` ≈ duration×10 with a warning in the log.

---

### M7 — Worker crash visibility + parent timeout in extraction scripts — ✅ DONE (`8377760`)

**Problem:** `extract_dataset_features.py:40-58` and `probe_features.py:40-47` redirect worker stderr to NUL (`os.dup2`) for log hygiene; unexpected worker death (segfault, OOM) prints nothing, and `mp.Pool.imap_unordered` can hang the parent forever. `chunksize=1` maximizes pickle round-trips.

**Reason:** Diagnostics and robustness were traded for quiet logs; no timeout or crash channel exists.

**Solution:** Keep the stderr redirect but add a parent-side consumption loop with `timeout=` on `next()`; on `multiprocessing.TimeoutError`, `pool.terminate()`, and report which item hung. Ensure `_process_one` returns `traceback.format_exc()` strings on unexpected exceptions (it already returns `f"{type(exc).__name__}: {exc}"` — keep, extend with the worker PID).

**Why this solution:** Hours-long dataset runs must fail loudly and fast rather than hang silently; terminating the pool on a stuck worker is the standard mitigation.

**Instructions:**

1. Replace `for entry in pool.imap_unordered(...)` with an explicit `it = iter(pool.imap_unordered(...))` + `next(it, timeout=...)` loop inside `try/except TimeoutError`.
2. On timeout: `pool.terminate()`, print the pending item index, exit non-zero.
3. Same change in `probe_features.py:185`.

**Verification:** simulate by killing a worker (or a pathological video) → parent terminates with a clear message instead of hanging.

---

### M8 — Unbiased dataset caps and deterministic CSV row order (future extractions only) — ✅ DONE (`8377760`)

**Problem:** `extract_dataset_features.py:110-116` uses `sorted(...)[:max_per_class]` — the alphabetically-first N files (with named layouts like FF++ `id0_id1_…` this systematically selects the first actor pairs). And line 234 uses `imap_unordered` → CSV row order varies between runs; `data.py` group iteration order (dict insertion = first-seen order) feeds `rng.shuffle(mixed)`, so regenerating the CSV can change the split assignment even with the same seed.

**Reason:** Capping and row order were never made deterministic or unbiased.

**Solution:** Before capping, seeded-shuffle the sorted list (`np.random.RandomState(0).shuffle(list)`); after collecting, write the CSV sorted by `video_path`. **The current 16-row CSV is NOT regenerated** — this fix governs future extractions (per user decision).

**Why this solution:** Reproducible splits across extraction runs; fair caps; zero impact on the trained model today.

**Instructions:**

1. Edit `collect_samples` caps.
2. Edit `main()`: `out_df = out_df.sort_values("video_path")` before `to_csv`.
3. Do not rerun extraction against the archive now.

**Verification:** two consecutive extractions (on a small `--max-per-class`) produce identical, sorted CSVs.

---

### M9 — Data-quality transparency (documentation only) — ✅ DONE (`8377760` + `9da797f`)

**Problem:** All 16 labeled rows have negative SNR (−1.4 to −8.1 dB); HR features lie on a coarse ~24.3 BPM grid; the train set is 10 rows — metrics are indicative only.

**Reason:** Short compressed DFDC clips + coarse PSD binning at 10 fps.

**Solution:** State this explicitly in the README model section (folded into H3) and in `retrain_dfdc.py`/`extract_dataset_features.py` output summaries. Optional future work (note in plan, do not implement now): zero-padded Welch (`nfft=`) for smoother HR estimates.

**Why this solution:** Honest reporting without code churn; the HR quantization is information-theoretic (bin width), not fixable by better code at this clip length.

**Instructions:** Text additions only.

**Verification:** README contains the data-quality caveat.

---

### M10 — Make `--all` complete without xgboost — ✅ DONE (`1173767`)

**Problem:** `quantum/evaluation.py:222` imports `XGBClassifier` inside `run_baselines` with no guard; xgboost appears in NO requirements file. On a fresh host, `python -m quantum.pipeline --all` crashes at the baselines step after all training work.

**Reason:** The lazy-import optimization assumed xgboost was installed; the dependency was never recorded.

**Solution:** Add `xgboost>=2.0.0` to root `requirements.txt` AND wrap the import in try/except so the baseline is reported as `{"skipped": "xgboost not installed"}`.

**Why this solution:** `--all` must always complete (the VQC is the deliverable; baselines are comparison only); the requirements file stays complete for CI and other machines.

**Instructions:**

1. Root `requirements.txt`: add xgboost under "Shared numeric / ML".
2. `evaluation.py:222`: guard `from xgboost import XGBClassifier` and skip the entry.

**Verification:** `python -m quantum.pipeline --all` completes on a host without xgboost (baseline `skipped`); with xgboost installed, baselines run.

---

### M11 — Guard AUC computation for single-class folds — ✅ DONE (`1173767`)

**Problem:** `quantum/evaluation.py:82` and `quantum/plots.py:28` call `roc_auc_score(y_true, prob_real)` unconditionally → `ValueError` when a split contains one class (reachable with tiny datasets; today's 3-row test split is 2/1).

**Reason:** No class-count check before the sklearn call.

**Solution:** `if len(set(y_true)) < 2: auc = None` (metrics: `"auc_roc": None`; plots: skip the ROC panel) before computing.

**Why this solution:** Two-line guard; prevents `--all` from crashing on future small splits.

**Instructions:** Edit both call sites; verify baseline plots still render (test split has both classes today).

**Verification:** `python -m quantum.pipeline --all` green; `metrics_quantum.json` unchanged.

---

## Phase 3 — Low severity (quick wins, one batch)

- **L1** ⏳ OPEN `frontend/package.json:2` — `"name": "frontemd"` → `"frontend"`. *Why: correct metadata; no functional impact.*
- **L2** ⏳ OPEN `WORKING/RPPG/rppg-pipeline/run_on_video.py:7` — docstring `examples/run_on_video.py` → `rppg-pipeline/run_on_video.py`.
- **L3** ⏳ OPEN `streamlit_app.py:119-128` — model path via env `RPPG_CLASSIFIER_PKL` with the current default; leave the widget quirk. *Why: makes the demo usable from any cwd.*
- **L4** ⏳ OPEN `WORKING/frame/app/pipeline.py:92-94` — remove the webcam `0 → 1` remap hack; pass the source through as-is. *Why: surprising undocumented behavior.*
- **L5** ⏳ OPEN `WORKING/RPPG/rppg/model_utils.py` — record expected SHA-256 for `face_landmarker.task` and the Haar XML; verify after download, raise on mismatch; keep the `latest` URL but document it as a moving target. *Why: silent model drift breaks reproducibility.*
- **L6** ✅ already fixed this session (`AGENTS.md` + `RPPG/README.md` corrected) — no action.
- **L7** ✅ DONE `WORKING/quantum/plots.py` — `_sklearn` docstring now reads "~2.4 s" (measured).
- **L8** ⏳ OPEN Delete stale root `result.json` (untracked; references an FF++ clip; `/api/previous` reads `output/pipeline/` only). *Why: removes a misleading leftover artifact.*
- **L9** ⏳ OPEN `git rm SKILL.md` — it is the frontend-design agent skill, already installed at `C:\Users\JHASHANK\.claude\skills\frontend-design\SKILL.md` (confirm the copy exists before deleting). *Why: unrelated clutter in a research repo.*
- **L10** ⏳ OPEN `quantum/config.py:73` — add a comment: batch 32 with ≤32 train rows = single batch (benign). No code change.
- **L11** ⏳ OPEN `quantum/vqc.py:58` — cache the `.cpu()` weight tensor across forwards (micro-opt). *Only if the inference path is touched anyway; otherwise skip.*
- **L12** ⏳ OPEN `WORKING/frame/requirements.txt:9` — `opencv-python==4.6.0.66` → `>=4.8.0,<5.0.0` (matches RPPG + root merged file). *Why: removes the install-order foot-gun.*

---

## Phase 4 — Potential risks (documented; no code unless stated)

- **P1 — DFDC subject leakage (document only, per user decision).** ⏳ OPEN `data.py:30-41` groups per clip because DFDC random 10-char IDs encode no subject pairing and the official DFDC subject-metadata JSON is unavailable. Add one sentence to the README data section: "DFDC clips are grouped per-clip (no subject IDs available); if the DFDC metadata JSON is obtained later, `_infer_subject_key` should map clips → subjects." No code change now.
- **P2 — Non-finite features at inference.** ⏳ OPEN Add to `predict_features` (`quantum/pipeline.py`): `if not np.isfinite(x_scaled).all(): return INCONCLUSIVE payload` (guards the M5 `-inf` path end-to-end). Small, recommended.
- **P3 — GPU memory under 2 concurrent jobs.** ⏳ OPEN Run a 2-job load test (two browser tabs). If OOM appears, add a frame-stage device knob (`--device cpu`) used under pressure. No change preemptively.
- **P4 — pkl trust.** ⏳ OPEN Document in README/AGENTS: `output/rppg/` must remain write-protected; `rppg_classifier.pkl` is `pickle.load`-ed. No code change.
- **P5 — mediapipe pin.** ⏳ OPEN Bump `mediapipe>=0.10.9` → `>=0.10.30` in `WORKING/RPPG/requirements.txt` (matches the Tasks API documented in `face_roi.py`); checksum pinning handled by L5.
- **P6 — split-balance edge (multi-clip groups).** ⏳ OPEN Revisit only if P1 lands (subject metadata creates multi-clip groups).

---

## Verification matrix (run after each phase)

| # | Check                  | Command (cwd)                                                                    | Expect                                                             |
| - | ---------------------- | -------------------------------------------------------------------------------- | ------------------------------------------------------------------ |
| 1 | Quantum self-tests     | `WORKING/`: `python -m quantum.tests`                                        | 5/5 pass                                                           |
| 2 | Full quantum flow      | `WORKING/`: `python -m quantum.pipeline --all`                               | green; metrics == canonical (unless dataset intentionally changed) |
| 3 | E2E inference          | `WORKING/`: `python run_pipeline.py --source <canonical video> --method POS` | `prob_real=0.6155 → UNCERTAIN`                                  |
| 4 | Degenerate rejection   | pipeline on a static-image mp4                                                   | INCONCLUSIVE + warning                                             |
| 5 | Server smoke           | `frontend/`: `python server.py`; curl allowed/evil origin, bad body          | 403 evil POST, 415 bad body, ACAO echo for allowed                 |
| 6 | Frontend build         | `frontend/`: `npm run build`                                                 | builds; no "8 feature" strings in`src/`                          |
| 7 | Doc integrity          | `Select-String` for `469                                                       | 669                                                                |
| 8 | Extraction determinism | two small`--max-per-class` runs (temp output)                                  | identical sorted CSVs                                              |

## Commit plan (only on explicit user request; matches repo style)

1. `docs: fix README data/model claims, feature counts, and broken frame-stage commands` — H3, M1, M2 (doc sites), L2, L7, L9, P1 note.
2. `fix: reject degenerate rPPG signals; guard cross-check and artifact consistency` — H1, H2, M3, M4, M5, M10, M11, P2.
3. `fix: frame/RPPG robustness and hygiene` — M6, M7, M8, L1, L3, L4, L5, L8, L12 (+P5 if applied).

Rollback note: every phase is independent; H1/H2/M3/M4/M5 changes are small and revertible, and Phases 1-2 never touch the 16-row CSV or the trained artifacts, so canonical metrics must remain exactly as recorded in Phase 0.
