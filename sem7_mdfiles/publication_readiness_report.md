# Publication Readiness Evidence Report

**Project:** Deepfake Video Detection for KYC — rPPG + visual fusion with QAOA feature selection and a Hybrid Variational Quantum Classifier (DFDC)
**Report date:** 2026-10-19
**Repository commit analysed:** `114e0bde71deb1d5f452c80a6d18724b1944b452` (branch: current working tree, uncommitted analysis artifacts excluded)
**Evidence basis:** repository source code, configuration, generated artifacts under `WORKING/output/`, git history, and scripts newly created in `paper_experiments/`. Documentation files were used only as secondary context; conflicts are flagged in Appendix B.
**Status vocabulary:** **[FOUND]** (directly extracted, source cited) · **[COMPUTED]** (executed in this audit; command and environment in Appendix A) · **[MISSING]** (not present and not computable from the repository) · **[HUMAN-VERIFY]** (requires author confirmation).

---

## Executive Summary

| # | Item | Status | Key finding | Blocking? |
|---|------|--------|-------------|-----------|
| 1 | Full-dataset retraining and evaluation | **[FOUND]** + **[MISSING]** | Full-DFDC rPPG extraction is **incomplete and single-class**: current CSV holds **1,024 FAKE-only rows** of 1,966 Fake / 2,127 Real videos on disk; the recorded quantum metrics come from a deleted **38-video smoke run** (test n = 8). | **Y** |
| 2 | Leakage-tested split | **[FOUND]** + **[COMPUTED]** + **[MISSING]** | Split is deterministic, leakage-guarded, and train-only in all fitting — **but identity/source grouping is explicitly unavailable for the current DFDC layout** (`data.py` docstring: subject-level separation "unrecoverable from filenames"); grouping degenerates to per-clip keys. Near-duplicate source leakage is therefore *structural*, not testable away. | **Y** |
| 3 | Confidence intervals / repeated runs | **[MISSING]** | Single fixed seed (42) everywhere; no multi-seed runs; recorded 5-fold CV gives mean ± std only. Wilson 95% CIs computed on recorded counts show unusable width (accuracy CI [0.215, 0.785] at n = 8). N ≥ 5-seed script provided, cannot execute until item 1 resolves. | **Y** |
| 4 | Controlled robustness experiments | **[MISSING]** | No degradation machinery or results anywhere in the code/artifacts. Reproducible ffmpeg grid script provided in `paper_experiments/robustness_degrade.py`. | Y (for low-resolution claims) |
| 5 | Ablation studies | **[MISSING]** | Ablation feature sets and the ablation runner were **removed** (`config.py` `PHASE6_FEATURE_SETS = {}`, sweep module deleted); only the fused configuration exists. | **Y** |
| 6 | Classical baseline comparisons | **[FOUND]** + **[MISSING]** | Six sklearn/XGBoost baselines exist and share split/CV with the VQC; no CNN/ViT baselines, no hyperparameter tuning, and XGBoost early-stopping monitors training data (biased). Parameter counts **[COMPUTED]**. | N (comparative claim limited) |
| 7 | Latency / memory / model size / throughput | **[MISSING]** + **[FOUND]** | Instrumentation exists (wall-clock, memory snapshot, artifact checksums) but **no measured latency/size artifacts**; simulator type is recorded (exact statevector, analytic — no shots). Benchmark script provided, not executed. | N (works claim) |
| 8 | DFDC provenance / licensing / duplicates | **[FOUND]** + **[HUMAN-VERIFY]** | On-disk dataset = flattened `Real/` (2,127) + `Fake/` (1,966) folders, **no metadata files, no part structure, no source-to-fake map**; duplicate policy exists at feature-build level; dataset acquisition and licence terms unconfirmed. | **Y** (provenance section) |
| 9 | Thresholds & inconclusive pathway | **[FOUND]** | Complete threshold inventory assembled (§9); Youden threshold validation-calibrated; display-level bands untethered from calibration; NN `INCONCLUSIVE` binary inconsistency between docs and `run_pipeline.py`. | N |
| 10 | Fairness analysis | **[MISSING]** | No demographic or capture-condition metadata exists on disk (folder contains only MP4s, no JSON/CSV); no per-group metrics possible. | Y (if fairness is claimed) |
| 11 | Author contributions | **[FOUND]** + **[HUMAN-VERIFY]** | Git history has a **single committer identity** (148 commits, `NYN-05`); no LICENSE/CONTRIBUTORS files; README names a 3-person team — attribution is documentation only. | **Y** (submission requirement) |
| 12 | ViT verification | **[FOUND]** (verdict) | **Not implemented.** No ViT/attention/transformer code exists in the model path; visual backbone is ResNet50. ViT appears only in the *titles* of reference PDFs. Remove or implement. | **Y** (if ViT is claimed) |

**Top-level verdict:** the repository demonstrates a well-engineered, contract-tested *pipeline*, but **the quantitative evidence base required for publication currently does not exist on disk**: no complete training table (single class only), no current metrics artifacts (deleted), single-seed results on n = 8 test clips, no robustness/ablation/latency measurements. Items 1, 2 (metadata), 3, 5, 10 and 12 are the critical path.

---

## 1. Full-Dataset Retraining and Evaluation

### Findings

- **Entry points [FOUND]:**
  - Training: `python -m quantum.pipeline --all` (`WORKING/quantum/pipeline.py::run_pipeline_for_feature_set`, steps `[1/6] build_dataset` → `[6/6] baselines`), orchestrated one-shot by `setup_and_run.ps1` (flags: `-Quick`, `-CpuOnly`, `-Resume`, `-Start/-End`, `-GpuWorkers`).
  - Feature extraction: `RPPG/extract_dataset_features.py` (rPPG table), `visual/pipeline.py` (visual + fused table), `RPPG/train_classifier.py` (cross-check RF; `--incremental` mode).
  - Evaluation is embedded in stage `[4/6]` of `--all` (`quantum/evaluation.py::evaluate_quantum_model`) writing `metrics_quantum.json`; baselines in `[5/6]` writing `metrics_baselines.json`.
- **Dataset actually used [COMPUTED — file-system + artifact audit]:**
  - On disk (`DFDC_Dataset/`): **Real = 2,127 videos, Fake = 1,966 videos** (4,093 MP4 files total; flat layouts, 0 metadata files — §8).
  - Current feature table (`WORKING/output/rppg/dataset_features.csv`, header + row count computed): **1,024 rows, all `label = 1` (FAKE), all `source = DFDC_Dataset/Fake`**; 20 feature columns + `label`, `video_path`, `source`.
  - Extraction state (`WORKING/output/rppg/checkpoint.json`): `processed_count: 1024`, `no_features_count: 81`, `failed_count: 0`, `last_completed_index: 1104`, `total_samples: 0` (total unknown), config: `POS`, blur 3.0, brightness 15–245, `min_usable_frames: 24`, `min_sqi: 0.05`, `max_nan_features: 2`, `max_per_class: null`.
  - **Conclusion:** *full DFDC was never extracted end-to-end*. The current table is ~52 % of the Fake class and **0 % of the Real class**; it cannot support training or evaluation of a REAL/FAKE classifier as-is.
- **Recorded results provenance [FOUND, with artifact caveat]:** the published-looking metrics (VQC test + CV; baselines) in `PROJECT_RESEARCH_REPORT.md` §11–§13 were transcribed on 2026-10-11 from `WORKING/output/quantum/metrics_quantum.json` / `metrics_baselines.json` / `training_log.jsonl` produced by a **38-video smoke run (20 fake / 18 real; test split n = 8)**. Those artifact files **no longer exist** in the output tree (verified by recursive listing; only `rppg/checkpoint.json`, `rppg/dataset_features.csv`, `rppg/extraction.lck` remain). All such numbers are therefore **recorded-from-deleted-artifact figures** and must be regenerated before publication.
- **Frames per clip [MISSING]:** per-clip and per-class frame counts are produced transiently by stage 1 (`frame_extraction_summary.json`) but no dataset-scale frame inventory artifact exists.
- **Hyperparameters, seeds, optimiser, loss [FOUND]:**

| Component | Setting | Source |
|---|---|---|
| Split | seed 42, val 0.20, test 0.20, per-class greedy grouped | `quantum/config.py` `DataConfig` (lines 220–231) |
| HR plausibility filter | 30–220 BPM, non-finite drop | `config.py:225-227`; `data.py::_load_feature_rows` |
| QAOA | p = 3 layers, COBYLA `max_iter=500`, redundancy 0.3, cardinality 0.5, `target_features=3`, 8 restarts, seed 42, torch-native simulator | `config.py:235-250`; `qaoa.py` |
| VQC | `qml_layers=3`, `hidden_units=8`, dropout 0.2, **epochs 80**, **batch 256**, **AdamW lr 5e-2, wd 1e-2**, focal loss **α=0.45 (REAL-weighted), γ=1.0, label smoothing 0.03**, cosine-warmup LR (3 warm-up epochs), grad clip 1.0, early stopping patience 12 / min_delta 1e-4, **seed 42** | `config.py:253-284`; `vqc.py::train_vqc` (`torch.manual_seed(cfg.seed)` at `vqc.py:457`) |
| Pre-selection toward QAOA | classically pre-select **18 of 59** AUC-ranked candidates (statevector 20-wire ceiling), `verify_hamiltonian` assert < 1e-6 | `quantum/pipeline.py:174-181`; `qaoa.py` |
| rPPG cross-check RF | 600 trees, `min_samples_leaf=2`, `class_weight="balanced_subsample"`, seed 42 | `RPPG/train_classifier.py` |

### Gaps

- Real-class rPPG extraction (2,127 clips), visual extraction, fusion, and a full `--all` run — none have completed; frames-per-clip accounting absent.
- Current artifacts absent → recorded metrics unreproducible as stored.

### Recommended actions (protocol; heavy steps require author confirmation per ground rule 4)

1. Complete extraction: `python RPPG/extract_dataset_features.py --workers 0 --gpu --resume` (resume-safe; hours).
2. `python visual/pipeline.py --video-root <DFDC_DATASET_PATH> --rppg-csv WORKING/output/rppg/dataset_features.csv --fuse --create-splits`.
3. `python -m quantum.pipeline --all` → regenerate `metrics_quantum.json`, `metrics_baselines.json`, `training_log.jsonl`, plots; then reconcile README/paper tables with the new artifacts.
4. Record dataset provenance manifest (columns `video_sha256`, `source_id`) during extraction — this simultaneously unblocks items 2, 8 and 10 (see §2, §8).

---

## 2. Leakage-Tested Train / Validation / Test Split

### Findings — how the split is made [FOUND]

- File/function: `quantum/data.py::_grouped_train_val_test_split` (lines 227–298) called with `DataConfig(seed=42, val_ratio=0.2, test_ratio=0.2)`.
- Mechanism: per-class greedy group allocation — mixed-label groups first, then single-class groups shuffled by `np.random.RandomState(cfg.seed)` (line 253); each group is placed wholly into one split; `_assert_no_group_leakage` (lines 301–317) aborts if any group key spans splits. Deterministic and per-class balanced.
- Group keys: `_infer_subject_key` (lines 106–121) prefers metadata columns `GROUP_COLUMNS = (subject_id, identity_id, original_video_id, source_video_id, source_id, group_id, video_sha256, content_sha256, file_sha256, sha256)`; **fallback = `clip:<video_path>`**. The module docstring (lines 10–14) states the limitation explicitly: *"each clip is treated as an individual group ('clip:<path>') — a documented limitation (DFDC subject-level separation is unrecoverable from filenames)"*.
- Preprocessing-statistic isolation [FOUND]: the z-score scaler is fitted on train only (`quantum/pipeline.py` step `[2/6]`, `quantum/scaling.py`); the shared visual PCA is fitted **on training-partition videos only** (`visual/pipeline.py::_fit_shared_pca`, persisted read-only artifact `visual_pca.npz`); QAOA selection runs on train only; all three are consumed read-only at inference (`pipeline.py::predict_features` lines 85–115).

### Leakage audit table [COMPUTED]

Executed `paper_experiments/leakage_audit.py` (command/env in Appendix A; full JSON: `paper_experiments/leakage_audit_result.json`):

| ID | Check | Result | Evidence |
|---|---|---|---|
| A1 | Duplicate `video_path` / row integrity of rPPG CSV | 1,024 rows, 1,024 unique paths, 0 dropped by HR/non-finite filters; loader raises on duplicates (not triggered) | `data.py::_load_feature_rows` |
| A1b | Class balance of current table | **FAKE = 1,024; REAL = 0** — single-class table | computed counts |
| A2 | Subject-group key provenance | **0 rows carry metadata group keys; 1,024/1,024 fall back to `clip:<path>`; 1,024 distinct groups → `CLIP_LEVEL_GROUPING_ONLY`** | `_infer_subject_key` over all rows |
| A3 | Split determinism (same seed ×3) | `IDENTICAL_ACROSS_RUNS`; split sizes **train 614 / val 205 / test 205** | `_grouped_train_val_test_split` |
| A4 | Group overlap across splits | `NO_GROUP_SPANS_MULTIPLE_SPLITS` (0 groups span splits) | `_assert_no_group_leakage` + explicit scan |
| A5 | Scaler / PCA / QAOA fitted on train only | `TRAIN_ONLY_FITTING_ENFORCED_IN_CODE` | `pipeline.py`, `visual/pipeline.py` |
| — | Near-duplicate frame/clip hashes | **Not computable: CSV content-hash columns empty and per-frame hashes not persisted** | `GROUP_COLUMNS` all absent (A2) |

### Gaps and the structural risk

- **Identity/source-level leakage is unguarded and unmeasurable today.** DFDC fake clips are manipulations of *source* real videos; with only flattened clip IDs (00000.mp4 vs random-string.mp4, no `dfdc_metadata`), the same underlying identity/source can appear in train **and** test via independent clips. The grouping machinery exists and is correct, but the **datum (source mapping) does not**.
- Near-duplicate detection (hash/SSIM over clips) was never implemented; `video_sha256`/`content_sha256` support exists in code but requires those columns to be populated (loader resolves hash→group conflicts, `data.py:204-212`).
- The split was **not meaningful today anyway** — a single-class table cannot produce a REAL/FAKE split (A1b).

### Recommended actions

1. **Populate provenance columns at extraction time** (DFDC part-3-style fake↔source mapping or content hashes): extend `extract_dataset_features.py` to emit `source_id`/`video_sha256` per row so `_infer_subject_key` uses metadata. This converts A2 to `METADATA_GROUPED`.
2. Add a near-duplicate audit (perceptual hash over first/last frames or whole-clip hash) as a gate in `visual/pipeline.py::create_experiment_splits`.
3. Re-run the audit after regeneration; keep `_assert_no_group_leakage` as the hard gate (already test-covered: `tests/quantum/test_quantum.py`, `test_no_group_leakage`).
4. If source mapping can be recovered from the acquisition site (see §8), republish dataset manifests with the paper.

---

## 3. Confidence Intervals and Repeated-Run Variability

### Findings

- **[FOUND]** The only repeated-evaluation machinery is subject-grouped cross-validation on the *training* split: `quantum/evaluation.py::run_cv` (lines 179–204; `StratifiedKFold`/grouped variants, seed 42) → mean ± std recorded. Recorded VQC CV (transcribed artifact): accuracy 0.700 ± 0.197, AUC 0.700 ± 0.267, balanced accuracy 0.667 ± 0.230, F1 0.467 ± 0.400 (n = 8 fold sizes 4–5; `PROJECT_RESEARCH_REPORT.md` §11).
- **[FOUND]** Every configured seed is the fixed `seed: int = 42` (`config.py` lines 222, 241, 281; baselines seed 42 `evaluation.py:425`). A repeated-seed evaluation utility once existed but was **removed** in the 2026-10-11 consolidation (`changes.md`, "removed … `repeated_seed_evaluation` wrappers"). No multi-seed artifacts exist anywhere in the tree.
- **[COMPUTED]** 95% Wilson score intervals for the recorded test-split counts (computed *from recorded counts*, not a new model run; script `paper_experiments/ci_from_recorded.py`, results in `ci_from_recorded_result.json`):

| Metric (recorded) | Estimate | 95% Wilson CI |
|---|---|---|
| Accuracy (4/8) | 0.500 | **[0.215, 0.785]** |
| Recall REAL / sensitivity (3/4) | 0.750 | [0.301, 0.954] |
| Specificity / FAKE recall (1/4) | 0.250 | [0.046, 0.699] |
| Precision REAL (3/6) | 0.500 | [0.188, 0.812] |
| False acceptance rate (3/4) | 0.750 | [0.301, 0.954] |
| False rejection rate (1/4) | 0.250 | [0.046, 0.699] |

  These intervals quantify why the current evidence cannot support comparative claims: every interval spans chance level (0.5) or worse.
- **[MISSING]** ROC-AUC / PR-AUC / EER **confidence intervals**: single test-split values only (AUC 0.5625, PR-AUC 0.6679 recorded); EER is not computed anywhere in `evaluation.py` (searched) — no EER value exists at all.
- **[MISSING]** N ≥ 5-seed repeated runs. Protocol script written: `paper_experiments/multi_seed_repeats.py` (seeds 42,1,2,3,4/…; identical split via seed-42 dataset; per-seed VQC via `VQCConfig(seed=s)` → `torch.manual_seed`; baselines via `random_state=s`; per-seed JSON + aggregated mean ± std and both t-based and bootstrap 95% CIs; includes EER). It also states plainly that `PR-AUC`/`EER` must be added to the per-seed aggregation because the evaluator does not yet emit EER.

### Gaps / Recommended actions

1. Execute item 1 regeneration, then run `multi_seed_repeats.py --seeds 5` (fast on trained-artifact path; VQC retraining per seed ~minutes at this scale).
2. Add `eer()` to `quantum/evaluation.py` (10/11-point interpolation) and include `pr_auc`, `eer` in per-seed summaries.
3. Prefer **bootstrap CIs over clips** (identity-level resampling post-metadata-fix) over t-based intervals for publication; the script emits both.

---

## 4. Controlled Robustness Experiments

### Findings

- **[FOUND — absence]** A repository-wide search for degradation handling (resolution/bitrate/CRF/codec/noise/fps variation *as experimental factors*) returns no experiment code and no result tables; `PROJECT_RESEARCH_REPORT.md` §12 reached the same conclusion from artifacts. Robustness is only *implicit* in defensive thresholds (§9) and the "designed for low-resolution videos" qualifier in `README.md`.
- **[FOUND]** Related but non-experimental machinery that does exist: stage-1 quality gates and frame-fps comparison docs (`MAJ_OUTPUT_ROOT/frames/docs/frame_sampling_rate_comparison.md` writer at `run_pipeline.py:106-109`) — descriptive, not a controlled study.
- **[MISSING]** All condition × metric results.

### Provided protocol (not executed)

`paper_experiments/robustness_degrade.py` generates a single-factor ffmpeg grid on ONE source clip, isolated per condition under `paper_experiments/robustness_clips/<condition>/`, all else held fixed, and writes a results template `robustness_table_TEMPLATE.md` (condition × accuracy / balanced-accuracy / recall_REAL / specificity / ROC-AUC / observed verdict). Grid:

| Factor | Values |
|---|---|
| Resolution | 720p, 480p, 360p, 240p (`-vf scale=-2:H`) |
| Frame rate | 15, 10, 5 fps (`-r`) |
| Codec/CRF | libx264 CRF 23 / 28 / 35 / 42 |
| Bitrate | 4000k / 1000k / 500k / 200k (min=max buf) |
| Noise | gaussian `noise=alls=10|30:allf=t` |

Each condition is then evaluated with the standard `run_pipeline.py` path so the pipeline (not a bespoke evaluation) measures end-system robustness.

### Recommended actions

1. Author confirmation → run the grid on a representative 10-clip set (balanced, metadata-grouped post-item-2), fill the template, and add a *within-clip* paired analysis (same source identity across degradations) to the paper.
2. Include stage-1 and stage-2 *acceptance-rate* per condition (rejection counts change first at low quality) — this is the mechanism by which the pipeline becomes INCONCLUSIVE, which is itself a publishable observation.

---

## 5. Ablation Studies

### Findings

- **[FOUND]** The ablation capability was **retired**: `quantum/config.py:142-145` keeps only degenerate legacy placeholders (`RPPG_BASE_FEATURE_NAMES = RPPG_FEATURE_NAMES`, `CROSS_ROI_FEATURE_NAMES = []`, `PHASE6_FEATURE_SETS = {}`, `PHASE7_ABLATION_SETS = {}`), `FEATURE_SET_CONFIGS = {"fused": …}` only (`pipeline.py:43-51`); the sweep/experiment runner and ensemble modules were deleted on 2026-10-11 (`changes.md`).
- **[FOUND]** The only *partial* ablation evidence recorded anywhere: rPPG-only metrics are unavailable; however the rPPG-side RF cross-check trains on 20 rPPG features (`train_classifier.py`) and the smoke-run `selection_comparison_fused.json` recorded QAOA top-3 == classical top-3 (3/3 overlap; `PROJECT_RESEARCH_REPORT.md` §14). This is a selection-overlap observation, not an ablation.
- **[MISSING]** All configurations requested by the checklist under an identical split+seed: rPPG-only, visual-only, fused, with/without feature selection, with/without VQC (e.g., same features → small MLP instead of VQC). None exist as artifacts, and **not all are runnable today** (visual-only requires the visual table — absent; with/without VQC requires trained fused data — absent).

### Recommended actions

1. After item 1, reintroduce a *paper-local* ablation evaluator in `paper_experiments/` that reads `data_fused.npz` and slices feature groups by the canonical name lists (`RPPG_FEATURE_NAMES`, `VISUAL_FEATURE_NAMES`) — no changes to the fused-only contract, keeping AGENTS.md constraints intact.
2. Matrix to deliver (same seed=42 split; per-seed loops optional post-item 3): {rPPG-only, visual-only, fused} × {QAOA-3, all-selected-18, all-59} × {VQC, best-classical}. Report every cell or mark it consciously out of scope.

---

## 6. Classical Baseline Comparisons

### Findings

- **[FOUND]** Six classical baselines are implemented and evaluated on the *same QAOA-selected features, same split and same grouped CV* as the VQC (`quantum/evaluation.py::run_baselines`, lines 424–424ff; models defined lines 450–478): RandomForest(200), MLP(32,16, max_iter 500), LogisticRegression (balanced, max_iter 2000), calibrated LinearSVC (sigmoid, cv 3), GaussianNB, XGBoost (300 rounds, lr 0.1, depth 3, `tree_method=hist`, CUDA when available). Recorded smoke figures exist (transcribed; n = 8): e.g. RF acc 0.500/AUC 0.500; GaussianNB and XGBoost 0.625; LR/SVC AUC 0.750; RF CV-AUC 0.933 vs test 0.500 (§6.4 of the previous report; artifacts deleted).
- **[FOUND — deviation]** **XGBoost early stopping checks training data**: `model.fit(X_train, y_train, eval_set=[(X_train, y_train)])` with `early_stopping_rounds=20` (lines 466–487). Monitoring loss on the *training* set can stop the booster at an optimistic point; a validation `eval_set` (or none) should be used. Flagged as a questionable practice under ground rule 5.
- **[FOUND]** No CNN/ViT-style deep baselines exist in the codebase; the RF cross-check (`RPPG/train_classifier.py`) is rPPG-side only and is documented as a diagnostic, not a decision path.
- **[COMPUTED]** Trainable-parameter counts for the recorded smoke configuration (script `paper_experiments/param_counts.py`, results in `param_counts_result.json`):

| Model (as configured) | Trainable parameters |
|---|---|
| Hybrid VQC — 3 selected inputs, 3 layers, head 8 (wires = 3) | **68** (27 quantum + 41 classical) |
| Hybrid VQC if no classical pre-selection (18 inputs) | 323 |
| MLP baseline (hidden 32-16, input = 3 selected) | 673 |
| LogisticRegression (input 3) | 4 |
| RandomForest (200 trees) / XGBoost (300 rounds·3-depth) | variable per fit — **[MISSING]** (node counts not persisted) |

- **[FOUND]** Tuning budgets: **none** — all baselines run at single fixed hyperparameter settings (`random_state=seed`); no search/scheduling code exists. The VQC's early stopping is validation-based (correct), in contrast to the XGBoost issue above.

### Gaps / Recommended actions

1. Fix the XGBoost `eval_set` to use a validation slice of the *training* data (or disable early stopping) before the publication run; document the change.
2. Persist per-fit `n_estimators`/tree counts (or `get_params`) into `metrics_baselines.json` for the parameter-count table.
3. For fairness of the *quantum* claim, add a parametrised classical head-only ablation (same 68/323-parameter budget) — connects to item 5.

---

## 7. Reproducible Latency, Memory, Model Size, Throughput

### Findings

- **[FOUND]** Simulator / hardware nature of the quantum component:
  - QAOA backend: **exact torch-native dense statevector simulator**, complex128, CPU process-safe (`qaoa_sim.py::QAOASimulator`; config `device="auto"`, `config.py:244-249`); PennyLane `lightning.qubit` retained as cross-check. Evaluation is **analytic** (exact expectations) — i.e. *no shot sampling occurs anywhere in the pipeline*; a "shots" parameter does not exist in the code.
  - VQC inference: `QuantumLayerTorch`, exact complex128 statevector (`vqc.py`); analytic probabilities, no shots.
- **[FOUND]** Latency/memory instrumentation that exists but has no stored results: wall-clock in `run_pipeline.py` diagnostics (`processing_time_seconds`), `memory_snapshot()` from `quantum/operations.py` (RSS/CUDA alloc/reserved — recorded as null placeholders in the absent result JSON lineage), server-side 30-minute hard cap (`frontend/server.py`), and docstring-level micro-claims (statevector "≈0.3–0.5 ms @ 20 wires", "≈20× vs PennyLane"; sklearn import ≈12–13 s comments). None of these are backed by a benchmark artifact in the tree.
- **[MISSING]** Per-stage latency (frames/rPPG/visual/selection/classifier), peak RAM/VRAM, model sizes on disk (`hybrid_vqc_fused.pt`, `feature_scaler_fused.json`, `qaoa_selection_fused.json`, `rppg_classifier.pkl`, `visual_pca.npz` — all currently absent), throughput (videos/s), warm-up protocol. Parameter counts for the VQC: **[COMPUTED]** 68 trainable (3-wire configuration; §6).
- **[COMPUTED]** Hardware/software environment (live probe; full record Appendix A): Intel Core (Raptor Lake, Family 6 Model 183), **NVIDIA GeForce RTX 4050 Laptop GPU, 6,140 MB VRAM**, CUDA available with torch 2.5.1+cu121, Python 3.11.9, Windows (build 26200), PennyLane 0.44.1, NumPy 1.26.4, scikit-learn 1.9.0, SciPy 1.17.1, OpenCV 4.8.1, ONNX Runtime GPU 1.20.2. Batch size (VQC): 256. A PennyLane deprecation warning against NumPy < 2 was emitted during the probe — recorded as environment tension (Appendix B).

### Provided protocol (not executed; requires author confirmation)

`paper_experiments/latency_benchmark.py`: warm-up 2 + repeats ≥ 5 per stage via subprocess timing around each stage's public entry point, `psutil` peak RSS + CUDA `max_memory_allocated`, disk-size collection of the artifact set, throughput = 1/(sum of means); emits `benchmark_env.json` + per-stage mean ± std table.

### Recommended actions

1. After artifacts are regenerated, run the benchmark once per device mode (CPU-only, CUDA) with identical video set; quote *simulator-analytic* (statevector, no shots; "shot count" = analytic/infinite) explicitly in the paper to avoid reviewer ambiguity.
2. Publish hardware in the reproducibility appendix (Appendix A table) rather than prose.

---

## 8. DFDC Provenance, Licensing, Source-Part Mapping, Duplicates

### Findings

- **Dataset layout on disk [COMPUTED — recursive scan]:** `DFDC_Dataset/Real/` = 2,127 MP4s named `00000.mp4`, `00001.mp4`, …; `DFDC_Dataset/Fake/` = 1,966 MP4s with alphanumeric stems (`aaagqkcdis.mp4`, …); **zero subdirectories; zero JSON/CSV/TXT/XLSX metadata files anywhere** in the tree (search over `*.json,*.csv,*.txt,*.md,*.xlsx` returned none; 4,093 total files, all MP4).
- **Path wiring [FOUND]:** `.env`/`DFDC_DATASET_PATH` must point at `DFDC_Dataset` (default set by `setup_and_run.ps1:60-62`); extractors walk `Real/` and `Fake/` folders and label by folder (rPPG CSV `1 = fake`) — the only "source" column emitted today is the folder name (`…/DFDC_Dataset/Fake`), as visible in the CSV first row.
- **Source-to-fake mapping [MISSING]:** no `dfdc` part/metadata JSON; `_infer_split_key` even documents "DFDC has no official train/val/test folder structure" (`data.py:124-130`). Because real stems are numeric and fake stems are unrelated random strings, the fake→source relation is **unrecoverable from files** (consistent with `data.py:10-12`).
- **Duplicate policy [FOUND]:** at the feature-table level, duplicate `video_path` is fatal (`data.py:200-201`) and hash→group consistency is enforced if hash columns exist (`data.py:204-212`); at the visual level, duplicate `video_path` is rejected in `create_experiment_splits`, and the same video cannot appear in two splits (`_assert_no_group_leakage`). **Near-duplicate content detection: absent** (see §2).
- **Licensing and acquisition [HUMAN-VERIFY]:** no LICENSE file exists in the repository, and no document records where the 4,093-clip corpus was obtained (folder naming `archive/DFDC_Dataset` suggests a Kaggle "archive/" extraction, but that is inference, not evidence). Authors must confirm: (a) acquisition source and date; (b) DFDC's licence/terms permit their use and redistribution-quotes; (c) any additional consent/attribution requirements; (d) institutional ethics approval for KYC data use. The DFDC dataset's own separate licence terms apply to the data, independently of the code licence.

### Recommended actions

1. Regenerate or download the official DFDC metadata (parts + fake→source JSON) or re-derive `source_id`/`video_sha256` at extraction; store as a repository manifest (data stays gitignored).
2. Create `LICENSE` (code) and a DATASET statement section in the paper; both are [HUMAN-VERIFY] items.

---

## 9. Decision Thresholds and the Inconclusive / Human-Review Pathway

### Threshold inventory [FOUND]

| Threshold | Value | Location | Chosen how |
|---|---|---|---|
| Frame blur rejection | Laplacian variance < 3.0 | `frame/pipeline.py:75` region (constants block; relaxed for DFDC) | Hard-coded (documented as "relaxed for short/low-quality videos") |
| Frame dark / overexposed | mean < 30.0 / > 235.0 | `frame/pipeline.py`:77-79 | Hard-coded |
| Face too small / extreme pose | area < 0.002 / aspect 0.65–1.7 | `frame/pipeline.py`:80-83 | Hard-coded (DFDC 112×112 tiles comment) |
| YOLO face confidence / input size | 0.25 / 320 | `run_pipeline.py:98-99` (frame stage call) | Hard-coded |
| Composite quality weights | 0.4 blur / 0.3 brightness / 0.3 face | `frame/processing.py:175` | Hard-coded |
| Minimum accepted sequence | 32 frames @ 10 FPS | `frame/pipeline.py:61` | Hard-coded |
| rPPG per-frame gates (direct-video path) | blur ≥ 3.0, brightness 15–245 | `RPPG/pipeline.py:130-131` region; also in checkpoint config | Hard-coded |
| Minimum usable frames | **24** (26→ no; dataset extraction same; the `--min-usable-frames 48` exists only for the incremental trainer) | `checkpoint.json config`; `RPPG/pipeline.py` | Hard-coded; tuned during development (`changes.md`) |
| Minimum SQI | 0.05 | `checkpoint.json config`; `RPPG/pipeline.py:130-131` | Hard-coded |
| ROI validity floor | ≥ 30 % valid samples per ROI | `RPPG/pipeline.py` combination path | Hard-coded |
| ROI combination weights | 0.35 / 0.35 / 0.30 | `RPPG/pipeline.py:132` | Hard-coded |
| Band-pass / detrend | 0.7–4.0 Hz; λ = 300 | `RPPG/pipeline.py:128-129`; `RPPG/preprocessing.py:34` | Hard-coded (physiological literature defaults) |
| HR plausibility | 30–220 BPM | `config.py:226-227` | Hard-coded |
| Max NaN features per row | 2 | `checkpoint.json config` | Hard-coded |
| rPPG→quantum label flip | `1 - csv_label` | `data.py:43-61` | Contract |
| Decision threshold (default) | `prob_real ≥ τ → REAL`; τ = 0.5 default | `config.py:288-292` (`DecisionConfig`) | Default; superseded by checkpoint value |
| Decision threshold (calibrated) | Youden-J on **validation** probabilities, persisted in checkpoint metadata; recorded smoke value **0.50676** | `evaluation.py::optimal_threshold_youden` (lines 278–289); applied in `pipeline.py:109-114` | **Validation-calibrated** |
| Confidence (display) | `2·|p−0.5|` | `quantum/explain.py` via `predict_features` | Hard-coded formula |
| Frontend display bands | confWord (≥0.6 High, ≥0.3 Moderate), probabilityWord (0.75/0.55/0.45/0.25) | `frontend/src/lib.js:52-66` | Display-only, uncalibrated |

### Inconsistencies flagged [FOUND] (also Appendix B)

- `run_pipeline.py` prints 4 stages `[N/4]` while the web layer keys on `[N/3]` (`frontend/server.py:85 STAGE_TAGS`; `App.jsx:97` regex) → stage events dead.
- `AGENTS.md`/README describe only REAL/FAKE outcomes; the executable contract retains `INCONCLUSIVE` with exit 3 (`run_pipeline.py:338-345`; `frontend/contract.py::VERDICTS = {"REAL","FAKE","INCONCLUSIVE"}`), and the frontend has an Inconclusive copy block (`lib.js:26-35`). Wording/logic must be reconciled once.
- The UI's "confidence" bands are cosmetic and disconnected from the calibrated probability — a reviewer-visible drift.

### Calibration evidence [FOUND — recorded] and proposal

- Recorded smoke-run calibration: **ECE 0.1881**, **Brier 0.2465**, reliability diagram `calibration_curve.png` generated by `evaluation.py` (artifacts deleted; regenerate). The VQC's recorded ECE was the lowest among the recorded models (§6.4), but n = 8.
- **Proposed scheme (grounded in existing code):**
  1. Keep the Youden-validated threshold τ* (already implemented, persistence-, and perish-verified via checkpoint metadata + schema digests).
  2. Add an **inconclusive band** on *evidence quality* rather than a second probability cut: gate on the existing, already-implemented **PQS** (`RPPG/pqs.py`, tiers HIGH ≥ 0.7 / LOW ≤ 0.3) plus `signal_quality_index` — e.g. outputs with `PQS < q_val` (q_val chosen on validation to meet a target coverage × precision trade-off) are labelled `INCONCLUSIVE` (exit 3 already carries this semantics: §AGENTS contract; the frontend already renders it).
  3. Optionally add a margin band `|p − τ*| < δ` calibrated on validation to reach a target precision on the *decided* subset; δ replaces the today-hard-cut binary semantics without deleting the REAL/FAKE contract (AGENTS.md forbids a three-class verdict; `INCONCLUSIVE` is a *pipeline* state, not a classifier class).
  4. Report ECE/Brier/reliability per threshold scheme; the evaluation code already computes all three (`evaluation.py:64-81`).

---

## 10. Fairness Analysis

### Findings

- **[COMPUTED — absence]** A recursive scan of `DFDC_Dataset/` found **only MP4 files (4,093)** and **no metadata files of any kind**; the rPPG CSV carries only features + `label` + `video_path` + `source` (folder name). Therefore no skin tone, gender, age, lighting, camera, or resolution metadata exists in the repository.
- **[FOUND — absence]** No fairness module, stratified evaluation, or per-group metric code exists anywhere in `WORKING/` or `tests/` (searched for demographic/group-stratified_evaluation patterns).
- **[HUMAN-VERIFY]** Whether DFDC's official distribution (as downloaded) *did* include demographic metadata (the public DFDC metadata files include some attributes) — i.e., whether the omission is a local subsetting artefact or a characteristic of this particular acquisition. Authors must confirm what was downloaded.

### What is needed

1. Re-obtain or reconstruct official DFDC metadata (fake→source mapping, and whatever attributes the shipped metadata includes).
2. Decide annotation scope: at minimum, binary perceived skin tone (e.g., Fitzpatrick or Monk scale buckets), gender presentation, and age band — annotated per *identity*, with inter-annotator agreement protocol if manual.
3. Capture-condition attributes are partially derivable automatically and should be persisted at extraction: source resolution/fps/bitrate (ffprobe), mean luminance (already computed per frame — currently discarded), frame acceptance rate per clip.
4. Then extend `quantum/evaluation.py` with a per-group metrics report (TPR/FPR/F1/AUC per group + gaps), reusing the existing grouped-split machinery for grouping integrity.

---

## 11. Author Contributions and Final Author List

### Contribution evidence table [COMPUTED — git; command in Appendix A]

| Evidence type | Finding | Source |
|---|---|---|
| Git committer identities | **Exactly one**: `NYN-05 <jnyn2005@gmail.com>`, **148 commits** up to `114e0bd` | `git log --format="%an|%ae"` group counts |
| Module-lineage (file-commit events per package) | RPPG 6,805; quantum 206; frame 56; visual 25 | `git log --numstat` aggregated by top-level package |
| CONTRIBUTORS / LICENSE files | **None exist** (checked `LICENSE`, `LICENSE.md`, `LICENSE.txt`, `CONTRIBUTORS`, `CONTRIBUTING.md`) | filesystem probe |
| README team table | Lists **Jhashank** (rPPG→classifier + quantum), **Sumit** (input→ROI), **Aswin** (preprocessing/eval/integration) — as documentation, unsupported by git identity | `README.md` Team section |
| File headers | No per-author headers/models/copyright notices in `WORKING/**` source files | grep |
| Prior reports | `research_paper_report.md` / `problem_and_bottleneck.md` authored as project documents (no named author list inside) | docs review |

**Deviation flagged (ground rule 5):** all git history is under a single identity, so git **cannot** substantiate or refute per-person contributions; the reorganisation history (bulk refactors recorded in `changes.md`) further obscures authorship. Any per-author contribution statement must therefore be assembled from project knowledge by the authors — the table above is the *only* evidence the repository itself contains.

**[HUMAN-VERIFY]** Final author list, author order, and the CRediT-style contribution statement must be decided by the team. Recommended submission hygiene: decide whether `NYN-05` is an organisation account representing the team (state this in the paper) and align the git identity with the author names or document why it does not.

---

## 12. Vision Transformer (ViT) Verification

### Findings

- **[FOUND]** A case-sensitive search for genuine ViT markers (`vit_base`, `vision_transformer`, `models.vit`, `MultiheadAttention`, patch-embedding class definitions, `ViT\b`) across all Python sources in `WORKING/` returned **zero matches**. The visual backbone in the fused path is **ResNet50** (torchvision `IMAGENET1K_V2`, FC-stripped, GAP → PCA-16; `visual/extractor.py`).
- **[FOUND — frontend]** The only "ViT-like" strings in the entire repository are (a) the *filenames* of reference PDFs in the research corpus (`QHybrid_XceptionViT_QNN_2025.pdf`, `Multimodal_Vision_Transformer_Forensics_...pdf`, `FakeFormer_...`) and (b) incidental substrings such as "Activity"/"sensitivity"/"positive" — neither is an implementation.
- **[FOUND]** Proof that the training/inference path contains no second backbone: the fused contract enumerates all 59 features by name (`FUSED_FEATURE_NAMES`, 20 + 39; `config.py:134`), the 39 visual features are all ResNet50-PCA/LBP/GLCM/colour/DCT (`extractor.py::VISUAL_FEATURE_NAMES`), and `predict_features` accepts exactly this dict. No transformer code participates.

### Verdict: **Not implemented.**

No ViT architecture (patch embedding, attention layers, pretrained ViT weights) exists anywhere in the codebase, let alone in the inference or training path.

### Recommended actions

1. **Remove** any ViT-based contribution claim from the paper draft (only QAOA + hybrid VQC + rPPG/visual fusion is implementable as claimed, evidenced here) **or** implement a genuine ViT branch (patch-embed → transformer encoder → head, pretrained weights frozen/tunable, integrated as extra visual features appended to `VISUAL_FEATURE_NAMES` with contract/test updates) — the latter is a documented, large change and must not silently enter the fused contract.
2. If the team's earlier slide text mentions ViT (observed in review PDFs, not in code), correct the claim before submission.

---

## Appendix A — Experiment Commands and Environment

### A.1 Environment (live probe; [COMPUTED] 2026-10-19)

| Item | Value |
|---|---|
| OS | Windows, build 26200 (Windows-10-10.0.26200-SP0 as reported by `platform`) |
| Python | 3.11.9 (`venv\Scripts\python.exe`) |
| CPU | Intel64 Family 6 Model 183 Stepping 1 (Raptor Lake laptop), GenuineIntel |
| GPU | **NVIDIA GeForce RTX 4050 Laptop GPU**, 6,140 MB VRAM, `torch.cuda.is_available() == True` |
| torch / torchvision / aud| 2.5.1+cu121 / 0.20.1 / (torchaudio installed, unused in path) |
| PennyLane | 0.44.1 (lightning installed); emits deprecation warning vs NumPy 1.26.4 |
| NumPy / SciPy / scikit-learn / OpenCV / ONNX Runtime GPU | 1.26.4 / 1.17.1 / 1.9.0 / 4.8.1 / 1.20.2 |
| VQC batch size | 256 (`config.py:259`) |
| Quantum execution | **simulator**: exact complex128 statevector, analytic expectations (**no shots**) — `qaoa_sim.py`, `vqc.QuantumLayerTorch` |

### A.2 Commands executed in this audit ([COMPUTED]; each < 1 min, read-only)

```text
git rev-parse HEAD
git log --format="%an|%ae" | group-count
Get-ChildItem DFDC_Dataset -Recurse … (structure + counts: Real 2,127 / Fake 1,966; 0 md/Json)
Import-CSV WORKING\output\rppg\dataset_features.csv | group source,label   # 1,024 × (Fake,1)
Get-Content WORKING\output\rppg\checkpoint.json                            # extraction state
.\venv\Scripts\python.exe paper_experiments\leakage_audit.py               # split/leakage audit
.\venv\Scripts\python.exe paper_experiments\ci_from_recorded.py            # Wilson CIs
.\venv\Scripts\python.exe paper_experiments\param_counts.py                # parameter counts
.\venv\Scripts\python.exe -c "<environment probe: torch/pennylane/versions>"
```

Generated artifacts (new files only; no source modified): `paper_experiments/leakage_audit.py` + `leakage_audit_result.json`, `ci_from_recorded.py` + `ci_from_recorded_result.json`, `param_counts.py` + `param_counts_result.json`, `multi_seed_repeats.py`, `robustness_degrade.py`, `latency_benchmark.py`, `README.md` (protocol).

### A.3 Heavy runs prepared but **not executed** (awaiting author confirmation, per ground rule 4)

- Full-DFDC extraction + fusion + `python -m quantum.pipeline --all` (hours; overwrites/regenerates `MAJ_OUTPUT_ROOT` artifacts).
- `robustness_degrade.py` grid (requires ffmpeg; writes ~15 clips × set).
- `latency_benchmark.py` (≥ 7 full pipeline executions; > 30 min budget).
- `multi_seed_repeats.py --seeds 5` (after item 1 regeneration).

---

## Appendix B — Documentation vs Code Discrepancies

| # | Discrepancy | Code truth (source cited) | Documentation claim (source) |
|---|---|---|---|
| B1 | Published performance table | Recorded artifact figures: acc 0.500 / bal 0.500 / AUC 0.5625 (transcribed from deleted `metrics_quantum.json`; §1) | `README.md` table shows acc 0.552 / bal 0.499 / AUC 0.535 with "TBD" classical column |
| B2 | Stage tags | `run_pipeline.py` prints `[1/4]…[4/4]` | `frontend/server.py:85` + `App.jsx:97` + `lib.js::stageActive` key on `[N/3]`; stage SSE events never fire |
| B3 | Feature count copy | 20 rPPG / 59 fused (`RPPG/features.py`; `config.py:134`) | `frontend/src/lib.js:43` says "Measuring **29** physiological features"; `run_pipeline.py:14` docstring says "24-feature vector" |
| B4 | Frontend component names | Actual components: `VerdictCard`, `KeyMetrics`, `SignalPanel`, `CrossCheck`, `PreviousRun`, `ProcessingView`… | `frontend/README.md` describes nonexistent `VerdictGauge`, "Insights", `useJob`/`useSSE` |
| B5 | Decision outcomes | Executable contract: REAL / FAKE + pipeline-state `INCONCLUSIVE` (exit 3; `contract.py`, `run_pipeline.py:338-345`) | `AGENTS.md` headline says "two decision outcomes: REAL and FAKE" (INCONCLUSIVE governed elsewhere in the same file) |
| B6 | Version manifests | `requirements.txt`: pandas **==3.0.5**, mediapipe **==1.0.1**, xgboost 3.2.0 | `pyproject.toml`: pandas >=2.0.0, mediapipe >=0.10.9, xgboost >=2.0.0 — drifting pins between the two manifests |
| B7 | Output-root default | Code fallback = `Scrape/output` (`config.py:216`, `run_pipeline.py:60`) when `MAJ_OUTPUT_ROOT` unset | `setup_and_run.ps1` default fallback = `WORKING\output`; `.env` sets `WORKING\output` — three resolution sites, one divergence risk |
| B8 | Dependency maintenance | PennyLane 0.44.1 warns NumPy 1.26.4 is below its maintained floor (probe output) | README install guidance implies current pins are final |
| B9 | XGBoost early stopping | `eval_set=[(X_train, y_train)]` monitors training loss (`evaluation.py:487`) | Documented as "with grouped CV" only — no caveat given |
| B10 | Dataset scope | On-disk corpus = 4,093 flattened clips, no metadata; extraction table Fake-only (1,024) | Docs describe "DFDC only" as if fully available/trained |
| B11 | Seed policy | All stages fixed seed 42; repeated-seed wrappers removed on 2026-10-11 (`changes.md`) | Prior docs mention "repeated-seed summaries" as a capability |

---

## Appendix C — Prioritised To-Do List Before Submission

**P0 — evidence-critical (blocks all quantitative claims)**
1. Complete Real-class rPPG extraction (2,127 clips) and Fake remainder; persist `source_id`/`video_sha256` per row (items 1, 2, 8).
2. Regenerate visual + fused tables and run `python -m quantum.pipeline --all` end-to-end; reconcile README with regenerated `metrics_*.json` (items 1, B1).
3. Re-run leakage audit (A2 must flip to metadata-grouped) + near-duplicate gate (item 2).
4. Execute `multi_seed_repeats.py --seeds 5` (+ add EER/PR-AUC to the evaluator); report mean ± std + bootstrap CIs (item 3).

**P1 — claim-critical**
5. Decide the ViT story: remove from claims (recommended; code shows none) or implement properly (item 12).
6. Run the robustness grid + latency benchmark after author confirmation; fill templates (items 4, 7).
7. Build the paper-local ablation evaluator for rPPG-only/visual-only/no-selection/param-matched-classical-head (items 5, 6); fix the XGBoost validation-set early stopping first (B9).
8. Obtain/confirm dataset licence terms + acquisition record; add repository LICENSE (items 8, 11).

**P2 — hygiene (before camera-ready)**
9. Fix `[N/3]` vs `[N/4]` stage tags and UI "29 features" text; align AGENTS.md wording on INCONCLUSIVE (B2, B3, B5).
10. Persist per-fit baseline parameter counts; document seed policy; record hardware in the repro appendix (items 6, 7, A.1).
11. Fairness: metadata re-acquisition + per-group metrics reporter + annotation protocol (item 10).
12. Unify version pins between `requirements.txt` and `pyproject.toml`; address PennyLane/NumPy maintenance warning (B6, B8).

---

*Report ends. Every [COMPUTED] number above is reproducible with the commands in Appendix A; every [FOUND] number cites its file; every [MISSING] item names exactly what must be produced; every [HUMAN-VERIFY] item is a decision reserved to the authors.*
