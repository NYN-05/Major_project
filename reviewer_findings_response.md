# Reviewer Findings Response — Item-by-Item Project Status

**Project:** Deepfake Video Detection for KYC — rPPG + visual fusion, QAOA feature selection, hybrid VQC (DFDC)
**Response date:** 2026-10-21
**Evidence basis:** this repository's code and artifacts; the 2026-10-19 code audit (`publication_readiness_report.md`, commit `114e0bd`); `for_synopsys.md`; `research_paper_report.md`. Every claim cites its source; every gap is named exactly.
**Status vocabulary:** **[EXISTS]** = implemented/verified in code or artifacts · **[PENDING RUN]** = machinery exists, result requires the full-dataset run · **[READY PROTOCOL]** = script provided, execution awaits author confirmation · **[HUMAN DECISION]** = reserved to the team, cannot be produced by the repository · **[OUT OF SCOPE]** = justified in §4.

## 0. Corrections that post-date the reviewer's table

Two facts changed after the review was drawn up, and one fix landed on 2026-10-21:

| Fact | At review time (≈2026-10-19) | Now (2026-10-21) |
|---|---|---|
| rPPG feature table | 1,024 rows, FAKE-only | **Complete.** All 4,093 clips walked (`output/rppg/checkpoint.json`: `processed_count 3914`, `no_features_count 179`, `failed_count 0`, `gated_count 0`); **3,914 labelled rows — 2,087 REAL / 1,827 FAKE** |
| Visual + fused table | absent (stage 5 crashed near video 1950/3,914 by memory exhaustion) | **Fixed 2026-10-21.** Streaming, checkpoint-resumable extraction (`visual/pipeline.py` rewrite: per-video `parts/` checkpoint store, OOM retry, fail-fast exit 2; smoke test 11/11 PASS). Stages 5–7 complete on the next full `setup_and_run.ps1` rerun — the rerun is author-triggered and was deliberately not auto-executed |
| Quantum stage | unchanged | unchanged (fused-only contract; `python -m quantum.tests` 9/9 runnable checks pass) |

So item 1 is now *partially satisfied* on the rPPG side and the remaining chain (visual → fuse → quantum train/evaluate) is one rerun away — **pending, not structurally absent**.

## 1. Results labeling — the reviewer's closing paragraph (accepted)

Recorded figures: 38-clip smoke run (20 fake / 18 real), **n = 8 test split**, VQC accuracy **0.500** (AUC 0.5625), GaussianNB and XGBoost **0.625** on the same split. On this evidence:

- At n = 8, the 95% Wilson interval for accuracy 0.500 is **[0.215, 0.785]** — every recorded metric spans chance level and no comparative claim is supportable (`paper_experiments/ci_from_recorded_result.json`).
- The smoke metric artifacts (`metrics_quantum.json`, `metrics_baselines.json`) were deleted after transcription, so all tracked documents carry recorded-from-deleted-artifact figures.
- **Required wording (adopted project-wide):** these numbers may only be labelled **"preliminary feasibility results"** — proof that the pipeline trains, evaluates, and emits metrics end-to-end — never final performance. The paper must not state "quantum beats classical" (or the reverse) from the smoke run; classical models matching/exceeding the VQC on this tiny split must be reported as-is in that label.
- Final performance tables must come only from regenerated full-dataset artifacts (`WORKING/output/quantum/metrics_quantum.json`, `metrics_baselines.json`, `training_log.jsonl` from `python -m quantum.pipeline --all`), which do not exist yet.

## 2. Item-by-item response

| # | Requirement (reviewer verdict) | Project answer — current status | Scope | Path to close |
|---|---|---|---|---|
| 1 | Full-dataset training on 4,093 clips — Missing | **[PENDING RUN]** rPPG stage complete: 3,914/4,093 clips labelled, both classes (2,087 REAL / 1,827 FAKE). The 179-clip deficit = clips yielding no usable rPPG under the evidence-quality floor (≥24 usable frames @10 fps, SQI ≥ 0.05, blur ≥ 3.0 — `checkpoint.json config`), i.e. clips that legitimately abstain (→ INCONCLUSIVE) rather than corrupt data. Visual, fusion, quantum train/eval pending | In scope — pending rerun | Full `setup_and_run.ps1` rerun (stage 5 now resume-safe/fixed) |
| 2 | Final generalization metrics — Missing | **[PENDING RUN]** Evaluation code [EXISTS]: `quantum/evaluation.py::evaluate_quantum_model` (stage 4/6) + `run_baselines` (stage 5/6); metric artifacts deleted → must be regenerated | In scope — pending rerun | Regenerate via `python -m quantum.pipeline --all`; then multi-seed repeatability (`paper_experiments/multi_seed_repeats.py`) and bootstrap CIs; add EER to the evaluator |
| 3 | Full-dataset identity/duplicate leakage verification — Missing | **[EXISTS]** (machinery) + **[MISSING]** (data basis). Deterministic grouped split + hard `_assert_no_group_leakage` gate; duplicate `video_path` fatal; sha256→group conflict check (`quantum/data.py`); split/scaler/PCA/QAOA train-only fitting — all verified. BUT: on-disk corpus is flattened folders with **zero metadata files** → grouping degenerates to clip-level (structural); near-duplicate audit absent; `paper_experiments/leakage_audit.py` result is stale (ran on the old 1,024-row table) | In scope (machinery + provenance columns); official source mapping = external/user | Re-run leakage audit on the 3,914-row table; persist `video_sha256`/`source_id` per row at extraction; re-obtain official DFDC fake→source metadata (author action, §8) |
| 4 | Controlled resolution/compression/codec/bitrate/noise experiments — Missing | **[READY PROTOCOL]** No experiment results exist anywhere [MISSING].<br>`paper_experiments/robustness_degrade.py` provides the reproducible single-factor ffmpeg grid (720p→240p; 15/10/5 fps; CRF 23−42; 4 M→200 k; noise σ10/30) evaluated through the standard `run_pipeline.py` path | **Out of scope of current code work** (§4.3); executable when authorized | Author confirmation → run grid on a balanced 10-clip set + within-clip paired analysis; include per-condition acceptance rates (the abstention mechanism) |
| 5 | rPPG-only / visual-only / fused / classifier ablations — Missing | **[MISSING]** (results) + **[EXISTS]** (part): six classical baselines share split+CV with the VQC (`quantum/evaluation.py::run_baselines`); ablation feature-set machinery was retired 2026-10-11 (repo rulebook: fused-only). Feature-name groups `RPPG_FEATURE_NAMES` (20) / `VISUAL_FEATURE_NAMES` (39) still exist, so a paper-local ablation evaluator is possible without touching the fused contract | In scope — pending rerun (+ paper_experiments script) | After item 1: ablation matrix {rPPG-only, visual-only, fused} × {QAOA-3, all-18, all-59} × {VQC, best classical} on the same seed-42 split |
| 6 | Latency, memory, throughput, model-size benchmarks — Missing | **[READY PROTOCOL]** + partial [EXISTS]: instrumentation in the pipeline (wall clock, `memory_snapshot`, artifact checksums); parameter counts [COMPUTED]: **VQC 68** (27 quantum + 41 classical, 3-wire) / 323 if 18 inputs; MLP 673; LR 4; quantum backend = exact complex128 statevector, **analytic (no shots)** — must be quoted as such | In scope — pending rerun (+ one benchmark run) | Run `paper_experiments/latency_benchmark.py` per device mode after artifacts regenerate; persist per-fit tree counts for RF/XGBoost |
| 7 | Fairness / demographic subgroup analysis — Missing | **[OUT OF SCOPE]** (§4.4) — no demographic or capture-condition metadata exists on disk (corpus = MP4s only, verified twice); no fairness code exists; evaluation cannot be stratified today | Out of scope unless metadata re-obtained | Re-obtain official DFDC metadata (if it ships demographic attributes) or annotate identities; add per-group metrics reporter |
| 8 | Official DFDC provenance, license, manifest, source mapping — Missing | **[HUMAN DECISION]** + manifest population in scope. On-disk layout verified: `Real/` = 2,127 numeric-named MP4s, `Fake/` = 1,966 alphanumeric MP4s, **zero metadata files**; acquisition site unrecorded (Kaggle-`archive/` naming is inference, not evidence); no repo LICENSE | Human + extraction change | Author records acquisition source/date/licence terms; obtain official metadata (fake→source map — also unlocks items 3, 7); populate provenance columns at extraction; add LICENSE |
| 9 | Ethics / consent / privacy / data-processing documentation — Missing | **[HUMAN DECISION]** Document deliverable, not code. Repo facts that feed it: only public DFDC prototype clips are processed (no live KYC subjects, no personal data); uploaded user videos are ephemeral (job records expire after 1 h, frame sequences after 24 h, per `frontend/server.py` documented behavior); upload validation + size limits + traversal protection enforced | Human | Write the ethics/data-processing section in the paper; confirm DFDC licence terms (§8) govern the dataset |
| 10 | Functional abstention / human-review pathway — Missing | **[EXISTS — partially wired]** `run_pipeline.py:338-345`: rPPG feature vector absent → verdict `INCONCLUSIVE`, exit code 3; `contract.py::VERDICTS` includes INCONCLUSIVE; frontend renders it; evidence-quality machinery [EXISTS] (PQS tiers ≥0.7 / ≤0.3 in `RPPG/pqs.py`, SQI, blur/brightness floors per stage). Gaps: PQS-based inconclusive band not wired into the decision path; doc wording inconsistent; no explicit reviewer-workflow artifact | In scope — small wiring + doc fixes | Gate on PQS/SQI (validation-chosen q_val) → INCONCLUSIVE; align docs; keep REAL/FAKE classifier contract unchanged |
| 11 | Consistent evaluation vs deployment thresholds — Needs correction | **[ACCEPTED — inventory exists]** Calibrated: Youden-τ from validation probabilities persisted in checkpoint (smoke value 0.50676) supersedes the default 0.5 (`config.py:289-292`). Uncorrected inconsistencies: UI confidence bands hard-coded and uncalibrated (`frontend/src/lib.js:52-66`); stage tags `[N/3]` vs pipeline's `[N/4]` (`frontend/server.py:85` vs `run_pipeline.py`) → stage events dead; "29 features" / "24-feature vector" copy vs actual 20/39/59; AGENTS headline "two outcomes" vs executable INCONCLUSIVE | In scope — actionable small fixes | Fix stage tags + display copy; tie UI bands to the calibrated τ*; document the threshold scheme (τ*, PQS band, margin δ) once |
| 12 | Author contributions and order — Needs confirmation | **[HUMAN DECISION]** All git history is one committer identity (re-verified today: `NYN-05`, 152 commits, HEAD `bd12c16`) — git cannot substantiate per-person contributions; README documents a 3-person team with module split (Jhashank — rPPG→classifier + quantum; Sumit — input→ROI; Aswin — preprocessing/eval/integration); no CONTRIBUTORS/LICENSE files | Human | Team finalizes author list/order + CRediT statement; decide whether the shared git identity is the team (state in the paper) |
| 13 | ViT implementation (if claimed) | **[EXISTS as a verdict: NOT IMPLEMENTED]** Case-insensitive searches across the codebase find no ViT/transformer/attention code in any model path; visual backbone = ResNet50 (`visual/extractor.py`); the fused contract enumerates all 20+39 features by name — no transformer participates. "ViT" appears only in reference-paper titles | **Out of scope for the codebase** (§4.1); paper-claim correction required | Remove the ViT claim from the paper (recommended), or implement a genuine ViT branch as a documented contract change (contradicts the current supported-architecture rulebook) |
## 3. Detail notes (with evidence cites)

### 3.1 Full-dataset training (items 1–2)

- **Completed today [EXISTS]:** rPPG extraction over the whole corpus — `WORKING/output/rppg/checkpoint.json`: `last_completed_index 4092` (walked 4,093 clips), `processed_count 3914`, `no_features_count 179`, `failed_count 0`, config POS / min-usable-frames 24 / min-SQI 0.05 / blur ≥ 3.0 / brightness 15–245. Class distribution verified directly from `dataset_features.csv`: **2,087 REAL (label 0) vs 1,827 FAKE (label 1)** — both classes present, no regurgitation of the old FAKE-only table.
- **How to state the 3,914 vs 4,093 mismatch in the paper:** 4,093 distributed clips; 3,914 (95.6%) yield a labelled rPPG row; 179 (4.4%) fall below the minimum-evidence floor and are treated by design as *no physiological signal* (at inference these would exit `INCONCLUSIVE`, not fake a verdict). This is an abstention statistic, not lost data — and it is exactly the behavior the reviewer asks to formalize in item 10.
- **Remaining for "training on 4,093 clips":** stage 5/6/7. All machinery is fixed and validated as of today (smoke test 11/11: streaming extraction, checkpoint resume from 4/5, fail-fast exit 2, bit-exact feature parity vs the old path, fusion join integrity). The full rerun is a single authored command: `.\setup_and_run.ps1` (resume-safe; skips completed rPPG work; visual stage streams per video). Heavy compute was intentionally **not** auto-triggered in this response session.
- **Generalization metrics (item 2)** are emitted by the same rerun (stages [4/6] evaluate, [5/6] baselines) plus two provided repeatability scripts (`multi_seed_repeats.py`, bootstrap over clips) that currently cannot run without the regenerated artifacts. Known evaluator gap to fix in the same pass: add `eer()` (absent from `evaluation.py`); include PR-AUC/EER in per-seed summaries.

### 3.2 Leakage verification (item 3)

What the code guarantees today, audited and test-covered:

- Split: deterministic per-class grouped split (seed 42), each group wholly in one split, hard abort `_assert_no_group_leakage` if any group spans splits (`quantum/data.py`); leakage regression test `test_no_group_leakage` passes today (9/9 runnable quantum checks).
- Fitting isolation: z-score scaler train-only; shared visual PCA train-only; QAOA selection train-only — all consumed read-only at inference.
- Duplicates: duplicate `video_path` aborts table build; sha256→group conflicts abort if hash columns exist.

What is structurally unmeasurable on this corpus copy: the dataset folder contains **only MP4s in two flat folders** — no subject/identity/source metadata — so every group key degenerates to `clip:<path>`. Same-identity cross-split leakage (DFDC fakes share source identities) cannot be excluded without the official fake→source mapping or per-clip content hashes. Near-duplicate perceptual detection was never implemented.

Closing path (three parts, one of them external):
1. Re-run `paper_experiments/leakage_audit.py` on the current 3,914-row table (the stored result predates completion).
2. Extend `RPPG/extract_dataset_features.py` to emit `video_sha256`/`source_id` per row (loader already consumes these columns) — then re-run audit → grouping becomes hash/metadata-based.
3. Acquire the official DFDC metadata (fake→source JSON / part manifests) — author/external action; the repository cannot fabricate it.

### 3.3 Robustness experiments (item 4) — out of current execution scope

No degradation-experiment results exist anywhere in the tree [verified 2026-10-19 audit §4; unchanged]. What IS provided: `paper_experiments/robustness_degrade.py` — reproducible single-factor ffmpeg grid, one clip at a time, each condition isolated, then evaluated through the standard `run_pipeline.py` (so stage-1/2 acceptance rates per condition are captured — the conditions under which the system goes INCONCLUSIVE are themselves measurable). This item has deliberately **not been executed**: it is a consuming experiment on top of trained artifacts (blocked by item 1 anyway) and each run consumes author compute budget; per the project's ground rules, heavy runs require author confirmation. It is listed in §4.3 with the exact reason and is not part of the detection system itself.

### 3.4 Ablations (item 5)

- **What exists:** classifier-axis ablation evidence [EXISTS] — six classical models (RandomForest, MLP, LogisticRegression, calibrated LinearSVC, GaussianNB, XGBoost) trained and evaluated on the **same QAOA-selected features, same split, same grouped CV** as the VQC (`quantum/evaluation.py::run_baselines`). Recorded smoke figures exist (GNB/XGB 0.625 etc., §1) — preliminary label applies.
- **What was retired:** the rPPG-only/visual-only feature-set machinery (`PHASE6_FEATURE_SETS = {}`, sweep runner deleted 2026-10-11). This is a deliberate rulebook constraint (fused-only inference; retired feature sets must not be reintroduced) — not an oversight.
- **Compliant path:** build the ablation evaluator in `paper_experiments/` that reads the regenerated `data_fused.npz` and slices feature groups by the canonical name lists (`RPPG_FEATURE_NAMES`, `VISUAL_FEATURE_NAMES`). This evaluates feature-axis ablations **after** the fused contract is built, without modifying the supported fused-only architecture. Matrix: {rPPG-only, visual-only, fused} × {QAOA top-3, all pre-selected 18, all 59} × {VQC, best classical}. Known baseline deviation to fix first: XGBoost early stopping monitors the training set (`eval_set=[(X_train, y_train)]`) — switch to a validation slice before the publication run.

### 3.5 Benchmarks (item 6)

Instrumentation exists in the pipeline (per-stage wall clock, `memory_snapshot()` RSS/CUDA, artifact checksums, 30-min server job cap). Provided protocol: `paper_experiments/latency_benchmark.py` (warm-up + ≥5 repeats per stage, peak RSS/VRAM capture, artifact disk sizes, throughput). Computed so far [COMPUTED 2026-10-19]: VQC trainable parameters **68** (27 quantum + 41 classical) at 3 wires / **323** at 18 inputs; MLP baseline 673; LogisticRegression 4. Also essential and often mis-stated: the quantum components are **simulators** — exact complex128 statevector, analytic expectation values, **no shot sampling** — the paper must say "simulator" explicitly. Actual runs are blocked on regenerated artifacts (item 1) and one authorized benchmark session.

### 3.6 Fairness (item 7) — out of scope for this cycle

The corpus on disk carries no demographic attributes (verified twice: recursive scans find only MP4s), and no fairness/per-group evaluation code exists. There is nothing to stratify by today — any subgroup claim would be fabricated. Fairness analysis requires (a) official DFDC metadata re-acquisition (some shipping metadata may include attributes — unverified), or (b) an annotation protocol (identity-level perceived skin tone/gender/age with inter-annotator agreement). Both are data-governance decisions outside the codebase's current mandate; the code change itself (per-group metrics reporter reusing the grouped-split machinery) is small **if** the data arrives. §4.4 documents this boundary.

### 3.7 Provenance, license, manifest (item 8)

Verified on disk (2026-10-21): `DFDC_Dataset/Real/` = `DFDC_Dataset/Real/` = 2,127 MP4s (248 zero-padded numeric-stem clips such as `00000.mp4`-`00247.mp4` appear in the label table; the remaining Real rows use alphanumeric stems); `DFDC_Dataset/Fake/` = 1,966 MP4s with alphanumeric stems; **zero JSON/CSV/TXT metadata files anywhere** in the tree. The rPPG CSV's only provenance column is `source` = folder name. No `LICENSE` file exists in the repo; no document records where this 4,093-clip corpus was acquired (the Kaggle-style `archive/DFDC_Dataset` naming is an inference, not a record). Therefore the mapping from each fake to its source real video — the one fact that both leakage verification (item 3) and any official-manifest statement (this item) require — is unrecoverable from the local copy. Recovery requires the official DFDC distribution metadata; the repository can then (and only then) populate `source_id`/`video_sha256` at extraction and republish the manifest with the paper. Licence terms and acquisition records are author-verification items by nature — they cannot be generated from this repository.
### 3.8 Ethics / privacy documentation (item 9)

The repository side of this story is factual and small: only public DFDC prototype clips (someone else's dataset, standard research provenance) are processed in the offline pipeline; there is no live KYC subject, no personal-data collection, no user upload retained long-term — `frontend/server.py` demonstrably enforces upload size/signature limits, path-traversal protection, and expiry (job records 1 h, frame sequences 24 h); rPPG features are aggregate physiological statistics, not identity artifacts. The **ethics/consent/data-processing document itself is a paper deliverable** (a [HUMAN DECISION]): it must state the dataset's own consent/licence position (item 8) and the system's intended KYC deployment boundary. The repository cannot author consent facts it does not possess; it supplies the data-processing inventory above.

### 3.9 Abstention / human-review pathway (item 10)

Contrary to "missing", the mechanism [EXISTS] in the executable contract and just needs finishing + doc alignment:

- `WORKING/run_pipeline.py:338-345` — when rPPG yields no usable feature vector, the result is `verdict: INCONCLUSIVE` and the process exits **code 3**; `frontend/contract.py::VERDICTS = {"REAL","FAKE","INCONCLUSIVE"}`; the frontend has an Inconclusive rendering block.
- The evidence-quality machinery that should *drive* abstention already exists: per-frame quality floors at stage 1 (blur/brightness/face area), rPPG-side gates (min usable frames 24, SQI ≥ 0.05, ROI validity ≥ 30%), and the **PQS quality tiers** (`RPPG/pqs.py`: HIGH ≥ 0.7, LOW ≤ 0.3).
- Gap 1: the decision path currently cuts hard at the calibrated threshold; the PQS/SQI-based inconclusive band (validation-chosen `q_val`) is specified but not wired in.
- Gap 2: documentation inconsistency — AGENTS.md/README describe only REAL/FAKE while the executable contract and UI retain INCONCLUSIVE.
- Gap 3: an explicit "send to human review" workflow artifact (e.g., a queue/export of inconclusive cases) does not exist beyond the frontend's passthrough state.
Fixing all three is a small, contract-preserving change (INCONCLUSIVE stays a pipeline state; the classifier remains two-outcome) and is listed in §5.

### 3.10 Threshold consistency (item 11)

Full inventory assembled in `publication_readiness_report.md` §9 (stage-1 quality gates, rPPG gates, decision threshold, display bands). Calibration is **correct in the training/eval layer**: the persisted checkpoint threshold is Youden-validated on validation probabilities (smoke value 0.50676) and supersedes the `DecisionConfig` default 0.5 (`quantum/config.py:289-292`) at inference. Accepted inconsistencies to correct:

| # | Inconsistency | Where | Fix |
|---|---|---|---|
| T1 | UI confidence bands ("High/Moderate", probability words 0.75/0.55/0.45/0.25) are hard-coded, uncalibrated display artifacts | `frontend/src/lib.js:52-66` | Derive bands from the calibrated τ* (and the planned margin δ) or label them "indicative" |
| T2 | Web layer keys on `[N/3]` stage tags; pipeline prints `[N/4]` → stage SSE events never fire | `frontend/server.py:85`, `src/lib.js:48-50` vs `run_pipeline.py` | Align tag counts |
| T3 | UI copy says "Measuring **29** physiological features"; a docstring says "24-feature vector"; contract is 20/39/59 | `src/lib.js:43`, `run_pipeline.py:14` | Correct copy |
| T4 | AGENTS headline "two decision outcomes" vs executable INCONCLUSIVE contract | `AGENTS.md`, `README.md` | One reconciled wording (see 3.9) |
| T5 | Evaluation threshold (calibrated τ*) vs deployment display semantics (uncalibrated bands) — the drift the reviewer is pointing at | — | Single documented threshold scheme: τ* for the verdict, optional PQS evidence band + margin δ for INCONCLUSIVE, display derived from it |

These are small code/copy fixes; they are recorded here as the correction list and deliberately **not applied unilaterally** in this response (they touch user-facing behavior; the maintainer schedules them with the rerun).

### 3.11 Author contributions (item 12)

Git reality: **one committer identity** — re-verified today: `NYN-05 <jnyn2005@gmail.com>`, **152 commits**, HEAD `bd12c16`. The README documents a three-person team with an explicit module split (Jhashank — rPPG→classifier + quantum; Sumit — input→ROI; Aswin — preprocessing/eval/integration), but git cannot confirm or refute per-person authorship (bulk points under one identity + the documented module reorganisations). There is no CONTRIBUTORS/LICENSE file. Final author list, author order, and a CRediT contribution statement are **[HUMAN DECISION]** items that the team must confirm — the report recommends deciding explicitly whether the shared git identity represents the team and stating that in the paper.

### 3.12 ViT (item 13)

Verified again on 2026-10-21 (case-insensitive sweep of the codebase): no ViT, no vision-transformer, no attention module exists in any model path; the visual backbone is **ResNet50** (`visual/extractor.py`), and the fused feature contract enumerates all 20 rPPG + 39 visual features by name (`quantum/config.py::FUSED_FEATURE_NAMES`) — a transformer cannot hide inside it. "ViT" appears in this repository only as words inside reference-paper filenames. **Answer to "if the paper claims to use ViT": it must not.** Two options exist for the paper: (a) remove the ViT claim (recommended — the honest, evidence-matched fix), or (b) implement a genuine ViT branch, which would be a large documented architecture change running against the current supported-architecture rulebook (fused-only ResNet50+handcrafted; no silent contract extensions). Under the present project constraints, (b) is **out of scope of this codebase cycle**.

## 4. Out-of-scope register (with justification)

| § | Item | Why out of scope here | What would pull it in |
|---|---|---|---|
| 4.1 | Implementing a ViT backbone | Contradicts the supported-architecture rulebook (fused-only; visual stage = ResNet50 + handcrafted descriptors; contract is name-frozen and test-locked). No ViT claim should remain in the paper | A deliberate, documented architecture change ordered by the team |
| 4.2 | Official DFDC provenance, licence terms, acquisition record | Facts about data acquisition and legal terms; not producible from repository contents | Team re-obtains official metadata/licence documentation and records provenance |
| 4.3 | Executing the robustness grid (item 4) | A consuming experiment requiring author compute authorization; blocked on trained full-dataset artifacts anyway; protocol is complete and ready | Author go-ahead after the rerun → `paper_experiments/robustness_degrade.py` |
| 4.4 | Fairness/demographic analysis (item 7) | No demographic metadata exists on disk; per-group metrics would be fabricated inputs | Official metadata (if it ships attributes) or an agreed annotation protocol |
| 4.5 | Author order / CRediT statement (item 12) | Attribution decision reserved to the team; git identity is shared, not per-person | Team confirmation |
| 4.6 | Ethics/consent document (item 9) | Consent/legal narrative for the paper; repository can only supply the data-processing inventory | Team-authored paper section grounded in §3.8 |

Everything else in the reviewer's list is **in scope** and either (a) already implemented, (b) closable by the one full-dataset rerun, or (c) closable by the small enumerated fixes in §3.10/§3.11/§3.9.

## 5. What the next full rerun closes (single authored command)

`.\setup_and_run.ps1` (from the repo root; stages 1–4 resume-complete instantly, stage 5+ run):

| Closes | Because |
|---|---|
| Item 1 (training on 4,093-clip corpus minus the 179 abstaining clips) | stage 5 streams/regenerates the visual table + fusion over all 3,914 labelled clips |
| Item 2 (final generalization metrics) | stages 6–7 produce `metrics_quantum.json`, `metrics_baselines.json`, training log, plots — replacing all deleted smoke artifacts |
| Item 5 (ablations, partially) | the regenerated `data_fused.npz` unlocks the paper-local ablation evaluator |
| Item 6 (benchmarks) | regenerated artifacts unlock `latency_benchmark.py` + persisted model sizes |
| Item 3 (partially) | leakage audit re-runnable on the full table; structural metadata fix is the extraction-column change noted in §3.2 |
| §1 labeling obligation | new artifacts fully replace the "preliminary feasibility" numbers, which are then retired to that label permanently |

Subsequent authorized sessions: multi-seed/CI pass; robustness grid; latency benchmark; ablation matrix.

## 6. Summary of corrections owed to the paper (accepting every reviewer finding)

1. Label the 38-clip smoke results (VQC 0.500; GNB/XGB 0.625; n = 8) **"preliminary feasibility ONLY"** — or replace them with regenerated full-dataset tables after the rerun (preferred).
2. Remove any ViT claim.
3. State the quantum components are simulators (exact statevector, analytic, no shots).
4. Reconcile the threshold/deployment story with the calibrated τ* + planned evidence-band abstention (fix T1–T5).
5. State dataset provenance honestly (flattened local copy, 4,093 clips walked, 3,914 labelled, no official metadata locally) and only then proceed to license/acquisition confirmation.
6. Team decisions required: author list/order + CRediT; ethics/data-processing section; robustness-grid and benchmark authorization.

*Response ends. Every number in this document is either directly re-verified on 2026-10-21 (checkpoint.json, CSV counts, DFDC folder scan, git log, code grep) or cites its earlier audit's source; nothing is asserted from memory of a deleted artifact beyond the explicit "preliminary feasibility" label.*
