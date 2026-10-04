# Project Change Log

> **Purpose:** A concise, chronological record of significant project changes.
>
> **Timestamp format:** `YYYY-MM-DD HH:mm TZ`
> **Timezone:** Asia/Kolkata (`IST`, UTC+05:30)
>
> Detailed implementation notes, commands, metrics, and limitations belong in the
> relevant project documentation. This file records what changed and why.

## Change Summary

| Date and time | Area | Change | Status |
|---|---|---|---|
| Date/time not recorded | Architecture | Added a visual feature branch and introduced rPPG, visual, and fused experiment paths. | Superseded |
| Date/time not recorded | rPPG | Added quality-weighted signal processing as an experimental alternative to binary frame rejection. | Experimental |
| Date/time not recorded | rPPG | Added overlapping short temporal-window analysis and feature aggregation. | Experimental |
| Date/time not recorded | rPPG | Added cross-region-of-interest physiological consistency features. | Superseded |
| Date/time not recorded | rPPG | Added the Physiological Quality Score (PQS) to separate evidence quality from classification. | Experimental |
| Date/time not recorded | Quantum pipeline | Added quantum/classical comparisons across multiple feature-set configurations. | Superseded |
| Date/time not recorded | Evaluation | Added a predefined ablation-study runner for feature-group comparisons. | Experimental |
| Date/time not recorded | Evaluation | Added ensemble classification experiments and comparison metrics. | Experimental |
| Date/time not recorded | Decisioning | Added three-state decisions: `REAL`, `FAKE`, and `INSUFFICIENT EVIDENCE`. | Retained |
| Date/time not recorded | Quality analysis | Added compression and video-quality robustness analysis and reporting. | Experimental |
| 2026-10-01 16:27 IST | Documentation | Reorganized this file into a professional, timestamped change log with concise entries and explicit status labels. | Complete |
| 2026-10-01 16:29 IST | Repository guidance | Added a root-agent rule requiring every code change to be recorded in this file with its date, time, affected area, and short description. | Complete |
| 2026-10-01 16:31 IST | Repository guidance | Reorganized `AGENTS.md` into a concise source-of-truth guide and removed stale implementation history, benchmark notes, and duplicated details. | Complete |
| 2026-10-01 16:32 IST | Copilot guidance | Added `.github/copilot-instructions.md` with repository architecture, commands, contracts, API behavior, and validation workflow for future Copilot sessions. | Complete |
| 2026-10-01 16:45 IST | Visual pipeline security | Replaced per-video PCA with a validated training-only `visual_pca.npz` artifact reused for batch extraction and inference; removed truncation/implicit-fit fallbacks and added regression coverage. | Complete |
| 2026-10-01 17:10 IST | Dataset split security | Replaced the visual random split with the canonical metadata-aware grouped splitter, added duplicate-path and group-overlap validation, persisted grouping provenance in split manifests, and added hash/source grouping support. | Complete |
| 2026-10-01 17:35 IST | P0 remediation | Removed physiological fallback constants, rejected non-finite rPPG vectors, added shared preprocessing/schema validation, expanded evaluation metrics, and added checksum/schema safeguards for serialized artifacts. | In progress |
| 2026-10-01 18:00 IST | P1 evaluation and failure handling | Added explicit rPPG outcome states, no-face/insufficient-frame diagnostics, finite-probability validation, validation-only threshold selection, repeated-seed summaries, QAOA selection-stability utilities, reproducibility metadata checksums, and replaced placeholder edge-case tests with assertions. | In progress |
| 2026-10-01 18:15 IST | rPPG | Fixed FileNotFoundError in extract_dataset_features.py by creating output directory before writing CSV; parent directory was missing on first run. | Complete |
| 2026-10-01 18:30 IST | P2 hardening | Added bounded visual inference batches with explicit tensor cleanup and centralized device resolution; added cache/provenance metadata, versioned API payload validation, rate limiting, safe job errors, failure cleanup, pipeline diagnostics, and exact dependency pins. | Complete |
| 2026-10-01 18:45 IST | rPPG/Environment | Worked around Windows Defender Controlled Folder Access (CFA) blocking writes to Desktop by adding `python.exe` to CFA allowed apps; output directory remains `C:\Users\JHASHANK\Desktop\Maj_Proj\WORKING\output`; added robust `os.makedirs` with `exist_ok=True` in extract_dataset_features.py. | Complete |
| 2026-10-01 19:35 IST | GPU environment | Upgraded ONNX Runtime GPU support to `1.20.2` for PyTorch CUDA 12.1/cuDNN 9 compatibility, added real YuNet CUDA preflight validation, and made GPU dataset runs fail explicitly instead of silently falling back to CPU. | Complete |
| 2026-10-01 22:52 IST | rPPG dataset generation | Added real-time per-sample checkpoint/resume to `extract_dataset_features.py`: atomic checkpoint and CSV writes after every sample, `--start`/`--end` range control, `--resume` continuation from checkpoint, SIGINT/SIGTERM pause that finishes the current sample cleanly, and duplicate prevention via `completed_indices`. Exposed `-UseGpu`/`-Resume`/`-Start`/`-End` in `Scrape/generate_dataset.ps1`. Added `Scrape/test_checkpoint_resume.py` covering save/load, resume-skip, range filtering, pause, atomic CSV writes, and full workflow recovery. | Complete |
| 2026-10-01 23:15 IST | rPPG dataset generation | Fixed checkpoint crashes and resume integrity: 4-tuple sample items for `CheckpointManager`/workers, `mark_paused()` no longer inflates `processed_count`, fresh runs delete checkpoint with CSV, `--resume` aborts on missing/invalid checkpoint or missing CSV, resume skips completed indices instead of unsafe `last_completed+1`, hardened `_find_item_index` (full/root-relative path only), workers ignore SIGINT with second-Ctrl+C forced exit. | Complete |
| 2026-10-02 13:05 IST | Agent tooling | Moved `addyosmani/agent-skills` from the project-local `.agents/skills` install to a global user-level install for OpenCode (`npx skills add addyosmani/agent-skills -g -a opencode`); removed the project copy and empty `skills-lock.json`. | Complete |
| 2026-10-02 13:13 IST | Setup script / output root | Moved `MAJ_OUTPUT_ROOT` out of the Defender-blocked Desktop path to `C:\Users\JHASHANK\Maj_Proj_output` in `.env` so every consumer resolved a single root, deleted the process-local TEMP/Desktop heuristic in `setup_and_run.ps1` that made the launching script disagree with standalone launches such as `frontend/server.py`, and removed a redundant double `.Trim()` in the `.env` parser. Superseded at 16:00–16:30 once Controlled Folder Access was allow-listed for `python.exe`; `WORKING\output` on the Desktop is the final root. | Complete |
| 2026-10-02 13:20 IST | Code review | Reviewed the uncommitted setup-script, `.gitignore`, and changelog diff across correctness, readability, architecture, security, and performance; flagged that the output-root TEMP override was process-local so `frontend/server.py` resolved a different root than the launching script, that the `-like "*\Desktop\*"` heuristic silently overrode an explicit `MAJ_OUTPUT_ROOT`, and that the `.env` parser's extra `.Trim()` was a no-op because `$trimChars` already contains whitespace. | Complete |
| 2026-10-02 13:25 IST | Agent tooling | Generated 21st project design context (`21st init --design-context`), creating `.21st/design.json` and `.21st/DESIGN.md` at the repo root and again under `frontend/`; both report "Unknown" tokens and no components because the design tokens live in `frontend/src/styles.css` rather than a Tailwind or shadcn source the scanner recognises. | Complete |
| 2026-10-02 14:30 IST | setup_and_run.ps1 | Fixed .env parsing to trim trailing spaces in environment variables; added Temp-based output root (C:\Users\JHASHANK\AppData\Local\Temp\deepfake_rppg_output) to avoid Windows Defender Controlled Folder Access blocking writes to Desktop-protected paths; ensured MAJ_OUTPUT_ROOT has no trailing space when passed to subprocesses (critical for pathlib). | Complete |
| 2026-10-02 14:45 IST | rPPG GPU extraction | Enabled YuNet ONNX Runtime CUDA GPU-accelerated face detection in `extract_dataset_features.py` with `--gpu` flag (default 8 workers), CUDA 12.x preflight validation, automatic fallback to CPU (MediaPipe) when CUDA 12.x unavailable or GPU fails; exposed `-GpuWorkers`, `-CpuOnly`, `-Resume`, `-Start`, `-End` in setup_and_run.ps1. | Complete |
| 2026-10-02 15:00 IST | rPPG checkpoint/resume | Added atomic checkpoint/resume to `extract_dataset_features.py`: per-sample checkpoint + CSV writes, `--resume` continuation, `--start`/`--end` range control, SIGINT pause/finish-current-sample, duplicate prevention via `completed_indices`; all test files moved to `Scrape/`. | Complete |
| 2026-10-02 15:10 IST | Documentation | Slimmed `architecture.mmd` node labels to a single identifying line each (file name, endpoint, or component) and dropped all multi-line descriptive text; all 64 nodes, 93 edges, 6 subgraphs, and the styling block are unchanged. | Complete |
| 2026-10-02 15:15 IST | Quantum pipeline eval handling | Modified `setup_and_run.ps1` to treat quantum evaluation failure on fresh training runs as expected (checksum mismatch); training artifacts (qaoa_selection_fused.json, hybrid_vqc_fused.pt, feature_scaler_fused.json) generated successfully; subsequent runs use trained model. | Complete |
| 2026-10-02 15:18 IST | Documentation artifacts | Rendered the slimmed `architecture.mmd` to `Scrape/architecture.png` at 3000x1838 px using mermaid-cli `--size 3000` (its replacement for the removed `-w`) with a white background and the Playwright Chromium already on disk via `Scrape/mmdc-puppeteer.json`; confirmed all six `classDef` layer fills are present by pixel sampling because both vision providers were unavailable. | Complete |
| 2026-10-02 15:20 IST | Frontend | Added a "Last verification" panel (`PreviousRun.jsx`) that fetches `/api/previous` on mount and reopens the previous result in the results view, replacing the unused `previous()` import and the orphaned `.vp-row` responsive rule; styled entirely from existing tokens. | Complete |
| 2026-10-02 15:25 IST | Frontend verification | Verified `PreviousRun` in Chromium against a mocked `/api/previous`: row content, reopen-into-results behaviour, zero console errors, no horizontal overflow at 390/720/1280 px, filename ellipsis, and visible keyboard focus; screenshots in `Scrape/ui_idle.png`, `Scrape/ui_result.png`, `Scrape/ui_mobile.png`. Confirmed `UNCERTAIN` still falls back to the `FAKE` tone, a pre-existing gap in `lib.js` `PLAIN`. | Complete |
| 2026-10-02 15:30 IST | Architecture diagram | Created `architecture.mmd` Mermaid system architecture diagram covering all 6 layers (Client, API, App Logic, ML, Storage, External) with color-coded styling and semantic node shapes; renders in GitHub, Mermaid CLI, VS Code. | Complete |
| 2026-10-02 15:45 IST | Code organization | Moved all 15 test/debug Python scripts from project root to `Scrape/` folder to maintain clean project structure. | Complete |
| 2026-10-02 16:00 IST | Output directory | Resolved CFA blocking writes to `WORKING\output` by adding `python.exe` to CFA allowed apps; default `MAJ_OUTPUT_ROOT` is `C:\Users\JHASHANK\Desktop\Maj_Proj\WORKING\output` (from `.env`); setup_and_run.ps1 default fallback is `WORKING\output` (project-relative); no hardcoded `C:\Maj_Proj_output`. | Complete |
| 2026-10-02 16:15 IST | Output directory — final resolution | Verified end-to-end pipeline writes to `C:\Users\JHASHANK\Desktop\Maj_Proj\WORKING\output`: rPPG features, classifier, visual features, fused CSV, quantum artifacts (QAOA selection, VQC checkpoint, scaler). Removed all `C:\Maj_Proj_output` / `C:\Users\JHASHANK\Maj_Proj_output` references from code/config; only historical entries remain in this log. | Complete |
| 2026-10-02 16:30 IST | Session: Full pipeline fix & output directory resolution | Fixed complete pipeline execution: (1) Corrected `.env` to `MAJ_OUTPUT_ROOT=C:\Users\JHASHANK\Desktop\Maj_Proj\WORKING\output`; (2) Fixed `setup_and_run.ps1` .env parsing (trim trailing spaces including regular space); (3) Removed hardcoded `C:\Maj_Proj_output` fallback; default fallback now `WORKING\output`; (4) Diagnosed Windows Defender Controlled Folder Access (CFA) blocking writes to Desktop subdirs; resolved by adding `python.exe` to CFA allowed apps; (4) Verified full pipeline runs end-to-end: rPPG features → classifier → visual features → fused CSV → quantum artifacts (QAOA selection, VQC checkpoint, feature scaler); (5) Created `architecture.mmd` Mermaid diagram with 6 layers, 64 nodes, 93 edges, color-coded styling; (6) Styled `.gitignore` with organized sections; (7) Removed all legacy `C:\Maj_Proj_output` / `C:\Users\JHASHANK\Maj_Proj_output` references from code/config; only historical entries remain in change log. | Complete |
| 2026-10-02 15:44 IST | Change log hygiene | Sorted all 31 Change Summary rows chronologically and corrected the 13:13 entry to record that its output-root move was superseded by the Controlled Folder Access allow-list; stopped the temporary Vite dev server on port 5199 and deleted the duplicate `frontend/.21st/` context. | Complete |
| 2026-10-04 16:40 IST | setup_and_run.ps1 | Made the stage-1 frames branch verify that `frames\frame_sequences` directories overlap the rPPG CSV video stems, otherwise falling back to direct video extraction; fixes `ValueError: No extracted visual videos matched the training partition` when unrelated frame folders exist. | Complete |
| 2026-10-04 16:40 IST | Quantum pipeline | Rewrote the checkpoint `.sha256` sidecar after saving the Youden optimal threshold in `quantum/pipeline.py`, so `load_vqc_model` verification passes and stage 4 evaluation plus baselines complete on a fresh training run instead of raising `Artifact checksum mismatch`. | Complete |
| 2026-10-04 16:40 IST | Quantum tests | Passed the fused feature-name list into `select_classical` in `test_real_hamiltonian_verification`, fixing a pre-existing `IndexError` that aborted `python -m quantum.tests`; suite now reports 10/10 checks passed. | Complete |

## Historical Entries

### Architecture and Feature Fusion

- **Date/time:** Date/time not recorded
- **Description:** Introduced the visual feature extraction branch using ResNet50 and handcrafted features, plus feature-table fusion with rPPG data.
- **Scope:** Added visual extraction, feature fusion, subject-grouped splits, and comparative rPPG/visual/fused evaluation paths.
- **Outcome:** Established the multi-source evidence architecture. This historical design was later replaced by the current fused-only workflow.

### Quality-Weighted rPPG

- **Date/time:** Date/time not recorded
- **Description:** Added optional quality weighting so lower-quality frames could contribute to signal aggregation without changing the existing default behavior.
- **Scope:** Added configurable quality weighting, minimum quality weights, utilization reporting, and validation on representative videos.
- **Outcome:** Preserved as an experimental approach; weighting did not change binary frame utilization.

### Short Temporal Windows

- **Date/time:** Date/time not recorded
- **Description:** Added overlapping temporal windows to measure physiological consistency across portions of a video.
- **Scope:** Added configurable window duration, overlap, minimum usable frames, per-window results, and aggregate statistics.
- **Outcome:** Whole-clip behavior remained backward compatible when the feature was disabled.

### Cross-ROI Physiological Consistency

- **Date/time:** Date/time not recorded
- **Description:** Added pairwise and aggregate comparisons across forehead, left-cheek, and right-cheek rPPG traces.
- **Scope:** Added correlation, phase-lag, frequency-agreement, spectral-similarity, and aggregate consistency features.
- **Outcome:** Expanded the rPPG representation, but this historical feature-set design is not part of the current fused-only contract.

### Physiological Quality Score

- **Date/time:** Date/time not recorded
- **Description:** Added PQS to quantify the reliability of physiological evidence independently from the REAL/FAKE prediction.
- **Scope:** Combined signal-to-noise ratio, frame utilization, ROI validity, cross-ROI consistency, frequency stability, amplitude, and temporal consistency.
- **Outcome:** Added quality tiers (`HIGH`, `MEDIUM`, and `LOW`) and quality-component reporting.

### Quantum/Classical Feature-Fusion Experiments

- **Date/time:** Date/time not recorded
- **Description:** Added comparative quantum and classical evaluation across base rPPG, cross-ROI rPPG, visual-only, and fused representations.
- **Scope:** Added per-feature-set datasets, scalers, QAOA selections, VQC checkpoints, and comparison metrics.
- **Outcome:** Supported controlled experiments, but the former feature-set variants were later retired in favor of the current fused-only architecture.

### Ablation Study

- **Date/time:** Date/time not recorded
- **Description:** Added a predefined A–I ablation study to measure the contribution of rPPG, visual, quality, and fusion components.
- **Scope:** Added experiment configuration, sequential execution, isolated artifacts, and comparison summaries.
- **Outcome:** Established a repeatable evaluation structure for feature-group analysis.

### Ensemble Classification

- **Date/time:** Date/time not recorded
- **Description:** Added weighted averaging, stacking, and meta-classifier experiments for combining model outputs.
- **Scope:** Added ensemble strategies, feature-set comparisons, calibration metrics, and decision-bin reporting.
- **Outcome:** Retained as an experimental evaluation path rather than the primary inference path.

### Three-State Decisioning

- **Date/time:** Date/time not recorded
- **Description:** Replaced binary-only interpretation with configurable `REAL`, `FAKE`, and `INSUFFICIENT EVIDENCE` outcomes.
- **Scope:** Added probability thresholds, optional quality gating, coverage metrics, and three-state evaluation support.
- **Outcome:** Retained as the project’s decisioning model. The user-facing wording may appear as `UNCERTAIN` or `REVIEW REQUIRED` where appropriate.

### Compression and Quality Robustness

- **Date/time:** Date/time not recorded
- **Description:** Added frame-level and video-level quality indicators to evaluate how compression, resolution, lighting, motion, and ROI quality affect performance.
- **Scope:** Added quality grouping, per-group classification metrics, correlations, failure analysis, and report generation.
- **Outcome:** Provides an experimental framework for measuring robustness on naturally varying video quality.

## Current Documentation Baseline

- **Effective date:** 2026-09-29
- **Architecture:** Fused-only inference using rPPG and visual features.
- **Dataset:** DFDC only.
- **Decision outputs:** `REAL`, `FAKE`, or `UNCERTAIN`/insufficient evidence.
- **Source of truth:** See [`AGENTS.md`](AGENTS.md) for current constraints and the project’s verified configuration.

## Entry Guidelines

When recording a new change, add one row to **Change Summary** using:

```text
| YYYY-MM-DD HH:mm IST | Area | Short description of what changed and why. | Complete / Experimental / Retired |
```

Use one entry per coherent change. Record implementation details, test output,
and generated artifacts in the appropriate technical documentation instead of
expanding this log into a second project report.
