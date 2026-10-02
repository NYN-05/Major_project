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
| 2026-10-01 18:00 IST | P1 evaluation and failure handling | Added explicit rPPG outcome states, no-face/insufficient-frame diagnostics, finite-probability validation, validation-only threshold selection, repeated-seed summaries, QAOA selection-stability utilities, reproducibility metadata checksums, and replaced placeholder edge-case tests with assertions. | In progress |
| 2026-10-01 17:35 IST | P0 remediation | Removed physiological fallback constants, rejected non-finite rPPG vectors, added shared preprocessing/schema validation, expanded evaluation metrics, and added checksum/schema safeguards for serialized artifacts. | In progress |
| 2026-10-01 18:15 IST | rPPG | Fixed FileNotFoundError in extract_dataset_features.py by creating output directory before writing CSV; parent directory was missing on first run. | Complete |
| 2026-10-01 18:30 IST | P2 hardening | Added bounded visual inference batches with explicit tensor cleanup and centralized device resolution; added cache/provenance metadata, versioned API payload validation, rate limiting, safe job errors, failure cleanup, pipeline diagnostics, and exact dependency pins. | Complete |
| 2026-10-01 18:45 IST | rPPG/Environment | Worked around Windows Defender Controlled Folder Access blocking writes to Desktop by moving dataset output to `C:\Users\JHASHANK\Maj_Proj_output`; added robust `os.makedirs` with `exist_ok=True` in extract_dataset_features.py. | Complete |
| 2026-10-01 19:35 IST | GPU environment | Upgraded ONNX Runtime GPU support to `1.20.2` for PyTorch CUDA 12.1/cuDNN 9 compatibility, added real YuNet CUDA preflight validation, and made GPU dataset runs fail explicitly instead of silently falling back to CPU. | Complete |
| 2026-10-01 22:52 IST | rPPG dataset generation | Added real-time per-sample checkpoint/resume to `extract_dataset_features.py`: atomic checkpoint and CSV writes after every sample, `--start`/`--end` range control, `--resume` continuation from checkpoint, SIGINT/SIGTERM pause that finishes the current sample cleanly, and duplicate prevention via `completed_indices`. Exposed `-UseGpu`/`-Resume`/`-Start`/`-End` in `Scrape/generate_dataset.ps1`. Added `Scrape/test_checkpoint_resume.py` covering save/load, resume-skip, range filtering, pause, atomic CSV writes, and full workflow recovery. | Complete |
| 2026-10-01 23:15 IST | rPPG dataset generation | Fixed checkpoint crashes and resume integrity: 4-tuple sample items for `CheckpointManager`/workers, `mark_paused()` no longer inflates `processed_count`, fresh runs delete checkpoint with CSV, `--resume` aborts on missing/invalid checkpoint or missing CSV, resume skips completed indices instead of unsafe `last_completed+1`, hardened `_find_item_index` (full/root-relative path only), workers ignore SIGINT with second-Ctrl+C forced exit. | Complete |

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
