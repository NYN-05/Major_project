# AGENTS.md

Deepfake-video detection for KYC using rPPG (physiological evidence) + hybrid quantum-classical ML (PennyLane + PyTorch). Low-resolution input, decision bins REAL / FAKE / UNCERTAIN.

## Layout

All active code lives under `WORKING/`; the repo root holds docs/PDFs and README. Single git repo (no nested repos). `FF++/` is the gitignored dataset.

```
WORKING/
  frame/    stage 1: frame sampling, YOLO face detection, quality filtering (has its own app/ + requirements.txt)
  RPPG/     stage 2: MediaPipe face ROIs -> POS/CHROM pulse -> 8 physiological features (+ rppg-pipeline/, requirements.txt)
  quantum/  stage 3: QAOA feature selection -> hybrid VQC -> P(real) -> verdict (run via `python -m quantum.pipeline`)
  run_pipeline.py  end-to-end orchestrator: frames -> rPPG -> quantum -> verdict
```

## Commands (all from `WORKING/`)

- `python -m quantum.pipeline --all` — full quantum flow: build real dataset, QAOA selection (8→6), train VQC, evaluate, baselines. Regenerates `quantum/output/*` (`data.npz`, `qaoa_selection.json`, `hybrid_vqc.pt`, metrics, plots).
- `python run_pipeline.py --source path/video.mp4 [--method POS|CHROM] [--out result.json]` — end-to-end inference. Requires the quantum artifacts above.
- Rebuild rPPG training data: `python rppg-pipeline/extract_dataset_features.py` from `WORKING/RPPG/` (writes `dataset_features.csv`), then rerun `python -m quantum.pipeline --all`.
- Standalone frame stage: `python app/main.py --source test.mp4 --save-metadata` from `WORKING/frame/`.

The `quantum.*` imports and the `sys.path` insertions in `run_pipeline.py` assume the working directory is `WORKING/`. Do not run from the repo root.

## Verified constraints (do not break)

- **No synthetic data.** The quantum layer consumes only the real rPPG feature table `RPPG/dataset_features.csv` (currently 18 labeled rows). Never reintroduce a generator or a transform/bridge layer.
- **Feature contract:** `FEATURE_NAMES` in `quantum/config.py` must stay identical in name AND order to `RPPGFeatures.feature_names()` in `RPPG/rppg/features.py` (duplicated on purpose; `data.py`, `qaoa.py`, and `pipeline.py` all index by it). Keep the two lists in sync.
- **Label conventions differ per stage — do not unify:**
  - rPPG CSV: `1 = fake, 0 = real`
  - quantum: `LABEL_REAL = 1`, `LABEL_FAKE = 0`; `quantum/data.py` flips (`y = 1 - csv_label`)
  - rPPG RandomForest cross-check in `run_pipeline.py`: `1 = DEEPFAKE`
- **Gitignored artifacts:** `*.csv`, `*.json`, `*.pkl`, `*.mp4`, and all `output/` dirs are untracked — `dataset_features.csv`, `quantum/output/*`, and the trained models will not appear in `git status`. Regenerating them is normal.
- **rPPG returns `features=None`** when usable frames < `min_usable_frames` (48); `run_pipeline.py` then emits INCONCLUSIVE and exits 3. New code must handle `None`.
- **Frame stage does not feed rPPG.** `frame/` and `RPPG/` each open the video independently (YOLO quality stats vs. MediaPipe ROIs); frame outputs serve reporting only. Pre-existing design — do not refactor without approval.
- RandomForest cross-check is an optional side path; the final verdict comes exclusively from the quantum stage.
- Small training set (10 train rows) makes QAOA mutual-information weights degenerate (`"success": false`, mostly-zero weights in `qaoa_selection.json`). Expected — data-quantity issue, not a code bug.
- rPPG needs MediaPipe Face Landmarker; the model auto-downloads on first run (internet required). In `frame/`, only `yolov8n-face-lindevs.pt` auto-downloads; missing other presets raise `FileNotFoundError`.

## Verification

No test suite, no CI, no lint config in the repo. Verify changes by running `python -m quantum.pipeline --all` (or at least `--build-data --select --train`) and `python run_pipeline.py --source <some video> --method POS`.

`Docs/projec_audit.md` predates the real-data refactor; its "bridged bug" finding is already fixed in `run_pipeline.py` — ignore stale recommendations in it.
