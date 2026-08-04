# Interface Contract: rPPG Feature File (Component 3 -> Component 2)

This document defines the exact data format your rPPG pipeline (Component 3) must
produce so that the Quantum ML component (Component 2, this `quantum/` folder) can
consume it without any code changes.

## 1. Feature Vector Contract

Each video must be summarized into exactly **9 features**, in this fixed order:

| # | Column name                | Meaning                                                      | Expected range |
|---|----------------------------|--------------------------------------------------------------|----------------|
| 0 | temporal_consistency       | Coherence of the recovered pulse signal across the sequence  | [0, 1]         |
| 1 | inter_region_agreement     | Agreement between left cheek, right cheek, forehead signals  | [0, 1]         |
| 2 | signal_stability           | Waveform stability of the rPPG signal over time              | [0, 1]         |
| 3 | amplitude_reliability      | Strength and reliability of the pulse amplitude              | [0, 1]         |
| 4 | rhythm_quality             | Periodicity / biological plausibility of the rhythm          | [0, 1]         |
| 5 | sync_behavior              | Synchronization behavior across facial regions               | [0, 1]         |
| 6 | frame_quality_score        | Mean frame quality score from the extraction pipeline        | [0, 1]         |
| 7 | roi_stability              | ROI consistency across the frame sequence                    | [0, 1]         |
| 8 | temporal_coverage_ratio    | Fraction of sampled frames accepted by quality filtering     | [0, 1]         |

Rules:

- Values must be finite numbers in `[0, 1]`. Out-of-range or NaN rows are rejected by
  `quantum/conditioning.py` (`FeatureValidationError`).
- `label` is **1 = real, 0 = fake**.
- Column order is fixed. Do not rename or reorder columns.

## 2. Accepted File Formats

### Option A: CSV (simplest)

Columns: `split,label,temporal_consistency,...,temporal_coverage_ratio`

One row per video. `split` must be one of `train`, `val`, `test`.

Example:

```csv
split,label,temporal_consistency,inter_region_agreement,signal_stability,amplitude_reliability,rhythm_quality,sync_behavior,frame_quality_score,roi_stability,temporal_coverage_ratio
train,1,0.78,0.71,0.80,0.66,0.74,0.69,0.85,0.82,0.90
train,0,0.31,0.25,0.40,0.22,0.35,0.28,0.62,0.55,0.71
```

### Option B: NPZ (numpy archive)

Keys: `X_train`, `y_train`, `X_val`, `y_val`, `X_test`, `y_test`
Each `X_*` is shape `(n_samples, 9)` float32; each `y_*` is shape `(n_samples,)` int64.

This matches `quantum/dummy_data.py` output exactly, so the simplest path is:
produce your CSV, then run `quantum/tools_convert_csv_to_npz.py` (see below) or
copy the format of `quantum/output/data.npz`.

## 3. How to Plug Real Data In

1. Place the file at `quantum/output/data.npz` (or `features.csv`).
2. Run:

```powershell
$python = "C:\Users\JHASHANK\AppData\Local\Programs\Python\Python310\python.exe"
& $python "quantum\run_quantum.py" --all
```

The pipeline regenerates the QAOA selection, retrains the VQC, evaluates, compares
with classical baselines, and rewrites `quantum/docs/results_report.md`.

## 4. Contract Validation

`quantum/conditioning.py` enforces:

- 2D numeric matrix
- exactly 9 columns in schema order
- no NaN / infinite values
- all values within [0, 1]

If any rule fails, the component stops with `FeatureValidationError` instead of
silently training on corrupt data.
