# Project Scenario and Complete Matrices

Deepfake-video detection for KYC using rPPG (physiological evidence) + hybrid quantum-classical ML (PennyLane + PyTorch). Low-resolution input, decision bins REAL / FAKE / UNCERTAIN.

All numbers below are read verbatim from the regenerated artifacts in `WORKING/output/` (current frozen baseline `baseline_20260815_221652`, generated 2026-08-15 via `python -m quantum.pipeline --all`). Source files are listed in the final section.

---

## Part 1 — Detailed System Scenario

### 1.1 System Overview

A liveness / authenticity layer for KYC video onboarding. Instead of pixel-level artifacts, the system detects deepfakes from **physiological evidence** — the remote photoplethysmography (rPPG) pulse recovered from the face — and classifies it with a **hybrid quantum-classical model**: QAOA performs feature selection, and a PennyLane Variational Quantum Circuit (VQC) fused with a PyTorch ML head outputs P(real). The final decision is binned into three classes:

- **REAL** if `P(real) >= 0.7`
- **FAKE** if `P(real) <= 0.3`
- **UNCERTAIN** otherwise (0.3 < P(real) < 0.7)

### 1.2 Actors and Components

| Component | Role |
|---|---|
| **End user** | Uploads a selfie / onboarding video (≤ 200 MB; MP4/MOV/AVI/WebM, magic-byte validated) through the web UI; views verdict, waveform, frame thumbnails |
| **Frontend** (`frontend/`) | React + Vite UI (`src/`), API server `server.py` (stdlib-only): upload, SSE progress, artifact serving, concurrency cap (2 jobs → 429), 30-min hard pipeline timeout, job TTL 1 h / frame sequences 24 h, CORS restricted to localhost, sanitized inbox filenames `{job8}_{stem}.ext` |
| **Stage 1 — Frame sampling & face filtering** (`WORKING/frame/`) | 30 fps sampling → YOLOv8 face detection (`yolov8n-face-lindevs.pt`) → quality gate (face size, blur, pose, brightness) → accepted JPEGs + `frame_metadata.jsonl` |
| **Stage 2 — rPPG recovery** (`WORKING/RPPG/`) | MediaPipe Face Landmarker → face ROIs (forehead, left/right cheek) → POS or CHROM pulse extraction → 10 physiological features |
| **Stage 3 — Quantum verdict** (`WORKING/quantum/`) | QAOA feature selection (8 → 3 features) → hybrid VQC (torch MLP + PennyLane QNode) → `prob_real` → decision bin. Optional classical cross-checks (RandomForest side path only) |
| **Dashboard** | Verdict radial gauge, 6 insight metrics, live rPPG signal canvas, quantum-flow trace, file info, frame samples |

### 1.3 Detailed Scenario — Upload to Verdict (Runtime)

1. **Upload & validate.** `POST /api/detect` receives the raw body + `X-Filename`. The server streams to disk in 64 KB chunks (no 200 MB in-memory buffers), enforces the 200 MB cap, validates magic bytes (MP4/MOV `ftyp`, AVI `RIFF`, WebM `EBML`) → 415 friendly error otherwise, and stores the video as `{job8}_{stem}.ext`. Returns `{job: <id>}`.
2. **Stage 1 — Frames.** The video is decoded at native ~30 fps. Frames are sampled, YOLO locates faces, and a quality gate rejects frames by reason (`face_too_small`, `no_face`, `extreme_pose`, `blur`). Accepted JPEGs plus `frame_metadata.jsonl` are persisted under `output/frames/frame_sequences/<video>/`. Mean quality score and temporal coverage ratio are reported.
3. **Stage 2 — rPPG.** Stage-1 accepted frames (fallback: direct video read via `process_video`, recorded in `input_mode`) are processed by MediaPipe per frame. Per-frame skin-masked ROI means (YCrCb + `inRange`, computed once per frame) feed POS (default) or CHROM pulse extraction with vectorized sliding-window overlap-add. A single shared Welch PSD yields HR, SNR, spectral entropy, and SQI. Output: 10 physiological features. If usable frames < `min_usable_frames` (48) → `features=None` → **INCONCLUSIVE, exit code 3**.
4. **Stage 3 — Verdict.** Features are standardized with the stored scaler, the QAOA-selected 3 features are forwarded through the hybrid VQC (torch on CUDA when available; PennyLane QNode on `default.qubit` with `backprop`), producing `prob_real`. Binning per §1.1 yields the final verdict (REAL / FAKE / UNCERTAIN) with confidence `|prob_real − 0.5| × 2`.
5. **Result delivery.** The pipeline writes `result.json` (with `_signal` rel to the decimated stage-2 waveform JSON, same schema as `frontend/dump_signal.py`). The server serves the job synchronously: verdict, all stage stats, and the waveform are available via `GET /api/jobs/<id>`; SSE (`/api/jobs/<id>/events`) streams `line / stage / result / signal / error` events. The dashboard renders the verdict gauge, 6 insight metrics, the pulse waveform, the quantum-flow trace, and frame thumbnails. State machine: `idle → selected → running → done → error`.

### 1.4 Offline Training / Evaluation Scenario

1. **Data.** Real data only (no synthetic augmentation): DFDC archive (`archive/DFDC_Dataset`: 1727 real + 1566 fake clips on disk) + FF++ (train/val/test: FF-real + YouTube-real reals, FF-synthesis fakes) → feature table `output/rppg/dataset_features.csv` (3445 rows @ native 30 fps; 3444 after the implausible-value filter; labels in the CSV: `1 = fake, 0 = real`).
2. **Leakage-safe splits.** Seed 42, val 20% / test 20%. Grouping strategy: FF++ real/synthesis grouped by source subject (`ffpp:src:<first-id-token>`, 15 groups, min 10 / max 51 clips, 0 groups straddling splits), YouTube-real per-clip (`ffpp:yt:<clip-stem>`), DFDC per-clip (`clip:<path>`). CV uses `StratifiedGroupKFold(5)` on train with the same groups.
3. **QAOA selection.** Supervised discrimination weights (`2·|AUC_i − 0.5|`, exact Mann-Whitney, sign-agnostic, deterministic, no sklearn). Target 3 features. QAOA circuit on `lightning.qubit` (fallback `default.qubit`), precomputed `PauliRot` cost gates + X-mixer, 4 parallel restarts (`ProcessPoolExecutor`, seeds). Hamiltonian ≡ classical cost (hard-asserted < 1e-6; beta-alive regression-guarded in `quantum/tests.py`).
4. **VQC training.** HybridModel: PennyLane QNode (3 features → 3 qubits, `default.qubit`/backprop) + torch head, focal loss, 20 epochs, cosine LR. Single-forward val metrics per epoch.
5. **Evaluation.** Held-out test split (689 clips): accuracy, precision, recall, specificity, F1, AUC-ROC, PR-AUC, ECE, confusion matrix, decision bins, plus grouped 5-fold CV. Classical baselines (RandomForest, MLP, LogisticRegression, LinearSVC, GaussianNB, XGBoost) evaluated identically for reference. Plots: ROC, confusion matrix, calibration curve.

### 1.5 Verification History (context)

| Date | Event |
|---|---|
| 2026-08-13 | QAOA layer audit: beta-mixer regression + Hamiltonian mismatch fixed, regression-guarded; POS vectorization verified bit-identical; pipeline verdict `prob_real 0.6155 → UNCERTAIN` |
| 2026-08-15 | QAOA weights switched to supervised discrimination weights; full 3445-row extraction incl. FF++; leakage-fixed grouping; **frozen baseline snapshot** `output/quantum/baseline_20260815_221652/` |
| Status | Baseline is a majority-class predictor (100% UNCERTAIN bins); Phase 2 rPPG signal-recovery probe (`Scrape/probe_rppg_methods.py`) is the gate |

---

## Part 2 — All Available Matrices

**Label convention (quantum layer):** `LABEL_REAL = 1`, `LABEL_FAKE = 0` (the CSV labels are flipped in `quantum/data.py`). Confusion matrices are stored as `[[TN, FP], [FN, TP]]` with REAL as the positive class, i.e.:

| | Pred FAKE | Pred REAL |
|---|---|---|
| **Actual FAKE** | TN | FP |
| **Actual REAL** | FN | TP |

---

### 2.1 Dataset Composition Matrix

Source: `split_manifest.json` / `baseline_manifest.json` (seed 42, val 0.2, test 0.2, `filter_implausible=true`, 3445 → 3444 rows).

| Split | Rows | Real | Fake | Subject groups |
|---|---|---|---|---|
| Train | 2066 | 1129 | 937 | 1827 |
| Validation | 689 | 377 | 312 | 600 |
| Test | 689 | 377 | 312 | 612 |
| **Total** | **3444** | **1883** | **1561** | — |

Grouping strategy: FF++ source-subject groups = 15 (min size 10, max size 51, groups straddling splits = 0); YouTube-real per-clip; DFDC per-clip.

### 2.2 Hybrid VQC — Test Split (frozen baseline)

| | Pred FAKE | Pred REAL |
|---|---|---|
| **Actual FAKE** | 0 | 312 |
| **Actual REAL** | 0 | 377 |

Metrics: accuracy **0.5472** · precision 0.5472 · recall 1.0000 · specificity **0.0000** · F1 0.7073 · AUC-ROC **0.5032** · PR-AUC 0.5490 · ECE 0.0646 · balanced accuracy 0.5000.
Interpretation: majority-class predictor — every test clip predicted REAL; no genuine real/fake signal.

### 2.3 Hybrid VQC — Grouped 5-Fold CV

Mean: accuracy 0.5481 ± 0.0277 · AUC-ROC 0.5172 ± 0.0225 · PR-AUC 0.5700 ± 0.0366 · balanced accuracy 0.5000 ± 0.0000 · ECE 0.0664 ± 0.0246 · F1 0.7077 ± 0.0227 · recall 1.0 ± 0.0 · specificity 0.0 ± 0.0.

| Fold | CM [[TN,FP],[FN,TP]] | Acc | Prec | Rec | Spec | F1 | AUC | PR-AUC | ECE | BalAcc |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | [[0,198],[0,227]] | 0.5341 | 0.5341 | 1.0 | 0.0 | 0.6963 | 0.5590 | 0.6069 | 0.0715 | 0.5 |
| 2 | [[0,210],[0,227]] | 0.5195 | 0.5195 | 1.0 | 0.0 | 0.6837 | 0.5090 | 0.5308 | 0.0991 | 0.5 |
| 3 | [[0,147],[0,221]] | 0.6005 | 0.6005 | 1.0 | 0.0 | 0.7504 | 0.5052 | 0.6138 | 0.0227 | 0.5 |
| 4 | [[0,195],[0,229]] | 0.5401 | 0.5401 | 1.0 | 0.0 | 0.7014 | 0.5198 | 0.5716 | 0.0677 | 0.5 |
| 5 | [[0,187],[0,225]] | 0.5461 | 0.5461 | 1.0 | 0.0 | 0.7064 | 0.4932 | 0.5267 | 0.0711 | 0.5 |

### 2.4 Classical Baselines — Test Split

| Model | CM [[TN,FP],[FN,TP]] | Acc | Prec | Rec | Spec | F1 | AUC | PR-AUC | ECE | BalAcc |
|---|---|---|---|---|---|---|---|---|---|---|
| RandomForest | [[119,193],[126,251]] | 0.5370 | 0.5653 | 0.6658 | 0.3814 | 0.6114 | 0.5421 | 0.5895 | 0.1065 | 0.5236 |
| MLP | [[110,202],[119,258]] | 0.5341 | 0.5609 | 0.6844 | 0.3526 | 0.6165 | 0.5119 | 0.5459 | 0.0805 | 0.5185 |
| LogisticRegression | [[160,152],[153,224]] | **0.5573** | **0.5957** | 0.5942 | **0.5128** | 0.5950 | **0.5688** | 0.6033 | **0.0473** | **0.5535** |
| LinearSVC | [[0,312],[2,375]] | 0.5443 | 0.5459 | 0.9947 | 0.0000 | 0.7049 | 0.4576 | 0.5070 | 0.0064 | 0.4973 |
| GaussianNB | [[62,250],[59,318]] | 0.5515 | 0.5599 | 0.8435 | 0.1987 | 0.6730 | 0.5643 | 0.6021 | 0.0401 | 0.5211 |
| XGBoost | [[118,194],[114,263]] | 0.5530 | 0.5755 | 0.6976 | 0.3782 | 0.6307 | 0.5648 | 0.6033 | 0.0621 | 0.5379 |
| **Hybrid VQC** | [[0,312],[0,377]] | 0.5472 | 0.5472 | 1.0000 | 0.0000 | 0.7073 | 0.5032 | 0.5490 | 0.0646 | 0.5000 |

Best test AUC: LR 0.5688, XGB 0.5648, GNB 0.5643. Classical LR/GNB/XGB beat the VQC on AUC; the VQC's F1 is inflated by majority-class prediction.

### 2.5 Classical Baselines — Grouped 5-Fold CV (mean ± std)

| Model | Acc | AUC | BalAcc | ECE |
|---|---|---|---|---|
| RandomForest | 0.5086 ± 0.0287 | 0.5090 ± 0.0266 | 0.4975 ± 0.0244 | 0.1450 ± 0.0219 |
| MLP | 0.5227 ± 0.0104 | 0.5051 ± 0.0193 | 0.5036 ± 0.0117 | 0.0783 ± 0.0264 |
| LogisticRegression | 0.5357 ± 0.0270 | 0.5272 ± 0.0311 | 0.5301 ± 0.0268 | 0.0541 ± 0.0219 |
| LinearSVC | 0.5486 ± 0.0198 | 0.4816 ± 0.0230 | 0.5015 ± 0.0081 | 0.0275 ± 0.0190 |
| GaussianNB | 0.5446 ± 0.0205 | 0.5287 ± 0.0300 | 0.5129 ± 0.0105 | 0.0528 ± 0.0176 |
| XGBoost | 0.5023 ± 0.0158 | 0.5011 ± 0.0121 | 0.4925 ± 0.0143 | 0.1321 ± 0.0172 |

### 2.6 Classical Baselines — Per-Fold Confusion Matrices and Metrics (fully expanded)

#### RandomForest (5 folds)

| Fold | CM [[TN,FP],[FN,TP]] | Acc | Prec | Rec | Spec | F1 | AUC | PR-AUC | ECE | BalAcc |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | [[80,118],[84,143]] | 0.5247 | 0.5479 | 0.6300 | 0.4040 | 0.5861 | 0.5429 | 0.5580 | 0.1211 | 0.5170 |
| 2 | [[63,147],[88,139]] | 0.4622 | 0.4860 | 0.6123 | 0.3000 | 0.5419 | 0.4685 | 0.5002 | 0.1845 | 0.4562 |
| 3 | [[56,91],[77,144]] | 0.5435 | 0.6128 | 0.6516 | 0.3810 | 0.6316 | 0.5235 | 0.6124 | 0.1358 | 0.5163 |
| 4 | [[75,120],[96,133]] | 0.4906 | 0.5257 | 0.5808 | 0.3846 | 0.5519 | 0.4890 | 0.5329 | 0.1507 | 0.4827 |
| 5 | [[83,104],[93,132]] | 0.5218 | 0.5593 | 0.5867 | 0.4439 | 0.5727 | 0.5211 | 0.5659 | 0.1327 | 0.5153 |

#### MLP (5 folds)

| Fold | CM [[TN,FP],[FN,TP]] | Acc | Prec | Rec | Spec | F1 | AUC | PR-AUC | ECE | BalAcc |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | [[51,147],[51,176]] | 0.5341 | 0.5449 | 0.7753 | 0.2576 | 0.6400 | 0.5400 | 0.5732 | 0.0312 | 0.5165 |
| 2 | [[58,152],[60,167]] | 0.5149 | 0.5235 | 0.7357 | 0.2762 | 0.6117 | 0.4846 | 0.5014 | 0.0941 | 0.5059 |
| 3 | [[50,97],[81,140]] | 0.5163 | 0.5907 | 0.6335 | 0.3401 | 0.6114 | 0.4936 | 0.5999 | 0.1102 | 0.4868 |
| 4 | [[52,143],[64,165]] | 0.5118 | 0.5357 | 0.7205 | 0.2667 | 0.6145 | 0.5105 | 0.5524 | 0.0791 | 0.4936 |
| 5 | [[53,134],[57,168]] | 0.5364 | 0.5563 | 0.7467 | 0.2834 | 0.6376 | 0.4965 | 0.5429 | 0.0766 | 0.5150 |

#### LogisticRegression (5 folds)

| Fold | CM [[TN,FP],[FN,TP]] | Acc | Prec | Rec | Spec | F1 | AUC | PR-AUC | ECE | BalAcc |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | [[115,83],[107,120]] | 0.5529 | 0.5911 | 0.5286 | 0.5808 | 0.5581 | 0.5652 | 0.5943 | 0.0374 | 0.5547 |
| 2 | [[96,114],[111,116]] | 0.4851 | 0.5043 | 0.5110 | 0.4571 | 0.5077 | 0.4770 | 0.5007 | 0.0527 | 0.4841 |
| 3 | [[64,83],[87,134]] | 0.5380 | 0.6175 | 0.6063 | 0.4354 | 0.6119 | 0.5138 | 0.6211 | 0.0965 | 0.5209 |
| 4 | [[95,100],[85,144]] | 0.5637 | 0.5902 | 0.6288 | 0.4872 | 0.6089 | 0.5538 | 0.5760 | 0.0380 | 0.5580 |
| 5 | [[88,99],[91,134]] | 0.5388 | 0.5751 | 0.5956 | 0.4706 | 0.5852 | 0.5265 | 0.5632 | 0.0460 | 0.5331 |

#### LinearSVC (5 folds)

| Fold | CM [[TN,FP],[FN,TP]] | Acc | Prec | Rec | Spec | F1 | AUC | PR-AUC | ECE | BalAcc |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | [[0,198],[0,227]] | 0.5341 | 0.5341 | 1.0 | 0.0 | 0.6963 | 0.4874 | 0.5122 | 0.0202 | 0.5 |
| 2 | [[10,200],[4,223]] | 0.5332 | 0.5272 | 0.9824 | 0.0476 | 0.6862 | 0.4981 | 0.5192 | 0.0323 | 0.5150 |
| 3 | [[1,146],[6,215]] | 0.5870 | 0.5956 | 0.9729 | 0.0068 | 0.7388 | 0.4421 | 0.5610 | 0.0622 | 0.4898 |
| 4 | [[1,194],[0,229]] | 0.5425 | 0.5414 | 1.0 | 0.0051 | 0.7025 | 0.5078 | 0.5454 | 0.0105 | 0.5026 |
| 5 | [[0,187],[0,225]] | 0.5461 | 0.5461 | 1.0 | 0.0 | 0.7064 | 0.4725 | 0.5218 | 0.0121 | 0.5 |

#### GaussianNB (5 folds)

| Fold | CM [[TN,FP],[FN,TP]] | Acc | Prec | Rec | Spec | F1 | AUC | PR-AUC | ECE | BalAcc |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | [[32,166],[27,200]] | 0.5459 | 0.5464 | 0.8811 | 0.1616 | 0.6745 | 0.5755 | 0.5976 | 0.0250 | 0.5213 |
| 2 | [[42,168],[44,183]] | 0.5149 | 0.5214 | 0.8062 | 0.2000 | 0.6332 | 0.4891 | 0.5143 | 0.0730 | 0.5031 |
| 3 | [[26,121],[36,185]] | 0.5734 | 0.6046 | 0.8371 | 0.1769 | 0.7021 | 0.5132 | 0.6146 | 0.0528 | 0.5070 |
| 4 | [[33,162],[37,192]] | 0.5307 | 0.5424 | 0.8384 | 0.1692 | 0.6587 | 0.5481 | 0.5966 | 0.0698 | 0.5038 |
| 5 | [[40,147],[35,190]] | 0.5583 | 0.5638 | 0.8444 | 0.2139 | 0.6762 | 0.5174 | 0.5441 | 0.0435 | 0.5292 |

#### XGBoost (5 folds)

| Fold | CM [[TN,FP],[FN,TP]] | Acc | Prec | Rec | Spec | F1 | AUC | PR-AUC | ECE | BalAcc |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | [[62,136],[85,142]] | 0.4800 | 0.5108 | 0.6256 | 0.3131 | 0.5624 | 0.4792 | 0.5289 | 0.1421 | 0.4693 |
| 2 | [[76,134],[84,143]] | 0.5011 | 0.5162 | 0.6300 | 0.3619 | 0.5675 | 0.5022 | 0.5251 | 0.1376 | 0.4959 |
| 3 | [[65,82],[98,123]] | 0.5109 | 0.6000 | 0.5566 | 0.4422 | 0.5775 | 0.5000 | 0.6141 | 0.1472 | 0.4994 |
| 4 | [[77,118],[97,132]] | 0.4929 | 0.5280 | 0.5764 | 0.3949 | 0.5511 | 0.5114 | 0.5563 | 0.1350 | 0.4856 |
| 5 | [[66,121],[74,151]] | 0.5267 | 0.5551 | 0.6711 | 0.3529 | 0.6076 | 0.5129 | 0.5599 | 0.0988 | 0.5120 |

### 2.7 QAOA Selection Matrices

**Selected features (chosen restart seed 44, cost −1.3627, `success: false`):** `peak_prominence`, `left_right_cheek_correlation`, `spectral_entropy` (indices 9, 7, 3).

**Feature discrimination weights** (sign-agnostic AUC strength `2·|AUC−0.5|`; order = `FEATURE_NAMES`):

| # | Feature | Weight | # | Feature | Weight |
|---|---|---|---|---|---|
| 0 | heart_rate_bpm | 0.4573 | 5 | signal_quality_index | 0.8864 |
| 1 | snr_db | 0.8162 | 6 | cheek_forehead_correlation | 0.2401 |
| 2 | prv_std_ms | 0.3311 | 7 | left_right_cheek_correlation | 0.6037 |
| 3 | spectral_entropy | 0.8527 | 8 | hr_half_diff | 0.1890 |
| 4 | mad | 0.1281 | 9 | peak_prominence | 1.0000 |

**Final bit-flip marginal probabilities per feature:** [0.3090, 0.3349, 0.2963, 0.3494, 0.2900, 0.3459, 0.3068, 0.3584, 0.2885, 0.3628] (same order; max marginals at indices 9, 7, 3).

**QAOA restart matrix (4 restarts, parallel, `ProcessPoolExecutor`):**

| Restart seed | Cost |
|---|---|
| 1 | −0.0043 |
| 2 | −1.1747 |
| 3 (chosen) | **−1.3627** |
| 4 | −0.0509 |

**QAOA vs classical (AUC-greedy) reference selection:**

| | Features | Overlap |
|---|---|---|
| QAOA | peak_prominence, left_right_cheek_correlation, spectral_entropy | 2 (indices 3, 9) |
| Classical | peak_prominence, signal_quality_index, spectral_entropy | QAOA-only: left_right_cheek_correlation · classical-only: signal_quality_index |

### 2.8 Decision-Bin Matrix (VQC, test split)

| Bin | Count | Rate |
|---|---|---|
| REAL | 0 | 0.0% |
| FAKE | 0 | 0.0% |
| UNCERTAIN | 689 | 100.0% |
| `confirmed_accuracy` | null (no confirmed predictions) | |

ECE = 0.0646 — calibrated, but trivially (all probabilities near 0.5).

### 2.9 Feature Scaler Matrix (train-set mean / scale)

| Feature | Mean | Scale |
|---|---|---|
| heart_rate_bpm | 91.4161 | 38.2470 |
| snr_db | −4.0814 | 2.3070 |
| prv_std_ms | 127.7978 | 46.5828 |
| spectral_entropy | 0.8328 | 0.0632 |
| mad | 0.7667 | 0.0502 |
| signal_quality_index | 0.6240 | 0.0448 |
| cheek_forehead_correlation | 0.1547 | 0.2104 |
| left_right_cheek_correlation | 0.0722 | 0.2592 |
| hr_half_diff | 44.5070 | 37.2936 |
| peak_prominence | 3.8010 | 1.2035 |

### 2.10 End-to-End Pipeline Sample Run

`WORKING/output/pipeline/pipeline_result_aa716e0ddaf64bcd8e17fac19b314157.json` (POS method, stage-1 frames path; run 2026-08-15T16:45:05).

| Stage | Key stats |
|---|---|
| frames | 305 sampled → 191 accepted; rejections: face_too_small 97, no_face 1, extreme_pose 6, blur 15; mean quality 0.9204; mean face conf 0.8396; temporal coverage 0.6262 |
| rppg | POS, `input_mode: stage1_frames`, fps 30.0, 191 total / 186 usable frames, no warnings; HR 160.2 bpm, SNR −5.93 dB, PRV std 60.86 ms, spectral entropy 0.892, MAD 0.706, SQI 0.563, cheek-forehead corr −0.153, L/R-cheek corr −0.227, HR½ diff 94.7, peak prominence 2.655 |
| quantum | `prob_real 0.5894 → UNCERTAIN`, confidence 0.1789 |
| verdict | **UNCERTAIN**, confidence 0.1789 |

Note: this run predates the leakage-fix freeze and used the older selection (HR, SNR, MAD); the current model uses `peak_prominence`, `left_right_cheek_correlation`, `spectral_entropy` and the baseline results in §2.2–2.6.

### 2.11 Historical Verification Matrix (audit trail, 421-row DFDC-only and pre-freeze runs)

| Run | Selected features | Test acc | Test AUC | CV(5) AUC | ECE | Decision bins |
|---|---|---|---|---|---|---|
| Post-fix 2026-08-13 (421 rows) | hr_half_diff, cheek_forehead_corr, L/R-cheek corr, mad, HR, spectral_entropy | — | — | — | 0.1052 (buggy: 0.2478) | 100% UNCERTAIN |
| Post-selection-fix 2026-08-15 (421 rows) | heart_rate_bpm, snr_db, mad | 0.512 | 0.534 | 0.557 | 0.084 | 100% UNCERTAIN |
| Full extraction 2026-08-15 (3445 rows) | L/R-cheek corr, spectral_entropy, snr_db | 0.547 | 0.532 | 0.493 | 0.067 | 100% UNCERTAIN |
| **Frozen baseline 2026-08-15 (3444 rows)** | peak_prominence, L/R-cheek corr, spectral_entropy | 0.547 | 0.503 | 0.517 | 0.065 | 100% UNCERTAIN |

Context: per-feature |AUC − 0.5| ≤ ~0.06; HR-only logistic regression (AUC 0.595) beats the 10-feature VQC → feature dilution caps the ceiling; Phase 2 rPPG method/ROI probe is the gate.

### 2.12 Available Charts

`WORKING/output/quantum/` (+ `baseline_20260815_221652/`): `roc_curve.png`, `confusion_matrix.png`, `calibration_curve.png`. `Docs/charts/`: `latency_profile.png`, `dataset_composition.png`, `frame_funnel.png`, `model_comparison.png`, `qaoa_restarts.png`. Report figures under `Docs/maj_proj_report_sem_6/images/` (system architecture, workflow, ER / use-case / activity / class diagrams, Table 6.1 system scenario).

---

## Part 3 — Data Sources

| Data | File |
|---|---|
| VQC test metrics + CV + folds | `WORKING/output/quantum/metrics_quantum.json` (identical in `baseline_20260815_221652/`) |
| Baseline test metrics + CV + folds | `WORKING/output/quantum/metrics_baselines.json` |
| QAOA selection (weights, marginals, restarts) | `WORKING/output/quantum/qaoa_selection.json` |
| QAOA vs classical comparison | `WORKING/output/quantum/selection_comparison.json` |
| Split manifest (per-clip) | `WORKING/output/quantum/split_manifest.json` |
| Frozen-baseline summary | `WORKING/output/quantum/baseline_20260815_221652/baseline_manifest.json` |
| Feature scaler | `WORKING/output/quantum/feature_scaler.json` |
| Training log | `WORKING/output/quantum/training_log.jsonl` |
| E2E sample run | `WORKING/output/pipeline/pipeline_result_aa716e0ddaf64bcd8e17fac19b314157.json`, `signal_aa716e0ddaf64bcd8e17fac19b314157.json` |
| rPPG feature table | `WORKING/output/rppg/dataset_features.csv` (3445 rows; gitignored) |
| Model weights | `WORKING/output/quantum/hybrid_vqc.pt`, `rppg_classifier.pkl` (in `output/rppg/`, write-protected) |
