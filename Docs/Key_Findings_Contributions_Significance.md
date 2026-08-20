# Key Findings, Contributions, and Significance

**Project:** Deepfake Video Detection for KYC using rPPG and Hybrid Quantum-Classical Machine Learning

**Scope and evidence base.** This document is derived strictly from the implemented repository: the three-stage pipeline under `WORKING/` (frame sampling/quality, rPPG physiological feature extraction, QAOA feature selection + hybrid VQC decision), the end-to-end orchestrator `WORKING/run_pipeline.py`, the web frontend (`frontend/`), the documentation in `Docs/`, and the regenerated experiment artifacts in `WORKING/output/` (`metrics_quantum.json`, `metrics_baselines.json`, `qaoa_selection.json`, `selection_comparison.json`, `split_manifest.json`, `rppg_classifier_metadata.json`, `training_log.jsonl`, run logs in `Scrape/`). All quantitative claims cite the artifacts produced by the most recent full run (2026-08-19, torch-native simulators) or, where marked, documented earlier baselines. Label conventions: the quantum stage uses `LABEL_REAL = 1`, `LABEL_FAKE = 0`; the rPPG CSV uses the opposite (`1 = fake`). All quantum computation in this project is executed on **simulated** statevector backends on classical hardware — no quantum hardware was used.

---

## 1. Key Findings

### F1. A complete three-stage pipeline is implemented, integrated, and runs end-to-end

The pipeline `frames → rPPG → quantum → verdict (REAL / FAKE / UNCERTAIN)` is fully wired in `WORKING/run_pipeline.py`: stage 1 (YOLO face detection, quality gating, 30 fps sampling) hands its accepted JPEGs and `frame_metadata.jsonl` to stage 2 (`RPPGPipeline.process_frames`), which computes the 20-feature physiological vector consumed by stage 3 (`quantum.pipeline.predict_features`), which emits a KYC binned verdict with confidence. A documented sample run processed 305 sampled frames → 191 accepted → 186 usable rPPG frames at 30 fps and produced `UNCERTAIN` (P(real) = 0.589) (RMTT report §3.1). The stage-1 quality layer reports per-frame metadata and an rPPG-readiness summary (accepted count ≥ 64, continuity tolerance 35%, mean face ≥ 48×48 px, green-channel variation) (`WORKING/frame/app/pipeline.py:494-552`). The web frontend (`frontend/server.py`, stdlib-only) exposes upload validation, SSE progress events, artifact serving, and a dashboard; its E2E flow (idle → selected → running → done) is documented as verified including invalid-file handling (415) and sequential reruns.

### F2. The "real data only" constraint is honored end-to-end

The quantum layer consumes only the real labeled rPPG feature table `WORKING/output/rppg/dataset_features.csv` — 3473 rows (1921 real / 1552 fake) from DFDC and FaceForensics++ at native 30 fps (`rppg_classifier_metadata.json`). A plausibility filter (HR ∈ [30, 220] BPM, non-finite rejection) dropped 2 of 3473 rows at dataset build time (run log: "3471/3473 kept"), and the split is written to `split_manifest.json` (seed 42, val 0.2 / test 0.2) with per-sample group keys. The 20-name feature contract between `quantum/config.py` `FEATURE_NAMES` and `RPPGFeatures.feature_names()` is duplicated deliberately and enforced at runtime: missing CSV columns raise `ValueError` (`WORKING/quantum/data.py:99-107`), and a dedicated test asserts exact list equality (`quantum/tests.py:112-122`, PASS). No synthetic-data generator or transform/bridge layer exists in the repository.

### F3. Leakage-safe grouped evaluation is implemented and regression-tested

Data splits group FF++ clips by **source subject** (`ffpp:src:<first-id-token>`; a real clip `id0_0000.mp4` and its synthesis `id0_id16_0002.mp4` share a group), YouTube-real clips per clip, and DFDC clips per clip (a documented limitation: DFDC subject identity is unrecoverable from filenames) (`quantum/data.py:43-63`). `_assert_no_group_leakage` raises `AssertionError` if any group appears in more than one split (`data.py:231-247`). Cross-validation is 5-fold **group-aware** (`StratifiedGroupKFold`, fallback `GroupKFold`) on the train split only — the test split is untouched until final evaluation (`quantum/evaluation.py:128-172, 223-225`). Tests `test_ffpp_source_subject_grouping`, `test_no_group_leakage`, and `test_split_determinism` all PASS. This protocol is the direct response to an identified leakage flaw in earlier splits (commit `58fbf96`, README/AGENTS.md).

### F4. The QAOA layer is verification-clean, with a documented defect history

Two historical defects were found by audit (2026-08-13) and fixed with regression guards: (a) a dead beta mixer — perturbing β left the cost invariant — now guarded by `test_beta_alive`; (b) a Hamiltonian that did not reproduce the classical cost (max error ≈ 1.94, print-only) — now exact. The current `_cost_terms` reproduces `_classical_cost` exactly: `pipeline.py` hard-asserts `error < 1e-6`, the latest run measured **5.68e-14**, and the documented earlier verification measured 7.1e-15 on real data. The full self-test suite — 10 checks covering the beta-alive ansatz, Hamiltonian ≡ classical cost (synthetic and real data), feature-contract sync, split determinism, subject grouping, no group leakage, and cross-verification of both torch-native simulators against PennyLane — passes 10/10 (re-verified 2026-08-19).

### F5. QAOA converges, but its solutions are sensitive to restarts and hyperparameters

On the latest run, QAOA (p = 3 layers, COBYLA, 4 parallel restarts) selected `[cheek_forehead_correlation, left_right_cheek_correlation, signal_to_motion_ratio]` with cost −0.767, but the per-restart costs spanned **[11.73, −0.098, −0.767, 9.83]** and the optimizer reported `success: false` (`qaoa_selection.json`) — two of four restarts terminated far from the optimum, indicating a difficult optimization landscape for this QUBO-style objective. The greedy classical reference (top-k by discrimination AUC) agreed on 2 of 3 features (`selection_comparison.json`). A documented hyperparameter sweep phase (2026-08-18, `Scrape/phase1_sweep_qaoa_*.log`/`txt`) found the same pattern: for k = 3, p = 2 the QAOA solution cost 0.4995 with 2/3 overlap vs classical; for p = 3 the cost was −0.7072 with **0/3 overlap** — the selection itself flips with QAOA depth. The selection objective was also revised for cause: unsupervised mutual-information weights systematically excluded the two strongest features (HR, SNR); the current supervised discrimination weights (sign-agnostic exact Mann-Whitney AUC strength `2·|AUC−0.5|`, deterministic, no sklearn) fixed this, with documented CV-AUC evidence (0.484 → 0.569 on a 3-feature probe).

### F6. Core negative result: the decision layer is honest but the physiology features carry almost no class signal

This is the central experimental finding. On the held-out test split (694 clips: 384 real / 310 fake, 617 subject groups), the hybrid VQC achieves:

| Metric | VQC test | VQC 5-fold CV (mean ± std) | Best classical (LR test) |
|---|---|---|---|
| Accuracy | 0.552 | 0.553 ± 0.009 | 0.552 |
| Precision | 0.553 | 0.553 ± 0.009 | 0.601 |
| Recall | 0.997 | 1.000 ± 0.0 | 0.568 |
| Specificity | **0.000** | 0.000 ± 0.0 | 0.532 |
| F1 | 0.711 | 0.712 ± 0.008 | 0.584 |
| AUC-ROC | 0.535 | 0.556 ± 0.019 | **0.582** |
| PR-AUC | 0.582 | 0.615 ± 0.013 | 0.625 |
| ECE | 0.068 | 0.063 ± 0.008 | 0.054 |
| Balanced accuracy | 0.499 | 0.500 | 0.550 |
| Confusion matrix | [[0, 310], [1, 383]] | every fold: TN = 0 | — |

(Sources: `metrics_quantum.json`, `metrics_baselines.json`, `training_log.jsonl` — 27 epochs, cosine LR 0.01 → 0.0076, val loss plateau ~0.058.)

The confusion matrix shows the VQC is a **majority-class predictor**: it labels every test clip REAL (specificity 0.0, recall ≈ 1.0, balanced accuracy ≈ 0.5). Decision bins are **100% UNCERTAIN** (694/694 at the 0.3/0.7 thresholds), so no video receives a confident call — safe, but not yet discriminating. Critically, the classical baselines trained on the same selected features are equally weak (LR AUC 0.582, LinearSVC 0.581, GNB 0.568, XGB 0.546, RF 0.536, MLP 0.528), and per-feature discrimination strength is |AUC − 0.5| ≤ ~0.06 for all 20 features (README/AGENTS.md verification notes). An 8.2× training-data increase (421-row DFDC-only table → 3445-row DFDC+FF++ table, documented 2026-08-15) did not lift the ceiling; the documented AUC progression across frozen baselines (0.503 → 0.543 → 0.535) is flat within noise. **Conclusion supported by the evidence:** the bottleneck is the informativeness of the rPPG feature set itself, not the quantum or classical decision layer — the documented next gate is a Phase-2 rPPG method/ROI probe, not further model tuning.

### F7. Substantial engineering results: exact torch-native simulators and measured speedups

Because this host has no GPU circuit backend (PennyLane ≥ 0.39 removed `default.qubit.torch`; `lightning.gpu` has no Windows wheels), the project implements its own **exact complex128 statevector simulators**: `qaoa_sim.py` (QAOA) and `QuantumLayerTorch` (`vqc.py`), both cross-verified against PennyLane by the test suite (cost and marginals ≤ 1e-6; forward outputs ≤ 1e-5; checkpoints remain load-compatible because parameter names/shapes are preserved). Documented performance: QAOA circuit call ~**5.6 s (PennyLane) → ~0.3–0.5 ms (torch sim)** on the 20-wire selection problem (~20× faster, `qaoa_sim.py:4-5`, `qaoa.py:333`), ~5 µs for 3 wires. The VQC QNode runs in broadcast mode (batched rows), verified bit-identical to the per-row loop (max |diff| = 0.0), with documented ~28× training / ~156× inference gains (README). Other measured items: CUDA torch-head gain 12.44 s vs 14.35 s for a 15-epoch VQC train (README, pre-refactor measurement); inference ~275 ms cold / ~32 ms cached (README); lazy heavy imports defer sklearn (~12 s), xgboost (~13 s), and matplotlib (~0.76 s) until actually needed.

### F8. KYC-safe decision design, with explicitly documented evidence gaps

The decision design is defensible for its domain: P(real) is binned into REAL (≥ 0.7) / FAKE (≤ 0.3) / UNCERTAIN with confidence `2·|P(real) − 0.5|`; `features=None` (usable frames < 48) yields INCONCLUSIVE with exit code 3 (`run_pipeline.py:299-306`); the server caps concurrency at 2 jobs, validates magic bytes and 200 MB uploads, and serves artifacts path-traversal-protected. The frontend renders verdict gauge, signal canvas, and frame samples; E2E state-machine flows are documented as verified.

**Evidence gaps — explicitly not claimed:** (i) no ground-truth pulse validation (HR plausibility is recorded but never validated against a reference pulse sensor); (ii) no formal latency/throughput benchmark of the full pipeline (the README timings above are component-level); (iii) all quantum execution is simulated — no quantum hardware; (iv) the rPPG RandomForest cross-check inside `run_pipeline.py` still guards for a 10-feature classifier while the current pickle is 20-feature (`rppg_classifier_metadata.json`), so that cross-check is currently skipped as incompatible — a stale guard rather than a functional check.

---

## 2. Contributions

The project's contributions are organized into methodological, implementation, engineering, and evaluation categories. Standard libraries, published algorithms, and public datasets used by the project are listed separately in §2.2 and are **not** claimed as contributions.

### 2.1 Genuine contributions

**Methodological.**

- **QAOA feature-selection objective for physiological features.** The selection cost is a supervised QUBO: discrimination term (deterministic sign-agnostic Mann-Whitney AUC strength, `2·|AUC−0.5|`, computed by exact pair counting — no sklearn), correlation-redundancy penalty, and cardinality constraint, optimized by QAOA and interpreted through per-qubit measurement marginals (`qaoa.py:175-205, 368-429`). This is a concrete instantiation of quantum optimization on a real, non-synthetic feature-selection problem, with a documented rationale for replacing the unsupervised MI objective (which systematically excluded the two strongest features).
- **Exact QUBO → Ising Hamiltonian mapping.** The classical cost is mapped to a Pauli-Z Hamiltonian with a closed-form constant, including the −¼·Σq_ij cross terms inside the linear coefficients, so that ⟨H⟩ ≡ C(b) exactly for every bitstring (`qaoa.py:213-257`); this is enforced by a hard assert (< 1e-6) in the orchestration pipeline rather than a silent check.
- **Leakage-safe evaluation protocol.** Source-subject grouping for FF++ (first-id token), per-clip grouping for DFDC/YouTube-real, group-leakage assertion, grouped 5-fold CV on train-only, and a persisted split manifest — a reusable template for honest evaluation of clip-level deepfake models.
- **Feature-contract discipline across stages.** The 20-name/order contract between rPPG and quantum stages is duplicated, runtime-validated (CSV columns, scaler length, selection indices, checkpoint compatibility), embedded in `data.npz` and the scaler JSON, and regression-tested — preventing silent breakage between independently developed stages.

**Implementation.**

- **Exact torch-native statevector simulators** for both QAOA and the VQC layer, CPU-process-safe (workers never open CUDA contexts), cross-verified against PennyLane to ≤ 1e-5/1e-6, with ~20× QAOA speedup and load-compatible checkpoints — a practical alternative to library backends that are unavailable on Windows hosts.
- **GPU-first device auto-detection** with one-time precision (1e-6 vs classical cost) and speed (≤ 3× lightning) probes that gate CUDA use (`qaoa.py:26-131`), and sequential CV folds on GPU hosts to avoid shared-card OOM.
- **rPPG engineering:** motion-compensated ROIs (RANSAC on stabilization landmarks + EMA smoothing), skin-mask-refined cheek/forehead polygons, POS/CHROM pulse extraction, 20 physiological features from a single shared Welch PSD, NaN hygiene with median fallback and SQI gating, and the `min_usable_frames = 48` gate that produces a clean `features=None`/INCONCLUSIVE contract (`RPPG/rppg/`).
- **Stage-1 quality layer:** YOLO face detection with six rejection rules (blur/dark/overexposed/no-face/face-too-small/extreme-pose), a composite quality score, and an rPPG-readiness summary.
- **System integration:** end-to-end orchestrator with in-process signal export (`--signal-out`, matching verdict-computation fps), lazy import of the heavy quantum stack so SSE progress streams immediately, model caching keyed by checkpoint mtime/size, and a stdlib-only API server with upload validation, concurrency caps, TTL cleanup, and SSE events.

**Engineering optimizations (measured or verified).** Vectorized POS via `sliding_window_view` + batched matmul + `np.add.at` overlap-add (bit-identical to the loop); shared Welch PSD (one periodogram instead of four); `lru_cache` on Butterworth coefficients and the detrend matrix; `cv2.mean` ROI averaging (no masked-array allocation); skin-mask hoist per frame; Laplacian skipped for no-face frames; broadcast-mode QNode (bit-identical, documented ~28×/156×); lazy imports; single-forward validation metrics; parallel dataset extraction with per-item timeouts.

**Evaluation discipline.** A full metric suite (accuracy, precision, recall, specificity, F1, AUC-ROC, PR-AUC, ECE, confusion matrix, balanced accuracy, decision bins with uncertain-rate and confirmed accuracy), six classical baselines under the identical grouped-CV protocol, ROC/confusion/calibration plots, seed-pinned reproducibility (seed 42 throughout, `split_manifest.json`, checkpoints bundling scaler stats, selection, and all configs), and a crash-safe hyperparameter sweep harness (`quantum/sweep.py`: QAOA phase ×9 configs, VQC phase ~174 combos, leaderboard by AUC).

### 2.2 Reused components (not contributions)

The following are standard libraries, published algorithms, or public resources incorporated as-is: YOLOv8 face detection (ultralytics); MediaPipe Face Landmarker (auto-downloaded model); the POS pulse algorithm (Wang et al., 2017) and CHROM (de Haan & Jeanne, 2013); Tarvainen smoothness-priors detrending; PennyLane (circuit primitives `AngleEmbedding`, `StronglyEntanglingLayers`, `PauliRot`), PyTorch, scikit-learn, XGBoost, scipy COBYLA; the DFDC and FaceForensics++ datasets; standard classification metrics and plots; Haar cascade and YuNet ONNX face detectors. These are used for what they are; the project's contributions lie in the objectives, integration, validation, and honest evaluation built around them.

---

## 3. Significance

### 3.1 Practical significance

Video-KYC onboarding is a high-value attack surface where deepfakes defeat visual-artifact detectors, especially at low resolution and under compression. This project demonstrates a **physiological-evidence alternative**: a pulse signal recovered remotely from facial skin regions, which synthesized faces do not reproduce faithfully and which is more robust to compression than pixel artifacts. The deployed decision design is domain-appropriate — an UNCERTAIN bin with confidence, and an INCONCLUSIVE path when physiological evidence is unusable, prevents the costliest failure mode of an anti-fraud system: a confident wrong call. The honesty of the evaluation protocol matters operationally: because the negative result (F6) was produced under leakage-safe grouping with classical baselines under identical conditions, it reliably identifies the bottleneck (feature informativeness, not the decision layer) and redirects effort to the documented next step (rPPG method/ROI probing) instead of premature threshold tuning or model substitutions. Where useful: video-KYC/AML onboarding systems, liveness and anti-spoofing research, and low-resolution video forensics in general.

### 3.2 Academic significance

The project contributes a worked, reproducible case study of hybrid quantum-classical ML applied to **real, non-synthetic data**: a QAOA-based feature selector with an exactly verified Hamiltonian, feeding a hybrid VQC, under a rigorous evaluation protocol that includes leakage control — an area where quantum-ML papers frequently rely on synthetic or leakage-prone settings. The exact Ising-mapping derivation and the torch-native simulator engineering are reusable techniques for QAOA experiments without PennyLane-backend availability. Equally, the project demonstrates the scientific value of a **well-documented negative result**: it quantifies the ceiling of a 20-feature POS/CHROM-derived physiological descriptor (per-feature |AUC − 0.5| ≤ ~0.06, flat AUC across an 8.2× data increase) and shows classical and quantum models saturate at the same ceiling — evidence that future work should target signal recovery rather than classifiers. The verification-first discipline (regression tests for dead mixer and Hamiltonian mismatch, hard asserts, reproducibility artifacts) is a template for trustworthy QML experimentation.

### 3.3 What remains unproven

Consistent with the evidence-only mandate: (i) whether improved rPPG methods/ROIs or temporal modeling can lift discrimination above chance — the documented Phase-2 probe is the required gate; (ii) physiological validity against ground-truth pulse; (iii) end-to-end latency certification and deployment hardening; (iv) generalization beyond DFDC + FF++ (e.g., unseen generation methods, real-world KYC capture conditions); (v) behavior on actual quantum hardware. None of these are claimed; all are explicitly open.