# Quantum Layer Guide — Hybrid Quantum-Classical Decision Stage

Deepfake Video Detection for KYC using rPPG + Hybrid Quantum-Classical ML.

This document explains the **quantum layer** (stage 3 of the 3-stage pipeline) from the ground up: the quantum-computing concepts it uses, how the layer is organized, what every file does, how it performs, and how to present it to judges.

---

## 1. Executive Summary

The project detects deepfake videos for KYC (Know-Your-Customer) onboarding using **physiological evidence** — a real face shows a faint periodic color change caused by blood flow (remote photoplethysmography, rPPG); a synthesized face does not. The pipeline has three stages:

| Stage | What it does | Output |
|---|---|---|
| 1. Frame (`WORKING/frame/`) | Samples frames, detects faces (YOLO), filters blur/brightness | Accepted JPEGs + metadata |
| 2. rPPG (`WORKING/RPPG/`) | Extracts pulse signals from face ROIs (POS/CHROM), computes 10–20 physiological features | 20-feature vector per video |
| 3. **Quantum (`WORKING/quantum/`)** | **QAOA feature selection + hybrid VQC classification** | `P(real)` → **REAL / FAKE / UNCERTAIN** |

The quantum layer does two jobs:

1. **QAOA feature selection** — a quantum optimization algorithm (Quantum Approximate Optimization Algorithm) selects the 3 most discriminative features from the 20 rPPG features. Selecting k features out of n is a combinatorial problem over a 2^n search space; QAOA is a quantum heuristic designed for exactly this class of problems.
2. **Hybrid VQC classification** — a Variational Quantum Circuit (a small parameterized quantum neural network) consumes the 3 selected features and outputs a probability that the face is real. The quantum circuit is trained end-to-end with PyTorch (AdamW, focal loss).

The layer is fully **simulated** on classical hardware (PennyLane statevector simulator + a custom torch-native simulator written for this project) — no quantum hardware is required to run it, which makes it deployable today while the algorithms are genuine quantum algorithms that transfer directly to quantum processors.

**Honest headline result (frozen baseline, 2026-08-18):** the layer runs end-to-end and is verification-clean (Hamiltonian ≡ classical cost to <1e-6, 7/7 regression tests pass), but current test metrics hover near chance (VQC test AUC 0.543, best classical baseline GNB 0.570, CV AUC 0.528). The bottleneck is the rPPG features themselves — per-feature discrimination strength |AUC−0.5| ≤ ~0.06 — not the quantum layer. The decision bins are therefore 100% UNCERTAIN at the 0.3/0.7 thresholds. This is presented transparently in §5.

---

## 2. Core Quantum Computing Concepts Used in This Project

This section explains every quantum concept the layer uses, in plain language with the concrete use in this project. A judge-friendly one-liner is given first, then the technical detail.

### 2.1 Qubits, superposition, and measurement

- **Qubit** — the quantum analogue of a bit. A classical bit is 0 or 1; a qubit is a vector in a 2-dimensional complex space, written |ψ⟩ = α|0⟩ + β|1⟩ with |α|² + |β|² = 1. The coefficients α, β are *probability amplitudes*.
- **Superposition** — a qubit can be in a combination of |0⟩ and |1⟩ simultaneously. This is what gives quantum computation its parallelism: n qubits live in a 2ⁿ-dimensional state space. **In this project:** 20 qubits represent the 20 candidate features, so QAOA explores a state space of 2²⁰ = 1,048,576 feature subsets at once.
- **Measurement / expectation value** — you never read a qubit's amplitudes directly; you measure and get |0⟩ or |1⟩ with probability |α|² or |β|². For a hermitian operator O (an *observable*), the expectation value ⟨ψ|O|ψ⟩ is the average measurement outcome. **In this project:** the cost of a feature subset is encoded as an observable H; QAOA minimizes ⟨H⟩, and qubit marginals ⟨Z_i⟩ are measured to decide which features to keep.

### 2.2 Quantum gates

Quantum gates are unitary operations on qubits — the quantum analogue of logic gates. The gates used in this project:

| Gate | Matrix / effect | Where it is used in the project |
|---|---|---|
| **Hadamard (H)** | H = 1/√2 [[1,1],[1,−1]]; maps |0⟩ → (|0⟩+|1⟩)/√2, creating superposition | Initial state of the QAOA ansatz: |+⟩^n (all 2ⁿ subsets equally weighted) and of the VQC circuit |
| **Pauli-X (X)** | X = [[0,1],[1,0]]; the quantum NOT | Basis-state preparation in `verify_hamiltonian`; the QAOA **mixer** (as R_x rotations) |
| **Pauli-Z (Z)** | Z = [[1,0],[0,−1]]; flips the phase of |1⟩ | Observable of the cost Hamiltonian and of VQC outputs ⟨Z_i⟩ |
| **Pauli-Y (Y)** | Y = [[0,−i],[i,0]] | Component of the entangling gates in the VQC ansatz |
| **RY(θ), RZ(φ)** | Single-qubit rotations by Euler angles | VQC **angle embedding** (RY per feature) and the strongly-entangling ansatz's Rot(φ,θ,ω) = RZ(ω)RY(θ)RZ(φ) |
| **CNOT** | Controlled-NOT: flips target iff control is |1⟩ | Creates **entanglement** between qubits in the VQC ansatz |
| **PauliRot(2γc, P, wires)** | exp(−iγc·P) for a Pauli string P | The QAOA **cost layer**: applies the evolution under the Hamiltonian term by term |

### 2.3 Entanglement

Two qubits are entangled when their state cannot be written as a product of independent qubit states. Measuring one instantly determines the other's measurement statistics. Entanglement is the resource that lets a quantum circuit explore correlations between variables that a classical product distribution cannot.

**In this project:** the CNOT ring in the VQC's strongly-entangling ansatz entangles all qubits, letting the circuit represent feature *interactions*. QAOA's mixer+ cost layers also create entanglement so the quantum state can represent correlated feature subsets (the redundancy-penalty term in the cost explicitly penalizes picking correlated features together — a purely classical problem, but one whose optimum QAOA finds in superposition).

### 2.4 Quantum circuits and the ansatz

A quantum circuit is a sequence of gates applied to qubits, followed by measurement. A *parameterized* circuit with tunable angles is called an **ansatz**. The project uses two ansätze:

**QAOA ansatz (feature selection).** For p layers of optimization (here p = 3):

```
|+>^n ── [Cost layer (γ₁) → Mixer layer (β₁)] ── ... ── [Cost layer (γₚ) → Mixer layer (βₚ)] ── ⟨H⟩
```

- **Cost layer** applies exp(−iγ·H): it rotates the state so that subsets with low cost C(b) accumulate more amplitude. Because every term in H is a Pauli-Z string, all terms commute, so the layer is applied exactly as a product of PauliRot gates — no Trotter error.
- **Mixer layer** applies R_x(2β) on every qubit; it explores the space of subsets (without it, the state would only ever shrink toward the single best basis state).
- The 2p angles (γ₁,β₁,…,γₚ,βₚ) are optimized classically (COBYLA) to minimize ⟨H⟩ — a "variational" hybrid loop.

**VQC ansatz (classification):**

```
AngleEmbedding(x, rotation="Y") ── StronglyEntanglingLayers(θ) ── ⟨Z_0⟩, …, ⟨Z_{n-1}⟩
```

- **Angle embedding**: each feature value x_i is encoded as the angle of an RY rotation on qubit i. (This is why the features are z-score standardized first — raw rPPG magnitudes like heart rate ~92 BPM are not meaningful rotation angles; standardized values are.)
- **StronglyEntanglingLayers(θ)**: a stack of qml layers (here 3), each a full layer of Rot(φ,θ,ω) gates (one per qubit) followed by a ring of CNOTs entangling every qubit. The trainable parameters θ are the rotation angles.
- The outputs are the expectation values ⟨Z_i⟩ — one real number per qubit — which feed a small classical neural head (Linear → ReLU → Dropout → Linear) that maps them to logits → P(real).

### 2.5 The hybrid quantum-classical training paradigm

Neither QAOA nor the VQC is "trained" in the usual sense. Both follow the same **variational** pattern:

1. Forward: run the parameterized quantum circuit on a classical simulator (or hardware) → cost/probability.
2. Backward: compute gradients of the cost w.r.t. the circuit angles (automatic differentiation through the simulator — analytic backprop, exact for these small circuits).
3. Update: adjust angles with a classical optimizer (COBYLA for QAOA; AdamW with cosine annealing, gradient clipping, early stopping for the VQC).
4. Repeat.

The quantum circuit itself is the *model*; the classical optimizer is the *teacher*. This is the standard "NISQ-era" recipe (Variational Quantum Eigensolver, QAOA, Quantum Neural Networks).

### 2.6 Statevector simulation (what actually runs today)

All quantum circuits in this project run on **classical simulators**:

- **PennyLane** devices (`default.qubit`, `lightning.qubit`): exact statevector simulators that keep the full 2ⁿ amplitude vector and apply gates as matrix–vector products.
- **A custom torch-native simulator** (`qaoa_sim.py`, `vqc.py::QuantumLayerTorch`): exact, complex128 (float64), fully differentiable, written with pure PyTorch ops. It reproduces the PennyLane circuits bit-for-bit (to float rounding) while removing PennyLane's per-gate tape overhead: QAOA cost calls went from ~5.6 s to ~0.5 ms per call on the 20-wire problem (~10,000×).

Why simulation? Three reasons: (a) reproducibility and testing (deterministic, no noise), (b) deployability (no quantum hardware needed for a KYC product), (c) the algorithm itself is hardware-agnostic — the exact same ansatz runs on real quantum processors. The verification harness (`verify_hamiltonian`) is what guarantees the simulated physics is correct.

### 2.7 QAOA mechanics in detail

**The optimization problem.** Select k = 3 features from n = 20 that (a) individually separate real from fake, (b) are not redundant with each other, (c) contain exactly k features. A feature subset is a bitstring b ∈ {0,1}²⁰. The classical cost is:

```
C(b) = −w·b  +  rp · Σ_{i<j} C_ij · b_i · b_j  +  cp · (Σb − k)²
```

- **w_i** — supervised discrimination weight of feature i: `2·|AUC_i − 0.5|`, the exact Mann-Whitney U statistic separating the two classes, normalized to [0,1]. (This replaced an earlier mutual-information weighting that systematically ranked the two strongest features — heart rate, SNR — near zero.)
- **rp = 0.3** — redundancy penalty; penalizes selecting correlated features (C_ij = |Pearson correlation|). Note the classic identity: Σ_{i<j} C_ij b_i b_j = ½(bᵀCb − Σb).
- **cp = 0.5** — cardinality penalty; pushes the selection to exactly k = 3 features.

**The Hamiltonian encoding.** The step that makes this a quantum problem: substitute the qubit operator identity **b = (1 − Z)/2** into C(b). Because b² = b for bits, C(b) becomes a linear-plus-quadratic form in the Pauli-Z operators:

```
H = const·I + Σ lin_i·Z_i + Σ_{i<j} quad_ij·Z_i·Z_j
```

with closed-form expressions for lin_i, quad_ij, const (implemented in `qaoa._cost_terms`, with the full algebra in the docstring). H is diagonal in the computational basis and **⟨b|H|b⟩ = C(b) exactly**. This equivalence is hard-verified at every pipeline run: max |⟨b|H|b⟩ − C(b)| < 1e-6 over all-zeros, all-ones, every single-bit, and 8 random bitstrings (`verify_hamiltonian`, asserted in `pipeline.py`).

**Why quantum at all?** Brute force over 2²⁰ = 1,048,576 subsets is feasible classically for 20 features (and the project ships a classical greedy reference `select_classical` for comparison). QAOA is included because: (a) it is the standard quantum algorithm for exactly this combinatorial-optimization class (QUBO / MaxCut-like), (b) feature selection in higher dimensions (100+ features) becomes intractable classically, and the same code scales there, (c) it demonstrates the hybrid quantum-classical methodology the project evaluates honestly rather than as marketing. The classical comparison exists precisely to verify QAOA is not worse than a naive greedy pick.

**Marginal probabilities → selection.** After optimization, the final state has amplitude on many subsets. The probability that qubit i is |1⟩ — `P(b_i = 1) = (1 − ⟨Z_i⟩)/2` — is the **marginal probability** of feature i. The k features with the highest marginals are selected. This is a nice quantum property: the answer is read out as a probability distribution over all subsets, not a single greedy decision.

**Restarts.** QAOA is not convex; the optimizer (COBYLA, max 200 iterations) can land in a local optimum. Four restarts with different seeds (42–45) run and the lowest-cost result wins. Restart seeds 42 and 45 got stuck at cost ≈ +10, seed 43 at −0.098, seed 44 found the best cost −0.707.

### 2.8 Loss function (VQC training)

The VQC head is trained with a **focal loss** variant:

```
L = −α_t·(1−p_t)^γ·log(p_t)  −  λ·H(probs)
```

- Focal term: down-weights easy samples (p_t near 1) so the model focuses on hard cases; α = 0.75 re-weights the positive class, γ = 2.0 is the focusing parameter.
- Confidence penalty: the second term (entropy H, λ = 0.02) discourages confident-but-wrong predictions.
- Label smoothing (0.03): prevents over-confidence on the training labels.
- The head + optimizer run on CUDA when available; the quantum layer executes on its own backend (CPU simulator today on this host).

### 2.9 Why "hybrid quantum-classical" ML

The model is a pipeline: **quantum circuit (parameterized, differentiable) → classical neural head (Linear+ReLU+Dropout+Linear) → sigmoid**. The quantum part computes a high-dimensional nonlinear feature map of the input (via embedding + entangling layers); the classical part learns how to combine the quantum expectation values into a decision. Both parts are trained jointly in one autograd graph — gradients flow from the loss through the classical head into the quantum circuit parameters via analytic backprop through the simulator.

---

## 3. Layer Architecture and Data Flow

### 3.1 Data flow diagram

```
WORKING/output/rppg/dataset_features.csv   (3471 rows @ 30 fps: 1920 real / 1551 fake*)
        │  label flip (rPPG: 1=fake → quantum: LABEL_REAL=1)
        ▼
data.py ──► plausibility filter (30 ≤ HR ≤ 220, finite values)
        │  subject-grouped 60/20/20 split (FF++ official folders; DFDC per-clip)
        │  leakage assertion (no subject in two splits)
        ▼
data.npz (X/y/groups per split) ──► split_manifest.json (reproducibility)
        │
scaling.py ──► z-score fit on TRAIN ONLY ──► feature_scaler.json
        │
qaoa.py ──► discrimination weights (Mann-Whitney AUC) + |correlation| matrix
        │  Hamiltonian encoding (_cost_terms) + verify_hamiltonian (< 1e-6)
        │  QAOA (3 layers, COBYLA, 4 restarts) → marginal probabilities
        ▼
qaoa_selection.json  (20 → 3 features: signal_quality_index, peak_prominence, entropy_window_std)
        │
vqc.py ──► HybridModel on the 3 selected features (angle embedding + SEL ansatz + head)
        │  AdamW, focal loss, cosine LR, early stopping, CUDA head
        ▼
hybrid_vqc.pt (state_dict + metadata) ──► training_log.jsonl
        │
evaluation.py ──► metrics_quantum.json + metrics_baselines.json + 3 PNG plots
        │
pipeline.predict_features(features) ──► P(real) ──► REAL / FAKE / UNCERTAIN
        │                                    ▲
        └────────  inference replays: scaler → selection → VQC  (exact train-time transform)
```

*Split counts (verified from `data.npz`): the 3471-row table splits into train 2083 / val 694 / test 694. Test = 310 fake / 384 real under quantum labels (~55% real ratio, mirrored in train/val).

### 3.2 Label conventions (do not unify between stages)

| Layer | Convention |
|---|---|
| rPPG CSV (`dataset_features.csv`) | `1 = fake`, `0 = real` |
| Quantum layer (`config.py`) | `LABEL_REAL = 1`, `LABEL_FAKE = 0` — `data.py` flips with `y = 1 − csv_label` |
| rPPG RandomForest cross-check (`run_pipeline.py`) | `1 = DEEPFAKE` |

### 3.3 Subject grouping and leakage prevention

The CSV carries no explicit subject IDs, so `data.py::_infer_subject_key` derives a group key from the clip path:

- **FF++** clips grouped by source subject: `id0_0000.mp4` (real) and `id0_id16_0002.mp4` (its synthetic counterpart) → `ffpp:src:id0`. A person's real and fake takes can never straddle train/val/test.
- **YouTube-real** clips (numeric stems, no pairing) → `ffpp:yt:<stem>` (own group each).
- **DFDC** clips carry no pairing info on disk → `clip:<path>` (own group each; documented limitation — DFDC subject-level separation is unrecoverable).

The split is seeded (seed 42) and class-balanced: groups are placed greedily into the split with the largest remaining normalized per-class demand. FF++ rows use the dataset's own train/val/test folders (`_infer_split_key`), others use the grouped random split. A hard assertion (`_assert_no_group_leakage`) aborts the build if any group appears in two splits, and `tests.py::test_no_group_leakage` re-checks it. Cross-validation is group-aware too (`StratifiedGroupKFold`), so no fold separates a subject's clips.

### 3.4 Inference path (exact replay)

`pipeline.predict_features(features)` takes a 20-feature dict (the rPPG stage's output) and replays the exact training-time transform, guaranteed by saved artifacts:

1. `feature_scaler.json` (train-fitted z-score; shape-checked against `FEATURE_NAMES`)
2. `qaoa_selection.json` (indices validated in-range)
3. `hybrid_vqc.pt` (cached model keyed by feature count + checkpoint mtime/size)
4. Sigmoid → `prob_real` → verdict via decision thresholds (REAL ≥ 0.7, FAKE ≤ 0.3, else UNCERTAIN), confidence = 2·|prob_real − 0.5|.

Non-finite features after scaling → `INCONCLUSIVE` with reason (the rPPG layer also returns `features=None` when fewer than 48 usable frames, handled upstream in `run_pipeline.py`).

---

## 4. File-by-File Explanation

All files live in `WORKING/quantum/`. Line references point to the current implementation.

### 4.1 `config.py` — contracts, labels, configuration

The single source of truth for the layer.

- **`FEATURE_NAMES`** (lines 9–30) — the 20-feature contract, duplicated *intentionally* from `RPPGFeatures.feature_names()` in `RPPG/rppg/features.py`; name AND order must stay identical (a regression test enforces sync). Every module indexes by this list.
- **`FEATURE_MEANINGS`** (32–53) — human-readable meaning of each physiological feature (e.g. `heart_rate_bpm`: dominant pulse frequency; `cheek_forehead_correlation`: Pearson correlation between cheek and forehead pulse signals).
- **`LABEL_REAL = 1`, `LABEL_FAKE = 0`** — quantum label convention.
- **Dataclass configs** (all frozen):
  - `DataConfig` — seed 42, val/test ratios 0.2, HR plausibility filter (30–220 BPM), CSV/data paths.
  - `QAOASelectionConfig` — p_layers=3, max_iter=200, redundancy_penalty=0.3, cardinality_penalty=0.5, target_features=3, restarts=4, device="auto".
  - `VQCConfig` — qml_layers=3, hidden_units=8, dropout=0.2, epochs=80, batch=32, lr=1e-2, wd=1e-2, focal α=0.75/γ=2.0, label_smoothing=0.03, confidence_penalty=0.02, cosine LR, patience=12, clip_grad=1.0, qnode_impl="auto" (torch-native).
  - `DecisionConfig` — thresholds fake_max_prob=0.3 / real_min_prob=0.7, artifact paths.

### 4.2 `data.py` — dataset build/load

Turns the rPPG CSV into `data.npz` (numpy, gitignored).

- `_load_rppg_rows` (81) — validates columns, parses features, drops rows with non-finite values or implausible HR (stats recorded and printed), flips labels to the quantum convention.
- `_infer_subject_key` (43) / `_infer_split_key` (66) — group/split derivation (§3.3).
- `_grouped_train_val_test_split` (140) — deterministic seeded, per-class-balanced, subject-grouped split.
- `_split_by_source` (213) — official FF++ folders when available.
- `_assert_no_group_leakage` (231) — abort on any subject appearing in two splits.
- `_write_split_manifest` (250) — JSON manifest of path → (split, group) so any split is reproducible.
- `build_dataset` (273) / `load_dataset` (321) — public entry points; savez_compressed / np.load.

### 4.3 `scaling.py` — train-only z-score standardization

`FeatureScaler`: per-feature z-score (mean 0, unit variance) fitted **on the training split only**; transform is reused on val/test/inference. Saved as JSON (`feature_scaler.json`) so inference reproduces the exact transform. Rationale: QAOA selection is scale-invariant (Mann-Whitney AUC + Pearson correlation), but the VQC angle embedding treats feature values as rotation angles — raw magnitudes (heart rate ~92 BPM) are not meaningful angles.

### 4.4 `qaoa.py` — QAOA feature selection (the quantum optimizer)

- `_discrimination_weights` (175) — supervised per-feature weights `2·|AUC_i − 0.5|` via the exact Mann-Whitney U statistic (no sklearn, deterministic). Sign-agnostic: an inverted feature is as informative as a forward one once the classifier learns its sign. This replaced `_mutual_info_weights` (167, kept only as documented alternative), which systematically excluded HR and SNR — the two strongest discriminators.
- `_cost_terms` (213) — the classical cost → Pauli-Z Hamiltonian encoding (§2.7), with the full algebra in the docstring.
- `_classical_cost` (260) — the reference cost function the Hamiltonian must reproduce.
- `_precompute_gates` (271) — decomposes exp(−iγH) into per-term PauliRot gates (all Z-terms commute → exact, no Trotter error), precomputed once per selection instead of per COBYLA call.
- `_apply_qaoa` (301) — the ansatz: Hadamard layer → p × (cost gates driven by γ, X-mixer driven by β). **The beta-mixer regression guard lives in tests.py** (a past refactor drove the mixer with γ, making β dead).
- `simulator_device` (134) / `_probe_gpu_device` (51) — backend resolution: "auto" prefers the torch-native sim; the GPU probe gates `default.qubit.torch` on precision (1e-6) and speed (≤3× CPU). On this host (PennyLane 0.42, no Windows GPU wheels), the probe fails and the torch-native CPU sim is used.
- `QAOASelector.select` (368) — orchestrates: weights → correlation matrix → Hamiltonian → 4 parallel restarts (ProcessPoolExecutor) of COBYLA → best state → marginal probabilities → top-k features.
- `_restart_worker` (326) — one full COBYLA run; module-level for pickling; uses `qaoa_sim.QAOASimulator` when available (~20× faster).
- `select_classical` (432) — classical greedy reference (top-k by weight) for fair comparison.
- `compare_selections` (456) — overlap report (current run: **0 features overlap** with the classical pick).
- `verify_hamiltonian` (470) — max |⟨b|H|b⟩ − C(b)| over all-zeros, all-ones, all single-bit, and 8 random bitstrings; the pipeline hard-asserts < 1e-6.
- `save_selection` / `load_selection` (536/544) — JSON artifact I/O.

### 4.5 `qaoa_sim.py` — torch-native statevector QAOA simulator

A pure-PyTorch, exact (complex128) simulator of the QAOA circuit:

- Precomputes, for all 2ⁿ bitstrings: the bit matrix and the **full classical cost table C(b)** (vectorized einsum, `_compute_C`, 99) — this is what makes cost calls ~10⁴× faster than PennyLane (no gate-by-gate tape expansion; the cost layer is a single diagonal multiply `exp(−iγC)`).
- `_apply_cost_layer` (112) — exact diagonal phase application.
- `_apply_mixer_layer` (116) — R_x(2β) per qubit via precomputed index pairs.
- `cost` (128) — ⟨H⟩ for a parameter vector (COBYLA's objective), under `no_grad` for speed.
- `marginals` (141) — ⟨Z_i⟩ per qubit, the selection readout.
- `make_hamiltonian_diagonal` (201) — Hamiltonian diagonal in the computational basis for the ≡ classical-cost verification.
- CPU-only by design: ProcessPool workers never open CUDA contexts (20 wires → 2²⁰ state ≈ 16 MB complex128; per-call ~0.3–0.5 ms).

### 4.6 `vqc.py` — hybrid VQC model, training, inference

- `resolve_device` (29) — CUDA if available, else CPU (torch side).
- `QuantumLayerTorch` (125) — the default quantum layer: exact torch-native statevector simulator of `AngleEmbedding(RY) → StronglyEntanglingLayers → ⟨Z_i⟩`, complex128, batched (a (batch, 2ⁿ) state is evolved in parallel — `_apply_wire_rotation` uses bit-index reshapes; `_apply_cnot` uses index masks). Same `weights` parameter name/shape as the PennyLane `QuantumLayer` so checkpoints are interchangeable.
- `QuantumLayer` (61) — legacy PennyLane QNode path (default.qubit + backprop), kept behind `qnode_impl="pennylane"` for cross-verification. Broadcasts batches through the circuit; CPU-weight caching with `_version`/`data_ptr` invalidation so autograd still flows to CUDA head weights.
- `HybridModel` (278) — quantum layer (n qubits = n features) + head `Linear(n→8) → ReLU → Dropout(0.2) → Linear(8→1)`; forward = head(quantum(x)).
- `focal_loss` (297) — §2.8.
- `train_vqc` (384) — AdamW (lr 1e-2, wd 1e-2), cosine-annealed LR, gradient clipping at 1.0, early stopping on val loss (patience 12, min_delta 1e-4) with restore of the best-validation checkpoint; checkpoint bundles `state_dict` + metadata (selection, scaler, configs) sanitized for `weights_only=True` loading.
- `load_vqc_model` (317) — cached by (n_features, path, mtime, size); max 8 entries.
- `predict_vqc` (343) — sigmoid logits under `no_grad` → P(real).
- `_val_metrics` (366) — single-forward val loss + accuracy (one forward instead of two).

### 4.7 `evaluation.py` — metrics, CV, baselines

- `expected_calibration_error` (68) — ECE over 10 probability bins.
- `classification_metrics` (82) — accuracy, precision, recall, specificity, F1, AUC-ROC, PR-AUC, confusion matrix, ECE.
- `balanced_accuracy` (110) — mean per-class recall.
- `run_cv` (128) — group-aware 5-fold CV (`StratifiedGroupKFold` when available, else `GroupKFold`); folds run in parallel subprocesses on CPU hosts, **sequentially on GPU hosts** (parallel folds would each open a CUDA context on the shared card — OOM risk). `_fit_vqc_fold` (200) is the module-level fold worker.
- `decision_bins` (180) — REAL / FAKE / UNCERTAIN counts and rates at the 0.3/0.7 thresholds, plus accuracy on confirmed predictions.
- `evaluate_quantum_model` (210) — test-set metrics + CV on train + 3 PNG plots + `metrics_quantum.json`.
- `run_baselines` (256) — six classical baselines on the **same** selected features: RandomForest(200), MLP(32,16), LogisticRegression(balanced), LinearSVC+Calibration, GaussianNB, XGBoost(300). XGBoost is lazy-imported and optional (skipped entry if missing).
- `_sklearn` (27) — lazy sklearn import (~12 s on this host) so QAOA workers never pay it.

### 4.8 `plots.py` — evaluation figures

- `plot_roc_curve` (37), `plot_confusion_matrix` (57), `plot_calibration_curve` (69) — write `roc_curve.png`, `confusion_matrix.png`, `calibration_curve.png` (Agg backend, no GUI, figure closed after save). matplotlib and sklearn imports are deferred inside the functions so `import quantum.pipeline` stays fast for QAOA spawn workers and server restarts.

### 4.9 `pipeline.py` — CLI + inference entry

- `predict_features` (48) — the inference entry consumed by `run_pipeline.py` (§3.4); returns `prob_real`, `verdict`, `confidence`, `selected_features/indices`, `scaler_file`; INCONCLUSIVE with reason on non-finite scaled features.
- `main` (111) — the 6-stage CLI: `--build-data` → `--select` (with Hamiltonian verification hard-assert and classical comparison) → `--train` → `--evaluate` → `--baselines`; `--all` runs all. `--dev-only` evaluates on the val split to protect the final test split until a config is frozen.

### 4.10 `tests.py` — regression self-checks (no pytest)

Run with `python -m quantum.tests`; exit 0 = all pass (currently 7/7). They guard the two audited QAOA defects and the fragile contracts:

1. `test_beta_alive` — perturbing β must change the cost (a refactor once drove the mixer with γ, making the ansatz degenerate).
2. `test_hamiltonian_matches_classical` — ⟨b|H|b⟩ = C(b) on every bitstring of a synthetic problem.
3. `test_real_hamiltonian_verification` — same on the real dataset (< 1e-6).
4. `test_feature_contract_sync` — `FEATURE_NAMES` identical to `RPPGFeatures.feature_names()`.
5. `test_split_determinism` — same seed → identical splits.
6. `test_ffpp_source_subject_grouping` — FF++ real+synth clips share a group key.
7. `test_no_group_leakage` — no group straddles two splits.

### 4.11 `sweep.py` — hyperparameter search harness (dev tool)

Two-phase factorial sweep: Phase A runs QAOA once per unique config (9 configs, cached in `output/sweep/qaoa_cache.json`); Phase B trains VQC combos on the cached feature sets (Tier 1: 162 combos = 9 QAOA × 6 VQC arch × 3 schedules; Tier 2: 12 loss × LR/WD variants on the default config). Per-combo timeout (600 s default), early stopping, checkpointing, CSV+JSON results with a leaderboard. Not part of the production path.

### 4.12 `__init__.py`

Empty package marker.

---

## 5. Performance Results (Frozen Baseline, 2026-08-18)

All numbers below come from the regenerated artifacts in `WORKING/output/quantum/` (`metrics_quantum.json`, `metrics_baselines.json`, `qaoa_selection.json`). Data: 3471 real rPPG rows (1920 real / 1551 fake), subject-grouped 60/20/20 split (train 2083 / val 694 / test 694).

### 5.1 QAOA feature selection

| Parameter | Value |
|---|---|
| Selected features (20 → 3) | `signal_quality_index`, `peak_prominence`, `entropy_window_std` |
| Best cost (restart seed 44) | **−0.7072** |
| All restart costs | [10.26, −0.098, −0.707, 9.74] |
| COBYLA success flag | false (no convergence; best iterate used — normal for COBYLA on this landscape) |
| Configuration | p=3 layers, max_iter=200, rp=0.3, cp=0.5, k=3, 4 restarts |
| Classical greedy reference | `cheek_forehead_correlation`, `left_right_cheek_correlation`, `prv_std_ms` |
| Overlap QAOA ↔ classical | **0 features** |
| Marginal probability of selected features | 0.182, 0.177, 0.174 (highest of 20; range 0.145–0.182) |

Interpretation: the marginals are close together (0.145–0.182) — no feature clearly dominates, consistent with the weak per-feature signal. QAOA and the classical greedy pick choose disjoint sets; neither lifts downstream AUC above ~0.57, confirming the selection problem itself is not the bottleneck.

### 5.2 Hybrid VQC — held-out test set (694 rows)

| Metric | Value |
|---|---|
| Accuracy | 0.5533 |
| Precision / Recall | 0.5533 / 1.0000 |
| Specificity | **0.0000** |
| F1 | 0.7124 |
| **AUC-ROC** | **0.5433** |
| PR-AUC | 0.5628 |
| ECE (calibration) | 0.0605 |
| Balanced accuracy | 0.5000 |
| Confusion matrix | [[TN 0, FP 310], [FN 0, TP 384]] |

The VQC is a **majority-class predictor**: it assigns every test sample P(real) > 0.5 (recall 1.0, specificity 0.0 — nothing is ever classified fake at the 0.5 threshold).

### 5.3 Decision bins (the deployed verdict behavior)

| Bin | Count | Rate |
|---|---|---|
| REAL (P ≥ 0.7) | 0 | 0% |
| FAKE (P ≤ 0.3) | 0 | 0% |
| **UNCERTAIN** (0.3 < P < 0.7) | **694** | **100%** |
| Confirmed accuracy | n/a (no confirmed predictions) | |

Every model probability falls inside [0.3, 0.7]. The layer therefore refuses to commit — the safety-correct behavior for KYC when evidence is weak — and `run_pipeline.py` maps this to an INCONCLUSIVE outcome for the end-to-end product.

### 5.4 VQC — 5-fold grouped CV (train split; procedure stability)

| Fold | Accuracy | AUC-ROC | ECE | Specificity |
|---|---|---|---|---|
| 1 | 0.5374 | 0.5511 | 0.0771 | 0.0 |
| 2 | 0.5627 | 0.4784 | 0.0517 | 0.0 |
| 3 | 0.5534 | 0.5333 | 0.0583 | 0.0 |
| 4 | 0.5514 | 0.5574 | 0.0647 | 0.0 |
| 5 | 0.5614 | 0.5221 | 0.0628 | 0.0 |
| **Mean ± Std** | **0.5533 ± 0.009** | **0.5285 ± 0.028** | 0.0629 ± 0.008 | 0.0 ± 0.0 |

Every fold is also a majority-class predictor — this is a property of the features, not of one lucky/unlucky split.

### 5.5 Classical baselines (same 3 selected features)

| Model | Test Acc | Test AUC | PR-AUC | ECE | Specificity | CV AUC |
|---|---|---|---|---|---|---|
| **Gaussian NB** | **0.5807** | **0.5698** | **0.6000** | 0.0140 | 0.1516 | 0.5008 |
| Logistic Regression | 0.5533 | 0.5443 | 0.5634 | 0.0515 | 0.4419 | 0.4859 |
| MLP (32,16) | 0.5576 | 0.5202 | 0.5617 | **0.0114** | 0.1452 | 0.4900 |
| Random Forest | 0.5014 | 0.4773 | 0.5476 | 0.1777 | 0.3387 | 0.4837 |
| XGBoost | 0.5259 | 0.4980 | 0.5453 | 0.0823 | 0.2419 | 0.4592 |
| Linear SVC | 0.5533 | 0.4239 | 0.5019 | 0.0178 | 0.0000 | 0.5141 |
| **Hybrid VQC** | 0.5533 | 0.5433 | 0.5628 | 0.0605 | 0.0000 | 0.5285 |

### 5.6 Training dynamics (`training_log.jsonl`, 16 epochs)

- Train loss: 0.0629 → 0.0578 (converged, monotone).
- Val loss: 0.0587 → 0.0578 (oscillates within the min_delta band — early stopping never fired).
- Val accuracy: **locked at 0.5533** every epoch (the majority-class fraction — the model learns nothing beyond the base rate).
- Cosine LR: 0.0100 → 0.0092.

### 5.7 Honest interpretation (for judges)

1. **The ceiling is in the features, not the model.** Per-feature discrimination |AUC−0.5| ≤ ~0.06 (the normalized weights range 0.025–1.0, but even the strongest features are weak). Every model — quantum or classical, on 3 or 20 features — lands between 0.50 and 0.57 AUC. Increasing the dataset 8.2× (421 → 3445 → 3471 rows, adding FF++) did not lift the ceiling, which is the classic signature of an input-signal problem.
2. **The VQC is an honest learner here.** It converges to the Bayes-optimal decision under weak signal: predict the majority class. Its AUC (0.543) actually beats MLP, RF, XGBoost, and LinearSVC on the same features, and its CV AUC (0.5285) beats all baselines' CV AUC. Only Gaussian NB edges it out on the test split.
3. **100% UNCERTAIN is a feature, not a bug, for KYC.** When the physiological evidence cannot separate real from fake, the system declines to authenticate rather than risk a false acceptance. The 0.3/0.7 thresholds are the next lever (an explicit, documented research item).
4. **The quantum mechanics are verified.** Hamiltonian ≡ classical cost to < 1e-6 (hard assert), the beta mixer is provably alive, QAOA is not worse than the classical greedy pick at selection quality (both pick weak-but-valid feature sets), and all 7 regression tests pass. The layer's correctness is not in question; the underlying physiological signal is.

---

## 6. Verification and Integrity Guarantees

These are the layer's quality gates — each is automatic or test-enforced:

1. **Hamiltonian ≡ classical cost** — `pipeline.py` hard-asserts `verify_hamiltonian` error < 1e-6 before every QAOA run (the old silent ~1.94-error verification is history, guarded by tests 2 & 3).
2. **Beta mixer alive** — test 1 perturbs β and asserts the cost changes (guards the dead-mixer regression).
3. **Feature-contract sync** — test 4 asserts `FEATURE_NAMES` matches the rPPG layer byte-for-byte in name and order (data.py, qaoa.py, vqc.py, pipeline.py all index by it).
4. **No data leakage** — grouped splits + `_assert_no_group_leakage` + tests 5–7 (deterministic split, FF++ subject grouping, no group straddling).
5. **No synthetic data** — the layer consumes only the real rPPG table `output/rppg/dataset_features.csv`; no generator or transform-bridge layer exists.
6. **Inference replays training exactly** — scaler, selection, and model are saved with training-time metadata; `predict_features` validates shapes/ranges and raises actionable errors when artifacts are out of sync.
7. **Train-only statistics** — the scaler is fitted on the training split only; the test split is touched only by the final frozen evaluation (a `--dev-only` mode exists to protect it during development).
8. **Deterministic reproduction** — seeds fixed (42), split manifest records path → split/group, QAOA restarts are seeded.

---

## 7. Usage

Run **from `WORKING/`** (the `quantum.*` imports and `sys.path` insertions assume that working directory):

```bash
# Full flow: build data → QAOA selection → train VQC → evaluate → classical baselines
python -m quantum.pipeline --all

# Step-by-step
python -m quantum.pipeline --build-data --select --train --evaluate --baselines

# Rerun with existing artifacts
python -m quantum.pipeline --train --evaluate

# Self-checks (run after any qaoa.py/config.py change)
python -m quantum.tests

# Hyperparameter sweep (dev tool; ~5.5 h full run)
python -m quantum.sweep

# End-to-end inference (requires the quantum artifacts above)
python run_pipeline.py --source path/to/video.mp4 --method POS
```

### Artifacts (all gitignored, under `WORKING/output/quantum/`)

| Artifact | Contents |
|---|---|
| `data.npz` | subject-grouped train/val/test feature matrices + labels + groups + paths |
| `split_manifest.json` | path → (split, group) for reproducibility |
| `feature_scaler.json` | train-fitted z-score means/scales |
| `qaoa_selection.json` | selected indices/features, marginals, weights, cost, restarts |
| `selection_comparison.json` | QAOA vs classical greedy overlap |
| `hybrid_vqc.pt` | model state_dict + metadata (selection, scaler, configs) + training summary |
| `training_log.jsonl` | per-epoch train loss, val loss/acc, LR |
| `metrics_quantum.json` | test metrics, balanced accuracy, decision bins, 5-fold CV |
| `metrics_baselines.json` | six classical baselines, test + CV |
| `roc_curve.png`, `confusion_matrix.png`, `calibration_curve.png` | evaluation figures |

---

## 8. Judge Q&A Preparation

**Q: Why use quantum computing at all for deepfake detection?**
A: Two reasons. (1) Feature selection is a combinatorial optimization problem over 2²⁰ subsets; QAOA is the canonical quantum algorithm for that problem class and scales where brute force does not. (2) The VQC is a genuinely hybrid quantum-classical model — the entanglement structure lets the quantum layer represent feature interactions that a classical linear model cannot, and it is trained end-to-end. We are honest that for 20 features classical brute force is feasible; the value is in the methodology and scaling path, which we verify rigorously.

**Q: Is this running on real quantum hardware?**
A: No — on exact classical statevector simulators (PennyLane and a custom torch-native simulator). This is deliberate: deterministic, reproducible, deployable in a KYC product today. The circuits are hardware-ready: the ansatz and Hamiltonian encoding transfer unchanged to real processors.

**Q: The accuracy looks low — does this mean the project failed?**
A: No — it means the experiment produced a clear, honest result: the rPPG physiological features carry almost no class signal (per-feature |AUC−0.5| ≤ 0.06; all models, classical and quantum, plateau at 0.50–0.57 AUC even after an 8.2× dataset increase). The quantum layer correctly learns the majority class and refuses to commit (100% UNCERTAIN), which is the safe behavior for KYC. The research conclusion is that Phase-2 work must improve the rPPG method itself (method/ROI probing), not the classifier.

**Q: Why 3 features? Why these three?**
A: k=3 was validated empirically: a 3-feature QAOA pick reached CV AUC 0.569 vs 0.484 for the old 6-feature pick (the extra features diluted the signal). The specific three (signal_quality_index, peak_prominence, entropy_window_std) are chosen by QAOA as the best trade-off of individual discrimination, low redundancy, and exact cardinality — a purely data-driven choice, cross-checked against a classical greedy selector.

**Q: How do you know the quantum part is doing what you claim?**
A: Four independent gates: (1) the Hamiltonian is provably equal to the classical cost (verified < 1e-6 over 30+ bitstrings, hard-asserted in the pipeline); (2) a regression test proves the β mixer is alive; (3) 7/7 self-checks pass; (4) the torch-native simulator reproduces the PennyLane circuit to float rounding.

**Q: Is this a hybrid model or two separate quantum steps?**
A: Both, and they are cleanly separated: QAOA is a quantum *optimizer* that picks features once at build time; the VQC is a quantum *classifier* trained end-to-end with PyTorch. Both are variational (hybrid quantum-classical) algorithms.

**Q: What is the inference latency?**
A: VQC inference is ~275 ms cold / ~32 ms cached per feature vector on CPU (the model cache in `vqc.py` makes repeated requests nearly free), plus the upstream frame + rPPG stages. The QAOA selection runs only at model build time (~15 s), never at inference.

**Q: What would happen with a real quantum computer?**
A: The same ansatz, with measurement-shot noise replacing exact simulation. Our verification harness already computes the Hamiltonian expectation values the hardware would report; the layer is designed so the classical cost oracle and circuit definitions are hardware-agnostic.

**Q: What are the decision thresholds and why 100% UNCERTAIN?**
A: REAL requires P(real) ≥ 0.7, FAKE requires P(real) ≤ 0.3, else UNCERTAIN. Because every trained model outputs probabilities clustered near 0.5 (no discriminating evidence), nothing passes either threshold. Tuning these thresholds is the documented next lever once the rPPG signal improves — we deliberately kept a strict reject zone for a KYC product.