# Accuracy Improvement Plan (2026-11-13)

Owner approvals recorded in this document:
- **Approved:** update `AGENTS.md` to set `target_features = 8`.
- **Approved:** Step 5 FAR target **≤ 0.10** (validation-set, FAR-constrained threshold).
- **Deferred:** Phase 3 (deep PCA 64 + layer3 GPU re-extraction) — revisit later.
- Test set (782 frozen clips) stays untouched; all tuning runs on development folds only.

Goal: maximize model accuracy while strictly minimizing code changes — bug fixes,
config/hyperparameter tuning, loss and threshold adjustments first; no architectural
rewrites, no synthetic data, no dataset change (DFDC only, fused mode only).

---

## Bottleneck analysis (verified in code)

### Confirmed bugs

1. **Calibrated threshold loaded but never applied at inference.**
   `WORKING/quantum/pipeline.py:115` loads `opt_threshold` from checkpoint metadata,
   then `:118` builds `decision_cfg = DecisionConfig()` (hard default 0.5) and passes
   that to `build_explanation` (`:119`). `opt_threshold` is dead code. Every deployed
   verdict is cut at 0.5 despite README/AGENTS claiming τ* is consumed at inference.
   Evaluation (`WORKING/quantum/evaluation.py:390-406`) applies it correctly, so
   reported "deployed threshold" metrics do not match actual inference behavior.

2. **QAOA feature selection is a no-op.**
   `WORKING/quantum/pipeline.py:178` calls `select_classical()` intending 18 pre-select
   candidates, but `WORKING/quantum/qaoa.py:414` returns only
   `order[: cfg.target_features]` = 3 features. So `pre_select_k = min(18, 3) = 3` and
   QAOA selects 3 from a pool of 3. The model is effectively "classical top-3" (matches
   the Stage 3 finding: QAOA ran on only 3 candidate features in every run).

### Structural bottlenecks (ranked by measured impact)

| # | Bottleneck | Evidence | Fix cost |
|---|-----------|----------|----------|
| 1 | 3-feature regime (`config.py:246`) | Stage 4 dev nested CV: top-3 AUC 0.566–0.589 → top-8 0.594–0.656 → top-12 0.655–0.676; Stage 6 test: QAOA(8)→VQC 0.699 vs k=3 run 0.658 | 1 line (config) |
| 2 | VQC seed instability (seed SD 0.023 AUC), 68 params at k=3 | Stage 3 table; Stage 6 averaged 3 seeds for RF, VQC single-seed | ~30 lines (ensemble) |
| 3 | Threshold criterion is Youden, not KYC-cost — VQC test FAR = 0.556 | Stage 6 table: VQC FAR/FRR 0.556/0.194 | few lines |
| 4 | Calibration — VQC ECE 0.080 vs MLP 0.027 | README metrics table | ~20 lines (optional) |
| 5 | Training config — stopped at 23 epochs, lr 5e-2, focal α=0.45/γ=1.0, hidden 8 | `config.py:262-293` | config-only |
| 6 | Feature set — deep PCA 16, no layer3 (Stage 4 C1+C2 = +0.059 AUC, accepted) | Stage 4 results table | **deferred** (GPU re-extraction) |

### Explicitly out of scope

- **Data augmentation** — AGENTS.md prohibits synthetic data/generators outright.
- **RF verdict swap** (+0.11 AUC, biggest single lever) — violates "final verdict comes
  from the quantum stage"; frozen decision in `Docs/final_improvement_plan.md:8-9`.
  Needs an owner reversal, not a code change.
- **Phase 3 feature expansion** — deferred by owner (GPU re-extraction).

---

## Phase 0 — Correctness fixes (everything downstream depends on them)

### Step 1: Apply τ* at inference
- **File/lines:** `WORKING/quantum/pipeline.py:118-122`
- **Change:** `decision_cfg = replace(DecisionConfig(), decision_threshold=float(opt_threshold))`
  (import `replace` from `dataclasses`; same pattern as `evaluation.py:394`).
- **Rationale:** Aligns inference with the documented contract and with already-reported
  metrics. τ*=0.47942 < 0.5 balances the FAR/FRR pair Youden was chosen for.
- **Trade-off:** Changes deployed verdicts vs. current buggy behavior — quantify with one
  before/after end-to-end `run_pipeline.py` run on the same video. No test-set access.

### Step 2: Fix the QAOA pre-select truncation
- **Files/lines:** `WORKING/quantum/qaoa.py:403-414` (add optional `k` param:
  `k = cfg.target_features if k is None else int(k)`, use `order[:k]`);
  `WORKING/quantum/pipeline.py:178` (pass `k=pre_select_k`).
- **Change:** ~4 lines total.
- **Rationale:** Restores the architecture AGENTS.md describes; without it "QAOA
  selection" is a label on a classical top-3 pick. Value compounds with Step 3.
- **Trade-off:** Stage 4/5 showed QAOA top-k ≤ classical top-k at the same k, so this
  fix alone may not raise accuracy — it makes the step real. 18-qubit pre-select is the
  designed path (`pre_select_k = 18` already exists); 8 COBYLA restarts already exist.

**Phase 0 gate:** `python -m quantum.tests` (keep 13/13; add tests asserting
`predict_features` uses `metadata["decision_threshold"]` and that `select_classical`
honors `k`), then one `run_pipeline.py` end-to-end run. Log to `changes.md`.

---

## Phase 1 — Config-only tuning (zero architectural change)

All tuning on **development folds only** (`DataConfig.split_manifest` + `val_fold`,
`--dev-only`). The frozen test set was accessed once (Stage 6); no second look.

### Step 3: Raise `target_features` 3 → 8 (APPROVED)
- **File/line:** `WORKING/quantum/config.py:246` (`target_features: int = 3` → `8`)
- **Docs:** update `AGENTS.md` (approved) and the matching README constraint line so no
  stale "is 3" guidance remains.
- **Rationale:** Largest measured lever inside the quantum-verdict rule: +0.03–0.09 AUC
  in Stage 4/5, +0.04 AUC in Stage 6 (0.699 vs 0.658); grows VQC 68 → ~323 params.
  8 qubits keeps the statevector sim fast.
- **Trade-offs:** checkpoint/scaler/selection artifacts must be regenerated
  (`python -m quantum.pipeline --all`); inference validates index ranges against them.
  Not 12 because Stage 5: VQC at k=8 already trails classical −0.064; k=12 costs ~4×
  statevector time for +0.02 dev AUC.

### Step 4: Hyperparameter sweep of `VQCConfig` via the existing harness
- **Files:** driver → `Scrape/phase1_hparam_sweep.py` (harness code stays in `Scrape/`
  per repo rules); knobs in `WORKING/quantum/config.py:262-293`; existing helpers
  `quantum/evaluation.py:649` (`nested_group_cv`), `:681`
  (`make_grid_search_fit_predict`), `:712` (`make_random_search_fit_predict`).
- **Search space (priority order):**
  1. `epochs` 80→150, `patience` 12→20 (training stopped at 23 epochs — likely
     undertrained after warmup)
  2. `learning_rate` 5e-2 → {1e-2, 3e-2} with `cosine_warmup`
  3. `alpha` 0.45 → {0.40, 0.50, 0.55}, `gamma` 1.0 → {1.0, 2.0} (focal loss shape;
     steers FAR/FRR balance)
  4. `hidden_units` {8, 16}, `dropout` {0.1, 0.2}, `label_smoothing` {0, 0.03}
- **Rationale:** all fields already exist and flow into training (`vqc.py:339` focal
  loss, `vqc.py:443` train loop) — pure value changes. Expected: recover 0.01–0.03 AUC
  and cut seed variance (Stage 3 stability gate failed at SD 0.023).
- **Trade-offs:** cap at ~10–15 random configs (`make_random_search_fit_predict`,
  equal budget, grouped inner split). **Must not** read test metrics during search.

### Step 5: KYC-cost threshold — FAR ≤ 0.10 (APPROVED target)
- **Files/lines:** `WORKING/quantum/evaluation.py:298` (add FAR-constrained criterion
  beside `optimal_threshold_youden`; the non-Youden branch at `:292` already computes
  balanced accuracy); consumed at `WORKING/quantum/pipeline.py:272`.
- **Change:** new method (e.g. `far_constrained`): smallest threshold whose validation
  FAR ≤ 0.10; fallback to Youden when no threshold satisfies the constraint.
- **Rationale:** Youden optimizes sensitivity+specificity, ignoring that in KYC a false
  accept is the expensive error. Current VQC test FAR = 0.556 — dangerously permissive.
- **Trade-off:** FAR ↓ forces FRR ↑ (more genuine users rejected). Threshold stays
  validation-only (no test touch).

**Phase 1 gate:** `python -m quantum.tests` + `--all` retrain + `--dev-only`
evaluation; record dev-fold AUC/balanced-acc/FAR/FRR in `changes.md` and `Docs/`.

---

## Phase 2 — Small, targeted code additions

### Step 6: 3-seed ensemble at inference
- **Files:** `quantum/config.py` (seed list), `quantum/pipeline.py:62-122`
  (`predict_features`: load 3 checkpoints, average `prob_real`), `quantum/vqc.py`
  (train seeded — run `train_vqc` 3× with seeds 42/43/44).
- **Rationale:** Stage 6 already averages 3 seeds for RF; VQC is the only single-seed
  path. Seed SD 0.023 → ensemble compresses by ~√3 ≈ 0.013.
- **Trade-offs:** 3× training time (one-off), 3× checkpoints (`.sha256` sidecars per
  `write_sha256`), ~3× inference latency (still seconds, under the 30-min job timeout).
  Update `load_vqc_model` cache key and checkpoint-compat test.

### Step 7 (optional): Temperature scaling on validation probabilities
- **Files:** fit scalar T at `quantum/pipeline.py:267-280` (where τ* is computed
  post-training), persist in checkpoint metadata; apply in `predict_features`.
- **Rationale:** ECE 0.080 vs 0.027 — poorly calibrated probabilities make Step 5's
  threshold less stable across retrains. ~20 lines, validation-only fit.
- **Trade-off:** marginal accuracy change (calibration ≠ discrimination); mainly
  improves confidence display and threshold robustness. Skip if time-constrained.

---

## Phase 3 — Feature expansion (DEFERRED by owner)

Deep PCA 16 → 64 + layer3 GAP (Stage 4 C1+C2, +0.059 AUC, 5/5 folds) — requires full
visual re-extraction (GPU, hours), a `FUSED_FEATURE_NAMES` contract change (59 → 86+),
and persisted pooled-PCA fit for train/inference parity. Revisit later. Would be
implemented in `visual/extractor.py`, `quantum/config.py:109-139`,
`run_pipeline.py:233`, plus `test_feature_contract_sync`.

---

## Phase 4 — Owner decisions (flagged, not scheduled)

| Option | Gain | Blocker |
|---|---|---|
| RF/123-feature verdict swap | 0.812 vs 0.699 AUC (Stage 6) | AGENTS.md "verdict from quantum stage" + frozen decision to keep it |
| Data augmentation | unknown | AGENTS.md "no synthetic data" prohibition |
| Recover 181 rPPG-dropped clips (77% fake) | better FAKE recall, honest coverage | no code change (gates already params at `RPPG/pipeline.py:130-131`) but full re-extraction |

---

## Expected outcome

```
Phase 0 (session 1)   → correctness; tests green; no accuracy claim yet
Phase 1 Steps 3-4     → +0.03–0.09 AUC (k=8) + stability from tuning
Phase 1 Step 5        → FAR 0.556 → ≤0.10 on validation (KYC-critical number)
Phase 2 Step 6        → −0.013 seed SD, small AUC bump
Phase 3 (deferred)    → +0.059 AUC when re-extraction runs
```

Realistic target on own protocols: dev AUC ~0.72 → ~0.78 for the quantum route —
still short of the RF's 0.812, the honest ceiling unless the verdict-source rule is
revisited.

## Cross-cutting constraints

- `FUSED_FEATURE_NAMES` order-sync guarded by `test_feature_contract_sync`; fused-only;
  DFDC-only; label conventions untouched.
- New harness code lives in `Scrape/` only.
- Test metrics are never read during tuning (dev folds only).
- Each phase ends with `python -m quantum.tests` + a `changes.md` entry
  (`| YYYY-MM-DD HH:mm IST | Area | ... | Complete |`).

## Validation commands

```powershell
# From WORKING/
python -m quantum.tests                 # contract/regression suite (13+ tests)
python -m quantum.pipeline --all        # retrain fused artifacts after config changes
python -m quantum.pipeline --all --dev-only   # dev-only evaluation (no test touch)
python run_pipeline.py --source <video> --method POS --out <json>   # end-to-end
```
