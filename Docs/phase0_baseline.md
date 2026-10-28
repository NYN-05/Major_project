# Phase 0 – Leak-aware baseline (2026-10-28)

Protocol: `Scrape/phase0_protocol.py` (grouped 5-fold CV x 3 seeds, threshold 0.5,
n = 3,912: 2,086 real / 1,826 fake after the HR filter; majority-class accuracy 0.533).
Full table: `Scrape/output/phase0/phase0_baseline.md` (JSON alongside).

## Findings

1. **The 0.569 is a feature-funnel problem, not a model-family ceiling.** QAOA keeps 3 of 59
   features (`lbp_uniform_hist_0`, `deep_feat_5`, `lbp_uniform_hist_9`). Classical models on all
   59 features reach 0.76-0.82 accuracy (AUC 0.83-0.89) under the project's clip-level grouping.
2. **That 0.76-0.82 is optimistic (leakage).** DFDC fakes are derived from reals, and no
   `metadata.json` ships with this copy, so source/actor ids are unrecoverable and the split
   groups per clip. Re-grouping by visual-embedding clusters (heuristic proxy for same-source
   clips; 47% of multi-clip clusters mix real and fake) drops RandomForest fused from 0.794 to
   0.680 and visual-only from 0.822 to 0.684. Treat ~0.68 as the honest current ceiling.
3. **rPPG carries almost no signal on this data:** 0.54-0.56 accuracy / AUC ~0.55-0.57 alone,
   and fused is no better than visual-only (clip 0.794 vs 0.822; proxy 0.680 vs 0.684).
4. **Sanity:** label-permutation scores 0.504 accuracy / AUC ~0.50, so the CV code is not
   leaking labels.

## Implications for the 0.95 plan

- Phase 1 (learned visual features) is where gains will come from; Phase 3 (rPPG) is low value.
- All later results must be reported under the proxy-source grouping (or real source ids if
  DFDC `metadata.json` can be obtained - this would remove the heuristic).
- The QAOA 3-feature funnel should be widened in Phase 4.

## Code fixes made in this phase

- `quantum/evaluation.py:_aggregate_cv` (T7): CV confusion matrix is now pooled element-wise
  (`confusion_matrix_pooled`) instead of averaged to a scalar; `None` AUCs no longer crash it.
- `quantum/evaluation.py:evaluate_quantum_model` (T6): adds `metrics_at_deployed_threshold` and
  `deployed_threshold` (tau*); confusion-matrix plot now uses tau*. `metrics` stays at 0.5 so
  baseline comparisons are unchanged.
