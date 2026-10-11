"""Regression tests for QAOA feature-index mapping (defect D1).

D1: qaoa_selection_fused.json used to be persisted with selector-LOCAL
indices (into the classically pre-selected candidate matrix) while
predict_features() applied those indices to the full-width scaled feature
vector, so inference silently consumed different features than training.

These tests pin the invariant: the persisted selection must be expressed in
the ORIGINAL feature space, and the columns selected at inference time must
be identical to the columns used during training.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

WORKING_ROOT = str(Path(__file__).resolve().parents[2] / "WORKING")
if WORKING_ROOT not in sys.path:
    sys.path.insert(0, WORKING_ROOT)

from quantum.config import FUSED_FEATURE_NAMES, FEATURE_NAMES, QAOASelectionConfig
from quantum.qaoa import (
    QAOASelector,
    load_selection,
    remap_selection_to_original_space,
    save_selection,
    select_classical,
)


def _synthetic_full(n_samples=60, n_features=59, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.rand(n_samples, n_features)
    y = np.asarray([0] * (n_samples // 2) + [1] * (n_samples - n_samples // 2))
    return X, y


def test_fused_contract_is_59_features():
    assert len(FUSED_FEATURE_NAMES) == 59


def test_remap_maps_local_indices_to_original_space():
    full_names = [f"feat_{i}" for i in range(59)]
    preselect = [5, 17, 42, 0, 58, 3]  # local columns -> original indices
    local_names = [full_names[i] for i in preselect]
    selection = {
        "selected_indices": [0, 2, 4],
        "selected_features": [local_names[i] for i in [0, 2, 4]],
        "restarts": {
            "selection_stability": {
                "selection_frequency": {local_names[0]: 1.0, local_names[4]: 0.5},
            }
        },
    }
    remap_selection_to_original_space(selection, preselect, full_names)
    assert selection["selected_indices"] == [5, 42, 58]
    assert selection["selected_features"] == ["feat_5", "feat_42", "feat_58"]
    freq = selection["restarts"]["selection_stability"]["selection_frequency"]
    assert freq == {"feat_5": 1.0, "feat_58": 0.5}
    assert selection["restarts"]["index_space"] == "original_feature_space"


def test_remap_is_identity_without_preselect():
    selection = {"selected_indices": [7, 1, 3], "selected_features": ["a", "b", "c"]}
    out = remap_selection_to_original_space(selection, None, ["a"] * 10)
    assert out["selected_indices"] == [7, 1, 3]
    assert out["selected_features"] == ["a", "b", "c"]
    assert "index_space" not in out.get("restarts", {})


def test_select_uses_provided_feature_names():
    rng = np.random.RandomState(1)
    X = rng.rand(40, 6)
    y = np.asarray([0] * 20 + [1] * 20)
    cfg = QAOASelectionConfig(target_features=3, restarts=1, max_iter=12, p_layers=2)
    names = [f"col_{i}" for i in range(6)]
    res = QAOASelector(cfg).select(X, y, feature_names=names)
    assert res["selected_features"] == [names[i] for i in res["selected_indices"]]
    stability = res["restarts"]["selection_stability"]["selection_frequency"]
    assert all(k.startswith("col_") for k in stability)


def test_select_rejects_feature_names_length_mismatch():
    rng = np.random.RandomState(1)
    X = rng.rand(40, 6)
    y = np.asarray([0] * 20 + [1] * 20)
    cfg = QAOASelectionConfig(target_features=3, restarts=1, max_iter=8, p_layers=2)
    with pytest.raises(ValueError):
        QAOASelector(cfg).select(X, y, feature_names=["only", "five", "names", "here", "x"])


def test_select_default_names_stay_backward_compatible():
    rng = np.random.RandomState(2)
    X = rng.rand(40, 5)
    y = np.asarray([0] * 20 + [1] * 20)
    cfg = QAOASelectionConfig(target_features=2, restarts=1, max_iter=8, p_layers=2)
    res = QAOASelector(cfg).select(X, y)  # legacy default: RPPG alias covers 5 cols
    assert res["selected_features"] == [FEATURE_NAMES[i] for i in res["selected_indices"]]
    # wider-than-default X without explicit names must fail loudly (sweep.py path)
    X_wide = np.random.RandomState(3).rand(40, 25)
    with pytest.raises(ValueError):
        QAOASelector(cfg).select(X_wide, y)


def test_training_and_inference_use_identical_columns(tmp_path):
    """THE D1 regression: persisted selection must index the same columns
    that training used, when applied to the full-width feature vector."""
    X_full, y = _synthetic_full()
    full_names = list(FUSED_FEATURE_NAMES)

    # --- mirror quantum/pipeline.py training flow ---
    pre_cfg = QAOASelectionConfig(target_features=18)  # pre-select 59 -> 18
    classical_pre = select_classical(X_full, y, pre_cfg, feature_names=full_names)
    classical_pre = classical_pre["selected_indices"][:18]
    X_local = X_full[:, classical_pre]
    local_names = [full_names[i] for i in classical_pre]

    sel_cfg = QAOASelectionConfig(target_features=3, restarts=1, max_iter=12, p_layers=2)
    selection = QAOASelector(sel_cfg).select(X_local, y, feature_names=local_names)
    train_local_idx = list(selection["selected_indices"])

    selection = remap_selection_to_original_space(selection, classical_pre, full_names)
    sel_file = tmp_path / "qaoa_selection_test.json"
    save_selection(selection, sel_file)

    # --- mirror predict_features() inference flow ---
    loaded = load_selection(sel_file)
    infer_idx = [int(i) for i in loaded["selected_indices"]]
    assert all(0 <= i < len(full_names) for i in infer_idx), (
        "persisted indices must be valid positions in the full feature vector"
    )
    assert loaded["selected_features"] == [full_names[i] for i in infer_idx]

    # The columns inference picks from the full scaled vector must be exactly
    # the columns training used from the pre-selected matrix.
    np.testing.assert_array_equal(
        X_full[:, infer_idx],
        X_local[:, train_local_idx],
        err_msg="D1: inference feature columns differ from training feature columns",
    )


def test_predict_features_range_guard_accepts_remapped_indices(tmp_path):
    """The exact range predicate used by predict_features() must accept a
    remapped selection persisted against the 59-feature schema."""
    full_names = list(FUSED_FEATURE_NAMES)
    preselect = list(range(0, 54, 3))  # 18 candidates
    selection = {
        "selected_indices": [1, 2, 17],  # local
        "selected_features": ["x"] * 3,
    }
    remap_selection_to_original_space(selection, preselect, full_names)
    f = tmp_path / "sel.json"
    save_selection(selection, f)
    indices = [int(i) for i in load_selection(f)["selected_indices"]]
    # predict_features():  if not indices or any(i < 0 or i >= len(feature_names))
    assert indices and all(0 <= i < len(full_names) for i in indices)
