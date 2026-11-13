"""Self-checks for the quantum layer (no pytest dependency).

Run from WORKING/:  python -m quantum.tests
Exit code 0 = all checks passed. Each check is a plain function that
raises AssertionError on failure; main() reports results.

These guard the two audited QAOA defects (dead beta mixer, Hamiltonian
!= classical cost), the duplicated feature contract, and dataset
determinism.

Additional setup: WORKING directory must be in sys.path for quantum imports.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure WORKING root is importable for quantum.* modules
WORKING_ROOT = str(Path(__file__).resolve().parent.parent)
if WORKING_ROOT not in sys.path:
    sys.path.insert(0, WORKING_ROOT)

import numpy as np

from quantum.config import FEATURE_NAMES, DataConfig, QAOASelectionConfig, get_dfdc_dataset_path
from quantum.data import SPLITS
from quantum.qaoa import (
    _classical_cost,
    _cost_terms,
    _make_circuits,
    _normalize_weights,
    _discrimination_weights,
    verify_hamiltonian,
)

RPPG_DIR = Path(__file__).resolve().parent.parent / "RPPG"
if str(RPPG_DIR) not in sys.path:
    sys.path.insert(0, str(RPPG_DIR))


def _synthetic_problem(n=6, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.rand(20, n)
    y = np.asarray([0] * 10 + [1] * 10)
    weights = _normalize_weights(_discrimination_weights(X, y, 42))
    correlation = np.abs(np.corrcoef(X.T))
    return weights, correlation, QAOASelectionConfig()


def test_beta_alive():
    """Perturbing the beta slots must change the QAOA cost.

    Regression guard: the precompute-gates refactor drove the X-mixer
    with gamma, making beta dead (cost bit-identical under beta
    perturbation). See PROJECT_AUDIT_REPORT.md section 6.1.
    """
    weights, correlation, cfg = _synthetic_problem()
    coeffs, ops = _cost_terms(weights, correlation, cfg)
    cost_circuit, _ = _make_circuits(coeffs, ops, len(weights))
    rng = np.random.RandomState(1)
    params = rng.uniform(0.0, 0.3, size=2 * cfg.p_layers)
    perturbed = params.copy()
    for layer in range(cfg.p_layers):
        perturbed[2 * layer + 1] = 1.5 + layer
    base = float(cost_circuit(params))
    changed = float(cost_circuit(perturbed))
    assert abs(changed - base) > 1e-3, (
        f"beta slots are dead (cost {base:.6f} -> {changed:.6f}); "
        "the X-mixer must be driven by beta, not gamma"
    )


def test_hamiltonian_matches_classical():
    """expval(H) must equal _classical_cost on every bitstring.

    Regression guard: the original _cost_terms derivation did not
    reproduce _classical_cost (verify_hamiltonian printed ~1.94 and
    never failed). See PROJECT_AUDIT_REPORT.md section 6.2.
    """
    weights, correlation, cfg = _synthetic_problem(n=8)
    coeffs, ops = _cost_terms(weights, correlation, cfg)

    import pennylane as qml

    dev = qml.device("default.qubit", wires=len(weights))

    @qml.qnode(dev)
    def basis_cost(bitstring):
        for i, bit in enumerate(bitstring):
            if bit:
                qml.PauliX(wires=i)
        return qml.expval(qml.Hamiltonian(coeffs, ops))

    rng = np.random.RandomState(7)
    states = [np.zeros(len(weights), dtype=int), np.ones(len(weights), dtype=int)]
    states += [rng.randint(0, 2, size=len(weights)) for _ in range(32)]
    for bitstring in states:
        quantum = float(basis_cost(bitstring))
        classical = _classical_cost(bitstring, weights, correlation, cfg)
        assert abs(quantum - classical) < 1e-6, (
            f"bitstring {bitstring}: quantum={quantum:.6f} classical={classical:.6f}"
        )


def test_real_hamiltonian_verification():
    """verify_hamiltonian must be ~0 on the real fused feature table.
    
    Uses classical pre-selection (like the pipeline) to limit to 18 features,
    avoiding 2^29 memory explosion in make_hamiltonian_diagonal.
    
    Skipped if dataset CSV not found.
    """
    csv_file = DataConfig().csv_file.parent.parent / "visual" / "fused_features.csv"
    if not csv_file.exists():
        import pytest
        pytest.skip(f"fused features CSV not found at {csv_file}")
    from quantum.data import _load_feature_rows, FEATURE_SETS
    from quantum.qaoa import select_classical

    feature_names = FEATURE_SETS["fused"]
    X, y, _, _, _, _ = _load_feature_rows(csv_file, feature_names, DataConfig())
    # Feature count for fused mode
    assert X.shape[1] == len(feature_names) == 59
    assert set(y.tolist()) <= {0, 1}
    
    # Classical pre-selection to 18 features (pipeline default)
    classical_pre = select_classical(X, y, QAOASelectionConfig(), feature_names=feature_names)
    classical_pre = classical_pre["selected_indices"][:18]
    X_pre = X[:, classical_pre]
    
    error = verify_hamiltonian(X_pre, y, QAOASelectionConfig())
    assert error < 1e-6, f"verify_hamiltonian error on real data: {error:.2e}"


def test_feature_contract_sync():
    """FEATURE_NAMES (quantum/config.py) must match RPPGFeatures order."""
    try:
        from RPPG.features import RPPGFeatures
    except ImportError as exc:
        raise AssertionError(f"cannot import RPPG.features: {exc}")
    expected = list(RPPGFeatures.feature_names())
    assert list(FEATURE_NAMES) == expected, (
        f"feature contract drift:\n  quantum/config.py: {list(FEATURE_NAMES)}\n"
        f"  RPPGFeatures:      {expected}"
    )


def test_split_determinism():
    """Same seed -> identical split; per-class counts preserved; ratios honored."""
    from quantum.data import _grouped_train_val_test_split
    from quantum.config import DataConfig

    rng = np.random.RandomState(3)
    X = rng.rand(16, 10)
    y = np.asarray([1] * 8 + [0] * 8)
    groups = np.asarray([f"g{i}" for i in range(16)], dtype=object)
    paths = np.asarray([f"p{i}.mp4" for i in range(16)], dtype=object)
    cfg = DataConfig()
    first = _grouped_train_val_test_split(X, y, groups, paths, cfg)
    second = _grouped_train_val_test_split(X, y, groups, paths, cfg)
    for s in ("train", "val", "test"):
        np.testing.assert_array_equal(first[f"X_{s}"], second[f"X_{s}"])
        np.testing.assert_array_equal(first[f"y_{s}"], second[f"y_{s}"])
    total = 0
    for s in ("train", "val", "test"):
        assert set(first[f"y_{s}"].tolist()) <= {0, 1}
        total += len(first[f"y_{s}"])
    assert total == 16
    assert len(first["X_train"]) > 0 and len(first["X_test"]) > 0
    for c in (0, 1):
        per_class_total = sum(int((first[f"y_{s}"] == c).sum()) for s in ("train", "val", "test"))
        assert per_class_total == int((y == c).sum()), f"class {c} rows lost in the split"
        for s, ratio in (("val", cfg.val_ratio), ("test", cfg.test_ratio)):
            n_s = int((first[f"y_{s}"] == c).sum())
            expected = int(round(int((y == c).sum()) * ratio))
            assert n_s == expected, (
                f"class {c} {s} count {n_s} != round(per_class * {ratio}) = {expected}"
            )


def test_dfdc_subject_grouping():
    """DFDC clips are grouped per-clip (no subject info available on disk)."""
    from quantum.data import _infer_subject_key
    import os

    # Test DFDC grouping
    dfdc1 = {"video_path": r"DFDC_Dataset\Fake\video1.mp4"}
    dfdc2 = {"video_path": r"DFDC_Dataset\Real\video2.mp4"}
    
    # Each clip is its own group
    assert _infer_subject_key(dfdc1) == "clip:" + dfdc1["video_path"].replace("\\", "/").lower()
    assert _infer_subject_key(dfdc2) == "clip:" + dfdc2["video_path"].replace("\\", "/").lower()
    # Different clips get different groups
    assert _infer_subject_key(dfdc1) != _infer_subject_key(dfdc2)

    # Test DFDC grouping only if dataset path is available
    if os.environ.get("DFDC_DATASET_PATH"):
        from quantum.config import get_dfdc_dataset_path
        dfdc_root = get_dfdc_dataset_path()
        fake_dir = dfdc_root / "Fake"
        fake_files = list(fake_dir.rglob("*.mp4")) if fake_dir.exists() else []
        if fake_files:
            dfdc = {"video_path": str(fake_files[0])}
            assert _infer_subject_key(dfdc) == "clip:" + dfdc["video_path"].replace("\\", "/").lower()


def test_no_group_leakage():
    """A subject group must never straddle train/val/test; assertion fires if so."""
    from quantum.data import (
        _assert_no_group_leakage,
        _grouped_train_val_test_split,
        _infer_subject_key,
    )

    rng = np.random.RandomState(9)
    rows = []
    # Use DFDC-style paths for testing
    for i in range(4):
        rows.append({"video_path": f"DFDC_Dataset/Fake/video{i}.mp4"})
        rows.append({"video_path": f"DFDC_Dataset/Real/video{i+4}.mp4"})
    groups = np.asarray([_infer_subject_key(r) for r in rows], dtype=object)
    X = rng.rand(len(rows), 10)
    y = np.asarray([1, 0, 1, 0, 1, 0, 1, 0])  # quantum: 1 real, 0 fake
    paths = np.asarray([r["video_path"] for r in rows], dtype=object)
    split = _grouped_train_val_test_split(X, y, groups, paths, DataConfig())
    _assert_no_group_leakage(split)  # must not raise

    bad = {f"groups_{s}": split[f"groups_{s}"].copy() for s in ("train", "val", "test")}
    bad["groups_train"][0] = "leaked_group"
    bad["groups_test"][0] = "leaked_group"
    try:
        _assert_no_group_leakage(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("_assert_no_group_leakage did not fire on a straddling group")


def test_qaoa_sim_matches_pennylane():
    """Torch-native QAOA simulator must reproduce PennyLane cost/marginals."""
    from quantum.qaoa_sim import QAOASimulator

    weights, correlation, cfg = _synthetic_problem(n=8)
    coeffs, ops = _cost_terms(weights, correlation, cfg)
    n_wires = len(weights)

    sim = QAOASimulator(n_wires, cfg, weights=weights, correlation=correlation)
    cost_circuit, marginals_circuit = _make_circuits(coeffs, ops, n_wires)

    rng = np.random.RandomState(5)
    for _ in range(4):
        params = rng.uniform(0.0, 0.5, size=2 * cfg.p_layers)
        qml_cost = float(cost_circuit(params))
        sim_cost = sim.cost(params)
        assert abs(qml_cost - sim_cost) < 1e-6, (
            f"QAOA sim cost {sim_cost:.9f} != PL {qml_cost:.9f} at params {params}"
        )
        pl_marginals = np.asarray(marginals_circuit(params)).flatten()
        sim_marginals = sim.marginals(params)
        diff = np.abs(pl_marginals - sim_marginals).max()
        assert diff < 1e-6, f"QAOA sim marginals differ from PL: max |diff| = {diff:.3e}"


def test_torch_layer_matches_pennylane():
    """QuantumLayerTorch forward must match the PennyLane QuantumLayer."""
    import torch

    from quantum.vqc import QuantumLayer, QuantumLayerTorch
    from quantum.config import VQCConfig

    n = 4
    cfg = VQCConfig(qml_layers=3)
    torch_layer = QuantumLayerTorch(n, cfg)
    pl_layer = QuantumLayer(n, cfg)
    pl_layer.weights.data = torch_layer.weights.data.clone()

    rng = np.random.RandomState(11)
    x = torch.tensor(rng.randn(6, n).astype(np.float32))
    with torch.no_grad():
        out_torch = torch_layer(x)
        out_pl = pl_layer(x)
    diff = float(torch.max(torch.abs(out_torch - out_pl)))
    assert diff < 1e-5, (
        f"torch layer output diverges from PennyLane: max |diff| = {diff:.3e}"
    )


def test_checkpoint_state_dict_compat():
    """The torch layer must expose quantum.weights with the PL shape so
    existing hybrid_vqc.pt checkpoints load unchanged."""
    import torch

    from quantum.vqc import HybridModel
    from quantum.config import VQCConfig

    n = 4
    cfg = VQCConfig(qml_layers=3)
    model = HybridModel(n, cfg)
    state = model.state_dict()
    assert "quantum.weights" in state, f"missing quantum.weights: {list(state)}"
    expected = (cfg.qml_layers, n, 3)
    assert tuple(state["quantum.weights"].shape) == expected, (
        f"weights shape {tuple(state['quantum.weights'].shape)} != {expected}"
    )


def test_label_conversion():
    """Test explicit label conversion functions.

    Verifies:
    - CSV 0 -> quantum -> REAL
    - CSV 1 -> quantum -> FAKE
    - quantum 1 -> display REAL
    - quantum 0 -> display FAKE
    """
    from quantum.data import csv_to_quantum_label, quantum_to_display_label
    from quantum.config import LABEL_REAL, LABEL_FAKE

    # CSV 0 (real) -> quantum 1 (REAL)
    q_label = csv_to_quantum_label(0)
    assert q_label == LABEL_REAL, f"CSV 0 -> expected quantum {LABEL_REAL}, got {q_label}"
    display = quantum_to_display_label(q_label)
    assert display == "REAL", f"quantum 1 -> expected REAL, got {display}"

    # CSV 1 (fake) -> quantum 0 (FAKE)
    q_label = csv_to_quantum_label(1)
    assert q_label == LABEL_FAKE, f"CSV 1 -> expected quantum {LABEL_FAKE}, got {q_label}"
    display = quantum_to_display_label(q_label)
    assert display == "FAKE", f"quantum 0 -> expected FAKE, got {display}"

    # Edge case: invalid CSV label raises
    try:
        csv_to_quantum_label(2)
        assert False, "Should have raised ValueError for invalid CSV label"
    except ValueError:
        pass  # expected

    print("PASS test_label_conversion")


def test_group_bootstrap_and_nested_cv() -> None:
    """Group bootstrap CIs, paired difference, and nested-CV group disjointness.

    Uses small hand-built arrays only to exercise the evaluation code paths; no model
    training data is involved.
    """
    from quantum.evaluation import (
        group_bootstrap,
        make_grid_search_fit_predict,
        nested_group_cv,
        paired_group_bootstrap_diff,
    )

    rng = np.random.RandomState(0)
    n_groups, per = 40, 5
    groups = np.repeat(np.arange(n_groups), per)
    y = np.tile(np.array([1, 1, 1, 0, 0]), n_groups)
    perfect = y.astype(float) * 0.8 + 0.1
    res = group_bootstrap(y, perfect, groups, n_boot=200, seed=1)
    auc = res["metrics"]["auc"]
    assert auc["point"] == 1.0 and auc["ci95"] == [1.0, 1.0], auc
    assert res["n_groups"] == n_groups and res["n_clips"] == n_groups * per

    noisy = np.clip(perfect + rng.normal(0, 0.5, len(y)), 0, 1)
    ci = group_bootstrap(y, noisy, groups, n_boot=300, seed=2)["metrics"]["auc"]
    assert ci["ci95"][0] <= ci["point"] <= ci["ci95"][1], ci
    same = paired_group_bootstrap_diff(y, noisy, noisy, groups, n_boot=100, seed=3)
    assert same["difference_a_minus_b"] == 0.0 and same["ci95"] == [0.0, 0.0], same
    better = paired_group_bootstrap_diff(y, perfect, noisy, groups, n_boot=200, seed=4)
    assert better["difference_a_minus_b"] > 0 and better["ci95"][0] > 0, better

    # nested CV: folds are group-disjoint and the runner refuses overlapping groups
    X = np.column_stack([y + rng.normal(0, 0.3, len(y)), rng.normal(size=len(y))])
    outer = (groups % 4).astype(int)
    from sklearn.linear_model import LogisticRegression

    fp = make_grid_search_fit_predict(
        lambda params, seed: LogisticRegression(C=params["C"], max_iter=500), [{"C": 0.1}, {"C": 1.0}]
    )
    out = nested_group_cv(fp, X, y, groups, outer, seed=0)
    assert np.isfinite(out["scores"]).all() and len(out["folds"]) == 4
    assert all(f["auc"] > 0.8 for f in out["folds"]), out["folds"]
    bad_outer = np.arange(len(y)) % 4  # splits each group across folds
    try:
        nested_group_cv(fp, X, y, groups, bad_outer, seed=0)
        assert False, "overlapping groups must be rejected"
    except AssertionError as exc:
        assert "appear in both" in str(exc), exc

    # equal-budget random search: same number of configurations, deterministic, no outer leakage
    from quantum.evaluation import make_random_search_fit_predict

    rs = make_random_search_fit_predict(
        lambda params, seed: LogisticRegression(C=params["C"], max_iter=500),
        {"C": [0.01, 0.1, 1.0, 10.0]}, n_configs=3,
    )
    a = nested_group_cv(rs, X, y, groups, outer, seed=1)
    b = nested_group_cv(rs, X, y, groups, outer, seed=1)
    assert np.array_equal(a["scores"], b["scores"]), "random search must be deterministic per seed"
    assert all(f["n_configs"] == 3 for f in a["folds"]), a["folds"]


def test_provenance_record(tmp=None) -> None:
    import json, tempfile
    from quantum.config import VQCConfig
    from quantum.provenance import collect_provenance, write_json_with_provenance

    with tempfile.TemporaryDirectory() as d:
        out = write_json_with_provenance(
            Path(d) / "r.json", {"auc": 0.5}, seeds=[1, 2], configs={"vqc": VQCConfig()}
        )
        rec = json.loads(out.read_text())
    prov = rec["provenance"]
    assert rec["auc"] == 0.5 and prov["seeds"] == [1, 2], prov
    assert prov["configs"]["vqc"]["seed"] == VQCConfig().seed
    assert prov["git_commit"] is None or len(prov["git_commit"]) == 40
    assert collect_provenance()["python"], "python version missing"


def test_stage1_grouping_split() -> None:
    import tempfile
    from quantum.grouping import assert_test_unlocked, build_split, components

    assert len(set(components([(0, 1), (1, 2)], 5).tolist())) == 3
    rng = np.random.RandomState(0)
    n, k = 400, 100  # k source families of 4 clips; clips in a family have near-identical backgrounds
    fam = np.repeat(np.arange(k), n // k)
    base = rng.randn(k, 40)
    sig = base[fam] + 0.01 * rng.randn(n, 40)
    sig /= np.linalg.norm(sig, axis=1, keepdims=True)
    sim = sig @ sig.T
    y = (fam % 2).astype(int)
    stems = [f"c{i}" for i in range(n)]
    m = build_split(
        rng.randn(n, 3), y, fam.astype(str), stems, stems, sim, fam % 50, [], csv_sha256="x"
    )
    assert m["gate_passed"], m["overlap_test_vs_dev_by_grouping"]
    assert all(v == 0 for v in m["overlap_test_vs_dev_by_grouping"].values())
    fams = {}
    for i, st in enumerate(stems):
        fams.setdefault(int(fam[i]), set()).add(m["rows"][st]["split"] == "test")
    assert all(len(v) == 1 for v in fams.values()), "a source family straddles test and non-test"
    again = build_split(
        rng.randn(n, 3), y, fam.astype(str), stems, stems, sim, fam % 50, [], csv_sha256="x"
    )
    assert again["split_hash"] == m["split_hash"], "split must be deterministic"
    with tempfile.TemporaryDirectory() as d:
        assert_test_unlocked(d)
        (Path(d) / "TEST_ACCESSED.lock").write_text("x")
        try:
            assert_test_unlocked(d)
        except RuntimeError:
            return
        raise AssertionError("second test-set access must be blocked")


def test_inference_uses_checkpoint_threshold():
    """predict_features must cut at the calibrated threshold stored in the
    checkpoint metadata, not at the DecisionConfig() default of 0.5.

    Regression guard: opt_threshold was loaded but dead, so every deployed
    verdict was silently cut at 0.5 regardless of training-time calibration.
    """
    import tempfile

    import torch

    from quantum import pipeline as qpipe
    from quantum.config import VQCConfig, schema_digest, write_sha256
    from quantum.explain import Verdict
    from quantum.qaoa import save_selection
    from quantum.scaling import FeatureScaler
    from quantum.vqc import HybridModel

    names = list(qpipe.FUSED_FEATURE_NAMES)
    rng = np.random.RandomState(3)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        scaler_file = tmp_path / "scaler.json"
        FeatureScaler(names).fit(rng.randn(8, len(names))).save(scaler_file)

        selection_file = tmp_path / "selection.json"
        save_selection(
            {
                "selected_indices": [0, 1, 2],
                "selected_features": [names[i] for i in (0, 1, 2)],
            },
            selection_file,
        )

        checkpoint_file = tmp_path / "model.pt"

        def write_checkpoint(threshold: float) -> None:
            model = HybridModel(3, VQCConfig())
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "metadata": {
                        "feature_schema_sha256": schema_digest(names),
                        "decision_threshold": float(threshold),
                    },
                },
                checkpoint_file,
            )
            write_sha256(checkpoint_file)

        original = dict(qpipe.FEATURE_SET_CONFIGS["fused"])
        qpipe.FEATURE_SET_CONFIGS["fused"] = {
            **original,
            "scaler_file": scaler_file,
            "selection_file": selection_file,
            "checkpoint_file": checkpoint_file,
        }
        try:
            features = {name: 0.1 for name in names}
            # A calibrated cut of 1.0 must force FAKE and 0.0 must force REAL
            # for any sigmoid output in (0, 1). Hard-coding 0.5 (the old bug)
            # would give the same verdict for both calls and fail here.
            write_checkpoint(1.0)
            verdict_hi = qpipe.predict_features(features).final_verdict
            write_checkpoint(0.0)
            verdict_lo = qpipe.predict_features(features).final_verdict
        finally:
            qpipe.FEATURE_SET_CONFIGS["fused"] = original

    assert verdict_hi is Verdict.FAKE, (
        f"threshold 1.0 must yield FAKE, got {verdict_hi}"
    )
    assert verdict_lo is Verdict.REAL, (
        f"threshold 0.0 must yield REAL, got {verdict_lo}"
    )


def test_select_classical_k_override():
    """select_classical(k=...) must widen the candidate pool beyond
    cfg.target_features.

    Regression guard: the QAOA pre-select stage asked for 18 candidates but
    only ever received cfg.target_features (3), so QAOA searched 3 of 3
    features and the selection stage was effectively a no-op.
    """
    from quantum.qaoa import select_classical

    rng = np.random.RandomState(5)
    X = rng.randn(80, 24)
    y = (X[:, 0] + 0.5 * rng.randn(80) > 0).astype(int)
    names = [f"f{i}" for i in range(24)]
    cfg = QAOASelectionConfig(target_features=3)

    wide = select_classical(X, y, cfg, feature_names=names, k=18)
    default = select_classical(X, y, cfg, feature_names=names)

    assert len(wide["selected_indices"]) == 18, (
        f"k=18 must return 18 indices, got {len(wide['selected_indices'])}"
    )
    assert len(default["selected_indices"]) == 3, (
        f"default must return cfg.target_features=3, got {len(default['selected_indices'])}"
    )
    assert wide["selected_indices"][:3] == default["selected_indices"], (
        "k override must extend, not reorder, the default ranking"
    )
    assert len(set(wide["selected_indices"])) == 18, "selection must be unique features"


def test_far_constrained_threshold():
    """The KYC threshold cut must honour the FAR ceiling (PLAN.md Step 5).

    A FAKE clip accepted as REAL is the expensive error, so the deployed cut
    is the smallest validation threshold whose false-acceptance rate is
    <= far_target, not Youden's J-optimal cut (which ignores KYC cost).
    """
    from quantum.evaluation import (
        classification_metrics,
        choose_validation_threshold,
        optimal_threshold_far_constrained,
    )

    rng = np.random.RandomState(7)
    # Weakly-separating scores: Youden's J-optimal cut is permissive here.
    y = np.array([0] * 40 + [1] * 40)
    p = np.r_[rng.uniform(0.30, 0.95, 40), rng.uniform(0.35, 0.98, 40)]

    far_cut = optimal_threshold_far_constrained(y, p, far_target=0.10)
    youden_cut = choose_validation_threshold(y, p, method="youden")

    at_far = classification_metrics(y, p, decision_threshold=far_cut)
    assert at_far["false_acceptance_rate"] <= 0.10 + 1e-12, (
        f"FAR {at_far['false_acceptance_rate']:.3f} exceeds the 0.10 ceiling "
        f"at threshold {far_cut:.4f}"
    )

    # Minimality: the next candidate below the chosen cut must break the
    # ceiling, otherwise a lower (less user-rejecting) cut was available.
    candidates = np.unique(np.r_[0.0, p, 1.0])
    lower = float(candidates[candidates < far_cut].max())
    at_lower = classification_metrics(y, p, decision_threshold=lower)
    assert at_lower["false_acceptance_rate"] > 0.10, (
        f"threshold {far_cut:.4f} is not the smallest cut meeting the ceiling"
    )

    # When Youden violates the ceiling, the constrained cut must be stricter.
    at_youden = classification_metrics(y, p, decision_threshold=youden_cut)
    if at_youden["false_acceptance_rate"] > 0.10:
        assert far_cut > youden_cut, (
            f"constrained cut {far_cut:.4f} must exceed the permissive "
            f"Youden cut {youden_cut:.4f}"
        )

    # Fallback: no cut can satisfy the ceiling (fakes score exactly 1.0), so
    # the chooser must return the Youden cut instead of raising.
    y2 = np.array([0, 0, 1, 1])
    p2 = np.array([1.0, 1.0, 1.0, 1.0])
    fallback = optimal_threshold_far_constrained(y2, p2, far_target=0.10)
    assert fallback == choose_validation_threshold(y2, p2, method="youden"), (
        "infeasible FAR ceiling must fall back to Youden"
    )


def test_vqc_ensemble_averaging():
    """Seed-ensemble inference must average members and force eval mode.

    Guards PLAN.md Step 6: predict_features/evaluate_quantum_model average
    P(real) over every checkpoint on disk, and the checkpoint naming contract
    (member 0 keeps the legacy filename, members 1..n append "_seed<N>").
    """
    from quantum.config import VQC_ENSEMBLE_SEEDS, VQCConfig
    from quantum.vqc import (
        HybridModel,
        ensemble_checkpoint_files,
        predict_vqc,
        predict_vqc_ensemble,
    )

    files = ensemble_checkpoint_files(Path("/x/hybrid_vqc_fused.pt"))
    assert len(files) == len(VQC_ENSEMBLE_SEEDS) == 3, (
        f"expected a 3-member ensemble, got {len(files)}"
    )
    assert files[0] == Path("/x/hybrid_vqc_fused.pt"), (
        "member 0 must keep the legacy checkpoint filename"
    )
    assert files[1].name == "hybrid_vqc_fused_seed43.pt", files[1].name
    assert files[2].name == "hybrid_vqc_fused_seed44.pt", files[2].name

    cfg = VQCConfig()
    m1, m2 = HybridModel(4, cfg), HybridModel(4, cfg)
    rng = np.random.RandomState(11)
    X = rng.randn(12, 4).astype(np.float32)
    m1.eval()
    m2.eval()
    p1 = predict_vqc(m1, X)
    p2 = predict_vqc(m2, X)

    m1.train()  # ensemble must switch members back to eval mode itself
    avg = predict_vqc_ensemble([m1, m2], X)
    assert not m1.training, "ensemble must put members in eval mode"
    assert np.allclose(avg, (p1 + p2) / 2.0, atol=1e-6), (
        f"ensemble must return the mean of member probabilities, "
        f"max |diff| = {float(np.max(np.abs(avg - (p1 + p2) / 2.0))):.3e}"
    )


def main() -> None:
    checks = [
        test_beta_alive,
        test_hamiltonian_matches_classical,
        test_real_hamiltonian_verification,
        test_feature_contract_sync,
        test_split_determinism,
        test_dfdc_subject_grouping,
        test_no_group_leakage,
        test_qaoa_sim_matches_pennylane,
        test_torch_layer_matches_pennylane,
        test_checkpoint_state_dict_compat,
        test_group_bootstrap_and_nested_cv,
        test_provenance_record,
        test_stage1_grouping_split,
        test_inference_uses_checkpoint_threshold,
        test_select_classical_k_override,
        test_far_constrained_threshold,
        test_vqc_ensemble_averaging,
    ]
    failures = 0
    for check in checks:
        try:
            check()
            print(f"PASS  {check.__name__}")
        except AssertionError as exc:
            failures += 1
            print(f"FAIL  {check.__name__}: {exc}")
    print(f"\n{len(checks) - failures}/{len(checks)} checks passed")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()