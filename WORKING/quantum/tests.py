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

from quantum.config import FEATURE_NAMES, DataConfig, QAOASelectionConfig
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
    """verify_hamiltonian must be ~0 on the real rPPG feature table."""
    csv_file = DataConfig().csv_file
    if not csv_file.exists():
        raise AssertionError(f"dataset CSV not found at {csv_file}")
    from quantum.data import _load_rppg_rows

    X, y, _, _, _, _ = _load_rppg_rows(csv_file, DataConfig())
    assert X.shape[1] == len(FEATURE_NAMES) == 20
    assert set(y.tolist()) <= {0, 1}
    error = verify_hamiltonian(X, y, QAOASelectionConfig())
    assert error < 1e-6, f"verify_hamiltonian error on real data: {error:.2e}"


def test_feature_contract_sync():
    """FEATURE_NAMES (quantum/config.py) must match RPPGFeatures order."""
    try:
        from rppg.features import RPPGFeatures
    except ImportError as exc:
        raise AssertionError(f"cannot import rppg.features: {exc}")
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


def test_ffpp_source_subject_grouping():
    """FF++ real + its synthesis must share a group key (no leakage)."""
    from quantum.data import _infer_subject_key

    real = {"video_path": r"FF++\train\FF-real\id0_0000.mp4"}
    synth = {"video_path": r"FF++\train\FF-synthesis\id0_id16_0002.mp4"}
    yt = {"video_path": r"FF++\train\YouTube-real\00000.mp4"}
    dfdc = {"video_path": r"archive\DFDC_Dataset\Fake\aaaoqepxnf.mp4"}
    assert _infer_subject_key(real) == _infer_subject_key(synth) == "ffpp:src:id0"
    assert _infer_subject_key(yt) == "ffpp:yt:00000"
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
    for subj in ("id0", "id1", "id2", "id3"):
        rows.append({"video_path": rf"FF++\train\FF-real\{subj}_0000.mp4"})
        rows.append({"video_path": rf"FF++\train\FF-synthesis\{subj}_id9_0001.mp4"})
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


def main() -> None:
    checks = [
        test_beta_alive,
        test_hamiltonian_matches_classical,
        test_real_hamiltonian_verification,
        test_feature_contract_sync,
        test_split_determinism,
        test_ffpp_source_subject_grouping,
        test_no_group_leakage,
        test_qaoa_sim_matches_pennylane,
        test_torch_layer_matches_pennylane,
        test_checkpoint_state_dict_compat,
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