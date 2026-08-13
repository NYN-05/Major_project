import json

import numpy as np
import pennylane as qml
from scipy.optimize import minimize
from sklearn.feature_selection import mutual_info_classif

from quantum.config import FEATURE_NAMES, QAOASelectionConfig


def _normalize_weights(weights):
    peak = weights.max()
    return weights / peak if peak > 0.0 else weights


def _cost_terms(weights, correlation, cfg):
    n = len(weights)
    k = cfg.target_features
    rp = cfg.redundancy_penalty
    cp = cfg.cardinality_penalty
    lin = (
        weights / 2.0
        - (rp / 4.0) * (correlation.sum(axis=1) - 1.0)
        + cp * (4 * k - 2 * n + 1) / 4.0
    )
    quad = (rp / 4.0) * correlation + cp / 2.0
    constant = cp * (n - 2 * k) ** 2 / 4.0 + (rp / 4.0) * np.triu(correlation, 1).sum()
    coeffs = [float(constant)]
    ops = [qml.Identity(wires=0)]
    for i in range(n):
        if abs(lin[i]) > 1e-12:
            coeffs.append(float(lin[i]))
            ops.append(qml.PauliZ(i))
        for j in range(i + 1, n):
            value = quad[i, j]
            if abs(value) > 1e-12:
                coeffs.append(float(value))
                ops.append(qml.PauliZ(i) @ qml.PauliZ(j))
    return coeffs, ops


def _classical_cost(bitstring, weights, correlation, cfg):
    mi_term = -float(np.dot(weights, bitstring))
    redundancy = cfg.redundancy_penalty * (
        0.5 * float(bitstring @ correlation @ bitstring) - 0.5 * float(bitstring.sum())
    )
    cardinality = cfg.cardinality_penalty * (
        float(bitstring.sum()) - cfg.target_features
    ) ** 2
    return mi_term + redundancy + cardinality


def _make_circuits(coeffs, ops, wires):
    hamiltonian = qml.Hamiltonian(coeffs, ops)
    # QAOA stays on the reference statevector simulator: it is faster
    # than accelerated backends for these small Hamiltonian-exp circuits
    # and keeps the selected-feature output bit-identical across runs.
    dev = qml.device("default.qubit", wires=wires)

    def _apply_qaoa(params):
        for wire in range(wires):
            qml.Hadamard(wires=wire)
        for layer in range(len(params) // 2):
            gamma, beta = params[2 * layer], params[2 * layer + 1]
            for coeff, op in zip(coeffs, ops):
                qml.exp(op, -1j * coeff * gamma)
            for wire in range(wires):
                qml.exp(qml.PauliX(wire), -1j * beta)

    @qml.qnode(dev)
    def cost_circuit(params):
        _apply_qaoa(params)
        return qml.expval(hamiltonian)

    @qml.qnode(dev)
    def marginals_circuit(params):
        _apply_qaoa(params)
        return [qml.expval(qml.PauliZ(i)) for i in range(wires)]

    return cost_circuit, marginals_circuit


class QAOASelector:
    def __init__(self, cfg=None):
        self.cfg = cfg or QAOASelectionConfig()

    def select(self, X, y):
        cfg = self.cfg
        rng = np.random.RandomState(cfg.seed)
        weights = _normalize_weights(mutual_info_classif(X, y, random_state=cfg.seed))
        correlation = np.abs(np.corrcoef(X.T))
        coeffs, ops = _cost_terms(weights, correlation, cfg)
        cost_circuit, marginals_circuit = _make_circuits(coeffs, ops, X.shape[1])
        params0 = rng.uniform(0.0, 0.3, size=2 * cfg.p_layers)
        result = minimize(
            cost_circuit,
            params0,
            method="COBYLA",
            options={"maxiter": cfg.max_iter},
        )
        marginals = (1.0 - np.asarray(marginals_circuit(result.x))) / 2.0
        order = np.argsort(-marginals)
        selected = order[: cfg.target_features]
        return {
            "selected_indices": [int(i) for i in selected],
            "selected_features": [FEATURE_NAMES[i] for i in selected],
            "marginal_probabilities": [float(m) for m in marginals],
            "feature_weights": [float(w) for w in weights],
            "cost": float(result.fun),
            "success": bool(result.success),
        }


def select_classical(X, y, cfg=None):
    """Classical reference selection: top-k features by mutual information.

    Report-only selector used to sanity-check that the QAOA choice is not
    worse than a plain MI-greedy pick. Same output schema as
    QAOASelector.select() so downstream code can consume either.
    """
    cfg = cfg or QAOASelectionConfig()
    weights = _normalize_weights(mutual_info_classif(X, y, random_state=cfg.seed))
    order = np.argsort(-weights)
    selected = order[: cfg.target_features]
    marginals = np.zeros(X.shape[1])
    for pos, i in enumerate(selected):
        marginals[i] = 1.0 - pos / X.shape[1]
    return {
        "selected_indices": [int(i) for i in selected],
        "selected_features": [FEATURE_NAMES[i] for i in selected],
        "marginal_probabilities": [float(m) for m in marginals],
        "feature_weights": [float(w) for w in weights],
        "cost": None,
        "success": True,
    }


def compare_selections(qaoa_result, classical_result):
    """Summarize two selections for the comparison artifact."""
    qa = set(qaoa_result["selected_indices"])
    cl = set(classical_result["selected_indices"])
    return {
        "qaoa_features": qaoa_result["selected_features"],
        "classical_features": classical_result["selected_features"],
        "overlap_indices": sorted(qa & cl),
        "overlap_count": len(qa & cl),
        "qaoa_only": [FEATURE_NAMES[i] for i in sorted(qa - cl)],
        "classical_only": [FEATURE_NAMES[i] for i in sorted(cl - qa)],
    }


def verify_hamiltonian(X, y, cfg=None):
    cfg = cfg or QAOASelectionConfig()
    weights = _normalize_weights(mutual_info_classif(X, y, random_state=cfg.seed))
    correlation = np.abs(np.corrcoef(X.T))
    coeffs, ops = _cost_terms(weights, correlation, cfg)
    dev = qml.device("default.qubit", wires=X.shape[1])

    @qml.qnode(dev)
    def basis_cost(bitstring):
        for i, bit in enumerate(bitstring):
            if bit:
                qml.PauliX(wires=i)
        return qml.expval(qml.Hamiltonian(coeffs, ops))

    all_ones = np.ones(X.shape[1], dtype=int)
    single = np.zeros(X.shape[1], dtype=int)
    single[0] = 1
    max_error = 0.0
    for bitstring in (all_ones, single):
        quantum = float(basis_cost(bitstring))
        classical = _classical_cost(bitstring, weights, correlation, cfg)
        max_error = max(max_error, abs(quantum - classical))
    return max_error


def save_selection(selection, path=None):
    path = path or QAOASelectionConfig().selection_file
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(selection, fh, indent=2)
    return path


def load_selection(path=None):
    path = path or QAOASelectionConfig().selection_file
    with open(path) as fh:
        return json.load(fh)