import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pennylane as qml
from scipy.optimize import minimize

from quantum.config import FEATURE_NAMES, QAOASelectionConfig


def _numpy_qnode(qnode):
    """Adapt a QNode whose outputs are torch tensors (torch-backend
    devices) to plain numpy, so scipy COBYLA and downstream numpy code
    keep working unchanged."""

    def wrapped(*args):
        out = qnode(*args)
        if hasattr(out, "detach"):
            out = out.detach().cpu().numpy()
        return np.asarray(out, dtype=np.float64)

    return wrapped


def _make_torch_device(wires):
    """CUDA-capable torch statevector device (default.qubit.torch), or
    None when this PennyLane version/platform has no such backend."""
    if not torch_available():
        return None
    try:
        return qml.device(
            "default.qubit.torch", wires=wires, torch_device="cuda"
        )
    except Exception:
        return None


def torch_available():
    try:
        import torch  # noqa: F401

        return torch.cuda.is_available()
    except Exception:
        return False


_GPU_PROBE = {"done": False, "ok": False, "detail": ""}


def _probe_gpu_device():
    """One-time per-process gate deciding whether the GPU torch backend is
    usable for QAOA. Two conditions must hold:

    1. Precision: the float64 classical bitstring energy must be reproduced
       within 1e-6 (the pipeline hard-asserts Hamiltonian == classical
       cost at that tolerance; float32 statevectors do not pass).
    2. Speed: the GPU per-call cost must not exceed ~3x lightning.qubit's,
       otherwise QAOA would be net-slower on GPU (COBYLA is call-bound;
       tiny 8-qubit states make transfer overhead dominate).
    """
    if _GPU_PROBE["done"]:
        return _GPU_PROBE["ok"]
    _GPU_PROBE["done"] = True
    dev = _make_torch_device(8)
    if dev is None:
        _GPU_PROBE["detail"] = "default.qubit.torch not available (PennyLane >= 0.39 removed it)"
        return False
    try:
        rng = np.random.RandomState(0)
        lin = rng.uniform(-1.0, 1.0, 8)
        quad = np.triu(rng.uniform(-0.3, 0.8, (8, 8)), 1)
        quad = quad + quad.T
        ops = [qml.PauliZ(i) for i in range(8)]
        coeffs = [float(c) for c in lin]
        for i in range(8):
            for j in range(i + 1, 8):
                ops.append(qml.PauliZ(i) @ qml.PauliZ(j))
                coeffs.append(float(quad[i, j]))
        hamiltonian = qml.Hamiltonian(coeffs, ops)

        @qml.qnode(dev, interface="torch")
        def energy(bitstring):
            for i, bit in enumerate(bitstring):
                if bit:
                    qml.PauliX(wires=i)
            return qml.expval(hamiltonian)

        err = 0.0
        for _ in range(4):
            bs = rng.randint(0, 2, size=8)
            quantum = float(energy(bs))
            classical = float(lin @ bs + bs @ quad @ bs)
            err = max(err, abs(quantum - classical))
        if err >= 1e-6:
            _GPU_PROBE["detail"] = f"precision gate failed (max err {err:.2e} >= 1e-6)"
            return False

        try:
            import pennylane_lightning  # noqa: F401

            cpu_dev = qml.device("lightning.qubit", wires=8)
        except ImportError:
            cpu_dev = qml.device("default.qubit", wires=8)

        @qml.qnode(cpu_dev)
        def cpu_energy(bitstring):
            for i, bit in enumerate(bitstring):
                if bit:
                    qml.PauliX(wires=i)
            return qml.expval(hamiltonian)

        t0 = time.perf_counter()
        for _ in range(5):
            cpu_energy(rng.randint(0, 2, size=8))
        cpu_ms = (time.perf_counter() - t0) / 5 * 1e3
        t0 = time.perf_counter()
        for _ in range(5):
            energy(rng.randint(0, 2, size=8))
        gpu_ms = (time.perf_counter() - t0) / 5 * 1e3
        if gpu_ms > max(3.0 * cpu_ms, cpu_ms + 2.0):
            _GPU_PROBE["detail"] = (
                f"speed gate failed (gpu {gpu_ms:.3f} ms/call vs cpu {cpu_ms:.3f} ms/call)"
            )
            return False
        _GPU_PROBE["ok"] = True
        _GPU_PROBE["detail"] = f"precision 1e-6 OK, gpu {gpu_ms:.3f} ms vs cpu {cpu_ms:.3f} ms/call"
        return True
    except Exception as exc:  # noqa: BLE001
        _GPU_PROBE["detail"] = f"probe raised {type(exc).__name__}: {exc}"
        return False


def simulator_device(wires, cfg=None):
    """Return ``(dev, backend)`` describing the chosen QAOA simulator.

    ``dev`` is always a valid PennyLane device (for legacy/fallback code
    paths).  ``backend`` ∈ {"torch-native", "lightning", "default"}.
    When ``cfg.device`` is "auto" or "torch", ``backend="torch-native"``
    indicates the torch statevector sim should be used for the hot loop;
    ``dev`` is still a valid PL device for backward compat with
    ``_make_circuits`` and tests.
    """
    cfg = cfg or QAOASelectionConfig()
    device = getattr(cfg, "device", "auto")
    if device in ("auto", "torch"):
        try:
            import pennylane_lightning  # noqa: F401

            dev = qml.device("lightning.qubit", wires=wires)
        except ImportError:
            dev = qml.device("default.qubit", wires=wires)
        return dev, "torch-native"
    if device in ("pennylane",):
        try:
            import pennylane_lightning  # noqa: F401

            return qml.device("lightning.qubit", wires=wires), "lightning"
        except ImportError:
            return qml.device("default.qubit", wires=wires), "default"
    # Legacy "lightning"/"default" aliases
    if device == "lightning":
        return qml.device("lightning.qubit", wires=wires), "lightning"
    return qml.device("default.qubit", wires=wires), "default"


def _mutual_info_weights(X, y, seed):
    """MI feature weights. sklearn is imported lazily: importing it costs
    ~12s on this host, which would slow every QAOA worker process."""
    from sklearn.feature_selection import mutual_info_classif  # noqa: PLC0415

    return mutual_info_classif(X, y, random_state=seed)


def _discrimination_weights(X, y, seed=None):
    """Supervised per-feature discrimination strength (sign-agnostic AUC).

    weight_i = 2*|AUC_i - 0.5| in [0, 1], where AUC_i is the area under
    the ROC curve of feature i alone separating the two classes (the
    Mann-Whitney U statistic, computed exactly, no training needed).

    Replaces mutual-information weights: on the weak, noisy rPPG
    features MI ranked the strongest discriminators (heart_rate_bpm,
    snr_db) near zero, so the QAOA optimizer systematically selected
    non-informative features (measured CV AUC 0.484 vs 0.569 for the
    top-3 discriminators). The sign is dropped because an inverted
    feature (AUC < 0.5) is as informative as a forward one once the
    downstream classifier learns its sign.
    """
    y = np.asarray(y)
    pos_mask = y == 1
    n_pos = int(pos_mask.sum())
    n_neg = int((~pos_mask).sum())
    d = X.shape[1]
    if n_pos == 0 or n_neg == 0:
        return np.zeros(d, dtype=np.float64)
    weights = np.zeros(d, dtype=np.float64)
    for i in range(d):
        a = np.asarray(X[:, i], dtype=np.float64)[pos_mask]
        b = np.asarray(X[:, i], dtype=np.float64)[~pos_mask]
        gt = int((a[:, None] > b[None, :]).sum())
        eq = int((a[:, None] == b[None, :]).sum())
        auc = (gt + 0.5 * eq) / (n_pos * n_neg)
        weights[i] = 2.0 * abs(auc - 0.5)
    return weights


def _normalize_weights(weights):
    peak = weights.max()
    return weights / peak if peak > 0.0 else weights


def _cost_terms(weights, correlation, cfg):
    """Encode the classical selection cost as a Pauli-Z Hamiltonian.

    Classical cost over bitstrings (b_i in {0,1}):
        C(b) = -w.b + rp * sum_{i<j} C_ij b_i b_j + cp * (sum b - k)^2
    Substituting b = (1 - Z)/2 gives H = const + sum lin_i Z_i
    + sum_{i<j} quad_ij Z_i Z_j with:
        lin_i    =  w_i/2 + cp*(2k - 1)/2
                    - (rp/4)*(rowsum_i - 1) - (cp/2)*(n - 1)
        quad_ij  =  (rp/4)*C_ij + cp/2
        const    =  cp*k^2 - sum(w)/2 + (n/2)*cp*(1 - 2k)
                    + (rp/4)*triu(C, 1).sum() + cp*n*(n - 1)/4
    (The -(1/4)*sum_{j!=i} q_ij terms in lin_i come from expanding
    (1-Z_i)(1-Z_j)/4; omitting them breaks expval(H) == C(b).)
    """
    n = len(weights)
    k = cfg.target_features
    rp = cfg.redundancy_penalty
    cp = cfg.cardinality_penalty
    lin = (
        weights / 2.0
        + cp * (2 * k - 1) / 2.0
        - (rp / 4.0) * (correlation.sum(axis=1) - 1.0)
        - (cp / 2.0) * (n - 1)
    )
    quad = (rp / 4.0) * correlation + cp / 2.0
    constant = (
        cp * k**2
        - weights.sum() / 2.0
        + (n / 2.0) * cp * (1 - 2 * k)
        + (rp / 4.0) * np.triu(correlation, 1).sum()
        + cp * n * (n - 1) / 4.0
    )
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


def _precompute_gates(coeffs, ops, wires):
    """Decompose exp(-i*gamma*H) into per-term PauliRot gates, once.

    Every term in H is a Pauli-Z string (Z or Z@Z), so all terms commute
    and the product of per-term ``PauliRot(2*gamma*c, basis, wires)`` is
    exactly exp(-i*gamma*H) (no Trotter error). The identity term is a
    global phase and is dropped. Precomputing the gate list removes the
    per-call Hamiltonian-expansion overhead inside the QNode.

    Returns ``(cost_gates, mixer_gates)``: the Z-term gates are driven by
    ``gamma``; the single-qubit X rotations are the QAOA mixer and MUST be
    driven by ``beta`` (regression guard: cost must change when beta is
    perturbed — see ``tests.py``).
    """
    cost_gates = []
    for c, op in zip(coeffs, ops):
        if isinstance(op, qml.Identity):
            continue
        cost_gates.append((float(c), "Z" * len(op.wires), list(op.wires)))
    mixer_gates = [(1.0, "X", [w]) for w in range(wires)]
    return cost_gates, mixer_gates


def _make_circuits(coeffs, ops, wires, cfg=None):
    cfg = cfg or QAOASelectionConfig()
    hamiltonian = qml.Hamiltonian(coeffs, ops)
    dev, backend = simulator_device(wires, cfg)
    torch_backend = backend == "torch"
    cost_gates, mixer_gates = _precompute_gates(coeffs, ops, wires)

    def _apply_qaoa(params):
        for wire in range(wires):
            qml.Hadamard(wires=wire)
        for layer in range(len(params) // 2):
            gamma, beta = params[2 * layer], params[2 * layer + 1]
            for c, basis, gate_wires in cost_gates:
                qml.PauliRot(2.0 * gamma * c, basis, wires=gate_wires)
            for _, basis, gate_wires in mixer_gates:
                qml.PauliRot(2.0 * beta, basis, wires=gate_wires)

    @qml.qnode(dev, interface="torch" if torch_backend else "autograd")
    def cost_circuit(params):
        _apply_qaoa(params)
        return qml.expval(hamiltonian)

    @qml.qnode(dev, interface="torch" if torch_backend else "autograd")
    def marginals_circuit(params):
        _apply_qaoa(params)
        return [qml.expval(qml.PauliZ(i)) for i in range(wires)]

    if torch_backend:
        return _numpy_qnode(cost_circuit), _numpy_qnode(marginals_circuit)
    return cost_circuit, marginals_circuit


def _restart_worker(seed, weights, correlation, cfg, sim=None):
    """Run one full COBYLA QAOA optimization (used by parallel restarts).

    Must be a module-level function so ProcessPool can pickle it. Each
    restart gets its own fixed seed and returns the best state it found.

    When *sim* (a ``qaoa_sim.QAOASimulator``) is provided, the torch-native
    statevector path is used (~20× faster than PennyLane on this host).
    Otherwise the PennyLane QNode path is used.
"""
    if sim is not None:
        params0 = np.random.RandomState(seed).uniform(0.0, 0.3, size=2 * cfg.p_layers)
        result = minimize(
            sim.cost,
            params0,
            method="COBYLA",
            options={"maxiter": cfg.max_iter},
        )
        return {
            "seed": seed,
            "x": np.asarray(result.x, dtype=np.float64),
            "cost": float(result.fun),
            "success": bool(result.success),
        }
    coeffs, ops = _cost_terms(weights, correlation, cfg)
    cost_circuit, marginals_circuit = _make_circuits(coeffs, ops, len(weights), cfg)
    rng = np.random.RandomState(seed)
    params0 = rng.uniform(0.0, 0.3, size=2 * cfg.p_layers)
    result = minimize(
        cost_circuit,
        params0,
        method="COBYLA",
        options={"maxiter": cfg.max_iter},
    )
    return {
        "seed": seed,
        "x": np.asarray(result.x, dtype=np.float64),
        "cost": float(result.fun),
        "success": bool(result.success),
    }


class QAOASelector:
    def __init__(self, cfg=None):
        self.cfg = cfg or QAOASelectionConfig()

    def select(self, X, y):
        cfg = self.cfg
        weights = _normalize_weights(_discrimination_weights(X, y, cfg.seed))
        correlation = np.abs(np.corrcoef(X.T))
        n_restarts = max(1, cfg.restarts)
        seeds = [cfg.seed + i for i in range(n_restarts)]

        device = getattr(cfg, "device", "auto")
        use_sim = device in ("auto", "torch")
        backend_label = "torch-native" if use_sim else None

        if use_sim:
            from quantum.qaoa_sim import QAOASimulator

            sim = QAOASimulator(X.shape[1], cfg, weights=weights, correlation=correlation)
            restarts = [
                _restart_worker(seed, weights, correlation, cfg, sim=sim)
                for seed in seeds
            ]
        else:
            dev, backend_label = simulator_device(X.shape[1], cfg)
            jobs = [(seed, weights, correlation, cfg) for seed in seeds]
            if n_restarts > 1:
                try:
                    n_jobs = cfg.n_jobs if cfg.n_jobs > 0 else min(n_restarts, os.cpu_count() or 1)
                    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
                        restarts = list(pool.map(_restart_worker, *zip(*jobs)))
                except Exception:
                    restarts = [_restart_worker(*job) for job in jobs]
            else:
                restarts = [_restart_worker(*job) for job in jobs]

        best = min(restarts, key=lambda r: r["cost"])

        if use_sim:
            sim_marginals = sim.marginals(best["x"])
            order = np.argsort(-sim_marginals)
        else:
            cost_circuit, marginals_circuit = _make_circuits(
                *_cost_terms(weights, correlation, cfg), X.shape[1], cfg
            )
            sim_marginals = (1.0 - np.asarray(marginals_circuit(best["x"]))) / 2.0
            order = np.argsort(-sim_marginals)

        selected = order[: cfg.target_features]
        return {
            "selected_indices": [int(i) for i in selected],
            "selected_features": [FEATURE_NAMES[i] for i in selected],
            "marginal_probabilities": [float(m) for m in sim_marginals],
            "feature_weights": [float(w) for w in weights],
            "cost": best["cost"],
            "success": best["success"],
            "restarts": {
                "n_restarts": n_restarts,
                "chosen_seed": best["seed"],
                "all_costs": [r["cost"] for r in restarts],
            },
        }


def select_classical(X, y, cfg=None):
    """Classical reference selection: top-k features by discrimination AUC.

    Report-only selector used to sanity-check that the QAOA choice is not
    worse than a plain supervised-greedy pick. Same output schema as
    QAOASelector.select() so downstream code can consume either.
    """
    cfg = cfg or QAOASelectionConfig()
    weights = _normalize_weights(_discrimination_weights(X, y, cfg.seed))
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
    """Max |expval(H) - classical_cost| over representative bitstrings.

    Checks all-zeros, all-ones, every single-bit state, and a handful of
    random bitstrings so a coefficient-derivation slip (as in the original
    ``_cost_terms``) cannot pass by checking only two states. Returns the
    max absolute error; the caller must assert it is ~0.
    """
    cfg = cfg or QAOASelectionConfig()
    weights = _normalize_weights(_discrimination_weights(X, y, cfg.seed))
    correlation = np.abs(np.corrcoef(X.T))
    coeffs, ops = _cost_terms(weights, correlation, cfg)
    n = X.shape[1]

    device = getattr(cfg, "device", "auto")
    use_sim = device in ("auto", "torch")

    if use_sim:
        from quantum.qaoa_sim import make_hamiltonian_diagonal

        h_diag = make_hamiltonian_diagonal(coeffs, ops, n)
        rng = np.random.RandomState(cfg.seed)
        states = [np.zeros(n, dtype=int), np.ones(n, dtype=int)]
        for i in range(n):
            single = np.zeros(n, dtype=int)
            single[i] = 1
            states.append(single)
        states.extend(rng.randint(0, 2, size=(8, n)))

        max_error = 0.0
        for bitstring in states:
            idx = sum(int(b) << w for w, b in enumerate(bitstring))
            quantum = float(h_diag[idx])
            classical = _classical_cost(bitstring, weights, correlation, cfg)
            max_error = max(max_error, abs(quantum - classical))
        return max_error

    # PennyLane path (legacy)
    import pennylane as qml

    dev, backend = simulator_device(n, cfg)

    @qml.qnode(dev, interface="torch" if backend == "torch" else "autograd")
    def basis_cost(bitstring):
        for i, bit in enumerate(bitstring):
            if bit:
                qml.PauliX(wires=i)
        return qml.expval(qml.Hamiltonian(coeffs, ops))

    if backend == "torch":
        basis_cost = _numpy_qnode(basis_cost)
    rng = np.random.RandomState(cfg.seed)
    states = [np.zeros(n, dtype=int), np.ones(n, dtype=int)]
    for i in range(n):
        single = np.zeros(n, dtype=int)
        single[i] = 1
        states.append(single)
    states.extend(rng.randint(0, 2, size=(8, n)))
    max_error = 0.0
    for bitstring in states:
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
