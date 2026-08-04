import json
from pathlib import Path

import numpy as np
import pennylane as qml
from scipy.optimize import minimize

from quantum.config import QAOASelectionConfig
from quantum.features_schema import FEATURE_NAMES


def mutual_information_scores(X: np.ndarray, y: np.ndarray, bins: int = 10) -> np.ndarray:
    n_features = X.shape[1]
    scores = np.zeros(n_features)
    for i in range(n_features):
        values = X[:, i]
        hist_edges = np.percentile(values, np.linspace(0, 100, bins + 1))
        hist_edges = np.unique(hist_edges)
        if hist_edges.size < 2:
            continue
        disc = np.digitize(values, hist_edges[:-1])
        joint, _, _ = np.histogram2d(disc, y, bins=[hist_edges.size - 1, 2])
        joint = joint / max(joint.sum(), 1e-9)
        mi = 0.0
        for b in range(joint.shape[0]):
            for c in range(joint.shape[1]):
                pxy = joint[b, c]
                px = joint[b, :].sum()
                py = joint[:, c].sum()
                if pxy > 0 and px > 0 and py > 0:
                    mi += pxy * np.log2(pxy / (px * py))
        scores[i] = mi
    return scores / max(scores.max(), 1e-9)


def correlation_penalties(X: np.ndarray) -> np.ndarray:
    corr = np.corrcoef(X, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)
    return np.abs(corr)


class QAOASelector:
    def __init__(self, config: QAOASelectionConfig):
        self.config = config

    def select(self, X: np.ndarray, y: np.ndarray) -> dict:
        n = X.shape[1]
        w = mutual_information_scores(X, y)
        c = correlation_penalties(X)
        tri = np.triu_indices(n, k=1)
        c_sum = float(c[tri].sum())
        lam = self.config.redundancy_penalty
        mu = self.config.cardinality_penalty
        k = min(self.config.target_features, n)

        h = w / 2.0 - lam * (c.sum(axis=1) - 1.0) / 4.0 - mu * n / 2.0 + k * mu
        j_pair = lam * c / 4.0 + mu / 2.0
        const = (
            -w.sum() / 2.0
            + lam * c_sum / 4.0
            + mu * n * (n + 1) / 4.0
            - k * mu * n
            + mu * k**2
        )

        ham_ops = [qml.PauliZ(i) for i in range(n)]
        ham_ops += [qml.PauliZ(i) @ qml.PauliZ(j) for i, j in zip(*tri)]
        ham_coeffs = [*h.tolist(), *[j_pair[i, j] for i, j in zip(*tri)]]

        dev = qml.device(self.config.device, wires=n)

        def apply_cost_layer(gamma):
            for coeff, op in zip(ham_coeffs, ham_ops):
                qml.PauliRot(2.0 * gamma * coeff, "Z" * len(op.wires), wires=list(op.wires))

        def apply_mixer_layer(beta):
            for i in range(n):
                qml.RX(2.0 * beta, wires=i)

        def build_state(params):
            for i in range(n):
                qml.Hadamard(wires=i)
            p = self.config.p_layers
            gamma, beta = params[:p], params[p:]
            for layer in range(p):
                apply_cost_layer(gamma[layer])
                apply_mixer_layer(beta[layer])

        @qml.qnode(dev, interface="autograd")
        def qaoa_circuit(params):
            build_state(params)
            return qml.expval(qml.Hamiltonian(ham_coeffs, ham_ops))

        @qml.qnode(dev, interface="autograd")
        def qaoa_probs(params):
            build_state(params)
            return qml.probs(wires=range(n))

        def expected_cost(params):
            return float(qaoa_circuit(params)) + const

        rng = np.random.default_rng(0)
        init = rng.uniform(0.0, np.pi, size=2 * self.config.p_layers)
        result = minimize(expected_cost, init, method="COBYLA", options={"maxiter": self.config.max_iter, "tol": 1e-6})

        probs = np.asarray(qaoa_probs(result.x))
        best_index = int(np.argmax(probs))
        bitstring = [int(b) for b in f"{best_index:0{n}b}"]
        selected = np.where(np.asarray(bitstring) == 1)[0]

        marginals = np.zeros(n)
        for idx, prob in enumerate(probs):
            bits = [int(b) for b in f"{idx:0{n}b}"]
            marginals += prob * np.asarray(bits)

        selected = self._adjust_to_target(selected, w, k)

        return {
            "feature_names": FEATURE_NAMES,
            "importance_scores": w.tolist(),
            "marginal_probabilities": marginals.tolist(),
            "selected_indices": selected.tolist(),
            "selected_features": [FEATURE_NAMES[i] for i in selected],
            "redundancy_penalty": lam,
            "cardinality_penalty": mu,
            "target_features": k,
            "p_layers": self.config.p_layers,
            "optimizer": "COBYLA",
            "optimizer_iterations": int(result.nfev),
            "expected_cost_energy": float(qaoa_circuit(result.x)),
            "classical_cost_constant": float(const),
        }

    @staticmethod
    def _adjust_to_target(selected: np.ndarray, w: np.ndarray, k: int) -> np.ndarray:
        selected = sorted(set(int(i) for i in selected))
        if len(selected) == k:
            return np.asarray(selected)
        all_idx = list(range(len(w)))
        if len(selected) < k:
            missing = sorted(set(all_idx) - set(selected), key=lambda i: -w[i])
            selected.extend(missing[: k - len(selected)])
        else:
            by_score = sorted(selected, key=lambda i: w[i])
            selected = by_score[:k]
        return np.asarray(sorted(selected))


def verify_hamiltonian(X: np.ndarray, y: np.ndarray, n_check: int = 8) -> float:
    n = X.shape[1]
    w = mutual_information_scores(X, y)
    c = correlation_penalties(X)
    tri = np.triu_indices(n, k=1)
    c_sum = float(c[tri].sum())
    lam = 0.3
    mu = 0.5
    k = 6
    h = w / 2.0 - lam * (c.sum(axis=1) - 1.0) / 4.0 - mu * n / 2.0 + k * mu
    j_pair = lam * c / 4.0 + mu / 2.0
    const = (
        -w.sum() / 2.0
        + lam * c_sum / 4.0
        + mu * n * (n + 1) / 4.0
        - k * mu * n
        + mu * k**2
    )

    def classical_cost(z: np.ndarray) -> float:
        return float(-w @ z + lam * (z @ c @ z - z @ z) / 2.0 + mu * (z.sum() - k) ** 2)

    def hamiltonian_value(z: np.ndarray) -> float:
        sigma = 1.0 - 2.0 * z
        value = const
        value += float(h @ sigma)
        for i, j in zip(*tri):
            value += j_pair[i, j] * sigma[i] * sigma[j]
        return float(value)

    rng = np.random.default_rng(1)
    max_err = 0.0
    for _ in range(n_check):
        z = rng.integers(0, 2, size=n).astype(float)
        max_err = max(max_err, abs(classical_cost(z) - hamiltonian_value(z)))
    return max_err


def save_selection(selection: dict, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(selection, indent=2), encoding="utf-8")


def load_selection(path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
