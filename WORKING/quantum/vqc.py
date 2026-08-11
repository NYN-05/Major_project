import json
from pathlib import Path

import numpy as np
import pennylane as qml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from quantum.config import VQCConfig


class QuantumLayer(nn.Module):
    def __init__(self, n_qubits, cfg):
        super().__init__()
        # Reference statevector simulator + analytic backprop: measured
        # fastest for these small circuits (accelerated backends and adjoint
        # were slower in min-of-3 isolated benchmarks on this host).
        dev = qml.device("default.qubit", wires=n_qubits)
        weight_shape = qml.StronglyEntanglingLayers.shape(
            n_layers=cfg.qml_layers, n_wires=n_qubits
        )
        self.weights = nn.Parameter(0.1 * torch.randn(*weight_shape))

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(features, weights):
            qml.AngleEmbedding(features, wires=range(n_qubits), rotation="Y")
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self.circuit = circuit

    def forward(self, x):
        rows = [self.circuit(row, self.weights) for row in x]
        return torch.stack([torch.stack(row).float() for row in rows])


class HybridModel(nn.Module):
    def __init__(self, n_features, cfg):
        super().__init__()
        self.quantum = QuantumLayer(n_features, cfg)
        self.head = nn.Sequential(
            nn.Linear(n_features, cfg.hidden_units),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_units, 1),
        )

    def forward(self, x):
        return self.head(self.quantum(x)).squeeze(-1)


def focal_loss(logits, targets, cfg):
    probs = torch.sigmoid(logits)
    soft = targets * (1 - cfg.label_smoothing) + cfg.label_smoothing * 0.5
    pt = probs * soft + (1 - probs) * (1 - soft)
    alpha_t = cfg.alpha * soft + (1 - cfg.alpha) * (1 - soft)
    focal = -alpha_t * (1 - pt).pow(cfg.gamma) * torch.log(pt.clamp(min=1e-6))
    entropy = -probs * torch.log(probs.clamp(min=1e-6)) - (1 - probs) * torch.log(
        (1 - probs).clamp(min=1e-6)
    )
    return focal.mean() - cfg.confidence_penalty * entropy.mean()


def _sanitize_metadata(obj):
    """Recursively convert metadata to JSON-safe (weights_only-loadable) values.

    torch.load defaults to weights_only=True since PyTorch 2.6; Path objects
    in dataclass configs would otherwise break checkpoint reloading.
    """
    if isinstance(obj, dict):
        return {str(k): _sanitize_metadata(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_metadata(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj


def load_vqc_model(n_features, cfg=None):
    """Build the HybridModel and load the saved checkpoint (evaluation + inference).

    Compatible with both the current bundle format (dict with "state_dict"
    and "metadata") and legacy bare-state_dict checkpoints.
    """
    cfg = cfg or VQCConfig()
    model = HybridModel(n_features, cfg)
    ckpt = torch.load(cfg.checkpoint_file, map_location="cpu")
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    return model


def predict_vqc(model, X):
    """P(real) for a (n, n_features) matrix from a trained VQC."""
    with torch.no_grad():
        logits = model(torch.tensor(np.asarray(X, dtype=np.float32)))
    return torch.sigmoid(logits).numpy()


def _val_metrics(model, X_val, y_val, cfg):
    """Validation loss (same focal loss as training) + accuracy, eval-mode."""
    model.eval()
    with torch.no_grad():
        X = torch.tensor(np.asarray(X_val, dtype=np.float32))
        y = torch.tensor(np.asarray(y_val, dtype=np.float32))
        val_loss = float(focal_loss(model(X), y, cfg).item())
        probs = predict_vqc(model, X_val)
    val_acc = float(((probs >= 0.5).astype(int) == np.asarray(y_val)).mean())
    return val_loss, val_acc


def train_vqc(features, labels, cfg=None, X_val=None, y_val=None, metadata=None):
    """Train the hybrid VQC on the QAOA-selected features.

    Architecture, optimizer, loss, and hyperparameters are unchanged;
    `X_val`/`y_val` only add per-epoch validation monitoring (never used
    for updates). The checkpoint is saved together with `metadata`
    (QAOA selection, feature ordering, scaler, configs) for inference.
    """
    cfg = cfg or VQCConfig()
    torch.manual_seed(cfg.seed)
    X = torch.tensor(np.asarray(features, dtype=np.float32))
    y = torch.tensor(np.asarray(labels, dtype=np.float32))
    loader = DataLoader(
        TensorDataset(X, y),
        batch_size=cfg.batch_size,
        shuffle=True,
    )
    model = HybridModel(features.shape[1], cfg)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    monitor_val = X_val is not None and y_val is not None
    log = []
    for epoch in range(cfg.epochs):
        model.train()
        total_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = focal_loss(model(xb), yb, cfg)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(xb)
        record = {"epoch": epoch + 1, "train_loss": total_loss / len(features)}
        if monitor_val:
            val_loss, val_acc = _val_metrics(model, X_val, y_val, cfg)
            record["val_loss"] = val_loss
            record["val_accuracy"] = val_acc
        log.append(record)
    cfg.checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": model.state_dict(), "metadata": _sanitize_metadata(metadata or {})},
        cfg.checkpoint_file,
    )
    with open(cfg.log_file, "w") as fh:
        for row in log:
            fh.write(json.dumps(row) + "\n")
    return model