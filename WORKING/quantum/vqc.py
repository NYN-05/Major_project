import json

import numpy as np
import pennylane as qml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from quantum.config import VQCConfig


class QuantumLayer(nn.Module):
    def __init__(self, n_qubits, cfg):
        super().__init__()
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


def train_vqc(features, labels, cfg=None):
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
        log.append({"epoch": epoch + 1, "train_loss": total_loss / len(features)})
    cfg.checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), cfg.checkpoint_file)
    with open(cfg.log_file, "w") as fh:
        for row in log:
            fh.write(json.dumps(row) + "\n")
    return model