import pennylane as qml
import torch
import torch.nn as nn

from quantum.config import VQCConfig


def build_qnode(n_qubits: int, qml_layers: int, device_name: str):
    dev = qml.device(device_name, wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method="adjoint")
    def qnode(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    return qnode


class QuantumFeatureBlock(nn.Module):
    def __init__(self, n_qubits: int, qml_layers: int, device_name: str = "lightning.qubit"):
        super().__init__()
        qnode = build_qnode(n_qubits, qml_layers, device_name)
        weight_shapes = {"weights": (qml_layers, n_qubits, 3)}
        self.qlayer = qml.qnn.TorchLayer(qnode, weight_shapes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.qlayer(x)


class HybridVQC(nn.Module):
    def __init__(self, n_qubits: int, config: VQCConfig):
        super().__init__()
        self.n_qubits = n_qubits
        self.quantum = QuantumFeatureBlock(n_qubits, config.qml_layers, config.device)
        self.head = nn.Sequential(
            nn.Linear(n_qubits, config.hidden_units),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_units, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        quantum_out = self.quantum(x)
        return self.head(quantum_out)


class BalancedFocalLoss(nn.Module):
    def __init__(
        self,
        alpha: float = 0.75,
        gamma: float = 2.0,
        label_smoothing: float = 0.1,
        confidence_penalty: float = 0.3,
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.confidence_penalty = confidence_penalty

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        smoothed = targets * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
        probs = torch.sigmoid(logits)
        pt = probs * smoothed + (1.0 - probs) * (1.0 - smoothed)
        bce = nn.functional.binary_cross_entropy_with_logits(logits, smoothed, reduction="none")
        focal = ((1.0 - pt) ** self.gamma) * bce
        weights = targets * self.alpha + (1.0 - targets) * (1.0 - self.alpha)
        loss = (focal * weights).mean()
        if self.confidence_penalty > 0:
            loss = loss + self.confidence_penalty * ((probs - smoothed) ** 2).mean()
        return loss
