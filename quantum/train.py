import json

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from quantum.config import VQCConfig
from quantum.dummy_data import load_dataset
from quantum.vqc import BalancedFocalLoss, HybridVQC


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_loaders(data: dict, batch_size: int) -> tuple[DataLoader, DataLoader, DataLoader]:
    X_train = torch.tensor(data["X_train"], dtype=torch.float32)
    y_train = torch.tensor(data["y_train"], dtype=torch.float32)
    X_val = torch.tensor(data["X_val"], dtype=torch.float32)
    y_val = torch.tensor(data["y_val"], dtype=torch.float32)
    X_test = torch.tensor(data["X_test"], dtype=torch.float32)
    y_test = torch.tensor(data["y_test"], dtype=torch.float32)
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size)
    return train_loader, val_loader, test_loader


def train_vqc(
    data: dict,
    selected_indices: list[int],
    config: VQCConfig,
) -> HybridVQC:
    seed_everything(42)
    train_loader, val_loader, _ = build_loaders(data, config.batch_size)

    n_qubits = len(selected_indices)
    model = HybridVQC(n_qubits=n_qubits, config=config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    loss_fn = BalancedFocalLoss(
        alpha=config.alpha,
        gamma=config.gamma,
        label_smoothing=config.label_smoothing,
        confidence_penalty=config.confidence_penalty,
    )

    config.log_file.parent.mkdir(parents=True, exist_ok=True)
    with config.log_file.open("w", encoding="utf-8") as fp:
        fp.write("")

    best_val_acc = 0.0

    for epoch in range(1, config.epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch[:, selected_indices]
            optimizer.zero_grad()
            logits = model(X_batch).squeeze(-1)
            loss = loss_fn(logits, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * X_batch.size(0)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)
        scheduler.step()

        train_loss = running_loss / max(total, 1)
        train_acc = correct / max(total, 1)
        val_loss, val_acc = evaluate_epoch(model, val_loader, selected_indices, loss_fn)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "selected_indices": selected_indices,
                    "config": {
                        "n_qubits": n_qubits,
                        "qml_layers": config.qml_layers,
                        "epochs": epoch,
                    },
                },
                config.checkpoint_file,
            )

        record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "train_acc": round(train_acc, 6),
            "val_loss": round(val_loss, 6),
            "val_acc": round(val_acc, 6),
            "lr": round(scheduler.get_last_lr()[0], 8),
        }
        with config.log_file.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(record) + "\n")

    return model


def evaluate_epoch(model, loader: DataLoader, selected_indices: list[int], loss_fn) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch[:, selected_indices]
            logits = model(X_batch).squeeze(-1)
            loss = loss_fn(logits, y_batch)
            total_loss += loss.item() * X_batch.size(0)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)
    return total_loss / max(total, 1), correct / max(total, 1)


def predict_proba(model: HybridVQC, X: np.ndarray, selected_indices: list[int]) -> np.ndarray:
    model.eval()
    X_t = torch.tensor(X[:, selected_indices], dtype=torch.float32)
    with torch.no_grad():
        logits = model(X_t).squeeze(-1)
        return torch.sigmoid(logits).numpy()
