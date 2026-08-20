"""Hybrid quantum-classical VQC: model, training, and inference.

The model combines a quantum layer (angle embedding +
strongly-entangling ansatz, P(Z) expectation per qubit) with a small
classical head.  The default quantum layer is **QuantumLayerTorch** — an
exact, differentiable torch-native statevector simulator (complex128,
CPU/GPU) that matches the PennyLane circuit to within float rounding.
The legacy PennyLane QNode path (default.qubit + backprop) is retained
behind ``VQCConfig.qnode_impl="pennylane"`` for cross-verification.

Training uses a focal loss with label smoothing and a confidence
penalty; the torch side (head, loss, optimizer) runs on GPU by default.
Checkpoints bundle state_dict + metadata so inference can reproduce the
exact training-time transformation.
"""

import json
from pathlib import Path

import numpy as np
import pennylane as qml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from quantum.config import VQCConfig


def resolve_device():
    """CUDA when available, else CPU. The torch-side tensors (head,
    loss, optimizer) run on this device; the quantum layer executes on
    its own backend (see qnode_backend_name)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def qnode_backend_name(cfg=None):
    """Resolved quantum circuit backend name: "torch-native" (default),
    "pennylane", or "pennylane-gpu"."""
    cfg = cfg or VQCConfig()
    impl = getattr(cfg, "qnode_impl", "auto")
    if impl in ("auto", "torch"):
        return "torch-native"
    return "pennylane"


def _make_qnode_device(n_qubits, cfg):
    """GPU-first QNode device: default.qubit.torch on CUDA when available
    (PennyLane < 0.39), else default.qubit on CPU. Returns (dev, on_gpu)."""
    backend = getattr(cfg, "qnode_backend", "auto")
    if backend in ("auto", "torch") and torch.cuda.is_available():
        try:
            dev = qml.device(
                "default.qubit.torch", wires=n_qubits, torch_device="cuda"
            )
            return dev, True
        except Exception:
            pass
    return qml.device("default.qubit", wires=n_qubits), False


class QuantumLayer(nn.Module):
    def __init__(self, n_qubits, cfg):
        super().__init__()
        # GPU-first simulator (see _make_qnode_device); the reference CPU
        # default.qubit + analytic backprop was measured fastest among the
        # CPU backends for these small circuits.
        dev, self.on_gpu = _make_qnode_device(n_qubits, cfg)
        weight_shape = qml.StronglyEntanglingLayers.shape(
            n_layers=cfg.qml_layers, n_wires=n_qubits
        )
        self.weights = nn.Parameter(0.1 * torch.randn(*weight_shape))
        self.broadcast = bool(getattr(cfg, "broadcast_qnode", True))

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(features, weights):
            qml.AngleEmbedding(features, wires=range(n_qubits), rotation="Y")
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self.circuit = circuit

    def _run(self, x, w):
        if self.broadcast:
            # Batched QNode execution: a (batch, n_features) input is
            # broadcast through the circuit by PennyLane, replacing the
            # per-row Python loop. Verified bit-identical to the loop
            # path (max |diff| = 0.0) before enabling.
            out = self.circuit(x, w)
            if isinstance(out, (list, tuple)):
                out = torch.stack(out, dim=1)
            out = out.float()
            return out if out.dim() == 2 else out.unsqueeze(0)
        rows = [self.circuit(row, w) for row in x]
        return torch.stack([torch.stack(row).float() for row in rows])

    def forward(self, x):
        if self.on_gpu:
            # QNode executes on CUDA: inputs and weights stay on the GPU
            # tensor device, no .cpu() hop.
            try:
                return self._run(x, self.weights).to(self.weights.device)
            except Exception:
                # Some GPU backends reject batched execution; drop to the
                # per-row loop once and keep it for the model lifetime.
                self.broadcast = False
                return self._run(x, self.weights).to(self.weights.device)
        # The QNode executes on CPU; .cpu()/.to() are differentiable
        # identity ops in torch, so the CUDA parameter graph stays intact
        # (verified: grads flow back to the GPU weights).
        if x.is_cuda:
            x = x.cpu()
        # Cache the CPU copy across forwards, invalidated whenever the
        # parameter is mutated in place (optimizer step bumps _version,
        # .to()/.load_state_dict() reallocate -> data_ptr changes). The
        # cached tensor keeps its autograd graph, so training grads still
        # flow back to the GPU weights on a cache miss.
        key = (self.weights._version, self.weights.data_ptr())
        cached = getattr(self, "_w_cpu_key", None)
        if cached != key:
            self._w_cpu = self.weights.cpu()
            self._w_cpu_key = key
        return self._run(x, self._w_cpu).to(self.weights.device)


class QuantumLayerTorch(nn.Module):
    """Exact torch-native statevector quantum layer (complex128).

    Replicates the PennyLane circuit:
        AngleEmbedding(x, rotation="Y") + StronglyEntanglingLayers(weights)
        + [PauliZ(i) for i in range(n_qubits)]

    Same ``weights`` parameter name and shape as ``QuantumLayer`` so
    existing checkpoints remain loadable without modification.
    """

    def __init__(self, n_qubits, cfg):
        super().__init__()
        self.n_qubits = n_qubits
        if n_qubits > 1:
            self._ranges = tuple(
                (l % (n_qubits - 1)) + 1 for l in range(cfg.qml_layers)
            )
        else:
            self._ranges = (0,) * cfg.qml_layers
        weight_shape = qml.StronglyEntanglingLayers.shape(
            n_layers=cfg.qml_layers, n_wires=n_qubits
        )
        self.weights = nn.Parameter(0.1 * torch.randn(*weight_shape))
        self._build_gate_tables()

    def _build_gate_tables(self):
        """Precompute wire-structure tables: CNOT permutations and Z masks.

        CNOT permutations: for each (control, target) wire pair in the
        entangling layers, the permutation of computational-basis indices
        (MSB: wire w = bit n-1-w) that swaps |b_c=1, b_t=0> <-> |b_c=1,
        b_t=1>.  One index_select applies any CNOT.

        Z masks: bit (n-1-w) = 0 -> +1, = 1 -> -1, so expval(PauliZ(w))
        = sum_b p_b * z_mask[w, b].
        """
        n = self.n_qubits
        n_states = 1 << n
        bits = torch.arange(n_states, dtype=torch.long)

        self._cnot_perms = {}
        for l in range(len(self._ranges)):
            r = self._ranges[l]
            for w in range(n):
                control, target = w, (w + r) % n
                if (control, target) in self._cnot_perms:
                    continue
                perm = torch.arange(n_states, dtype=torch.long)
                ctrl_bit = (bits >> (n - 1 - control)) & 1
                tgt_bit = (bits >> (n - 1 - target)) & 1
                swap = (ctrl_bit == 1) & (tgt_bit == 0)
                lo = bits[swap]
                hi = lo | (1 << (n - 1 - target))
                perm[lo] = hi
                perm[hi] = lo
                self._cnot_perms[(control, target)] = perm

        self._z_masks = torch.stack(
            [1.0 - 2.0 * ((bits >> (n - 1 - w)) & 1).to(torch.float64)
             for w in range(n)]
        )
        self._cnot_perms_dev = {}

    def _apply_wire_rotation(self, state, w, mat):
        """Apply a 2x2 unitary to the qubit at position w in the state.

        Due to the reshape convention (k, 2, half), position w in the
        reshape actually targets bit (n-1-w) in the computational basis,
        which corresponds to PL's wire w (MSB convention).

        ``mat`` is (2, 2) (fixed gates) or (batch, 2, 2) (per-sample
        rotations).  One fused einsum per gate — no clone/slice copies.
        """
        k = 1 << w
        bsz = state.shape[0]
        half = state.shape[1] // (2 * k)
        sv = state.view(bsz, k, 2, half)
        if mat.dim() == 3:
            # mat (b, i, j) x sv (b, k, j, h) -> out (b, k, i, h)
            out = torch.einsum("bij,bkjh->bkih", mat, sv)
        else:
            out = torch.einsum("ij,bkjh->bkih", mat, sv)
        return out.reshape(bsz, -1)

    def _apply_cnot(self, state, control, target):
        """CNOT(control, target) on state (batch, 2^n).

        PennyLane uses MSB wire ordering: wire w = bit (n-1-w).
        The permutation over computational basis states is precomputed
        once per (control, target) pair; a single index_select applies it.
        """
        perm = self._cnot_perms[(control, target)]
        if perm.device != state.device:
            key = (control, target, state.device)
            cached = self._cnot_perms_dev.get(key)
            if cached is None:
                cached = perm.to(state.device)
                self._cnot_perms_dev[key] = cached
            perm = cached
        return state.index_select(1, perm)

    def _run_circuit(self, x):
        """Core circuit: embedding + entangling layers -> final state.

        Uses MSB wire ordering to match PennyLane: wire w = bit (n-1-w).
        """
        n = self.n_qubits
        bsz = x.shape[0]
        dtype = torch.complex128
        device = x.device

        # |0>^n initial state
        state = torch.zeros(bsz, 1 << n, dtype=dtype, device=device)
        state[:, 0] = 1.0
        x64 = x.to(torch.float64)

        # Angle embedding: RY(x_i) on wire w
        # _apply_wire_rotation(state, w, mat) targets bit (n-1-w) = PL wire w
        c = torch.cos(x64 / 2.0)
        s = torch.sin(x64 / 2.0)
        for w in range(n):
            ry = torch.stack([
                torch.stack([c[:, w] + 0j, -s[:, w] + 0j], dim=-1),
                torch.stack([s[:, w] + 0j, c[:, w] + 0j], dim=-1),
            ], dim=-2)
            state = self._apply_wire_rotation(state, w, ry)

        # StronglyEntanglingLayers: all Rot matrices are weight-derived, so
        # build them in one batched pass, then apply wire by wire.
        weights = self.weights
        L = weights.shape[0]
        w64 = weights.to(torch.float64)
        phi = w64[:, :, 0]
        theta = w64[:, :, 1]
        omega = w64[:, :, 2]
        ep = torch.exp(-1j * phi / 2.0)
        eq = torch.exp(1j * phi / 2.0)
        rz_phi = torch.stack([torch.stack([ep, torch.zeros_like(ep)], dim=-1),
                              torch.stack([torch.zeros_like(eq), eq], dim=-1)], dim=-2)
        ct = torch.cos(theta / 2.0)
        st = torch.sin(theta / 2.0)
        ry_theta = torch.stack([torch.stack([ct + 0j, -st + 0j], dim=-1),
                                torch.stack([st + 0j, ct + 0j], dim=-1)], dim=-2)
        eo = torch.exp(-1j * omega / 2.0)
        ef = torch.exp(1j * omega / 2.0)
        rz_omega = torch.stack([torch.stack([eo, torch.zeros_like(eo)], dim=-1),
                                torch.stack([torch.zeros_like(ef), ef], dim=-1)], dim=-2)
        rot_mats = rz_omega @ ry_theta @ rz_phi  # (L, n, 2, 2)

        for l in range(L):
            for w in range(n):
                state = self._apply_wire_rotation(state, w, rot_mats[l, w])
            for w in range(n):
                state = self._apply_cnot(state, w, (w + self._ranges[l]) % n)

        return state

    def forward(self, x):
        # Run the whole circuit on the weights device (CUDA when available).
        if x.device != self.weights.device:
            x = x.to(self.weights.device)
        state = self._run_circuit(x)
        probs = state.real ** 2 + state.imag ** 2
        masks = self._z_masks.to(probs.device)
        return (probs @ masks.T).float()


class HybridModel(nn.Module):
    def __init__(self, n_features, cfg):
        super().__init__()
        impl = getattr(cfg, "qnode_impl", "auto")
        if impl in ("auto", "torch"):
            self.quantum = QuantumLayerTorch(n_features, cfg)
        else:
            self.quantum = QuantumLayer(n_features, cfg)
        self.head = nn.Sequential(
            nn.Linear(n_features, cfg.hidden_units),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_units, 1),
        )

    def forward(self, x):
        return self.head(self.quantum(x)).squeeze(-1)


def focal_loss(logits, targets, cfg, weight=None):
    probs = torch.sigmoid(logits)
    soft = targets * (1 - cfg.label_smoothing) + cfg.label_smoothing * 0.5
    pt = probs * soft + (1 - probs) * (1 - soft)
    alpha_t = cfg.alpha * soft + (1 - cfg.alpha) * (1 - soft)
    focal = -alpha_t * (1 - pt).pow(cfg.gamma) * torch.log(pt.clamp(min=1e-6))
    entropy = -probs * torch.log(probs.clamp(min=1e-6)) - (1 - probs) * torch.log(
        (1 - probs).clamp(min=1e-6)
    )
    loss = focal.mean() - cfg.confidence_penalty * entropy.mean()
    if weight is not None:
        loss = loss * weight
    return loss


# Trained models are cached across calls (keyed by feature count + checkpoint
# identity) so repeated inference never rebuilds the PennyLane circuit or
# re-reads the checkpoint from disk. The cache is small and keyed by file
# mtime/size, so retraining invalidates the old entry automatically.
_MODEL_CACHE: dict[tuple, HybridModel] = {}
_MODEL_CACHE_MAX = 8


def load_vqc_model(n_features, cfg=None):
    """Build the HybridModel and load the saved checkpoint (evaluation + inference).

    Compatible with both the current bundle format (dict with "state_dict"
    and "metadata") and legacy bare-state_dict checkpoints. Results are
    cached: a second call with the same (n_features, checkpoint identity)
    returns the already-loaded model on the same device.
    """
    cfg = cfg or VQCConfig()
    stat = cfg.checkpoint_file.stat()
    key = (n_features, str(cfg.checkpoint_file), stat.st_mtime_ns, stat.st_size)
    cached = _MODEL_CACHE.get(key)
    if cached is not None:
        return cached
    model = HybridModel(n_features, cfg)
    ckpt = torch.load(cfg.checkpoint_file, map_location="cpu", weights_only=True)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.to(resolve_device())
    model.eval()
    if len(_MODEL_CACHE) >= _MODEL_CACHE_MAX:
        _MODEL_CACHE.clear()
    _MODEL_CACHE[key] = model
    return model


def predict_vqc(model, X):
    """P(real) for a (n, n_features) matrix from a trained VQC."""
    device = next(model.parameters()).device
    with torch.no_grad():
        logits = model(torch.tensor(np.asarray(X, dtype=np.float32), device=device))
    return torch.sigmoid(logits).cpu().numpy()


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


def _val_metrics(model, X_val, y_val, cfg):
    """Validation loss (same focal loss as training) + accuracy, eval-mode.

    Single forward pass: the logits feed both the loss and the probability
    threshold (previously the val set was evaluated twice per epoch).
    """
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        X = torch.tensor(np.asarray(X_val, dtype=np.float32), device=device)
        y = torch.tensor(np.asarray(y_val, dtype=np.float32), device=device)
        logits = model(X)
        val_loss = float(focal_loss(logits, y, cfg).item())
        probs = torch.sigmoid(logits).cpu().numpy()
    val_acc = float(((probs >= 0.5).astype(int) == np.asarray(y_val)).mean())
    return val_loss, val_acc


def train_vqc(features, labels, cfg=None, X_val=None, y_val=None, metadata=None):
    """Train the hybrid VQC on the QAOA-selected features.

    Training loop upgrades: cosine-annealed learning rate, gradient
    clipping, early stopping on validation loss with restore of the
    best-validation checkpoint (instead of keeping final-epoch weights).
    `X_val`/`y_val` are used only for monitoring/early stopping, never
    for gradient updates. The checkpoint is saved together with `metadata`
    (QAOA selection, feature ordering, scaler, configs) for inference.

    Class-balanced focal loss is applied if enabled in config.
    """
    cfg = cfg or VQCConfig()
    torch.manual_seed(cfg.seed)
    device = resolve_device()
    X = torch.tensor(np.asarray(features, dtype=np.float32), device=device)
    y = torch.tensor(np.asarray(labels, dtype=np.float32), device=device)
    loader = DataLoader(
        TensorDataset(X, y),
        batch_size=cfg.batch_size,
        shuffle=True,
    )
    model = HybridModel(features.shape[1], cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    monitor_val = X_val is not None and y_val is not None

    # Compute balanced class weight for focal loss (intervention A)
    # Weight for the positive class (REAL = 1) is inversely proportional to its frequency
    y_np = y.cpu().numpy()
    n_pos = int((y == 1).sum())  # class 1 = REAL in quantum convention
    n_neg = int((y == 0).sum())  # class 0 = FAKE in quantum convention
    if n_pos > 0 and n_neg > 0:
        balanced_weight = float(n_neg + n_pos) / (2.0 * max(n_pos, n_neg))  # ~ n_samples / (2 * max_class)
    else:
        balanced_weight = 1.0

    scheduler = None
    if cfg.lr_schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.epochs, eta_min=max(1e-5, cfg.learning_rate / 100.0)
        )
    elif cfg.lr_schedule == "plateau" and monitor_val:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=max(3, cfg.patience // 4)
        )

    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    log = []
    for epoch in range(cfg.epochs):
        model.train()
        total_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = focal_loss(model(xb), yb, cfg, weight=balanced_weight)
            loss.backward()
            if cfg.clip_grad is not None and cfg.clip_grad > 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad)
            optimizer.step()
            total_loss += loss.item() * len(xb)
        record = {
            "epoch": epoch + 1,
            "train_loss": total_loss / len(features),
            "lr": float(scheduler.get_last_lr()[0]) if scheduler else float(cfg.learning_rate),
        }
        if monitor_val:
            val_loss, val_acc = _val_metrics(model, X_val, y_val, cfg)
            record["val_loss"] = val_loss
            record["val_accuracy"] = val_acc
            if val_loss < best_val_loss - cfg.min_delta:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                best_epoch = epoch + 1
        log.append(record)
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(record["val_loss"] if monitor_val else record["train_loss"])
            else:
                scheduler.step()
        if monitor_val and best_state is not None and (epoch + 1 - best_epoch) > cfg.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    summary = {
        "epochs_run": epoch + 1,
        "best_epoch": best_epoch if best_state is not None else None,
        "best_val_loss": best_val_loss if best_state is not None else None,
        "early_stopped": best_state is not None and epoch + 1 - best_epoch > cfg.patience,
    }
    if cfg.save_checkpoint:
        cfg.checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": model.state_dict(),
                "metadata": _sanitize_metadata(metadata or {}),
                "training_summary": summary,
            },
            cfg.checkpoint_file,
        )
        with open(cfg.log_file, "w") as fh:
            for row in log:
                fh.write(json.dumps(row) + "\n")
    return model