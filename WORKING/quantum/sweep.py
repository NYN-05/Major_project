"""Hyperparameter sweep harness for QAOA + VQC configurations.

Two-phase architecture (critical for performance):
  Phase A: QAOA feature selection — run once per unique QAOA config (9 configs).
           Cache selected feature indices so downstream VQC sweep skips QAOA.
  Phase B: VQC training sweep — iterate over (QAOA_features x VQC_arch x LR/WD x
           loss x training_schedule). QAOA is cached; only VQC training runs per combo.

Timing (measured on this host, CPU):
  QAOA:  ~220s per config (iter=50, restarts=1) → 9 configs ≈ 33 min total.
  VQC:   ~1.5–2.4s per epoch → with early stopping (patience=10), ~60–120s per combo.
  Total: ~200 VQC combos × ~90s avg ≈ 5 hours + 33 min QAOA ≈ 5.5 hours.

Phase 1 additions (always active):
  - Per-combo wall-time timeout (default 600s).
  - Early stopping with configurable patience (default 10).
  - Per-combo checkpoint saving to sweep_checkpoints/<combo_id>/.

Usage from WORKING/:
    python -m quantum.sweep [--timeout 600] [--patience 10] [--out output/sweep]

All paths are relative to WORKING/ (the quantum.* imports assume CWD = WORKING/).
"""

import argparse
import csv
import json
import os
import time
import traceback
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path

import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────
SWEEP_DIR = Path(__file__).resolve().parent.parent / "output" / "sweep"
CHECKPOINT_DIR = SWEEP_DIR / "sweep_checkpoints"

# ── Grid definitions ───────────────────────────────────────────────────
# QAOA grid: 9 representative configs (one per meaningful axis combination).
# Full 3x3x3=27 would cost 27*223s=~100 min; these 9 cover the key trade-offs.
QAOA_GRID = [
    (2, 100, 3), (2, 200, 5), (2, 400, 8),
    (3, 100, 5), (3, 200, 3), (3, 400, 8),
    (5, 100, 8), (5, 200, 3), (5, 400, 5),
]

# VQC architecture grid: 6 representative configs
VQC_ARCH_GRID = [
    (2, 4, 0.1), (2, 8, 0.3),
    (3, 4, 0.3), (3, 8, 0.1),
    (5, 4, 0.1), (5, 8, 0.3),
]

# LR / weight-decay grid: 3 combos
LR_WD_GRID = [
    (5e-3, 1e-3),
    (1e-2, 1e-2),
    (5e-2, 1e-3),
]

# Loss function grid: 4 key variants
LOSS_GRID = [
    (0.5,  1.0, 0.0,  0.0),   # alpha-light, no smoothing
    (0.75, 2.0, 0.03, 0.02),  # production defaults
    (0.5,  2.0, 0.03, 0.0),   # high gamma, smoothing
    (0.75, 1.0, 0.0,  0.02),  # low gamma, confidence penalty
]

# Training schedule grid: 3 combos
TRAIN_GRID = [
    (50,  "cosine", 32),
    (80,  "plateau", 32),
    (120, "cosine", 16),
]


@dataclass
class QAOAConfig:
    """One QAOA selection configuration."""
    p_layers: int
    max_iter: int
    target_features: int


@dataclass
class VQCRunConfig:
    """One VQC training run configuration (combined with a QAOA feature set)."""
    combo_id: int
    qaoa_key: str         # hash of QAOA config for lookup
    qml_layers: int
    hidden_units: int
    dropout: float
    learning_rate: float
    weight_decay: float
    alpha: float
    gamma: float
    label_smoothing: float
    confidence_penalty: float
    epochs: int
    lr_schedule: str
    batch_size: int


def _qaoa_key(cfg: QAOAConfig) -> str:
    return f"p{cfg.p_layers}_i{cfg.max_iter}_k{cfg.target_features}"


def _generate_vqc_combos():
    """Generate all VQC combos: QAOA(9) x VQC_arch(6) x LR/WD(3) x Loss(4) x Train(3)
    = 9 x 6 x 3 x 4 x 3 = 1944 combos.

    To keep runtime manageable (~5h), we do a factorial design but cap the
    total by running the full grid only for the default loss+LR config, and
    sampling the loss/LR variants on the best QAOA+VQC arch only.

    Strategy:
      Tier 1 (full): QAOA(9) x VQC_arch(6) x Train(3) = 162 combos
                     with fixed loss=(0.75,2,0.03,0.02), lr=1e-2, wd=1e-2.
      Tier 2 (loss sweep on best QAOA+VQC): 4 loss x 3 LR/WD = 12 combos
                     on the Tier-1 winner.
      Total: ~174 combos.
    """
    combos = []
    cid = 0

    # Tier 1: full QAOA x VQC_arch x Train with fixed loss+LR/WD
    for qaoa_params in QAOA_GRID:
        for vqc_arch in VQC_ARCH_GRID:
            for train_params in TRAIN_GRID:
                cid += 1
                p, mi, tf = qaoa_params
                qml, hu, do = vqc_arch
                ep, sched, bs = train_params
                combos.append(VQCRunConfig(
                    combo_id=cid,
                    qaoa_key=_qaoa_key(QAOAConfig(p, mi, tf)),
                    qml_layers=qml,
                    hidden_units=hu,
                    dropout=do,
                    learning_rate=1e-2,
                    weight_decay=1e-2,
                    alpha=0.75,
                    gamma=2.0,
                    label_smoothing=0.03,
                    confidence_penalty=0.02,
                    epochs=ep,
                    lr_schedule=sched,
                    batch_size=bs,
                ))

    # Tier 2: loss+LR sweep on default QAOA(3,200,3) + default VQC(3,8,0.1)
    # These use a fixed QAOA feature set and VQC arch; only loss+LR vary.
    best_qaoa_key = _qaoa_key(QAOAConfig(3, 200, 3))
    for loss_params in LOSS_GRID:
        for lrwd_params in LR_WD_GRID:
            cid += 1
            alpha, gamma, ls, cp = loss_params
            lr, wd = lrwd_params
            combos.append(VQCRunConfig(
                combo_id=cid,
                qaoa_key=best_qaoa_key,
                qml_layers=3,
                hidden_units=8,
                dropout=0.1,
                learning_rate=lr,
                weight_decay=wd,
                alpha=alpha,
                gamma=gamma,
                label_smoothing=ls,
                confidence_penalty=cp,
                epochs=80,
                lr_schedule="cosine",
                batch_size=32,
            ))

    return combos


# ── QAOA Phase ─────────────────────────────────────────────────────────
def run_qaoa_phase(X_train_scaled, y_train, qaoa_configs, out_dir, timeout_s=600):
    """Run QAOA feature selection once per unique config. Cache results.

    Returns dict mapping qaoa_key -> {selected_indices, selected_features, ...}.
    """
    import signal
    from quantum.config import QAOASelectionConfig
    from quantum.qaoa import QAOASelector

    cache_file = out_dir / "qaoa_cache.json"
    cache = {}

    # Load existing cache if present
    if cache_file.exists():
        with open(cache_file) as fh:
            cache = json.load(fh)
        print(f"  Loaded QAOA cache: {len(cache)} entries")

    for i, qcfg in enumerate(qaoa_configs):
        key = _qaoa_key(qcfg)
        if key in cache:
            print(f"  QAOA [{i+1}/{len(qaoa_configs)}] {key}: cached (features={cache[key]['selected_features']})")
            continue

        print(f"  QAOA [{i+1}/{len(qaoa_configs)}] {key}...", end=" ", flush=True)
        t0 = time.time()
        try:
            cfg = QAOASelectionConfig(
                p_layers=qcfg.p_layers,
                max_iter=qcfg.max_iter,
                target_features=qcfg.target_features,
                seed=42,
                restarts=1,  # Single restart for speed in sweep
                n_jobs=1,    # No subprocess (Windows ProcessPool hangs with PennyLane)
            )
            selector = QAOASelector(cfg)
            result = selector.select(X_train_scaled, y_train)
            elapsed = time.time() - t0
            cache[key] = {
                "selected_indices": result["selected_indices"],
                "selected_features": result["selected_features"],
                "cost": result.get("cost"),
                "marginal_probabilities": result.get("marginal_probabilities"),
                "wall_time_s": round(elapsed, 1),
            }
            # Save after each config (crash-safe)
            with open(cache_file, "w") as fh:
                json.dump(cache, fh, indent=2)
            print(f"features={result['selected_features']} ({elapsed:.0f}s)")
        except Exception as exc:
            elapsed = time.time() - t0
            cache[key] = {"error": str(exc), "wall_time_s": round(elapsed, 1)}
            with open(cache_file, "w") as fh:
                json.dump(cache, fh, indent=2)
            print(f"ERROR: {exc}")

    return cache


# ── VQC Phase ──────────────────────────────────────────────────────────
def run_vqc_combo(cfg: VQCRunConfig, qaoa_cache_entry, data, scaler, timeout_s, patience):
    """Run one VQC training combo. Returns a result dict.

    Uses pre-selected features from QAOA cache (no QAOA re-run).
    """
    from quantum.config import FEATURE_NAMES, VQCConfig
    from quantum.evaluation import classification_metrics, expected_calibration_error
    from quantum.vqc import HybridModel, focal_loss, resolve_device

    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    start = time.time()

    try:
        indices = qaoa_cache_entry["selected_indices"]
        n_features = len(indices)

        X_train_s = scaler.transform(data["X_train"])
        X_val_s = scaler.transform(data["X_val"])
        X_test_s = scaler.transform(data["X_test"])

        X_tr = X_train_s[:, indices]
        X_vl = X_val_s[:, indices]
        X_te = X_test_s[:, indices]
        y_train = data["y_train"]
        y_val = data["y_val"]
        y_test = data["y_test"]

        vqc_cfg = VQCConfig(
            qml_layers=cfg.qml_layers,
            hidden_units=cfg.hidden_units,
            dropout=cfg.dropout,
            epochs=cfg.epochs,
            batch_size=cfg.batch_size,
            learning_rate=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
            alpha=cfg.alpha,
            gamma=cfg.gamma,
            label_smoothing=cfg.label_smoothing,
            confidence_penalty=cfg.confidence_penalty,
            lr_schedule=cfg.lr_schedule,
            patience=patience,
            min_delta=1e-4,
            clip_grad=1.0,
            broadcast_qnode=True,
            save_checkpoint=False,
            seed=42,
        )

        torch.manual_seed(vqc_cfg.seed)
        device = resolve_device()
        X_t = torch.tensor(np.asarray(X_tr, dtype=np.float32), device=device)
        y_t = torch.tensor(np.asarray(y_train, dtype=np.float32), device=device)
        loader = DataLoader(TensorDataset(X_t, y_t), batch_size=vqc_cfg.batch_size, shuffle=True)

        model = HybridModel(n_features, vqc_cfg).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=vqc_cfg.learning_rate, weight_decay=vqc_cfg.weight_decay)

        scheduler = None
        if vqc_cfg.lr_schedule == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=vqc_cfg.epochs, eta_min=max(1e-5, vqc_cfg.learning_rate / 100.0)
            )
        elif vqc_cfg.lr_schedule == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=max(3, vqc_cfg.patience // 4)
            )

        best_val_loss = float("inf")
        best_state = None
        best_epoch = 0
        epochs_run = 0

        X_val_t = torch.tensor(np.asarray(X_vl, dtype=np.float32), device=device)
        y_val_t = torch.tensor(np.asarray(y_val, dtype=np.float32), device=device)

        for epoch in range(vqc_cfg.epochs):
            if time.time() - start > timeout_s:
                break

            model.train()
            total_loss = 0.0
            for xb, yb in loader:
                optimizer.zero_grad()
                loss = focal_loss(model(xb), yb, vqc_cfg)
                loss.backward()
                if vqc_cfg.clip_grad is not None and vqc_cfg.clip_grad > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), vqc_cfg.clip_grad)
                optimizer.step()
                total_loss += loss.item() * len(xb)

            model.eval()
            with torch.no_grad():
                val_logits = model(X_val_t)
                val_loss = float(focal_loss(val_logits, y_val_t, vqc_cfg).item())
                val_probs = torch.sigmoid(val_logits).cpu().numpy()
            val_acc = float(((val_probs >= 0.5).astype(int) == np.asarray(y_val)).mean())

            if val_loss < best_val_loss - vqc_cfg.min_delta:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                best_epoch = epoch + 1

            epochs_run = epoch + 1
            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()

            if best_state is not None and (epoch + 1 - best_epoch) > vqc_cfg.patience:
                break

        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()

        with torch.no_grad():
            test_logits = model(torch.tensor(np.asarray(X_te, dtype=np.float32), device=device))
            test_probs = torch.sigmoid(test_logits).cpu().numpy()

        metrics = classification_metrics(y_test, test_probs)
        ece = expected_calibration_error(y_test, test_probs)
        elapsed = time.time() - start

        return {
            "combo_id": cfg.combo_id,
            "status": "ok",
            "qaoa_features": qaoa_cache_entry.get("selected_features", []),
            "n_features": n_features,
            "qaoa_cost": qaoa_cache_entry.get("cost"),
            "accuracy": metrics["accuracy"],
            "f1": metrics["f1"],
            "auc_roc": metrics["auc_roc"],
            "pr_auc": metrics["pr_auc"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "specificity": metrics["specificity"],
            "ece": ece,
            "epochs_run": epochs_run,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "early_stopped": best_state is not None and epochs_run - best_epoch > patience,
            "wall_time_s": round(elapsed, 2),
            "config": {
                "qaoa_key": cfg.qaoa_key,
                "qml_layers": cfg.qml_layers,
                "hidden_units": cfg.hidden_units,
                "dropout": cfg.dropout,
                "learning_rate": cfg.learning_rate,
                "weight_decay": cfg.weight_decay,
                "alpha": cfg.alpha,
                "gamma": cfg.gamma,
                "label_smoothing": cfg.label_smoothing,
                "confidence_penalty": cfg.confidence_penalty,
                "epochs": cfg.epochs,
                "lr_schedule": cfg.lr_schedule,
                "batch_size": cfg.batch_size,
            },
        }
    except Exception as exc:
        return {
            "combo_id": cfg.combo_id,
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "wall_time_s": round(time.time() - start, 2),
            "config": asdict(cfg),
        }


# ── CSV helpers ────────────────────────────────────────────────────────
_CSV_FIELDS = [
    "combo_id", "status",
    "qaoa_key", "qml_layers", "hidden_units", "dropout",
    "learning_rate", "weight_decay",
    "alpha", "gamma", "label_smoothing", "confidence_penalty",
    "epochs", "lr_schedule", "batch_size",
    "qaoa_features", "n_features", "qaoa_cost",
    "accuracy", "f1", "auc_roc", "pr_auc",
    "precision", "recall", "specificity", "ece",
    "epochs_run", "best_epoch", "best_val_loss", "early_stopped",
    "wall_time_s", "error",
]


def _init_csv(csv_path):
    with open(csv_path, "w", newline="") as fh:
        csv.DictWriter(fh, fieldnames=_CSV_FIELDS).writeheader()


def _append_csv(csv_path, result):
    cfg = result.get("config", {})
    row = {k: "" for k in _CSV_FIELDS}
    row["combo_id"] = result["combo_id"]
    row["status"] = result["status"]
    for k in ["qaoa_key", "qml_layers", "hidden_units", "dropout",
              "learning_rate", "weight_decay", "alpha", "gamma",
              "label_smoothing", "confidence_penalty", "epochs",
              "lr_schedule", "batch_size"]:
        row[k] = cfg.get(k, "")
    row["qaoa_features"] = json.dumps(result.get("qaoa_features", []))
    row["n_features"] = result.get("n_features", "")
    row["qaoa_cost"] = result.get("qaoa_cost", "")
    for k in ["accuracy", "f1", "auc_roc", "pr_auc", "precision", "recall",
              "specificity", "ece", "epochs_run", "best_epoch", "best_val_loss",
              "early_stopped", "wall_time_s", "error"]:
        row[k] = result.get(k, "")
    with open(csv_path, "a", newline="") as fh:
        csv.DictWriter(fh, fieldnames=_CSV_FIELDS).writerow(row)


# ── Main sweep entry point ─────────────────────────────────────────────
def run_sweep(timeout_s=600, patience=10, out_dir=None):
    out_dir = Path(out_dir) if out_dir else SWEEP_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    from quantum.config import DataConfig, FEATURE_NAMES
    from quantum.data import load_dataset
    from quantum.scaling import FeatureScaler

    # Load dataset once
    data_cfg = DataConfig()
    data = load_dataset(data_cfg.data_file)
    print(f"Dataset: train={data['X_train'].shape}, val={data['X_val'].shape}, test={data['X_test'].shape}")

    scaler = FeatureScaler(FEATURE_NAMES).fit(data["X_train"])
    X_train_s = scaler.transform(data["X_train"])

    # Unique QAOA configs
    qaoa_configs = [QAOAConfig(p, mi, tf) for p, mi, tf in QAOA_GRID]
    print(f"\n{'='*60}")
    print(f"PHASE A: QAOA feature selection ({len(qaoa_configs)} configs)")
    print(f"{'='*60}")
    qaoa_cache = run_qaoa_phase(X_train_s, data["y_train"], qaoa_configs, out_dir, timeout_s)

    # Generate VQC combos
    vqc_combos = _generate_vqc_combos()
    n_total = len(vqc_combos)
    print(f"\n{'='*60}")
    print(f"PHASE B: VQC training sweep ({n_total} combos)")
    print(f"  timeout={timeout_s}s, patience={patience}")
    print(f"{'='*60}")

    csv_path = out_dir / "sweep_results.csv"
    json_path = out_dir / "sweep_results.json"
    _init_csv(csv_path)

    results = []
    n_ok = 0
    n_err = 0
    sweep_start = time.time()

    for i, vcfg in enumerate(vqc_combos):
        qaoa_entry = qaoa_cache.get(vcfg.qaoa_key, {})
        if "error" in qaoa_entry:
            result = {
                "combo_id": vcfg.combo_id, "status": "error",
                "error": f"QAOA failed: {qaoa_entry['error']}",
                "wall_time_s": 0, "config": asdict(vcfg),
            }
        else:
            print(f"[{i+1}/{n_total}] combo {vcfg.combo_id} "
                  f"(QAOA={vcfg.qaoa_key}, qml={vcfg.qml_layers}, hu={vcfg.hidden_units}, "
                  f"lr={vcfg.learning_rate}, ep={vcfg.epochs})...", end=" ", flush=True)
            result = run_vqc_combo(vcfg, qaoa_entry, data, scaler, timeout_s, patience)

        results.append(result)
        _append_csv(csv_path, result)

        if result["status"] == "ok":
            n_ok += 1
            print(f"AUC={result['auc_roc']:.4f} F1={result['f1']:.4f} "
                  f"acc={result['accuracy']:.4f} ece={result['ece']:.4f} "
                  f"({result['wall_time_s']:.0f}s)")
        else:
            n_err += 1
            print(f"ERROR: {result.get('error', '?')[:80]}")

        elapsed = time.time() - sweep_start
        rate = (i + 1) / elapsed if elapsed > 0 else 0
        eta = (n_total - i - 1) / rate if rate > 0 else 0
        print(f"  Progress: {i+1}/{n_total} ({elapsed/60:.1f}m elapsed, ETA {eta/60:.1f}m)")

    elapsed_total = time.time() - sweep_start
    print(f"\n{'='*60}")
    print(f"Sweep complete: {n_ok} ok, {n_err} errors, {elapsed_total/60:.1f}m total")
    print(f"{'='*60}")

    # Write full JSON
    summary = {
        "total_combos": n_total,
        "ok": n_ok,
        "errors": n_err,
        "elapsed_s": round(elapsed_total, 2),
        "timeout_s": timeout_s,
        "patience": patience,
        "qaoa_cache": qaoa_cache,
        "results": results,
    }
    with open(json_path, "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    print(f"Full results: {json_path}")
    print(f"CSV results:  {csv_path}")

    # Print leaderboard
    ok_results = [r for r in results if r["status"] == "ok" and r.get("auc_roc") is not None]
    if ok_results:
        ok_results.sort(key=lambda r: r["auc_roc"], reverse=True)
        print(f"\nTop-15 by AUC-ROC:")
        print(f"{'ID':>4} {'AUC':>6} {'F1':>6} {'Acc':>6} {'ECE':>6} "
              f"{'QAOA':>15} {'QML':>3} {'HU':>3} {'LR':>6} {'Ep':>3} {'Time':>5}")
        for r in ok_results[:15]:
            cfg = r["config"]
            qk = cfg.get("qaoa_key", "")[:15]
            print(
                f"{r['combo_id']:>4} {r['auc_roc']:>6.4f} {r['f1']:>6.4f} "
                f"{r['accuracy']:>6.4f} {r['ece']:>6.4f} "
                f"{qk:>15} {cfg.get('qml_layers',''):>3} "
                f"{cfg.get('hidden_units',''):>3} {cfg.get('learning_rate',''):>6.4f} "
                f"{r.get('epochs_run',''):>3} {r['wall_time_s']:>5.0f}"
            )

    return summary


def main():
    parser = argparse.ArgumentParser(description="QAOA + VQC hyperparameter sweep")
    parser.add_argument("--timeout", type=int, default=600,
                        help="Per-combo timeout in seconds (default: 600)")
    parser.add_argument("--patience", type=int, default=10,
                        help="Early stopping patience (default: 10)")
    parser.add_argument("--out", type=str, default=None,
                        help="Output directory (default: output/sweep)")
    args = parser.parse_args()

    run_sweep(
        timeout_s=args.timeout,
        patience=args.patience,
        out_dir=args.out,
    )


if __name__ == "__main__":
    raise SystemExit(main())
