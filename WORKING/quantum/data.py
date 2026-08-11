"""
Real rPPG dataset for the quantum layer (no synthetic data).

Reads the labelled feature table produced by the rPPG pipeline
(RPPG/dataset_features.csv), converts its labels onto the quantum
convention (CSV: 1 = deepfake, 0 = real; quantum: LABEL_REAL = 1,
LABEL_FAKE = 0), and stores a stratified train/val/test split as
data.npz for QAOA selection, VQC training, and evaluation.
"""

import csv

import numpy as np
from sklearn.model_selection import train_test_split

from quantum.config import DataConfig, FEATURE_NAMES, LABEL_FAKE, LABEL_REAL

RPPG_LABEL_FAKE = 1  # rPPG CSV convention: 1 = deepfake, 0 = real


def _load_rppg_features(csv_file):
    if not csv_file.exists():
        raise FileNotFoundError(
            f"rPPG features file not found at {csv_file}. Extract it with the rPPG "
            "pipeline first (see RPPG/rppg-pipeline/extract_dataset_features.py)."
        )
    with open(csv_file, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise ValueError(f"No labelled samples found in {csv_file}")
    missing = [name for name in FEATURE_NAMES if name not in rows[0]]
    if missing:
        raise ValueError(f"{csv_file} is missing rPPG columns: {missing}")

    X = np.asarray(
        [[float(row[name]) for name in FEATURE_NAMES] for row in rows], dtype=np.float32
    )
    labels = np.asarray([int(round(float(row["label"]))) for row in rows], dtype=np.int64)
    if set(labels.tolist()) - {0, 1}:
        raise ValueError("rPPG labels must be 0 (real) or 1 (fake)")
    y = np.where(labels == RPPG_LABEL_FAKE, LABEL_FAKE, LABEL_REAL).astype(np.int64)
    return X, y


def _train_val_test_split(X, y, cfg):
    rest_ratio = 1.0 - cfg.train_ratio
    X_train, X_rest, y_train, y_rest = train_test_split(
        X, y, test_size=rest_ratio, stratify=y, random_state=cfg.seed
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_rest,
        y_rest,
        test_size=(1.0 - cfg.train_ratio - cfg.val_ratio) / rest_ratio,
        stratify=y_rest,
        random_state=cfg.seed,
    )
    return X_train, y_train, X_val, y_val, X_test, y_test


def build_dataset(cfg=None):
    """Build data.npz from the real rPPG feature table (rPPG layer output)."""
    cfg = cfg or DataConfig()
    X, y = _load_rppg_features(cfg.csv_file)
    X_train, y_train, X_val, y_val, X_test, y_test = _train_val_test_split(X, y, cfg)
    cfg.data_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cfg.data_file,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        feature_names=FEATURE_NAMES,
    )
    return load_dataset(cfg.data_file)


def load_dataset(path=None):
    path = path or DataConfig().data_file
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Build it first with: python -m quantum.run --build-data"
        )
    data = np.load(path)
    return {key: data[key] for key in ("X_train", "y_train", "X_val", "y_val", "X_test", "y_test")}
