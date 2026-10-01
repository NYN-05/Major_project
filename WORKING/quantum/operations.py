"""Shared operational safeguards for reproducible pipeline runs."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

OPERATIONS_VERSION = "p2-1"
LOGGER = logging.getLogger("deepfake_pipeline")


def resolve_torch_device(requested: str = "auto") -> torch.device:
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError(f"Unsupported device: {requested}")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device("cuda" if requested == "cuda" or (
        requested == "auto" and torch.cuda.is_available()
    ) else "cpu")


def memory_snapshot(device: torch.device | None = None) -> dict:
    snapshot = {"rss_mb": None, "cuda_allocated_mb": None, "cuda_reserved_mb": None}
    try:
        import psutil
        snapshot["rss_mb"] = round(psutil.Process().memory_info().rss / 2**20, 2)
    except ImportError:
        pass
    if device is not None and device.type == "cuda":
        snapshot["cuda_allocated_mb"] = round(torch.cuda.memory_allocated(device) / 2**20, 2)
        snapshot["cuda_reserved_mb"] = round(torch.cuda.memory_reserved(device) / 2**20, 2)
    return snapshot


def stable_hash(value) -> str:
    payload = json.dumps(value, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def cache_key(*, dataset_hash: str, preprocessing_version: str,
              model_version: str, feature_version: str, config_hash: str) -> str:
    return stable_hash({
        "dataset_hash": dataset_hash,
        "preprocessing_version": preprocessing_version,
        "model_version": model_version,
        "feature_version": feature_version,
        "config_hash": config_hash,
    })


def dataset_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(path.rglob("*")) if path.is_dir() else [path]:
        if item.is_file():
            stat = item.stat()
            digest.update(str(item.relative_to(path) if path.is_dir() else item.name).encode())
            digest.update(str(stat.st_size).encode())
            digest.update(str(stat.st_mtime_ns).encode())
    return digest.hexdigest()


def git_revision(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def experiment_record(*, experiment_id: str, dataset: str, split: str, seed: int,
                      features: list[str], model: str, hyperparameters: dict,
                      metrics: dict, device: str, started: float) -> dict:
    return {
        "experiment_id": experiment_id,
        "git_commit": git_revision(Path(__file__).resolve().parents[2]),
        "dataset": dataset,
        "split": split,
        "seed": int(seed),
        "features": features,
        "model": model,
        "hyperparameters": hyperparameters,
        "metrics": metrics,
        "runtime_seconds": round(time.time() - started, 3),
        "device": device,
        "operations_version": OPERATIONS_VERSION,
        "platform": platform.platform(),
    }


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
