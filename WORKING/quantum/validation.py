"""Shared validation and evaluation safeguards for P1 remediation."""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from pathlib import Path

import numpy as np


class PipelineStatus(StrEnum):
    VALID = "VALID"
    INSUFFICIENT_FRAMES = "INSUFFICIENT_FRAMES"
    NO_FACE = "NO_FACE"
    MULTIPLE_FACES = "MULTIPLE_FACES"
    LOW_QUALITY = "LOW_QUALITY"
    RPPG_INVALID = "RPPG_INVALID"
    MODEL_ERROR = "MODEL_ERROR"


def validate_numeric_features(values) -> dict:
    arr = np.asarray(values, dtype=np.float64)
    nan_count = int(np.isnan(arr).sum())
    inf_count = int(np.isinf(arr).sum())
    invalid_count = nan_count + inf_count
    if invalid_count:
        raise ValueError(
            f"Invalid feature vector: nan_count={nan_count}, inf_count={inf_count}"
        )
    return {"nan_count": nan_count, "inf_count": inf_count, "invalid_feature_count": 0}


def choose_validation_threshold(y_true, prob_real, *, method="youden"):
    """Choose a threshold using validation data only."""
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(prob_real, dtype=float)
    validate_numeric_features(p)
    if len(y) != len(p) or len(np.unique(y)) < 2:
        raise ValueError("Threshold selection requires paired validation data with both classes")
    candidates = np.unique(np.r_[0.0, p, 1.0])
    best = (float("-inf"), 0.5)
    for threshold in candidates:
        pred = p >= threshold
        tp = np.sum(pred & (y == 1))
        tn = np.sum(~pred & (y == 0))
        fp = np.sum(pred & (y == 0))
        fn = np.sum(~pred & (y == 1))
        sensitivity = tp / max(tp + fn, 1)
        specificity = tn / max(tn + fp, 1)
        score = sensitivity + specificity - 1 if method == "youden" else (sensitivity + specificity) / 2
        if score > best[0]:
            best = (float(score), float(threshold))
    return best[1]


def repeated_seed_summary(run_fn, seeds=(42, 123, 456, 789, 999)):
    """Run a deterministic experiment for each seed and summarize numeric metrics."""
    results = []
    for seed in seeds:
        result = dict(run_fn(int(seed)))
        result["seed"] = int(seed)
        results.append(result)
    metric_keys = sorted(
        key for key in results[0] if key != "seed" and isinstance(results[0][key], (int, float))
    )
    return {
        "seeds": list(map(int, seeds)),
        "runs": results,
        "mean": {key: float(np.mean([row[key] for row in results])) for key in metric_keys},
        "std": {key: float(np.std([row[key] for row in results])) for key in metric_keys},
    }


def selection_stability(selections):
    """Return per-feature selection frequency and pairwise Jaccard stability."""
    normalized = [set(map(str, selection)) for selection in selections]
    if not normalized:
        return {"selection_frequency": {}, "mean_jaccard": 0.0}
    counts = {}
    for selection in normalized:
        for feature in selection:
            counts[feature] = counts.get(feature, 0) + 1
    jaccards = []
    for i, left in enumerate(normalized):
        for right in normalized[i + 1:]:
            union = left | right
            jaccards.append(len(left & right) / len(union) if union else 1.0)
    total = len(normalized)
    return {
        "selection_frequency": {key: value / total for key, value in sorted(counts.items())},
        "mean_jaccard": float(np.mean(jaccards)) if jaccards else 1.0,
        "n_runs": total,
    }


def write_reproducibility_metadata(path: Path, metadata: dict) -> Path:
    """Write deterministic JSON metadata and its checksum."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(metadata)
    payload.setdefault("schema_version", "p1-1")
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_suffix(path.suffix + ".sha256").write_text(digest + "\n", encoding="ascii")
    return path
