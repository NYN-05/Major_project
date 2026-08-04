import json
from pathlib import Path

import numpy as np

from quantum.features_schema import FEATURE_NAMES, MAX_VALUE, MIN_VALUE


class FeatureValidationError(ValueError):
    pass


def validate_feature_matrix(X: np.ndarray) -> np.ndarray:
    if X.ndim != 2:
        raise FeatureValidationError(f"Feature matrix must be 2D, got shape {X.shape}")
    if X.shape[1] != len(FEATURE_NAMES):
        raise FeatureValidationError(
            f"Feature matrix must have {len(FEATURE_NAMES)} columns, got {X.shape[1]}"
        )
    if not np.issubdtype(X.dtype, np.floating) and not np.issubdtype(X.dtype, np.integer):
        raise FeatureValidationError(f"Feature matrix must be numeric, got dtype {X.dtype}")
    if not np.isfinite(X).all():
        raise FeatureValidationError("Feature matrix contains NaN or infinite values")
    if X.min() < MIN_VALUE - 1e-6 or X.max() > MAX_VALUE + 1e-6:
        raise FeatureValidationError(
            f"Feature values out of range [{MIN_VALUE}, {MAX_VALUE}], got min={X.min():.4f} max={X.max():.4f}"
        )
    return X.astype(np.float32)


def validate_labels(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y)
    if y.ndim != 1:
        raise FeatureValidationError(f"Labels must be 1D, got shape {y.shape}")
    if not np.isfinite(y).all():
        raise FeatureValidationError("Labels contain NaN or infinite values")
    return y.astype(np.int64)


class FeatureConditioner:
    def __init__(self, feature_names: list[str] | None = None):
        self.feature_names = list(feature_names or FEATURE_NAMES)
        self.min_ = None
        self.max_ = None

    def fit(self, X: np.ndarray) -> "FeatureConditioner":
        X = validate_feature_matrix(X)
        self.min_ = X.min(axis=0)
        self.max_ = X.max(axis=0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = validate_feature_matrix(X)
        if self.min_ is None or self.max_ is None:
            raise FeatureValidationError("Conditioner must be fit before transform")
        scale = np.maximum(self.max_ - self.min_, 1e-6)
        return np.clip((X - self.min_) / scale, MIN_VALUE, MAX_VALUE).astype(np.float32)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def to_dict(self) -> dict:
        return {
            "feature_names": self.feature_names,
            "min": self.min_.tolist() if self.min_ is not None else None,
            "max": self.max_.tolist() if self.max_ is not None else None,
        }

    def save(self, path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path) -> "FeatureConditioner":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        obj = cls(feature_names=data["feature_names"])
        obj.min_ = np.asarray(data["min"], dtype=np.float32)
        obj.max_ = np.asarray(data["max"], dtype=np.float32)
        return obj
