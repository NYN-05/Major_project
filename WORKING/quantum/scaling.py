"""
Train-only feature standardization for the quantum layer.

There is no feature-level scaling anywhere in the existing quantum path:
QAOA selection is scale-invariant (mutual information + Pearson
correlation), but the hybrid VQC angle embedding receives raw rPPG
magnitudes (e.g. heart_rate_bpm ~ 50-146) which are not meaningful
rotation angles. This module applies the codebase's existing
normalization idiom (z-score, as used on signal traces in
RPPG/rppg/preprocessing.py) to the feature vector.

The scaler is fitted on the TRAINING split only; validation, test, and
inference reuse the exact fitted transformation. The artifact is saved
as JSON in output/ (same convention as qaoa_selection.json) so
inference can reproduce training-time preprocessing.
"""

import json

import numpy as np

from quantum.config import FEATURE_NAMES, OUTPUT_DIR

SCALER_FILE = OUTPUT_DIR / "feature_scaler.json"


class FeatureScaler:
    """Per-feature z-score standardization (mean 0, unit variance).

    fit() must only ever see training data; transform() is then reused
    on validation, test, and inference inputs.
    """

    def __init__(self, feature_names=None):
        self.feature_names = list(feature_names or FEATURE_NAMES)
        self.mean_ = None
        self.scale_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0)
        self.scale_[self.scale_ < 1e-8] = 1.0
        return self

    def transform(self, X):
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("FeatureScaler must be fit before transform")
        return (np.asarray(X, dtype=np.float64) - self.mean_) / self.scale_

    def save(self, path=None):
        path = path or SCALER_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            json.dump(
                {
                    "feature_names": self.feature_names,
                    "mean": [float(m) for m in self.mean_],
                    "scale": [float(s) for s in self.scale_],
                },
                fh,
                indent=2,
            )
        return path

    @classmethod
    def load(cls, path=None):
        path = path or SCALER_FILE
        if not path.exists():
            raise FileNotFoundError(
                f"Feature scaler not found at {path}. Build it first with: "
                "python -m quantum.pipeline --all"
            )
        with open(path) as fh:
            payload = json.load(fh)
        scaler = cls(payload["feature_names"])
        scaler.mean_ = np.asarray(payload["mean"], dtype=np.float64)
        scaler.scale_ = np.asarray(payload["scale"], dtype=np.float64)
        return scaler