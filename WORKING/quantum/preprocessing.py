"""Canonical feature validation and scaling shared by training and inference."""

import numpy as np

from quantum.scaling import FeatureScaler


def validate_feature_schema(feature_names, expected_names) -> None:
    actual = list(feature_names)
    expected = list(expected_names)
    if actual != expected:
        raise ValueError(
            f"Feature schema mismatch: expected ordered schema {expected}, got {actual}"
        )


def transform_features(values, feature_names, scaler: FeatureScaler) -> np.ndarray:
    validate_feature_schema(scaler.feature_names, feature_names)
    transformed = scaler.transform(np.asarray(values, dtype=np.float64))
    if not np.isfinite(transformed).all():
        raise ValueError("Preprocessing produced non-finite scaled features")
    return transformed
