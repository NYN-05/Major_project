"""
visual
======
Visual feature extraction branch for deepfake detection.

Extracts compact visual representations from face crops/frames using
a pre-trained CNN backbone (ResNet50 by default). Designed to run
alongside the rPPG physiological branch for multi-source evidence fusion.
"""

from .extractor import (
    VisualFeatureExtractor,
    VisualFeatures,
    VISUAL_FEATURE_NAMES,
    compute_visual_features,
)

__all__ = [
    "VisualFeatureExtractor",
    "compute_visual_features",
    "VisualFeatures",
    "VISUAL_FEATURE_NAMES",
]