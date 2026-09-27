"""
visual
======
Visual feature extraction branch for deepfake detection.

Extracts compact visual representations from face crops/frames using
a pre-trained CNN backbone (ResNet50 by default). Designed to run
alongside the rPPG physiological branch for multi-source evidence fusion.
"""

from .extractor import VisualFeatureExtractor, compute_visual_features
from .features import VisualFeatures, VISUAL_FEATURE_NAMES

__all__ = [
    "VisualFeatureExtractor",
    "compute_visual_features",
    "VisualFeatures",
    "VISUAL_FEATURE_NAMES",
]