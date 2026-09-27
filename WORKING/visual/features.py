"""
visual/features.py
==================
Visual feature definitions and data classes.
"""

from dataclasses import dataclass, asdict
from typing import Dict, List
import numpy as np


@dataclass
class VisualFeatures:
    """
    Compact visual feature representation extracted from face crops.

    Uses the penultimate layer (global average pooling) of a pre-trained
    ResNet50 (2048-dim) or similar backbone. For the quantum pipeline,
    we project to a lower dimension (e.g., 16 features) via PCA or
    use the raw 2048-dim vector with a classical baseline.
    """
    # Deep CNN features (ResNet50 GAP output = 2048 dim)
    # We store a reduced set for practical use with the quantum pipeline
    deep_feat_0: float = 0.0
    deep_feat_1: float = 0.0
    deep_feat_2: float = 0.0
    deep_feat_3: float = 0.0
    deep_feat_4: float = 0.0
    deep_feat_5: float = 0.0
    deep_feat_6: float = 0.0
    deep_feat_7: float = 0.0
    deep_feat_8: float = 0.0
    deep_feat_9: float = 0.0
    deep_feat_10: float = 0.0
    deep_feat_11: float = 0.0
    deep_feat_12: float = 0.0
    deep_feat_13: float = 0.0
    deep_feat_14: float = 0.0
    deep_feat_15: float = 0.0

    # Handcrafted complementary features
    lbp_uniform_hist_0: float = 0.0
    lbp_uniform_hist_1: float = 0.0
    lbp_uniform_hist_2: float = 0.0
    lbp_uniform_hist_3: float = 0.0
    lbp_uniform_hist_4: float = 0.0
    lbp_uniform_hist_5: float = 0.0
    lbp_uniform_hist_6: float = 0.0
    lbp_uniform_hist_7: float = 0.0
    lbp_uniform_hist_8: float = 0.0
    lbp_uniform_hist_9: float = 0.0

    # Texture statistics (computed on grayscale face crop)
    texture_contrast: float = 0.0
    texture_energy: float = 0.0
    texture_homogeneity: float = 0.0
    texture_correlation: float = 0.0

    # Color statistics (mean/std per channel on face crop)
    color_mean_r: float = 0.0
    color_mean_g: float = 0.0
    color_mean_b: float = 0.0
    color_std_r: float = 0.0
    color_std_g: float = 0.0
    color_std_b: float = 0.0

    # Frequency domain (DCT/FFT energy in bands)
    freq_low_energy: float = 0.0
    freq_mid_energy: float = 0.0
    freq_high_energy: float = 0.0

    def to_vector(self) -> np.ndarray:
        """Fixed-order numeric feature vector."""
        return np.array([
            # Deep features (16 dims)
            self.deep_feat_0, self.deep_feat_1, self.deep_feat_2, self.deep_feat_3,
            self.deep_feat_4, self.deep_feat_5, self.deep_feat_6, self.deep_feat_7,
            self.deep_feat_8, self.deep_feat_9, self.deep_feat_10, self.deep_feat_11,
            self.deep_feat_12, self.deep_feat_13, self.deep_feat_14, self.deep_feat_15,
            # LBP histogram (10 dims)
            self.lbp_uniform_hist_0, self.lbp_uniform_hist_1, self.lbp_uniform_hist_2,
            self.lbp_uniform_hist_3, self.lbp_uniform_hist_4, self.lbp_uniform_hist_5,
            self.lbp_uniform_hist_6, self.lbp_uniform_hist_7, self.lbp_uniform_hist_8,
            self.lbp_uniform_hist_9,
            # Texture (4 dims)
            self.texture_contrast, self.texture_energy, self.texture_homogeneity,
            self.texture_correlation,
            # Color (6 dims)
            self.color_mean_r, self.color_mean_g, self.color_mean_b,
            self.color_std_r, self.color_std_g, self.color_std_b,
            # Frequency (3 dims)
            self.freq_low_energy, self.freq_mid_energy, self.freq_high_energy,
        ], dtype=np.float64)

    @staticmethod
    def feature_names() -> List[str]:
        return [
            # Deep features (16 dims)
            "deep_feat_0", "deep_feat_1", "deep_feat_2", "deep_feat_3",
            "deep_feat_4", "deep_feat_5", "deep_feat_6", "deep_feat_7",
            "deep_feat_8", "deep_feat_9", "deep_feat_10", "deep_feat_11",
            "deep_feat_12", "deep_feat_13", "deep_feat_14", "deep_feat_15",
            # LBP histogram (10 dims)
            "lbp_uniform_hist_0", "lbp_uniform_hist_1", "lbp_uniform_hist_2",
            "lbp_uniform_hist_3", "lbp_uniform_hist_4", "lbp_uniform_hist_5",
            "lbp_uniform_hist_6", "lbp_uniform_hist_7", "lbp_uniform_hist_8",
            "lbp_uniform_hist_9",
            # Texture (4 dims)
            "texture_contrast", "texture_energy", "texture_homogeneity",
            "texture_correlation",
            # Color (6 dims)
            "color_mean_r", "color_mean_g", "color_mean_b",
            "color_std_r", "color_std_g", "color_std_b",
            # Frequency (3 dims)
            "freq_low_energy", "freq_mid_energy", "freq_high_energy",
        ]

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


# Full feature list (39 features total)
VISUAL_FEATURE_NAMES = VisualFeatures.feature_names()


# For the quantum pipeline, we also define a compact subset
# that matches the QAOA selection target (3-6 features)
VISUAL_FEATURE_NAMES_COMPACT = [
    "deep_feat_0", "deep_feat_1", "deep_feat_2", "deep_feat_3",
    "lbp_uniform_hist_0", "lbp_uniform_hist_1",
    "texture_contrast", "texture_energy",
    "color_mean_r", "color_mean_g",
    "freq_low_energy", "freq_mid_energy",
]