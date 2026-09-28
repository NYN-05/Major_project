"""Pytest configuration and shared fixtures."""
import sys
from pathlib import Path

# Add WORKING directory to sys.path so tests can import from WORKING modules
WORKING_ROOT = Path(__file__).parent.parent / "WORKING"
if str(WORKING_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKING_ROOT))

# Also add the project root for tests.fixtures
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
import numpy as np
import cv2


@pytest.fixture
def sample_rppg_features():
    """Provide mock rPPG features for testing."""
    return {
        'heart_rate_bpm': 72.0,
        'snr_db': 10.0,
        'prv_std_ms': 20.0,
        'spectral_entropy': 0.5,
        'mad': 0.1,
        'signal_quality_index': 0.7,
        'cheek_forehead_correlation': 0.8,
        'left_right_cheek_correlation': 0.7,
        'hr_half_diff': 2.0,
        'peak_prominence': 2.0,
        'systolic_peak_width': 150.0,
        'diastolic_notch_ratio': 0.2,
        'forehead_cheek_phase_lag': 50.0,
        'signal_to_motion_ratio': 10.0,
        'peak_amplitude_variability': 0.1,
        'pulse_transit_time_proxy': 20.0,
        'spectral_flatness': 0.3,
        'spectral_centroid': 1.5,
        'kurtosis': 0.0,
        'phase_coherence_lr': 0.5,
        'phase_coherence_cf': 0.5,
        'pulse_cv_interval': 0.05,
        'zero_crossing_rate': 0.3,
        'cross_corr_lr': 0.8,
        'cross_corr_lf': 0.7,
        'cross_corr_rf': 0.7,
        'freq_agreement_lr': 0.9,
        'freq_agreement_lf': 0.85,
        'freq_agreement_rf': 0.85,
        'spectral_similarity_lr': 0.8,
        'spectral_similarity_lf': 0.8,
        'spectral_similarity_rf': 0.8,
        'cross_roi_corr_mean': 0.7,
        'cross_roi_corr_std': 0.1,
        'cross_roi_corr_min': 0.5,
        'cross_roi_corr_max': 0.9,
        'cross_roi_corr_cv': 0.2,
        'cross_roi_phase_lag_mean': 30.0,
        'cross_roi_phase_lag_std': 10.0,
        'cross_roi_phase_lag_min': 10.0,
        'cross_roi_phase_lag_max': 50.0,
        'cross_roi_coherence_mean': 0.7,
        'cross_roi_coherence_std': 0.1,
        'cross_roi_coherence_min': 0.5,
        'cross_roi_coherence_max': 0.9,
        'cross_roi_cross_corr_mean': 0.75,
        'cross_roi_freq_agreement_mean': 0.9,
        'cross_roi_spectral_similarity_mean': 0.85,
    }


@pytest.fixture
def sample_visual_features():
    """Provide mock visual features for testing."""
    features = {}
    for i in range(16):
        features[f'deep_feat_{i}'] = np.random.randn()
    for i in range(10):
        features[f'lbp_uniform_hist_{i}'] = np.random.rand()
    features.update({
        'texture_contrast': np.random.rand(),
        'texture_energy': np.random.rand(),
        'texture_homogeneity': np.random.rand(),
        'texture_correlation': np.random.rand(),
        'color_mean_r': np.random.rand(),
        'color_mean_g': np.random.rand(),
        'color_mean_b': np.random.rand(),
        'color_std_r': np.random.rand(),
        'color_std_g': np.random.rand(),
        'color_std_b': np.random.rand(),
        'freq_low_energy': np.random.rand(),
        'freq_mid_energy': np.random.rand(),
        'freq_high_energy': np.random.rand(),
    })
    return features


@pytest.fixture
def sample_video_frames():
    """Create sample video frames for testing."""
    for i in range(10):
        frame = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        # Add a simple face-like region
        h, w = 224, 224
        frame = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
        cv2.rectangle(frame, (w//3, h//3), (2*w//3, 2*h//3), (200, 150, 100), -1)
        yield frame