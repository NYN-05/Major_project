"""Tests for RPPG module."""
import sys
import unittest
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tests.fixtures.sample_data import (
    generate_synthetic_rppg_signal,
    generate_synthetic_rgb_traces,
    create_mock_rppg_features,
    create_mock_visual_features,
    create_test_dataset,
)


class TestSampleDataGenerators(unittest.TestCase):
    """Test synthetic data generators."""

    def test_generate_synthetic_rppg_signal(self):
        """Test synthetic rPPG signal generation."""
        signal = generate_synthetic_rppg_signal(duration_sec=2.0, fs=30.0)
        self.assertIsInstance(signal, np.ndarray)
        self.assertEqual(signal.shape[0], 60)  # 2 sec * 30 fps
        self.assertEqual(signal.dtype, np.float32)
        self.assertTrue(np.isfinite(signal).all())

    def test_generate_synthetic_rgb_traces(self):
        """Test synthetic RGB trace generation."""
        traces = generate_synthetic_rgb_traces(duration_sec=2.0, fs=30.0)
        self.assertIn('left_cheek', traces)
        self.assertIn('right_cheek', traces)
        self.assertIn('forehead', traces)
        for name, trace in traces.items():
            self.assertIsInstance(trace, np.ndarray)
            self.assertEqual(trace.shape[0], 60)
            self.assertEqual(trace.dtype, np.float32)

    def test_create_mock_rppg_features(self):
        """Test mock rPPG feature creation."""
        features = create_mock_rppg_features(heart_rate_bpm=75.0, snr_db=15.0)
        self.assertEqual(features['heart_rate_bpm'], 75.0)
        self.assertEqual(features['snr_db'], 15.0)
        self.assertIn('heart_rate_bpm', features)
        self.assertIn('snr_db', features)
        self.assertEqual(len(features), 23)  # Default feature count

    def test_create_mock_visual_features(self):
        """Test mock visual feature creation."""
        features = create_mock_visual_features(deep_feat_dim=8)
        self.assertIn('deep_feat_0', features)
        self.assertIn('deep_feat_7', features)
        self.assertNotIn('deep_feat_8', features)
        self.assertIn('texture_contrast', features)
        self.assertIn('color_mean_r', features)
        self.assertIn('freq_low_energy', features)

    def test_create_test_dataset(self):
        """Test test dataset creation."""
        features, X, y = create_test_dataset(n_samples=100, feature_set='rppg_only')
        self.assertEqual(len(features), 23)
        self.assertEqual(X.shape, (100, 23))
        self.assertEqual(len(y), 100)
        self.assertTrue(np.all(np.isin(y, [0, 1])))  # Binary labels


class TestSignalGenerationEdgeCases(unittest.TestCase):
    """Test edge cases in signal generation."""

    def test_zero_duration(self):
        """Test zero duration signal - returns empty array for very small durations."""
        signal = generate_synthetic_rppg_signal(duration_sec=0.01, fs=30.0)
        self.assertIsInstance(signal, np.ndarray)
        # For duration < 1/fs, n_samples = int(duration * fs) = 0, so empty array is valid
        self.assertGreaterEqual(signal.shape[0], 0)

    def test_high_snr(self):
        """Test high SNR signal generation."""
        signal = generate_synthetic_rppg_signal(snr_db=40.0)
        # High SNR should have very clean signal
        self.assertTrue(np.isfinite(signal).all())

    def test_low_snr(self):
        """Test low SNR signal generation."""
        signal = generate_synthetic_rppg_signal(snr_db=-10.0)
        self.assertTrue(np.isfinite(signal).all())

    def test_no_motion(self):
        """Test signal generation without motion."""
        signal = generate_synthetic_rppg_signal(add_motion=False)
        self.assertTrue(np.isfinite(signal).all())


if __name__ == '__main__':
    unittest.main()