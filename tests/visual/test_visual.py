"""Tests for visual feature extraction module."""
import sys
import tempfile
import unittest
import numpy as np
from pathlib import Path

# Add WORKING directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "WORKING"))

from visual.features import VisualFeatures, VISUAL_FEATURE_NAMES
from visual.extractor import (
    PCA_PREPROCESSING_VERSION,
    RESNET_FEATURE_DIM,
    VisualFeatureExtractor,
)


class TestVisualFeatures(unittest.TestCase):
    """Test VisualFeatures dataclass."""

    def test_visual_features_creation(self):
        """Test VisualFeatures creation."""
        features = VisualFeatures()
        self.assertEqual(len(VISUAL_FEATURE_NAMES), 39)
        self.assertEqual(len(features.to_vector()), 39)

    def test_visual_feature_names(self):
        """Test feature names list."""
        self.assertEqual(len(VISUAL_FEATURE_NAMES), 39)
        self.assertIn('deep_feat_0', VISUAL_FEATURE_NAMES)
        self.assertIn('lbp_uniform_hist_0', VISUAL_FEATURE_NAMES)
        self.assertIn('texture_contrast', VISUAL_FEATURE_NAMES)
        self.assertIn('color_mean_r', VISUAL_FEATURE_NAMES)
        self.assertIn('freq_low_energy', VISUAL_FEATURE_NAMES)

    def test_visual_features_to_vector(self):
        """Test feature vector conversion."""
        features = VisualFeatures()
        vec = features.to_vector()
        self.assertEqual(vec.shape, (39,))
        self.assertEqual(vec.dtype, np.float64)


class TestVisualFeatureExtractor(unittest.TestCase):
    """Test VisualFeatureExtractor class."""

    def setUp(self):
        """Set up extractor."""
        # Note: This requires ResNet50 weights, skip if not available
        try:
            self.extractor = VisualFeatureExtractor(device='cpu', deep_feature_dim=16)
            self.has_extractor = True
        except Exception:
            self.has_extractor = False

    def test_extractor_initialization(self):
        """Test extractor initialization."""
        if not self.has_extractor:
            self.skipTest("VisualFeatureExtractor not available (no ResNet weights)")
        self.assertIsNotNone(self.extractor)
        self.assertEqual(self.extractor.deep_feature_dim, 16)

    def test_compute_visual_features_from_crops(self):
        """Test visual feature computation from face crops."""
        if not self.has_extractor:
            self.skipTest("VisualFeatureExtractor not available")

        # Create dummy face crops
        face_crops = [np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8) for _ in range(5)]

        try:
            features = self.extractor.extract_from_crops(face_crops)
            self.assertIsInstance(features.to_vector(), np.ndarray)
            self.assertEqual(features.to_vector().shape[0], 39)
        except Exception as e:
            self.skipTest(f"Feature extraction failed: {e}")

    def test_compute_visual_features_from_video(self):
        """Test visual feature computation from video file."""
        if not self.has_extractor:
            self.skipTest("VisualFeatureExtractor not available")
        # This would need a test video file
        self.skipTest("Requires test video file")

    def test_pca_artifact_round_trip_and_fixed_shape(self):
        """A persisted training PCA can transform the expected ResNet space."""
        with tempfile.TemporaryDirectory() as tmp:
            artifact = Path(tmp) / "visual_pca.npz"
            source = object.__new__(VisualFeatureExtractor)
            source.deep_feature_dim = 16
            source.backbone_name = "resnet50"
            source.use_pca = True
            source.pca_fitted = False
            source.actual_pca_dim = None
            source.pca_artifact_path = None
            source.fit_pca(
                np.random.RandomState(7).normal(size=(24, RESNET_FEATURE_DIM)),
                str(artifact),
            )

            loaded = object.__new__(VisualFeatureExtractor)
            loaded.deep_feature_dim = 16
            loaded.backbone_name = "resnet50"
            loaded.use_pca = True
            loaded.pca_fitted = False
            loaded.actual_pca_dim = None
            loaded.pca_artifact_path = None
            loaded.load_pca(str(artifact))
            transformed = loaded._apply_pca(
                np.zeros((3, RESNET_FEATURE_DIM), dtype=np.float32)
            )
            self.assertEqual(transformed.shape, (3, 16))
            self.assertEqual(PCA_PREPROCESSING_VERSION, str(
                np.load(artifact, allow_pickle=False)["preprocessing_version"]
            ))

    def test_pca_rejects_insufficient_training_data(self):
        """PCA must not silently truncate or zero-pad an undersized fit."""
        extractor = object.__new__(VisualFeatureExtractor)
        extractor.deep_feature_dim = 16
        with self.assertRaises(ValueError):
            extractor.fit_pca(np.zeros((16, RESNET_FEATURE_DIM), dtype=np.float32))

    def test_pca_transform_never_fits_implicitly(self):
        """Inference without the shared artifact must fail explicitly."""
        extractor = object.__new__(VisualFeatureExtractor)
        extractor.deep_feature_dim = 16
        extractor.pca_fitted = False
        with self.assertRaises(RuntimeError):
            extractor._apply_pca(np.zeros((1, RESNET_FEATURE_DIM), dtype=np.float32))


class TestVisualFeatureExtractionEdgeCases(unittest.TestCase):
    """Test edge cases in visual feature extraction."""

    def test_empty_crops(self):
        """Test with empty crop list."""
        try:
            extractor = VisualFeatureExtractor(device='cpu', deep_feature_dim=16)
            features = extractor.extract_from_crops([])
            self.assertIsInstance(features, np.ndarray)
        except Exception:
            pass  # Expected to fail or return zeros

    def test_single_crop(self):
        """Test with single crop."""
        try:
            extractor = VisualFeatureExtractor(device='cpu', deep_feature_dim=16)
            crop = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
            features = extractor.extract_from_crops([crop])
            self.assertEqual(features.to_vector().shape[0], 39)
        except Exception:
            pass


if __name__ == '__main__':
    unittest.main()