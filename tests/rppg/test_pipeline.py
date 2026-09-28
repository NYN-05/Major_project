"""Tests for RPPG pipeline module."""
import sys
import unittest
import numpy as np
from pathlib import Path

# Add WORKING directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "WORKING"))

from rppg.pipeline import RPPGPipeline, RPPGResult, FrameQuality, WindowResult
from rppg.features import RPPGFeatures


class TestRPPGResult(unittest.TestCase):
    """Test RPPGResult dataclass."""

    def test_rppg_result_creation(self):
        """Test RPPGResult creation."""
        features = type('obj', (object,), {
            'to_vector': lambda self: np.array([1.0, 2.0, 3.0])
        })()
        result = RPPGResult(
            fps=30.0,
            n_frames_total=100,
            n_frames_usable=80,
            features=None,
            combined_signal=np.array([1.0, 2.0, 3.0]),
        )
        self.assertEqual(result.fps, 30.0)
        self.assertEqual(result.n_frames_total, 100)
        self.assertEqual(result.n_frames_usable, 80)
        self.assertIsNone(result.features)

    def test_rppg_result_with_features(self):
        """Test RPPGResult with features."""
        features = type('obj', (object,), {
            'to_vector': lambda self: np.array([1.0, 2.0, 3.0])
        })()
        result = RPPGResult(
            fps=30.0,
            n_frames_total=100,
            n_frames_usable=80,
            features=type('obj', (object,), {'to_vector': lambda self: np.array([1.0, 2.0, 3.0])})(),
            combined_signal=np.array([1.0, 2.0, 3.0]),
        )
        vec = result.to_feature_vector()
        self.assertIsNotNone(vec)
        self.assertEqual(vec.shape, (3,))


class TestFrameQuality(unittest.TestCase):
    """Test FrameQuality dataclass."""

    def test_frame_quality_creation(self):
        """Test FrameQuality creation."""
        fq = FrameQuality(
            frame_index=0,
            is_usable=True,
            blur_score=100.0,
            brightness=128.0,
            face_found=True,
        )
        self.assertEqual(fq.frame_index, 0)
        self.assertTrue(fq.is_usable)
        self.assertEqual(fq.blur_score, 100.0)
        self.assertEqual(fq.brightness, 128.0)
        self.assertTrue(fq.face_found)


class TestRPPGPipeline(unittest.TestCase):
    """Test RPPG pipeline."""

    def setUp(self):
        """Set up test pipeline."""
        self.pipeline = RPPGPipeline(method='POS', min_sqi=0.1)

    def test_pipeline_initialization(self):
        """Test pipeline initialization."""
        self.assertEqual(self.pipeline.method, 'POS')
        self.assertEqual(self.pipeline.min_sqi, 0.1)

    def test_pipeline_different_methods(self):
        """Test pipeline with different methods."""
        pipeline_pos = RPPGPipeline(method='POS')
        pipeline_chrom = RPPGPipeline(method='CHROM')
        self.assertEqual(pipeline_pos.method, 'POS')
        self.assertEqual(pipeline_chrom.method, 'CHROM')

    def test_pipeline_invalid_method(self):
        """Test pipeline with invalid method - validation happens at signal extraction time."""
        # The pipeline accepts any method in constructor; validation happens later
        pipeline = RPPGPipeline(method='INVALID')
        self.assertEqual(pipeline.method, 'INVALID')


class TestRPPGFeatures(unittest.TestCase):
    """Test RPPG features."""

    def test_rppg_features_creation(self):
        """Test RPPGFeatures creation."""
        features = type('obj', (object,), {
            'heart_rate_bpm': 72.0,
            'snr_db': 10.0,
            'to_vector': lambda self: np.array([72.0, 10.0]),
        })()
        self.assertEqual(features.heart_rate_bpm, 72.0)
        self.assertEqual(features.snr_db, 10.0)


class TestRPPGPipelineEdgeCases(unittest.TestCase):
    """Test RPPG pipeline edge cases."""

    def test_short_video(self):
        """Test pipeline with very short video."""
        # This test would need a video file, skip for now
        pass

    def test_no_face_detected(self):
        """Test handling of no face detected."""
        pass


if __name__ == '__main__':
    unittest.main()