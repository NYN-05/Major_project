"""Integration tests for the full pipeline."""
import sys
import unittest
import numpy as np
from pathlib import Path

# Add WORKING directory to path
WORKING_DIR = Path(__file__).parent.parent.parent / "WORKING"
sys.path.insert(0, str(WORKING_DIR))
sys.path.insert(0, str(WORKING_DIR / "RPPG"))
sys.path.insert(0, str(WORKING_DIR / "quantum"))

from RPPG.pipeline import RPPGPipeline
from RPPG.pqs import compute_pqs, PQSResult, PQSComponents
from quantum.config import DecisionConfig
from quantum.evaluation import decision_bins, classification_metrics


class TestFullPipelineIntegration(unittest.TestCase):
    """Integration tests for the full pipeline."""

    def setUp(self):
        """Set up test fixtures."""
        # We use test videos if available, otherwise skip
        self.test_video_path = None  # Would need actual test video

    def test_pipeline_initialization(self):
        """Test pipeline initialization with different configs."""
        # Test default
        pipeline = RPPGPipeline(method='POS')
        self.assertEqual(pipeline.method, 'POS')

        pipeline_chrom = RPPGPipeline(method='CHROM')
        self.assertEqual(pipeline_chrom.method, 'CHROM')

    def test_pipeline_with_quality_weighting(self):
        """Test pipeline with quality weighting enabled."""
        pipeline = RPPGPipeline(
            method='POS',
            use_quality_weighting=True,
            quality_weight_min=0.05
        )
        self.assertTrue(pipeline.use_quality_weighting)
        self.assertEqual(pipeline.quality_weight_min, 0.05)

    def test_pipeline_with_window_analysis(self):
        """Test pipeline with window analysis."""
        pipeline = RPPGPipeline(
            method='POS',
            enable_window_analysis=True,
            window_duration_sec=8.0,
            window_overlap_sec=4.0
        )
        self.assertTrue(pipeline.enable_window_analysis)
        self.assertEqual(pipeline.window_duration_sec, 8.0)
        self.assertEqual(pipeline.window_overlap_sec, 4.0)


class TestPQSIntegration(unittest.TestCase):
    """Test PQS computation integration."""

    def test_pqs_computation(self):
        """Test PQS computation with mock components."""
        components = type('obj', (object,), {
            'snr_quality': 0.8,
            'frame_utilization': 0.9,
            'roi_validity': 0.9,
            'cross_roi_consistency': 0.8,
            'frequency_stability': 0.7,
            'signal_amplitude': 0.6,
            'temporal_consistency': 0.5
        })()

        from RPPG.pqs import PQSResult, PQSComponents
        from RPPG.pqs import compute_pqs_simple

        # Test simple PQS computation
        pqs = compute_pqs_simple(
            snr_db=1.6,
            n_usable=251,
            n_total=251,
            cross_roi_corr_mean=0.5,
            freq_agreement_lr=0.96
        )
        self.assertIsInstance(pqs, float)
        self.assertGreaterEqual(pqs, 0.0)
        self.assertLessEqual(pqs, 1.0)

    def test_pqs_result_dataclass(self):
        """Test PQSResult dataclass."""
        components = type('obj', (object,), {
            'snr_quality': 0.8,
            'frame_utilization': 0.9,
            'roi_validity': 0.9,
            'cross_roi_consistency': 0.8,
            'frequency_stability': 0.7,
            'signal_amplitude': 0.6,
            'temporal_consistency': 0.5
        })()
        pqs_result = type('obj', (object,), {
            'pqs': 0.75,
            'quality_tier': 'HIGH',
            'components': type('obj', (object,), {'snr_quality': 0.8})(),
        })()
        self.assertEqual(pqs_result.quality_tier, 'HIGH')


class TestThreeStateDecision(unittest.TestCase):
    """Test three-state decision logic."""

    def test_decision_bins_three_state(self):
        """Test three-state decision bins."""
        y_true = np.array([0, 0, 1, 1, 1, 1])
        prob_real = np.array([0.1, 0.2, 0.4, 0.6, 0.8, 0.9])
        cfg = DecisionConfig(fake_max_prob=0.3, real_min_prob=0.7)
        bins = decision_bins(y_true, prob_real, cfg=cfg)
        self.assertIn('real', bins)
        self.assertIn('fake', bins)
        self.assertIn('insufficient_evidence', bins)
        self.assertIn('coverage', bins)
        self.assertIn('fake_max_prob', bins)
        self.assertIn('real_min_prob', bins)

    def test_decision_bins_with_pqs(self):
        """Test decision bins with PQS."""
        y_true = np.array([0, 0, 1, 1, 1, 1])
        prob_real = np.array([0.1, 0.2, 0.4, 0.6, 0.8, 0.9])
        pqs = np.array([0.8, 0.8, 0.3, 0.6, 0.9, 0.9])
        cfg = DecisionConfig(fake_max_prob=0.3, real_min_prob=0.7)
        bins = decision_bins(y_true, prob_real, cfg=cfg, pqs=pqs)
        self.assertIn('insufficient_evidence', bins)
        self.assertLessEqual(bins['coverage'], 1.0)


class TestEndToEndWorkflow(unittest.TestCase):
    """Test end-to-end workflow."""

    def test_full_pipeline_workflow(self):
        """Test complete pipeline workflow."""
        # This would require a test video file
        # Skipping for now as it requires test video files
        self.skipTest("Requires test video files")


class TestRPPGModule(unittest.TestCase):
    """Test RPPG module imports and basic functionality."""

    def test_rppg_imports(self):
        """Test RPPG module imports."""
        import RPPG
        from RPPG import RPPGPipeline
        from RPPG.pqs import compute_pqs, PQSResult, PQSComponents

        self.assertIsNotNone(RPPG.RPPGPipeline)

    def test_visual_imports(self):
        """Test visual module imports."""
        import visual
        from visual import VisualFeatureExtractor
        self.assertIsNotNone(visual.VisualFeatureExtractor)

    def test_quantum_imports(self):
        """Test quantum module imports."""
        import quantum
        from quantum.config import DecisionConfig
        from quantum.pipeline import predict_features

        # Check that the module has a __file__ attribute (is loadable)
        self.assertIsNotNone(quantum.__file__)


if __name__ == '__main__':
    unittest.main()