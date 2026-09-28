"""Tests for quantum config module."""
import sys
import unittest
import numpy as np
from pathlib import Path

# Add WORKING directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "WORKING"))

from quantum.config import (
    DecisionConfig, DataConfig, QAOASelectionConfig, VQCConfig,
    RPPG_FEATURE_NAMES, VISUAL_FEATURE_NAMES, FUSED_FEATURE_NAMES,
    RPPG_BASE_FEATURE_NAMES, CROSS_ROI_FEATURE_NAMES,
    PHASE6_FEATURE_SETS, PHASE7_ABLATION_SETS
)
from quantum.evaluation import (
    classification_metrics, balanced_accuracy, decision_bins,
    expected_calibration_error
)
from quantum.data import FEATURE_SETS


class TestQuantumConfig(unittest.TestCase):
    """Test quantum configuration classes."""

    def test_decision_config_defaults(self):
        """Test DecisionConfig default values."""
        cfg = DecisionConfig()
        self.assertEqual(cfg.fake_max_prob, 0.3)
        self.assertEqual(cfg.real_min_prob, 0.7)
        self.assertEqual(cfg.quality_threshold, 0.5)
        self.assertEqual(cfg.decision_threshold, 0.5)

    def test_data_config_defaults(self):
        """Test DataConfig default values."""
        cfg = DataConfig()
        self.assertEqual(cfg.seed, 42)
        self.assertEqual(cfg.val_ratio, 0.2)
        self.assertEqual(cfg.test_ratio, 0.2)
        self.assertTrue(cfg.filter_implausible)
        self.assertEqual(cfg.hr_min, 30.0)
        self.assertEqual(cfg.hr_max, 220.0)

    def test_qaoa_config_defaults(self):
        """Test QAOASelectionConfig defaults."""
        cfg = QAOASelectionConfig()
        self.assertEqual(cfg.p_layers, 3)
        self.assertEqual(cfg.target_features, 3)
        self.assertEqual(cfg.restarts, 8)

    def test_vqc_config_defaults(self):
        """Test VQCConfig defaults."""
        cfg = VQCConfig()
        self.assertEqual(cfg.qml_layers, 3)
        self.assertEqual(cfg.hidden_units, 8)
        self.assertEqual(cfg.epochs, 80)

    def test_feature_names_lengths(self):
        """Test feature name list lengths."""
        self.assertEqual(len(RPPG_FEATURE_NAMES), 48)
        self.assertEqual(len(VISUAL_FEATURE_NAMES), 39)
        self.assertEqual(len(FUSED_FEATURE_NAMES), 87)

    def test_phase6_feature_sets(self):
        """Test Phase 6 feature set definitions."""
        self.assertIn('rppg_base', PHASE6_FEATURE_SETS)
        self.assertIn('rppg_cross_roi', PHASE6_FEATURE_SETS)
        self.assertIn('visual_only', PHASE6_FEATURE_SETS)
        self.assertIn('fused', PHASE6_FEATURE_SETS)

    def test_phase7_ablation_sets(self):
        """Test Phase 7 ablation feature sets."""
        self.assertIn('A_rppg_only', PHASE7_ABLATION_SETS)
        self.assertIn('B_pos_only', PHASE7_ABLATION_SETS)
        self.assertIn('C_chrom_only', PHASE7_ABLATION_SETS)
        self.assertIn('D_pos_chrom', PHASE7_ABLATION_SETS)
        self.assertIn('E_rppg_quality', PHASE7_ABLATION_SETS)
        self.assertIn('F_rppg_cross_roi', PHASE7_ABLATION_SETS)
        self.assertIn('G_rppg_visual', PHASE7_ABLATION_SETS)
        self.assertIn('H_rppg_visual_quality', PHASE7_ABLATION_SETS)
        self.assertIn('I_full', PHASE7_ABLATION_SETS)


class TestEvaluationMetrics(unittest.TestCase):
    """Test evaluation metrics functions."""

    def setUp(self):
        self.y_true = np.array([0, 0, 1, 1, 1, 1])
        self.prob_real = np.array([0.1, 0.2, 0.4, 0.6, 0.8, 0.9])

    def test_classification_metrics_binary(self):
        """Test classification metrics for binary classification."""
        metrics = classification_metrics(self.y_true, self.prob_real)
        self.assertIn('accuracy', metrics)
        self.assertIn('precision', metrics)
        self.assertIn('recall', metrics)
        self.assertIn('f1', metrics)
        self.assertIn('auc_roc', metrics)
        self.assertIn('pr_auc', metrics)

    def test_classification_metrics_three_state(self):
        """Test classification metrics for three-state decision."""
        cfg = DC(fake_max_prob=0.3, real_min_prob=0.7)
        metrics = classification_metrics(self.y_true, self.prob_real, cfg=cfg)
        self.assertIn('accuracy', metrics)
        self.assertIn('coverage', metrics)
        self.assertIn('n_classified', metrics)
        self.assertIn('n_insufficient', metrics)
        self.assertIn('confusion_matrix_3class', metrics)

    def test_balanced_accuracy(self):
        """Test balanced accuracy calculation."""
        y_true = np.array([0, 0, 1, 1])
        prob_real = np.array([0.1, 0.2, 0.8, 0.9])
        cfg = DC(fake_max_prob=0.3, real_min_prob=0.7)
        bal_acc = balanced_accuracy(y_true, prob_real, cfg=cfg)
        self.assertIsInstance(bal_acc, float)
        self.assertGreaterEqual(bal_acc, 0.0)
        self.assertLessEqual(bal_acc, 1.0)

    def test_expected_calibration_error(self):
        """Test ECE calculation."""
        y_true = np.array([0, 0, 1, 1, 1, 1])
        prob_real = np.array([0.1, 0.2, 0.4, 0.6, 0.8, 0.9])
        ece = expected_calibration_error(y_true, prob_real, n_bins=5)
        self.assertIsInstance(ece, float)
        self.assertGreaterEqual(ece, 0.0)

    def test_decision_bins_binary(self):
        """Test decision bins for binary classification (using DecisionConfig)."""
        cfg = DecisionConfig()
        bins = decision_bins(self.y_true, self.prob_real, cfg=cfg)
        self.assertIn('real', bins)
        self.assertIn('fake', bins)
        self.assertIn('insufficient_evidence', bins)
        self.assertIn('coverage', bins)
        self.assertIn('fake_max_prob', bins)
        self.assertIn('real_min_prob', bins)
        self.assertIn('quality_threshold', bins)

    def test_decision_bins_three_state(self):
        """Test decision bins for three-state decision (using DecisionConfig)."""
        cfg = DecisionConfig(fake_max_prob=0.3, real_min_prob=0.7)
        bins = decision_bins(self.y_true, self.prob_real, cfg=cfg)
        self.assertIn('real', bins)
        self.assertIn('fake', bins)
        self.assertIn('insufficient_evidence', bins)
        self.assertIn('coverage', bins)

    def test_decision_bins_with_pqs(self):
        """Test decision bins with PQS."""
        cfg = DecisionConfig()
        pqs = np.array([0.8, 0.8, 0.3, 0.6, 0.9, 0.9])
        bins = decision_bins(self.y_true, self.prob_real, cfg=cfg, pqs=pqs)
        self.assertIn('insufficient_evidence', bins)
        self.assertLessEqual(bins['coverage'], 1.0)


if __name__ == '__main__':
    unittest.main()