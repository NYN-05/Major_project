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
)
from quantum.evaluation import (
    classification_metrics, balanced_accuracy, decision_bins,
    expected_calibration_error
)
from quantum.data import FEATURE_SETS


class TestQuantumConfig(unittest.TestCase):
    """Test quantum configuration classes."""

    def test_decision_config_defaults(self):
        """Test DecisionConfig default values (binary contract)."""
        cfg = DecisionConfig()
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
        # RPPG has 20 features (Phase 4 cross-ROI features removed)
        self.assertEqual(len(RPPG_FEATURE_NAMES), 20)
        self.assertEqual(len(VISUAL_FEATURE_NAMES), 39)
        # Fused = 20 rPPG + 39 visual = 59
        self.assertEqual(len(FUSED_FEATURE_NAMES), 59)

    def test_feature_sets_only_fused(self):
        """Test that only fused feature set is supported."""
        self.assertEqual(len(FEATURE_SETS), 1)
        self.assertIn('fused', FEATURE_SETS)
        self.assertEqual(FEATURE_SETS['fused'], FUSED_FEATURE_NAMES)


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

    def test_classification_metrics_with_threshold(self):
        """Test binary classification metrics honour a custom threshold."""
        metrics = classification_metrics(self.y_true, self.prob_real, decision_threshold=0.6)
        self.assertIn('accuracy', metrics)
        self.assertIn('coverage', metrics)
        self.assertIn('confusion_matrix_binary', metrics)

    def test_balanced_accuracy(self):
        """Test balanced accuracy calculation."""
        y_true = np.array([0, 0, 1, 1])
        prob_real = np.array([0.1, 0.2, 0.8, 0.9])
        bal_acc = balanced_accuracy(y_true, prob_real)
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
        self.assertIn('coverage', bins)
        self.assertIn('decision_threshold', bins)
        self.assertEqual(bins['real'] + bins['fake'], len(self.y_true))

    def test_decision_bins_custom_threshold(self):
        """Test decision bins honour a custom decision threshold."""
        cfg = DecisionConfig(decision_threshold=0.6)
        bins = decision_bins(self.y_true, self.prob_real, cfg=cfg)
        self.assertEqual(bins['decision_threshold'], 0.6)
        # 0.1 0.2 0.4 -> FAKE; 0.6 0.8 0.9 -> REAL
        self.assertEqual(bins['fake'], 3)
        self.assertEqual(bins['real'], 3)

    def test_decision_bins_with_pqs(self):
        """PQS argument is accepted and coverage stays 1.0 (binary contract)."""
        cfg = DecisionConfig()
        pqs = np.array([0.8, 0.8, 0.3, 0.6, 0.9, 0.9])
        bins = decision_bins(self.y_true, self.prob_real, cfg=cfg, pqs=pqs)
        self.assertIn('real', bins)
        self.assertIn('fake', bins)
        self.assertEqual(bins['coverage'], 1.0)


if __name__ == '__main__':
    unittest.main()