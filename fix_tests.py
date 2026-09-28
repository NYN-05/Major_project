with open('tests/quantum/test_config.py', 'r') as f:
    content = f.read()

# Fix test_classification_metrics_three_state
old = """    def test_classification_metrics_three_state(self):
        \"\"\"Test classification metrics for three-state decision.\"\"\"
        cfg = DecisionConfig(fake_max_prob=0.3, real_min_prob=0.7)
        metrics = classification_metrics(self.y_true, self.prob_real, fake_max_prob=0.3, real_min_prob=0.7)
        self.assertIn('accuracy', metrics)
        self.assertIn('coverage', metrics)
        self.assertIn('n_classified', metrics)
        self.assertIn('n_insufficient', metrics)
        self.assertIn('confusion_matrix_3class', metrics)"""

new = """    def test_classification_metrics_three_state(self):
        \"\"\"Test classification metrics for three-state decision.\"\"\"
        cfg = DC(fake_max_prob=0.3, real_min_prob=0.7)
        metrics = classification_metrics(self.y_true, self.prob_real, cfg=cfg)
        self.assertIn('accuracy', metrics)
        self.assertIn('coverage', metrics)
        self.assertIn('n_classified', metrics)
        self.assertIn('n_insufficient', metrics)
        self.assertIn('confusion_matrix_3class', metrics)"""

content = content.replace(old, new)

# Fix test_balanced_accuracy
old = """    def test_balanced_accuracy(self):
        \"\"\"Test balanced accuracy calculation.\"\"\"
        y_true = np.array([0, 0, 1, 1])
        prob_real = np.array([0.1, 0.2, 0.8, 0.9])
        bal_acc = balanced_accuracy(y_true, prob_real, fake_max_prob=0.3, real_min_prob=0.7)
        self.assertIsInstance(bal_acc, float)
        self.assertGreaterEqual(bal_acc, 0.0)
        self.assertLessEqual(bal_acc, 1.0)"""

new = """    def test_balanced_accuracy(self):
        \"\"\"Test balanced accuracy calculation.\"\"\"
        y_true = np.array([0, 0, 1, 1])
        prob_real = np.array([0.1, 0.2, 0.8, 0.9])
        cfg = DC(fake_max_prob=0.3, real_min_prob=0.7)
        bal_acc = balanced_accuracy(y_true, prob_real, cfg=cfg)
        self.assertIsInstance(bal_acc, float)
        self.assertGreaterEqual(bal_acc, 0.0)
        self.assertLessEqual(bal_acc, 1.0)"""

content = content.replace(old, new)

# Fix test_decision_bins_binary
old = """    def test_decision_bins_binary(self):
        \"\"\"Test decision bins for binary classification.\"\"\"
        bins = decision_bins(self.y_true, self.prob_real)
        self.assertIn('real', bins)
        self.assertIn('fake', bins)
        self.assertIn('threshold', bins)"""

new = """    def test_decision_bins_binary(self):
        \"\"\"Test decision bins for binary classification.\"\"\"
        bins = decision_bins(self.y_true, self.prob_real)
        self.assertIn('real', bins)
        self.assertIn('fake', bins)
        # Three-state decision_bins doesn't have 'threshold' key
        self.assertIn('insufficient_evidence', bins)
        self.assertIn('coverage', bins)"""

content = content.replace(old, new)

# Fix test_decision_bins_three_state
old = """    def test_decision_bins_three_state(self):
        \"\"\"Test decision bins for three-state decision.\"\"\"
        bins = decision_bins(self.y_true, self.prob_real, fake_max_prob=0.3, real_min_prob=0.7)
        self.assertIn('real', bins)
        self.assertIn('fake', bins)
        self.assertIn('insufficient_evidence', bins)
        self.assertIn('coverage', bins)"""

new = """    def test_decision_bins_three_state(self):
        \"\"\"Test decision bins for three-state decision.\"\"\"
        cfg = DC(fake_max_prob=0.3, real_min_prob=0.7)
        bins = decision_bins(self.y_true, self.prob_real, cfg=cfg)
        self.assertIn('real', bins)
        self.assertIn('fake', bins)
        self.assertIn('insufficient_evidence', bins)
        self.assertIn('coverage', bins)"""

content = content.replace(old, new)

# Fix test_decision_bins_with_pqs
old = """    def test_decision_bins_with_pqs(self):
        \"\"\"Test decision bins with PQS.\"\"\"
        pqs = np.array([0.8, 0.8, 0.3, 0.6, 0.9, 0.9])
        bins = decision_bins(self.y_true, self.prob_real, pqs=pqs)
        self.assertIn('insufficient_evidence', bins)
        self.assertLessEqual(bins['coverage'], 1.0)"""

new = """    def test_decision_bins_with_pqs(self):
        \"\"\"Test decision bins with PQS.\"\"\"
        pqs = np.array([0.8, 0.8, 0.3, 0.6, 0.9, 0.9])
        cfg = DC(fake_max_prob=0.3, real_min_prob=0.7, quality_threshold=0.5)
        bins = decision_bins(self.y_true, self.prob_real, cfg=cfg, pqs=pqs)
        self.assertIn('insufficient_evidence', bins)
        self.assertLessEqual(bins['coverage'], 1.0)"""

content = content.replace(old, new)

with open('tests/quantum/test_config.py', 'w') as f:
    f.write(content)
print('Fixed')