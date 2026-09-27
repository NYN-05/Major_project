"""
Ensemble Classification Module
==============================

Implements ensemble classification by combining predictions from multiple
models (visual, rPPG, quantum) using various fusion strategies:

1. Weighted Averaging - Simple weighted average of probabilities
2. Logistic Stacking - Meta-classifier (logistic regression) on model outputs
3. Meta-classifier - More complex meta-classifier (MLP, RF, etc.)

The ensemble combines predictions from:
- Visual model (trained on visual features)
- rPPG model (trained on rPPG features) 
- Quantum model (trained VQC on selected features)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable
import json
import numpy as np

from quantum.config import DecisionConfig, VQCConfig
from quantum.evaluation import run_baselines, evaluate_quantum_model
from quantum.vqc import load_vqc_model, predict_vqc, train_vqc
from quantum.scaling import FeatureScaler
from quantum.data import build_dataset, load_dataset, FEATURE_SETS
from quantum.config import DataConfig, QAOASelectionConfig, VQCConfig, DecisionConfig, OUTPUT_DIR


@dataclass
class ModelPrediction:
    """Individual model prediction output."""
    name: str
    prob_real: np.ndarray
    feature_set: str
    model_type: str  # "quantum", "classical", "visual"


@dataclass
class EnsembleResult:
    """Result of ensemble evaluation."""
    name: str
    method: str  # "weighted_avg", "logistic_stacking", "meta_classifier"
    weights: Optional[Dict[str, float]] = None
    metrics: Dict = field(default_factory=dict)
    cv_results: Optional[Dict] = None
    decision_bins: Optional[Dict] = None
    individual_metrics: Dict = field(default_factory=dict)
    
    def to_dict(self):
        return {
            "name": self.name,
            "method": self.method,
            "weights": self.weights,
            "metrics": self.metrics,
            "cv_results": self.cv_results,
            "decision_bins": self.decision_bins,
            "individual_metrics": self.individual_metrics,
        }


def train_individual_models(
    feature_sets: Dict[str, str],
    data_cfg: DataConfig,
    qaoa_cfg: QAOASelectionConfig,
    vqc_cfg: VQCConfig,
    decision_cfg: DecisionConfig,
    dev_only: bool = False,
    csv_file: Optional[str] = None,
    run_qaoa: bool = True,
    run_train: bool = True,
    run_eval: bool = True,
    run_baselines_flag: bool = True,
) -> Dict[str, Dict]:
    """
    Train individual models for each feature set and return their evaluation results.
    
    Returns dict mapping model name to its evaluation results (metrics, probs, etc.)
    """
    from quantum.pipeline import run_pipeline_for_feature_set
    
    results = {}
    for name, feature_set in feature_sets.items():
        print(f"\n{'='*60}")
        print(f"  Training individual model: {name} ({feature_set})")
        print(f"{'='*60}")
        
        result = run_pipeline_for_feature_set(
            feature_set=feature_set,
            data_cfg=data_cfg,
            qaoa_cfg=qaoa_cfg,
            vqc_cfg=vqc_cfg,
            decision_cfg=decision_cfg,
            dev_only=dev_only,
            csv_file=csv_file,
            run_qaoa=run_qaoa,
            run_train=run_train,
            run_eval=run_eval,
            run_baselines_flag=run_baselines_flag,
        )
        results[name] = result
    
    return results


def collect_model_probabilities(
    individual_results: Dict[str, Dict],
    feature_sets: Dict[str, str],
    data_cfg: DataConfig,
    vqc_cfg: VQCConfig,
    decision_cfg: DecisionConfig,
    dev_only: bool = True,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Collect validation/test probabilities from all trained models.
    
    Returns dict: {model_name: {'prob_real': array, 'y_true': array}}
    """
    from quantum.pipeline import load_vqc_model, predict_vqc
    from quantum.data import load_dataset
    from quantum.scaling import FeatureScaler
    from quantum.qaoa import load_selection
    from quantum.vqc import load_vqc_model, predict_vqc
    from quantum.scaling import FeatureScaler, SCALER_FILE
    import torch
    
    model_probs = {}
    
    for name, result in individual_results.items():
        feature_set = result.get('feature_set', 'rppg_only')
        feature_names = FEATURE_SETS.get(feature_set, [])
        
        # Load the dataset for this feature set
        if feature_set in ['rppg_only', 'rppg_base', 'rppg_cross_roi', 'visual_only', 'fused']:
            fs_cfg = FEATURE_SETS.get(feature_set, {})
            # Determine the right data file
            if 'visual' in feature_set:
                data_file = result.get('data_file', OUTPUT_DIR / f"data_{feature_set}.npz")
            else:
                data_file = OUTPUT_DIR / f"data_{feature_set}.npz"
            
            # Load data
            data = load_dataset(data_file)
            
            # Load scaler
            scaler_file = OUTPUT_DIR / f"feature_scaler_{feature_set}.json"
            scaler = FeatureScaler(feature_names=FEATURE_SETS[feature_set]).load(scaler_file)
            
            # Get test data
            eval_X = data["X_test"]
            eval_y = data["y_test"]
            X_test_scaled = scaler.transform(eval_X)
            
            # Load model and get probabilities
            selection = load_selection(OUTPUT_DIR / f"qaoa_selection_{feature_set}.json")
            indices = [int(i) for i in selection["selected_indices"]]
            
            model = load_vqc_model(len(indices))
            prob_real = predict_vqc(model, X_test_scaled[:, indices])
            
            model_probs[name] = {
                'prob_real': prob_real,
                'y_true': eval_y,
                'feature_set': feature_set,
            }
    
    return model_probs


def weighted_average_ensemble(
    model_probs: Dict[str, np.ndarray],
    weights: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    """
    Simple weighted average of model probabilities.
    
    Args:
        model_probs: Dict mapping model name to probability array
        weights: Optional weights for each model (will be normalized)
    
    Returns:
        Ensemble probability array
    """
    names = list(model_probs.keys())
    probs = np.column_stack([model_probs[n] for n in names])
    
    if weights is None:
        weights = np.ones(len(names)) / len(names)
    else:
        w = np.array([weights.get(n, 0.0) for n in names])
        w = w / w.sum() if w.sum() > 0 else np.ones(len(names)) / len(names)
    
    return (probs * w).sum(axis=1)


def train_logistic_stacking(
    val_probs: Dict[str, np.ndarray],
    y_val: np.ndarray,
) -> tuple:
    """
    Train logistic regression meta-classifier on validation probabilities.
    
    Returns:
        (meta_model, feature_names)
    """
    from sklearn.linear_model import LogisticRegression
    
    names = list(val_probs.keys())
    X_meta = np.column_stack([val_probs[n] for n in names])
    
    meta_model = LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42)
    meta_model.fit(X_meta, y_val)
    
    return meta_model, names


def predict_logistic_stacking(
    meta_model,
    test_probs: Dict[str, np.ndarray],
    feature_names: List[str],
) -> np.ndarray:
    """Predict using trained logistic stacking meta-model."""
    X_meta = np.column_stack([test_probs[n] for n in feature_names])
    return meta_model.predict_proba(X_meta)[:, 1]


def train_meta_classifier(
    val_probs: Dict[str, np.ndarray],
    y_val: np.ndarray,
    meta_model_type: str = "logistic_regression",
) -> tuple:
    """
    Train a meta-classifier on validation probabilities.
    
    Supported meta_model_types:
    - "logistic_regression"
    - "random_forest"
    - "mlp"
    - "linear_svc"
    - "gaussian_nb"
    - "xgboost"
    
    Returns:
        (meta_model, feature_names)
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.svm import LinearSVC
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.naive_bayes import GaussianNB
    
    try:
        from xgboost import XGBClassifier
        XGB_AVAILABLE = True
    except ImportError:
        XGB_AVAILABLE = False
    
    names = list(val_probs.keys())
    X_meta = np.column_stack([val_probs[n] for n in names])
    
    if meta_model_type == "logistic_regression":
        meta_model = LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42)
    elif meta_model_type == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        meta_model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    elif meta_model_type == "mlp":
        from sklearn.neural_network import MLPClassifier
        meta_model = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=500, random_state=42)
    elif meta_model_type == "linear_svc":
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.svm import LinearSVC
        meta_model = CalibratedClassifierCV(
            LinearSVC(class_weight='balanced', max_iter=5000, random_state=42),
            method="sigmoid",
            cv=3,
        )
    elif meta_model_type == "gaussian_nb":
        meta_model = GaussianNB()
    elif meta_model_type == "xgboost":
        from xgboost import XGBClassifier
        meta_model = XGBClassifier(
            n_estimators=300,
            learning_rate=0.1,
            max_depth=3,
            eval_metric="logloss",
            early_stopping_rounds=20,
            random_state=42,
            n_jobs=-1,
            tree_method="hist",
            device="cuda",
        )
    else:
        raise ValueError(f"Unknown meta_model_type: {meta_model_type}")
    
    meta_model.fit(X_meta, y_val)
    return meta_model, names


def evaluate_ensemble(
    ensemble_probs: np.ndarray,
    y_true: np.ndarray,
    decision_cfg: Optional[DecisionConfig] = None,
) -> Dict:
    """Evaluate ensemble predictions using standard metrics."""
    from quantum.evaluation import (
        classification_metrics,
        balanced_accuracy,
        decision_bins,
        optimal_threshold_youden,
    )
    
    decision_cfg = decision_cfg or DecisionConfig()
    metrics = classification_metrics(y_true, ensemble_probs)
    metrics["balanced_accuracy"] = balanced_accuracy(y_true, ensemble_probs)
    metrics["decision_bins"] = decision_bins(y_true, ensemble_probs)
    return metrics


def run_ensemble_experiment(
    feature_sets: Dict[str, str],
    data_cfg: DataConfig,
    qaoa_cfg: QAOASelectionConfig,
    vqc_cfg: VQCConfig,
    decision_cfg: DecisionConfig,
    dev_only: bool = False,
    csv_file: Optional[str] = None,
    ensemble_methods: List[str] = None,
    cv_folds: int = 5,
) -> Dict:
    """
    Run full ensemble experiment: train individual models, collect probabilities,
    apply ensemble methods, and compare results.
    
    Returns comprehensive results dictionary.
    """
    if ensemble_methods is None:
        ensemble_methods = ["weighted_avg", "logistic_stacking", "meta_rf", "meta_lr", "meta_mlp"]
    
    print(f"\n{'='*72}")
    print(f"  ENSEMBLE EXPERIMENT")
    print(f"{'='*72}")
    
    # Step 1: Train individual models
    print(f"\n[1/4] Training individual models...")
    individual_results = train_individual_models(
        feature_sets=feature_sets,
        data_cfg=data_cfg,
        qaoa_cfg=qaoa_cfg,
        vqc_cfg=vqc_cfg,
        decision_cfg=decision_cfg,
        dev_only=dev_only,
        csv_file=None,
        run_qaoa=True,
        run_train=True,
        run_eval=True,
        run_baselines_flag=True,
    )
    
    # Step 2: Collect model probabilities on validation set
    print(f"\n[2/4] Collecting model probabilities on validation set...")
    model_probs = collect_model_probabilities(
        individual_results={},
        feature_sets={},  # Will be filled from individual_results
        data_cfg=data_cfg,
        vqc_cfg=vqc_cfg,
        decision_cfg=DecisionConfig(),
        dev_only=dev_only,
    )
    
    # Step 3: Apply ensemble methods
    print(f"\n[3/4] Applying ensemble methods: {ensemble_methods}")
    
    # We need to re-collect properly using individual_results
    # For now, return the individual results for further processing
    
    return {
        "individual_results": individual_results,
        "ensemble_methods": ensemble_methods,
    }


def run_phase8_ensemble_comparison(
    data_cfg: DataConfig,
    qaoa_cfg: QAOASelectionConfig,
    vqc_cfg: VQCConfig,
    decision_cfg: DecisionConfig,
    dev_only: bool = False,
    csv_file: Optional[str] = None,
    run_qaoa: bool = True,
    run_train: bool = True,
    run_eval: bool = True,
    run_baselines_flag: bool = True,
) -> Dict:
    """
    Run Phase 8 ensemble comparison experiment.
    
    This runs the full ensemble comparison across multiple feature sets
    and ensemble methods, comparing quantum, classical, and ensemble approaches.
    """
    from quantum.config import PHASE6_FEATURE_SETS
    
    # Phase 8 feature sets for ensemble comparison
    feature_sets = {
        "rppg_only": "rppg_only",
        "visual_only": "visual_only",
        "fused": "fused",
    }
    
    # Override with dev_only
    if dev_only:
        # Use smaller subsets if needed
        pass
    
    results = {}
    
    for name, feature_set in feature_sets.items():
        print(f"\n{'='*60}")
        print(f"  Phase 8 Ensemble: {name} ({feature_set})")
        print(f"{'='*60}")
        
        # Run full pipeline for this feature set
        result = run_pipeline_for_feature_set(
            feature_set=feature_set,
            data_cfg=data_cfg,
            qaoa_cfg=qaoa_cfg,
            vqc_cfg=vqc_cfg,
            decision_cfg=decision_cfg,
            dev_only=dev_only,
            csv_file=None,
            run_qaoa=True,
            run_train=True,
            run_eval=True,
            run_baselines_flag=True,
        )
        results[name] = result
    
    # Collect quantum model probabilities for each feature set
    # For now, return individual results
    return results


# Export main functions
__all__ = [
    "weighted_average_ensemble",
    "train_logistic_stacking",
    "predict_logistic_stacking",
    "train_meta_classifier",
    "evaluate_ensemble",
    "run_ensemble_experiment",
    "run_phase8_ensemble_comparison",
]