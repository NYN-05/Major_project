import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from quantum.config import DecisionConfig
from quantum.dummy_data import load_dataset


def evaluate_classifier(name: str, model, X_train, y_train, X_test, y_test, scaler=None) -> dict:
    X_train_fit = scaler.transform(X_train) if scaler is not None else X_train
    X_test_fit = scaler.transform(X_test) if scaler is not None else X_test
    model.fit(X_train_fit, y_train)
    y_pred = model.predict(X_test_fit)
    y_proba = model.predict_proba(X_test_fit)[:, 1] if hasattr(model, "predict_proba") else None
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary", zero_division=0)
    return {
        "model": name,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc_roc": float(roc_auc_score(y_test, y_proba)) if y_proba is not None else None,
    }


def run_baselines(data: dict, decision_config: DecisionConfig) -> dict:
    scaler = StandardScaler().fit(data["X_train"])
    X_train_s = scaler.transform(data["X_train"])
    X_val_s = scaler.transform(data["X_val"])
    X_test_s = scaler.transform(data["X_test"])

    results = {}
    for name, model in (
        ("MLP", MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=500, random_state=42)),
        ("RandomForest", RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)),
        ("SVM", SVC(kernel="rbf", probability=True, random_state=42)),
    ):
        results[name] = evaluate_classifier(name, model, X_train_s, data["y_train"], X_test_s, data["y_test"])

    decision_config.metrics_baseline_file.parent.mkdir(parents=True, exist_ok=True)
    decision_config.metrics_baseline_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def load_baseline_metrics(path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
