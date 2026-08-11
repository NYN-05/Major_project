import json

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from quantum.config import DecisionConfig
from quantum.evaluate import classification_metrics


def run_baselines(X_train, y_train, X_test, y_test, decision_cfg=None, seed=42):
    decision_cfg = decision_cfg or DecisionConfig()
    models = {
        "random_forest": RandomForestClassifier(
            n_estimators=200, random_state=seed, n_jobs=-1
        ),
        "mlp": MLPClassifier(
            hidden_layer_sizes=(32, 16), max_iter=500, random_state=seed
        ),
    }
    results = {}
    for name, model in models.items():
        model.fit(np.asarray(X_train), np.asarray(y_train))
        prob_real = model.predict_proba(np.asarray(X_test))[:, 1]
        results[name] = classification_metrics(y_test, prob_real)
    with open(decision_cfg.metrics_baseline_file, "w") as fh:
        json.dump(results, fh, indent=2)
    return results