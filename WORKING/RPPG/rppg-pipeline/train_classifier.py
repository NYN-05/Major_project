"""
train_classifier.py
===================
Trains a class-balanced RandomForest model on the extracted rPPG features.
The model uses median imputation, stratified holdout evaluation, and a
slightly stronger forest configuration than the original baseline.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rppg.features import RPPGFeatures


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the rPPG deepfake classifier.")
    parser.add_argument("--features-csv", default=None, help="Path to dataset_features.csv")
    parser.add_argument("--model-out", default=None, help="Path to save the trained model")
    parser.add_argument("--metadata-out", default=None, help="Optional JSON file for training metadata")
    args = parser.parse_args()

    root = _repo_root()
    features_csv = Path(args.features_csv) if args.features_csv else root / "dataset_features.csv"
    model_path = Path(args.model_out) if args.model_out else root / "rppg_classifier.pkl"
    metadata_path = Path(args.metadata_out) if args.metadata_out else root / "rppg_classifier_metadata.json"

    if not features_csv.exists():
        print(f"Error: {features_csv} not found. Run extract_dataset_features.py first.")
        return

    df = pd.read_csv(features_csv)

    feature_cols = [name for name in RPPGFeatures.feature_names() if name in df.columns]
    missing_cols = [name for name in RPPGFeatures.feature_names() if name not in df.columns]
    if missing_cols:
        print(f"Error: missing required feature columns: {', '.join(missing_cols)}")
        return

    if "label" not in df.columns:
        print("Error: dataset_features.csv must include a label column.")
        return

    df = df.dropna(subset=["label"]).copy()
    df["label"] = df["label"].astype(int)

    if len(df) < 4:
        print("Not enough data to train a reliable model.")
        return

    fake_count = int((df["label"] == 1).sum())
    real_count = int((df["label"] == 0).sum())
    print(f"Training on {len(df)} samples ({fake_count} Fake, {real_count} Real)")

    X = df[feature_cols].values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2 if len(df) >= 10 else 0.33,
        random_state=42,
        stratify=y if min(fake_count, real_count) >= 2 else None,
    )

    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=600,
                    random_state=42,
                    class_weight="balanced_subsample",
                    min_samples_leaf=2,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    if min(fake_count, real_count) >= 2:
        n_splits = min(5, min(fake_count, real_count))
        if n_splits >= 2:
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            cv_scores = cross_val_score(model, X, y, cv=cv, scoring="balanced_accuracy", n_jobs=-1)
            print(f"Cross-validated balanced accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("\n--- Classification Report ---")
    try:
        print(classification_report(y_test, y_pred, target_names=["Real (0)", "Fake (1)"], labels=[0, 1]))
    except Exception:
        print(classification_report(y_test, y_pred))

    print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print(f"Balanced accuracy: {balanced_accuracy_score(y_test, y_pred):.3f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    forest = model.named_steps["clf"]
    importances = forest.feature_importances_
    print("\n--- Feature Importances ---")
    for name, importance in sorted(zip(feature_cols, importances), key=lambda item: item[1], reverse=True):
        print(f"{name:30s}: {importance:.4f}")

    with open(model_path, "wb") as handle:
        pickle.dump(model, handle)
    print(f"\nModel saved to {model_path}")

    metadata = {
        "features_csv": str(features_csv),
        "feature_columns": feature_cols,
        "training_samples": int(len(df)),
        "fake_samples": fake_count,
        "real_samples": real_count,
        "model_type": "Pipeline(SimpleImputer + RandomForestClassifier)",
    }
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    print(f"Training metadata saved to {metadata_path}")


if __name__ == "__main__":
    main()
