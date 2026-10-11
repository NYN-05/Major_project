"""
train_classifier.py
===================
Trains the class-balanced RandomForest cross-check model on extracted rPPG
features. Two modes:

  Default (CSV): train from an existing feature table
      (MAJ_OUTPUT_ROOT/rppg/dataset_features.csv), with median imputation,
      stratified holdout evaluation, and cross-validation.

  --incremental: walk the DFDC dataset at DFDC_DATASET_PATH, extract the
      20 rPPG features per video with RPPGPipeline, append rows to
      dataset_features_dfdc.csv (crash-safe), and retrain + overwrite
      rppg_classifier.pkl every --checkpoint-every videos. Videos whose
      feature row already exists in the CSV are skipped, so re-running
      continues where it stopped.

Both modes write the SAME artifacts used elsewhere:
  MAJ_OUTPUT_ROOT/rppg/rppg_classifier.pkl (+ .sha256)
  MAJ_OUTPUT_ROOT/rppg/rppg_classifier_metadata.json

Usage (from WORKING/):
  python RPPG/train_classifier.py
  python RPPG/train_classifier.py --incremental --max-videos 20
  python RPPG/train_classifier.py --incremental --checkpoint-every 10
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
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

_WORKING_DIR = Path(__file__).resolve().parent.parent
if str(_WORKING_DIR) not in sys.path:
    sys.path.insert(0, str(_WORKING_DIR))

from quantum.config import write_sha256  # noqa: E402
from RPPG.features import RPPGFeatures  # noqa: E402

FEATURE_COLS = RPPGFeatures.feature_names()


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _output_dir() -> Path:
    return Path(os.environ.get("MAJ_OUTPUT_ROOT", _repo_root().parent / "Scrape" / "output")) / "rppg"


def build_model() -> Pipeline:
    """Shared RandomForest pipeline (median imputation + balanced forest)."""
    return Pipeline(
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


def _save_model_and_metadata(model: Pipeline, feature_cols: list[str], df: pd.DataFrame,
                             features_csv: Path, model_path: Path, metadata_path: Path,
                             extra_metadata: dict | None = None) -> None:
    fake_count = int((df["label"] == 1).sum())
    real_count = int((df["label"] == 0).sum())
    with open(model_path, "wb") as handle:
        pickle.dump(model, handle)
    write_sha256(model_path)
    metadata = {
        "features_csv": str(features_csv),
        "feature_columns": list(feature_cols),
        "training_samples": int(len(df)),
        "fake_samples": fake_count,
        "real_samples": real_count,
        "model_type": "Pipeline(SimpleImputer + RandomForestClassifier)",
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def train_from_csv(args: argparse.Namespace) -> None:
    out_dir = _output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    features_csv = Path(args.features_csv) if args.features_csv else out_dir / "dataset_features.csv"
    model_path = Path(args.model_out) if args.model_out else out_dir / "rppg_classifier.pkl"
    metadata_path = Path(args.metadata_out) if args.metadata_out else out_dir / "rppg_classifier_metadata.json"

    if not features_csv.exists():
        print(f"Error: {features_csv} not found. Run extract_dataset_features.py first.")
        return

    df = pd.read_csv(features_csv)

    feature_cols = [name for name in FEATURE_COLS if name in df.columns]
    missing_cols = [name for name in FEATURE_COLS if name not in df.columns]
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

    model = build_model()

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

    _save_model_and_metadata(model, feature_cols, df, features_csv, model_path, metadata_path)
    print(f"\nModel saved to {model_path}")
    print(f"Training metadata saved to {metadata_path}")


# ---------------------------------------------------------------------------
# Incremental DFDC mode (formerly retrain_dfdc.py)
# ---------------------------------------------------------------------------

def _videos(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(
        p for p in directory.rglob("*")
        if p.is_file() and p.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv", ".webm"}
    )


def collect_videos(dfdc_root: Path) -> list[tuple[int, Path]]:
    samples = [(1, p) for p in _videos(dfdc_root / "Fake")] + [(0, p) for p in _videos(dfdc_root / "Real")]
    return samples


def load_done(features_csv: Path) -> set[str]:
    if not features_csv.exists():
        return set()
    df = pd.read_csv(features_csv)
    if "video_path" not in df.columns:
        return set()
    return set(df["video_path"].astype(str).tolist())


def _train_and_save_incremental(rows: list[dict], features_csv: Path, model_path: Path,
                                metadata_path: Path) -> tuple[int, int]:
    """Train on all accumulated rows and overwrite the model + metadata."""
    df = pd.DataFrame(rows)
    df = df.dropna(subset=["label"]).copy()
    df["label"] = df["label"].astype(int)

    fake_count = int((df["label"] == 1).sum())
    real_count = int((df["label"] == 0).sum())
    print(f"  [checkpoint] training on {len(df)} samples ({fake_count} Fake, {real_count} Real)")

    model = build_model()
    model.fit(df[FEATURE_COLS].values, df["label"].values)
    _save_model_and_metadata(
        model, FEATURE_COLS, df, features_csv, model_path, metadata_path,
        extra_metadata={"checkpoint": "incremental every N videos"},
    )
    return fake_count, real_count


def train_incremental(args: argparse.Namespace) -> None:
    from quantum.config import get_dfdc_dataset_path
    from RPPG import RPPGPipeline

    out_dir = _output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    features_csv = out_dir / "dataset_features_dfdc.csv"
    model_path = out_dir / "rppg_classifier.pkl"
    metadata_path = out_dir / "rppg_classifier_metadata.json"

    dfdc_root = get_dfdc_dataset_path()
    videos = collect_videos(dfdc_root)
    if not videos:
        print(f"No videos found under {dfdc_root}")
        return

    done = set() if args.force else load_done(features_csv)
    pending = [(label, path) for label, path in videos if str(path) not in done]
    if args.max_videos is not None:
        pending = pending[: args.max_videos]

    print(f"Total videos: {len(videos)} | already done: {len(done)} | to process: {len(pending)}"
          + (f" (capped by --max-videos {args.max_videos})" if args.max_videos and len(done) + len(pending) < len(videos) else ""))

    if not pending:
        print("Nothing to process (all videos already in CSV). Use --force to rebuild.")
        return

    pipeline = RPPGPipeline(method=args.method, min_usable_frames=args.min_usable_frames)

    rows = []
    if features_csv.exists():
        rows = pd.read_csv(features_csv).to_dict("records")

    processed = 0
    failed = 0
    no_features = 0
    t_start = time.time()

    for label, video_path in pending:
        label_name = "Fake" if label == 1 else "Real"
        t0 = time.time()
        try:
            result = pipeline.process_video(str(video_path))
        except Exception as exc:
            failed += 1
            print(f"  ERROR {label_name} {video_path.name}: {exc}")
            continue

        if result.features is None:
            no_features += 1
            print(f"  skip {label_name} {video_path.name} (no features, usable={result.n_frames_usable})")
            continue

        feat = result.features.to_dict()
        feat["label"] = label
        feat["video_path"] = str(video_path)
        feat["method"] = args.method
        rows.append(feat)

        processed += 1
        print(f"  [{processed:5d}] OK {label_name} {video_path.name} ({time.time() - t0:.1f}s) "
              f"usable={result.n_frames_usable}")

        pd.DataFrame(rows).to_csv(features_csv, index=False)

        if processed % args.checkpoint_every == 0:
            fake_n, real_n = _train_and_save_incremental(rows, features_csv, model_path, metadata_path)
            dt = time.time() - t_start
            rate = processed / dt if dt > 0 else 0
            eta = (len(pending) - processed) / rate if rate > 0 else float("inf")
            print(f"  [checkpoint] model saved -> {model_path} ({fake_n} Fake / {real_n} Real) "
                  f"| {rate:.2f} vid/s | ETA {eta/60:.0f} min")

    if processed % args.checkpoint_every != 0 and processed > 0:
        _train_and_save_incremental(rows, features_csv, model_path, metadata_path)
        print(f"[final] model saved -> {model_path}")

    dt = time.time() - t_start
    print(f"\nDone. processed={processed} no_features={no_features} failed={failed} in {dt/60:.1f} min")
    print(f"Features -> {features_csv}")
    print(f"Model    -> {model_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the rPPG deepfake classifier.")
    parser.add_argument("--features-csv", default=None, help="Path to dataset_features.csv")
    parser.add_argument("--model-out", default=None, help="Path to save the trained model")
    parser.add_argument("--metadata-out", default=None, help="Optional JSON file for training metadata")
    parser.add_argument("--incremental", action="store_true",
                        help="Walk DFDC_DATASET_PATH, extract features per video, and retrain incrementally")
    # Incremental-mode options
    parser.add_argument("--method", default="POS", choices=["POS", "CHROM"],
                        help="[incremental] rPPG method")
    parser.add_argument("--min-usable-frames", type=int, default=48,
                        help="[incremental] minimum usable frames per video")
    parser.add_argument("--checkpoint-every", type=int, default=10,
                        help="[incremental] save model every N processed videos")
    parser.add_argument("--max-videos", type=int, default=None,
                        help="[incremental] process at most this many videos (smoke test)")
    parser.add_argument("--force", action="store_true",
                        help="[incremental] re-process videos already in the CSV")
    args = parser.parse_args()

    if args.incremental:
        train_incremental(args)
    else:
        train_from_csv(args)


if __name__ == "__main__":
    main()
