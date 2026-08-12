"""
retrain_dfdc.py
===============
Retrain the rPPG RandomForest classifier on archive/DFDC_Dataset
(Fake/ + Real/) end-to-end.

Pipeline per batch of videos:
  1. Extract the 8 rPPG features per video (same RPPGPipeline used by
     the quantum stage; features=None videos are skipped).
  2. Accumulate rows and append them to dataset_features_dfdc.csv
     (incremental, so a crash never loses finished work).
  3. Every --checkpoint-every successfully processed videos, retrain the
     classifier on ALL accumulated rows and save it to the SAME pkl path
     used by run_pipeline.py / streamlit_app.py
     (WORKING/output/rppg/rppg_classifier.pkl) plus rppg_classifier_metadata.json.

Resume support: videos whose feature row already exists in the CSV are
skipped, so re-running continues where it stopped.

Usage (from WORKING/RPPG):
  venv\Scripts\python.exe rppg-pipeline\retrain_dfdc.py
  venv\Scripts\python.exe rppg-pipeline\retrain_dfdc.py --max-videos 20   # smoke test
  venv\Scripts\python.exe rppg-pipeline\retrain_dfdc.py --checkpoint-every 10
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
from sklearn.pipeline import Pipeline

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rppg import RPPGPipeline  # noqa: E402
from rppg.features import RPPGFeatures  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT.parent / "output" / "rppg"
DATA_ROOT = REPO_ROOT / "archive" / "DFDC_Dataset"
FAKE_DIR = DATA_ROOT / "Fake"
REAL_DIR = DATA_ROOT / "Real"

FEATURES_CSV = OUTPUT_DIR / "dataset_features_dfdc.csv"
MODEL_PATH = OUTPUT_DIR / "rppg_classifier.pkl"
METADATA_PATH = OUTPUT_DIR / "rppg_classifier_metadata.json"

FEATURE_COLS = RPPGFeatures.feature_names()


def _videos(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv", ".webm"})


def collect_videos() -> list[tuple[int, Path]]:
    samples = [(1, p) for p in _videos(FAKE_DIR)] + [(0, p) for p in _videos(REAL_DIR)]
    return samples


def load_done() -> set[str]:
    if not FEATURES_CSV.exists():
        return set()
    df = pd.read_csv(FEATURES_CSV)
    if "video_path" not in df.columns:
        return set()
    return set(df["video_path"].astype(str).tolist())


def build_model() -> Pipeline:
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


def train_and_save(rows: list[dict]) -> tuple[int, int]:
    """Train on all accumulated rows and overwrite MODEL_PATH + METADATA_PATH."""
    df = pd.DataFrame(rows)
    df = df.dropna(subset=["label"]).copy()
    df["label"] = df["label"].astype(int)

    fake_count = int((df["label"] == 1).sum())
    real_count = int((df["label"] == 0).sum())
    print(f"  [checkpoint] training on {len(df)} samples ({fake_count} Fake, {real_count} Real)")

    X = df[FEATURE_COLS].values
    y = df["label"].values

    model = build_model()
    model.fit(X, y)

    with open(MODEL_PATH, "wb") as handle:
        pickle.dump(model, handle)

    metadata = {
        "features_csv": str(FEATURES_CSV),
        "feature_columns": FEATURE_COLS,
        "training_samples": int(len(df)),
        "fake_samples": fake_count,
        "real_samples": real_count,
        "model_type": "Pipeline(SimpleImputer + RandomForestClassifier)",
        "checkpoint": "incremental every N videos",
    }
    with open(METADATA_PATH, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    return fake_count, real_count


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrain rPPG classifier on archive/DFDC_Dataset.")
    parser.add_argument("--method", default="POS", choices=["POS", "CHROM"])
    parser.add_argument("--min-usable-frames", type=int, default=48)
    parser.add_argument("--checkpoint-every", type=int, default=10, help="Save model every N processed videos")
    parser.add_argument("--max-videos", type=int, default=None, help="Process at most this many videos (smoke test)")
    parser.add_argument("--force", action="store_true", help="Re-process videos already in the CSV")
    args = parser.parse_args()

    videos = collect_videos()
    if not videos:
        print(f"No videos found under {DATA_ROOT}")
        return

    done = set() if args.force else load_done()
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
    if FEATURES_CSV.exists():
        rows = pd.read_csv(FEATURES_CSV).to_dict("records")

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

        pd.DataFrame(rows).to_csv(FEATURES_CSV, index=False)

        if processed % args.checkpoint_every == 0:
            fake_n, real_n = train_and_save(rows)
            dt = time.time() - t_start
            rate = processed / dt if dt > 0 else 0
            eta = (len(pending) - processed) / rate if rate > 0 else float("inf")
            print(f"  [checkpoint] model saved -> {MODEL_PATH} ({fake_n} Fake / {real_n} Real) "
                  f"| {rate:.2f} vid/s | ETA {eta/60:.0f} min")

    if processed % args.checkpoint_every != 0 and processed > 0:
        train_and_save(rows)
        print(f"[final] model saved -> {MODEL_PATH}")

    dt = time.time() - t_start
    print(f"\nDone. processed={processed} no_features={no_features} failed={failed} in {dt/60:.1f} min")
    print(f"Features -> {FEATURES_CSV}")
    print(f"Model    -> {MODEL_PATH}")


if __name__ == "__main__":
    main()