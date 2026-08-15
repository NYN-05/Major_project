"""
Real rPPG dataset for the quantum layer (no synthetic data).

Reads the labelled feature table produced by the rPPG pipeline
(WORKING/output/rppg/dataset_features.csv), converts its labels onto the quantum
convention (CSV: 1 = deepfake, 0 = real; quantum: LABEL_REAL = 1,
LABEL_FAKE = 0), and stores a subject-grouped train/val/test split as
data.npz for QAOA selection, VQC training, and evaluation.

Subject grouping: the CSV carries no explicit subject IDs, so the
group key is derived from the clip path. FF++ clips are grouped by
source subject so a real clip and its synthesized counterpart can
never straddle train/val/test:
  - FF-real / FF-synthesis (e.g. id0_0000.mp4, id0_id16_0002.mp4)
    -> "ffpp:src:<source-subject>" (first "id" token of the stem).
  - YouTube-real clips (e.g. 00000.mp4) -> "ffpp:yt:<stem>" (each
    unpaired YouTube clip is its own group).
DFDC clips carry no pairing or subject information on disk, so each
is treated as an individual group ("clip:<path>") - a documented
limitation (DFDC subject-level separation is unrecoverable here).
The split is seeded and balanced per class, and a leakage assertion
aborts the build if any group key appears in more than one split.
"""

import csv
import json

import numpy as np

from quantum.config import (
    DataConfig,
    FEATURE_NAMES,
    LABEL_FAKE,
    LABEL_REAL,
    OUTPUT_DIR,
)

RPPG_LABEL_FAKE = 1  # rPPG CSV convention: 1 = deepfake, 0 = real

SPLITS = ("train", "val", "test")


def _infer_subject_key(row):
    """Derive a subject group key from the video path.

    FF++ clips are grouped by source subject (first "id" token of the
    filename stem) so a real clip and its synthesized counterpart stay
    in the same split: id0_0000.mp4 and id0_id16_0002.mp4 -> "ffpp:src:id0".
    YouTube-real clips (numeric stems, no pairing) are their own group.
    DFDC clip names carry no subject info -> clip id (documented
    limitation: DFDC subject-level separation is unrecoverable on disk).
    """
    path = row["video_path"].replace("\\", "/").strip().lower()
    if "/ff++/" in path or path.startswith("ff++/"):
        stem = path.rsplit("/", 1)[-1].rsplit(".", 1)[0]
        folder = path.rstrip("/").rsplit("/", 2)[-2]
        if folder == "youtube-real" or not stem.split("_")[0].startswith("id"):
            return "ffpp:yt:" + stem
        return "ffpp:src:" + stem.split("_")[0]
    if "archive (1)" in path:
        stem = path.rsplit("/", 1)[-1].rsplit(".", 1)[0]
        return "subj:" + stem
    return "clip:" + path


def _infer_split_key(row):
    """Official dataset split when the source layout declares one.

    FF++ clips live under FF++/<split>/..., so their train/val/test
    folders are used directly (no random regrouping); anything else
    returns None and falls back to the seeded grouped random split.
    """
    path = row["video_path"].replace("\\", "/").strip().lower()
    if "/ff++/" in path:
        for split in SPLITS:
            if f"/ff++/{split}/" in path:
                return split
    return None


def _load_rppg_rows(csv_file, cfg=None):
    """Load the real rPPG feature table.

    Returns X (n x len(FEATURE_NAMES), FEATURE_NAMES order), y (quantum convention:
    1 = real, 0 = fake), subject groups, video paths, and a dict of
    filtering stats. When cfg.filter_implausible is set, rows with a
    heart rate outside [hr_min, hr_max] or with non-finite feature
    values are dropped (plausibility hygiene for KYC signals).
    """
    if not csv_file.exists():
        raise FileNotFoundError(
            f"rPPG features file not found at {csv_file}. Extract it with the rPPG "
            "pipeline first (see RPPG/rppg-pipeline/extract_dataset_features.py)."
        )
    with open(csv_file, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise ValueError(f"No labelled samples found in {csv_file}")
    missing = [name for name in FEATURE_NAMES if name not in rows[0]]
    if missing:
        raise ValueError(f"{csv_file} is missing rPPG columns: {missing}")

    filter_stats = {"total": len(rows), "dropped_hr": 0, "dropped_invalid": 0}
    keep = []
    for row in rows:
        try:
            values = [float(row[name]) for name in FEATURE_NAMES]
        except (TypeError, ValueError):
            keep.append(None)
            continue
        if not np.isfinite(values).all():
            filter_stats["dropped_invalid"] += 1
            keep.append(None)
            continue
        hr = values[FEATURE_NAMES.index("heart_rate_bpm")]
        if cfg is not None and cfg.filter_implausible and not (cfg.hr_min <= hr <= cfg.hr_max):
            filter_stats["dropped_hr"] += 1
            keep.append(None)
            continue
        keep.append(row)

    kept_rows = [row for row in keep if row is not None]
    filter_stats["kept"] = len(kept_rows)
    if not kept_rows:
        raise ValueError(f"No labelled samples survived filtering in {csv_file}")

    X = np.asarray(
        [[float(row[name]) for name in FEATURE_NAMES] for row in kept_rows], dtype=np.float32
    )
    labels = np.asarray([int(round(float(row["label"]))) for row in kept_rows], dtype=np.int64)
    if set(labels.tolist()) - {0, 1}:
        raise ValueError("rPPG labels must be 0 (real) or 1 (fake)")
    y = np.where(labels == RPPG_LABEL_FAKE, LABEL_FAKE, LABEL_REAL).astype(np.int64)
    groups = np.asarray([_infer_subject_key(row) for row in kept_rows], dtype=object)
    paths = np.asarray([row["video_path"].strip() for row in kept_rows], dtype=object)
    split_keys = np.asarray([_infer_split_key(row) for row in kept_rows], dtype=object)
    return X, y, groups, paths, split_keys, filter_stats


def _grouped_train_val_test_split(X, y, groups, paths, cfg):
    """Deterministic subject-grouped train/val/test split, balanced per class.

    Groups are assigned greedily to the split with the largest remaining
    (normalized) per-class demand, so each split ends up with roughly
    val_ratio / test_ratio of every class (train = remainder) while never
    separating the samples of one subject. Seeded via cfg.seed.
    """
    n_classes = int(y.max()) + 1
    per_class = [int((y == c).sum()) for c in range(n_classes)]
    target = {s: [0] * n_classes for s in SPLITS}
    for c in range(n_classes):
        val_c = int(round(per_class[c] * cfg.val_ratio))
        test_c = int(round(per_class[c] * cfg.test_ratio))
        train_c = per_class[c] - val_c - test_c
        if train_c < 0:
            raise ValueError("Not enough samples per class for the requested ratios")
        target["train"][c] = train_c
        target["val"][c] = val_c
        target["test"][c] = test_c

    grouped = {}
    for i, g in enumerate(groups):
        grouped.setdefault(g, []).append(i)

    rng = np.random.RandomState(cfg.seed)
    mixed, single = [], {c: [] for c in range(n_classes)}
    for idxs in grouped.values():
        labels = set(y[idxs].tolist())
        if len(labels) == 1:
            single[labels.pop()].append(idxs)
        else:
            mixed.append(idxs)

    assigned = {s: [0] * n_classes for s in SPLITS}
    splits = {s: [] for s in SPLITS}

    def remaining(s, classes):
        return sum(target[s][c] - assigned[s][c] for c in classes)

    def place(idxs):
        classes = set(y[idxs].tolist())
        scores = {}
        for s in SPLITS:
            need = remaining(s, classes)
            if need > 0:
                scores[s] = need / max(1.0, sum(target[s]))
        if scores:
            s = max(scores, key=scores.get)
        else:
            s = min(SPLITS, key=lambda s: sum(assigned[s]))
        splits[s].append(idxs)
        for c in classes:
            assigned[s][c] += int((y[idxs] == c).sum())

    rng.shuffle(mixed)
    for idxs in mixed:
        place(idxs)
    for c in range(n_classes):
        rng.shuffle(single[c])
        for idxs in single[c]:
            place(idxs)

    data = {}
    for s in SPLITS:
        idx = np.concatenate(splits[s]) if splits[s] else np.array([], dtype=np.int64)
        data[f"X_{s}"] = X[idx]
        data[f"y_{s}"] = y[idx]
        data[f"groups_{s}"] = groups[idx]
        data[f"paths_{s}"] = paths[idx]
    return data


def _split_by_source(X, y, groups, paths, split_keys):
    """Explicit train/val/test split from the dataset's own folders.

    Used when every row carries an official split hint (FF++ layout);
    returns the same dict keys as _grouped_train_val_test_split.
    """
    data = {}
    for s in SPLITS:
        mask = split_keys == s
        if not mask.any():
            raise ValueError(f"Official split '{s}' has no samples")
        data[f"X_{s}"] = X[mask]
        data[f"y_{s}"] = y[mask]
        data[f"groups_{s}"] = groups[mask]
        data[f"paths_{s}"] = paths[mask]
    return data


def _assert_no_group_leakage(split):
    """Fail loudly if any subject group appears in more than one split.

    Subject grouping (see _infer_subject_key) exists so a person's real
    and fake takes stay in the same split; if a group key ever straddles
    train/val/test that is data leakage and the build must abort.
    """
    seen = {}
    for s in SPLITS:
        for g in split[f"groups_{s}"]:
            g = str(g)
            if g in seen and seen[g] != s:
                raise AssertionError(
                    f"Subject-group leakage: group '{g}' appears in both "
                    f"'{seen[g]}' and '{s}' splits. Fix _infer_subject_key."
                )
            seen[g] = s


def _write_split_manifest(split, cfg):
    """Record path -> (split, group) so any future split can be reproduced.

    The manifest lives next to data.npz (gitignored) and is consumed by
    the baseline harness in Phase 1.
    """
    manifest = {
        "seed": cfg.seed,
        "val_ratio": cfg.val_ratio,
        "test_ratio": cfg.test_ratio,
        "filter_implausible": cfg.filter_implausible,
        "rows": {},
    }
    for s in SPLITS:
        for p, g in zip(split[f"paths_{s}"], split[f"groups_{s}"]):
            manifest["rows"][str(p)] = {"split": s, "group": str(g)}
    out = OUTPUT_DIR / "split_manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        json.dump(manifest, fh, indent=2)
    return out


def build_dataset(cfg=None):
    """Build data.npz from the real rPPG feature table (rPPG layer output)."""
    cfg = cfg or DataConfig()
    X, y, groups, paths, split_keys, stats = _load_rppg_rows(cfg.csv_file, cfg)
    if cfg.filter_implausible:
        print(
            f"  Plausibility filter: {stats['kept']}/{stats['total']} kept "
            f"(HR out of [{cfg.hr_min}, {cfg.hr_max}]: {stats['dropped_hr']}, "
            f"non-finite features: {stats['dropped_invalid']})"
        )
    use_official = bool(split_keys.size) and all(k is not None for k in split_keys)
    if use_official:
        print("  Using official dataset train/val/test folders (no regrouping).")
        split = _split_by_source(X, y, groups, paths, split_keys)
    else:
        split = _grouped_train_val_test_split(X, y, groups, paths, cfg)
    _assert_no_group_leakage(split)
    manifest_file = _write_split_manifest(split, cfg)
    print(f"  Split manifest written: {manifest_file}")
    cfg.data_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cfg.data_file,
        X_train=split["X_train"],
        y_train=split["y_train"],
        X_val=split["X_val"],
        y_val=split["y_val"],
        X_test=split["X_test"],
        y_test=split["y_test"],
        groups_train=np.asarray(split["groups_train"], dtype=str),
        groups_val=np.asarray(split["groups_val"], dtype=str),
        groups_test=np.asarray(split["groups_test"], dtype=str),
        paths_train=np.asarray(split["paths_train"], dtype=str),
        paths_val=np.asarray(split["paths_val"], dtype=str),
        paths_test=np.asarray(split["paths_test"], dtype=str),
        feature_names=FEATURE_NAMES,
    )
    print(f"  Built subject-grouped split from {len(X)} real rPPG samples:")
    for s in SPLITS:
        ys = split[f"y_{s}"]
        n_groups = len(set(split[f"groups_{s}"].tolist()))
        print(
            f"    {s}: {len(ys)} samples "
            f"({int((ys == LABEL_REAL).sum())} real / {int((ys == LABEL_FAKE).sum())} fake, "
            f"{n_groups} subject groups)"
        )
    return load_dataset(cfg.data_file)


def load_dataset(path=None):
    path = path or DataConfig().data_file
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Build it first with: python -m quantum.pipeline --build-data"
        )
    data = np.load(path)
    return {
        key: data[key]
        for key in (
            "X_train", "y_train", "X_val", "y_val", "X_test", "y_test",
            "groups_train", "groups_val", "groups_test",
        )
    }
