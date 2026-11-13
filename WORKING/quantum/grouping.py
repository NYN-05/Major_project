"""Stage 1: source-group verification and the frozen group-disjoint test split.

Run from WORKING/:  python -m quantum.grouping [--out DIR]

Predefined rules (fixed before looking at outcomes, see Docs/final_improvement_plan.md):
  * Union grouping = connected components over
      - clips in the same ResNet-proxy cluster (ward on 16 deep dims, n//5 clusters),
      - clips whose background signatures have cosine similarity >= LINK_THR (0.995; chosen from
        a 400-clip ground-truth validation: recall 1.0),
      - exact duplicate files.
    If the largest component exceeds 5% of the clips, fall back in order to (b) union without the
    proxy clusters, then (c) background link at 0.999.
  * Test set = one fold (20%) of a seeded StratifiedGroupKFold over union groups.
  * Guard band = development clips purged if they share a proxy cluster or a background@0.95
    connected component with any test clip. Purged clips are used nowhere.
  * Development folds = seeded 5-fold StratifiedGroupKFold over the remaining groups.
Gate: zero test/dev overlap under clip, proxy, background@0.95 and union groupings; development
folds group-disjoint under the union grouping; no giant component.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

from quantum.config import FUSED_FEATURE_NAMES, OUTPUT_ROOT, DataConfig
from quantum.provenance import sha256_file

LINK_THR, GUARD_THR, GIANT_FRAC = 0.995, 0.95, 0.05
TEST_SEED, DEV_SEED, N_FOLDS = 2026, 0, 5
MANIFEST_NAME = "split_manifest_stage1.json"
LOCK_NAME = "TEST_LOCK.json"
ACCESSED_NAME = "TEST_ACCESSED.lock"


def signature_from_frames(frames_rgb_112: np.ndarray) -> np.ndarray:
    """Background signature: median over frames of the outer ring of a 32x32 colour thumbnail
    (face centre masked), mean-centred and L2-normalised, so a dot product is cosine similarity.

    ``frames_rgb_112``: N x 112 x 112 x 3 uint8 (RGB).
    """
    import cv2

    th = np.stack(
        [cv2.resize(f, (32, 32), interpolation=cv2.INTER_AREA) for f in frames_rgb_112]
    ).astype(np.float32)
    med = np.median(th, axis=0)
    mask = np.ones((32, 32), bool)
    mask[8:24, 8:24] = False
    v = med[mask].reshape(-1)
    v = v - v.mean()
    return v / (np.linalg.norm(v) + 1e-8)


def components(adj_pairs, n: int) -> np.ndarray:
    """Union-find connected components over index pairs; returns a root id per node."""
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i, j in adj_pairs:
        parent[find(i)] = find(j)
    return np.array([find(i) for i in range(n)])


def dense_ids(arr) -> np.ndarray:
    return np.unique(arr, return_inverse=True)[1]


def union_groups(n, proxy, sim, link_thr, use_proxy, dup_pairs) -> np.ndarray:
    ii, jj = np.where(np.triu(sim >= link_thr, 1))
    pairs = list(zip(ii.tolist(), jj.tolist())) + list(dup_pairs)
    if use_proxy:
        first: dict = {}
        for i, c in enumerate(proxy):
            if c in first:
                pairs.append((first[c], i))
            else:
                first[c] = i
    return dense_ids(components(pairs, n))


def build_split(X_all, y, g_clip, stems, paths, sim, proxy, dup_pairs, *, csv_sha256="", names=None):
    """Pure split construction from precomputed signals; returns the manifest dict."""
    from sklearn.model_selection import StratifiedGroupKFold

    n = len(y)
    attempts = []
    for label, thr, use_proxy in (
        ("a: proxy+bg@0.995+dups", LINK_THR, True),
        ("b: bg@0.995+dups", LINK_THR, False),
        ("c: bg@0.999+dups", 0.999, False),
    ):
        grp = union_groups(n, proxy, sim, thr, use_proxy, dup_pairs)
        sizes = np.bincount(grp)
        attempts.append({
            "rule": label, "n_groups": int(len(sizes)), "largest": int(sizes.max()),
            "largest_frac": float(sizes.max() / n),
        })
        if sizes.max() <= GIANT_FRAC * n:
            break
    union_rule = attempts[-1]["rule"]
    sizes = np.bincount(grp)
    giant_ok = bool(sizes.max() <= GIANT_FRAC * n)

    folds = list(StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=TEST_SEED).split(X_all, y, grp))
    dev_idx, test_idx = folds[0]
    dev_set = set(dev_idx.tolist())

    # whole connected components, not just direct neighbours, so no purged-to-dev chain survives
    ii, jj = np.where(np.triu(sim >= GUARD_THR, 1))
    bg95 = dense_ids(components(list(zip(ii.tolist(), jj.tolist())), n))
    test_proxy = set(proxy[test_idx].tolist())
    test_bg = set(bg95[test_idx].tolist())
    purge = np.array([(i in dev_set) and (proxy[i] in test_proxy or bg95[i] in test_bg) for i in range(n)])
    dev_keep = np.array([i for i in dev_idx if not purge[i]], dtype=int)
    purged = np.array([i for i in dev_idx if purge[i]], dtype=int)

    dev_folds = list(
        StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=DEV_SEED)
        .split(X_all[dev_keep], y[dev_keep], grp[dev_keep])
    )
    dev_fold_of = np.full(n, -1)
    for k, (_, te) in enumerate(dev_folds):
        dev_fold_of[dev_keep[te]] = k
    split_of = np.array(["dev"] * n, dtype=object)
    split_of[test_idx] = "test"
    split_of[purged] = "purged"

    groupings = {
        "clip": dense_ids(np.array([str(g) for g in g_clip])), "proxy": proxy,
        "background@0.95": bg95, "union": grp,
    }
    overlap = {k: int(len(set(g[test_idx].tolist()) & set(g[dev_keep].tolist()))) for k, g in groupings.items()}
    fold_overlap = {}
    for k, g in groupings.items():
        per = [set(g[dev_keep[te]].tolist()) for _, te in dev_folds]
        fold_overlap[k] = int(sum(len(per[a] & per[b]) for a in range(N_FOLDS) for b in range(a + 1, N_FOLDS)))
    gate = bool(giant_ok and all(v == 0 for v in overlap.values()) and fold_overlap["union"] == 0)

    def stat(idx):
        return {
            "clips": int(len(idx)), "real": int((y[idx] == 1).sum()), "fake": int((y[idx] == 0).sum()),
            "union_groups": int(len(set(grp[idx].tolist()))),
        }

    split_hash = hashlib.sha256(json.dumps(
        [[stems[i], int(grp[i]), str(split_of[i]), int(dev_fold_of[i])] for i in range(n)],
        separators=(",", ":")).encode()).hexdigest()
    return {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "fused_features_csv_sha256": csv_sha256,
        "feature_names": list(names or []),
        "rules": {
            "link_thr": LINK_THR, "guard_thr": GUARD_THR, "giant_fraction_cap": GIANT_FRAC,
            "test_seed": TEST_SEED, "dev_seed": DEV_SEED, "n_folds": N_FOLDS,
            "test_fraction": 1 / N_FOLDS, "union_rule_used": union_rule, "grouping_attempts": attempts,
        },
        "stats": {
            "all": stat(np.arange(n)), "test": stat(test_idx), "dev": stat(dev_keep),
            "purged": stat(purged), "dev_folds": [stat(dev_keep[te]) for _, te in dev_folds],
        },
        "largest_union_group": int(sizes.max()),
        "overlap_test_vs_dev_by_grouping": overlap,
        "dev_fold_group_overlap_by_grouping": fold_overlap,
        "gate_passed": gate,
        "split_hash": split_hash,
        "caveat": (
            "Groups are heuristics validated on a 400-clip ground-truth sample (not on these clips). "
            "181 clips dropped upstream by the rPPG step are not part of this split."
        ),
        "rows": {
            stems[i]: {
                "video_path": str(paths[i]), "label_real1": int(y[i]), "group": f"u{int(grp[i])}",
                "split": str(split_of[i]), "dev_fold": int(dev_fold_of[i]),
            }
            for i in range(n)
        },
    }


def test_set_accessed(out_dir: str | Path) -> bool:
    return (Path(out_dir) / ACCESSED_NAME).exists()


def assert_test_unlocked(out_dir: str | Path) -> None:
    """Call before any test-set evaluation; a second look is contaminated."""
    if test_set_accessed(out_dir):
        raise RuntimeError(f"Test set already evaluated ({Path(out_dir) / ACCESSED_NAME}); a second look is contaminated")


def build_stage1(csv_file=None, frames_root=None, out_dir=None, device=None, overwrite=False) -> dict:
    """Compute signals (GPU when available), build the split, write manifest + test lock."""
    import torch
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.preprocessing import StandardScaler

    from quantum.data import _load_feature_rows

    csv_file = Path(csv_file or OUTPUT_ROOT / "visual" / "fused_features.csv")
    frames_root = Path(frames_root or OUTPUT_ROOT / "visual" / "frames112")
    out_dir = Path(out_dir or OUTPUT_ROOT / "stage1")
    if (out_dir / MANIFEST_NAME).exists() and not overwrite:
        raise FileExistsError(f"{out_dir / MANIFEST_NAME} exists; the frozen split is not overwritten")
    out_dir.mkdir(parents=True, exist_ok=True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    X_all, y, g_clip, paths, _, _ = _load_feature_rows(csv_file, FUSED_FEATURE_NAMES, DataConfig())
    n = len(y)
    stems = [Path(p).stem for p in paths]
    names = list(FUSED_FEATURE_NAMES)

    S = torch.tensor(
        np.stack([signature_from_frames(np.load(frames_root / f"{s}.npy")) for s in stems]), device=device
    )
    sim = (S @ S.T).cpu().numpy()
    np.fill_diagonal(sim, 1.0)
    Z = StandardScaler().fit_transform(X_all[:, [names.index(f"deep_feat_{i}") for i in range(16)]])
    proxy = AgglomerativeClustering(n_clusters=n // 5, linkage="ward").fit_predict(Z)
    byhash: dict = {}
    for i, p in enumerate(paths):
        byhash.setdefault(hashlib.sha256(Path(p).read_bytes()).hexdigest(), []).append(i)
    dup_pairs = [(v[0], j) for v in byhash.values() if len(v) > 1 for j in v[1:]]

    manifest = build_split(
        X_all, y, g_clip, stems, paths, sim, proxy, dup_pairs,
        csv_sha256=sha256_file(csv_file), names=names,
    )
    (out_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=1))
    lock = {
        "split_hash": manifest["split_hash"], "test_clips": manifest["stats"]["test"]["clips"],
        "test_accessed": False,
        "policy": (
            f"The test set is evaluated exactly once, in Stage 6, after code, configs and threshold are "
            f"frozen. Create {ACCESSED_NAME} in this folder when doing so; a second evaluation is contaminated."
        ),
    }
    (out_dir / LOCK_NAME).write_text(json.dumps(lock, indent=2))
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--csv", default=None)
    ap.add_argument("--frames-root", default=None)
    ap.add_argument("--out", default=None, help="default: MAJ_OUTPUT_ROOT/stage1")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    manifest = build_stage1(args.csv, args.frames_root, args.out, overwrite=args.overwrite)
    print(json.dumps({k: v for k, v in manifest.items() if k not in ("rows", "feature_names")}, indent=2))
    sys.exit(0 if manifest["gate_passed"] else 1)


if __name__ == "__main__":
    main()
