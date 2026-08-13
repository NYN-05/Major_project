"""
probe_features.py
=================
Quick featurability probe: process a subset of clips and compute a richer
signal-level feature vector per clip (window jitter, spectral shape,
cross-region phase coherence, two-half HR consistency, pulse regularity)
on top of the 10 canonical features. Prints per-feature AUC per class to
judge whether ANY rPPG-derived signal separates real vs fake before
committing to a full re-extraction.

Usage (from WORKING/RPPG/):
    python rppg-pipeline/probe_features.py --ffpp --max-per-class 120
"""

import argparse
import glob
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
from scipy import signal as sig
from scipy.stats import kurtosis as _kurtosis

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rppg import RPPGPipeline  # noqa: E402

BAND_LO, BAND_HI = 0.7, 4.0

# Upper bound (seconds) for one clip before the parent gives up on the worker.
ITEM_TIMEOUT_S = 600

_WORKER: dict = {}


def _init_worker(method: str, target_fps: Optional[float], min_usable: int) -> None:
    try:
        os.dup2(os.open(os.devnull, os.O_WRONLY), 2)
    except OSError:
        pass
    _WORKER["pipeline"] = RPPGPipeline(
        method=method, target_fps=target_fps, min_usable_frames=min_usable
    )


def _psd(x, fs):
    f, p = sig.welch(x, fs=fs, nperseg=min(64, len(x)), noverlap=0)
    inband = (f >= BAND_LO) & (f <= BAND_HI)
    return f, p, inband


def _hr_of(x, fs):
    f, p, inband = _psd(x, fs)
    if not inband.any():
        return np.nan
    peak = f[inband][np.argmax(p[inband])]
    return peak * 60.0


def _snr_of(x, fs):
    f, p, inband = _psd(x, fs)
    signal_pow = p[inband].sum()
    noise_pow = p[~inband].sum()
    if noise_pow <= 0 or signal_pow <= 0:
        return np.nan
    return 10.0 * np.log10(signal_pow / noise_pow)


def extra_features(sig_comb, sig_l, sig_r, sig_f, fs):
    out = {}

    # --- spectral shape of the combined signal ---
    f, p, inband = _psd(sig_comb, fs)
    pb = p[inband]
    if len(pb) > 1 and np.all(pb > 0):
        out["spectral_flatness"] = float(
            np.exp(np.mean(np.log(pb))) / np.mean(pb)
        )
    else:
        out["spectral_flatness"] = np.nan
    out["spectral_centroid"] = float(np.sum(pb * f[inband]) / pb.sum())
    out["peak_prominence"] = float(pb.max() / np.mean(pb))

    # --- time-domain pulse regularity ---
    peaks, _ = sig.find_peaks(sig_comb, distance=int(0.35 * fs))
    if len(peaks) >= 3:
        iv = np.diff(peaks) / fs
        out["pulse_cv_interval"] = float(np.std(iv) / np.mean(iv))
    else:
        out["pulse_cv_interval"] = np.nan
    out["zero_crossing_rate"] = float(
        np.mean(np.abs(np.diff(np.sign(sig_comb)))) / 2.0
    )
    out["kurtosis"] = float(_kurtosis(sig_comb))

    # --- two-half HR consistency ---
    n = len(sig_comb)
    half = max(15, n // 2)
    hr1 = _hr_of(sig_comb[:half], fs)
    hr2 = _hr_of(sig_comb[-half:], fs)
    out["hr_half_diff"] = float(abs(hr1 - hr2)) if np.isfinite(hr1 + hr2) else np.nan

    # --- cross-region phase coherence (Hilbert) ---
    def _phase_coherence(a, b):
        if a is None or b is None or len(a) < 8:
            return np.nan
        pa = np.angle(sig.hilbert(a))
        pb = np.angle(sig.hilbert(b))
        d = (pa - pb + np.pi) % (2 * np.pi) - np.pi
        return float(np.std(d))

    out["phase_coherence_LR"] = _phase_coherence(sig_l, sig_r)
    out["phase_coherence_CF"] = _phase_coherence(sig_l, sig_f)

    # --- window jitter: HR/SNR variance across windows ---
    ws = max(10, n // 3)
    hrs, snrs = [], []
    for k in range(0, n - ws + 1, ws):
        w = sig_comb[k : k + ws]
        h, s = _hr_of(w, fs), _snr_of(w, fs)
        if np.isfinite(h):
            hrs.append(h)
        if np.isfinite(s):
            snrs.append(s)
    out["hr_window_jitter"] = float(np.std(hrs)) if len(hrs) >= 2 else np.nan
    out["snr_window_jitter"] = float(np.std(snrs)) if len(snrs) >= 2 else np.nan
    return out


def _process_one(item: Tuple[int, Path, str]) -> dict:
    label, video_path, source = item
    entry = {"label": label, "source": source}
    try:
        r = _WORKER["pipeline"].process_video(str(video_path))
    except Exception as exc:  # noqa: BLE001
        entry["error"] = f"{type(exc).__name__} (pid {os.getpid()}): {exc}"
        return entry
    if r.features is None:
        entry["no_features"] = True
        return entry
    fv = r.features.to_dict()
    feats = extra_features(
        r.combined_signal, r.left_cheek_signal, r.right_cheek_signal, r.forehead_signal, r.fps
    )
    entry.update(fv)
    entry.update(feats)
    return entry


def collect_ffpp(max_per_class: int) -> List[Tuple[int, Path, str]]:
    root = Path(__file__).resolve().parent.parent.parent.parent / "FF++"
    fake, real = [], []
    for split in ("train",):
        fake += glob.glob(str(root / split / "FF-synthesis" / "*.mp4"))
        real += glob.glob(str(root / split / "FF-real" / "*.mp4"))
        real += glob.glob(str(root / split / "YouTube-real" / "*.mp4"))
    fake = sorted(fake)[:max_per_class]
    real = sorted(real)[:max_per_class]
    return [(1, Path(p), "FF-synthesis") for p in fake] + [
        (0, Path(p), "FF-real") for p in real
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ffpp", action="store_true")
    parser.add_argument("--max-per-class", type=int, default=120)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    items = []
    if args.ffpp:
        items = collect_ffpp(args.max_per_class)
    if not items:
        print("No clips selected. Pass --ffpp.")
        return 1

    print(f"Probing {len(items)} clips ...")
    with mp.Pool(args.workers, initializer=_init_worker, initargs=("POS", 10.0, 30)) as pool:
        results = iter(pool.imap_unordered(_process_one, items, chunksize=1))
        entries = []
        done = 0
        while done < len(items):
            try:
                entries.append(results.next(timeout=ITEM_TIMEOUT_S))
            except mp.TimeoutError:
                pool.terminate()
                raise SystemExit(
                    f"FATAL: worker hung for > {ITEM_TIMEOUT_S}s on item {done + 1}/{len(items)} "
                    "(no result received). Pool terminated."
                ) from None
            except StopIteration:
                break
            done += 1

    rows = [e for e in entries if "error" not in e and not e.get("no_features")]
    print(f"  ok={len(rows)} failed={sum('error' in e for e in entries)} no-feat={sum(e.get('no_features', False) for e in entries)}")

    if not rows:
        print("No usable clips.")
        return 1
    from sklearn.metrics import roc_auc_score

    y = np.array([int(r["label"]) for r in rows])
    keys = sorted({k for r in rows for k in r if k not in ("label", "source")})
    print(f"  {'feature':32s} {'AUC':>6s}")
    for k in keys:
        x = np.array([r.get(k, np.nan) for r in rows], dtype=float)
        valid = np.isfinite(x)
        if valid.sum() < 20 or len(set(y[valid])) < 2:
            continue
        auc = roc_auc_score(y[valid], x[valid])
        print(f"  {k:32s} {auc:.3f}")

    if args.output:
        import csv

        with open(args.output, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys + ["label", "source"])
            w.writeheader()
            w.writerows(rows)
        print(f"Saved {len(rows)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
