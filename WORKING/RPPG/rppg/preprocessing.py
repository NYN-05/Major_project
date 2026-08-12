"""
preprocessing.py
=================
Signal cleaning utilities applied to raw and reconstructed rPPG
traces: detrending, bandpass filtering, normalization, and simple
frame-quality-driven interpolation for dropped/occluded frames.
"""

from functools import lru_cache
from typing import Optional

import numpy as np
from scipy import signal, sparse
from scipy.sparse.linalg import spsolve


def interpolate_missing(trace: np.ndarray) -> np.ndarray:
    """
    Linearly interpolate NaN gaps (e.g. from occluded/rejected frames)
    in a 1D signal. Edge NaNs are filled with the nearest valid value.
    """
    trace = trace.astype(np.float64).copy()
    nans = np.isnan(trace)
    if not nans.any():
        return trace
    if nans.all():
        raise ValueError("Signal is entirely missing; cannot interpolate.")

    idx = np.arange(len(trace))
    trace[nans] = np.interp(idx[nans], idx[~nans], trace[~nans])
    return trace


def detrend_tarvainen(trace: np.ndarray, lam: float = 300.0) -> np.ndarray:
    """
    Smoothness-priors detrending (Tarvainen et al., 2002).

    Removes slow non-stationary trends (illumination drift, gradual
    lighting changes) while preserving the higher-frequency
    pulsatile component. This is the standard detrending approach
    used in HRV/rPPG literature, and is more principled than a
    simple moving-average subtraction.

    Parameters
    ----------
    trace : 1D array
    lam   : smoothing regularization strength. Higher = smoother
            trend removed (less aggressive), lower = more aggressive
            high-pass behavior. 300 is a reasonable default for
            ~30fps windows of a few seconds.
    """
    trace = np.asarray(trace, dtype=np.float64)
    n = len(trace)
    if n < 3:
        return trace - np.mean(trace)

    regularized = _detrend_matrix(n, lam)
    trend = spsolve(regularized, trace)
    return trace - trend


@lru_cache(maxsize=16)
def _detrend_matrix(n: int, lam: float):
    """Smoothness-priors regularization matrix (identical for any trace
    of the same length), built once and reused across the per-ROI and
    combined signals of a video."""
    identity = sparse.eye(n, format="csc")
    d2 = sparse.diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(n - 2, n), format="csc")
    return (identity + (lam ** 2) * (d2.T @ d2)).tocsc()


def bandpass_filter(
    trace: np.ndarray,
    fs: float,
    low_hz: float = 0.7,
    high_hz: float = 4.0,
    order: int = 4,
) -> np.ndarray:
    """
    Zero-phase Butterworth bandpass filter restricting the signal to
    the plausible human heart-rate band.

    0.7-4.0 Hz corresponds to 42-240 BPM, covering resting through
    elevated heart rates while rejecting baseline drift and
    high-frequency camera/compression noise.
    """
    trace = np.asarray(trace, dtype=np.float64)
    nyq = fs / 2.0
    low = max(low_hz / nyq, 1e-4)
    high = min(high_hz / nyq, 0.999)
    if low >= high:
        raise ValueError("Invalid filter band for given sampling rate.")

    b, a = _bandpass_coefficients(order, fs, low, high)
    padlen = 3 * max(len(a), len(b))
    if len(trace) <= padlen:
        # Too short to filtfilt safely; fall back to lfilter.
        return signal.lfilter(b, a, trace)
    return signal.filtfilt(b, a, trace)


@lru_cache(maxsize=16)
def _bandpass_coefficients(order: int, fs: float, low: float, high: float):
    """Butterworth coefficients — identical for all signals of a video,
    so the design is computed once instead of once per ROI signal."""
    return signal.butter(order, [low, high], btype="band")


def zscore_normalize(trace: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-variance normalization of a signal window."""
    trace = np.asarray(trace, dtype=np.float64)
    std = trace.std()
    if std < 1e-8:
        return trace - trace.mean()
    return (trace - trace.mean()) / std


def clean_signal(
    trace: np.ndarray,
    fs: float,
    low_hz: float = 0.7,
    high_hz: float = 4.0,
    detrend_lambda: float = 300.0,
) -> np.ndarray:
    """Full cleaning chain: interpolate -> detrend -> bandpass -> normalize."""
    trace = interpolate_missing(trace)
    trace = detrend_tarvainen(trace, lam=detrend_lambda)
    trace = bandpass_filter(trace, fs=fs, low_hz=low_hz, high_hz=high_hz)
    trace = zscore_normalize(trace)
    return trace
