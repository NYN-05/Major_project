"""
signal_extraction.py
=====================
Chrominance-based rPPG signal reconstruction from RGB traces.

Implements two well-validated classical methods that outperform
naive green-channel averaging, especially under the illumination
variation and compression noise typical of low-resolution KYC video:

  * CHROM  - de Haan & Jeanne (2013), "Robust Pulse Rate From
             Chrominance-Based rPPG"
  * POS    - Wang et al. (2017), "Algorithmic Principles of Remote
             Photoplethysmography" (Plane-Orthogonal-to-Skin)

POS generally shows better robustness under less-controlled capture
conditions and is used as the default in this pipeline; CHROM is
provided for comparison/ablation.
"""

from typing import Tuple

import numpy as np


def _normalize_temporal(rgb_window: np.ndarray) -> np.ndarray:
    """
    Temporally normalize each RGB channel within a window by its own
    mean (per-channel), as required by both CHROM and POS. This
    removes DC skin-tone/illumination-level differences so only the
    relative pulsatile modulation remains.

    Parameters
    ----------
    rgb_window : array of shape (T, 3)

    Returns
    -------
    array of shape (T, 3), each column divided by its temporal mean.
    """
    means = rgb_window.mean(axis=0)
    means = np.where(means < 1e-6, 1e-6, means)
    return rgb_window / means


def chrom_method(rgb_window: np.ndarray) -> np.ndarray:
    """
    CHROM: linear combination of chrominance signals designed to be
    insensitive to specular reflection and motion.

    X = 3R - 2G
    Y = 1.5R + G - 1.5B
    S = X - alpha * Y,  alpha = std(X)/std(Y)

    Parameters
    ----------
    rgb_window : array (T, 3), raw mean RGB per frame over a window.

    Returns
    -------
    1D pulse signal of length T.
    """
    rgb_n = _normalize_temporal(rgb_window)
    r, g, b = rgb_n[:, 0], rgb_n[:, 1], rgb_n[:, 2]

    x = 3 * r - 2 * g
    y = 1.5 * r + g - 1.5 * b

    std_x = x.std()
    std_y = y.std()
    alpha = std_x / std_y if std_y > 1e-8 else 0.0

    return x - alpha * y


def pos_method(rgb_window: np.ndarray, fs: float, window_sec: float = 1.6) -> np.ndarray:
    """
    POS (Plane-Orthogonal-to-Skin): projects the temporally normalized
    RGB signal onto a plane orthogonal to the skin-tone direction in
    a rotated color space, then combines the two projected components
    weighted by their relative standard deviation. Operates on
    overlapping sub-windows and overlap-adds the result, as specified
    in the original algorithm.

    Parameters
    ----------
    rgb_window : array (T, 3)
    fs         : sampling rate (fps) of the input trace
    window_sec : length of the internal POS sub-window in seconds
                 (paper default ~1.6s)

    Returns
    -------
    1D pulse signal of length T.
    """
    rgb = np.asarray(rgb_window, dtype=np.float64)
    t_len = rgb.shape[0]
    win_len = max(3, int(round(window_sec * fs)))

    S = np.zeros(t_len, dtype=np.float64)
    weight_sum = np.zeros(t_len, dtype=np.float64)

    # Projection matrix onto plane orthogonal to skin-tone vector
    # (fixed matrix from Wang et al., derived analytically).
    P = np.array([[0, 1, -1], [-2, 1, 1]], dtype=np.float64)

    n_windows = t_len - win_len + 1
    if n_windows <= 0:
        weight_sum[weight_sum == 0] = 1.0
        return S / weight_sum

    # Vectorized form of the original per-window loop (identical math):
    # sliding windows -> temporal normalization -> projection ->
    # per-window alpha weighting -> overlap-add.
    windows = np.lib.stride_tricks.sliding_window_view(rgb, win_len, axis=0)
    means = windows.mean(axis=1)
    means = np.where(means < 1e-6, 1e-6, means)
    segments = windows / means[:, None, :]

    projected = segments @ P.T  # (n_windows, win_len, 2) -> [S1, S2]
    s1 = projected[..., 0]
    s2 = projected[..., 1]

    std1 = s1.std(axis=1)
    std2 = s2.std(axis=1)
    alpha = np.where(std2 > 1e-8, std1 / std2, 0.0)

    h = s1 + alpha[:, None] * s2
    h = h - h.mean(axis=1)[:, None]

    idx = np.arange(n_windows)[:, None] + np.arange(win_len)[None, :]
    np.add.at(S, idx, h)
    np.add.at(weight_sum, idx, 1.0)

    weight_sum[weight_sum == 0] = 1.0
    return S / weight_sum


def extract_pulse_signal(
    rgb_window: np.ndarray,
    fs: float,
    method: str = "POS",
) -> np.ndarray:
    """
    Dispatch helper: reconstruct a pulse signal from a raw mean-RGB
    trace using the requested method.

    Parameters
    ----------
    rgb_window : array (T, 3)
    fs         : sampling rate (fps)
    method     : "POS" or "CHROM"
    """
    method = method.upper()
    if method == "POS":
        return pos_method(rgb_window, fs=fs)
    elif method == "CHROM":
        return chrom_method(rgb_window)
    else:
        raise ValueError(f"Unknown rPPG method '{method}'. Use 'POS' or 'CHROM'.")


def combine_roi_signals(
    signals: list,
    weights: Tuple[float, ...] = None,
) -> np.ndarray:
    """
    Combine multiple ROI-derived pulse signals (e.g. left cheek,
    right cheek, forehead) into a single signal via weighted average.
    Signals must already be the same length and comparable scale
    (e.g. both z-score normalized) before combining.

    Parameters
    ----------
    signals : list of 1D arrays (same length)
    weights : optional weights, defaults to equal weighting.
    """
    valid = [s for s in signals if s is not None]
    if not valid:
        raise ValueError("No valid ROI signals to combine.")

    stacked = np.vstack(valid)
    if weights is None:
        weights = np.ones(len(valid)) / len(valid)
    else:
        weights = np.asarray(weights[: len(valid)], dtype=np.float64)
        weights = weights / weights.sum()

    return np.average(stacked, axis=0, weights=weights)
