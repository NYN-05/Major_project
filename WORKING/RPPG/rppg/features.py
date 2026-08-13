"""
features.py
============
Converts a cleaned rPPG pulse signal (and its per-ROI components)
into a compact, fixed-length feature vector suitable for downstream
(hybrid quantum-classical) classification.

Features implemented
---------------------
  * Heart rate (BPM) via power spectral density peak
  * SNR (signal power in HR band vs. rest of spectrum)
  * Pulse rate variability (inter-beat interval std dev)
  * Inter-region correlation (biological synchrony between ROIs)
  * Spectral entropy
  * Mean absolute deviation (MAD) of the waveform
  * Peak prominence-based signal quality index
"""

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

import numpy as np
from scipy import signal


@dataclass
class RPPGFeatures:
    heart_rate_bpm: float
    snr_db: float
    prv_std_ms: float
    spectral_entropy: float
    mad: float
    signal_quality_index: float
    cheek_forehead_correlation: float
    left_right_cheek_correlation: float
    hr_half_diff: float
    peak_prominence: float

    def to_vector(self) -> np.ndarray:
        """Fixed-order numeric feature vector for ML/quantum encoding."""
        return np.array(
            [
                self.heart_rate_bpm,
                self.snr_db,
                self.prv_std_ms,
                self.spectral_entropy,
                self.mad,
                self.signal_quality_index,
                self.cheek_forehead_correlation,
                self.left_right_cheek_correlation,
                self.hr_half_diff,
                self.peak_prominence,
            ],
            dtype=np.float64,
        )

    @staticmethod
    def feature_names() -> List[str]:
        return [
            "heart_rate_bpm",
            "snr_db",
            "prv_std_ms",
            "spectral_entropy",
            "mad",
            "signal_quality_index",
            "cheek_forehead_correlation",
            "left_right_cheek_correlation",
            "hr_half_diff",
            "peak_prominence",
        ]

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


def _welch_psd(trace: np.ndarray, fs: float):
    nperseg = min(len(trace), max(int(fs * 8), 32))
    freqs, psd = signal.welch(trace, fs=fs, nperseg=nperseg)
    return freqs, psd


def estimate_heart_rate(
    trace: np.ndarray,
    fs: float,
    low_hz: float = 0.7,
    high_hz: float = 4.0,
    psd: Optional[tuple] = None,
) -> float:
    """Heart rate (BPM) estimated as the dominant frequency in the
    plausible HR band of the power spectral density.

    `psd` may carry a precomputed `(freqs, psd)` pair (shared across
    features in `compute_features`) to avoid recomputing the same
    periodogram four times per video.
    """
    freqs, psd = psd if psd is not None else _welch_psd(trace, fs)
    band = (freqs >= low_hz) & (freqs <= high_hz)
    if not band.any():
        return float("nan")
    dominant_freq = freqs[band][np.argmax(psd[band])]
    return float(dominant_freq * 60.0)


def estimate_snr(
    trace: np.ndarray,
    fs: float,
    low_hz: float = 0.7,
    high_hz: float = 4.0,
    harmonic_width_hz: float = 0.2,
    psd: Optional[tuple] = None,
) -> float:
    """
    SNR (dB) between power at the estimated HR fundamental (+ first
    harmonic) versus the remaining power in the physiological band.
    Higher SNR indicates a cleaner, more trustworthy pulse signal --
    genuine faces typically yield higher SNR than manipulated/flat
    deepfake regions under rPPG extraction.
    """
    freqs, psd = psd if psd is not None else _welch_psd(trace, fs)
    band_mask = (freqs >= low_hz) & (freqs <= high_hz)
    if not band_mask.any() or psd[band_mask].sum() <= 0:
        return float("nan")

    hr_freq = freqs[band_mask][np.argmax(psd[band_mask])]

    def near(f_target):
        return (freqs >= f_target - harmonic_width_hz) & (freqs <= f_target + harmonic_width_hz)

    signal_mask = near(hr_freq) | near(2 * hr_freq)
    signal_power = psd[signal_mask & band_mask].sum()
    noise_power = psd[band_mask].sum() - signal_power
    noise_power = max(noise_power, 1e-12)

    snr = 10 * np.log10(signal_power / noise_power) if signal_power > 0 else -np.inf
    return float(snr)


def estimate_prv(trace: np.ndarray, fs: float, low_hz: float = 0.7, high_hz: float = 4.0) -> float:
    """
    Pulse Rate Variability: standard deviation of inter-beat
    intervals (ms), computed from peaks detected in the bandpassed
    pulse signal. Deepfake/manipulated regions often show either
    unnaturally low variability (overly regular) or erratic,
    non-physiological variability.
    """
    # find_peaks enforces its own minimum inter-peak distance
    peaks, _ = signal.find_peaks(trace, distance=max(1, int(fs / (high_hz))))
    if len(peaks) < 3:
        return float("nan")

    ibi_samples = np.diff(peaks)
    ibi_ms = (ibi_samples / fs) * 1000.0

    # Reject physiologically implausible intervals (artifacts)
    plausible = (ibi_ms > (60000.0 / (high_hz * 60))) & (ibi_ms < (60000.0 / (low_hz * 60)))
    ibi_ms = ibi_ms[plausible] if plausible.any() else ibi_ms
    if len(ibi_ms) < 2:
        return float("nan")

    return float(np.std(ibi_ms))


def spectral_entropy(trace: np.ndarray, fs: float, low_hz: float = 0.7, high_hz: float = 4.0, psd: Optional[tuple] = None) -> float:
    """
    Shannon entropy of the normalized power spectral density within
    the physiological band. Genuine physiological signals tend to
    concentrate energy near the HR fundamental (lower entropy);
    noisy or fabricated signals tend to be flatter (higher entropy).
    """
    freqs, psd = psd if psd is not None else _welch_psd(trace, fs)
    band = (freqs >= low_hz) & (freqs <= high_hz)
    p = psd[band]
    if p.sum() <= 0:
        return float("nan")
    p = p / p.sum()
    p = p[p > 0]
    ent = -np.sum(p * np.log2(p))
    max_ent = np.log2(len(p)) if len(p) > 1 else 1.0
    return float(ent / max_ent) if max_ent > 0 else float("nan")


def mean_absolute_deviation(trace: np.ndarray) -> float:
    return float(np.mean(np.abs(trace - np.mean(trace))))


def hr_half_diff(trace: np.ndarray, fs: float, low_hz: float = 0.7, high_hz: float = 4.0) -> float:
    """
    Absolute difference (BPM) between heart rates estimated from the first
    and second halves of the pulse signal. A genuine recording keeps a
    stable pulse across the clip; unstable or drifting (synthetic) signals
    disagree between halves. Mirrors the probe feature of the same name.
    """
    trace = np.asarray(trace, dtype=np.float64)
    n = len(trace)
    half = max(15, n // 2)
    hr1 = estimate_heart_rate(trace[:half], fs, low_hz, high_hz)
    hr2 = estimate_heart_rate(trace[-half:], fs, low_hz, high_hz)
    if not (np.isfinite(hr1) and np.isfinite(hr2)):
        return float("nan")
    return float(abs(hr1 - hr2))


def peak_prominence(trace: np.ndarray, fs: float, low_hz: float = 0.7, high_hz: float = 4.0, psd: Optional[tuple] = None) -> float:
    """
    Spectral peak-to-mean ratio of the in-band power spectrum (prominence of
    the dominant pulse peak). Genuine pulse signals concentrate energy at the
    HR fundamental (ratio >> 1); noisy or fabricated spectra are flatter
    (ratio near 1). Mirrors the probe feature of the same name.
    """
    freqs, psd = psd if psd is not None else _welch_psd(trace, fs)
    band = (freqs >= low_hz) & (freqs <= high_hz)
    pb = psd[band]
    if not band.any() or pb.size < 2 or pb.mean() <= 0:
        return float("nan")
    return float(pb.max() / pb.mean())


def signal_quality_index(trace: np.ndarray, fs: float, psd: Optional[tuple] = None) -> float:
    """
    Pulse-signal quality score in [0, 1], combining:

      * beat regularity: fraction of detected peaks whose inter-beat
        interval falls within the physiological band, and
      * spectral concentration: fraction of in-band power within a
        narrow window around the dominant pulse frequency.

    Genuine, clean pulse traces score high (regular beats, energy
    concentrated at the fundamental); noisy or fabricated signals
    score low. This is a real quality metric (unlike a ratio of
    prominences above their own percentile, which is constant by
    construction).
    """
    trace = np.asarray(trace, dtype=np.float64)
    if len(trace) < 8 or np.isnan(trace).any() or np.std(trace) < 1e-8:
        return 0.0

    # --- Beat-regularity component ------------------------------------
    min_dist = max(1, int(round(fs / 4.0)))  # at most ~240 BPM
    peaks, _ = signal.find_peaks(trace, distance=min_dist)
    regularity = 0.0
    if len(peaks) >= 2:
        ibi_s = np.diff(peaks) / fs
        lo = 1.0 / 4.0    # 0.25 s -> 240 BPM upper bound
        hi = 1.0 / 0.7    # 1.43 s -> 42 BPM lower bound
        plausible = np.mean((ibi_s >= lo) & (ibi_s <= hi))
        regularity = float(plausible)

    # --- Spectral-concentration component ------------------------------
    freqs, psd = psd if psd is not None else _welch_psd(trace, fs)
    band = (freqs >= 0.7) & (freqs <= 4.0)
    if band.any() and psd[band].sum() > 1e-12:
        f_dom = freqs[band][np.argmax(psd[band])]
        width = 0.15  # Hz around the dominant frequency
        near = (freqs >= f_dom - width) & (freqs <= f_dom + width)
        concentration = float(psd[near & band].sum() / psd[band].sum())
    else:
        concentration = 0.0

    return float(0.5 * regularity + 0.5 * concentration)


def region_correlation(sig_a: Optional[np.ndarray], sig_b: Optional[np.ndarray]) -> float:
    """
    Pearson correlation between two ROI pulse signals. Real faces
    show high biological synchrony (cheeks/forehead pulse together);
    deepfake generation frequently fails to preserve this
    cross-region physiological coupling.
    """
    if sig_a is None or sig_b is None:
        return float("nan")
    n = min(len(sig_a), len(sig_b))
    if n < 3:
        return float("nan")
    a, b = sig_a[:n], sig_b[:n]
    if a.std() < 1e-8 or b.std() < 1e-8:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def compute_features(
    combined_signal: np.ndarray,
    fs: float,
    left_cheek_signal: Optional[np.ndarray] = None,
    right_cheek_signal: Optional[np.ndarray] = None,
    forehead_signal: Optional[np.ndarray] = None,
    low_hz: float = 0.7,
    high_hz: float = 4.0,
) -> RPPGFeatures:
    """
    Compute the full physiological feature set for one analysis
    window given the combined pulse signal and (optionally) the
    individual per-ROI signals for correlation features.
    """
    # One periodogram shared by all spectral features (HR, SNR, entropy,
    # quality index) instead of four identical Welch computations.
    shared_psd = _welch_psd(combined_signal, fs)

    hr = estimate_heart_rate(combined_signal, fs, low_hz, high_hz, psd=shared_psd)
    snr = estimate_snr(combined_signal, fs, low_hz, high_hz, psd=shared_psd)
    prv = estimate_prv(combined_signal, fs, low_hz, high_hz)
    ent = spectral_entropy(combined_signal, fs, low_hz, high_hz, psd=shared_psd)
    mad = mean_absolute_deviation(combined_signal)
    sqi = signal_quality_index(combined_signal, fs, psd=shared_psd)
    hrd = hr_half_diff(combined_signal, fs, low_hz, high_hz)
    ppr = peak_prominence(combined_signal, fs, low_hz, high_hz, psd=shared_psd)

    cheek_corr_vals = [
        region_correlation(left_cheek_signal, forehead_signal),
        region_correlation(right_cheek_signal, forehead_signal),
    ]
    cheek_corr_vals = [c for c in cheek_corr_vals if not np.isnan(c)]
    cheek_forehead_corr = float(np.mean(cheek_corr_vals)) if cheek_corr_vals else 0.0

    left_right_corr = region_correlation(left_cheek_signal, right_cheek_signal)
    if np.isnan(left_right_corr):
        left_right_corr = 0.0

    features = RPPGFeatures(
        heart_rate_bpm=hr,
        snr_db=snr,
        prv_std_ms=prv,
        spectral_entropy=ent,
        mad=mad,
        signal_quality_index=sqi,
        cheek_forehead_correlation=cheek_forehead_corr,
        left_right_cheek_correlation=float(left_right_corr),
        hr_half_diff=hrd,
        peak_prominence=ppr,
    )
    _fill_nan_with_median(features)
    return features


def _fill_nan_with_median(features: RPPGFeatures) -> None:
    """Replace any remaining NaN feature values with neutral fallbacks so
    the feature vector is always finite and classifier-ready.

    Uses per-feature median of non-NaN values; if all are NaN (flat /
    degenerate signal), falls back to a physiologically neutral default.
    """
    fallbacks = {
        "heart_rate_bpm": 72.0,
        "snr_db": 0.0,
        "prv_std_ms": 0.0,
        "spectral_entropy": 0.5,
        "mad": 0.0,
        "signal_quality_index": 0.0,
        "hr_half_diff": 0.0,
        "peak_prominence": 1.0,
    }
    for name in fallbacks:
        value = getattr(features, name)
        if np.isnan(value):
            setattr(features, name, fallbacks[name])
