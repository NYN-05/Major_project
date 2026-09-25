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
  * Systolic peak width (morphology)
  * Diastolic notch ratio (morphology)
  * Forehead-cheek phase lag (spatial)
  * Signal-to-motion ratio (artifact rejection)
  * Peak amplitude variability (natural variation)
  * Pulse transit time proxy (inter-ROI timing)
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
    systolic_peak_width: float
    diastolic_notch_ratio: float
    forehead_cheek_phase_lag: float
    signal_to_motion_ratio: float
    peak_amplitude_variability: float
    pulse_transit_time_proxy: float
    # --- Probe features (Phase 4 upstream improvement) ---
    spectral_flatness: float
    spectral_centroid: float
    kurtosis: float
    phase_coherence_lr: float
    phase_coherence_cf: float
    pulse_cv_interval: float
    zero_crossing_rate: float

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
                self.systolic_peak_width,
                self.diastolic_notch_ratio,
                self.forehead_cheek_phase_lag,
                self.signal_to_motion_ratio,
                self.peak_amplitude_variability,
                self.pulse_transit_time_proxy,
                # Probe features (Phase 4)
                self.spectral_flatness,
                self.spectral_centroid,
                self.kurtosis,
                self.phase_coherence_lr,
                self.phase_coherence_cf,
                self.pulse_cv_interval,
                self.zero_crossing_rate,
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
            "systolic_peak_width",
            "diastolic_notch_ratio",
            "forehead_cheek_phase_lag",
            "signal_to_motion_ratio",
            "peak_amplitude_variability",
            "pulse_transit_time_proxy",
            # Probe features (Phase 4)
            "spectral_flatness",
            "spectral_centroid",
            "kurtosis",
            "phase_coherence_lr",
            "phase_coherence_cf",
            "pulse_cv_interval",
            "zero_crossing_rate",
        ]

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


def _welch_psd(trace: np.ndarray, fs: float, nperseg: Optional[int] = None):
    """
    Compute power spectral density with zero-padding for finer
    frequency resolution. Zero-padding to nperseg > len(trace)
    interpolates the spectrum without adding new information,
    resolving the ~12 BPM quantization grid of short clips.

    Default zero-pads to at least 256 samples (or fs*8) for
    ~0.12 Hz resolution at 30fps, vs native ~0.2 Hz at 148 frames.
    """
    n = len(trace)
    if nperseg is None:
        # Default: zero-pad to at least 256 samples (or fs*8, whichever is larger)
        # This gives ~0.12 Hz resolution at 30fps instead of ~0.2 Hz native
        nperseg = max(int(fs * 8), 256)
    # Manual zero-padding: scipy.signal.welch ignores nperseg > len(signal)
    if nperseg > n:
        trace = np.pad(trace, (0, nperseg - n), mode="constant")
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
    shoulder_hz: float = 0.45,
    psd: Optional[tuple] = None,
) -> float:
    """
    SNR (dB) between power at the estimated HR fundamental (+ first
    harmonic) versus the remaining power in the physiological band.
    Higher SNR indicates a cleaner, more trustworthy pulse signal --
    genuine faces typically yield higher SNR than manipulated/flat
    deepfake regions under rPPG extraction.

    The signal windows sit at +/- `harmonic_width_hz` around the
    fundamental (and its first harmonic); the excluded `shoulder_hz`
    band counts the local spectral floor around each peak as noise, so
    sharp, concentrated peaks score higher than flat spectra.
    """
    freqs, psd = psd if psd is not None else _welch_psd(trace, fs)
    band_mask = (freqs >= low_hz) & (freqs <= high_hz)
    if not band_mask.any() or psd[band_mask].sum() <= 0:
        return float("nan")

    hr_freq = freqs[band_mask][np.argmax(psd[band_mask])]

    def near(f_target, width):
        return (freqs >= f_target - width) & (freqs <= f_target + width)

    signal_mask = near(hr_freq, harmonic_width_hz) | near(2 * hr_freq, harmonic_width_hz)
    noise_mask = ~(near(hr_freq, shoulder_hz) | near(2 * hr_freq, shoulder_hz))
    signal_power = psd[signal_mask & band_mask].sum()
    noise_power = psd[noise_mask & band_mask].sum()
    noise_power = max(noise_power, 1e-12)

    snr = 10 * np.log10(signal_power / noise_power) if signal_power > 0 else float("nan")
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


def systolic_peak_width(trace: np.ndarray, fs: float, low_hz: float = 0.7, high_hz: float = 4.0) -> float:
    """
    Width (ms) of the systolic peak in the pulse waveform. Real pulses
    have consistent peak widths; deepfake signals often show distorted
    morphology with wider or irregular peaks.
    """
    trace = np.asarray(trace, dtype=np.float64)
    if len(trace) < 8:
        return float("nan")
    from scipy.signal import find_peaks
    min_dist = max(1, int(fs / high_hz))
    peaks, _ = find_peaks(trace, distance=min_dist, prominence=0.1 * np.std(trace))
    if len(peaks) < 2:
        return float("nan")
    widths_samples, _, _, _ = signal.peak_widths(trace, peaks, rel_height=0.5)
    widths_ms = (widths_samples / fs) * 1000.0
    return float(np.median(widths_ms))


def diastolic_notch_ratio(trace: np.ndarray, fs: float, low_hz: float = 0.7, high_hz: float = 4.0) -> float:
    """
    Ratio of dicrotic notch depth to systolic peak height. Real arterial
    pulses show a distinct dicrotic notch; synthetic signals often lack
    this morphological feature (ratio near 0).
    """
    from scipy.signal import find_peaks
    trace = np.asarray(trace, dtype=np.float64)
    if len(trace) < 16:
        return float("nan")
    min_dist = max(1, int(fs / high_hz))
    systolic_peaks, _ = find_peaks(trace, distance=min_dist, prominence=0.05 * np.std(trace))
    if len(systolic_peaks) < 2:
        return float("nan")
    notch_depths = []
    for i in range(len(systolic_peaks) - 1):
        start = systolic_peaks[i]
        end = systolic_peaks[i + 1]
        segment = trace[start:end]
        if len(segment) > 3:
            min_idx = np.argmin(segment[1:-1]) + 1
            peak_val = trace[start]
            notch_val = segment[min_idx]
            if peak_val > notch_val:
                depth = (peak_val - notch_val) / (peak_val + 1e-12)
                notch_depths.append(depth)
    if not notch_depths:
        return 0.0
    return float(np.mean(notch_depths))


def forehead_cheek_phase_lag(forehead_signal: Optional[np.ndarray], cheek_signal: Optional[np.ndarray], fs: float) -> float:
    """
    Phase lag (ms) between forehead and cheek pulse signals. Real blood
    flow propagation creates a measurable delay; deepfake signals lack
    this physiological timing.

    Constrains search to physiologically plausible range (< 200ms)
    to avoid spurious correlation peaks at large lags.
    """
    if forehead_signal is None or cheek_signal is None:
        return float("nan")
    n = min(len(forehead_signal), len(cheek_signal))
    if n < 32:
        return float("nan")
    fh = forehead_signal[:n]
    ch = cheek_signal[:n]
    if fh.std() < 1e-8 or ch.std() < 1e-8:
        return float("nan")
    # Constrain to physiologically plausible lag range: ±200ms
    max_lag_samples = int(0.2 * fs)  # 200ms
    max_lag_samples = min(max_lag_samples, n - 1)
    correlation = np.correlate(fh - fh.mean(), ch - ch.mean(), mode="full")
    lags = np.arange(-n + 1, n)
    # Only consider lags within ±max_lag_samples
    valid = (lags >= -max_lag_samples) & (lags <= max_lag_samples)
    if not valid.any():
        return float("nan")
    lag_samples = lags[valid][np.argmax(correlation[valid])]
    return float(abs(lag_samples) / fs * 1000.0)


def signal_to_motion_ratio(trace: np.ndarray, fs: float, low_hz: float = 0.7, high_hz: float = 4.0, psd: Optional[tuple] = None) -> float:
    """
    Ratio of physiological signal power to motion artifact power.
    Real videos have higher SMR (clean pulse); deepfakes often have
    motion artifacts that corrupt the rPPG signal.
    """
    freqs, psd = psd if psd is not None else _welch_psd(trace, fs)
    physio_band = (freqs >= low_hz) & (freqs <= high_hz)
    motion_band = (freqs >= 0.1) & (freqs < low_hz)
    physio_power = psd[physio_band].sum() if physio_band.any() else 0
    motion_power = psd[motion_band].sum() if motion_band.any() else 1e-12
    return float(10 * np.log10(max(physio_power, 1e-12) / max(motion_power, 1e-12)))


def peak_amplitude_variability(trace: np.ndarray, fs: float, low_hz: float = 0.7, high_hz: float = 4.0) -> float:
    """
    Coefficient of variation of systolic peak amplitudes. Real pulses
    show natural amplitude variation; deepfake signals often have
    unnaturally consistent or erratic amplitudes.
    """
    from scipy.signal import find_peaks
    trace = np.asarray(trace, dtype=np.float64)
    if len(trace) < 16:
        return float("nan")
    min_dist = max(1, int(fs / high_hz))
    peaks, _ = find_peaks(trace, distance=min_dist, prominence=0.05 * np.std(trace))
    if len(peaks) < 3:
        return float("nan")
    amplitudes = trace[peaks]
    mean_amp = np.mean(amplitudes)
    if mean_amp < 1e-8:
        return float("nan")
    return float(np.std(amplitudes) / mean_amp)


def pulse_transit_time_proxy(sig_a: Optional[np.ndarray], sig_b: Optional[np.ndarray], fs: float) -> float:
    """
    Proxy for pulse transit time: time delay (ms) between two ROI
    signals. Real blood flow has measurable propagation delays;
    deepfake signals lack this physiological timing structure.
    """
    return forehead_cheek_phase_lag(sig_a, sig_b, fs)


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


# --- New probe features (Phase 4 upstream improvement) ---

def spectral_flatness(trace: np.ndarray, fs: float, low_hz: float = 0.7, high_hz: float = 4.0, psd: Optional[tuple] = None) -> float:
    """Geometric mean / arithmetic mean of in-band PSD. Flat spectra (noise) -> 1.0; tonal (pulse) -> < 1.0."""
    freqs, psd = psd if psd is not None else _welch_psd(trace, fs)
    band = (freqs >= low_hz) & (freqs <= high_hz)
    pb = psd[band]
    if len(pb) < 2 or np.any(pb <= 0):
        return float("nan")
    return float(np.exp(np.mean(np.log(pb))) / np.mean(pb))


def spectral_centroid(trace: np.ndarray, fs: float, low_hz: float = 0.7, high_hz: float = 4.0, psd: Optional[tuple] = None) -> float:
    """Center of mass of the in-band power spectrum (Hz). Higher = more high-frequency content."""
    freqs, psd = psd if psd is not None else _welch_psd(trace, fs)
    band = (freqs >= low_hz) & (freqs <= high_hz)
    pb = psd[band]
    if pb.sum() <= 0:
        return float("nan")
    return float(np.sum(pb * freqs[band]) / pb.sum())


def kurtosis(trace: np.ndarray) -> float:
    """Fisher kurtosis of the time-domain pulse waveform. Heavy-tailed (artifacts) -> high kurtosis."""
    from scipy.stats import kurtosis as _kurtosis
    trace = np.asarray(trace, dtype=np.float64)
    if len(trace) < 4:
        return float("nan")
    return float(_kurtosis(trace))


def _phase_coherence(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    """Phase coherence (std of instantaneous phase difference) between two signals. Lower = more coherent."""
    if a is None or b is None or len(a) < 8:
        return float("nan")
    pa = np.angle(signal.hilbert(a))
    pb = np.angle(signal.hilbert(b))
    d = (pa - pb + np.pi) % (2 * np.pi) - np.pi
    return float(np.std(d))


def pulse_cv_interval(trace: np.ndarray, fs: float) -> float:
    """Coefficient of variation of inter-beat intervals. Regular pulse -> low CV."""
    peaks, _ = signal.find_peaks(trace, distance=max(1, int(fs / 4.0)))
    if len(peaks) < 3:
        return float("nan")
    iv = np.diff(peaks) / fs
    mean_iv = np.mean(iv)
    if mean_iv <= 0:
        return float("nan")
    return float(np.std(iv) / mean_iv)


def zero_crossing_rate(trace: np.ndarray) -> float:
    """Zero-crossing rate of the pulse waveform. Higher = more high-frequency content/noise."""
    trace = np.asarray(trace, dtype=np.float64)
    if len(trace) < 2:
        return float("nan")
    return float(np.mean(np.abs(np.diff(np.sign(trace)))) / 2.0)


def _window_jitter(
    trace: np.ndarray,
    fs: float,
    low_hz: float = 0.7,
    high_hz: float = 4.0,
    window_s: float = 2.0,
    max_windows: int = 5,
    feature: str = "hr",  # "hr" or "snr"
) -> float:
    """Std of per-window HR or SNR estimates across the clip. Stable pulse -> low jitter.

    Adapts window size for short signals and uses 50% overlapping windows
    to ensure at least 2 windows for stability estimation.
    """
    trace = np.asarray(trace, dtype=np.float64)
    n = len(trace)
    min_frames = int(1.0 * fs)
    if n < min_frames:
        return float("nan")

    # Adaptive window size: target 2s windows, but shrink for short signals
    target_wlen = int(round(window_s * fs))
    if n < 2 * target_wlen:
        wlen = max(int(0.8 * fs), n // 2)
    else:
        wlen = target_wlen

    # Use 50% overlapping windows for more samples
    stride = wlen // 2
    n_windows = min(max_windows, max(2, (n - wlen) // stride + 1))

    vals = []
    for i in range(n_windows):
        start = i * stride
        end = min(start + wlen, n)
        seg = trace[start:end]
        if len(seg) < int(0.5 * fs):
            continue
        psd = _welch_psd(seg, fs)
        if feature == "hr":
            v = estimate_heart_rate(seg, fs, low_hz, high_hz, psd=psd)
        else:
            v = estimate_snr(seg, fs, low_hz, high_hz, psd=psd)
        if np.isfinite(v):
            vals.append(v)
    return float(np.std(vals)) if len(vals) >= 2 else float("nan")


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
    # Use zero-padding to next power of 2 for finer frequency resolution
    # (resolves ~12 BPM quantization grid of short clips).
    n = len(combined_signal)
    nperseg = 1 << (n - 1).bit_length()  # next power of 2 >= n
    nperseg = max(nperseg, 256)  # at least 256 for ~0.12 Hz resolution at 30fps
    shared_psd = _welch_psd(combined_signal, fs, nperseg=nperseg)

    hr = estimate_heart_rate(combined_signal, fs, low_hz, high_hz, psd=shared_psd)
    snr = estimate_snr(combined_signal, fs, low_hz, high_hz, psd=shared_psd)
    prv = estimate_prv(combined_signal, fs, low_hz, high_hz)
    ent = spectral_entropy(combined_signal, fs, low_hz, high_hz, psd=shared_psd)
    mad = mean_absolute_deviation(combined_signal)
    sqi = signal_quality_index(combined_signal, fs, psd=shared_psd)
    hrd = hr_half_diff(combined_signal, fs, low_hz, high_hz)
    ppr = peak_prominence(combined_signal, fs, low_hz, high_hz, psd=shared_psd)
    spw = systolic_peak_width(combined_signal, fs, low_hz, high_hz)
    dnr = diastolic_notch_ratio(combined_signal, fs, low_hz, high_hz)
    smr = signal_to_motion_ratio(combined_signal, fs, low_hz, high_hz, psd=shared_psd)
    pav = peak_amplitude_variability(combined_signal, fs, low_hz, high_hz)

    # --- Probe features (Phase 4) ---
    spf = spectral_flatness(combined_signal, fs, low_hz, high_hz, psd=shared_psd)
    sce = spectral_centroid(combined_signal, fs, low_hz, high_hz, psd=shared_psd)
    kur = kurtosis(combined_signal)
    pci = pulse_cv_interval(combined_signal, fs)
    zcr = zero_crossing_rate(combined_signal)
    pclr = _phase_coherence(left_cheek_signal, right_cheek_signal)
    pccf = _phase_coherence(forehead_signal, left_cheek_signal)

    cheek_corr_vals = [
        region_correlation(left_cheek_signal, forehead_signal),
        region_correlation(right_cheek_signal, forehead_signal),
    ]
    cheek_corr_vals = [c for c in cheek_corr_vals if not np.isnan(c)]
    cheek_forehead_corr = float(np.mean(cheek_corr_vals)) if cheek_corr_vals else 0.0

    left_right_corr = region_correlation(left_cheek_signal, right_cheek_signal)
    if np.isnan(left_right_corr):
        left_right_corr = 0.0

    lag_vals = [
        forehead_cheek_phase_lag(forehead_signal, left_cheek_signal, fs),
        forehead_cheek_phase_lag(forehead_signal, right_cheek_signal, fs),
    ]
    lag_vals = [l for l in lag_vals if not np.isnan(l)]
    phase_lag = float(np.mean(lag_vals)) if lag_vals else float("nan")
    ptt = pulse_transit_time_proxy(forehead_signal, left_cheek_signal, fs)
    if np.isnan(ptt):
        ptt = pulse_transit_time_proxy(forehead_signal, right_cheek_signal, fs)

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
        systolic_peak_width=spw,
        diastolic_notch_ratio=dnr,
        forehead_cheek_phase_lag=phase_lag,
        signal_to_motion_ratio=smr,
        peak_amplitude_variability=pav,
        pulse_transit_time_proxy=ptt,
        # Probe features (Phase 4)
        spectral_flatness=spf,
        spectral_centroid=sce,
        kurtosis=kur,
        phase_coherence_lr=pclr,
        phase_coherence_cf=pccf,
        pulse_cv_interval=pci,
        zero_crossing_rate=zcr,
    )
    raw_nan_count = sum(
        1
        for name in RPPGFeatures.feature_names()
        if not np.isfinite(getattr(features, name))
    )
    features._raw_nan_count = raw_nan_count  # transient; excluded from asdict()/to_vector()
    _fill_nan_with_median(features)
    return features


def _fill_nan_with_median(features: RPPGFeatures) -> None:
    """Replace any remaining NaN feature values with hardcoded neutral
    fallbacks so the feature vector is always finite and classifier-ready.

    Note: these are fixed "average human" constants, NOT per-feature medians.
    Degenerate signals (several NaNs or zero SQI) are rejected earlier by the
    RPPGPipeline gate, so this fill only ever fires for a rare single-NaN case.
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
        "systolic_peak_width": 350.0,
        "diastolic_notch_ratio": 0.0,
        "forehead_cheek_phase_lag": 0.0,
        "signal_to_motion_ratio": 0.0,
        "peak_amplitude_variability": 0.0,
        "pulse_transit_time_proxy": 0.0,
        # Probe features (Phase 4)
        "spectral_flatness": 1.0,
        "spectral_centroid": 1.5,
        "kurtosis": 0.0,
        "phase_coherence_lr": 1.0,
        "phase_coherence_cf": 1.0,
        "pulse_cv_interval": 0.05,
        "zero_crossing_rate": 0.5,
    }
    for name in fallbacks:
        value = getattr(features, name)
        if np.isnan(value):
            setattr(features, name, fallbacks[name])
