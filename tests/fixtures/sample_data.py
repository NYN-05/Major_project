"""Test fixtures and sample data generators."""
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path
import cv2


def generate_synthetic_rppg_signal(
    duration_sec: float = 5.0,
    fs: float = 30.0,
    heart_rate_bpm: float = 72.0,
    snr_db: float = 10.0,
    add_motion: bool = True
) -> np.ndarray:
    """Generate a synthetic rPPG signal for testing."""
    n_samples = int(duration_sec * fs)
    t = np.arange(n_samples) / fs
    
    # Heart rate signal (sinusoidal)
    hr_hz = heart_rate_bpm / 60.0
    signal = np.sin(2 * np.pi * hr_hz * t)
    
    # Add harmonics
    signal += 0.3 * np.sin(4 * np.pi * hr_hz * t)
    signal += 0.1 * np.sin(6 * np.pi * hr_hz * t)
    
    # Add motion artifacts if requested
    if add_motion:
        motion = 0.5 * np.sin(2 * np.pi * 0.5 * t)  # Low frequency motion
        motion += 0.2 * np.random.randn(n_samples)  # Random noise
        signal += motion
    
    # Add noise based on SNR
    signal_power = np.mean(signal ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.sqrt(noise_power) * np.random.randn(n_samples)
    signal += noise
    
    return signal.astype(np.float32)


def generate_synthetic_rgb_traces(
    duration_sec: float = 5.0,
    fs: float = 30.0,
    heart_rate_bpm: float = 72.0,
    snr_db: float = 10.0
) -> Dict[str, np.ndarray]:
    """Generate synthetic RGB traces for three ROIs."""
    n_samples = int(duration_sec * fs)
    t = np.arange(n_samples) / fs
    hr_hz = heart_rate_bpm / 60.0
    
    # Base pulse signal
    pulse = np.sin(2 * np.pi * hr_hz * t)
    
    # ROI-specific variations
    traces = {}
    for roi_name, phase_shift, amplitude in [
        ('left_cheek', 0.0, 1.0),
        ('right_cheek', 0.1, 0.95),
        ('forehead', 0.2, 0.8),
    ]:
        trace = amplitude * np.sin(2 * np.pi * hr_hz * t + phase_shift)
        
        # Add noise based on SNR
        signal_power = np.mean(trace ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = np.sqrt(noise_power) * np.random.randn(len(t))
        trace += noise
        
        # Add motion artifact (low frequency)
        trace += 0.1 * np.sin(2 * np.pi * 0.3 * t)
        
        traces[roi_name] = trace.astype(np.float32)
    
    return traces


def create_mock_rppg_features(
    heart_rate_bpm: float = 72.0,
    snr_db: float = 10.0,
    **kwargs
) -> Dict[str, float]:
    """Create mock rPPG features for testing."""
    defaults = {
        'heart_rate_bpm': heart_rate_bpm,
        'snr_db': snr_db,
        'prv_std_ms': 20.0,
        'spectral_entropy': 0.5,
        'mad': 0.1,
        'signal_quality_index': 0.7,
        'cheek_forehead_correlation': 0.8,
        'left_right_cheek_correlation': 0.7,
        'hr_half_diff': 2.0,
        'peak_prominence': 2.0,
        'systolic_peak_width': 150.0,
        'diastolic_notch_ratio': 0.2,
        'forehead_cheek_phase_lag': 50.0,
        'signal_to_motion_ratio': 10.0,
        'peak_amplitude_variability': 0.1,
        'pulse_transit_time_proxy': 20.0,
        'spectral_flatness': 0.3,
        'spectral_centroid': 1.5,
        'kurtosis': 0.0,
        'phase_coherence_lr': 0.5,
        'phase_coherence_cf': 0.5,
        'pulse_cv_interval': 0.05,
        'zero_crossing_rate': 0.3,
    }
    defaults.update(kwargs)
    return defaults


def create_mock_visual_features(
    deep_feat_dim: int = 16,
    **kwargs
) -> Dict[str, float]:
    """Create mock visual features for testing."""
    defaults = {}
    # Deep features
    for i in range(deep_feat_dim):
        defaults[f'deep_feat_{i}'] = np.random.randn()
    # LBP histogram
    for i in range(10):
        defaults[f'lbp_uniform_hist_{i}'] = np.random.rand()
    # Texture
    defaults.update({
        'texture_contrast': np.random.rand(),
        'texture_energy': np.random.rand(),
        'texture_homogeneity': np.random.rand(),
        'texture_correlation': np.random.rand(),
    })
    # Color
    defaults.update({
        'color_mean_r': np.random.rand(),
        'color_mean_g': np.random.rand(),
        'color_mean_b': np.random.rand(),
        'color_std_r': np.random.rand(),
        'color_std_g': np.random.rand(),
        'color_std_b': np.random.rand(),
    })
    # Frequency
    defaults.update({
        'freq_low_energy': np.random.rand(),
        'freq_mid_energy': np.random.rand(),
        'freq_high_energy': np.random.rand(),
    })
    defaults.update(kwargs)
    return defaults


def create_test_dataset(
    n_samples: int = 100,
    feature_set: str = 'rppg_only',
    class_balance: float = 0.5,
    random_state: int = 42
) -> tuple:
    """Create a synthetic test dataset."""
    np.random.seed(random_state)
    
    if feature_set == 'rppg_only':
        n_features = 23
        features = [f'feat_{i}' for i in range(n_features)]
    elif feature_set == 'visual_only':
        n_features = 39
        features = [f'feat_{i}' for i in range(n_features)]
    elif feature_set == 'fused':
        n_features = 62
        features = [f'feat_{i}' for i in range(n_features)]
    else:
        n_features = 23
        features = [f'feat_{i}' for i in range(n_features)]
    
    X = np.random.randn(n_samples, len(features)).astype(np.float32)
    y = np.random.binomial(1, 1 - np.random.random(), n_samples).astype(np.int64)
    # Ensure class balance
    n_pos = int(n_samples * class_balance)
    y[:n_pos] = 1
    y[n_pos:] = 0
    np.random.shuffle(y)
    
    return features, X, y


def create_test_video_frames(
    n_frames: int = 30,
    frame_size: tuple = (224, 224),
    add_face: bool = True
) -> List[np.ndarray]:
    """Create synthetic video frames for testing."""
    frames = []
    for i in range(n_frames):
        frame = np.random.randint(0, 255, (*frame_size, 3), dtype=np.uint8)
        if add_face:
            # Add a simple face-like region
            h, w = frame_size
            cv2.rectangle(frame, (w//3, h//3), (2*w//3, 2*h//3), (200, 150, 100), -1)
        frames.append(frame)
    return frames


# Test data paths
TEST_DATA_DIR = Path(__file__).parent
SAMPLE_VIDEO_PATH = None  # Set to actual test video path if available