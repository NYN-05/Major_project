"""
run_on_video.py
=================
Example usage of the rPPG pipeline.

Usage:
    python examples/run_on_video.py path/to/kyc_video.mp4

Prints the extracted physiological feature vector and saves a plot
of the cleaned pulse waveform + power spectrum to
'rppg_output.png' for visual sanity-checking.
"""

import argparse
import os
import sys

import numpy as np

# Allow running directly from the examples/ folder without installing
# the package.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rppg import RPPGPipeline

OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "output", "rppg",
)


def main():
    parser = argparse.ArgumentParser(description="Run rPPG feature extraction on a video.")
    parser.add_argument("video_path", help="Path to a KYC video file (mp4, avi, etc.)")
    parser.add_argument("--method", default="POS", choices=["POS", "CHROM"],
                         help="rPPG reconstruction method (default: POS)")
    parser.add_argument("--fps", type=float, default=None,
                         help="Force a target sampling fps (default: use video's native fps)")
    parser.add_argument("--plot", action="store_true", help="Save a diagnostic plot")
    parser.add_argument("--out", default=os.path.join(OUTPUT_DIR, "rppg_output.png"), help="Path for diagnostic plot")
    args = parser.parse_args()

    pipeline = RPPGPipeline(method=args.method, target_fps=args.fps)
    result = pipeline.process_video(args.video_path)

    print(f"\nVideo: {args.video_path}")
    print(f"FPS used: {result.fps:.2f}")
    print(f"Frames total / usable: {result.n_frames_total} / {result.n_frames_usable}")

    if result.warnings:
        print("\nWarnings:")
        for w in result.warnings:
            print(f"  - {w}")

    if result.features is None:
        print("\nNo feature vector produced (insufficient usable frames).")
        return

    print("\nPhysiological feature vector:")
    for name, value in result.features.to_dict().items():
        print(f"  {name:32s}: {value:.4f}")

    vector = result.to_feature_vector()
    print(f"\nRaw vector (for classifier input): {np.array2string(vector, precision=4)}")

    if args.plot:
        _save_diagnostic_plot(result, args.out)
        print(f"\nDiagnostic plot saved to: {args.out}")


def _save_diagnostic_plot(result, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import signal as sp_signal

    fig, axes = plt.subplots(2, 1, figsize=(10, 6))

    t = np.arange(len(result.combined_signal)) / result.fps
    axes[0].plot(t, result.combined_signal, color="crimson")
    axes[0].set_title("Cleaned rPPG Pulse Signal")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Amplitude (z-scored)")

    freqs, psd = sp_signal.welch(result.combined_signal, fs=result.fps,
                                  nperseg=min(len(result.combined_signal), int(result.fps * 8)))
    mask = (freqs >= 0.5) & (freqs <= 5.0)
    axes[1].plot(freqs[mask] * 60, psd[mask], color="navy")
    axes[1].axvline(result.features.heart_rate_bpm, color="crimson", linestyle="--",
                     label=f"Estimated HR: {result.features.heart_rate_bpm:.1f} BPM")
    axes[1].set_title("Power Spectral Density")
    axes[1].set_xlabel("Heart Rate (BPM)")
    axes[1].set_ylabel("Power")
    axes[1].legend()

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150)


if __name__ == "__main__":
    main()
