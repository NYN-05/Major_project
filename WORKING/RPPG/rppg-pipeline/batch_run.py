"""
batch_run.py

Run the rPPG pipeline on all videos in archive (1)/video and save
feature vectors and diagnostic plots.
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rppg import RPPGPipeline

RPPG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(os.path.dirname(RPPG_ROOT), 'output', 'rppg')

VIDEO_DIR = os.path.join(RPPG_ROOT, 'archive (1)', 'video')
OUT_CSV = os.path.join(OUTPUT_DIR, 'batch_results.csv')


def save_diagnostic_plot(result, out_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy import signal as sp_signal

    fig, axes = plt.subplots(2, 1, figsize=(10, 6))

    t = np.arange(len(result.combined_signal)) / result.fps
    axes[0].plot(t, result.combined_signal, color='crimson')
    axes[0].set_title('Cleaned rPPG Pulse Signal')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Amplitude (z-scored)')

    freqs, psd = sp_signal.welch(result.combined_signal, fs=result.fps,
                                  nperseg=min(len(result.combined_signal), int(result.fps * 8)))
    mask = (freqs >= 0.5) & (freqs <= 5.0)
    axes[1].plot(freqs[mask] * 60, psd[mask], color='navy')
    if result.features is not None:
        axes[1].axvline(result.features.heart_rate_bpm, color='crimson', linestyle='--',
                         label=f"Estimated HR: {result.features.heart_rate_bpm:.1f} BPM")
        axes[1].legend()
    axes[1].set_title('Power Spectral Density')
    axes[1].set_xlabel('Heart Rate (BPM)')
    axes[1].set_ylabel('Power')

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    files = [f for f in os.listdir(VIDEO_DIR) if f.lower().endswith(('.mp4', '.mov', '.avi', '.mkv'))]
    files.sort()
    rows = []

    pipeline = RPPGPipeline(method='POS')

    for fname in files:
        path = os.path.join(VIDEO_DIR, fname)
        print('Processing', path)
        try:
            res = pipeline.process_video(path)
        except Exception as e:
            print('Error processing', fname, e)
            rows.append({'file': fname, 'status': 'error', 'error': str(e)})
            continue

        row = {'file': fname, 'status': 'ok' if res.features is not None else 'no_features',
               'n_frames_total': res.n_frames_total, 'n_frames_usable': res.n_frames_usable}
        if res.features is not None:
            row.update(res.features.to_dict())
            plot_path = os.path.join(OUTPUT_DIR, f'rppg_{os.path.splitext(fname)[0]}.png')
            try:
                save_diagnostic_plot(res, plot_path)
                row['plot'] = os.path.basename(plot_path)
            except Exception as e:
                row['plot'] = None
                print('Could not save plot for', fname, e)
        else:
            row['plot'] = None
            row['warnings'] = ';'.join(res.warnings)

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print('Batch complete. Results saved to', OUT_CSV)


if __name__ == '__main__':
    main()
