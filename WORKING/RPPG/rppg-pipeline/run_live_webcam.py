"""
run_live_webcam.py
===================
Live webcam demonstration of the rPPG pipeline.
Continuously captures video, maintains a rolling buffer of frames,
and updates the estimated Heart Rate (BPM) in real-time.
"""

import os
import sys
import time
import collections

import cv2
import numpy as np

# Allow running directly from the examples/ folder without installing the package.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rppg.face_roi import FaceROIExtractor
from rppg.preprocessing import clean_signal
from rppg.signal_extraction import extract_pulse_signal, combine_roi_signals
from rppg.features import compute_features

def draw_rois(frame, rois):
    # Draw convex hulls of ROIs for visual feedback
    if rois.left_cheek is not None:
        cv2.polylines(frame, [rois.left_cheek], isClosed=True, color=(0, 255, 0), thickness=2)
    if rois.right_cheek is not None:
        cv2.polylines(frame, [rois.right_cheek], isClosed=True, color=(0, 255, 0), thickness=2)
    if rois.forehead is not None:
        cv2.polylines(frame, [rois.forehead], isClosed=True, color=(255, 0, 0), thickness=2)

def main():
    print("Initializing Live rPPG Webcam Feed...")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    # Try to set higher resolution/FPS if supported
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # Rolling window parameters
    WINDOW_SECONDS = 10.0 # Maintain 10 seconds of data for rPPG
    MAX_FRAMES = 300      # Upper limit buffer size
    MIN_USABLE_FRAMES = 60 # Minimum frames before we attempt HR calculation

    # Deques for storing the traces
    timestamps = collections.deque(maxlen=MAX_FRAMES)
    left_trace = collections.deque(maxlen=MAX_FRAMES)
    right_trace = collections.deque(maxlen=MAX_FRAMES)
    forehead_trace = collections.deque(maxlen=MAX_FRAMES)

    frame_idx = 0
    current_hr = None
    current_snr = None

    print("Press 'q' in the video window to quit.")
    
    with FaceROIExtractor() as extractor:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame.")
                break
            
            t_now = time.time()
            
            # Detect face and extract ROIs
            face = extractor.detect(frame, frame_idx)
            
            display_frame = frame.copy()
            
            if face.found:
                rois = extractor.extract_rois(frame, face)
                
                l_val = extractor.mean_rgb(frame, rois.left_cheek)
                r_val = extractor.mean_rgb(frame, rois.right_cheek)
                f_val = extractor.mean_rgb(frame, rois.forehead)
                
                # Only add if we got valid RGB values (not None)
                left_trace.append(l_val if l_val is not None else np.full(3, np.nan))
                right_trace.append(r_val if r_val is not None else np.full(3, np.nan))
                forehead_trace.append(f_val if f_val is not None else np.full(3, np.nan))
                timestamps.append(t_now)
                
                draw_rois(display_frame, rois)
            else:
                # Add NaNs if no face is found
                left_trace.append(np.full(3, np.nan))
                right_trace.append(np.full(3, np.nan))
                forehead_trace.append(np.full(3, np.nan))
                timestamps.append(t_now)

            # Keep deque strictly within the rolling window based on time
            while len(timestamps) > 0 and (t_now - timestamps[0]) > WINDOW_SECONDS:
                timestamps.popleft()
                left_trace.popleft()
                right_trace.popleft()
                forehead_trace.popleft()

            # Calculate HR if we have enough frames
            if len(timestamps) >= MIN_USABLE_FRAMES:
                # Calculate real FPS of the rolling window
                elapsed_time = timestamps[-1] - timestamps[0]
                if elapsed_time > 0:
                    real_fps = (len(timestamps) - 1) / elapsed_time
                    
                    # Convert traces to arrays
                    left_arr = np.array(left_trace)
                    right_arr = np.array(right_trace)
                    forehead_arr = np.array(forehead_trace)
                    
                    def process_roi(arr):
                        # Filter out missing frames (NaNs)
                        valid_ratio = 1.0 - np.isnan(arr).any(axis=1).mean()
                        if valid_ratio < 0.3:
                            return None
                        
                        # Interpolate small gaps
                        filled = arr.copy()
                        for c in range(3):
                            col = filled[:, c]
                            nans = np.isnan(col)
                            if nans.any() and not nans.all():
                                idx = np.arange(len(col))
                                col[nans] = np.interp(idx[nans], idx[~nans], col[~nans])
                                filled[:, c] = col
                        
                        if np.isnan(filled).any():
                            return None
                            
                        # Use POS method
                        return extract_pulse_signal(filled, fs=real_fps, method="POS")
                    
                    l_sig = process_roi(left_arr)
                    r_sig = process_roi(right_arr)
                    f_sig = process_roi(forehead_arr)
                    
                    if l_sig is not None or r_sig is not None or f_sig is not None:
                        try:
                            # Combine signals
                            combined_raw = combine_roi_signals(
                                [l_sig, r_sig, f_sig], 
                                weights=(0.35, 0.35, 0.30)
                            )
                            # Clean signal
                            combined_clean = clean_signal(combined_raw, fs=real_fps, low_hz=0.7, high_hz=4.0)
                            
                            # Clean per ROI (optional for features, but required by compute_features signature)
                            l_clean = clean_signal(l_sig, fs=real_fps, low_hz=0.7, high_hz=4.0) if l_sig is not None else None
                            r_clean = clean_signal(r_sig, fs=real_fps, low_hz=0.7, high_hz=4.0) if r_sig is not None else None
                            f_clean = clean_signal(f_sig, fs=real_fps, low_hz=0.7, high_hz=4.0) if f_sig is not None else None
                            
                            feats = compute_features(
                                combined_signal=combined_clean,
                                fs=real_fps,
                                left_cheek_signal=l_clean,
                                right_cheek_signal=r_clean,
                                forehead_signal=f_clean,
                                low_hz=0.7,
                                high_hz=4.0
                            )
                            
                            if not np.isnan(feats.heart_rate_bpm):
                                current_hr = feats.heart_rate_bpm
                                current_snr = feats.snr_db
                        except Exception as e:
                            # If signal extraction fails for some reason, just skip updating this frame
                            pass

            # Overlay info
            if current_hr is not None:
                text = f"HR: {current_hr:.1f} BPM"
                snr_text = f"SNR: {current_snr:.1f} dB" if current_snr is not None else ""
                
                # Draw text with outline for better visibility
                cv2.putText(display_frame, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(display_frame, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2, cv2.LINE_AA)
                
                cv2.putText(display_frame, snr_text, (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(display_frame, snr_text, (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 1, cv2.LINE_AA)
            else:
                msg = "Buffering..." if face.found else "No face detected"
                cv2.putText(display_frame, msg, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

            # Calculate and show FPS
            if len(timestamps) > 1:
                inst_fps = 1.0 / (timestamps[-1] - timestamps[-2])
                cv2.putText(display_frame, f"FPS: {inst_fps:.1f}", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

            cv2.imshow("Live rPPG", display_frame)

            frame_idx += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
