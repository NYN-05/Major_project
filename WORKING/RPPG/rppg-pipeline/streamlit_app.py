import streamlit as st
import tempfile
import os
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import signal as sp_signal

from rppg import RPPGPipeline


st.set_page_config(page_title="rPPG Pipeline Demo", layout="wide")

st.title("rPPG Pipeline — Demo GUI")

st.sidebar.header("Input video")
uploaded = st.sidebar.file_uploader("Upload a video (mp4/avi)", type=["mp4", "avi"], accept_multiple_files=False)
use_example = st.sidebar.button("Use example test video")

st.sidebar.header("Pipeline options")
method = st.sidebar.selectbox("Reconstruction method", ["POS", "CHROM"], index=0)
fps = st.sidebar.number_input("Target FPS (leave 0 for native)", value=0.0, min_value=0.0, step=1.0)
blur_threshold = st.sidebar.slider("Blur threshold (Laplacian var)", 0.0, 100.0, 15.0)
# set a dynamic default for min frames when user selects a target fps
initial_min = 48 if fps <= 0.0 else max(12, int(fps * 4))
min_frames = st.sidebar.number_input("Min usable frames", value=initial_min, min_value=4, step=1)
apply_mediapipe = st.sidebar.checkbox("Prefer MediaPipe (if installed)", value=False)

run = st.sidebar.button("Run pipeline")

def plot_result(result):
    fig, axes = plt.subplots(2, 1, figsize=(10, 5))
    if result.combined_signal is not None:
        t = np.arange(len(result.combined_signal)) / result.fps
        axes[0].plot(t, result.combined_signal, color="crimson")
        axes[0].set_title("Cleaned rPPG Pulse Signal")
        axes[0].set_xlabel("Time (s)")
    else:
        axes[0].text(0.5, 0.5, "No combined signal", ha="center")

    if result.combined_signal is not None:
        freqs, psd = sp_signal.welch(result.combined_signal, fs=result.fps, nperseg=min(len(result.combined_signal), int(result.fps * 8)))
        mask = (freqs >= 0.5) & (freqs <= 5.0)
        axes[1].plot(freqs[mask] * 60, psd[mask], color="navy")
        axes[1].set_title("Power Spectral Density")
        axes[1].set_xlabel("Heart Rate (BPM)")
    else:
        axes[1].text(0.5, 0.5, "No PSD", ha="center")

    fig.tight_layout()
    return fig


def run_pipeline_on_file(path):
    st.info(f"Processing: {path}")
    pipeline = RPPGPipeline(method=method, target_fps=(None if fps <= 0 else float(fps)), blur_threshold=blur_threshold, min_usable_frames=min_frames)
    result = pipeline.process_video(path)
    return result


video_path = None
if use_example:
    video_path = os.path.join(os.path.dirname(__file__), "test_video.avi")
    if not os.path.exists(video_path):
        st.error("Example video not found. Run rppg-pipeline/_make_test_video.py first.")

if uploaded is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded.name)[1])
    tfile.write(uploaded.read())
    tfile.flush()
    tfile.close()
    video_path = tfile.name

if run:
    if not video_path:
        st.error("Please upload a video or use the example test video.")
    else:
        # check estimated sampled frames for warning/auto-adjust
        try:
            cap = cv2.VideoCapture(video_path)
            native_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            cap.release()
        except Exception:
            native_fps = 25.0
            total_frames = 0

        if fps > 0:
            sample_stride = max(1, round(native_fps / float(fps)))
        else:
            sample_stride = 1

        est_samples = max(0, total_frames // sample_stride) if total_frames > 0 else None
        if est_samples is not None and est_samples < min_frames:
            st.warning(
                f"With target FPS={fps}, estimated sampled frames={est_samples} is less than min usable frames={min_frames}. "
                "Lowering min usable frames so processing can proceed."
            )
            min_frames = max(4, est_samples)

        with st.spinner("Running pipeline — this may take a while..."):
            pipeline = RPPGPipeline(method=method, target_fps=(None if fps <= 0 else float(fps)), blur_threshold=blur_threshold, min_usable_frames=min_frames)
            result = pipeline.process_video(video_path)

        st.subheader("Summary")
        st.write(f"FPS used: {result.fps:.2f}")
        st.write(f"Frames total / usable: {result.n_frames_total} / {result.n_frames_usable}")
        if result.warnings:
            st.warning("\n".join(result.warnings))

        if result.features is not None:
            st.subheader("Physiological feature vector")
            feat = result.features.to_dict()
            st.json(feat)
            
            # --- Integration of Classifier ---
            model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "rppg_classifier.pkl")
            if os.path.exists(model_path):
                st.subheader("Deepfake Detection Result")
                try:
                    import pickle
                    with open(model_path, 'rb') as f:
                        clf = pickle.load(f)
                    
                    vector = result.to_feature_vector().reshape(1, -1)
                    
                    if np.isnan(vector).any():
                        st.warning("Cannot run classifier: Feature vector contains NaN values.")
                    else:
                        pred = clf.predict(vector)[0]
                        proba = clf.predict_proba(vector)[0]
                        
                        if pred == 1:
                            confidence = proba[1] * 100
                            st.error(f"🚨 **DEEPFAKE DETECTED** (Confidence: {confidence:.1f}%)")
                        else:
                            confidence = proba[0] * 100
                            st.success(f"✅ **REAL VIDEO** (Confidence: {confidence:.1f}%)")
                except Exception as e:
                    st.error(f"Could not load or run classifier: {e}")
            else:
                st.info("Train a classifier using `rppg-pipeline/train_classifier.py` to enable deepfake prediction here.")
        else:
            st.info("No features produced (insufficient usable frames)")

        st.subheader("Diagnostic plots")
        fig = plot_result(result)
        st.pyplot(fig)

        # cleanup temp file if uploaded
        if uploaded is not None and video_path and os.path.exists(video_path):
            try:
                os.remove(video_path)
            except Exception:
                pass

st.markdown("---")
st.markdown("Notes: For best results install `mediapipe` and provide `face_landmarker.task`. The app uses an OpenCV fallback when MediaPipe is unavailable.")
