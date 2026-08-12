import os
import sys
import cv2
import numpy as np
from rppg.face_roi import FaceROIExtractor

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "output", "rppg")


def main(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print("Cannot open video", path)
        return

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    print(f"Video: {path}\nFrames: {total}, FPS: {fps}")

    counts = []
    blurs = []
    brights = []

    with FaceROIExtractor() as ext:
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            face = ext.detect(frame, idx)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            bright = float(gray.mean())
            blurs.append(blur)
            brights.append(bright)

            rois = ext.extract_rois(frame, face)
            lc = rc = fc = 0
            if rois.left_cheek is not None:
                lc = int(cv2.countNonZero(rois.left_cheek))
            if rois.right_cheek is not None:
                rc = int(cv2.countNonZero(rois.right_cheek))
            if rois.forehead is not None:
                fc = int(cv2.countNonZero(rois.forehead))
            counts.append((lc, rc, fc, face.found))
            idx += 1

    cap.release()

    counts = np.array(counts, dtype=object)
    usable_mask = (np.array([c[3] for c in counts]) & (np.array(blurs) >= 15.0))

    print(f"Total frames processed: {len(counts)}")
    print(f"Frames passing blur>=15: {(np.array(blurs)>=15.0).sum()}")
    print(f"Frames with face found: {(np.array([c[3] for c in counts])).sum()}")

    # show a few sample frames where ROIs are non-empty
    nonzero_idx = [i for i, c in enumerate(counts) if c[0] + c[1] + c[2] > 0]
    print(f"Frames with any ROI pixels: {len(nonzero_idx)} (examples: {nonzero_idx[:5]})")

    if nonzero_idx:
        # save overlay images for up to 3 example frames
        cap = cv2.VideoCapture(path)
        saved = 0
        pick = set(nonzero_idx[:3])
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx in pick:
                face = ext.detect(frame, idx)
                rois = ext.extract_rois(frame, face)
                vis = frame.copy()
                if rois.left_cheek is not None:
                    vis[rois.left_cheek > 0] = (0, 0, 255)
                if rois.right_cheek is not None:
                    vis[rois.right_cheek > 0] = (0, 255, 0)
                if rois.forehead is not None:
                    vis[rois.forehead > 0] = (255, 0, 0)
                outp = os.path.join(OUTPUT_DIR, f"debug_frame_{idx}.png")
                os.makedirs(OUTPUT_DIR, exist_ok=True)
                cv2.imwrite(outp, vis)
                print("Wrote", outp)
                saved += 1
            idx += 1
            if saved >= len(pick):
                break
        cap.release()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python debug_run.py <video_path>")
    else:
        main(sys.argv[1])
