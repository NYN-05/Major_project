from pathlib import Path

import cv2
import numpy as np

out_path = Path(__file__).resolve().parent / 'test_video.avi'
out_path.parent.mkdir(parents=True, exist_ok=True)
w, h = 320, 240
fourcc = cv2.VideoWriter_fourcc(*'XVID')
vw = cv2.VideoWriter(str(out_path), fourcc, 20.0, (w, h))
for i in range(120):
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    # draw a simple face-like pattern with slight motion
    cx = w // 2 + int(5 * np.sin(i * 0.2))
    cy = h // 2
    cv2.circle(frame, (cx, cy), 60, (255, 220, 200), -1)
    cv2.circle(frame, (cx - 20, cy - 10), 8, (0, 0, 0), -1)
    cv2.circle(frame, (cx + 20, cy - 10), 8, (0, 0, 0), -1)
    cv2.ellipse(frame, (cx, cy + 10), (20, 10), 0, 0, 180, (0, 0, 0), 2)
    vw.write(frame)
vw.release()
print('WROTE', out_path)
