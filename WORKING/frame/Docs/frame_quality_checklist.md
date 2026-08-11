# Frame Quality Checklist

## Rule Definitions
- Blur rule: reject when Laplacian variance < 8.0.
- Brightness dark rule: reject when gray mean < 45.0.
- Brightness overexposed rule: reject when gray mean > 220.0.
- Face visibility rule: reject when no detectable face is present.
- Face size rule: reject when detected face is too small for reliable physiological signal.
- Extreme pose rule: reject when detected face touches frame edge margin or has unusual face-box aspect ratio.
- Composite quality score: w1*Blur + w2*Brightness + w3*FaceConfidence.

## Rejection Categories
- blur
- dark_frame
- overexposed_frame
- no_face
- face_too_small
- extreme_pose

## Reporting Checklist
- Mark example accepted frames.
- Mark example rejected frames for each category.
- Verify quality flags are exported into metadata.