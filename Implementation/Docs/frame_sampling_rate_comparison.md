# Frame Sampling Rate Comparison

Project-standard baseline sampling rate: 10.0 FPS
Minimum sequence length target for temporal modeling: 64 frames

| Video | Sampling FPS | Est. Frames | Interval (ms) | Compute Cost vs Baseline | Signal Fidelity Proxy | Sufficient (>= min length) |
|---|---:|---:|---:|---:|---|---|
| test.mp4 | 5.0 | 63 | 200.00 | 0.50x | Lower temporal fidelity | No |
| test.mp4 | 10.0 | 127 | 100.00 | 1.00x | Baseline temporal fidelity | Yes |
| test.mp4 | 15.0 | 190 | 66.67 | 1.50x | Higher temporal fidelity | Yes |

Baseline decision:
- Selected 10.0 FPS as the default project setting for baseline experiments.
- Lower rates reduce compute but can weaken temporal signal fidelity.
- Higher rates preserve temporal detail but increase storage and processing cost.
- Accuracy impact must be validated in downstream classifier experiments.