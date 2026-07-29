# Tool Run Instructions

Use the Python interpreter that already has OpenCV and NumPy available in this workspace:

```powershell
$python = "C:\Users\JHASHANK\AppData\Local\Programs\Python\Python310\python.exe"
```

Run the commands from `D:\Implementation` unless a different output path is shown.

## `video_frame_extractor.py`

Purpose: extract video frames into `frames/<split>/<video_name>/` and write extraction reports.

Run frame extraction:

```powershell
& $python "tools\video_frame_extractor.py" extract `
  --input-root "d:/Implementation/celebdf_split" `
  --output-root "d:/Implementation/frames"
```

Run the FPS benchmark mode:

```powershell
& $python "tools\video_frame_extractor.py" benchmark `
  --input-root "d:/Implementation/celebdf_split" `
  --rates 1 5 10
```

Manual FPS control:

- Edit `DEFAULT_TARGET_FPS` near the top of the file to change the default FPS.
- Or pass `--target-fps <value>` on the command line.

## `dataset_audit.py`

Purpose: scan the raw dataset and generate inventory, metadata, and structure reports.

Run the audit:

```powershell
& $python "tools\dataset_audit.py"
```

The script auto-detects the dataset root from the current directory or the workspace root and writes reports to `dataset_audit_reports/`.

## `face_roi_pipeline.py`

Purpose: detect, stabilize, align, and crop faces from already extracted frames.

Run one sample video:

```powershell
& $python "tools\face_roi_pipeline.py" run `
  --input-root "d:/Implementation/frames" `
  --output-root "C:/Users/JHASHANK/AppData/Local/Temp/face_roi_validation" `
  --videos "test/id10_id7_0006"
```

Run the full extracted-frame set:

```powershell
& $python "tools\face_roi_pipeline.py" run `
  --input-root "d:/Implementation/frames" `
  --output-root "C:/Users/JHASHANK/AppData/Local/Temp/face_roi_validation"
```

Notes:

- `--output-root` must stay outside the input root.
- Use a drive with free space for outputs; this workspace's D: drive can fill quickly.
- Outputs are written under `annotations/`, `faces/`, `debug/`, and `faces/reports/` inside the chosen output root.

## `_probe_face_pipeline.py`

Purpose: temporary diagnostic script used to validate the face pipeline module.

Run it only for debugging:

```powershell
& $python "tools\_probe_face_pipeline.py"
```

This script loads `face_roi_pipeline.py` and runs a hardcoded sample, so it is not the normal entrypoint for face processing.