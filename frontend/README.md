# frontemd

Web UI for the deepfake-verification project (rPPG + hybrid quantum-classical
decision layer). Upload a KYC video, watch the three-stage pipeline run live,
and read the verdict with its evidence dossier (accepted frames, physiological
features, quantum probabilities and plots, the pulse waveform of the analyzed
video).

The UI drives the **existing** Python pipeline untouched — the backend spawns
`WORKING/run_pipeline.py` as a subprocess. No existing project code is modified.

## Stack

- React + Vite (dev server on `http://localhost:5173`)
- Backend: `server.py` — pure Python stdlib (`http.server`), no pip installs.
  Serves upload, progress (SSE), results, and artifact files from `WORKING/output/`.

## Run it

Terminal 1 — backend (must use the Python that has torch/pennylane/mediapipe):

```
python server.py          # listens on http://127.0.0.1:8000
```

Terminal 2 — UI:

```
npm install
npm run dev               # http://localhost:5173
```

Then drop a video onto the upload bay. First run of the rPPG stage may
download the MediaPipe face-landmarker model (internet required once).

## Production build

```
npm run build
npm run preview           # serves the built app (still expects /api on :8000)
```

## Layout

```
frontemd/
  server.py          stdlib API backend (upload → subprocess → SSE → artifacts)
  dump_signal.py     reconstructs the pulse waveform from stage-1 frames
  src/App.jsx        verdict rig, progress rail, evidence cards
  src/api.js         thin fetch/EventSource client
  src/styles.css     instrument-panel design system
```

## Notes

- Videos are stored temporarily in `WORKING/output/frontend_inbox/`.
- Each job runs the pipeline at 10 fps sampling; verdicts take 1–3 minutes
  depending on video length.
- A run's evidence (frames, plots, waveform) is readable from the artifact
  endpoints at any time — `/api/health` lists what exists.
