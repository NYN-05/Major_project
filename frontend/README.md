# Frontend - Web UI for Deepfake KYC Verification

Web UI for the deepfake-verification project (rPPG + hybrid quantum-classical
decision layer). Upload a KYC video, watch the three-stage pipeline run live,
and read the verdict with its evidence dossier (accepted frames, physiological
features, quantum probabilities and plots, the pulse waveform of the analyzed
video).

The UI drives the **existing** Python pipeline untouched — the backend spawns
`WORKING/run_pipeline.py` as a subprocess. No existing project code is modified.

## Stack

- **Frontend**: React + Vite (dev server on `http://localhost:5173`)
- **Backend**: `server.py` — pure Python stdlib (`http.server`), no pip installs.
  Serves upload, progress (SSE), results, and artifact files from `WORKING/output/`.

## Layout

```
frontend/
├── server.py          # stdlib API backend (upload -> subprocess -> SSE -> artifacts)
├── dump_signal.py     # Reconstructs rPPG waveform from stage-1 frames for UI visualization
├── src/
│   ├── App.jsx        # Main app: verdict rig, progress rail, evidence cards
│   ├── api.js         # Thin fetch/EventSource client
│   ├── styles.css     # Instrument-panel design system
│   ├── components/    # React components (VerdictGauge, SignalCanvas, FrameStrip, etc.)
│   └── hooks.js       # Custom hooks (useJob, useSSE, etc.)
├── package.json
├── vite.config.js
└── README.md
```

## Run It

### Terminal 1 — Backend (must use the Python env with torch/pennylane/mediapipe)

```bash
cd frontend
python server.py          # Listens on http://127.0.0.1:8000 (port via FRONTEND_PORT env)
```

### Terminal 2 — Frontend

```bash
cd frontend
npm install
npm run dev               # http://localhost:5173 (proxies /api to backend)
```

Then drop a video onto the upload bay. First run of the rPPG stage may
download the MediaPipe face-landmarker model (internet required once).

## Production Build

```bash
npm run build             # Outputs to dist/
npm run preview           # Serves built app (still expects /api on :8000)
```

## API Contract (`server.py`)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/detect` | POST | Upload video (raw body + `X-Filename` header) → `{job: <id>}` |
| `/api/jobs/<id>` | GET | Job status: `{done, error, video, signal, lines[-400:], result}` |
| `/api/jobs/<id>/events` | GET | SSE stream: `line` / `stage` / `result` / `signal` / `error` |
| `/api/previous` | GET | Last canonical pipeline result (with `_signal`) |
| `/api/health` | GET | `{ok, platform, running, has_previous, sequences, artifacts}` |
| `/api/artifacts?dir=` | GET | List files under `output/<rel>` |
| `/api/files?rel=` | GET | Serve artifact file (path traversal protected) |

### Upload Limits & Validation

- **Max size**: 200 MB
- **Magic-byte validation**: MP4/MOV `ftyp`, AVI `RIFF`, WebM `EBML`
- **Concurrency cap**: Max 2 simultaneous jobs (returns 429 on overflow)
- **Job TTL**: 1 hour; frame sequences: 24 hours
- **Sanitized filenames**: `{job8}_{stem}.ext` for thumbnail consistency
- **CORS**: Restricted to localhost origins (`localhost:5173`, `localhost:8000`)
- **Hard timeout**: 30 minutes per pipeline run (worker killed)
- **Upload streaming**: 64 KB chunks (no 200 MB in-memory buffer)

## Frontend State Machine

```
idle
  -> selected (preview + metadata + Start button)
  -> running (7-stage pipeline + live panel + creeping progress + continuous sheen)
  -> done (pipeline persists with 100% bar + result strip;
           dashboard: Verdict radial gauge, Insights 6 metrics,
           Signal canvas, Quantum flow, FileInfo, FrameSamples)
  -> error (inline banner)
```

**Theme toggle** persists `rppgqc.theme` in localStorage; respects `prefers-reduced-motion`.

## Pipeline Stages (as shown in UI)

1. **Upload** - Video received, validated (magic-byte check, 200 MB limit)
2. **Frames** - Frame sampling at 30 fps + YOLO face detection + quality gates
3. **rPPG** - MediaPipe ROIs → POS/CHROM pulse → 10 physiological features
4. **Quantum** - QAOA feature selection (10 → 3) → Hybrid VQC → P(real)
5. **Verdict** - Decision bins: REAL (≥0.7), FAKE (≤0.3), UNCERTAIN
6. **Artifacts** - Result JSON, signal waveform, frame thumbnails, plots

## Artifacts Served

The UI reads artifacts from `WORKING/output/` via `/api/files` and `/api/artifacts`:

| Source | Artifacts |
|--------|-----------|
| Stage 1 (frames) | `output/frames/frame_sequences/<job>/frames/*.jpg`, `frame_metadata.jsonl` |
| Stage 2 (rPPG) | `output/rppg/plots/`, `dataset_features.csv` |
| Stage 3 (quantum) | `output/quantum/plots/`, `metrics_quantum.json`, `hybrid_vqc.pt` |
| Pipeline | `output/pipeline/pipeline_result.json` |

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `FRONTEND_PORT` | Backend API port | `8000` |
| `FRONTEMD_PORT` | Deprecated alias for `FRONTEND_PORT` | `8000` |

## Notes

- Videos are stored temporarily in `WORKING/output/frontend_inbox/`
- Each job runs the pipeline at 30 fps sampling; verdicts take 1–4 minutes
  depending on video length
- A run's evidence (frames, plots, waveform) is readable from the artifact
  endpoints at any time — `/api/health` lists what exists
- No JS errors expected; console 404s = missing frame thumbnails for stale runs (gitignored)
- The UI drives the **existing** Python pipeline untouched — the backend spawns
  `WORKING/run_pipeline.py` as a subprocess. No existing project code is modified.

## Verified E2E Flow

- `idle` → `selected` → `running` → `done` with signal canvas, frame thumbnails (5 frames), theme toggle, sequential rerun
- Invalid file → 415 friendly error
- Responsive: 1920→375px no overflow