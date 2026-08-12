"""
server.py
=========
Minimal stdlib-only backend for the frontemd React UI.

Endpoints
---------
POST /api/detect                    upload video (raw body) -> {job}
GET  /api/jobs/<id>                 {done, error, lines, result}
GET  /api/jobs/<id>/events          SSE: line / stage / result / error events
GET  /api/health                    {ok, artifacts...}
GET  /api/previous                  last canonical pipeline result, if any
GET  /api/artifacts?dir=<rel>       list files under output/<rel>
GET  /api/files?rel=output/...      serve an artifact file

The heavy lifting is delegated to the existing stack unchanged: each job
spawns `python run_pipeline.py` inside WORKING/ as a subprocess. No
existing project code is modified.
"""

from __future__ import annotations

import json
import mimetypes
import os
import queue
import re
import subprocess
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

HERE = Path(__file__).resolve().parent
WORKING = HERE.parent / "WORKING"
OUTPUT_ROOT = WORKING / "output"
INBOX = OUTPUT_ROOT / "frontend_inbox"
RESULTS_DIR = OUTPUT_ROOT / "pipeline"
CANONICAL_RESULT = RESULTS_DIR / "pipeline_result.json"

PORT = int(os.environ.get("FRONTEMD_PORT", "8000"))
JOB_TTL_SECONDS = 60 * 60

STAGE_TAGS = ("[1/3]", "[2/3]", "[3/3]")

JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()


def _trim_jobs() -> None:
    cutoff = time.time() - JOB_TTL_SECONDS
    with JOBS_LOCK:
        for job_id in [j for j, job in JOBS.items() if job["created"] < cutoff]:
            JOBS.pop(job_id, None)


def _mime(path: Path) -> str:
    return mimetypes.guess_type(path.name)[0] or "application/octet-stream"


def _run_job(job: dict, video_path: Path) -> None:
    run_pipeline = WORKING / "run_pipeline.py"
    out_json = RESULTS_DIR / f"pipeline_result_{job['id']}.json"
    cmd = [
        sys.executable,
        str(run_pipeline),
        "--source",
        str(video_path),
        "--out",
        str(out_json),
    ]
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(WORKING),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.rstrip("\r\n")
            if not line:
                continue
            job["lines"].append(line)
            job["queue"].put(("line", line))
            if any(tag in line for tag in STAGE_TAGS):
                job["queue"].put(("stage", line))
        proc.wait()
        if not out_json.exists():
            raise RuntimeError("pipeline produced no result JSON")
        result = json.loads(out_json.read_text(encoding="utf-8"))
        result["_signal"] = _dump_signal(job, video_path.name)
        job["result"] = result
        job["video"] = video_path.name
        job["done"] = True
        job["queue"].put(("result", result))
    except Exception as exc:  # noqa: BLE001 - surface to the UI
        job["error"] = str(exc)
        job["done"] = True
        job["queue"].put(("error", str(exc)))


def _dump_signal(job: dict, video_name: str) -> str | None:
    """Reuse the stage-1 accepted frames to reconstruct the real pulse
    waveform for the verdict rig (best-effort; ~30s extra)."""
    stem = Path(video_name).stem
    frames_dir = OUTPUT_ROOT / "frames" / "frame_sequences" / stem / "frames"
    metadata = OUTPUT_ROOT / "frames" / "frame_sequences" / stem / "frame_metadata.jsonl"
    if not frames_dir.is_dir():
        return None
    sig_file = RESULTS_DIR / f"signal_{job['id']}.json"
    cmd = [
        sys.executable,
        str(HERE / "dump_signal.py"),
        "--frames-dir",
        str(frames_dir),
        "--metadata",
        str(metadata),
        "--out",
        str(sig_file),
    ]
    try:
        subprocess.run(cmd, cwd=str(WORKING), timeout=300, capture_output=True, text=True)
    except (subprocess.TimeoutExpired, OSError):
        return None
    if sig_file.exists():
        return f"pipeline/{sig_file.name}"
    return None


class Handler(BaseHTTPRequestHandler):
    server_version = "frontemd/0.1"

    # -- helpers ---------------------------------------------------------

    def _cors(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _json(self, payload, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self._cors()
        self.end_headers()

    # -- routes ----------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)
        try:
            if path == "/api/health":
                self._health()
            elif path == "/api/previous":
                self._previous()
            elif path == "/api/artifacts":
                self._artifacts(query.get("dir", [""])[0])
            elif path == "/api/files":
                self._files(query.get("rel", [""])[0])
            elif (m := re.fullmatch(r"/api/jobs/([0-9a-f-]+)/events", path)):
                self._sse_stream(m.group(1))
            elif (m := re.fullmatch(r"/api/jobs/([0-9a-f-]+)", path)):
                self._job(m.group(1))
            else:
                self._json({"error": "not found"}, 404)
        except (BrokenPipeError, ConnectionResetError):
            return

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path != "/api/detect":
            self._json({"error": "not found"}, 404)
            return
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            self._json({"error": "empty upload"}, 400)
            return
        body = self.rfile.read(length)

        _trim_jobs()
        job_id = uuid.uuid4().hex
        INBOX.mkdir(parents=True, exist_ok=True)
        video_path = INBOX / f"{job_id}.mp4"
        video_path.write_bytes(body)

        job = {
            "id": job_id,
            "created": time.time(),
            "lines": [],
            "queue": queue.Queue(),
            "done": False,
            "error": None,
            "result": None,
            "video": None,
        }
        with JOBS_LOCK:
            JOBS[job_id] = job
        threading.Thread(target=_run_job, args=(job, video_path), daemon=True).start()
        self._json({"job": job_id})

    # -- endpoint implementations ----------------------------------------

    def _health(self) -> None:
        quantum = OUTPUT_ROOT / "quantum"
        frames_root = OUTPUT_ROOT / "frames" / "frame_sequences"
        sequences = sorted(
            p.name for p in frames_root.iterdir() if p.is_dir() and (p / "frames").is_dir()
        ) if frames_root.is_dir() else []
        self._json(
            {
                "ok": True,
                "platform": sys.platform,
                "running": len(JOBS),
                "has_previous": CANONICAL_RESULT.exists(),
                "sequences": sequences,
                "artifacts": {
                    "quantum_plots": sorted(
                        p.name for p in quantum.iterdir() if p.suffix.lower() in {".png", ".svg"}
                    ) if quantum.is_dir() else [],
                    "rppg_png": (OUTPUT_ROOT / "rppg" / "rppg_output.png").exists(),
                    "crosscheck_model": (OUTPUT_ROOT / "rppg" / "rppg_classifier.pkl").exists(),
                },
            }
        )

    def _previous(self) -> None:
        if not CANONICAL_RESULT.exists():
            self._json({"result": None})
            return
        result = json.loads(CANONICAL_RESULT.read_text(encoding="utf-8"))
        signal = None
        for sig in RESULTS_DIR.glob("signal_*.json"):
            signal = f"pipeline/{sig.name}"
            break
        result["_signal"] = signal
        self._json({"result": result})

    def _job(self, job_id: str) -> None:
        job = JOBS.get(job_id)
        if job is None:
            self._json({"error": "unknown job"}, 404)
            return
        self._json(
            {
                "done": job["done"],
                "error": job["error"],
                "video": job["video"],
                "lines": job["lines"][-400:],
                "result": job["result"],
            }
        )

    def _artifacts(self, rel_dir: str) -> None:
        root = OUTPUT_ROOT
        target = (root / rel_dir).resolve()
        if not target.is_relative_to(root.resolve()) or not target.is_dir():
            self._json({"error": "invalid dir"}, 404)
            return
        files = []
        for p in sorted(target.rglob("*")):
            if p.is_file():
                files.append(
                    {
                        "name": p.name,
                        "rel": str(p.relative_to(root)).replace("\\", "/"),
                        "size": p.stat().st_size,
                    }
                )
        self._json({"dir": rel_dir, "files": files})

    def _files(self, rel: str) -> None:
        root = OUTPUT_ROOT
        target = (root / rel).resolve()
        if not target.is_relative_to(root.resolve()) or not target.is_file():
            self._json({"error": "not found"}, 404)
            return
        body = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", _mime(target))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def _sse(self, kind: str, payload) -> None:
        data = json.dumps({"kind": kind, "data": payload})
        try:
            self.wfile.write(f"data: {data}\n\n".encode("utf-8"))
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            raise

    def _sse_stream(self, job_id: str) -> None:
        job = JOBS.get(job_id)
        if job is None:
            self._json({"error": "unknown job"}, 404)
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self._cors()
        self.end_headers()
        q = job["queue"]
        try:
            while True:
                try:
                    kind, payload = q.get(timeout=20)
                except queue.Empty:
                    self._sse("ping", {"t": time.time()})
                    continue
                self._sse(kind, payload)
                if kind in ("result", "error"):
                    return
        except (BrokenPipeError, ConnectionResetError):
            return

    def log_message(self, fmt: str, *args) -> None:  # noqa: A003 - stdlib signature
        if fmt.startswith("GET /api") or fmt.startswith("POST /api"):
            pass  # keep console quiet about API chatter
        else:
            super().log_message(fmt, *args)


def main() -> None:
    INBOX.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    httpd = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"[frontemd] backend on http://127.0.0.1:{PORT}  (WORKING={WORKING})")
    print("[frontemd] start the UI with: npm run dev   (http://localhost:5173)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[frontemd] stopped")
        httpd.server_close()


if __name__ == "__main__":
    main()