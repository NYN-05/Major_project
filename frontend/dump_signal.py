"""
dump_signal.py
==============
Reconstruct the pulse waveform for a previously analyzed video by
re-running rPPG over the stage-1 accepted frames (the same frames the
pipeline already produced). Emits a compact decimated JSON for the UI's
verdict-rig waveform. Best-effort: prints a payload with signal=None on
failure.

Usage:
    python dump_signal.py --frames-dir <dir> [--metadata <jsonl>] [--fps 10.0] --out <out.json>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "WORKING" / "RPPG"))

from rppg import RPPGPipeline  # noqa: E402

MAX_POINTS = 600


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames-dir", required=True)
    parser.add_argument("--metadata", default=None)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--method", default="POS", choices=["POS", "CHROM"])
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    payload: dict = {"fps": args.fps, "n": 0, "signal": None, "error": None}
    try:
        pipeline = RPPGPipeline(method=args.method)
        result = pipeline.process_frames(args.frames_dir, metadata_path=args.metadata, fps=args.fps)
        payload["fps"] = float(result.fps)
        payload["n"] = int(result.n_frames_usable)
        if result.combined_signal is None:
            payload["error"] = "no signal (insufficient usable frames)"
        else:
            sig = np.asarray(result.combined_signal, dtype=np.float64)
            if len(sig) > MAX_POINTS:
                step = int(np.ceil(len(sig) / MAX_POINTS))
                sig = sig[::step]
            payload["signal"] = [
                None if np.isnan(float(s)) else round(float(s), 6) for s in sig
            ]
    except Exception as exc:  # noqa: BLE001 - best-effort dump
        payload["error"] = f"{type(exc).__name__}: {exc}"

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload), encoding="utf-8")
    print("signal dump done", flush=True)


if __name__ == "__main__":
    main()