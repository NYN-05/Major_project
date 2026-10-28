#!/usr/bin/env python
"""
setup_progress.py
=================
Streaming console-UI wrapper for the long-running training stages
(rPPG extraction, visual extraction). Used by setup_and_run.ps1.

Spawns a child command with unbuffered stdout, tees every raw line to a log
file, filters known benign framework noise, parses machine-readable progress
lines emitted by the wrapped stage, and renders a live single-line progress
bar with throughput and ETA. Non-progress lines of interest are shown as dim
event lines below the bar. Always exits with the child's exit code.

Usage:
    python setup_progress.py --log LOGFILE [--label LABEL] [--total N] -- CHILD [ARGS ...]

Machine-line contract (emitted by the wrapped stage on stdout):
    TOTAL samples=<int> pending=<int>
    PROGRESS received=<int> total=<int> ok=<int> err=<int> nofeat=<int> gated=<int> elapsed=<float>
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from typing import Dict, Optional

# Framework noise filtered from the console (still written to the log file).
BENIGN_MARKERS = (
    "face_landmarker_graph.cc",
    "inference_feedback_manager.cc",
    "Created TensorFlow Lite XNNPACK delegate",
    "Sets FaceBlendshapesGraph acceleration",
    "Feedback manager requires a model",
)
# Lines relayed to the log only: per-video chatter superseded by the live bar.
LOG_ONLY_PREFIXES = ("Processing ", "    ... ")
# tqdm-style download bars (e.g. torch model weights) stay in the log too;
# their constant redraws would fight the live bar in a TTY.
import re as _re
LOG_ONLY_PATTERN = _re.compile(r"^\s*\d+%\|")

TOTAL_PREFIX = "TOTAL "
PROGRESS_PREFIX = "PROGRESS "

ANSI = {
    "reset": "\x1b[0m",
    "bold": "\x1b[1m",
    "dim": "\x1b[2m",
    "gray": "\x1b[90m",
    "red": "\x1b[91m",
    "green": "\x1b[92m",
    "yellow": "\x1b[93m",
    "cyan": "\x1b[96m",
}

USE_COLOR = False

# Progress bar block characters.
BAR_FILL = "\u2588"   # solid block
BAR_EMPTY = "\u2591"  # light shade


def _c(key: str) -> str:
    return ANSI[key] if USE_COLOR else ""


def _enable_windows_vt() -> None:
    """Best-effort enablement of ANSI escape processing on Windows consoles."""
    if os.name != "nt":
        return
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        for std_handle in (-11, -12):  # STD_OUTPUT_HANDLE, STD_ERROR_HANDLE
            handle = kernel32.GetStdHandle(std_handle)
            mode = ctypes.c_uint32()
            if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
                # ENABLE_VIRTUAL_TERMINAL_PROCESSING
                kernel32.SetConsoleMode(handle, mode.value | 0x0004)
    except Exception:
        pass


def _parse_kv(line: str) -> Optional[Dict[str, str]]:
    """Parse 'PREFIX k=v k=v ...' into a dict, tolerating malformed lines."""
    try:
        return dict(kv.split("=", 1) for kv in line.split(" ")[1:] if "=" in kv)
    except ValueError:
        return None


def _fmt_duration(seconds: float) -> str:
    s = max(0, int(round(seconds)))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{sec:02d}s"
    if m:
        return f"{m}m{sec:02d}s"
    return f"{sec}s"


def _metrics(state: Dict) -> Dict:
    """Compute percent, rate and ETA from the current state."""
    total = state["total"]
    received = state["received"]
    pct = (received / total) if total > 0 else 0.0
    elapsed = max(1e-6, time.monotonic() - state["t0"])
    rate = (received / elapsed) if received > 0 else 0.0
    eta = ((total - received) / rate) if rate > 0 and total > received else 0.0
    return {"pct": pct, "rate": rate, "eta": eta, "elapsed": elapsed}


def _render_bar(state: Dict, label: str, width: int) -> str:
    m = _metrics(state)
    total = state["total"] if state["total"] > 0 else None
    pct_txt = f"{m['pct'] * 100:5.1f}%"
    cnt_txt = f"{state['received']}/{total if total is not None else '?'}"
    stat_txt = (
        f"ok {state['ok']} \u00b7 gated {state['gated']} \u00b7 "
        f"nofeat {state['nofeat']} \u00b7 err {state['err']}"
    )
    spd_txt = f"{m['rate']:.2f} vid/s \u00b7 ETA {_fmt_duration(m['eta'])}"

    measure = f"{label} {pct_txt} {cnt_txt}  {stat_txt}  {spd_txt}  []"
    bar_w = max(10, min(60, width - len(measure) - 2))
    filled = int(bar_w * m["pct"])
    err_color = "red" if state["err"] else "gray"

    return (
        f"{_c('bold')}{_c('cyan')}{label}{_c('reset')} "
        f"[{_c('green')}{BAR_FILL * filled}{_c('dim')}{BAR_EMPTY * (bar_w - filled)}{_c('reset')}] "
        f"{_c('bold')}{pct_txt}{_c('reset')} {cnt_txt}  "
        f"{_c('green')}\u2713 {state['ok']}{_c('reset')} \u00b7 "
        f"{_c('yellow')}gated {state['gated']}{_c('reset')} \u00b7 "
        f"{_c('gray')}nofeat {state['nofeat']}{_c('reset')} \u00b7 "
        f"{_c(err_color)}\u2717 {state['err']}{_c('reset')}  "
        f"{_c('dim')}{spd_txt}{_c('reset')}"
    )


def _draw_bar(state: Dict, label: str) -> None:
    if not USE_COLOR:
        return
    width = shutil.get_terminal_size((100, 24)).columns
    sys.stdout.write("\r\x1b[2K" + _render_bar(state, label, max(48, width - 1)))
    sys.stdout.flush()


def _relay(line: str, state: Dict, label: str) -> None:
    """Show an event line below the bar, then redraw the bar."""
    if USE_COLOR:
        sys.stdout.write("\r\x1b[2K")
        print(f"{ANSI['gray']}{line}{ANSI['reset']}", flush=True)
        _draw_bar(state, label)
    else:
        print(line, flush=True)


def _plain_progress(state: Dict, label: str) -> None:
    """Non-TTY fallback: print a plain line at coarse intervals."""
    m = _metrics(state)
    total = state["total"]
    interval = max(1, total // 20) if total > 0 else 25
    if state["received"] == 0 or state["received"] % interval != 0:
        if not (total > 0 and state["received"] == total):
            return
    print(
        f"[{label}] {state['received']}/{total if total > 0 else '?'} "
        f"({m['pct'] * 100:.1f}%) ok {state['ok']} \u00b7 gated {state['gated']} \u00b7 "
        f"nofeat {state['nofeat']} \u00b7 err {state['err']} \u00b7 "
        f"{m['rate']:.2f} vid/s \u00b7 ETA {_fmt_duration(m['eta'])}",
        flush=True,
    )


def _apply_progress(state: Dict, kv: Optional[Dict[str, str]], label: str) -> None:
    if not kv:
        return

    def _int(key: str) -> int:
        try:
            return int(kv.get(key, 0))
        except (TypeError, ValueError):
            return 0

    received = _int("received")
    if kv.get("total", "").isdigit() and int(kv["total"]) > 0:
        state["total"] = int(kv["total"])
    state["received"] = received
    state["ok"] = _int("ok")
    state["err"] = _int("err")
    state["nofeat"] = _int("nofeat")
    state["gated"] = _int("gated")

    if USE_COLOR:
        total = state["total"]
        pct = (received / total) if total > 0 else 0.0
        milestone = int(pct * 10)
        if milestone > state["milestone"] and 0 < milestone < 10:
            state["milestone"] = milestone
            sys.stdout.write("\r\x1b[2K")
            print(
                f"{ANSI['dim']}\u2500\u2500 {pct * 100:.0f}% \u00b7 "
                f"{received}/{total} \u2500\u2500{ANSI['reset']}",
                flush=True,
            )
        _draw_bar(state, label)
    else:
        _plain_progress(state, label)


def _finish(state: Dict, label: str, returncode: int) -> None:
    m = _metrics(state)
    symbol = "\u2713" if returncode == 0 else "\u2717"
    color = "green" if returncode == 0 else "red"
    if USE_COLOR:
        sys.stdout.write("\r\x1b[2K")
    print(
        f"{_c(color)}{symbol} {label}{_c('reset')} \u2014 "
        f"{state['received']} samples in {_fmt_duration(m['elapsed'])} \u00b7 "
        f"{_c('green')}ok {state['ok']}{_c('reset')} \u00b7 "
        f"{_c('yellow')}gated {state['gated']}{_c('reset')} \u00b7 "
        f"{_c('gray')}nofeat {state['nofeat']}{_c('reset')} \u00b7 "
        f"{_c('red' if state['err'] else 'gray')}err {state['err']}{_c('reset')} \u00b7 "
        f"{m['rate']:.2f} vid/s",
        flush=True,
    )


def main() -> int:
    global USE_COLOR

    argv = sys.argv[1:]
    if "--" not in argv:
        print(
            "setup_progress.py: missing '--' separator before the child command",
            file=sys.stderr,
        )
        return 2
    sep = argv.index("--")
    pre, cmd = argv[:sep], argv[sep + 1:]
    if not cmd:
        print("setup_progress.py: no child command after '--'", file=sys.stderr)
        return 2

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[2] if __doc__ else "")
    parser.add_argument("--log", required=True, help="log file receiving every raw child line")
    parser.add_argument("--label", default="", help="short stage label shown in the progress bar")
    parser.add_argument("--total", type=int, default=0, help="fallback total if the stage emits no TOTAL line")
    args = parser.parse_args(pre)

    # Python children need -u so their stdout reaches us unbuffered.
    child = list(cmd)
    if "python" in os.path.basename(child[0]).lower():
        child.insert(1, "-u")

    label = args.label or "progress"
    log_dir = os.path.dirname(os.path.abspath(args.log))
    os.makedirs(log_dir, exist_ok=True)

    USE_COLOR = sys.stdout.isatty() and sys.stderr.isatty() and os.environ.get("NO_COLOR") is None
    if USE_COLOR:
        _enable_windows_vt()

    state: Dict = {
        "total": args.total,
        "received": 0,
        "ok": 0,
        "err": 0,
        "nofeat": 0,
        "gated": 0,
        "milestone": 0,
        "t0": time.monotonic(),
    }

    try:
        proc = subprocess.Popen(
            child,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            bufsize=1,
        )
    except FileNotFoundError as exc:
        print(f"setup_progress.py: cannot start child: {exc}", file=sys.stderr)
        return 127

    with open(args.log, "w", encoding="utf-8", errors="replace", buffering=1) as log_f:
        try:
            assert proc.stdout is not None
            for raw in proc.stdout:
                log_f.write(raw)
                line = raw.rstrip("\r\n")
                if not line.strip():
                    continue
                if any(marker in line for marker in BENIGN_MARKERS):
                    continue
                if line.startswith(TOTAL_PREFIX):
                    kv = _parse_kv(line)
                    if kv and kv.get("pending", "").isdigit() and int(kv["pending"]) > 0:
                        state["total"] = int(kv["pending"])
                        _relay(line, state, label)
                    continue
                if line.startswith(PROGRESS_PREFIX):
                    _apply_progress(state, _parse_kv(line), label)
                    continue
                if line.startswith(LOG_ONLY_PREFIXES) or LOG_ONLY_PATTERN.match(line):
                    continue
                _relay(line, state, label)
            proc.wait()
        except KeyboardInterrupt:
            print(f"\n[{label}] interrupted - terminating child...", flush=True)
            try:
                proc.terminate()
                proc.wait(timeout=10)
            except Exception:
                proc.kill()
            log_f.write(f"\n[interrupted] child terminated after {time.monotonic() - state['t0']:.1f}s\n")
            return 130

    _finish(state, label, proc.returncode or 0)
    return proc.returncode if proc.returncode is not None else 1


if __name__ == "__main__":
    sys.exit(main())
