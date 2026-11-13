"""Run provenance: commit hash, environment, seeds, and configs stored with every result."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

WORKING_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = WORKING_ROOT.parent


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, timeout=30
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _jsonable(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return _jsonable(dataclasses.asdict(obj))
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "item") and callable(obj.item):  # numpy scalars
        return obj.item()
    return obj if isinstance(obj, (str, int, float, bool, type(None))) else repr(obj)


def _package_versions() -> dict[str, str | None]:
    from importlib import metadata

    names = ("numpy", "scikit-learn", "torch", "pennylane", "pandas", "opencv-python", "mediapipe")
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def collect_provenance(
    *,
    seeds: list[int] | None = None,
    configs: dict[str, Any] | None = None,
    files: dict[str, str | Path] | None = None,
) -> dict[str, Any]:
    """Describe how a result was produced.

    ``configs`` may hold dataclasses or dicts; ``files`` maps a label to an input file whose
    sha256 is recorded (for example the feature table).
    """
    status = _git("status", "--porcelain")
    record: dict[str, Any] = {
        "created": datetime.now().astimezone().isoformat(timespec="seconds"),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(status) if status is not None else None,
        "git_dirty_files": status.splitlines() if status else [],
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": _package_versions(),
        "seeds": list(seeds) if seeds is not None else None,
        "configs": _jsonable(configs or {}),
        "input_sha256": {k: sha256_file(v) for k, v in (files or {}).items() if Path(v).is_file()},
    }
    try:
        import torch

        record["cuda"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except ImportError:
        record["cuda"] = None
    return record


def write_json_with_provenance(path: str | Path, result: dict, **provenance_kwargs: Any) -> Path:
    """Write ``result`` as JSON with a ``provenance`` block added (never overwrites that key)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {**result, "provenance": result.get("provenance") or collect_provenance(**provenance_kwargs)}
    path.write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")
    return path
