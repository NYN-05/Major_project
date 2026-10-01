"""Integrity helpers for model and preprocessing artifacts."""

import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_sha256(path: Path, digest_path: Path | None = None) -> str:
    """Verify an artifact against its adjacent checksum sidecar."""
    digest_path = digest_path or path.with_suffix(path.suffix + ".sha256")
    if not path.is_file():
        raise FileNotFoundError(f"Artifact not found: {path}")
    if not digest_path.is_file():
        raise RuntimeError(
            f"Refusing to load unsigned artifact {path}; missing checksum {digest_path}"
        )
    expected = digest_path.read_text(encoding="ascii").strip().split()[0].lower()
    actual = sha256_file(path)
    if expected != actual:
        raise RuntimeError(
            f"Artifact checksum mismatch for {path}: expected {expected}, got {actual}"
        )
    return actual


def write_sha256(path: Path) -> Path:
    digest_path = path.with_suffix(path.suffix + ".sha256")
    digest_path.write_text(sha256_file(path) + "\n", encoding="ascii")
    return digest_path


def schema_digest(feature_names) -> str:
    payload = json.dumps(list(feature_names), separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
