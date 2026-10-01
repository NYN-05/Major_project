"""Versioned frontend/backend payload validation."""

from __future__ import annotations

API_VERSION = "1"
PIPELINE_VERSION = "p2-1"
VERDICTS = {"REAL", "FAKE", "UNCERTAIN", "INCONCLUSIVE", "REVIEW REQUIRED"}


def validate_result_payload(payload: dict) -> dict:
    if not isinstance(payload, dict):
        raise ValueError("pipeline result must be an object")
    verdict = payload.get("verdict")
    if not isinstance(verdict, dict) or not isinstance(verdict.get("label"), str):
        raise ValueError("pipeline result is missing verdict.label")
    if verdict["label"] not in VERDICTS:
        raise ValueError(f"unsupported verdict label: {verdict['label']}")
    confidence = verdict.get("confidence")
    if confidence is not None and (not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1):
        raise ValueError("verdict.confidence must be null or within [0, 1]")
    payload.setdefault("api_version", API_VERSION)
    payload.setdefault("pipeline_version", PIPELINE_VERSION)
    return payload
