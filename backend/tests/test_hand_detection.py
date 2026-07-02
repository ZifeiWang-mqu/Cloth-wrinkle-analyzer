"""Detector-compatibility tests (MediaPipe Tasks vs legacy vs absent).

All scenarios are simulated by monkeypatching the module-level probes, so
these run deterministically with or without MediaPipe installed.
"""

from __future__ import annotations

import numpy as np

from app.services import hand_detection as hd


def _img() -> np.ndarray:
    return np.full((120, 120, 3), 230, dtype=np.uint8)


class _MpStubNoSolutions:
    """Mimics MediaPipe >= 0.10.x builds without ``mp.solutions``."""

    solutions = None


def test_note_when_mediapipe_missing(monkeypatch):
    monkeypatch.setattr(hd, "_try_import_mediapipe", lambda: None)
    det = hd.detect_hands(_img())
    assert det.detected is False
    assert det.note == "mediapipe_not_installed"
    assert det.to_debug()["note"] == "mediapipe_not_installed"


def test_note_when_no_api_available(monkeypatch):
    """MediaPipe importable but neither Tasks API nor legacy solutions exist."""
    monkeypatch.setattr(hd, "_try_import_mediapipe", lambda: _MpStubNoSolutions())
    monkeypatch.setattr(hd, "_tasks_modules", lambda: None)
    det = hd.detect_hands(_img())
    assert det.detected is False
    assert det.note == "hand_landmarker_unavailable"


def test_note_when_model_file_missing(monkeypatch):
    """Tasks API available (like MediaPipe 0.10.35) but no .task model file."""
    monkeypatch.setattr(hd, "_try_import_mediapipe", lambda: _MpStubNoSolutions())
    monkeypatch.setattr(hd, "_tasks_modules", lambda: (object(), object()))
    monkeypatch.setattr(hd, "_resolve_model", lambda: None)
    det = hd.detect_hands(_img())
    assert det.detected is False
    assert det.note == "hand_model_missing"


def test_model_missing_surfaces_in_api_message(monkeypatch):
    """End-to-end: the API's fallback issue explains the missing model file."""
    import cv2
    from fastapi.testclient import TestClient

    from app.main import app

    monkeypatch.setattr(hd, "_try_import_mediapipe", lambda: _MpStubNoSolutions())
    monkeypatch.setattr(hd, "_tasks_modules", lambda: (object(), object()))
    monkeypatch.setattr(hd, "_resolve_model", lambda: None)

    img = np.full((200, 200, 3), 235, dtype=np.uint8)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    with TestClient(app) as client:
        r = client.post(
            "/api/inspect-hand",
            files={"image": ("t.png", buf.tobytes(), "image/png")},
        )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["debug"]["hand"]["note"] == "hand_model_missing"
    failed = body["issues"][0]
    assert failed["type"] == "hand_detection_failed"
    assert "hand_landmarker.task" in failed["message"]


def test_reset_hand_landmarker_clears_cache():
    hd._LANDMARKER = object()
    hd._LANDMARKER_PATH = "x"
    hd.reset_hand_landmarker()
    assert hd._LANDMARKER is None and hd._LANDMARKER_PATH is None
