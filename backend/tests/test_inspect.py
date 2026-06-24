"""API-level tests using FastAPI's TestClient.

A temporary data directory is configured via env BEFORE importing the app so
tests never touch the real ``data/`` folder.
"""

from __future__ import annotations

import os
import tempfile

# Must be set before importing app.settings (settings are cached at import).
_TMP = tempfile.mkdtemp(prefix="wrinkle_test_")
os.environ["WRINKLE_DATA_DIR"] = _TMP

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402


def _png_bytes(with_lines: bool = True) -> bytes:
    img = np.full((256, 200, 3), 230, dtype=np.uint8)
    if with_lines:
        # Several horizontal "wrinkles" — should be detectable by the pipeline.
        for y in range(40, 220, 18):
            cv2.line(img, (20, y), (180, y), (60, 60, 60), 2)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_inspect_basic(client):
    r = client.post(
        "/api/inspect-wrinkle",
        files={"image": ("test.png", _png_bytes(), "image/png")},
        data={"garment_type": "skirt"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["inspection_id"]
    assert body["result"] in {"ok", "needs_review"}
    assert 0.0 <= body["overall_score"] <= 1.0
    assert isinstance(body["issues"], list)
    assert "scores" in body["debug"]
    assert "anomaly_model" in body["debug"]["scores"]
    assert len(body["debug"]["scores"]) == 7


def test_inspect_with_region(client):
    region = '{"x": 10, "y": 10, "w": 120, "h": 200}'
    r = client.post(
        "/api/inspect-wrinkle",
        files={"image": ("test.png", _png_bytes(), "image/png")},
        data={"garment_type": "dress", "selected_region": region},
    )
    assert r.status_code == 200, r.text
    assert r.json()["debug"]["garment_region_used"] is True


def test_inspect_rejects_bad_extension(client):
    r = client.post(
        "/api/inspect-wrinkle",
        files={"image": ("notes.txt", b"not an image", "text/plain")},
    )
    assert r.status_code == 400


def test_inspect_bad_region_json(client):
    r = client.post(
        "/api/inspect-wrinkle",
        files={"image": ("test.png", _png_bytes(), "image/png")},
        data={"selected_region": "{not json"},
    )
    assert r.status_code == 400


def test_feedback_roundtrip(client):
    r = client.post(
        "/api/inspect-wrinkle",
        files={"image": ("test.png", _png_bytes(), "image/png")},
        data={"garment_type": "skirt"},
    )
    inspection_id = r.json()["inspection_id"]

    fb = client.post(
        "/api/feedback",
        json={
            "inspection_id": inspection_id,
            "issue_id": None,
            "feedback": "false_positive",
            "comment": "  これは自然な皺です  ",
        },
    )
    assert fb.status_code == 200, fb.text
    assert fb.json()["status"] == "saved"
    assert fb.json()["feedback_id"]


def test_feedback_unknown_inspection(client):
    fb = client.post(
        "/api/feedback",
        json={"inspection_id": "does-not-exist", "feedback": "correct"},
    )
    assert fb.status_code == 404
