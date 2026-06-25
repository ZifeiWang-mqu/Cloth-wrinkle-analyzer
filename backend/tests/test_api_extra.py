"""Tests for the improvement-phase endpoints: base64, model/status, inspection,
and richer feedback (missed_issue)."""

from __future__ import annotations

import base64

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app


def _png_bytes() -> bytes:
    img = np.full((220, 180, 3), 235, dtype=np.uint8)
    for y in range(30, 200, 16):
        cv2.line(img, (15, y), (165, y), (50, 50, 50), 2)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_model_status(client):
    r = client.get("/api/model/status")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["model_type"] in {"dummy", "mahalanobis"}
    assert isinstance(body["available_garment_models"], list)
    assert body["thresholds_loaded"] is True  # shipped config exists
    assert "version" in body


def test_inspect_base64_matches_shape(client):
    data_url = "data:image/png;base64," + base64.b64encode(_png_bytes()).decode()
    r = client.post(
        "/api/inspect-base64",
        json={"image_base64": data_url, "garment_type": "skirt", "source": "photoshop"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["inspection_id"]
    assert body["result"] in {"ok", "needs_review"}
    assert "anomaly_model" in body["debug"]["scores"]
    # New per-issue fields are present when issues exist.
    for issue in body["issues"]:
        assert {"score", "threshold", "flagged", "evidence_boxes"} <= set(issue)
        assert isinstance(issue["evidence_boxes"], list)
        assert len(issue["evidence_boxes"]) >= 1  # precise or fallback


def test_inspect_base64_rejects_bad_data(client):
    r = client.post("/api/inspect-base64", json={"image_base64": "not-base64!!"})
    assert r.status_code == 400


def test_get_inspection_roundtrip(client):
    raw = base64.b64encode(_png_bytes()).decode()
    ins = client.post("/api/inspect-base64", json={"image_base64": raw}).json()
    iid = ins["inspection_id"]

    r = client.get(f"/api/inspection/{iid}")
    assert r.status_code == 200, r.text
    detail = r.json()
    assert detail["inspection_id"] == iid
    assert "issues" in detail
    assert detail["result"] in {"ok", "needs_review"}


def test_get_inspection_404(client):
    assert client.get("/api/inspection/nope").status_code == 404


def test_missed_issue_feedback(client):
    raw = base64.b64encode(_png_bytes()).decode()
    ins = client.post("/api/inspect-base64", json={"image_base64": raw}).json()

    fb = client.post(
        "/api/feedback",
        json={
            "inspection_id": ins["inspection_id"],
            "feedback": "missed_issue",
            "issue_type": "joint_inconsistency",
            "garment_type": "shirt",
            "image_id": "eval_001.png",
            "corrected_bbox": {"x": 10, "y": 20, "w": 40, "h": 30},
            "source": "web",
            "comment": "肘の皺が不足",
        },
    )
    assert fb.status_code == 200, fb.text
    assert fb.json()["status"] == "saved"

    # The feedback should come back in the inspection detail.
    detail = client.get(f"/api/inspection/{ins['inspection_id']}").json()
    assert any(f["feedback"] == "missed_issue" for f in detail["feedback"])


# --- Improvement phase 4 (SAM / pose / illustration model) ------------------ #
def test_model_status_has_capability_fields(client):
    body = client.get("/api/model/status").json()
    assert "sam_available" in body
    assert "mediapipe_available" in body
    assert "illustration_feedback_model" in body
    assert "ready" in body["illustration_feedback_model"]


def test_capabilities_endpoint(client):
    r = client.get("/api/debug/capabilities")
    assert r.status_code == 200, r.text
    caps = r.json()
    assert caps["segmentation"]["opencv_available"] is True
    assert "mediapipe_available" in caps["pose"]


def test_inspect_base64_with_segmentation(client):
    data_url = "data:image/png;base64," + base64.b64encode(_png_bytes()).decode()
    r = client.post(
        "/api/inspect-base64",
        json={
            "image_base64": data_url,
            "use_segmentation": True,
            "return_debug_overlays": True,
        },
    )
    assert r.status_code == 200, r.text
    debug = r.json()["debug"]
    assert debug["segmentation"] is not None
    assert debug["segmentation"]["enabled"] is True
    assert "final_score" in debug["model_scores"]


def test_model_reload(client):
    r = client.post("/api/model/reload")
    assert r.status_code == 200
    assert r.json()["status"] == "reloaded"
