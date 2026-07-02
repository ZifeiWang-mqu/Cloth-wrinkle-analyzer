"""API tests for POST /api/inspect-hand.

The fallback path (no MediaPipe / no hand found) is the CI default — these
tests must pass WITHOUT mediapipe installed. Detection-success paths are
exercised by monkeypatching ``detect_hands``.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.schemas import CATEGORY_HAND, HandIssueType
from app.services.hand_detection import DetectedHand, HandDetection
from tests.test_hand_rules import natural_landmarks, zigzag_landmarks


def _png_bytes() -> bytes:
    img = np.full((300, 300, 3), 235, dtype=np.uint8)
    for y in range(60, 260, 20):
        cv2.line(img, (30, y), (270, y), (60, 60, 60), 2)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_inspect_hand_fallback_never_crashes(client):
    """Without MediaPipe (CI default) the endpoint returns an informational
    hand_detection_failed issue instead of erroring."""
    r = client.post(
        "/api/inspect-hand",
        files={"image": ("test.png", _png_bytes(), "image/png")},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["result"] in {"ok", "needs_review"}
    types = [i["type"] for i in body["issues"]]
    assert HandIssueType.hand_detection_failed.value in types
    failed = body["issues"][0]
    assert failed["category"] == CATEGORY_HAND
    assert failed["flagged"] is False
    assert body["debug"]["hand"]["detected"] is False
    assert any(n.startswith("mode: hand") for n in body["debug"]["notes"])
    # no feature diagnostics when nothing was detected
    assert "features" not in body["debug"]["hand"]


def test_inspect_hand_with_region_hint(client):
    region = '{"x": 40, "y": 40, "w": 150, "h": 150}'
    r = client.post(
        "/api/inspect-hand",
        files={"image": ("test.png", _png_bytes(), "image/png")},
        data={"selected_region": region},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["debug"]["garment_region_used"] is True
    # fallback issue bbox reflects the selected region
    if body["issues"] and body["issues"][0]["type"] == "hand_detection_failed":
        assert body["issues"][0]["bbox"]["w"] == 150


def test_inspect_hand_rejects_bad_upload(client):
    r = client.post(
        "/api/inspect-hand",
        files={"image": ("notes.txt", b"not an image", "text/plain")},
    )
    assert r.status_code == 400


def _fake_detection(points) -> HandDetection:
    return HandDetection(
        detected=True,
        backend="mediapipe",
        hands=[
            DetectedHand(
                landmarks=points,
                handedness="Right",
                confidence=0.9,
                bbox={"x": 60.0, "y": 80.0, "w": 180.0, "h": 200.0},
            )
        ],
    )


def test_inspect_hand_detected_natural(client, monkeypatch):
    monkeypatch.setattr(
        "app.services.hand_inspection_service.detect_hands",
        lambda image, region=None, **kw: _fake_detection(natural_landmarks()),
    )
    r = client.post(
        "/api/inspect-hand",
        files={"image": ("test.png", _png_bytes(), "image/png")},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["result"] == "ok"
    assert body["issues"] == []
    assert body["debug"]["hand"]["detected"] is True

    # Stage 4a diagnostics: feature snapshot present and rounded
    features = body["debug"]["hand"]["features"]
    assert len(features) == 1
    f = features[0]
    assert f["degenerate"] is False
    assert set(f["length_ratios"]) == {"thumb", "index", "middle", "ring", "pinky"}
    assert 0.5 < f["length_ratios"]["pinky"] < 1.0
    assert "min_interior_deg" in f and "wrist_palm_ratio" in f
    assert len(f["adjacent_pairs"]) == 3

    # per-rule explanations: all five rules, all zero, each with a reason
    details = body["debug"]["hand"]["rule_details"]
    assert len(details) == 1 and details[0]["hand"] == 0
    rules = {r_["rule"]: r_ for r_ in details[0]["rules"]}
    assert len(rules) == 5
    for r_ in rules.values():
        assert r_["score"] == 0.0
        assert r_["fires_when"]
        assert r_["reason"]
    assert rules["finger_length_anomaly"]["reason"] == "values within natural bands"


def test_inspect_hand_landmark_overlays_flag(client, monkeypatch):
    monkeypatch.setattr(
        "app.services.hand_inspection_service.detect_hands",
        lambda image, region=None, **kw: _fake_detection(natural_landmarks()),
    )
    # flag off (default) -> no overlays
    r = client.post(
        "/api/inspect-hand",
        files={"image": ("test.png", _png_bytes(), "image/png")},
    )
    assert r.json()["debug"]["overlays"] is None
    # flag on -> 21 landmark points for the one hand
    r = client.post(
        "/api/inspect-hand",
        files={"image": ("test.png", _png_bytes(), "image/png")},
        data={"return_debug_overlays": "true"},
    )
    overlays = r.json()["debug"]["overlays"]
    assert overlays is not None
    assert len(overlays["hand_landmarks"]) == 1
    assert len(overlays["hand_landmarks"][0]) == 21


def test_inspect_hand_detected_anomalous(client, monkeypatch):
    monkeypatch.setattr(
        "app.services.hand_inspection_service.detect_hands",
        lambda image, region=None, **kw: _fake_detection(zigzag_landmarks()),
    )
    r = client.post(
        "/api/inspect-hand",
        files={"image": ("test.png", _png_bytes(), "image/png")},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["result"] == "needs_review"
    flagged = [i for i in body["issues"] if i["flagged"]]
    assert flagged and flagged[0]["type"] == HandIssueType.joint_angle_anomaly.value
    assert all(i["category"] == CATEGORY_HAND for i in body["issues"])
    assert flagged[0]["evidence_boxes"]

    # diagnostics explain the fired rule
    rules = {r_["rule"]: r_ for r_ in body["debug"]["hand"]["rule_details"][0]["rules"]}
    joint = rules[HandIssueType.joint_angle_anomaly.value]
    assert joint["score"] >= 0.5
    assert "zigzag" in joint["detail"]
    assert joint["reason"].startswith("anomaly condition met")

    # persisted row is retrievable via the existing inspection-detail endpoint
    detail = client.get(f"/api/inspection/{body['inspection_id']}")
    assert detail.status_code == 200, detail.text
    assert detail.json()["garment_type"] == "hand"


def test_wrinkle_endpoint_still_works(client):
    r = client.post(
        "/api/inspect-wrinkle",
        files={"image": ("test.png", _png_bytes(), "image/png")},
        data={"garment_type": "skirt"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    # wrinkle issues (if any) default to category "wrinkle"
    assert all(i.get("category", "wrinkle") == "wrinkle" for i in body["issues"])
