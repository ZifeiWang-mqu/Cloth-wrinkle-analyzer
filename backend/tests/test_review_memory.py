"""Tests for the review-memory foundation (Phase 1-3).

Order inside this module matters: the empty-memory test runs before any review
is saved (the session-scoped test DB is shared, and no other module saves
reviews). No MediaPipe or external APIs are required anywhere.
"""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.db.database import SessionLocal
from app.db.models import ReviewFeedback, ReviewMemory
from app.main import app
from app.services.embeddings import LocalHashEmbedder, cosine_similarity
from tests.test_hand_rules import natural_landmarks
from tests.test_inspect_hand import _fake_detection

_IDS: dict[str, str] = {}  # populated across ordered tests


def _png_bytes() -> bytes:
    img = np.full((250, 220, 3), 235, dtype=np.uint8)
    for y in range(40, 220, 18):
        cv2.line(img, (20, y), (200, y), (60, 60, 60), 2)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


# --------------------------------------------------------------------------- #
# Embedder units (pure)
# --------------------------------------------------------------------------- #
def test_embedder_deterministic_and_normalized():
    e = LocalHashEmbedder()
    v1 = e.embed("hand false negative extra digit")
    v2 = e.embed("hand false negative extra digit")
    assert v1 == v2 and len(v1) == 256
    assert abs(sum(x * x for x in v1) - 1.0) < 1e-3  # L2-normalized
    assert cosine_similarity(v1, v2) == pytest.approx(1.0, abs=1e-6)


def test_embedder_similarity_ordering():
    e = LocalHashEmbedder()
    q = e.embed("hand detector confidence landmark rules zero")
    similar = e.embed("hand landmark rules returned zero, detector confidence high")
    different = e.embed("skirt gravity wrinkle density garment fabric")
    assert cosine_similarity(q, similar) > cosine_similarity(q, different)


# --------------------------------------------------------------------------- #
# API — errors and empty state (BEFORE any review exists)
# --------------------------------------------------------------------------- #
def test_search_empty_memory_returns_empty(client):
    r = client.post(
        "/api/review-memory/search", json={"query_text": "hand detector zero rules"}
    )
    assert r.status_code == 200, r.text
    assert r.json()["cases"] == []


def test_review_unknown_inspection_404(client):
    r = client.post(
        "/api/review", json={"inspection_id": "nope", "user_verdict": "correct"}
    )
    assert r.status_code == 404


def test_search_unknown_inspection_404(client):
    r = client.post("/api/review-memory/search", json={"inspection_id": "nope"})
    assert r.status_code == 404


def test_search_requires_query_or_id(client):
    r = client.post("/api/review-memory/search", json={})
    assert r.status_code == 400


# --------------------------------------------------------------------------- #
# Save reviews (hand + wrinkle)
# --------------------------------------------------------------------------- #
def test_save_review_hand_false_negative(client, monkeypatch):
    monkeypatch.setattr(
        "app.services.hand_inspection_service.detect_hands",
        lambda image, region=None, **kw: _fake_detection(natural_landmarks()),
    )
    ins = client.post(
        "/api/inspect-hand",
        files={"image": ("t.png", _png_bytes(), "image/png")},
    ).json()
    _IDS["hand_inspection"] = ins["inspection_id"]

    r = client.post(
        "/api/review",
        json={
            "inspection_id": ins["inspection_id"],
            "user_verdict": "false_negative",
            "corrected_issue_type": "extra_digit_like_shape",
            "user_comment": "looks like six fingers",
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "saved" and body["mode"] == "hand"
    _IDS["hand_review"] = body["review_id"]

    # Summary carries the key fields
    s = body["summary_text"]
    assert "Hand false_negative" in s
    assert "extra_digit_like_shape" in s
    assert "0.900" in s  # detector confidence from the fake detection
    assert "All landmark rules returned 0.0." in s
    assert "landmark-normalization failure" in s
    assert 'User comment: "looks like six fingers"' in s

    # DB rows: snapshot copied server-side; memory has a real embedding
    db = SessionLocal()
    try:
        review = db.get(ReviewFeedback, body["review_id"])
        assert review is not None and review.mode == "hand"
        assert review.app_result == ins["result"]
        assert json.loads(review.scores_json)  # hand rule scores snapshotted
        assert json.loads(review.debug_snapshot_json)["hand"]["detected"] is True
        assert review.image_path_ref is None  # include_image_crop defaulted False

        memory = db.get(ReviewMemory, body["memory_id"])
        assert memory is not None and memory.feedback_id == review.id
        assert memory.issue_type == "extra_digit_like_shape"
        vec = json.loads(memory.embedding_json)
        assert len(vec) == 256 and any(v != 0 for v in vec)
    finally:
        db.close()


def test_save_review_wrinkle(client):
    ins = client.post(
        "/api/inspect-wrinkle",
        files={"image": ("t.png", _png_bytes(), "image/png")},
        data={"garment_type": "skirt"},
    ).json()
    _IDS["wrinkle_inspection"] = ins["inspection_id"]

    r = client.post(
        "/api/review",
        json={"inspection_id": ins["inspection_id"], "user_verdict": "correct"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["mode"] == "wrinkle"
    _IDS["wrinkle_review"] = body["review_id"]
    assert "Wrinkle correct." in body["summary_text"]
    assert "Garment type: skirt." in body["summary_text"]
    assert "App result:" in body["summary_text"]


# --------------------------------------------------------------------------- #
# Retrieval
# --------------------------------------------------------------------------- #
def test_search_by_inspection_id_returns_saved_case_first(client):
    r = client.post(
        "/api/review-memory/search",
        json={"inspection_id": _IDS["hand_inspection"], "top_k": 5},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["query_summary"].startswith("Hand inspection.")
    cases = body["cases"]
    assert cases, "expected at least the saved hand case"
    # mode auto-inferred from the inspection -> only hand cases returned
    assert all(c["mode"] == "hand" for c in cases)
    top = cases[0]
    assert top["feedback_id"] == _IDS["hand_review"]
    assert top["verdict"] == "false_negative"
    assert top["issue_type"] == "extra_digit_like_shape"
    assert 0.0 < top["similarity"] <= 1.0


def test_review_only_label_accepted_and_in_summary(client):
    """Detection-failed hand inspection reviewed with a review-only label."""
    # No MediaPipe in CI -> detection fails; a region hint is provided.
    ins = client.post(
        "/api/inspect-hand",
        files={"image": ("t.png", _png_bytes(), "image/png")},
        data={"selected_region": '{"x": 20, "y": 20, "w": 120, "h": 120}'},
    ).json()

    r = client.post(
        "/api/review",
        json={
            "inspection_id": ins["inspection_id"],
            "user_verdict": "false_negative",
            "corrected_issue_type": "missing_finger",
            "user_comment": "the hand has 4 fingers",
        },
    )
    assert r.status_code == 200, r.text
    s = r.json()["summary_text"]
    assert "Hand false_negative: user reported missing_finger." in s
    assert "The detector found no hand despite a user-selected region" in s
    # merged phrasing replaces the two disjointed sentences
    assert "A user-selected region hint was used." not in s


def test_unknown_correction_label_rejected(client):
    r = client.post(
        "/api/review",
        json={
            "inspection_id": _IDS["hand_inspection"],
            "user_verdict": "false_negative",
            "corrected_issue_type": "totally_made_up_label",
        },
    )
    assert r.status_code == 422
    assert "corrected_issue_type" in r.text


def test_detector_types_still_valid_corrections(client):
    r = client.post(
        "/api/review",
        json={
            "inspection_id": _IDS["hand_inspection"],
            "user_verdict": "false_positive",
            "corrected_issue_type": "joint_angle_anomaly",
        },
    )
    assert r.status_code == 200, r.text


def test_mode_filter(client):
    r = client.post(
        "/api/review-memory/search",
        json={"query_text": "inspection result", "mode": "wrinkle"},
    )
    cases = r.json()["cases"]
    assert cases and all(c["mode"] == "wrinkle" for c in cases)


def test_verdict_filter(client):
    r = client.post(
        "/api/review-memory/search",
        json={"query_text": "hand", "verdict": "false_negative"},
    )
    cases = r.json()["cases"]
    assert cases and all(c["verdict"] == "false_negative" for c in cases)

    r = client.post(
        "/api/review-memory/search",
        json={"query_text": "hand", "verdict": "unclear"},
    )
    assert r.json()["cases"] == []