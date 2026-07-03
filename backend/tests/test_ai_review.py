"""Tests for the AI visual review endpoint (OpenAI is always mocked).

The single network seam is ``app.services.ai_review._post_responses``;
availability is controlled via ``app.services.ai_review._resolved_key``.
No real API calls are made anywhere.
"""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.db.database import SessionLocal
from app.db.models import Inspection
from app.main import app
from app.services import ai_review
from app.services.ai_review import select_crop
from tests.test_hand_rules import natural_landmarks
from tests.test_inspect_hand import _fake_detection


def _png_bytes() -> bytes:
    img = np.full((300, 260, 3), 235, dtype=np.uint8)
    for y in range(50, 260, 18):
        cv2.line(img, (25, y), (235, y), (60, 60, 60), 2)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


_MODEL_JSON = {
    "summary": "指の本数は5本に見えます。",
    "detected_visual_problems": [
        {"type": "merged_fingers", "description": "薬指と中指が一体化して見えます。", "confidence": 0.6}
    ],
    "detector_comparison": "検出器は異常なしでしたが、輪郭上は結合の疑いがあります。",
    "recommended_action": "薬指と中指の間に境界線を追加してください。",
    "limitations": "この切り抜きだけでは手首の接続は判断できません。",
}


def _canned(obj: dict) -> dict:
    """Responses-API-shaped payload carrying structured output text."""
    return {
        "output": [
            {
                "type": "message",
                "content": [{"type": "output_text", "text": json.dumps(obj, ensure_ascii=False)}],
            }
        ]
    }


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def with_key(monkeypatch):
    monkeypatch.setattr(ai_review, "_resolved_key", lambda: "test-key")


def _make_hand_inspection(client, monkeypatch, region: str | None = None) -> str:
    monkeypatch.setattr(
        "app.services.hand_inspection_service.detect_hands",
        lambda image, region=None, **kw: _fake_detection(natural_landmarks()),
    )
    data = {"selected_region": region} if region else {}
    r = client.post(
        "/api/inspect-hand",
        files={"image": ("t.png", _png_bytes(), "image/png")},
        data=data,
    )
    return r.json()["inspection_id"]


# --------------------------------------------------------------------------- #
# Availability / not-found
# --------------------------------------------------------------------------- #
def test_503_when_key_missing(client, monkeypatch):
    monkeypatch.setattr(ai_review, "_resolved_key", lambda: "")
    r = client.post(
        "/api/inspection/ai-review", json={"inspection_id": "whatever"}
    )
    assert r.status_code == 503
    assert "OPENAI_API_KEY" in r.text


def test_404_unknown_inspection(client, with_key):
    r = client.post("/api/inspection/ai-review", json={"inspection_id": "nope"})
    assert r.status_code == 404


def test_404_missing_image_file(client, with_key, monkeypatch):
    iid = _make_hand_inspection(client, monkeypatch)
    db = SessionLocal()
    try:
        row = db.get(Inspection, iid)
        row.image_path = "/nonexistent/gone.png"
        db.commit()
    finally:
        db.close()
    r = client.post("/api/inspection/ai-review", json={"inspection_id": iid})
    assert r.status_code == 404
    assert "画像ファイル" in r.text


# --------------------------------------------------------------------------- #
# Happy paths (mocked model)
# --------------------------------------------------------------------------- #
def test_happy_path_hand(client, with_key, monkeypatch):
    captured: dict = {}

    def fake_post(payload, api_key, timeout_s):
        captured["payload"] = payload
        captured["api_key"] = api_key
        return _canned(_MODEL_JSON)

    monkeypatch.setattr(ai_review, "_post_responses", fake_post)
    iid = _make_hand_inspection(client, monkeypatch)

    r = client.post(
        "/api/inspection/ai-review", json={"inspection_id": iid, "language": "ja"}
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["summary"] == _MODEL_JSON["summary"]
    assert body["detected_visual_problems"][0]["type"] == "merged_fingers"
    assert body["meta"]["mode"] == "hand"
    assert body["meta"]["language"] == "ja"
    # natural fake hand -> no flagged issues, no region -> full-image crop
    assert body["meta"]["crop"]["source"] == "full_image"

    # outbound payload sanity: structured outputs + image + hand context
    payload = captured["payload"]
    assert captured["api_key"] == "test-key"
    fmt = payload["text"]["format"]
    assert fmt["type"] == "json_schema" and fmt["strict"] is True
    user_content = payload["input"][1]["content"]
    assert any(
        c["type"] == "input_image" and c["image_url"].startswith("data:image/jpeg;base64,")
        for c in user_content
    )
    user_text = next(c["text"] for c in user_content if c["type"] == "input_text")
    assert "finger count" in user_text
    assert '"detected": true' in user_text  # hand context JSON embedded


def test_happy_path_wrinkle(client, with_key, monkeypatch):
    monkeypatch.setattr(ai_review, "_post_responses", lambda *a, **k: _canned(_MODEL_JSON))
    ins = client.post(
        "/api/inspect-wrinkle",
        files={"image": ("t.png", _png_bytes(), "image/png")},
        data={"garment_type": "skirt"},
    ).json()

    r = client.post(
        "/api/inspection/ai-review",
        json={"inspection_id": ins["inspection_id"], "language": "en"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["meta"]["mode"] == "wrinkle"
    assert body["meta"]["crop"]["source"] in {"issue_bbox", "full_image"}


def test_region_override_crop(client, with_key, monkeypatch):
    monkeypatch.setattr(ai_review, "_post_responses", lambda *a, **k: _canned(_MODEL_JSON))
    iid = _make_hand_inspection(client, monkeypatch)
    r = client.post(
        "/api/inspection/ai-review",
        json={"inspection_id": iid, "region": {"x": 20, "y": 30, "w": 100, "h": 120}},
    )
    assert r.status_code == 200, r.text
    crop = r.json()["meta"]["crop"]
    assert crop["source"] == "request_region"
    assert crop["w"] == 100 and crop["h"] == 120


def test_502_on_malformed_model_json(client, with_key, monkeypatch):
    monkeypatch.setattr(
        ai_review,
        "_post_responses",
        lambda *a, **k: {
            "output": [
                {"type": "message", "content": [{"type": "output_text", "text": "not json"}]}
            ]
        },
    )
    iid = _make_hand_inspection(client, monkeypatch)
    r = client.post("/api/inspection/ai-review", json={"inspection_id": iid})
    assert r.status_code == 502
    assert "解析" in r.text


def test_502_on_upstream_error(client, with_key, monkeypatch):
    import httpx

    def boom(*a, **k):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(ai_review, "_post_responses", boom)
    iid = _make_hand_inspection(client, monkeypatch)
    r = client.post("/api/inspection/ai-review", json={"inspection_id": iid})
    assert r.status_code == 502
    assert "OpenAI" in r.text


def test_502_on_401_gives_key_guidance(client, with_key, monkeypatch):
    import httpx

    def unauthorized(*a, **k):
        req = httpx.Request("POST", ai_review.OPENAI_RESPONSES_URL)
        resp = httpx.Response(401, request=req)
        raise httpx.HTTPStatusError("401 Unauthorized", request=req, response=resp)

    monkeypatch.setattr(ai_review, "_post_responses", unauthorized)
    iid = _make_hand_inspection(client, monkeypatch)
    r = client.post("/api/inspection/ai-review", json={"inspection_id": iid})
    assert r.status_code == 502
    assert "OpenAI APIキーが無効" in r.text
    assert "WRINKLE_OPENAI_API_KEY" in r.text
    assert "key_source=" in r.text
    # the actual key value never appears (only a short prefix is allowed)
    assert "test-key-secret" not in r.text


# --------------------------------------------------------------------------- #
# Japanese wording sanitizer
# --------------------------------------------------------------------------- #
def test_ja_sanitizer_replaces_technical_terms():
    from app.services.ai_review import sanitize_ja_review

    parsed = {
        "summary": "これは偽陰性の可能性があります。",
        "detector_comparison": "検出結果は False Positive かもしれません。",
        "recommended_action": "クロップを広げて確認してください。",
        "limitations": "true negative の判断はできません。",
        "detected_visual_problems": [
            {"type": "missing_finger", "description": "偽陽性ではありません。", "confidence": 0.5}
        ],
    }
    sanitize_ja_review(parsed)
    assert "偽陰性" not in parsed["summary"]
    assert "検出では見つけられなかった可能性" in parsed["summary"]
    assert "False Positive" not in parsed["detector_comparison"]  # case-insensitive
    assert "誤って検出された可能性" in parsed["detector_comparison"]
    assert "クロップ" not in parsed["recommended_action"]
    assert "選択範囲" in parsed["recommended_action"]
    assert "true negative" not in parsed["limitations"]
    desc = parsed["detected_visual_problems"][0]
    assert "偽陽性" not in desc["description"]
    # enum value untouched
    assert desc["type"] == "missing_finger"


def test_ja_sanitizer_applied_end_to_end(client, with_key, monkeypatch):
    dirty = dict(_MODEL_JSON)
    dirty["summary"] = "偽陰性の可能性があります（false negative）。"
    monkeypatch.setattr(ai_review, "_post_responses", lambda *a, **k: _canned(dirty))
    iid = _make_hand_inspection(client, monkeypatch)
    r = client.post(
        "/api/inspection/ai-review", json={"inspection_id": iid, "language": "ja"}
    )
    assert r.status_code == 200, r.text
    summary = r.json()["summary"]
    assert "偽陰性" not in summary and "false negative" not in summary
    assert "検出では見つけられなかった可能性" in summary


def test_ja_prompt_contains_wording_instruction(client, with_key, monkeypatch):
    captured: dict = {}

    def fake_post(payload, api_key, timeout_s):
        captured["payload"] = payload
        return _canned(_MODEL_JSON)

    monkeypatch.setattr(ai_review, "_post_responses", fake_post)
    iid = _make_hand_inspection(client, monkeypatch)
    client.post("/api/inspection/ai-review", json={"inspection_id": iid, "language": "ja"})
    system_text = captured["payload"]["input"][0]["content"][0]["text"]
    assert "偽陰性" in system_text and "平易な表現" in system_text


# --------------------------------------------------------------------------- #
# Key resolution / diagnostics (safe — never the full key)
# --------------------------------------------------------------------------- #
def test_resolved_key_is_stripped(monkeypatch):
    from app.settings import settings

    monkeypatch.setattr(settings, "openai_api_key", "  sk-test-abc123\n")
    assert settings.resolved_openai_key == "sk-test-abc123"


def test_key_diagnostics_source_and_prefix(monkeypatch):
    from app.settings import settings

    # WRINKLE_ key wins and shadows a different OPENAI_API_KEY
    monkeypatch.setattr(settings, "openai_api_key", "sk-wrinkle-1234567890")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-other-key")
    diag = settings.openai_key_diagnostics()
    assert diag["key_source"] == "WRINKLE_OPENAI_API_KEY"
    assert diag["key_prefix"] == "sk-wrink"
    assert diag["key_length"] == len("sk-wrinkle-1234567890")
    assert diag["shadows_openai_api_key"] is True
    # full key never present in the diagnostics values
    assert "sk-wrinkle-1234567890" not in str(diag)

    # fallback path
    monkeypatch.setattr(settings, "openai_api_key", "")
    diag = settings.openai_key_diagnostics()
    assert diag["key_source"] == "OPENAI_API_KEY"
    assert diag["shadows_openai_api_key"] is False

    # none
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    diag = settings.openai_key_diagnostics()
    assert diag["key_source"] == "none" and diag["key_prefix"] is None


def test_capabilities_exposes_key_diagnostics_not_key(client, monkeypatch):
    from app.settings import settings

    monkeypatch.setattr(settings, "openai_api_key", "sk-diagtest-abcdefer123456")
    r = client.get("/api/debug/capabilities")
    ai = r.json()["ai_review"]
    assert ai["available"] is True
    assert ai["key_source"] == "WRINKLE_OPENAI_API_KEY"
    assert ai["key_prefix"] == "sk-diagt"
    assert ai["key_length"] == len("sk-diagtest-abcdefer123456")
    assert "sk-diagtest-abcdefer123456" not in r.text


# --------------------------------------------------------------------------- #
# Crop selection (pure)
# --------------------------------------------------------------------------- #
def _img(w=300, h=200) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_crop_priority_request_region_first():
    crop, box = select_crop(
        _img(),
        region_override={"x": 10, "y": 10, "w": 50, "h": 40},
        selected_region={"x": 0, "y": 0, "w": 200, "h": 100},
        issues=[{"flagged": True, "bbox": {"x": 5, "y": 5, "w": 20, "h": 20}}],
    )
    assert box["source"] == "request_region"
    assert crop.shape[:2] == (40, 50)


def test_crop_selected_region_polygon():
    crop, box = select_crop(
        _img(),
        region_override=None,
        selected_region={"points": [[10, 20], [110, 20], [110, 90], [10, 90]]},
        issues=[],
    )
    assert box["source"] == "selected_region"
    assert box["x"] == 10 and box["y"] == 20 and box["w"] == 100 and box["h"] == 70


def test_crop_issue_bbox_union_with_evidence_and_padding():
    issues = [
        {"flagged": False, "bbox": {"x": 0, "y": 0, "w": 10, "h": 10}},
        {
            "flagged": True,
            "bbox": {"x": 100, "y": 50, "w": 40, "h": 40},
            "evidence_boxes": [{"x": 150, "y": 60, "w": 30, "h": 20}],
        },
    ]
    crop, box = select_crop(_img(), None, None, issues)
    assert box["source"] == "issue_bbox"
    # union x-range 100..180 padded by 20% of max span (80*0.2=16) -> starts <100
    assert box["x"] < 100 and box["x"] + box["w"] > 180


def test_crop_full_image_fallback_and_clamping():
    crop, box = select_crop(_img(), None, None, [])
    assert box["source"] == "full_image" and crop.shape[:2] == (200, 300)
    # out-of-bounds override gets clamped, never crashes
    crop, box = select_crop(_img(), {"x": -50, "y": -50, "w": 10_000, "h": 10_000}, None, [])
    assert box["source"] == "request_region"
    assert box["x"] == 0 and box["y"] == 0 and box["w"] <= 300 and box["h"] <= 200