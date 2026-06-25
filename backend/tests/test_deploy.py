"""Deployment-focused tests: SAM status/fallback, line filter, capabilities.

These run WITHOUT SAM installed (CI default): SAM is reported unavailable and
the pipeline falls back. The real SAM integration test is gated behind
WRINKLE_RUN_SAM_TESTS=true and a present checkpoint.
"""

from __future__ import annotations

import base64
import os

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.segmentation import GarmentRegion, segment_garment
from app.settings import settings


def _png() -> bytes:
    img = np.full((200, 180, 3), 235, dtype=np.uint8)
    for y in range(30, 180, 16):
        cv2.line(img, (15, y), (165, y), (40, 40, 40), 2)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_status_segmentation_block(client):
    body = client.get("/api/model/status").json()
    seg = body["segmentation"]
    for key in (
        "sam_available",
        "sam_loaded",
        "checkpoint_path",
        "checkpoint_exists",
        "model_type",
        "device",
        "fallback_provider",
        "last_error",
    ):
        assert key in seg, key
    # SAM not installed in CI -> unavailable.
    assert isinstance(seg["sam_available"], bool)


def test_capabilities_doc_shape(client):
    caps = client.get("/api/debug/capabilities").json()
    assert set(["sam", "opencv", "mediapipe"]).issubset(caps.keys())
    assert caps["opencv"]["available"] is True
    assert "loaded" in caps["sam"]
    assert "device" in caps["sam"]


def test_inspect_use_segmentation_true_fallback(client):
    data_url = "data:image/png;base64," + base64.b64encode(_png()).decode()
    r = client.post(
        "/api/inspect-base64",
        json={"image_base64": data_url, "use_segmentation": True, "segmentation_provider": "sam"},
    )
    assert r.status_code == 200, r.text
    seg = r.json()["debug"]["segmentation"]
    assert seg["enabled"] is True
    # SAM unavailable -> geometric fallback, never crashes.
    assert seg["provider"] in {"sam", "opencv"}
    lf = r.json()["debug"]["line_filter"]
    assert "raw_lines" in lf and "kept_lines" in lf


def test_segment_garment_sam_falls_back_without_dependency():
    img = np.full((200, 200, 3), 220, dtype=np.uint8)
    region = GarmentRegion(0, 0, 200, 200, used_selection=False, source="full_image")
    res = segment_garment(img, region, settings, use_segmentation=True, provider_override="sam")
    assert res.enabled is True
    assert res.mask_available is True  # fell back to a geometric mask
    assert res.provider == "opencv"
    assert res.fallback_used is True


def test_line_filter_removes_outside_mask():
    from app.services.wrinkle_edges import extract_wrinkle_candidates

    img = np.full((200, 200, 3), 235, dtype=np.uint8)
    for y in range(20, 180, 14):
        cv2.line(img, (5, y), (195, y), (40, 40, 40), 2)
    region = GarmentRegion(0, 0, 200, 200, used_selection=False, source="full_image")
    mask = np.zeros((200, 200), dtype=np.uint8)
    mask[:, :100] = 255
    cand = extract_wrinkle_candidates(img, region, seg_mask=mask, settings=settings)
    lf = cand.line_filter
    assert lf["raw_lines"] >= lf["kept_lines"]
    assert "removed_outside_mask" in lf


@pytest.mark.skipif(
    os.environ.get("WRINKLE_RUN_SAM_TESTS") != "true",
    reason="SAM integration tests disabled (set WRINKLE_RUN_SAM_TESTS=true)",
)
def test_sam_integration(client):
    data_url = "data:image/png;base64," + base64.b64encode(_png()).decode()
    r = client.post(
        "/api/inspect-base64",
        json={"image_base64": data_url, "use_segmentation": True, "segmentation_provider": "sam"},
    )
    assert r.status_code == 200
    seg = r.json()["debug"]["segmentation"]
    assert seg["mask_available"] is True
