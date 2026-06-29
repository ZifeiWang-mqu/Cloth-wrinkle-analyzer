"""Tests for local evidence-box generation."""

from __future__ import annotations

from app.services.evidence import (
    build_evidence_for_issues,
    clamp_box,
    line_to_evidence_box,
    merge_small_overlapping_boxes,
)
from app.services.feature_extraction import Features
from app.services.pose import PoseResult
from app.services.region_geometry import RegionGeometry
from app.services.wrinkle_edges import WrinkleCandidates, WrinkleLine

REGION = RegionGeometry(bbox={"x": 0, "y": 0, "w": 200, "h": 200})


def test_line_to_box_and_clamp():
    b = line_to_evidence_box(WrinkleLine(10, 100, 70, 102), 0, 0, 8)
    assert b["w"] >= 60  # ~length + 2*pad
    c = clamp_box({"x": -5, "y": -5, "w": 10, "h": 10}, 100, 100)
    assert c["x"] == 0 and c["y"] == 0 and c["w"] >= 1


def test_merge_overlapping():
    a = {"x": 0, "y": 0, "w": 50, "h": 50, "score": 0.5}
    b = {"x": 5, "y": 5, "w": 50, "h": 50, "score": 0.9}
    merged = merge_small_overlapping_boxes([a, b])
    assert len(merged) == 1
    assert merged[0]["score"] == 0.9


def test_gravity_evidence_is_local_not_broad():
    # Near-horizontal lines drive gravity inconsistency.
    lines = [WrinkleLine(20, 100 + i, 80, 101 + i) for i in range(0, 50, 12)]
    cand = WrinkleCandidates(lines=lines, offset=(0, 0), crop_shape=(200, 200))
    issues = [
        {"type": "gravity_inconsistency", "bbox": {"x": 0, "y": 0, "w": 200, "h": 200}, "score": 0.8}
    ]
    build_evidence_for_issues(issues, cand, Features(num_lines=len(lines)), PoseResult(), REGION, (200, 200))
    ev = issues[0]["evidence_boxes"]
    assert len(ev) >= 1
    assert all(not e.get("fallback_broad_bbox") for e in ev)
    # Every evidence box is far smaller than the whole 200x200 region.
    assert all(e["w"] * e["h"] < 0.25 * 200 * 200 for e in ev)


def test_fallback_when_no_lines():
    cand = WrinkleCandidates(lines=[], offset=(0, 0), crop_shape=(200, 200))
    issues = [
        {"type": "shadow_wrinkle_mismatch", "bbox": {"x": 10, "y": 10, "w": 120, "h": 120}, "score": 0.5}
    ]
    build_evidence_for_issues(issues, cand, Features(), PoseResult(), REGION, (200, 200))
    ev = issues[0]["evidence_boxes"]
    assert len(ev) == 1
    assert ev[0]["fallback_broad_bbox"] is True


def test_total_box_cap():
    lines = [WrinkleLine(5 + 30 * i, 10, 25 + 30 * i, 12) for i in range(6)]
    cand = WrinkleCandidates(lines=lines, offset=(0, 0), crop_shape=(400, 400))
    issues = [
        {"type": "gravity_inconsistency", "bbox": {"x": 0, "y": 0, "w": 400, "h": 400}, "score": 0.8},
        {"type": "body_volume_inconsistency", "bbox": {"x": 0, "y": 0, "w": 400, "h": 400}, "score": 0.7},
    ]
    build_evidence_for_issues(issues, cand, Features(num_lines=len(lines)), PoseResult(), RegionGeometry(bbox={"x": 0, "y": 0, "w": 400, "h": 400}), (400, 400))
    total = sum(len(i["evidence_boxes"]) for i in issues)
    assert total <= 24
