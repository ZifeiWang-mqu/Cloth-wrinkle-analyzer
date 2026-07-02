"""Unit tests for hand features + rules using synthetic landmarks.

No MediaPipe required: we hand-craft 21-point landmark sets (MediaPipe Hands
index order) for a natural hand and several anomalous variants, and assert the
rules fire only where expected.
"""

from __future__ import annotations

from app.schemas import CATEGORY_HAND, HandIssueType
from app.services.hand_detection import DetectedHand
from app.services.hand_features import extract_hand_features
from app.services.hand_rule_engine import (
    integrate_hand_issues,
    score_finger_length_anomaly,
    score_finger_overlap_anomaly,
    score_joint_angle_anomaly,
    score_thumb_position_anomaly,
    score_wrist_connection_anomaly,
)
from app.settings import settings

_BBOX = {"x": 60.0, "y": 80.0, "w": 180.0, "h": 200.0}


def natural_landmarks() -> list[tuple[float, float]]:
    """A plausibly proportioned open hand, fingers up, thumb on index side."""
    return [
        (150, 260),  # 0 wrist
        (112, 232), (95, 210), (85, 192), (78, 178),      # thumb 1-4
        (120, 180), (117, 150), (115, 128), (114, 110),   # index 5-8
        (145, 175), (143, 140), (142, 115), (141, 95),    # middle 9-12
        (170, 180), (172, 148), (173, 125), (174, 107),   # ring 13-16
        (192, 190), (195, 168), (197, 152), (198, 140),   # pinky 17-20
    ]


def zigzag_landmarks() -> list[tuple[float, float]]:
    """Middle finger zig-zags (PIP bends one way, DIP the other)."""
    pts = natural_landmarks()
    pts[10] = (120, 150)  # middle PIP
    pts[11] = (145, 130)  # middle DIP
    pts[12] = (120, 110)  # middle TIP
    return pts


def _hand(points, confidence=0.9) -> DetectedHand:
    return DetectedHand(landmarks=points, handedness="Right", confidence=confidence, bbox=dict(_BBOX))


def _features(points, confidence=0.9):
    return extract_hand_features(_hand(points, confidence))


def test_natural_hand_triggers_no_rules():
    f = _features(natural_landmarks())
    assert not f.degenerate
    for scorer in (
        score_joint_angle_anomaly,
        score_finger_length_anomaly,
        score_thumb_position_anomaly,
        score_wrist_connection_anomaly,
        score_finger_overlap_anomaly,
    ):
        assert scorer(f).score == 0.0, scorer.__name__


def test_zigzag_finger_flags_joint_angle_only():
    f = _features(zigzag_landmarks())
    res = score_joint_angle_anomaly(f)
    assert res.score >= settings.issue_threshold
    assert "middle:zigzag" in res.detail
    assert res.extra["boxes"], "should carry local evidence boxes"
    # isolation: other rules stay quiet
    assert score_finger_length_anomaly(f).score == 0.0
    assert score_thumb_position_anomaly(f).score == 0.0


def test_overlong_pinky_flags_length():
    pts = natural_landmarks()
    pts[17], pts[18], pts[19], pts[20] = (192, 190), (197, 150), (200, 115), (202, 85)
    res = score_finger_length_anomaly(_features(pts))
    assert res.score >= settings.issue_threshold
    assert "pinky" in res.detail


def test_thumb_on_pinky_side_flags():
    pts = natural_landmarks()
    pts[1], pts[2], pts[3], pts[4] = (205, 225), (215, 205), (222, 190), (228, 178)
    res = score_thumb_position_anomaly(_features(pts))
    assert res.score >= settings.issue_threshold
    assert res.detail == "thumb_on_pinky_side"


def test_merged_fingers_flag_overlap():
    pts = natural_landmarks()
    # ring chain nearly coincides with middle chain
    pts[13], pts[14], pts[15], pts[16] = (148, 177), (146, 142), (145, 117), (144, 97)
    res = score_finger_overlap_anomaly(_features(pts))
    assert res.score >= settings.issue_threshold
    assert "middle+ring" in res.detail


def test_integrate_natural_hand_is_ok():
    hand = _hand(natural_landmarks())
    payload = integrate_hand_issues([(hand, extract_hand_features(hand))], settings, (400, 400))
    assert payload["result"] == "ok"
    assert payload["overall_score"] == 0.0
    assert payload["issues"] == []


def test_integrate_zigzag_hand_flags_and_shapes_issue():
    hand = _hand(zigzag_landmarks())
    payload = integrate_hand_issues([(hand, extract_hand_features(hand))], settings, (400, 400))
    assert payload["result"] == "needs_review"
    flagged = [i for i in payload["issues"] if i["flagged"]]
    assert flagged and flagged[0]["type"] == HandIssueType.joint_angle_anomaly.value
    issue = flagged[0]
    assert issue["category"] == CATEGORY_HAND
    assert issue["id"].endswith("-h0")
    assert issue["evidence_boxes"] and issue["evidence_boxes"][0]["source"] == "hand_landmark"
    assert 0.0 <= issue["confidence"] <= 1.0


def test_low_confidence_hand_informational_issue():
    hand = _hand(natural_landmarks(), confidence=0.3)
    payload = integrate_hand_issues([(hand, extract_hand_features(hand))], settings, (400, 400))
    types = [i["type"] for i in payload["issues"]]
    assert HandIssueType.low_confidence_hand.value in types
    low = next(i for i in payload["issues"] if i["type"] == HandIssueType.low_confidence_hand.value)
    assert low["flagged"] is False
