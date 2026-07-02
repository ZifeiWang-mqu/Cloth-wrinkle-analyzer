"""Stage-1 schema tests for hand inspection support.

Confirms the additive changes are fully backward-compatible:
- existing wrinkle issues (no ``category`` key) still validate and default to
  ``category="wrinkle"``,
- hand issue types validate through the ``IssueType | HandIssueType`` union,
- arbitrary strings are still rejected (no unrestricted ``type``).
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.schemas import (
    CATEGORY_HAND,
    CATEGORY_WRINKLE,
    HandIssueType,
    InspectResponse,
    Issue,
    IssueType,
)

_BBOX = {"x": 10, "y": 20, "w": 90, "h": 120}


def _issue(**overrides) -> dict:
    base = {
        "id": "gravity_inconsistency-0",
        "type": "gravity_inconsistency",
        "label": "重力と皺の矛盾",
        "severity": "medium",
        "bbox": _BBOX,
        "confidence": 0.72,
        "message": "テスト",
    }
    base.update(overrides)
    return base


def test_wrinkle_issue_without_category_defaults_to_wrinkle():
    issue = Issue(**_issue())
    assert issue.category == CATEGORY_WRINKLE
    assert issue.type is IssueType.gravity_inconsistency
    # category serialises so future clients can rely on it
    assert issue.model_dump()["category"] == "wrinkle"


def test_hand_issue_type_validates_via_union():
    issue = Issue(
        **_issue(
            id="finger_length_anomaly-0",
            type="finger_length_anomaly",
            category=CATEGORY_HAND,
            label="Unnatural finger length",
            severity="high",
            confidence=0.86,
            message="Finger length ratios are far outside natural range.",
        )
    )
    assert issue.type is HandIssueType.finger_length_anomaly
    assert issue.category == CATEGORY_HAND


@pytest.mark.parametrize("hand_type", [t.value for t in HandIssueType])
def test_all_hand_types_accepted(hand_type: str):
    issue = Issue(**_issue(id=f"{hand_type}-0", type=hand_type, category=CATEGORY_HAND))
    assert issue.type.value == hand_type


def test_unknown_type_still_rejected():
    with pytest.raises(ValidationError):
        Issue(**_issue(type="totally_made_up_type"))


def test_inspect_response_accepts_mixed_categories():
    resp = InspectResponse(
        inspection_id="abc",
        result="needs_review",
        overall_score=0.5,
        issues=[
            _issue(),  # wrinkle, no category key (legacy shape)
            _issue(
                id="joint_angle_anomaly-0",
                type="joint_angle_anomaly",
                category=CATEGORY_HAND,
                label="Impossible joint angle",
            ),
        ],
        debug={"pose_detected": False},
    )
    cats = [i.category for i in resp.issues]
    assert cats == [CATEGORY_WRINKLE, CATEGORY_HAND]
    # round-trip: stored-JSON style dicts (as in DB issues_json) revalidate
    for raw in [i.model_dump() for i in resp.issues]:
        assert Issue(**raw).category in {CATEGORY_WRINKLE, CATEGORY_HAND}
