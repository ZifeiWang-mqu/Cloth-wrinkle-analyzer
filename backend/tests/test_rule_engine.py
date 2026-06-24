"""Unit tests for the rule engine and individual scorers."""

from __future__ import annotations

from app.schemas import IssueType
from app.services.feature_extraction import Features
from app.services.pose import PoseResult
from app.services.rule_engine import (
    integrate_scores,
    run_all_scores,
    score_gravity_inconsistency,
    score_line_density,
    score_shadow_wrinkle_mismatch,
)
from app.settings import settings

REGION_CTX = {
    "bbox": {"x": 0, "y": 0, "w": 200, "h": 400},
    "image_w": 200,
    "image_h": 400,
    "region_diag": 447.0,
}


def _vertical_drape() -> Features:
    return Features(
        num_lines=12,
        area_px=80000.0,
        edge_density=0.04,
        line_count_density=3.0,
        line_length_sum=600.0,
        mean_line_length=50.0,
        dominant_line_angle=90.0,
        orientation_dispersion=0.3,
        vertical_fraction=0.8,
        horizontal_fraction=0.1,
        angle_diff_from_gravity=0.0,
        gradient_coherence=0.5,
        shadow_wrinkle_perpendicularity=0.9,
    )


def _horizontal_mess() -> Features:
    return Features(
        num_lines=12,
        area_px=80000.0,
        edge_density=0.04,
        line_count_density=3.0,
        line_length_sum=600.0,
        mean_line_length=50.0,
        dominant_line_angle=5.0,
        orientation_dispersion=0.3,
        vertical_fraction=0.1,
        horizontal_fraction=0.8,
        angle_diff_from_gravity=85.0,
        gradient_coherence=0.5,
        shadow_wrinkle_perpendicularity=0.1,
    )


def test_all_scores_in_range():
    feats = _horizontal_mess()
    pose = PoseResult()
    results = run_all_scores(feats, "skirt", pose, REGION_CTX)
    assert set(results.keys()) == set(IssueType)
    for res in results.values():
        assert 0.0 <= res.score <= 1.0
        assert 0.0 <= res.confidence <= 1.0


def test_gravity_high_for_horizontal_skirt():
    high = score_gravity_inconsistency(_horizontal_mess(), "skirt", REGION_CTX)
    low = score_gravity_inconsistency(_vertical_drape(), "skirt", REGION_CTX)
    assert high.score > low.score
    assert high.score >= settings.issue_threshold
    assert low.score < settings.issue_threshold


def test_gravity_weaker_for_unknown_than_skirt():
    skirt = score_gravity_inconsistency(_horizontal_mess(), "skirt", REGION_CTX)
    unknown = score_gravity_inconsistency(_horizontal_mess(), "unknown", REGION_CTX)
    assert skirt.score >= unknown.score


def test_few_lines_yield_no_gravity_issue():
    feats = _horizontal_mess()
    feats.num_lines = 1
    res = score_gravity_inconsistency(feats, "skirt", REGION_CTX)
    assert res.score == 0.0


def test_joint_skipped_without_pose():
    from app.services.rule_engine import score_joint_inconsistency

    res = score_joint_inconsistency(_horizontal_mess(), PoseResult(), REGION_CTX)
    assert res.score == 0.0


def test_density_extremes_flagged():
    feats = _vertical_drape()
    feats.line_count_density = 30.0  # way over baseline
    over = score_line_density(feats, "skirt", REGION_CTX)
    assert over.score >= settings.issue_threshold

    feats.line_count_density = 0.0  # way under baseline
    under = score_line_density(feats, "skirt", REGION_CTX)
    assert under.score >= settings.issue_threshold


def test_shadow_mismatch_needs_coherence():
    feats = _horizontal_mess()
    feats.gradient_coherence = 0.0  # untrusted shading
    res = score_shadow_wrinkle_mismatch(feats)
    assert res.score == 0.0


def test_integrate_scores_shape():
    payload = integrate_scores(_horizontal_mess(), "skirt", PoseResult(), REGION_CTX, settings)
    assert {"issues", "overall_score", "result", "scores", "model_scores"} <= set(payload.keys())
    assert set(payload["model_scores"].keys()) == {
        "rule_score",
        "anomaly_score",
        "illustration_model_score",
        "final_score",
    }
    assert 0.0 <= payload["overall_score"] <= 1.0
    assert payload["result"] in {"ok", "needs_review"}
    # 6 rule scores + the learned anomaly_model score.
    assert len(payload["scores"]) == 7
    assert "anomaly_model" in payload["scores"]
    for issue in payload["issues"]:
        assert {"id", "type", "label", "severity", "bbox", "confidence", "message"} <= set(issue)
        assert 0.0 <= issue["confidence"] <= 1.0
        assert issue["severity"] in {"low", "medium", "high"}
