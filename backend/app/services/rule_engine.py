"""Rule-based scoring for the six wrinkle-consistency requirements.

Each ``score_*`` function returns a :class:`ScoreResult` (score in 0..1, a
confidence, an optional issue bbox, and a short detail). ``integrate_scores``
combines them into the API's issue list + overall score.

These are intentionally simple, transparent heuristics (MVP). They are easy to
read, test, and later replace with learned thresholds derived from the photo
reference dataset (see ``reference_stats``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from app.schemas import IssueType, Severity
from app.services.explanation import generate_explanation, label_for
from app.services.feature_extraction import Features
from app.services.pose import PoseResult

# Rough default density expectations (line_count_density = lines per 10k px).
# Replaced later by stats computed from the 100-photo reference set.
DEFAULT_DENSITY_STATS: dict[str, tuple[float, float]] = {
    "shirt": (3.5, 2.5),
    "skirt": (3.0, 2.2),
    "pants": (3.2, 2.3),
    "dress": (3.0, 2.2),
    "jacket": (4.0, 2.8),
    "unknown": (3.3, 2.6),
}

# Garments that should mostly show gravity-driven (vertical) drape.
_HANGING_WEIGHT = {
    "skirt": 1.0,
    "dress": 1.0,
    "pants": 0.9,
    "jacket": 0.7,
    "shirt": 0.6,
    "unknown": 0.5,
}


@dataclass
class ScoreResult:
    score: float = 0.0
    confidence: float = 0.0
    bbox: dict[str, float] | None = None  # full-image coords; None -> use region
    detail: str = ""
    extra: dict = field(default_factory=dict)


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


def _smoothstep(edge0: float, edge1: float, x: float) -> float:
    """Smooth 0->1 ramp. Supports descending ramps (edge0 > edge1) too:
    e.g. ``_smoothstep(0.7, 0.15, x)`` returns ~1 for small x, ~0 for large x.
    """
    if edge0 == edge1:
        return 0.0 if x < edge0 else 1.0
    t = _clamp01((x - edge0) / (edge1 - edge0))
    return t * t * (3 - 2 * t)


def _region_bbox(region_context: dict) -> dict[str, float]:
    return dict(region_context.get("bbox", {"x": 0, "y": 0, "w": 1, "h": 1}))


# --------------------------------------------------------------------------- #
# 8.1 Gravity
# --------------------------------------------------------------------------- #
def score_gravity_inconsistency(
    features: Features, garment_type: str, region_context: dict
) -> ScoreResult:
    if features.num_lines < 3:
        return ScoreResult(0.0, 0.1, detail="few_lines")

    weight = _HANGING_WEIGHT.get(garment_type, 0.5)
    # 0 when wrinkles fall vertically, 1 when they run horizontally.
    horizontalness = features.angle_diff_from_gravity / 90.0
    raw = horizontalness * (0.5 + 0.5 * features.horizontal_fraction)
    score = _clamp01(weight * raw)
    confidence = _clamp01(0.35 + 0.4 * features.horizontal_fraction + 0.2 * weight)
    return ScoreResult(
        score=score,
        confidence=confidence,
        detail=f"horiz_frac={features.horizontal_fraction:.2f}",
    )


# --------------------------------------------------------------------------- #
# 8.2 Joint
# --------------------------------------------------------------------------- #
def score_joint_inconsistency(
    features: Features, pose_landmarks: PoseResult, region_context: dict
) -> ScoreResult:
    if not pose_landmarks.detected or features.nearest_joint is None:
        return ScoreResult(0.0, 0.0, detail="no_pose")

    dist = features.nearest_joint_dist_norm or 1.0
    if dist > 0.3:
        return ScoreResult(0.0, 0.2, detail="region_not_near_joint")

    # Near a joint we EXPECT compression wrinkles. Too few -> suspicious.
    # Also penalise wrinkles running straight through (low dispersion) at a bend.
    sparse = _smoothstep(1.0, 0.2, features.line_count_density)  # high when sparse
    straight_through = _smoothstep(0.5, 0.1, features.orientation_dispersion)
    score = _clamp01(max(0.5 * sparse, 0.6 * straight_through))
    confidence = _clamp01(0.4 + 0.4 * (1.0 - dist))
    return ScoreResult(
        score=score,
        confidence=confidence,
        detail=f"joint={features.nearest_joint},dist={dist:.2f}",
    )


# --------------------------------------------------------------------------- #
# 8.3 Tension
# --------------------------------------------------------------------------- #
def score_tension_ambiguity(
    features: Features, pose_landmarks: PoseResult, garment_type: str
) -> ScoreResult:
    if features.convergence_strength < 0.35 or features.convergence_point is None:
        return ScoreResult(0.0, 0.2, detail="no_strong_convergence")

    cx, cy = features.convergence_point
    base = features.convergence_strength

    if pose_landmarks.detected:
        anchors = ["left_shoulder", "right_shoulder", "left_wrist", "right_wrist",
                   "left_hip", "right_hip"]
        # Distance from the convergence point to the nearest plausible anchor.
        min_d = None
        for name in anchors:
            p = pose_landmarks.point(name)
            if p is None:
                continue
            d = math.hypot(p[0] - cx, p[1] - cy)
            min_d = d if min_d is None else min(min_d, d)
        if min_d is None:
            far = 0.5
        else:
            far = _smoothstep(120.0, 320.0, min_d)  # px; far from any anchor
        score = _clamp01(base * (0.3 + 0.7 * far))
        confidence = _clamp01(0.4 + 0.4 * base)
        return ScoreResult(score, confidence, bbox=_point_bbox(cx, cy),
                           detail=f"far_from_anchor={far:.2f}")

    # No pose: tension anchors are usually at garment edges. Without landmarks we
    # cannot verify them, so report a weaker, lower-confidence signal.
    return ScoreResult(
        score=_clamp01(base * 0.5),
        confidence=_clamp01(0.25 + 0.4 * base),
        bbox=_point_bbox(cx, cy),
        detail="no_pose_convergence",
    )


def _point_bbox(cx: float, cy: float, size: float = 60.0) -> dict[str, float]:
    return {"x": max(0.0, cx - size / 2), "y": max(0.0, cy - size / 2), "w": size, "h": size}


# --------------------------------------------------------------------------- #
# 8.4 Body volume
# --------------------------------------------------------------------------- #
def score_body_volume_inconsistency(
    features: Features, pose_landmarks: PoseResult, region_context: dict
) -> ScoreResult:
    if features.num_lines < 3:
        return ScoreResult(0.0, 0.1, detail="few_lines")

    diag = float(region_context.get("region_diag", 0.0)) or 1.0
    # Ruler-straight, parallel wrinkles ignore curved volume.
    straightness = _smoothstep(0.7, 0.15, features.orientation_dispersion)
    long_factor = _smoothstep(0.25, 0.75, features.mean_line_length / diag)
    # Vertical drape is natural; penalise mostly when NOT aligned with gravity.
    not_drape = 0.4 + 0.6 * (features.angle_diff_from_gravity / 90.0)
    score = _clamp01(straightness * long_factor * not_drape)
    confidence = _clamp01(0.3 + 0.4 * straightness)
    return ScoreResult(score, confidence, detail=f"straight={straightness:.2f}")


# --------------------------------------------------------------------------- #
# 8.5 Line density
# --------------------------------------------------------------------------- #
def score_line_density(
    features: Features,
    garment_type: str,
    region_context: dict,
    reference_stats: dict | None = None,
) -> ScoreResult:
    stats = reference_stats or DEFAULT_DENSITY_STATS
    mean, std = stats.get(garment_type, DEFAULT_DENSITY_STATS["unknown"])
    std = max(std, 1e-3)

    z = abs(features.line_count_density - mean) / (2.0 * std)
    deviation = _clamp01(z)  # both over- and under-wrinkled count
    # Over-concentration in a single patch.
    concentration = _smoothstep(1.5, 4.0, features.patch_density_max_z)
    score = _clamp01(max(deviation, concentration))
    confidence = 0.45 if reference_stats else 0.3  # lower until real stats exist
    return ScoreResult(
        score=score,
        confidence=confidence,
        detail=f"lcd={features.line_count_density:.2f},z={z:.2f}",
    )


# --------------------------------------------------------------------------- #
# 8.6 Shadow / wrinkle mismatch
# --------------------------------------------------------------------------- #
def score_shadow_wrinkle_mismatch(
    features: Features,
    image_patch=None,
    estimated_light_direction: float | None = None,
) -> ScoreResult:
    if features.num_lines < 3:
        return ScoreResult(0.0, 0.1, detail="few_lines")

    # Real wrinkle shading is perpendicular to the wrinkle. Parallel gradients
    # (perpendicularity ~ 0) suggest shading that ignores the wrinkle geometry.
    mismatch = 1.0 - features.shadow_wrinkle_perpendicularity
    trust = min(1.0, features.gradient_coherence * 2.0)
    score = _clamp01(mismatch * trust)
    confidence = _clamp01(0.25 + 0.5 * features.gradient_coherence)
    return ScoreResult(
        score=score,
        confidence=confidence,
        detail=f"perp={features.shadow_wrinkle_perpendicularity:.2f},coh={features.gradient_coherence:.2f}",
    )


# --------------------------------------------------------------------------- #
# Integration
# --------------------------------------------------------------------------- #
_SCORERS = [
    IssueType.gravity_inconsistency,
    IssueType.joint_inconsistency,
    IssueType.tension_ambiguity,
    IssueType.body_volume_inconsistency,
    IssueType.density_inconsistency,
    IssueType.shadow_wrinkle_mismatch,
]


def _severity(score: float, settings) -> Severity:
    if score >= settings.severity_high:
        return Severity.high
    if score >= settings.severity_medium:
        return Severity.medium
    return Severity.low


def run_all_scores(
    features: Features,
    garment_type: str,
    pose: PoseResult,
    region_context: dict,
    reference_stats: dict | None = None,
) -> dict[IssueType, ScoreResult]:
    """Run the six scorers and return a mapping of type -> ScoreResult."""
    return {
        IssueType.gravity_inconsistency: score_gravity_inconsistency(
            features, garment_type, region_context
        ),
        IssueType.joint_inconsistency: score_joint_inconsistency(
            features, pose, region_context
        ),
        IssueType.tension_ambiguity: score_tension_ambiguity(
            features, pose, garment_type
        ),
        IssueType.body_volume_inconsistency: score_body_volume_inconsistency(
            features, pose, region_context
        ),
        IssueType.density_inconsistency: score_line_density(
            features, garment_type, region_context, reference_stats
        ),
        IssueType.shadow_wrinkle_mismatch: score_shadow_wrinkle_mismatch(
            features, None, features.estimated_light_angle
        ),
    }


def integrate_scores(
    features: Features,
    garment_type: str,
    pose: PoseResult,
    region_context: dict,
    settings,
    reference_stats: dict | None = None,
    anomaly_score: float = 0.0,
    thresholds=None,
) -> dict:
    """Combine per-requirement scores into the inspection result payload.

    ``thresholds`` (optional :class:`Thresholds`) supplies per-issue-type /
    per-garment judgment thresholds; without it ``settings.issue_threshold`` is
    used. Issues with score >= ``settings.min_report_score`` are returned (each
    annotated with ``score``, ``threshold`` and ``flagged``) so the frontend can
    re-filter independently. ``anomaly_score`` (0..1) blends into
    ``overall_score`` via ``settings.anomaly_weight``.

    Returns: ``issues``, ``overall_score``, ``result``, ``scores``.
    """
    results = run_all_scores(features, garment_type, pose, region_context, reference_stats)
    region_bbox = _region_bbox(region_context)
    floor = float(getattr(settings, "min_report_score", settings.issue_threshold))

    def _effective(issue_type_value: str) -> float:
        if thresholds is not None and getattr(thresholds, "loaded", False):
            return float(thresholds.effective(issue_type_value, garment_type))
        return float(settings.issue_threshold)

    issues: list[dict] = []
    scores: dict[str, float] = {}
    any_flagged = False
    for idx, issue_type in enumerate(_SCORERS):
        res = results[issue_type]
        scores[issue_type.value] = round(res.score, 4)
        if res.score < floor:
            continue
        eff = _effective(issue_type.value)
        flagged = res.score >= eff
        any_flagged = any_flagged or flagged
        bbox = res.bbox or region_bbox
        issues.append(
            {
                "id": f"{issue_type.value}-{idx}",
                "type": issue_type.value,
                "label": label_for(issue_type.value),
                "severity": _severity(res.score, settings).value,
                "bbox": {
                    "x": float(bbox["x"]),
                    "y": float(bbox["y"]),
                    "w": float(bbox["w"]),
                    "h": float(bbox["h"]),
                },
                "confidence": round(_clamp01(res.confidence), 4),
                "message": generate_explanation(issue_type.value, res.score, features),
                "score": round(res.score, 4),
                "threshold": round(eff, 4),
                "flagged": flagged,
            }
        )

    # Aggregate the six rule scores, then blend in the learned anomaly signal.
    if scores:
        max_s = max(scores.values())
        mean_s = sum(scores.values()) / len(scores)
    else:
        max_s = mean_s = 0.0

    anomaly = _clamp01(anomaly_score)
    w = _clamp01(getattr(settings, "anomaly_weight", 0.15))
    rule_overall = 0.6 * max_s + 0.4 * mean_s
    overall = round(_clamp01((1.0 - w) * rule_overall + w * anomaly), 4)

    scores["anomaly_model"] = round(anomaly, 4)
    result = "needs_review" if (any_flagged or overall >= settings.review_threshold) else "ok"

    return {
        "issues": issues,
        "overall_score": overall,
        "result": result,
        "scores": scores,
    }
