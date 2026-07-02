"""Rule-based scoring for hand-unnaturalness (MVP).

Five landmark-geometry rules, each returning the existing
:class:`~app.services.rule_engine.ScoreResult`. Thresholds are deliberately
GENEROUS: 2D projections of real poses (foreshortening, fists, side views)
must not fire. ``integrate_hand_issues`` turns per-hand results into issues in
the same shape the wrinkle engine produces, with ``category="hand"``.

Labels/messages are English for the MVP (Japanese localisation is a later,
frontend-driven step).
"""

from __future__ import annotations

from app.schemas import CATEGORY_HAND, HandIssueType, Severity
from app.services.hand_detection import DetectedHand
from app.services.hand_features import HandFeatures, Point
from app.services.rule_engine import ScoreResult

# Generous natural bands for finger length ratios (vs middle finger, 2D).
_LENGTH_BANDS: dict[str, tuple[float, float]] = {
    "thumb": (0.35, 1.20),
    "index": (0.50, 1.20),
    "middle": (1.00, 1.00),  # reference — never flagged by ratio
    "ring": (0.50, 1.20),
    "pinky": (0.30, 1.00),
}

_FOLD_DEG = 25.0  # interior angle below this = folded back on itself
_ZIGZAG_FLEX_DEG = 35.0  # both joints bent past this in OPPOSITE directions
_OVERLAP_TIP = 0.12  # fraction of palm width
_OVERLAP_PIP = 0.15
_WRIST_BAND = (0.40, 3.00)

HAND_LABELS: dict[str, str] = {
    HandIssueType.joint_angle_anomaly.value: "Impossible joint angle",
    HandIssueType.finger_length_anomaly.value: "Unnatural finger length",
    HandIssueType.thumb_position_anomaly.value: "Thumb on the wrong side",
    HandIssueType.wrist_connection_anomaly.value: "Implausible wrist connection",
    HandIssueType.finger_overlap_anomaly.value: "Fingers appear merged",
    HandIssueType.low_confidence_hand.value: "Low-confidence hand detection",
    HandIssueType.hand_detection_failed.value: "Hand detection unavailable",
}


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


def _count_score(count: int) -> float:
    """1 anomaly -> 0.5, each extra -> +0.25 (capped at 1.0)."""
    return _clamp01(0.5 + 0.25 * (count - 1)) if count > 0 else 0.0


def _points_box(points: list[Point], pad: float) -> dict[str, float]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return {
        "x": max(0.0, min(xs) - pad),
        "y": max(0.0, min(ys) - pad),
        "w": (max(xs) - min(xs)) + 2 * pad,
        "h": (max(ys) - min(ys)) + 2 * pad,
    }


# --------------------------------------------------------------------------- #
# Rules (each: HandFeatures -> ScoreResult; evidence boxes in extra["boxes"])
# --------------------------------------------------------------------------- #
def score_joint_angle_anomaly(f: HandFeatures) -> ScoreResult:
    if f.degenerate:
        return ScoreResult(0.0, 0.1, detail="degenerate_hand")
    pad = 0.15 * f.palm_width
    bad: list[str] = []
    boxes: list[dict] = []
    by_finger: dict[str, list] = {}
    for b in f.bends:
        by_finger.setdefault(b.finger, []).append(b)
    for finger, bends in by_finger.items():
        fold = any(b.interior_deg < _FOLD_DEG for b in bends)
        zigzag = False
        if len(bends) == 2:
            b1, b2 = bends
            flex1, flex2 = 180.0 - b1.interior_deg, 180.0 - b2.interior_deg
            zigzag = (
                flex1 > _ZIGZAG_FLEX_DEG
                and flex2 > _ZIGZAG_FLEX_DEG
                and b1.sign != 0
                and b2.sign != 0
                and b1.sign != b2.sign
            )
        if fold or zigzag:
            bad.append(f"{finger}:{'fold' if fold else 'zigzag'}")
            boxes.append(_points_box(f.finger_points(finger), pad))
    score = _count_score(len(bad))
    conf = _clamp01(0.4 + 0.5 * f.confidence) if bad else 0.3
    return ScoreResult(score, conf, detail=",".join(bad) or "ok", extra={"boxes": boxes})


def score_finger_length_anomaly(f: HandFeatures) -> ScoreResult:
    if f.degenerate:
        return ScoreResult(0.0, 0.1, detail="degenerate_hand")
    pad = 0.15 * f.palm_width
    bad: list[str] = []
    boxes: list[dict] = []
    for name, ratio in f.length_ratios.items():
        if name == "middle":
            continue
        lo, hi = _LENGTH_BANDS[name]
        near_zero = f.finger_lengths[name] < 0.08 * f.palm_width
        if near_zero or not (lo <= ratio <= hi):
            bad.append(f"{name}={ratio:.2f}")
            boxes.append(_points_box(f.finger_points(name), pad))
    score = _count_score(len(bad))
    conf = _clamp01(0.35 + 0.4 * f.confidence) if bad else 0.3
    return ScoreResult(score, conf, detail=",".join(bad) or "ok", extra={"boxes": boxes})


def score_thumb_position_anomaly(f: HandFeatures) -> ScoreResult:
    if f.degenerate:
        return ScoreResult(0.0, 0.1, detail="degenerate_hand")
    # Only judge when index and pinky clearly sit on opposite sides of the
    # palm axis (otherwise the view is too ambiguous in 2D).
    if f.index_side == 0 or f.pinky_side == 0 or f.index_side == f.pinky_side:
        return ScoreResult(0.0, 0.15, detail="ambiguous_view")
    if f.thumb_side == f.pinky_side:
        pad = 0.15 * f.palm_width
        return ScoreResult(
            0.7,
            _clamp01(0.3 + 0.5 * f.confidence),
            detail="thumb_on_pinky_side",
            extra={"boxes": [_points_box(f.finger_points("thumb"), pad)]},
        )
    return ScoreResult(0.0, 0.4, detail="ok")


def score_wrist_connection_anomaly(f: HandFeatures) -> ScoreResult:
    if f.degenerate:
        return ScoreResult(0.0, 0.1, detail="degenerate_hand")
    lo, hi = _WRIST_BAND
    if lo <= f.wrist_palm_ratio <= hi:
        return ScoreResult(0.0, 0.4, detail=f"ratio={f.wrist_palm_ratio:.2f}")
    return ScoreResult(
        0.6,
        _clamp01(0.3 + 0.4 * f.confidence),
        detail=f"ratio={f.wrist_palm_ratio:.2f}",
        extra={"boxes": [dict(f.bbox)] if f.bbox else []},
    )


def score_finger_overlap_anomaly(f: HandFeatures) -> ScoreResult:
    if f.degenerate:
        return ScoreResult(0.0, 0.1, detail="degenerate_hand")
    pad = 0.15 * f.palm_width
    merged: list[str] = []
    boxes: list[dict] = []
    for a, b, tip_d, pip_d in f.adjacent_pairs:
        if tip_d < _OVERLAP_TIP and pip_d < _OVERLAP_PIP:
            merged.append(f"{a}+{b}")
            boxes.append(_points_box(f.finger_points(a) + f.finger_points(b), pad))
    score = _count_score(len(merged))
    conf = _clamp01(0.35 + 0.45 * f.confidence) if merged else 0.3
    return ScoreResult(score, conf, detail=",".join(merged) or "ok", extra={"boxes": boxes})


_HAND_SCORERS: dict[HandIssueType, callable] = {
    HandIssueType.joint_angle_anomaly: score_joint_angle_anomaly,
    HandIssueType.finger_length_anomaly: score_finger_length_anomaly,
    HandIssueType.thumb_position_anomaly: score_thumb_position_anomaly,
    HandIssueType.wrist_connection_anomaly: score_wrist_connection_anomaly,
    HandIssueType.finger_overlap_anomaly: score_finger_overlap_anomaly,
}

# Human-readable firing conditions per rule (diagnostics only).
RULE_FIRES_WHEN: dict[str, str] = {
    HandIssueType.joint_angle_anomaly.value: (
        f"interior angle < {_FOLD_DEG:.0f} deg (fold) OR both joints flexed > "
        f"{_ZIGZAG_FLEX_DEG:.0f} deg in opposite directions (zigzag)"
    ),
    HandIssueType.finger_length_anomaly.value: (
        "length ratio vs middle outside bands "
        + ", ".join(f"{k} {lo:.2f}-{hi:.2f}" for k, (lo, hi) in _LENGTH_BANDS.items() if k != "middle")
        + ", or finger length < 0.08 x palm width"
    ),
    HandIssueType.thumb_position_anomaly.value: (
        "thumb MCP on the pinky side of the wrist->middle-MCP axis "
        "(only judged when index/pinky sit on clearly opposite sides)"
    ),
    HandIssueType.wrist_connection_anomaly.value: (
        f"wrist-palm ratio outside {_WRIST_BAND[0]:.2f}-{_WRIST_BAND[1]:.2f}"
    ),
    HandIssueType.finger_overlap_anomaly.value: (
        f"adjacent fingertips < {_OVERLAP_TIP:.2f} AND PIPs < {_OVERLAP_PIP:.2f} x palm width"
    ),
}


def _zero_reason(detail: str) -> str:
    """Short human explanation for a 0.0 score (diagnostics only)."""
    if detail == "degenerate_hand":
        return "palm too small to measure — hand geometry collapsed"
    if detail == "ambiguous_view":
        return "index/pinky not on opposite sides — 2D view too ambiguous to judge the thumb"
    if detail == "ok" or detail.startswith("ratio="):
        return "values within natural bands"
    return "no anomaly condition met"


_MESSAGES: dict[str, str] = {
    HandIssueType.joint_angle_anomaly.value: "A finger bends at an angle that is anatomically implausible (folded back or zig-zagging).",
    HandIssueType.finger_length_anomaly.value: "Finger length proportions fall outside the natural range for a hand.",
    HandIssueType.thumb_position_anomaly.value: "The thumb appears to attach on the pinky side of the palm.",
    HandIssueType.wrist_connection_anomaly.value: "The wrist position is implausible relative to the palm.",
    HandIssueType.finger_overlap_anomaly.value: "Adjacent fingers overlap so closely they may be drawn as one.",
    HandIssueType.low_confidence_hand.value: "The hand detector was unsure about this region; results are indicative only.",
}


def _severity(score: float, settings) -> str:
    if score >= settings.severity_high:
        return Severity.high.value
    if score >= settings.severity_medium:
        return Severity.medium.value
    return Severity.low.value


def _clamp_box(b: dict, iw: int, ih: int) -> dict:
    x = max(0.0, min(float(b["x"]), iw - 1.0))
    y = max(0.0, min(float(b["y"]), ih - 1.0))
    w = max(1.0, min(float(b["w"]), iw - x))
    h = max(1.0, min(float(b["h"]), ih - y))
    return {"x": round(x, 1), "y": round(y, 1), "w": round(w, 1), "h": round(h, 1)}


def integrate_hand_issues(
    hands: list[tuple[DetectedHand, HandFeatures]],
    settings,
    image_shape: tuple[int, int],
) -> dict:
    """Run all hand rules and build the issues payload (wrinkle-compatible).

    Returns ``{"issues", "overall_score", "result", "scores"}`` where each
    issue carries ``category="hand"``, a validated ``HandIssueType``, and
    landmark-derived ``evidence_boxes``.
    """
    ih, iw = image_shape
    issues: list[dict] = []
    scores: dict[str, float] = {}
    all_scores: list[float] = []
    rule_details: list[dict] = []
    any_flagged = False
    floor = float(settings.min_report_score)
    threshold = float(settings.issue_threshold)

    for idx, (hand, feats) in enumerate(hands):
        hand_rules: list[dict] = []
        for issue_type, scorer in _HAND_SCORERS.items():
            res = scorer(feats)
            key = issue_type.value
            scores[key] = max(scores.get(key, 0.0), round(res.score, 4))
            all_scores.append(res.score)
            hand_rules.append(
                {
                    "rule": key,
                    "score": round(res.score, 4),
                    "detail": res.detail,
                    "fires_when": RULE_FIRES_WHEN[key],
                    "reason": (
                        f"anomaly condition met ({res.detail})"
                        if res.score > 0
                        else _zero_reason(res.detail)
                    ),
                }
            )
            if res.score < floor:
                continue
            flagged = res.score >= threshold
            any_flagged = any_flagged or flagged
            boxes = [
                {**_clamp_box(b, iw, ih), "source": "hand_landmark"}
                for b in res.extra.get("boxes", [])
                if b.get("w", 0) > 0 and b.get("h", 0) > 0
            ]
            bbox = _clamp_box(hand.bbox, iw, ih) if hand.bbox else {"x": 0, "y": 0, "w": iw, "h": ih}
            issues.append(
                {
                    "id": f"{key}-h{idx}",
                    "type": key,
                    "category": CATEGORY_HAND,
                    "label": HAND_LABELS[key],
                    "severity": _severity(res.score, settings),
                    "bbox": bbox,
                    "confidence": round(_clamp01(res.confidence), 4),
                    "message": f"{_MESSAGES[key]} ({res.detail})",
                    "score": round(res.score, 4),
                    "threshold": round(threshold, 4),
                    "flagged": flagged,
                    "evidence_boxes": boxes
                    or [{**bbox, "fallback_broad_bbox": True, "source": "hand_bbox"}],
                }
            )

        # Informational: unsure detector (kept below the judgment threshold).
        if feats.confidence < 0.5:
            key = HandIssueType.low_confidence_hand.value
            bbox = _clamp_box(hand.bbox, iw, ih) if hand.bbox else {"x": 0, "y": 0, "w": iw, "h": ih}
            issues.append(
                {
                    "id": f"{key}-h{idx}",
                    "type": key,
                    "category": CATEGORY_HAND,
                    "label": HAND_LABELS[key],
                    "severity": Severity.low.value,
                    "bbox": bbox,
                    "confidence": round(_clamp01(feats.confidence), 4),
                    "message": _MESSAGES[key],
                    "score": 0.3,
                    "threshold": round(threshold, 4),
                    "flagged": False,
                    "evidence_boxes": [{**bbox, "fallback_broad_bbox": True, "source": "hand_bbox"}],
                }
            )
            scores[key] = max(scores.get(key, 0.0), 0.3)

        rule_details.append({"hand": idx, "rules": hand_rules})

    if all_scores:
        overall = _clamp01(0.6 * max(all_scores) + 0.4 * (sum(all_scores) / len(all_scores)))
    else:
        overall = 0.0
    result = "needs_review" if (any_flagged or overall >= settings.review_threshold) else "ok"
    return {
        "issues": issues,
        "overall_score": round(overall, 4),
        "result": result,
        "scores": scores,
        # Diagnostics only — surfaced under debug.hand.rule_details.
        "rule_details": rule_details,
    }
