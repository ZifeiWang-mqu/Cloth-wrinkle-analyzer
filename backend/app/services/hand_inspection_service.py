"""Orchestration for the hand-unnaturalness inspection (MVP).

Mirrors ``inspection_service.run_inspection`` but for hands, reusing the same
persistence (``Inspection`` row, mode marked in debug notes) and the same
``InspectResponse`` shape so the existing frontend display works unchanged.
Never crashes on detector failure: returns an informational
``hand_detection_failed`` issue instead.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
from sqlalchemy.orm import Session

from app.db.models import Inspection
from app.schemas import CATEGORY_HAND, DebugInfo, HandIssueType, InspectResponse, Severity
from app.services.hand_detection import detect_hands
from app.services.hand_features import extract_hand_features
from app.services.hand_rule_engine import HAND_LABELS, integrate_hand_issues
from app.services.segmentation import get_garment_region
from app.settings import settings

logger = logging.getLogger(__name__)


def _detection_failed_issue(note: str | None, bbox: dict, threshold: float) -> dict:
    """Informational issue for 'detector unavailable / found nothing'.

    Honest by design: score 0.0 and flagged False — this is guidance, not a
    detected drawing problem. The distinct type lets the frontend surface it.
    """
    key = HandIssueType.hand_detection_failed.value
    reason = {
        "mediapipe_not_installed": "MediaPipe is not installed on the server.",
        "hand_model_missing": "The hand landmark model file (hand_landmarker.task) is not installed on the server — see README to download it into data/models/hand/.",
        "hand_landmarker_unavailable": "This MediaPipe build exposes neither the Tasks API nor the legacy hands API.",
        "no_hands_found": "No hand could be detected in the image/region (stylized hands often defeat photo-trained detectors).",
    }.get(note or "", "Hand detection did not produce a usable result.")
    return {
        "id": f"{key}-0",
        "type": key,
        "category": CATEGORY_HAND,
        "label": HAND_LABELS[key],
        "severity": Severity.low.value,
        "bbox": bbox,
        "confidence": 0.2,
        "message": f"{reason} Try selecting the hand area with the lasso tool and re-running.",
        "score": 0.0,
        "threshold": threshold,
        "flagged": False,
        "evidence_boxes": [{**bbox, "fallback_broad_bbox": True, "source": "hand_bbox"}],
    }


def run_hand_inspection(
    *,
    image_bgr: np.ndarray,
    region_dict: dict | None,
    db: Session,
    saved_path: Path,
    image_filename: str | None,
    source: str = "web",
    return_debug_overlays: bool = False,
) -> InspectResponse:
    """Detect hands, score MVP rules, persist, and return an InspectResponse."""
    t0 = time.perf_counter()
    ih, iw = image_bgr.shape[:2]
    notes: list[str] = ["mode: hand"]
    issues: list[dict] = []
    overall = 0.0
    result_str = "ok"
    scores: dict[str, float] = {}
    hand_debug: dict = {}
    overlays: dict | None = None

    try:
        region = get_garment_region(image_bgr, region_dict)
        detection = detect_hands(image_bgr, region if region.used_selection else None)
        hand_debug = detection.to_debug()

        if not detection.detected:
            bbox = region.as_dict() if region.used_selection else {"x": 0, "y": 0, "w": iw, "h": ih}
            issues = [
                _detection_failed_issue(
                    detection.note, bbox, float(settings.issue_threshold)
                )
            ]
            notes.append(f"hand_detection: {detection.note}")
        else:
            hands = [(h, extract_hand_features(h)) for h in detection.hands]
            payload = integrate_hand_issues(hands, settings, (ih, iw))
            issues = payload["issues"]
            overall = payload["overall_score"]
            result_str = payload["result"]
            scores = payload["scores"]
            # Diagnostics: raw feature values + per-rule "why (not) fired".
            hand_debug["features"] = [feats.to_debug() for _, feats in hands]
            hand_debug["rule_details"] = payload.get("rule_details", [])
            if return_debug_overlays:
                overlays = {
                    "hand_landmarks": [
                        [[round(x, 1), round(y, 1)] for x, y in h.landmarks]
                        for h in detection.hands
                    ]
                }
    except Exception as exc:  # pragma: no cover - defensive top-level guard
        logger.exception("Hand inspection pipeline failed")
        notes.append(f"pipeline_error: {exc}")

    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    debug = DebugInfo(
        pose_detected=False,
        garment_region_used=bool(region_dict),
        num_wrinkle_candidates=0,
        processing_time_ms=elapsed_ms,
        scores=scores,
        notes=notes,
        hand=hand_debug or None,
        overlays=overlays,
    )

    row = Inspection(
        image_path=str(saved_path),
        image_filename=image_filename or saved_path.name,
        garment_type="hand",  # data-level mode marker; DB schema unchanged
        selected_region_json=json.dumps(region_dict) if region_dict else None,
        overall_score=overall,
        result=result_str,
        issues_json=json.dumps(issues, ensure_ascii=False),
        debug_json=json.dumps(debug.model_dump(exclude={"overlays"}), ensure_ascii=False),
    )
    db.add(row)
    db.commit()
    db.refresh(row)

    return InspectResponse(
        inspection_id=row.id,
        result=result_str,
        overall_score=overall,
        issues=issues,
        debug=debug,
    )
