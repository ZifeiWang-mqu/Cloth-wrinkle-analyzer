"""Shared inspection logic used by all inspect endpoints.

Both ``/api/inspect-wrinkle`` (file upload) and ``/api/inspect-base64``
(external tools) decode an image to a BGR ndarray + a persisted path, then call
:func:`run_inspection`. This avoids duplicating the pipeline/persistence logic.
"""

from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path

import numpy as np
from sqlalchemy.orm import Session

from app.db.models import Inspection
from app.schemas import DebugInfo, InspectResponse
from app.services.anomaly_model import get_anomaly_model
from app.services.feature_extraction import extract_features
from app.services.pose import estimate_pose
from app.services.reference_stats import load_density_stats
from app.services.rule_engine import integrate_scores
from app.services.segmentation import get_garment_region
from app.services.thresholds import load_thresholds
from app.services.wrinkle_edges import extract_wrinkle_candidates
from app.settings import settings

logger = logging.getLogger(__name__)


def run_inspection(
    *,
    image_bgr: np.ndarray,
    garment: str,
    region_dict: dict | None,
    db: Session,
    saved_path: Path,
    image_filename: str | None,
    source: str = "web",
) -> InspectResponse:
    """Run the full inspection pipeline, persist history, and return the result.

    Image-processing failures never raise: a 200 result with ``debug.notes`` is
    returned instead (requirement §13: never crash on a bad image / missing
    model).
    """
    t0 = time.perf_counter()
    notes: list[str] = []
    issues: list[dict] = []
    overall = 0.0
    result_str = "ok"
    scores: dict[str, float] = {}
    num_candidates = 0
    pose_detected = False
    region_used = False

    try:
        region = get_garment_region(image_bgr, region_dict)
        region_used = region.used_selection
        pose = estimate_pose(image_bgr)
        pose_detected = pose.detected

        candidates = extract_wrinkle_candidates(image_bgr, region)
        num_candidates = candidates.count

        features = extract_features(candidates, pose)
        anomaly_score = get_anomaly_model().score(features, garment)

        ih, iw = image_bgr.shape[:2]
        region_context = {
            "bbox": region.as_dict(),
            "image_w": iw,
            "image_h": ih,
            "region_diag": math.hypot(region.w, region.h),
        }

        thresholds = load_thresholds(settings.thresholds_path)
        density_stats, _ = load_density_stats(settings.reference_stats_path)

        payload = integrate_scores(
            features,
            garment,
            pose,
            region_context,
            settings,
            reference_stats=density_stats,
            anomaly_score=anomaly_score,
            thresholds=thresholds,
        )
        issues = payload["issues"]
        overall = payload["overall_score"]
        result_str = payload["result"]
        scores = payload["scores"]
        if not pose.detected:
            notes.append("pose_not_detected: 関節系の判定は低信頼度です。")
        if not thresholds.loaded:
            notes.append("thresholds_not_loaded: 既定の閾値を使用しています。")
    except Exception as exc:  # pragma: no cover - defensive top-level guard
        logger.exception("Inspection pipeline failed")
        notes.append(f"pipeline_error: {exc}")

    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    debug = DebugInfo(
        pose_detected=pose_detected,
        garment_region_used=region_used,
        num_wrinkle_candidates=num_candidates,
        processing_time_ms=elapsed_ms,
        scores=scores,
        notes=notes,
    )

    row = Inspection(
        image_path=str(saved_path),
        image_filename=image_filename or saved_path.name,
        garment_type=garment,
        selected_region_json=json.dumps(region_dict) if region_dict else None,
        overall_score=overall,
        result=result_str,
        issues_json=json.dumps(issues, ensure_ascii=False),
        debug_json=json.dumps(debug.model_dump(), ensure_ascii=False),
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
