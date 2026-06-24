"""POST /api/inspect-wrinkle — the main inspection endpoint."""

from __future__ import annotations

import json
import logging
import math
import time

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.models import Inspection
from app.schemas import DebugInfo, GarmentType, InspectResponse
from app.services import image_io
from app.services.anomaly_model import get_anomaly_model
from app.services.feature_extraction import extract_features
from app.services.pose import estimate_pose
from app.services.rule_engine import integrate_scores
from app.services.segmentation import get_garment_region
from app.services.wrinkle_edges import extract_wrinkle_candidates
from app.settings import settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["inspect"])


def _parse_region(selected_region: str | None) -> dict | None:
    if not selected_region:
        return None
    try:
        data = json.loads(selected_region)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"selected_region が不正なJSONです: {exc}")
    if not isinstance(data, dict):
        raise HTTPException(status_code=400, detail="selected_region はオブジェクトである必要があります。")
    return data


def _normalize_garment(garment_type: str | None) -> str:
    if not garment_type:
        return GarmentType.unknown.value
    try:
        return GarmentType(garment_type).value
    except ValueError:
        return GarmentType.unknown.value


@router.post("/inspect-wrinkle", response_model=InspectResponse)
async def inspect_wrinkle(
    image: UploadFile = File(...),
    garment_type: str | None = Form(None),
    selected_region: str | None = Form(None),
    db: Session = Depends(get_db),
) -> InspectResponse:
    t0 = time.perf_counter()
    garment = _normalize_garment(garment_type)
    region_dict = _parse_region(selected_region)

    raw = await image.read()
    try:
        saved_path, _safe_name, img = image_io.save_upload(raw, image.filename, settings)
    except image_io.UploadError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    notes: list[str] = []
    issues: list[dict] = []
    overall = 0.0
    result_str = "ok"
    scores: dict[str, float] = {}
    num_candidates = 0
    pose_detected = False
    region_used = False

    # The pipeline is wrapped so processing failures still return a 200 with
    # debug info (requirement 14: never crash on image-processing errors).
    try:
        region = get_garment_region(img, region_dict)
        region_used = region.used_selection
        pose = estimate_pose(img)
        pose_detected = pose.detected

        candidates = extract_wrinkle_candidates(img, region)
        num_candidates = candidates.count

        features = extract_features(candidates, pose)

        # Learned anomaly model (trained Mahalanobis model if available, else
        # a heuristic fallback). Contributes to overall_score via settings.
        anomaly_score = get_anomaly_model().score(features, garment)

        ih, iw = img.shape[:2]
        region_context = {
            "bbox": region.as_dict(),
            "image_w": iw,
            "image_h": ih,
            "region_diag": math.hypot(region.w, region.h),
        }
        payload = integrate_scores(
            features, garment, pose, region_context, settings, anomaly_score=anomaly_score
        )
        issues = payload["issues"]
        overall = payload["overall_score"]
        result_str = payload["result"]
        scores = payload["scores"]
        if not pose.detected:
            notes.append("pose_not_detected: 関節系の判定は低信頼度です。")
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

    # Persist inspection history.
    row = Inspection(
        image_path=str(saved_path),
        image_filename=image.filename or saved_path.name,
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
