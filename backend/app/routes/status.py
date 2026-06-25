"""Status / retrieval endpoints for external tools.

GET /api/model/status      — model + reference stats + thresholds status
GET /api/inspection/{id}   — fetch a past inspection result + its feedback
"""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app import __version__
from app.db.database import get_db
from app.db.models import Inspection
from app.schemas import (
    DebugInfo,
    InspectionDetail,
    Issue,
    ModelStatus,
    StoredFeedback,
)
from app.services import capabilities, illustration_model
from app.services.anomaly_model import describe_anomaly_model
from app.services.reference_stats import load_density_stats
from app.services.segmentation import sam_status
from app.services.thresholds import load_thresholds
from app.settings import settings

router = APIRouter(prefix="/api", tags=["status"])


@router.get("/model/status", response_model=ModelStatus)
def model_status() -> ModelStatus:
    desc = describe_anomaly_model()
    th = load_thresholds(settings.thresholds_path)
    _, rs_loaded = load_density_stats(settings.reference_stats_path)
    caps = capabilities.get_capabilities(settings)
    return ModelStatus(
        model_loaded=bool(desc["loaded"]),
        model_type=str(desc["type"]),
        model_path=str(settings.model_path),
        reference_stats_loaded=bool(rs_loaded),
        reference_stats_path=str(settings.reference_stats_path),
        thresholds_loaded=th.loaded,
        thresholds_path=str(settings.thresholds_path),
        available_garment_models=list(desc["garments"]),
        sam_available=bool(caps["segmentation"]["sam_available"]),
        sam_checkpoint_present=bool(caps["segmentation"]["sam_checkpoint_present"]),
        mediapipe_available=bool(caps["pose"]["mediapipe_available"]),
        segmentation=sam_status(settings),
        illustration_feedback_model=illustration_model.describe(),
        version=__version__,
    )


@router.get("/debug/capabilities", tags=["status"])
def debug_capabilities() -> dict:
    """Which optional features are available in this environment."""
    return capabilities.get_capabilities(settings)


@router.post("/model/reload", tags=["status"])
def model_reload() -> dict:
    """Hot-reload thresholds / anomaly model / reference stats / illustration model."""
    from app.services.anomaly_model import reset_anomaly_model_cache
    from app.services.illustration_model import reset_cache as reset_illu
    from app.services.reference_stats import reset_cache as reset_refstats
    from app.services.segmentation import reset_sam_segmenter
    from app.services.thresholds import reset_cache as reset_thresholds

    reset_anomaly_model_cache()
    reset_thresholds()
    reset_refstats()
    reset_illu()
    reset_sam_segmenter()
    return {"status": "reloaded"}


@router.get("/inspection/{inspection_id}", response_model=InspectionDetail)
def get_inspection(
    inspection_id: str, db: Session = Depends(get_db)
) -> InspectionDetail:
    row = db.get(Inspection, inspection_id)
    if row is None:
        raise HTTPException(status_code=404, detail="inspection_id が見つかりません。")

    try:
        issues = [Issue(**i) for i in json.loads(row.issues_json or "[]")]
    except Exception:
        issues = []
    try:
        debug = DebugInfo(**json.loads(row.debug_json or "{}"))
    except Exception:
        debug = None

    feedback = [
        StoredFeedback(
            id=f.id,
            issue_id=f.issue_id,
            feedback=f.feedback,
            image_id=f.image_id,
            garment_type=f.garment_type,
            issue_type=f.issue_type,
            corrected_type=f.corrected_type,
            comment=f.comment,
            created_at=f.created_at.isoformat() if f.created_at else None,
        )
        for f in row.feedback
    ]

    return InspectionDetail(
        inspection_id=row.id,
        image_path=row.image_path,
        image_filename=row.image_filename,
        garment_type=row.garment_type,
        result=row.result,
        overall_score=row.overall_score,
        issues=issues,
        debug=debug,
        feedback=feedback,
        created_at=row.created_at.isoformat() if row.created_at else None,
    )
