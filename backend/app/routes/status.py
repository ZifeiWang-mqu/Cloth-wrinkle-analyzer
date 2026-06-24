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
from app.services.anomaly_model import describe_anomaly_model
from app.services.reference_stats import load_density_stats
from app.services.thresholds import load_thresholds
from app.settings import settings

router = APIRouter(prefix="/api", tags=["status"])


@router.get("/model/status", response_model=ModelStatus)
def model_status() -> ModelStatus:
    desc = describe_anomaly_model()
    th = load_thresholds(settings.thresholds_path)
    _, rs_loaded = load_density_stats(settings.reference_stats_path)
    return ModelStatus(
        model_loaded=bool(desc["loaded"]),
        model_type=str(desc["type"]),
        model_path=str(settings.model_path),
        reference_stats_loaded=bool(rs_loaded),
        reference_stats_path=str(settings.reference_stats_path),
        thresholds_loaded=th.loaded,
        thresholds_path=str(settings.thresholds_path),
        available_garment_models=list(desc["garments"]),
        version=__version__,
    )


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
