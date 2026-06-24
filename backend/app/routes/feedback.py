"""POST /api/feedback — store user corrections for future training data.

Captures rich context (image_id, garment_type, issue_type, original/corrected
bbox, confidence, severity, source, comment) so the data can later seed an
illustration-specific dataset. Supports manually-added ``missed_issue`` entries.
"""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.models import Inspection, IssueFeedback
from app.schemas import FeedbackRequest, FeedbackResponse

router = APIRouter(prefix="/api", tags=["feedback"])


@router.post("/feedback", response_model=FeedbackResponse)
def submit_feedback(
    payload: FeedbackRequest, db: Session = Depends(get_db)
) -> FeedbackResponse:
    inspection = db.get(Inspection, payload.inspection_id)
    if inspection is None:
        raise HTTPException(status_code=404, detail="inspection_id が見つかりません。")

    # Backfill context from the inspection when the client didn't supply it.
    image_id = payload.image_id or inspection.image_filename
    garment_type = payload.garment_type or inspection.garment_type

    row = IssueFeedback(
        inspection_id=payload.inspection_id,
        issue_id=payload.issue_id,
        feedback=payload.feedback.value,
        image_id=image_id,
        garment_type=garment_type,
        issue_type=payload.issue_type,
        original_bbox_json=(
            json.dumps(payload.original_bbox.model_dump())
            if payload.original_bbox
            else None
        ),
        corrected_bbox_json=(
            json.dumps(payload.corrected_bbox.model_dump())
            if payload.corrected_bbox
            else None
        ),
        corrected_type=payload.corrected_type,
        confidence=payload.confidence,
        severity=payload.severity,
        source=payload.source,
        comment=payload.comment,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return FeedbackResponse(status="saved", feedback_id=row.id)
