"""POST /api/feedback — store user corrections for future training data."""

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

    row = IssueFeedback(
        inspection_id=payload.inspection_id,
        issue_id=payload.issue_id,
        feedback=payload.feedback.value,
        corrected_bbox_json=(
            json.dumps(payload.corrected_bbox.model_dump())
            if payload.corrected_bbox
            else None
        ),
        corrected_type=payload.corrected_type,
        comment=payload.comment,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return FeedbackResponse(status="saved", feedback_id=row.id)
