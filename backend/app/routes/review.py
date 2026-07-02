"""Review-memory endpoints.

POST /api/review                 — save a per-inspection review (snapshot built
                                   server-side from the stored Inspection row)
POST /api/review-memory/search   — retrieve similar reviewed cases

No external API calls; retrieval uses the local deterministic embedder.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.models import Inspection
from app.schemas import (
    MemoryCase,
    MemorySearchRequest,
    MemorySearchResponse,
    ReviewRequest,
    ReviewResponse,
)
from app.services import review_memory

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["review"])


@router.post("/review", response_model=ReviewResponse)
def save_review(payload: ReviewRequest, db: Session = Depends(get_db)) -> ReviewResponse:
    inspection = db.get(Inspection, payload.inspection_id)
    if inspection is None:
        raise HTTPException(status_code=404, detail="inspection_id が見つかりません。")

    review, memory = review_memory.save_review(db, inspection, payload)
    return ReviewResponse(
        status="saved",
        review_id=review.id,
        memory_id=memory.id,
        mode=review.mode,
        summary_text=memory.summary_text,
    )


@router.post("/review-memory/search", response_model=MemorySearchResponse)
def search_review_memory(
    payload: MemorySearchRequest, db: Session = Depends(get_db)
) -> MemorySearchResponse:
    mode = payload.mode
    if payload.inspection_id:
        inspection = db.get(Inspection, payload.inspection_id)
        if inspection is None:
            raise HTTPException(status_code=404, detail="inspection_id が見つかりません。")
        snapshot = review_memory.build_snapshot(inspection)
        query_text = review_memory.build_case_summary(snapshot)  # neutral query flavor
        if mode is None:
            mode = snapshot["mode"]  # same-mode cases by default
    elif payload.query_text and payload.query_text.strip():
        query_text = payload.query_text.strip()
    else:
        raise HTTPException(
            status_code=400, detail="inspection_id か query_text のいずれかが必要です。"
        )

    cases = review_memory.search_memory(
        db, query_text, mode=mode, verdict=payload.verdict, top_k=payload.top_k
    )
    return MemorySearchResponse(
        query_summary=query_text,
        cases=[MemoryCase(**c) for c in cases],
    )
