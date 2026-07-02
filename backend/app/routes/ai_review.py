"""POST /api/inspection/ai-review — optional GPT visual second opinion.

Loads the stored inspection + its image, crops the most relevant region, and
asks an OpenAI vision model (Responses API, Structured Outputs) to review it.
Requires a server-side API key; cleanly disabled (503) without one.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
from fastapi import APIRouter, Depends, HTTPException
from pydantic import ValidationError
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.models import Inspection
from app.schemas import AIReviewRequest, AIReviewResponse
from app.services import ai_review
from app.services.review_memory import infer_mode

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["ai-review"])


@router.post("/inspection/ai-review", response_model=AIReviewResponse)
def ai_visual_review(
    payload: AIReviewRequest, db: Session = Depends(get_db)
) -> AIReviewResponse:
    available, reason = ai_review.is_available()
    if not available:
        raise HTTPException(
            status_code=503,
            detail=f"AI review is not configured ({reason}). サーバー側の OPENAI_API_KEY を設定してください。",
        )

    inspection = db.get(Inspection, payload.inspection_id)
    if inspection is None:
        raise HTTPException(status_code=404, detail="inspection_id が見つかりません。")

    image_path = Path(inspection.image_path or "")
    image = cv2.imread(str(image_path)) if image_path.exists() else None
    if image is None:
        raise HTTPException(
            status_code=404,
            detail="この検査の画像ファイルが見つかりません（削除された可能性があります）。",
        )

    mode = payload.mode if payload.mode in ("hand", "wrinkle") else infer_mode(inspection)
    region_override = payload.region.model_dump() if payload.region else None

    try:
        parsed = ai_review.run_ai_review(
            inspection=inspection,
            image_bgr=image,
            mode=mode,
            language=payload.language,
            region_override=region_override,
        )
        return AIReviewResponse(**parsed)
    except ai_review.AIReviewUpstreamError as exc:
        raise HTTPException(status_code=502, detail=f"OpenAI 呼び出しに失敗しました: {exc}")
    except (ai_review.AIReviewParseError, ValidationError) as exc:
        raise HTTPException(
            status_code=502, detail=f"AI応答の解析に失敗しました: {str(exc)[:200]}"
        )
