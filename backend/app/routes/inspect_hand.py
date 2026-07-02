"""POST /api/inspect-hand — hand-unnaturalness inspection (MVP).

Separate from the wrinkle endpoints on purpose: the existing
``/api/inspect-wrinkle`` contract stays untouched. Accepts the same multipart
shape (image + optional ``selected_region`` lasso/bbox JSON used as a hand
region hint) and returns the same ``InspectResponse`` schema with
``category="hand"`` issues.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.routes.inspect import _parse_region
from app.schemas import InspectResponse
from app.services import image_io
from app.services.hand_inspection_service import run_hand_inspection
from app.settings import settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["inspect-hand"])


@router.post("/inspect-hand", response_model=InspectResponse)
async def inspect_hand(
    image: UploadFile = File(...),
    selected_region: str | None = Form(None),
    return_debug_overlays: bool = Form(False),
    db: Session = Depends(get_db),
) -> InspectResponse:
    region_dict = _parse_region(selected_region)

    raw = await image.read()
    try:
        saved_path, _safe_name, img = image_io.save_upload(raw, image.filename, settings)
    except image_io.UploadError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return run_hand_inspection(
        image_bgr=img,
        region_dict=region_dict,
        db=db,
        saved_path=saved_path,
        image_filename=image.filename,
        source="web",
        return_debug_overlays=return_debug_overlays,
    )
