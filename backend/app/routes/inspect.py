"""Inspection endpoints.

POST /api/inspect-wrinkle   — multipart file upload (web UI)
POST /api/inspect-base64    — base64 / data URL (Photoshop & external tools)

Both share :func:`app.services.inspection_service.run_inspection`.
"""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.schemas import GarmentType, InspectBase64Request, InspectResponse
from app.services import image_io
from app.services.inspection_service import run_inspection
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
    use_segmentation: bool = Form(True),
    segmentation_provider: str | None = Form(None),
    use_pose_advanced: bool = Form(True),
    use_illustration_model: bool = Form(True),
    return_debug_overlays: bool = Form(False),
    db: Session = Depends(get_db),
) -> InspectResponse:
    garment = _normalize_garment(garment_type)
    region_dict = _parse_region(selected_region)

    raw = await image.read()
    try:
        saved_path, _safe_name, img = image_io.save_upload(raw, image.filename, settings)
    except image_io.UploadError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return run_inspection(
        image_bgr=img,
        garment=garment,
        region_dict=region_dict,
        db=db,
        saved_path=saved_path,
        image_filename=image.filename,
        source="web",
        use_segmentation=use_segmentation,
        segmentation_provider=segmentation_provider,
        use_pose_advanced=use_pose_advanced,
        use_illustration_model=use_illustration_model,
        return_debug_overlays=return_debug_overlays,
    )


@router.post("/inspect-base64", response_model=InspectResponse)
def inspect_base64(
    payload: InspectBase64Request, db: Session = Depends(get_db)
) -> InspectResponse:
    """Inspect a base64-encoded image (same result shape as the upload route)."""
    garment = _normalize_garment(payload.garment_type)
    region_dict = payload.selected_region or None  # polygon or rect dict
    try:
        raw, ext = image_io.decode_base64(payload.image_base64)
        img = image_io.decode_image(raw)  # validate before persisting
        saved_path = image_io.persist_bytes(raw, settings, ext)
    except image_io.UploadError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return run_inspection(
        image_bgr=img,
        garment=garment,
        region_dict=region_dict,
        db=db,
        saved_path=saved_path,
        image_filename=f"base64{ext}",
        source=payload.source or "external",
        use_segmentation=payload.use_segmentation,
        segmentation_provider=payload.segmentation_provider,
        use_pose_advanced=payload.use_pose_advanced,
        use_illustration_model=payload.use_illustration_model,
        return_debug_overlays=payload.return_debug_overlays,
    )
