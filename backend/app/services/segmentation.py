"""Garment-region selection.

MVP: the user selects an area either as a freehand **lasso (polygon)** or a
rectangle, or we fall back to the whole image. A polygon additionally yields a
binary mask so downstream wrinkle extraction only considers lines INSIDE the
drawn shape. The function signature and return shape are designed so a future
SAM / Segment-Anything mask can be dropped in without changing callers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class GarmentRegion:
    """A clamped integer pixel bbox, optional polygon, plus provenance."""

    x: int
    y: int
    w: int
    h: int
    used_selection: bool  # True if the user provided the region
    source: str  # "user_polygon" | "user_bbox" | "full_image"
    # Polygon vertices in FULL-image pixel coords (None for rect / whole image).
    polygon: list[tuple[float, float]] | None = field(default=None)

    def as_dict(self) -> dict[str, int]:
        return {"x": self.x, "y": self.y, "w": self.w, "h": self.h}

    def crop(self, image: np.ndarray) -> np.ndarray:
        return image[self.y : self.y + self.h, self.x : self.x + self.w]

    def crop_mask(self, crop_shape: tuple[int, int]) -> np.ndarray | None:
        """Binary mask (uint8 0/255) for the polygon in the cropped frame.

        Returns None when there is no polygon (rect / whole image).
        """
        if not self.polygon or len(self.polygon) < 3:
            return None
        h, w = crop_shape
        mask = np.zeros((h, w), dtype=np.uint8)
        pts = np.array(
            [[int(round(px - self.x)), int(round(py - self.y))] for px, py in self.polygon],
            dtype=np.int32,
        )
        cv2.fillPoly(mask, [pts], 255)
        return mask


def _clamp_bbox(
    x: float, y: float, w: float, h: float, img_w: int, img_h: int
) -> tuple[int, int, int, int]:
    x = int(max(0, min(x, img_w - 1)))
    y = int(max(0, min(y, img_h - 1)))
    w = int(max(1, min(w, img_w - x)))
    h = int(max(1, min(h, img_h - y)))
    return x, y, w, h


def _polygon_region(points: list, img_w: int, img_h: int) -> GarmentRegion | None:
    """Build a GarmentRegion from polygon points ``[[x, y], ...]``."""
    clean: list[tuple[float, float]] = []
    for p in points:
        try:
            px = float(max(0, min(p[0], img_w - 1)))
            py = float(max(0, min(p[1], img_h - 1)))
            clean.append((px, py))
        except (TypeError, IndexError, ValueError):
            continue
    if len(clean) < 3:
        return None
    xs = [p[0] for p in clean]
    ys = [p[1] for p in clean]
    x, y, w, h = _clamp_bbox(
        min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys), img_w, img_h
    )
    return GarmentRegion(
        x, y, w, h, used_selection=True, source="user_polygon", polygon=clean
    )


def get_garment_region(
    image: np.ndarray, selected_region: dict | None = None
) -> GarmentRegion:
    """Return the region to analyse.

    ``selected_region`` may be:
      - ``{"points": [[x, y], ...]}``  -> freehand lasso (polygon), or
      - ``{"x", "y", "w", "h"}``       -> rectangle (legacy), or
      - ``None`` / empty               -> whole image.
    All coordinates are clamped to the image bounds so a slightly off selection
    never crashes the pipeline.
    """
    img_h, img_w = image.shape[:2]

    if selected_region:
        points = selected_region.get("points")
        if isinstance(points, list) and len(points) >= 3:
            region = _polygon_region(points, img_w, img_h)
            if region is not None:
                return region

        if all(k in selected_region for k in ("x", "y", "w", "h")):
            x, y, w, h = _clamp_bbox(
                float(selected_region.get("x", 0)),
                float(selected_region.get("y", 0)),
                float(selected_region.get("w", img_w)),
                float(selected_region.get("h", img_h)),
                img_w,
                img_h,
            )
            return GarmentRegion(x, y, w, h, used_selection=True, source="user_bbox")

    return GarmentRegion(0, 0, img_w, img_h, used_selection=False, source="full_image")


# --------------------------------------------------------------------------- #
# Garment segmentation (SAM with OpenCV fallback)
# --------------------------------------------------------------------------- #
@dataclass
class SegmentationResult:
    enabled: bool = False
    provider: str = "none"  # "sam" | "opencv" | "none"
    mask: np.ndarray | None = None  # full-image uint8 0/255
    mask_available: bool = False
    mask_area_ratio: float = 0.0
    fallback_used: bool = False
    reason: str | None = None

    def to_debug(self) -> dict:
        return {
            "enabled": self.enabled,
            "provider": self.provider,
            "mask_available": self.mask_available,
            "mask_area_ratio": round(self.mask_area_ratio, 4),
            "fallback_used": self.fallback_used,
            "reason": self.reason,
        }


def _full_polygon_mask(shape: tuple[int, int], polygon) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)
    pts = np.array([[int(round(px)), int(round(py))] for px, py in polygon], np.int32)
    cv2.fillPoly(mask, [pts], 255)
    return mask


def _full_bbox_mask(shape: tuple[int, int], region: GarmentRegion) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[region.y : region.y + region.h, region.x : region.x + region.w] = 255
    return mask


# Module-level SAM predictor cache (keyed by checkpoint path).
_SAM_PREDICTOR: dict[str, object] = {}


def _sam_mask(image: np.ndarray, region: GarmentRegion, settings) -> np.ndarray | None:
    """Run SAM with a bbox prompt. Returns a full-image uint8 mask or None.

    Optional: requires `segment_anything` + `torch` + a checkpoint. Any failure
    returns None so the caller can fall back.
    """
    try:
        checkpoint = settings.resolved_sam_checkpoint
        if checkpoint is None or not checkpoint.exists():
            return None
        from segment_anything import SamPredictor, sam_model_registry  # type: ignore

        key = str(checkpoint)
        predictor = _SAM_PREDICTOR.get(key)
        if predictor is None:
            sam = sam_model_registry[settings.sam_model_type](checkpoint=str(checkpoint))
            predictor = SamPredictor(sam)
            _SAM_PREDICTOR[key] = predictor

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        predictor.set_image(rgb)
        box = np.array(
            [region.x, region.y, region.x + region.w, region.y + region.h],
            dtype=np.float32,
        )
        masks, scores, _ = predictor.predict(box=box, multimask_output=True)
        if masks is None or len(masks) == 0:
            return None
        best = masks[int(np.argmax(scores))]
        return (best.astype(np.uint8)) * 255
    except Exception as exc:  # pragma: no cover - optional / env dependent
        logger.warning("SAM segmentation failed: %s", exc)
        return None


def segment_garment(
    image: np.ndarray,
    region: GarmentRegion,
    settings,
    use_segmentation: bool,
    provider_override: str | None = None,
) -> SegmentationResult:
    """Produce a garment mask (full-image) with graceful fallback.

    Order: requested provider -> opencv geometric fallback. A user lasso polygon
    is always honoured (intersected later). Masks outside the configured area
    ratio band trigger a fallback.
    """
    if not use_segmentation:
        return SegmentationResult(enabled=False, provider="none")

    h, w = image.shape[:2]
    total = float(h * w)
    provider = (provider_override or settings.segmentation_provider or "opencv").lower()
    result = SegmentationResult(enabled=True, provider=provider)

    mask: np.ndarray | None = None

    # Try SAM first if requested.
    if provider == "sam":
        mask = _sam_mask(image, region, settings)
        if mask is None:
            result.fallback_used = True
            result.reason = "sam_unavailable_or_failed"
            provider = "opencv"
            result.provider = "opencv"

    # OpenCV geometric fallback (or explicit opencv provider).
    if mask is None and provider in ("opencv", "none"):
        if region.polygon and len(region.polygon) >= 3:
            mask = _full_polygon_mask((h, w), region.polygon)
            result.reason = result.reason or "user_polygon"
        elif region.used_selection:
            mask = _full_bbox_mask((h, w), region)
            result.fallback_used = True
            result.reason = result.reason or "bbox_fallback"
        else:
            mask = np.full((h, w), 255, dtype=np.uint8)
            result.fallback_used = True
            result.reason = result.reason or "whole_image_fallback"

    if mask is None:
        result.mask_available = False
        result.reason = result.reason or "no_mask"
        return result

    area_ratio = float(np.count_nonzero(mask)) / (total + 1e-9)

    # If a *real* (SAM) mask is implausibly small/large, fall back to bbox/whole.
    if result.provider == "sam" and not (
        settings.segmentation_min_area_ratio
        <= area_ratio
        <= settings.segmentation_max_area_ratio
    ):
        logger.info("SAM mask area ratio %.3f out of band; falling back.", area_ratio)
        if region.used_selection:
            mask = _full_bbox_mask((h, w), region)
        else:
            mask = np.full((h, w), 255, dtype=np.uint8)
        result.provider = "opencv"
        result.fallback_used = True
        result.reason = "sam_area_out_of_band"
        area_ratio = float(np.count_nonzero(mask)) / (total + 1e-9)

    result.mask = mask
    result.mask_available = True
    result.mask_area_ratio = area_ratio
    return result


# --- Future hook ----------------------------------------------------------- #
def segment_with_sam(image: np.ndarray, region: GarmentRegion | None = None, garment_type=None):  # noqa: D401
    """Thin wrapper kept for API compatibility / future FastSAM/SAM2 swap."""
    from app.settings import settings as _settings

    if region is None:
        h, w = image.shape[:2]
        region = GarmentRegion(0, 0, w, h, used_selection=False, source="full_image")
    return _sam_mask(image, region, _settings)
