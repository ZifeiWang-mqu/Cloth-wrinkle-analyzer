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
from pathlib import Path

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
# Garment segmentation (SAM with geometric fallback)
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
    device: str | None = None
    model_type: str | None = None
    last_error: str | None = None

    def to_debug(self) -> dict:
        return {
            "enabled": self.enabled,
            "provider": self.provider,
            "mask_available": self.mask_available,
            "mask_area_ratio": round(self.mask_area_ratio, 4),
            "fallback_used": self.fallback_used,
            "reason": self.reason,
            "device": self.device,
            "model_type": self.model_type,
            "last_error": self.last_error,
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


@dataclass
class SegmentResult:
    """Uniform return type for every ``Segmenter.segment()`` implementation.

    ``mask`` is a full-image uint8 0/255 mask (or None when segmentation could
    not produce one); ``reason`` is a short tag describing how the mask was
    derived (e.g. "sam", "user_polygon", "manual_bbox", "whole_image") or why it
    is absent. Having one shape lets ``segment_garment`` treat any segmenter the
    same way without branching on the concrete class.
    """

    mask: np.ndarray | None = None
    reason: str | None = None


class BaseSegmenter:
    name = "base"

    def segment(
        self, image: np.ndarray, region: GarmentRegion, garment_type: str | None = None
    ) -> SegmentResult:
        raise NotImplementedError


class FallbackSegmenter(BaseSegmenter):
    """Geometric fallback: lasso polygon > manual bbox > whole image."""

    name = "opencv"

    def __init__(self, settings):
        self.settings = settings

    def segment(
        self, image: np.ndarray, region: GarmentRegion, garment_type: str | None = None
    ) -> SegmentResult:
        h, w = image.shape[:2]
        if region.polygon and len(region.polygon) >= 3:
            return SegmentResult(_full_polygon_mask((h, w), region.polygon), "user_polygon")
        if region.used_selection:
            return SegmentResult(_full_bbox_mask((h, w), region), "manual_bbox")
        return SegmentResult(np.full((h, w), 255, dtype=np.uint8), "whole_image")


class SamSegmenter(BaseSegmenter):
    """SAM-based segmenter. Optional (torch + segment_anything + checkpoint).

    Lazy-loads the model on first use; supports cpu/cuda (auto-falls back to cpu
    if CUDA is unavailable); can optionally download the checkpoint at runtime.
    Tracks ``last_error`` and ``loaded`` for /api/model/status.
    """

    name = "sam"

    def __init__(self, settings):
        self.settings = settings
        self.model_type = settings.sam_model_type
        self.device = settings.sam_device
        self._predictor = None
        self.loaded = False
        self.last_error: str | None = None

    def available(self) -> bool:
        try:
            import importlib.util

            return (
                importlib.util.find_spec("segment_anything") is not None
                and importlib.util.find_spec("torch") is not None
            )
        except Exception:  # pragma: no cover
            return False

    def ensure_checkpoint(self) -> Path | None:
        cp = self.settings.resolved_sam_checkpoint
        if cp is not None and cp.exists():
            return cp
        if self.settings.sam_auto_download and self.settings.sam_download_url:
            target = cp or (self.settings.sam_dir / Path(self.settings.sam_download_url).name)
            try:
                import urllib.request

                target.parent.mkdir(parents=True, exist_ok=True)
                logger.info("Downloading SAM checkpoint -> %s", target)
                urllib.request.urlretrieve(self.settings.sam_download_url, str(target))
                return target if target.exists() else None
            except Exception as exc:  # pragma: no cover - network dependent
                self.last_error = f"download_failed: {exc}"
                return None
        return None

    def load(self) -> bool:
        if self.loaded:
            return True
        if not self.available():
            self.last_error = "segment_anything/torch not installed"
            return False
        try:
            import torch  # type: ignore
            from segment_anything import SamPredictor, sam_model_registry  # type: ignore

            cp = self.ensure_checkpoint()
            if cp is None or not cp.exists():
                self.last_error = self.last_error or "checkpoint_missing"
                return False
            sam = sam_model_registry[self.model_type](checkpoint=str(cp))
            device = self.device
            if device == "cuda" and not torch.cuda.is_available():
                logger.info("CUDA unavailable; using CPU for SAM.")
                device = "cpu"
            sam.to(device)
            self.device = device
            self._predictor = SamPredictor(sam)
            self.loaded = True
            self.last_error = None
            logger.info("SAM loaded (%s, %s).", self.model_type, device)
            return True
        except Exception as exc:  # pragma: no cover - optional / env dependent
            self.last_error = str(exc)
            logger.warning("SAM load failed: %s", exc)
            return False

    def segment(
        self, image: np.ndarray, region: GarmentRegion, garment_type: str | None = None
    ) -> SegmentResult:
        if not self.load():
            return SegmentResult(mask=None, reason=None)  # last_error holds the cause
        try:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self._predictor.set_image(rgb)
            box = np.array(
                [region.x, region.y, region.x + region.w, region.y + region.h],
                dtype=np.float32,
            )
            masks, scores, _ = self._predictor.predict(box=box, multimask_output=True)
            if masks is None or len(masks) == 0:
                return SegmentResult(mask=None, reason="sam_no_mask")
            best = masks[int(np.argmax(scores))]
            return SegmentResult(mask=(best.astype(np.uint8)) * 255, reason="sam")
        except Exception as exc:  # pragma: no cover - optional / env dependent
            self.last_error = str(exc)
            logger.warning("SAM segmentation failed: %s", exc)
            return SegmentResult(mask=None, reason=None)

    def status(self) -> dict:
        cp = self.settings.resolved_sam_checkpoint
        return {
            "sam_available": self.available(),
            "sam_loaded": self.loaded,
            "checkpoint_path": str(cp)
            if cp
            else (self.settings.sam_checkpoint_path or None),
            "checkpoint_exists": bool(cp and cp.exists()),
            "model_type": self.model_type,
            "device": self.device,
            "last_error": self.last_error,
        }


_SAM_SEGMENTER: SamSegmenter | None = None


def get_sam_segmenter(settings) -> SamSegmenter:
    global _SAM_SEGMENTER
    if _SAM_SEGMENTER is None:
        _SAM_SEGMENTER = SamSegmenter(settings)
    return _SAM_SEGMENTER


def reset_sam_segmenter() -> None:
    global _SAM_SEGMENTER
    _SAM_SEGMENTER = None


def sam_status(settings) -> dict:
    """Status block for /api/model/status (does not force a load)."""
    seg = get_sam_segmenter(settings)
    st = seg.status()
    st.update(
        {
            "enabled": settings.enable_auto_segmentation
            or settings.segmentation_provider == "sam",
            "provider": settings.segmentation_provider,
            "fallback_provider": settings.segmentation_fallback,
        }
    )
    return st


def segment_garment(
    image: np.ndarray,
    region: GarmentRegion,
    settings,
    use_segmentation: bool,
    provider_override: str | None = None,
) -> SegmentationResult:
    """Produce a garment mask (full-image) with graceful fallback.

    Order: requested provider (SAM) -> geometric fallback. A user lasso polygon
    is always honoured (intersected later). SAM masks outside the configured
    area-ratio band trigger a fallback. Never raises.
    """
    if not use_segmentation:
        return SegmentationResult(enabled=False, provider="none")

    h, w = image.shape[:2]
    total = float(h * w)
    provider = (provider_override or settings.segmentation_provider or "opencv").lower()
    result = SegmentationResult(
        enabled=True,
        provider=provider,
        device=settings.sam_device,
        model_type=settings.sam_model_type,
    )

    mask: np.ndarray | None = None

    if provider == "sam":
        seg = get_sam_segmenter(settings)
        sr = seg.segment(image, region)
        result.device = seg.device
        result.last_error = seg.last_error
        mask = sr.mask
        if mask is not None:
            ar = float(np.count_nonzero(mask)) / (total + 1e-9)
            if not (
                settings.segmentation_min_area_ratio
                <= ar
                <= settings.segmentation_max_area_ratio
            ):
                logger.info("SAM mask area ratio %.3f out of band; falling back.", ar)
                mask = None
                result.reason = "sam_area_out_of_band"
        if mask is not None:
            result.provider = "sam"
        else:
            result.fallback_used = True
            result.reason = result.reason or (seg.last_error or "sam_unavailable")

    if mask is None:
        sr = FallbackSegmenter(settings).segment(image, region)
        mask = sr.mask
        result.provider = "opencv"
        result.reason = result.reason or sr.reason
        if provider == "sam":
            result.fallback_used = True

    if mask is None:
        result.mask_available = False
        result.reason = result.reason or "no_mask"
        return result

    result.mask = mask
    result.mask_available = True
    result.mask_area_ratio = float(np.count_nonzero(mask)) / (total + 1e-9)
    return result


# --- Compatibility wrapper -------------------------------------------------- #
def segment_with_sam(
    image: np.ndarray, region: GarmentRegion | None = None, garment_type=None
) -> np.ndarray | None:  # noqa: D401
    """Run SAM directly (returns a full-image mask or None)."""
    from app.settings import settings as _settings

    if region is None:
        h, w = image.shape[:2]
        region = GarmentRegion(0, 0, w, h, used_selection=False, source="full_image")
    return get_sam_segmenter(_settings).segment(image, region).mask
