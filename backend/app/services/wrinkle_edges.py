"""Wrinkle-candidate extraction.

Pipeline (MVP, classic CV — no learning required):
    crop -> grayscale -> denoise -> CLAHE contrast -> Canny -> HoughLinesP

Each detected segment is a *wrinkle candidate*. Coordinates are expressed in
the cropped region's local frame; callers add the region offset when drawing
on the full image. Skeletonization is available via scikit-image when present
and is used to refine the edge map, but failures fall back to raw Canny.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from app.services.segmentation import GarmentRegion

logger = logging.getLogger(__name__)


@dataclass
class WrinkleLine:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def length(self) -> float:
        return math.hypot(self.x2 - self.x1, self.y2 - self.y1)

    @property
    def angle_deg(self) -> float:
        """Axial angle in [0, 180). 0 = horizontal, 90 = vertical."""
        ang = math.degrees(math.atan2(self.y2 - self.y1, self.x2 - self.x1))
        if ang < 0:
            ang += 180.0
        return ang

    @property
    def midpoint(self) -> tuple[float, float]:
        return (self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0

    def as_dict(self) -> dict[str, float]:
        return {
            "x1": round(self.x1, 1),
            "y1": round(self.y1, 1),
            "x2": round(self.x2, 1),
            "y2": round(self.y2, 1),
            "angle": round(self.angle_deg, 1),
            "length": round(self.length, 1),
        }


@dataclass
class WrinkleCandidates:
    lines: list[WrinkleLine] = field(default_factory=list)
    edge_map: np.ndarray | None = None  # uint8 (crop frame, polygon-masked)
    gray: np.ndarray | None = None  # preprocessed grayscale crop
    crop: np.ndarray | None = None  # BGR crop
    offset: tuple[int, int] = (0, 0)  # (x, y) of crop within full image
    crop_shape: tuple[int, int] = (0, 0)  # (h, w)
    # Effective analysed area in px: polygon mask area when a lasso was used,
    # otherwise the crop area. Used for density normalisation.
    area_px: float = 0.0
    mask: np.ndarray | None = None  # uint8 0/255 combined mask (crop frame)
    line_filter: dict[str, int] = field(default_factory=dict)  # raw/kept/removed_*

    @property
    def count(self) -> int:
        return len(self.lines)


def _preprocess(crop_bgr: np.ndarray) -> np.ndarray:
    """Grayscale + denoise + contrast normalization."""
    if crop_bgr.ndim == 3:
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = crop_bgr
    # Edge-preserving denoise keeps wrinkle ridges while killing texture noise.
    gray = cv2.bilateralFilter(gray, d=5, sigmaColor=50, sigmaSpace=50)
    # Local contrast normalization (CLAHE) makes faint line art detectable.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    return gray


def _auto_canny(gray: np.ndarray, sigma: float = 0.33) -> np.ndarray:
    """Canny with thresholds derived from the image median (robust default)."""
    v = float(np.median(gray))
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    if upper <= lower:
        lower, upper = 50, 150
    return cv2.Canny(gray, lower, upper)


def _combine_masks(
    poly_mask: np.ndarray | None, seg_crop: np.ndarray | None
) -> np.ndarray | None:
    if poly_mask is not None and seg_crop is not None:
        return cv2.bitwise_and(poly_mask, seg_crop)
    return poly_mask if poly_mask is not None else seg_crop


def _filter_lines(
    lines: list[WrinkleLine],
    mask: np.ndarray | None,
    crop_shape: tuple[int, int],
    settings,
    min_len: float,
) -> tuple[list[WrinkleLine], dict[str, int]]:
    """Drop lines that are likely NOT wrinkles (outline/edge/contour).

    Returns the kept lines and a ``line_filter`` debug dict with raw/kept and
    per-reason removed counts.
    """
    h, w = crop_shape
    diag = math.hypot(h, w)
    lf = {
        "raw_lines": len(lines),
        "kept_lines": 0,
        "removed_outside_mask": 0,
        "removed_near_boundary": 0,
        "removed_touches_edge": 0,
        "removed_too_long": 0,
        "removed_too_short": 0,
    }
    if not lines:
        return [], lf

    dist = None
    margin = 0.0
    if mask is not None:
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
        margin = float(getattr(settings, "mask_boundary_margin_ratio", 0.0) or 0.0) * diag
    max_len = 0.9 * diag
    short_len = 0.4 * float(min_len)

    kept: list[WrinkleLine] = []
    for ln in lines:
        if ln.length < short_len:
            lf["removed_too_short"] += 1
            continue
        mx, my = ln.midpoint
        mxi = int(min(max(0, mx), w - 1))
        myi = int(min(max(0, my), h - 1))
        if mask is not None and mask[myi, mxi] == 0:
            lf["removed_outside_mask"] += 1
            continue
        if dist is not None and margin > 0 and dist[myi, mxi] < margin:
            lf["removed_near_boundary"] += 1
            continue
        if (
            min(ln.x1, ln.x2) <= 1
            or min(ln.y1, ln.y2) <= 1
            or max(ln.x1, ln.x2) >= w - 2
            or max(ln.y1, ln.y2) >= h - 2
        ):
            lf["removed_touches_edge"] += 1
            continue
        if ln.length > max_len:
            lf["removed_too_long"] += 1
            continue
        kept.append(ln)
    lf["kept_lines"] = len(kept)
    return kept, lf


def extract_wrinkle_candidates(
    image: np.ndarray,
    region: GarmentRegion,
    seg_mask: np.ndarray | None = None,
    settings=None,
) -> WrinkleCandidates:
    """Detect candidate wrinkle line segments inside ``region``.

    ``seg_mask`` (full-image uint8) restricts analysis to a garment mask
    (from SAM/OpenCV); it is intersected with any lasso polygon. Lines outside
    the mask, near the mask boundary, touching the crop border, or implausibly
    long are filtered out. Never raises — returns an empty set on failure.
    """
    try:
        crop = region.crop(image)
        if crop.size == 0:
            return WrinkleCandidates(offset=(region.x, region.y))

        gray = _preprocess(crop)
        edges = _auto_canny(gray)

        h, w = gray.shape[:2]

        # Combined mask = lasso polygon AND segmentation mask (either may be None).
        poly_mask = region.crop_mask((h, w))
        seg_crop = None
        if seg_mask is not None:
            seg_crop = seg_mask[region.y : region.y + h, region.x : region.x + w]
            if seg_crop.shape[:2] != (h, w):
                seg_crop = cv2.resize(seg_crop, (w, h), interpolation=cv2.INTER_NEAREST)
        mask = _combine_masks(poly_mask, seg_crop)
        if mask is not None:
            edges = cv2.bitwise_and(edges, edges, mask=mask)

        # Optional skeletonization to thin thick strokes before Hough.
        edge_for_hough = edges
        try:
            from skimage.morphology import skeletonize  # type: ignore

            skel = skeletonize(edges > 0)
            edge_for_hough = (skel.astype(np.uint8)) * 255
        except Exception:  # pragma: no cover - skimage optional / edge cases
            pass
        min_len = max(12, int(0.06 * min(h, w)))
        raw = cv2.HoughLinesP(
            edge_for_hough,
            rho=1,
            theta=np.pi / 180.0,
            threshold=30,
            minLineLength=min_len,
            maxLineGap=6,
        )

        lines: list[WrinkleLine] = []
        if raw is not None:
            for seg in raw[:, 0, :]:
                x1, y1, x2, y2 = (float(v) for v in seg)
                lines.append(WrinkleLine(x1, y1, x2, y2))

        lines, line_filter = _filter_lines(lines, mask, (h, w), settings, min_len)

        area_px = float(np.count_nonzero(mask)) if mask is not None else float(h * w)
        return WrinkleCandidates(
            lines=lines,
            edge_map=edges,
            gray=gray,
            crop=crop,
            offset=(region.x, region.y),
            crop_shape=(h, w),
            area_px=area_px,
            mask=mask,
            line_filter=line_filter,
        )

    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Wrinkle extraction failed: %s", exc)
        return WrinkleCandidates(offset=(region.x, region.y))
