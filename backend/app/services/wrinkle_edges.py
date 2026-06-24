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
    edge_map: np.ndarray | None = None  # uint8 (crop frame)
    gray: np.ndarray | None = None  # preprocessed grayscale crop
    crop: np.ndarray | None = None  # BGR crop
    offset: tuple[int, int] = (0, 0)  # (x, y) of crop within full image
    crop_shape: tuple[int, int] = (0, 0)  # (h, w)

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


def extract_wrinkle_candidates(
    image: np.ndarray, region: GarmentRegion
) -> WrinkleCandidates:
    """Detect candidate wrinkle line segments inside ``region``.

    Never raises on degenerate input — returns an empty candidate set instead.
    """
    try:
        crop = region.crop(image)
        if crop.size == 0:
            return WrinkleCandidates(offset=(region.x, region.y))

        gray = _preprocess(crop)
        edges = _auto_canny(gray)

        # Optional skeletonization to thin thick strokes before Hough.
        edge_for_hough = edges
        try:
            from skimage.morphology import skeletonize  # type: ignore

            skel = skeletonize(edges > 0)
            edge_for_hough = (skel.astype(np.uint8)) * 255
        except Exception:  # pragma: no cover - skimage optional / edge cases
            pass

        h, w = gray.shape[:2]
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

        return WrinkleCandidates(
            lines=lines,
            edge_map=edges,
            gray=gray,
            crop=crop,
            offset=(region.x, region.y),
            crop_shape=(h, w),
        )

    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Wrinkle extraction failed: %s", exc)
        return WrinkleCandidates(offset=(region.x, region.y))
