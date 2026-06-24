"""Garment-region selection.

MVP: the user selects an area either as a freehand **lasso (polygon)** or a
rectangle, or we fall back to the whole image. A polygon additionally yields a
binary mask so downstream wrinkle extraction only considers lines INSIDE the
drawn shape. The function signature and return shape are designed so a future
SAM / Segment-Anything mask can be dropped in without changing callers.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


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


# --- Future hook ----------------------------------------------------------- #
def segment_with_sam(image: np.ndarray, selected_region: dict | None = None):  # noqa: D401
    """Placeholder for SAM/Segment-Anything integration (later phase)."""
    raise NotImplementedError("SAM segmentation is planned for a later phase.")
