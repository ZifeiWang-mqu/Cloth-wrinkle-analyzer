"""Garment-region selection.

MVP: the user manually selects a rectangle, or we fall back to the whole
image. The function signature and return shape are designed so a future
SAM / Segment-Anything mask can be dropped in without changing callers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GarmentRegion:
    """A clamped, integer pixel bbox plus provenance metadata."""

    x: int
    y: int
    w: int
    h: int
    used_selection: bool  # True if the user provided the region
    source: str  # "user_bbox" | "full_image"

    def as_dict(self) -> dict[str, int]:
        return {"x": self.x, "y": self.y, "w": self.w, "h": self.h}

    def crop(self, image: np.ndarray) -> np.ndarray:
        return image[self.y : self.y + self.h, self.x : self.x + self.w]


def _clamp_bbox(
    x: float, y: float, w: float, h: float, img_w: int, img_h: int
) -> tuple[int, int, int, int]:
    x = int(max(0, min(x, img_w - 1)))
    y = int(max(0, min(y, img_h - 1)))
    w = int(max(1, min(w, img_w - x)))
    h = int(max(1, min(h, img_h - y)))
    return x, y, w, h


def get_garment_region(
    image: np.ndarray, selected_region: dict | None = None
) -> GarmentRegion:
    """Return the region to analyse.

    Parameters
    ----------
    image:
        BGR or grayscale ndarray (H, W[, C]).
    selected_region:
        Optional ``{"x","y","w","h"}`` dict from the frontend. Values are
        clamped to the image bounds so a slightly off selection never crashes
        the pipeline.
    """
    img_h, img_w = image.shape[:2]

    if selected_region:
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
    """Placeholder for SAM/Segment-Anything integration.

    Intentionally not implemented in the MVP. When added, it should return a
    binary mask (and optionally a tight bbox) compatible with the rest of the
    pipeline.
    """
    raise NotImplementedError("SAM segmentation is planned for a later phase.")
