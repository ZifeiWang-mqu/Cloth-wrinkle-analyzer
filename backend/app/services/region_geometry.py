"""Typed geometry of the analysed region.

Replaces the former untyped ``region_context`` dict that was threaded through
scoring and evidence generation, so callers get attribute access + type
checking (``geometry.region_diag``) instead of ``.get("region_diag", default)``
scattered across modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RegionGeometry:
    """Geometry of the region under inspection, in original-image pixels.

    ``bbox`` is the analysed region's bounding box as ``{"x","y","w","h"}``;
    ``image_w``/``image_h`` are the full image dimensions; ``region_diag`` is the
    region's diagonal length (used to normalise line-length thresholds).
    """

    bbox: dict[str, float] = field(default_factory=lambda: {"x": 0, "y": 0, "w": 1, "h": 1})
    image_w: int = 0
    image_h: int = 0
    region_diag: float = 0.0
