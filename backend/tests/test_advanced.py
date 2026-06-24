"""Unit tests for segmentation fallback, joint geometry, and capabilities."""

from __future__ import annotations

import numpy as np

from app.services import capabilities
from app.services.pose import (
    JointContext,
    classify_side_of_joint,
    compute_joint_angle,
    estimate_joint_bend_direction,
)
from app.services.segmentation import GarmentRegion, segment_garment
from app.services.wrinkle_edges import extract_wrinkle_candidates
from app.settings import settings


# --- Joint geometry --------------------------------------------------------- #
def test_joint_angle_straight_and_right():
    # Straight line -> 180 degrees.
    assert abs(compute_joint_angle((0, 0), (1, 0), (2, 0)) - 180.0) < 1e-6
    # Right angle at b.
    assert abs(compute_joint_angle((0, 1), (0, 0), (1, 0)) - 90.0) < 1e-6


def test_bend_direction_and_side_classification():
    a, b, c = (0.0, -1.0), (0.0, 0.0), (1.0, 0.0)  # 90-degree bend
    comp = estimate_joint_bend_direction(a, b, c)
    ctx = JointContext(
        joint_name="left_elbow",
        joint_type="elbow",
        proximal=a,
        joint=b,
        distal=c,
        angle_degrees=90.0,
        bend_strength=1.0,
        compression_vector=comp,
        stretch_vector=(-comp[0], -comp[1]),
        confidence=0.9,
    )
    # A point along +compression vector is "compression"; opposite is "stretch".
    p_comp = (b[0] + comp[0] * 5, b[1] + comp[1] * 5)
    p_str = (b[0] - comp[0] * 5, b[1] - comp[1] * 5)
    assert classify_side_of_joint(p_comp, ctx) == "compression"
    assert classify_side_of_joint(p_str, ctx) == "stretch"


# --- Segmentation fallback -------------------------------------------------- #
def _img() -> np.ndarray:
    return np.full((200, 200, 3), 220, dtype=np.uint8)


def test_segment_disabled():
    img = _img()
    region = GarmentRegion(0, 0, 200, 200, used_selection=False, source="full_image")
    res = segment_garment(img, region, settings, use_segmentation=False)
    assert res.enabled is False
    assert res.mask is None


def test_segment_opencv_fallback_whole_image():
    img = _img()
    region = GarmentRegion(0, 0, 200, 200, used_selection=False, source="full_image")
    res = segment_garment(
        img, region, settings, use_segmentation=True, provider_override="opencv"
    )
    assert res.enabled is True
    assert res.mask_available is True
    assert res.mask is not None
    assert res.mask_area_ratio > 0.9  # whole image


def test_seg_mask_restricts_lines():
    import cv2

    img = np.full((200, 200, 3), 235, dtype=np.uint8)
    # Lines across the whole image.
    for y in range(20, 180, 14):
        cv2.line(img, (5, y), (195, y), (40, 40, 40), 2)

    region = GarmentRegion(0, 0, 200, 200, used_selection=False, source="full_image")
    # Mask only the left half.
    mask = np.zeros((200, 200), dtype=np.uint8)
    mask[:, :100] = 255

    masked = extract_wrinkle_candidates(img, region, seg_mask=mask, settings=settings)
    unmasked = extract_wrinkle_candidates(img, region, seg_mask=None, settings=settings)
    # Restricting to half the image should not yield more lines than the full one.
    assert masked.count <= unmasked.count
    # All kept line midpoints must fall inside the mask.
    for ln in masked.lines:
        mx, my = ln.midpoint
        assert mask[int(my), int(mx)] > 0


# --- Capabilities ----------------------------------------------------------- #
def test_capabilities_structure():
    caps = capabilities.get_capabilities(settings)
    assert "segmentation" in caps
    assert "pose" in caps
    assert "illustration_model" in caps
    assert caps["segmentation"]["opencv_available"] is True
