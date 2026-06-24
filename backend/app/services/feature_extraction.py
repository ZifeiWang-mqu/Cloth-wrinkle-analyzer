"""Structural feature extraction from wrinkle candidates.

We deliberately extract *structural* features (line angle, density, convergence,
gradient orientation, contrast) rather than colour/texture, because the goal is
to compare illustrations against photo-derived statistics across a large domain
gap (real fabric vs. line art). See requirements section 2.

All features degrade gracefully: with no lines, sensible neutral defaults are
returned so the rule engine still runs.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import cv2
import numpy as np

from app.services.pose import PoseResult
from app.services.wrinkle_edges import WrinkleCandidates, WrinkleLine

# Image convention: +y points DOWN, so gravity is the vertical axis (90 deg).
GRAVITY_ANGLE_DEG = 90.0


@dataclass
class Features:
    # Counts / density
    num_lines: int = 0
    area_px: float = 0.0
    edge_density: float = 0.0  # edge pixels / area
    line_count_density: float = 0.0  # lines per 10k px
    line_length_sum: float = 0.0
    mean_line_length: float = 0.0

    # Orientation (axial, degrees in [0,180); 0=horizontal, 90=vertical)
    dominant_line_angle: float = GRAVITY_ANGLE_DEG
    orientation_dispersion: float = 1.0  # 0 aligned, 1 isotropic
    vertical_fraction: float = 0.0  # length share within +-30deg of vertical
    horizontal_fraction: float = 0.0  # length share within +-30deg of horizontal
    gravity_angle: float = GRAVITY_ANGLE_DEG
    angle_diff_from_gravity: float = 0.0  # |dominant - gravity|, 0..90

    # Convergence / radiality
    convergence_strength: float = 0.0  # 0..1
    convergence_point: tuple[float, float] | None = None  # full-image coords

    # Shadow / light
    gradient_orientation: float = 0.0  # axial deg
    gradient_coherence: float = 0.0  # 0..1
    estimated_light_angle: float = 0.0  # full 360 deg, dir of increasing bright
    shadow_wrinkle_perpendicularity: float = 0.0  # 0 parallel .. 1 perpendicular
    local_contrast: float = 0.0  # std/255

    # Patch density (for line-density rule)
    patch_density_std: float = 0.0
    patch_density_max_z: float = 0.0

    # Pose-derived context
    pose_detected: bool = False
    nearest_joint: str | None = None
    nearest_joint_dist_norm: float | None = None  # dist / region diag

    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        # tuples -> lists for JSON friendliness; round floats lightly
        if self.convergence_point is not None:
            d["convergence_point"] = [
                round(self.convergence_point[0], 1),
                round(self.convergence_point[1], 1),
            ]
        for k, v in d.items():
            if isinstance(v, float):
                d[k] = round(v, 4)
        return d


# --------------------------------------------------------------------------- #
# Circular (axial) statistics helpers
# --------------------------------------------------------------------------- #
def _axial_mean(angles_deg: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    """Return (mean_axial_angle_deg, resultant_length R) for period-180 data."""
    if angles_deg.size == 0 or weights.sum() <= 0:
        return GRAVITY_ANGLE_DEG, 0.0
    rad2 = np.deg2rad(angles_deg) * 2.0
    c = float(np.sum(weights * np.cos(rad2)))
    s = float(np.sum(weights * np.sin(rad2)))
    r = math.hypot(c, s) / float(weights.sum())
    mean = math.degrees(0.5 * math.atan2(s, c))
    if mean < 0:
        mean += 180.0
    return mean, r


def _axial_distance(a: float, b: float) -> float:
    """Smallest angle between two axial (period-180) orientations, 0..90."""
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


# --------------------------------------------------------------------------- #
# Convergence detection
# --------------------------------------------------------------------------- #
def _line_intersection(
    a: WrinkleLine, b: WrinkleLine
) -> tuple[float, float] | None:
    x1, y1, x2, y2 = a.x1, a.y1, a.x2, a.y2
    x3, y3, x4, y4 = b.x1, b.y1, b.x2, b.y2
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-6:
        return None  # parallel
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
    return px, py


def _convergence(
    lines: list[WrinkleLine], crop_shape: tuple[int, int]
) -> tuple[float, tuple[float, float] | None]:
    """Estimate how strongly the lines converge to a common point.

    Returns (strength 0..1, point in crop coords). Strength is the fraction of
    pairwise intersections falling within a small radius of their median.
    """
    if len(lines) < 3:
        return 0.0, None
    h, w = crop_shape
    diag = math.hypot(h, w)
    # Use the longest lines (cap for performance) and require a min length.
    sel = sorted(lines, key=lambda ln: ln.length, reverse=True)[:50]
    sel = [ln for ln in sel if ln.length >= 0.05 * diag]
    if len(sel) < 3:
        return 0.0, None

    pts: list[tuple[float, float]] = []
    bound_lo = -0.5 * w, -0.5 * h
    bound_hi = 1.5 * w, 1.5 * h
    for i in range(len(sel)):
        for j in range(i + 1, len(sel)):
            if _axial_distance(sel[i].angle_deg, sel[j].angle_deg) < 10:
                continue  # near-parallel: unstable intersection
            p = _line_intersection(sel[i], sel[j])
            if p is None:
                continue
            if bound_lo[0] <= p[0] <= bound_hi[0] and bound_lo[1] <= p[1] <= bound_hi[1]:
                pts.append(p)
    if len(pts) < 3:
        return 0.0, None

    arr = np.array(pts)
    center = np.median(arr, axis=0)
    radius = 0.15 * diag
    within = np.sum(np.linalg.norm(arr - center, axis=1) <= radius)
    strength = float(within) / float(len(pts))
    return strength, (float(center[0]), float(center[1]))


# --------------------------------------------------------------------------- #
# Patch density
# --------------------------------------------------------------------------- #
def _patch_density(edge_map: np.ndarray, grid: int = 4) -> tuple[float, float]:
    h, w = edge_map.shape[:2]
    if h < grid or w < grid:
        return 0.0, 0.0
    ys = np.linspace(0, h, grid + 1, dtype=int)
    xs = np.linspace(0, w, grid + 1, dtype=int)
    dens = []
    for gy in range(grid):
        for gx in range(grid):
            patch = edge_map[ys[gy] : ys[gy + 1], xs[gx] : xs[gx + 1]]
            if patch.size:
                dens.append(float(np.count_nonzero(patch)) / float(patch.size))
    if not dens:
        return 0.0, 0.0
    d = np.array(dens)
    std = float(d.std())
    mean = float(d.mean())
    max_z = float((d.max() - mean) / (std + 1e-6)) if std > 0 else 0.0
    return std, max_z


# --------------------------------------------------------------------------- #
# Gradient / light
# --------------------------------------------------------------------------- #
def _gradient_stats(gray: np.ndarray) -> tuple[float, float, float]:
    """Return (axial gradient orientation deg, coherence 0..1, light angle deg)."""
    if gray is None or gray.size == 0:
        return 0.0, 0.0, 0.0
    g = gray.astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)

    jxx = float(np.mean(gx * gx))
    jyy = float(np.mean(gy * gy))
    jxy = float(np.mean(gx * gy))
    denom = jxx + jyy + 1e-6
    coherence = math.sqrt((jxx - jyy) ** 2 + 4 * jxy**2) / denom
    orient = math.degrees(0.5 * math.atan2(2 * jxy, (jxx - jyy)))
    if orient < 0:
        orient += 180.0

    # Light direction: mean (signed) gradient vector -> direction of brightening.
    lx, ly = float(np.mean(gx)), float(np.mean(gy))
    light = math.degrees(math.atan2(ly, lx))  # -180..180
    return orient, float(min(1.0, coherence)), light


# --------------------------------------------------------------------------- #
# Pose context
# --------------------------------------------------------------------------- #
_JOINTS = ["left_elbow", "right_elbow", "left_knee", "right_knee", "left_hip", "right_hip"]


def _nearest_joint(
    region_center: tuple[float, float], pose: PoseResult, diag: float
) -> tuple[str | None, float | None]:
    if not pose.detected:
        return None, None
    best_name, best_d = None, None
    for name in _JOINTS:
        p = pose.point(name)
        if p is None:
            continue
        d = math.hypot(p[0] - region_center[0], p[1] - region_center[1])
        if best_d is None or d < best_d:
            best_name, best_d = name, d
    if best_name is None or best_d is None:
        return None, None
    return best_name, best_d / (diag + 1e-6)


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
def extract_features(
    candidates: WrinkleCandidates,
    pose: PoseResult | None = None,
) -> Features:
    """Compute structural features from detected wrinkle candidates."""
    pose = pose or PoseResult()
    feats = Features(pose_detected=pose.detected)

    h, w = candidates.crop_shape if candidates.crop_shape != (0, 0) else (0, 0)
    feats.area_px = float(h * w)
    ox, oy = candidates.offset

    lines = candidates.lines
    feats.num_lines = len(lines)

    # Edge density from the raw edge map.
    if candidates.edge_map is not None and candidates.edge_map.size:
        feats.edge_density = float(np.count_nonzero(candidates.edge_map)) / float(
            candidates.edge_map.size
        )
        feats.patch_density_std, feats.patch_density_max_z = _patch_density(
            candidates.edge_map
        )

    if feats.area_px > 0:
        feats.line_count_density = feats.num_lines / feats.area_px * 10000.0

    if lines:
        lengths = np.array([ln.length for ln in lines], dtype=np.float64)
        angles = np.array([ln.angle_deg for ln in lines], dtype=np.float64)
        feats.line_length_sum = float(lengths.sum())
        feats.mean_line_length = float(lengths.mean())

        dom, r = _axial_mean(angles, lengths)
        feats.dominant_line_angle = dom
        feats.orientation_dispersion = float(1.0 - r)
        feats.angle_diff_from_gravity = _axial_distance(dom, GRAVITY_ANGLE_DEG)

        total_len = float(lengths.sum()) + 1e-6
        vert = sum(
            ln.length for ln in lines if _axial_distance(ln.angle_deg, 90.0) <= 30.0
        )
        horiz = sum(
            ln.length for ln in lines if _axial_distance(ln.angle_deg, 0.0) <= 30.0
        )
        feats.vertical_fraction = float(vert / total_len)
        feats.horizontal_fraction = float(horiz / total_len)

        strength, point = _convergence(lines, (h, w))
        feats.convergence_strength = strength
        if point is not None:
            feats.convergence_point = (point[0] + ox, point[1] + oy)

    # Gradient / shadow stats.
    grad_orient, coherence, light = _gradient_stats(candidates.gray)
    feats.gradient_orientation = grad_orient
    feats.gradient_coherence = coherence
    feats.estimated_light_angle = light
    feats.local_contrast = (
        float(np.std(candidates.gray) / 255.0) if candidates.gray is not None else 0.0
    )
    if lines:
        # Real wrinkle shading runs perpendicular to the wrinkle line.
        d = _axial_distance(grad_orient, feats.dominant_line_angle)  # 0..90
        # 0 deg (parallel) -> 0 ; 90 deg (perpendicular) -> 1
        feats.shadow_wrinkle_perpendicularity = float(math.sin(math.radians(d)))

    # Pose context relative to region centre.
    diag = math.hypot(h, w)
    region_center = (ox + w / 2.0, oy + h / 2.0)
    name, dist = _nearest_joint(region_center, pose, diag)
    feats.nearest_joint = name
    feats.nearest_joint_dist_norm = dist

    return feats
