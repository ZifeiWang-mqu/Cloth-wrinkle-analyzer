"""Geometric features from 21 hand landmarks (pure math, no dependencies).

Everything here works on the ``DetectedHand`` landmark list in image pixels,
so it is fully unit-testable with synthetic hands — no MediaPipe required.
Angles are 2D projections; thresholds downstream must stay generous because
foreshortening legitimately distorts lengths and angles.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from app.services.hand_detection import (
    INDEX,
    MIDDLE,
    PINKY,
    RING,
    THUMB,
    WRIST,
    DetectedHand,
)

FINGERS: dict[str, tuple[int, int, int, int]] = {
    "thumb": THUMB,
    "index": INDEX,
    "middle": MIDDLE,
    "ring": RING,
    "pinky": PINKY,
}

Point = tuple[float, float]


@dataclass
class JointBend:
    """Signed bend at one joint: 0 = straight, +/- = bend direction (2D)."""

    finger: str
    joint: str  # "pip" | "dip" (thumb: "mcp" | "ip")
    interior_deg: float  # 180 = straight, small = folded
    sign: float  # bend direction (+1/-1/0)


@dataclass
class HandFeatures:
    confidence: float = 0.0
    bbox: dict[str, float] = field(default_factory=dict)
    palm_width: float = 0.0
    finger_lengths: dict[str, float] = field(default_factory=dict)
    length_ratios: dict[str, float] = field(default_factory=dict)  # vs middle
    bends: list[JointBend] = field(default_factory=list)
    thumb_side: float = 0.0  # sign vs palm axis
    index_side: float = 0.0
    pinky_side: float = 0.0
    wrist_palm_ratio: float = 0.0  # dist(wrist, middle_mcp) / palm_width
    adjacent_pairs: list[tuple[str, str, float, float]] = field(default_factory=list)
    # (finger_a, finger_b, tip_dist/palm_w, pip_dist/palm_w)
    degenerate: bool = False  # palm too small to measure anything

    def finger_points(self, name: str) -> list[Point]:
        return self._points.get(name, [])

    _points: dict[str, list[Point]] = field(default_factory=dict, repr=False)

    def to_debug(self) -> dict:
        """Rounded diagnostic snapshot of every value the rules judge.

        Purely informational (surfaced in ``debug.hand.features``) so users can
        see WHY rules did or did not fire on a detected hand.
        """
        if self.degenerate:
            return {"degenerate": True, "palm_width": round(self.palm_width, 1)}
        min_interior: dict[str, float] = {}
        bend_signs: dict[str, list[float]] = {}
        for b in self.bends:
            cur = min_interior.get(b.finger)
            if cur is None or b.interior_deg < cur:
                min_interior[b.finger] = b.interior_deg
            bend_signs.setdefault(b.finger, []).append(b.sign)
        return {
            "degenerate": False,
            "confidence": round(self.confidence, 3),
            "palm_width": round(self.palm_width, 1),
            "bbox": self.bbox,
            "length_ratios": {k: round(v, 3) for k, v in self.length_ratios.items()},
            "min_interior_deg": {k: round(v, 1) for k, v in min_interior.items()},
            "bend_signs": bend_signs,
            "thumb_side": self.thumb_side,
            "index_side": self.index_side,
            "pinky_side": self.pinky_side,
            "wrist_palm_ratio": round(self.wrist_palm_ratio, 3),
            "adjacent_pairs": [
                {"pair": f"{a}+{b}", "tip": round(t, 3), "pip": round(p, 3)}
                for a, b, t, p in self.adjacent_pairs
            ],
        }


def _dist(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _signed_angle(a: Point, b: Point, c: Point) -> tuple[float, float]:
    """Interior angle at ``b`` (deg) and bend sign via 2D cross product."""
    ux, uy = a[0] - b[0], a[1] - b[1]
    vx, vy = c[0] - b[0], c[1] - b[1]
    dot = ux * vx + uy * vy
    cross = ux * vy - uy * vx
    ang = math.degrees(math.atan2(abs(cross), dot))
    sign = 0.0 if abs(cross) < 1e-9 else math.copysign(1.0, cross)
    return ang, sign


def _side(axis_a: Point, axis_b: Point, p: Point) -> float:
    """Sign of ``p`` relative to the line axis_a -> axis_b."""
    cross = (axis_b[0] - axis_a[0]) * (p[1] - axis_a[1]) - (
        axis_b[1] - axis_a[1]
    ) * (p[0] - axis_a[0])
    return 0.0 if abs(cross) < 1e-9 else math.copysign(1.0, cross)


def extract_hand_features(hand: DetectedHand) -> HandFeatures:
    lm = hand.landmarks
    f = HandFeatures(confidence=hand.confidence, bbox=dict(hand.bbox))

    wrist = lm[WRIST]
    index_mcp, middle_mcp, pinky_mcp = lm[INDEX[0]], lm[MIDDLE[0]], lm[PINKY[0]]
    f.palm_width = _dist(index_mcp, pinky_mcp)
    if f.palm_width < 4.0:  # degenerate/collapsed hand — nothing is measurable
        f.degenerate = True
        return f

    # Per-finger chains, lengths, and joint bends.
    for name, ids in FINGERS.items():
        chain = [lm[i] for i in ids]
        f._points[name] = chain
        f.finger_lengths[name] = sum(_dist(chain[i], chain[i + 1]) for i in range(3))
        j1, j2 = ("mcp", "ip") if name == "thumb" else ("pip", "dip")
        a1, s1 = _signed_angle(chain[0], chain[1], chain[2])
        a2, s2 = _signed_angle(chain[1], chain[2], chain[3])
        f.bends.append(JointBend(name, j1, a1, s1))
        f.bends.append(JointBend(name, j2, a2, s2))

    middle_len = f.finger_lengths.get("middle", 0.0) or 1e-6
    f.length_ratios = {
        name: length / middle_len for name, length in f.finger_lengths.items()
    }

    # Thumb placement: which side of the wrist->middle_mcp palm axis?
    f.thumb_side = _side(wrist, middle_mcp, lm[THUMB[1]])  # thumb MCP
    f.index_side = _side(wrist, middle_mcp, index_mcp)
    f.pinky_side = _side(wrist, middle_mcp, pinky_mcp)

    # Wrist-to-palm connection plausibility.
    f.wrist_palm_ratio = _dist(wrist, middle_mcp) / f.palm_width

    # Adjacent finger proximity (merged-finger signal).
    order = ["index", "middle", "ring", "pinky"]
    for a, b in zip(order, order[1:]):
        tip_d = _dist(f._points[a][3], f._points[b][3]) / f.palm_width
        pip_d = _dist(f._points[a][1], f._points[b][1]) / f.palm_width
        f.adjacent_pairs.append((a, b, tip_d, pip_d))

    return f
