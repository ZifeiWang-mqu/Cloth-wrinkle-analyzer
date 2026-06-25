"""Local evidence-box generation for issues.

Turns each coarse issue into a small set of *local* evidence boxes around the
specific wrinkle lines / dense patches / convergence point that drove its score,
so the frontend overlay can highlight precise areas instead of the whole
garment region. Reuses already-computed line/patch features — small JSON only,
no images/masks. Never raises; falls back to the issue bbox (marked
``fallback_broad_bbox``) when precise evidence can't be computed.
"""

from __future__ import annotations

import math

import numpy as np

from app.services.feature_extraction import Features
from app.services.pose import PoseResult
from app.services.wrinkle_edges import WrinkleCandidates, WrinkleLine

MAX_BOXES_PER_ISSUE = 4
MAX_TOTAL_BOXES = 24
MERGE_IOU = 0.55
MAX_BOX_AREA_RATIO = 0.25  # vs region area -> considered "too broad"


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #
def _padding(iw: int, ih: int) -> float:
    return float(max(8, int(0.015 * min(iw, ih))))


def clamp_box(b: dict, iw: int, ih: int) -> dict:
    x = max(0.0, min(float(b["x"]), iw - 1.0))
    y = max(0.0, min(float(b["y"]), ih - 1.0))
    w = max(1.0, min(float(b["w"]), iw - x))
    h = max(1.0, min(float(b["h"]), ih - y))
    out = dict(b)
    out.update({"x": round(x, 1), "y": round(y, 1), "w": round(w, 1), "h": round(h, 1)})
    return out


def line_to_evidence_box(ln: WrinkleLine, ox: float, oy: float, pad: float) -> dict:
    x1, y1 = ln.x1 + ox, ln.y1 + oy
    x2, y2 = ln.x2 + ox, ln.y2 + oy
    return {
        "x": min(x1, x2) - pad,
        "y": min(y1, y2) - pad,
        "w": abs(x2 - x1) + 2 * pad,
        "h": abs(y2 - y1) + 2 * pad,
    }


def _iou(a: dict, b: dict) -> float:
    ax2, ay2 = a["x"] + a["w"], a["y"] + a["h"]
    bx2, by2 = b["x"] + b["w"], b["y"] + b["h"]
    ix1, iy1 = max(a["x"], b["x"]), max(a["y"], b["y"])
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw_, ih_ = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw_ * ih_
    if inter <= 0:
        return 0.0
    union = a["w"] * a["h"] + b["w"] * b["h"] - inter
    return inter / union if union > 0 else 0.0


def merge_small_overlapping_boxes(boxes: list[dict], thr: float = MERGE_IOU) -> list[dict]:
    used = [False] * len(boxes)
    out: list[dict] = []
    for i in range(len(boxes)):
        if used[i]:
            continue
        cur = dict(boxes[i])
        for j in range(i + 1, len(boxes)):
            if used[j] or _iou(cur, boxes[j]) <= thr:
                continue
            x1 = min(cur["x"], boxes[j]["x"])
            y1 = min(cur["y"], boxes[j]["y"])
            x2 = max(cur["x"] + cur["w"], boxes[j]["x"] + boxes[j]["w"])
            y2 = max(cur["y"] + cur["h"], boxes[j]["y"] + boxes[j]["h"])
            cur.update({"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1})
            cur["score"] = max(cur.get("score") or 0.0, boxes[j].get("score") or 0.0)
            used[j] = True
        out.append(cur)
    return out


def limit_evidence_boxes(boxes: list[dict], n: int) -> list[dict]:
    return sorted(boxes, key=lambda b: b.get("score") or 0.0, reverse=True)[:n]


def _axial_dist(a: float, b: float) -> float:
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


# --------------------------------------------------------------------------- #
# Per-issue-type evidence
# --------------------------------------------------------------------------- #
def _gravity_boxes(lines, ox, oy, pad) -> list[dict]:
    scored = []
    for ln in lines:
        horiz = _axial_dist(ln.angle_deg, 90.0) / 90.0  # 0 vertical .. 1 horizontal
        if horiz < 0.4:
            continue
        scored.append((horiz * ln.length, horiz, ln))
    scored.sort(key=lambda t: t[0], reverse=True)
    out = []
    for _w, horiz, ln in scored[:6]:
        b = line_to_evidence_box(ln, ox, oy, pad)
        b.update({"score": round(min(1.0, horiz), 3), "reason": "horizontal_wrinkle_candidate", "source": "wrinkle_line"})
        out.append(b)
    return out


def _body_volume_boxes(lines, ox, oy, pad, diag) -> list[dict]:
    scored = [(ln.length, ln) for ln in lines if ln.length > 0.12 * diag]
    scored.sort(key=lambda t: t[0], reverse=True)
    out = []
    for length, ln in scored[:6]:
        b = line_to_evidence_box(ln, ox, oy, pad)
        b.update({"score": round(min(1.0, length / diag), 3), "reason": "long_straight_parallel_wrinkle", "source": "wrinkle_line"})
        out.append(b)
    return out


def _density_boxes(candidates: WrinkleCandidates, ox, oy) -> list[dict]:
    em = candidates.edge_map
    if em is None or em.size == 0:
        return []
    h, w = em.shape[:2]
    grid = 4
    ys = np.linspace(0, h, grid + 1, dtype=int)
    xs = np.linspace(0, w, grid + 1, dtype=int)
    cells = []
    for gy in range(grid):
        for gx in range(grid):
            patch = em[ys[gy] : ys[gy + 1], xs[gx] : xs[gx + 1]]
            if patch.size == 0:
                continue
            d = float(np.count_nonzero(patch)) / patch.size
            cells.append((d, int(xs[gx]), int(ys[gy]), int(xs[gx + 1] - xs[gx]), int(ys[gy + 1] - ys[gy])))
    if not cells:
        return []
    ds = np.array([c[0] for c in cells])
    mean, std = float(ds.mean()), float(ds.std()) or 1e-6
    out = []
    for d, px, py, pw, ph in cells:
        z = (d - mean) / std
        if z < 1.0:
            continue
        out.append({"x": px + ox, "y": py + oy, "w": pw, "h": ph, "score": round(float(min(1.0, z / 3.0)), 3), "reason": "high_density_patch", "source": "patch"})
    return out


def _tension_boxes(features: Features, lines, ox, oy, pad) -> list[dict]:
    cp = features.convergence_point
    if cp is None:
        return []
    cx, cy = cp
    size = max(24.0, pad * 3)
    out = [{"x": cx - size / 2, "y": cy - size / 2, "w": size, "h": size, "score": round(float(features.convergence_strength), 3), "reason": "unanchored_convergence_point", "source": "convergence"}]
    scored = sorted(lines, key=lambda ln: math.hypot(ln.midpoint[0] + ox - cx, ln.midpoint[1] + oy - cy))
    for ln in scored[:3]:
        b = line_to_evidence_box(ln, ox, oy, pad)
        b.update({"score": 0.6, "reason": "converging_wrinkle_lines", "source": "wrinkle_line"})
        out.append(b)
    return out


def _joint_boxes(features: Features, pose: PoseResult, lines, ox, oy, pad, diag) -> list[dict]:
    if not pose.detected or not features.nearest_joint:
        return []
    p = pose.point(features.nearest_joint)
    if p is None:
        return []
    jx, jy = p
    size = max(40.0, pad * 4)
    out = [{"x": jx - size / 2, "y": jy - size / 2, "w": size, "h": size, "score": round(float(features.joint_bend_strength or 0.5), 3), "reason": "near_joint_region", "source": "joint"}]
    scored = sorted(lines, key=lambda ln: math.hypot(ln.midpoint[0] + ox - jx, ln.midpoint[1] + oy - jy))
    for ln in scored[:3]:
        if math.hypot(ln.midpoint[0] + ox - jx, ln.midpoint[1] + oy - jy) > 0.25 * diag:
            break
        b = line_to_evidence_box(ln, ox, oy, pad)
        b.update({"score": 0.5, "reason": "wrinkle_near_joint", "source": "wrinkle_line"})
        out.append(b)
    return out


def _shadow_boxes(lines, ox, oy, pad) -> list[dict]:
    out = []
    for ln in sorted(lines, key=lambda ln: ln.length, reverse=True)[:3]:
        b = line_to_evidence_box(ln, ox, oy, pad)
        b.update({"score": 0.4, "reason": "local_shadow_wrinkle_mismatch", "source": "wrinkle_line"})
        out.append(b)
    return out


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def build_evidence_for_issues(
    issues: list[dict],
    candidates: WrinkleCandidates | None,
    features: Features,
    pose: PoseResult,
    region_context: dict,
    image_shape: tuple[int, int],
) -> list[dict]:
    """Attach an ``evidence_boxes`` list to each issue dict (in place)."""
    ih, iw = image_shape
    diag = math.hypot(iw, ih)
    pad = _padding(iw, ih)
    region_bbox = region_context.get("bbox", {"x": 0, "y": 0, "w": iw, "h": ih})
    region_area = max(1.0, float(region_bbox.get("w", iw)) * float(region_bbox.get("h", ih)))
    lines = list(candidates.lines) if candidates else []
    ox, oy = candidates.offset if candidates else (0, 0)

    total = 0
    for issue in issues:
        boxes: list[dict] = []
        try:
            t = issue.get("type")
            if t == "gravity_inconsistency":
                boxes = _gravity_boxes(lines, ox, oy, pad)
            elif t == "body_volume_inconsistency":
                boxes = _body_volume_boxes(lines, ox, oy, pad, diag)
            elif t == "density_inconsistency":
                boxes = _density_boxes(candidates, ox, oy) if candidates else []
            elif t == "tension_ambiguity":
                boxes = _tension_boxes(features, lines, ox, oy, pad)
            elif t == "joint_inconsistency":
                boxes = _joint_boxes(features, pose, lines, ox, oy, pad, diag)
            elif t == "shadow_wrinkle_mismatch":
                boxes = _shadow_boxes(lines, ox, oy, pad)
        except Exception:  # pragma: no cover - never break inspection
            boxes = []

        boxes = [clamp_box(b, iw, ih) for b in boxes]
        boxes = [b for b in boxes if (b["w"] * b["h"]) <= MAX_BOX_AREA_RATIO * region_area]
        boxes = merge_small_overlapping_boxes(boxes)
        boxes = limit_evidence_boxes(boxes, MAX_BOXES_PER_ISSUE)
        if total + len(boxes) > MAX_TOTAL_BOXES:
            boxes = boxes[: max(0, MAX_TOTAL_BOXES - total)]
        total += len(boxes)

        if boxes:
            for b in boxes:
                b.setdefault("fallback_broad_bbox", False)
        else:
            bb = issue.get("bbox", region_bbox)
            boxes = [
                clamp_box(
                    {
                        "x": bb["x"],
                        "y": bb["y"],
                        "w": bb["w"],
                        "h": bb["h"],
                        "score": issue.get("score"),
                        "reason": "fallback_broad_bbox",
                        "source": "issue_bbox",
                        "fallback_broad_bbox": True,
                    },
                    iw,
                    ih,
                )
            ]
        issue["evidence_boxes"] = boxes
    return issues
