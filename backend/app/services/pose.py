"""Pose estimation wrapper.

MediaPipe Pose is optional. If it is not installed (or fails to run on the
given image) we degrade gracefully: ``pose_detected`` is False and downstream
rules that need landmarks become low-confidence or are skipped. The API never
fails because of a missing pose.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

# Landmark indices we care about (MediaPipe Pose 33-point model).
_LANDMARK_IDS = {
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
}


@dataclass
class PoseResult:
    detected: bool = False
    # name -> {"x": px, "y": px, "visibility": 0..1}
    landmarks: dict[str, dict[str, float]] = field(default_factory=dict)
    backend: str = "none"  # "mediapipe" | "none"

    def point(self, name: str) -> tuple[float, float] | None:
        lm = self.landmarks.get(name)
        if lm is None:
            return None
        return lm["x"], lm["y"]

    def visibility(self, name: str) -> float:
        lm = self.landmarks.get(name)
        return float(lm.get("visibility", 0.0)) if lm else 0.0


# --------------------------------------------------------------------------- #
# Joint geometry (advanced joint rule)
# --------------------------------------------------------------------------- #
# joint_name -> (proximal, joint, distal, joint_type)
_JOINT_TRIPLETS = [
    ("left_elbow", "left_shoulder", "left_elbow", "left_wrist", "elbow"),
    ("right_elbow", "right_shoulder", "right_elbow", "right_wrist", "elbow"),
    ("left_knee", "left_hip", "left_knee", "left_ankle", "knee"),
    ("right_knee", "right_hip", "right_knee", "right_ankle", "knee"),
    ("left_hip", "left_shoulder", "left_hip", "left_knee", "hip"),
    ("right_hip", "right_shoulder", "right_hip", "right_knee", "hip"),
]


@dataclass
class JointContext:
    joint_name: str
    joint_type: str
    proximal: tuple[float, float]
    joint: tuple[float, float]
    distal: tuple[float, float]
    angle_degrees: float
    bend_strength: float  # 0 (straight) .. 1 (sharply bent)
    compression_vector: tuple[float, float]  # unit vector toward concave side
    stretch_vector: tuple[float, float]
    confidence: float

    def to_debug(self) -> dict:
        return {
            "joint_name": self.joint_name,
            "joint_type": self.joint_type,
            "angle_degrees": round(self.angle_degrees, 1),
            "bend_strength": round(self.bend_strength, 3),
            "confidence": round(self.confidence, 3),
        }


def _unit(v: tuple[float, float]) -> tuple[float, float]:
    n = math.hypot(v[0], v[1])
    if n < 1e-9:
        return (0.0, 0.0)
    return (v[0] / n, v[1] / n)


def compute_joint_angle(
    a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]
) -> float:
    """Interior angle at vertex ``b`` (degrees, 0..180)."""
    v1 = (a[0] - b[0], a[1] - b[1])
    v2 = (c[0] - b[0], c[1] - b[1])
    n1 = math.hypot(*v1)
    n2 = math.hypot(*v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return 180.0
    cosv = (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)
    cosv = max(-1.0, min(1.0, cosv))
    return math.degrees(math.acos(cosv))


def estimate_joint_bend_direction(
    a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]
) -> tuple[float, float]:
    """Unit vector at ``b`` pointing toward the concave (compression) side."""
    u1 = _unit((a[0] - b[0], a[1] - b[1]))
    u2 = _unit((c[0] - b[0], c[1] - b[1]))
    bis = _unit((u1[0] + u2[0], u1[1] + u2[1]))
    return bis


def classify_side_of_joint(
    point: tuple[float, float], ctx: JointContext
) -> str:
    """Return "compression" | "stretch" | "neutral" for a point near the joint."""
    d = (point[0] - ctx.joint[0]) * ctx.compression_vector[0] + (
        point[1] - ctx.joint[1]
    ) * ctx.compression_vector[1]
    if d > 1e-6:
        return "compression"
    if d < -1e-6:
        return "stretch"
    return "neutral"


def get_joint_contexts(
    pose: PoseResult, image_shape: tuple[int, int] | None = None
) -> list[JointContext]:
    """Build joint contexts for elbows/knees/hips from detected landmarks."""
    if not pose.detected:
        return []
    contexts: list[JointContext] = []
    for joint_name, prox, jnt, dist, jtype in _JOINT_TRIPLETS:
        a, b, c = pose.point(prox), pose.point(jnt), pose.point(dist)
        if a is None or b is None or c is None:
            continue
        vis = (
            pose.visibility(prox) + pose.visibility(jnt) + pose.visibility(dist)
        ) / 3.0
        if vis < 0.2:
            continue
        angle = compute_joint_angle(a, b, c)
        bend_strength = max(0.0, min(1.0, (180.0 - angle) / 90.0))
        comp = estimate_joint_bend_direction(a, b, c)
        contexts.append(
            JointContext(
                joint_name=joint_name,
                joint_type=jtype,
                proximal=a,
                joint=b,
                distal=c,
                angle_degrees=angle,
                bend_strength=bend_strength,
                compression_vector=comp,
                stretch_vector=(-comp[0], -comp[1]),
                confidence=float(vis),
            )
        )
    return contexts


def _try_import_mediapipe():
    try:
        import mediapipe as mp  # type: ignore

        return mp
    except Exception:  # pragma: no cover - depends on environment
        return None


def estimate_pose(image: np.ndarray) -> PoseResult:
    """Estimate body landmarks from a BGR image.

    Returns an empty (``detected=False``) result if MediaPipe is unavailable
    or no person is found. Any internal error is swallowed and logged.
    """
    mp = _try_import_mediapipe()
    if mp is None:
        logger.info("MediaPipe not installed; skipping pose estimation.")
        return PoseResult(detected=False, backend="none")

    try:
        import cv2

        h, w = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        with mp.solutions.pose.Pose(
            static_image_mode=True,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.4,
        ) as pose:
            res = pose.process(rgb)

        if not res.pose_landmarks:
            return PoseResult(detected=False, backend="mediapipe")

        landmarks: dict[str, dict[str, float]] = {}
        for name, idx in _LANDMARK_IDS.items():
            lm = res.pose_landmarks.landmark[idx]
            landmarks[name] = {
                "x": float(lm.x) * w,
                "y": float(lm.y) * h,
                "visibility": float(lm.visibility),
            }
        return PoseResult(detected=True, landmarks=landmarks, backend="mediapipe")

    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Pose estimation failed: %s", exc)
        return PoseResult(detected=False, backend="mediapipe")
