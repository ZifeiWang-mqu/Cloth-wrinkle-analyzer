"""Pose estimation wrapper.

MediaPipe Pose is optional. If it is not installed (or fails to run on the
given image) we degrade gracefully: ``pose_detected`` is False and downstream
rules that need landmarks become low-confidence or are skipped. The API never
fails because of a missing pose.
"""

from __future__ import annotations

import logging
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
