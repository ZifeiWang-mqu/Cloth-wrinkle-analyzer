"""Hand landmark detection wrapper.

MediaPipe is OPTIONAL, mirroring the graceful-degradation pattern of
``pose.py``. Two MediaPipe APIs are supported:

1. **Tasks API** (``mp.tasks.vision.HandLandmarker``) — required for MediaPipe
   >= ~0.10.x builds that no longer ship ``mp.solutions``. Needs a
   ``hand_landmarker.task`` model file (NOT committed to git); its location is
   resolved from settings (``WRINKLE_HAND_MODEL_PATH`` or
   ``data/models/hand/hand_landmarker.task``).
2. **Legacy solutions API** (``mp.solutions.hands``) — used as a fallback for
   older MediaPipe installs where it still exists.

If MediaPipe is missing, the Tasks model file is absent, or detection fails,
we return ``HandDetection(detected=False)`` with a machine-readable ``note``
and the API never crashes. Illustrations are out-of-domain for MediaPipe
(trained on photos), so callers should treat a miss as "detector couldn't find
a hand", not "there is no hand".

An optional region hint (the user's lasso/bbox selection) crops the image
before detection, which measurably helps on stylized art.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from app.services.segmentation import GarmentRegion

logger = logging.getLogger(__name__)

# MediaPipe Hands landmark indices (21-point model, identical in both APIs).
WRIST = 0
THUMB = (1, 2, 3, 4)  # CMC, MCP, IP, TIP
INDEX = (5, 6, 7, 8)  # MCP, PIP, DIP, TIP
MIDDLE = (9, 10, 11, 12)
RING = (13, 14, 15, 16)
PINKY = (17, 18, 19, 20)
NUM_LANDMARKS = 21


@dataclass
class DetectedHand:
    """One detected hand, landmarks in FULL-image pixel coordinates."""

    landmarks: list[tuple[float, float]]  # exactly 21 (x, y) points
    handedness: str = "unknown"  # "Left" | "Right" | "unknown"
    confidence: float = 0.0
    bbox: dict[str, float] = field(default_factory=dict)  # padded {x,y,w,h}


@dataclass
class HandDetection:
    detected: bool = False
    hands: list[DetectedHand] = field(default_factory=list)
    backend: str = "none"  # "mediapipe_tasks" | "mediapipe_solutions" | "none"
    note: str | None = None
    # notes: "mediapipe_not_installed" | "hand_model_missing" |
    #        "hand_landmarker_unavailable" | "no_hands_found" | "tasks_error: …"

    def to_debug(self) -> dict:
        return {
            "detected": self.detected,
            "backend": self.backend,
            "num_hands": len(self.hands),
            "confidences": [round(h.confidence, 3) for h in self.hands],
            "note": self.note,
        }


# --------------------------------------------------------------------------- #
# Availability probes (module-level so tests can monkeypatch them)
# --------------------------------------------------------------------------- #
def _try_import_mediapipe():
    try:
        import mediapipe as mp  # type: ignore

        return mp
    except Exception:  # pragma: no cover - depends on environment
        return None


def _tasks_modules():
    """Return ``(mp_tasks, mp_vision)`` or None if the Tasks API is unavailable."""
    try:  # pragma: no cover - depends on environment
        from mediapipe.tasks import python as mp_tasks  # type: ignore
        from mediapipe.tasks.python import vision as mp_vision  # type: ignore

        return mp_tasks, mp_vision
    except Exception:
        return None


def _resolve_model() -> Path | None:
    """Path to the HandLandmarker ``.task`` model, or None if not installed.

    Settings import is deferred so this module stays importable without the
    web-framework dependencies.
    """
    from app.settings import settings

    return settings.resolved_hand_model


# Cached HandLandmarker (model load is not free); keyed by model path.
_LANDMARKER = None
_LANDMARKER_PATH: str | None = None


def reset_hand_landmarker() -> None:
    global _LANDMARKER, _LANDMARKER_PATH
    _LANDMARKER = None
    _LANDMARKER_PATH = None


def _get_landmarker(mp_tasks, mp_vision, model_path: Path, max_hands: int, min_confidence: float):
    global _LANDMARKER, _LANDMARKER_PATH
    if _LANDMARKER is not None and _LANDMARKER_PATH == str(model_path):
        return _LANDMARKER
    options = mp_vision.HandLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_hands=max_hands,
        min_hand_detection_confidence=min_confidence,
    )
    _LANDMARKER = mp_vision.HandLandmarker.create_from_options(options)
    _LANDMARKER_PATH = str(model_path)
    logger.info("HandLandmarker loaded (%s).", model_path)
    return _LANDMARKER


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _landmarks_bbox(
    points: list[tuple[float, float]], img_w: int, img_h: int, pad_ratio: float = 0.15
) -> dict[str, float]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    w = max(xs) - min(xs)
    h = max(ys) - min(ys)
    pad = pad_ratio * max(w, h, 1.0)
    x = max(0.0, min(xs) - pad)
    y = max(0.0, min(ys) - pad)
    return {
        "x": round(x, 1),
        "y": round(y, 1),
        "w": round(min(w + 2 * pad, img_w - x), 1),
        "h": round(min(h + 2 * pad, img_h - y), 1),
    }


def _build_hand(
    points: list[tuple[float, float]],
    handedness: str,
    confidence: float,
    img_w: int,
    img_h: int,
) -> DetectedHand | None:
    if len(points) != NUM_LANDMARKS:
        return None
    return DetectedHand(
        landmarks=points,
        handedness=handedness,
        confidence=confidence,
        bbox=_landmarks_bbox(points, img_w, img_h),
    )


def _detect_with_tasks(
    mp, mp_tasks, mp_vision, model_path: Path, rgb: np.ndarray,
    ox: float, oy: float, sw: int, sh: int, img_w: int, img_h: int,
    max_hands: int, min_confidence: float,
) -> list[DetectedHand]:
    landmarker = _get_landmarker(mp_tasks, mp_vision, model_path, max_hands, min_confidence)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
    result = landmarker.detect(mp_image)
    hands: list[DetectedHand] = []
    hand_landmarks = result.hand_landmarks or []
    handedness_list = result.handedness or []
    for i, lms in enumerate(hand_landmarks):
        points = [(float(lm.x) * sw + ox, float(lm.y) * sh + oy) for lm in lms]
        label, score = "unknown", 0.0
        if i < len(handedness_list) and handedness_list[i]:
            cat = handedness_list[i][0]
            label, score = str(cat.category_name), float(cat.score)
        hand = _build_hand(points, label, score, img_w, img_h)
        if hand:
            hands.append(hand)
    return hands


def _detect_with_solutions(
    mp, rgb: np.ndarray,
    ox: float, oy: float, sw: int, sh: int, img_w: int, img_h: int,
    max_hands: int, min_confidence: float,
) -> list[DetectedHand]:
    with mp.solutions.hands.Hands(
        static_image_mode=True,
        max_num_hands=max_hands,
        min_detection_confidence=min_confidence,
    ) as detector:
        res = detector.process(rgb)
    hands: list[DetectedHand] = []
    if not res.multi_hand_landmarks:
        return hands
    handedness_list = res.multi_handedness or []
    for i, hand_lms in enumerate(res.multi_hand_landmarks):
        points = [(float(lm.x) * sw + ox, float(lm.y) * sh + oy) for lm in hand_lms.landmark]
        label, score = "unknown", 0.0
        if i < len(handedness_list) and handedness_list[i].classification:
            cls = handedness_list[i].classification[0]
            label, score = str(cls.label), float(cls.score)
        hand = _build_hand(points, label, score, img_w, img_h)
        if hand:
            hands.append(hand)
    return hands


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def detect_hands(
    image: np.ndarray,
    region: GarmentRegion | None = None,
    max_hands: int = 2,
    min_confidence: float = 0.3,
) -> HandDetection:
    """Detect hands in a BGR image. Never raises.

    Tries the MediaPipe Tasks API first (current versions), then the legacy
    solutions API (older versions). ``region`` (when the user made a
    selection) crops the search area; returned landmarks are always mapped
    back to full-image coordinates.
    """
    mp = _try_import_mediapipe()
    if mp is None:
        logger.info("MediaPipe not installed; hand detection unavailable.")
        return HandDetection(detected=False, backend="none", note="mediapipe_not_installed")

    try:
        import cv2

        img_h, img_w = image.shape[:2]
        ox = oy = 0
        search = image
        if region is not None and region.used_selection:
            search = region.crop(image)
            ox, oy = region.x, region.y
            if search.size == 0:
                search, ox, oy = image, 0, 0
        sh, sw = search.shape[:2]
        rgb = cv2.cvtColor(search, cv2.COLOR_BGR2RGB)

        note: str | None = None

        # --- Preferred: Tasks API (MediaPipe >= 0.10 without mp.solutions) ---
        tasks = _tasks_modules()
        if tasks is not None:
            mp_tasks, mp_vision = tasks
            model_path = _resolve_model()
            if model_path is None:
                note = "hand_model_missing"
            else:
                try:
                    hands = _detect_with_tasks(
                        mp, mp_tasks, mp_vision, model_path, rgb,
                        ox, oy, sw, sh, img_w, img_h, max_hands, min_confidence,
                    )
                    if hands:
                        return HandDetection(detected=True, hands=hands, backend="mediapipe_tasks")
                    return HandDetection(
                        detected=False, backend="mediapipe_tasks", note="no_hands_found"
                    )
                except Exception as exc:  # pragma: no cover - env dependent
                    logger.warning("HandLandmarker (Tasks API) failed: %s", exc)
                    note = f"tasks_error: {exc}"

        # --- Fallback: legacy solutions API (older MediaPipe installs) ---
        if getattr(mp, "solutions", None) is not None:
            try:
                hands = _detect_with_solutions(
                    mp, rgb, ox, oy, sw, sh, img_w, img_h, max_hands, min_confidence
                )
                if hands:
                    return HandDetection(
                        detected=True, hands=hands, backend="mediapipe_solutions"
                    )
                return HandDetection(
                    detected=False, backend="mediapipe_solutions", note="no_hands_found"
                )
            except Exception as exc:  # pragma: no cover - env dependent
                logger.warning("Legacy hands API failed: %s", exc)
                note = note or f"error: {exc}"

        return HandDetection(
            detected=False,
            backend="mediapipe",
            note=note or "hand_landmarker_unavailable",
        )

    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Hand detection failed: %s", exc)
        return HandDetection(detected=False, backend="mediapipe", note=f"error: {exc}")
