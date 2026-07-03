"""Runtime capability detection.

Reports which optional features are actually available in this environment so
the API/UI can adapt and never assume a missing dependency. Nothing here raises.
"""

from __future__ import annotations

import importlib.util

from app.settings import Settings


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:  # pragma: no cover - defensive
        return False


def sam_importable() -> bool:
    # SAM needs both the library and torch.
    return _module_available("segment_anything") and _module_available("torch")


def mediapipe_available() -> bool:
    return _module_available("mediapipe")


def sklearn_available() -> bool:
    return _module_available("sklearn") and _module_available("joblib")


def get_capabilities(settings: Settings) -> dict:
    sam_ok = sam_importable()
    checkpoint = settings.resolved_sam_checkpoint
    checkpoint_present = checkpoint is not None and checkpoint.exists()

    if sam_ok and checkpoint_present:
        seg_provider = "sam"
    else:
        seg_provider = "opencv"  # always-available fallback

    # SAM runtime state (does not force a load).
    sam_loaded = False
    sam_device = settings.sam_device
    try:
        from app.services.segmentation import get_sam_segmenter

        st = get_sam_segmenter(settings).status()
        sam_loaded = bool(st.get("sam_loaded", False))
        sam_device = st.get("device", sam_device)
    except Exception:  # pragma: no cover - defensive
        pass

    illu_path = settings.illustration_model_path
    illu_available = sklearn_available() and illu_path.exists()
    if not sklearn_available():
        illu_reason = "scikit-learn / joblib not installed"
    elif not illu_path.exists():
        illu_reason = "model file not found"
    else:
        illu_reason = None

    return {
        # Doc-shaped capability blocks.
        "sam": {
            "available": sam_ok,
            "loaded": sam_loaded,
            "checkpoint_exists": checkpoint_present,
            "device": sam_device,
        },
        "opencv": {"available": True},
        "mediapipe": {"available": mediapipe_available()},
        # Backward-compatible grouped blocks (used elsewhere).
        "segmentation": {
            "sam_available": sam_ok,
            "sam_checkpoint_present": checkpoint_present,
            "opencv_available": True,
            "effective_provider": seg_provider,
        },
        "pose": {
            "mediapipe_available": mediapipe_available(),
        },
        "illustration_model": {
            "available": illu_available,
            "sklearn_available": sklearn_available(),
            "reason": illu_reason,
        },
        # Optional GPT visual second opinion (server-side key only; the key
        # itself is never exposed here — source/prefix/length diagnostics only).
        "ai_review": {
            "available": bool(settings.resolved_openai_key),
            "model": settings.openai_model,
            **settings.openai_key_diagnostics(),
        },
    }
