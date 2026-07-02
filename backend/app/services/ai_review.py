"""AI visual review — optional GPT second opinion on top of the detector.

Sends a focused image crop + the stored inspection context to an OpenAI
vision-capable model via the **Responses API with Structured Outputs**
(strict JSON schema), using plain httpx — no SDK dependency. The service is
isolated behind small functions so it can be migrated or mocked easily:

- ``_resolved_key()``      — availability seam (tests monkeypatch this)
- ``_post_responses()``    — the ONLY network call (tests monkeypatch this)

Privacy: the API key is server-side only and never logged; image bytes are
never logged (sizes only); responses are returned, not stored.
"""

from __future__ import annotations

import base64
import json
import logging

import cv2
import numpy as np

from app.schemas import VISUAL_PROBLEM_TYPES
from app.settings import settings

logger = logging.getLogger(__name__)

OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"


class AIReviewUpstreamError(Exception):
    """OpenAI HTTP/timeout failure (route maps to 502)."""


class AIReviewParseError(Exception):
    """Model output could not be parsed/validated (route maps to 502)."""


# --------------------------------------------------------------------------- #
# Availability
# --------------------------------------------------------------------------- #
def _resolved_key() -> str:
    return settings.resolved_openai_key


def is_available() -> tuple[bool, str | None]:
    if not _resolved_key():
        return False, "OPENAI_API_KEY missing"
    return True, None


# --------------------------------------------------------------------------- #
# Context (compact, from the stored Inspection row only)
# --------------------------------------------------------------------------- #
def build_context(inspection) -> dict:
    """Trim the persisted inspection into a compact JSON context for the model."""
    from app.services.review_memory import build_snapshot  # deferred (sqlalchemy)

    snapshot = build_snapshot(inspection)
    issues = [
        {
            "type": i.get("type"),
            "label": i.get("label"),
            "severity": i.get("severity"),
            "score": i.get("score"),
            "flagged": i.get("flagged"),
            "bbox": i.get("bbox"),
            "evidence_boxes": [
                {k: b.get(k) for k in ("x", "y", "w", "h")}
                for b in (i.get("evidence_boxes") or [])[:3]
            ],
        }
        for i in (snapshot.get("issues") or [])
    ]
    ctx: dict = {
        "mode": snapshot["mode"],
        "result": snapshot["result"],
        "overall_score": snapshot["overall_score"],
        "scores": snapshot["scores"],
        "issues": issues,
        "selected_region_used": snapshot["selected_region"],
    }
    debug = snapshot.get("debug") or {}
    if snapshot["mode"] == "hand":
        hand = debug.get("hand") or {}
        ctx["hand"] = {
            "detected": hand.get("detected"),
            "backend": hand.get("backend"),
            "confidences": hand.get("confidences"),
            "note": hand.get("note"),
            "features": hand.get("features"),
            "rule_details": hand.get("rule_details"),
        }
    else:
        ctx["garment_type"] = snapshot.get("garment_type")
        seg = debug.get("segmentation") or {}
        ctx["segmentation"] = {
            "provider": seg.get("provider"),
            "mask_area_ratio": seg.get("mask_area_ratio"),
        }
    return ctx


# --------------------------------------------------------------------------- #
# Crop selection (pure; unit-tested without DB)
# --------------------------------------------------------------------------- #
def _clamp_box(b: dict, iw: int, ih: int) -> dict[str, float]:
    x = max(0.0, min(float(b.get("x", 0)), iw - 1.0))
    y = max(0.0, min(float(b.get("y", 0)), ih - 1.0))
    w = max(1.0, min(float(b.get("w", iw)), iw - x))
    h = max(1.0, min(float(b.get("h", ih)), ih - y))
    return {"x": round(x, 1), "y": round(y, 1), "w": round(w, 1), "h": round(h, 1)}


def _region_to_bbox(selected_region: dict | None) -> dict | None:
    """Stored selected_region (rect or lasso polygon) -> bbox dict."""
    if not isinstance(selected_region, dict):
        return None
    points = selected_region.get("points")
    if isinstance(points, list) and len(points) >= 3:
        try:
            xs = [float(p[0]) for p in points]
            ys = [float(p[1]) for p in points]
        except (TypeError, ValueError, IndexError):
            return None
        return {"x": min(xs), "y": min(ys), "w": max(xs) - min(xs), "h": max(ys) - min(ys)}
    if all(k in selected_region for k in ("x", "y", "w", "h")):
        return selected_region
    return None


def _issue_bbox(issues: list[dict]) -> dict | None:
    """Union of the first flagged issue's bbox + its evidence boxes, padded 20%."""
    flagged = next((i for i in issues if i.get("flagged")), None)
    if not flagged or not isinstance(flagged.get("bbox"), dict):
        return None
    boxes = [flagged["bbox"]] + [
        b for b in (flagged.get("evidence_boxes") or []) if isinstance(b, dict)
    ]
    try:
        x1 = min(float(b["x"]) for b in boxes)
        y1 = min(float(b["y"]) for b in boxes)
        x2 = max(float(b["x"]) + float(b["w"]) for b in boxes)
        y2 = max(float(b["y"]) + float(b["h"]) for b in boxes)
    except (KeyError, TypeError, ValueError):
        return None
    pad = 0.2 * max(x2 - x1, y2 - y1, 1.0)
    return {"x": x1 - pad, "y": y1 - pad, "w": (x2 - x1) + 2 * pad, "h": (y2 - y1) + 2 * pad}


def select_crop(
    image: np.ndarray,
    region_override: dict | None,
    selected_region: dict | None,
    issues: list[dict],
) -> tuple[np.ndarray, dict]:
    """Pick the most focused crop. Returns (crop_bgr, crop_box_with_source)."""
    ih, iw = image.shape[:2]
    for candidate, source in (
        (region_override, "request_region"),
        (_region_to_bbox(selected_region), "selected_region"),
        (_issue_bbox(issues), "issue_bbox"),
    ):
        if candidate:
            box = _clamp_box(candidate, iw, ih)
            crop = image[
                int(box["y"]) : int(box["y"] + box["h"]),
                int(box["x"]) : int(box["x"] + box["w"]),
            ]
            if crop.size > 0:
                return crop, {**box, "source": source}
    return image, {"x": 0.0, "y": 0.0, "w": float(iw), "h": float(ih), "source": "full_image"}


def encode_image_data_url(crop: np.ndarray, max_px: int) -> str:
    """Resize (max side <= max_px) and encode as a JPEG data URL."""
    h, w = crop.shape[:2]
    scale = max_px / max(h, w)
    if scale < 1.0:
        crop = cv2.resize(crop, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        raise AIReviewParseError("failed to encode image crop")
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


# --------------------------------------------------------------------------- #
# Prompts + structured output schema
# --------------------------------------------------------------------------- #
RESPONSE_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "detected_visual_problems": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "enum": list(VISUAL_PROBLEM_TYPES)},
                    "description": {"type": "string"},
                    "confidence": {"type": "number"},
                },
                "required": ["type", "description", "confidence"],
                "additionalProperties": False,
            },
        },
        "detector_comparison": {"type": "string"},
        "recommended_action": {"type": "string"},
        "limitations": {"type": "string"},
    },
    "required": [
        "summary",
        "detected_visual_problems",
        "detector_comparison",
        "recommended_action",
        "limitations",
    ],
    "additionalProperties": False,
}

_HAND_CHECKLIST = """Check specifically:
- finger count (count visible fingers carefully)
- extra finger-like shapes
- missing fingers
- merged fingers drawn as one mass
- malformed fingertips (blobs, claws)
- distorted wrist/palm connection
- whether the detector's MediaPipe 21-point skeleton may have normalized away a visual artifact (it always outputs a plausible 5-finger skeleton)."""

_WRINKLE_CHECKLIST = """Check specifically:
- whether the detector's flagged wrinkle issues look visually plausible
- whether any flagged issue may be a false positive
- whether wrinkles, shadows, and the garment's 3D shape align
- what the user should visually inspect next."""


def build_prompts(mode: str, language: str, context: dict) -> tuple[str, str]:
    lang_name = "Japanese" if language == "ja" else "English"
    system = (
        "You are a professional illustration reviewer assisting an automated "
        "art-error detector. You receive one image crop from an illustration "
        "and the detector's findings as JSON. Judge only what is visible; be "
        "honest about uncertainty and never invent problems. Write ALL "
        f"free-text fields in {lang_name}. The 'type' field must use the "
        "allowed enum values (English)."
    )
    checklist = _HAND_CHECKLIST if mode == "hand" else _WRINKLE_CHECKLIST
    user = (
        f"Mode: {mode} inspection.\n{checklist}\n\n"
        "Detector context (JSON):\n"
        f"{json.dumps(context, ensure_ascii=False)}\n\n"
        "In 'detector_comparison', compare your visual assessment with the "
        "detector's findings: state whether the detector likely missed "
        "something (false negative) or over-flagged something (false "
        "positive). In 'limitations', state what you cannot judge from this "
        "crop alone."
    )
    return system, user


# --------------------------------------------------------------------------- #
# Network seam + output extraction
# --------------------------------------------------------------------------- #
def _post_responses(payload: dict, api_key: str, timeout_s: float) -> dict:
    """The ONLY network call. Tests monkeypatch this function."""
    import httpx  # deferred; already a project dependency

    resp = httpx.post(
        OPENAI_RESPONSES_URL,
        json=payload,
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=timeout_s,
    )
    resp.raise_for_status()
    return resp.json()


def extract_output_text(resp: dict) -> str:
    """Pull the structured-output text from a Responses API result."""
    if isinstance(resp.get("output_text"), str) and resp["output_text"].strip():
        return resp["output_text"]
    parts: list[str] = []
    for item in resp.get("output", []) or []:
        if item.get("type") == "message":
            for content in item.get("content", []) or []:
                if content.get("type") == "output_text" and content.get("text"):
                    parts.append(content["text"])
    text = "".join(parts).strip()
    if not text:
        raise AIReviewParseError("empty model output")
    return text


def run_ai_review(
    *,
    inspection,
    image_bgr: np.ndarray,
    mode: str,
    language: str,
    region_override: dict | None,
) -> dict:
    """Full flow: context -> crop -> prompt -> OpenAI -> validated dict.

    Returns the parsed model dict plus a ``meta`` block. Raises
    AIReviewUpstreamError / AIReviewParseError for the route to map to 502.
    """
    import httpx  # for exception types only

    context = build_context(inspection)
    selected_region = None
    try:
        selected_region = json.loads(inspection.selected_region_json or "null")
    except (json.JSONDecodeError, TypeError):
        selected_region = None

    crop, crop_box = select_crop(
        image_bgr, region_override, selected_region, context.get("issues") or []
    )
    data_url = encode_image_data_url(crop, settings.ai_review_max_image_px)

    system, user = build_prompts(mode, language, context)
    payload = {
        "model": settings.openai_model,
        "input": [
            {"role": "system", "content": [{"type": "input_text", "text": system}]},
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": user},
                    {"type": "input_image", "image_url": data_url},
                ],
            },
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "ai_visual_review",
                "strict": True,
                "schema": RESPONSE_SCHEMA,
            }
        },
        "temperature": 0.2,
        "max_output_tokens": 900,
    }

    logger.info(
        "AI review: model=%s mode=%s crop=%s image_b64_len=%d",
        settings.openai_model,
        mode,
        crop_box["source"],
        len(data_url),
    )

    try:
        raw = _post_responses(payload, _resolved_key(), settings.ai_review_timeout_s)
    except httpx.HTTPError as exc:
        raise AIReviewUpstreamError(str(exc)[:200]) from exc

    try:
        parsed = json.loads(extract_output_text(raw))
    except json.JSONDecodeError as exc:
        raise AIReviewParseError(f"model returned non-JSON output: {str(exc)[:120]}") from exc
    if not isinstance(parsed, dict):
        raise AIReviewParseError("model output is not a JSON object")

    parsed["meta"] = {
        "model": settings.openai_model,
        "mode": mode,
        "language": language,
        "crop": crop_box,
    }
    return parsed
