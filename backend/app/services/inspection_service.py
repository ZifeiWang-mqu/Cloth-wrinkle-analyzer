"""Shared inspection logic used by all inspect endpoints.

Both ``/api/inspect-wrinkle`` (file upload) and ``/api/inspect-base64``
(external tools) decode an image to a BGR ndarray + a persisted path, then call
:func:`run_inspection`. This avoids duplicating the pipeline/persistence logic.

Pipeline: region -> [segmentation mask] -> masked wrinkle lines -> features
(+ advanced joint geometry) -> anomaly + illustration scores -> rules.
All optional stages degrade gracefully (requirement §1/§13).
"""

from __future__ import annotations

import base64
import json
import logging
import math
import time
from pathlib import Path

import cv2
import numpy as np
from sqlalchemy.orm import Session

from app.db.models import Inspection
from app.schemas import DebugInfo, InspectResponse
from app.services import capabilities
from app.services.anomaly_model import get_anomaly_model
from app.services.feature_extraction import extract_features
from app.services.illustration_model import get_illustration_model, is_ready
from app.services.pose import estimate_pose, get_joint_contexts
from app.services.reference_stats import load_density_stats
from app.services.rule_engine import integrate_scores
from app.services.segmentation import get_garment_region, segment_garment
from app.services.thresholds import load_thresholds
from app.services.wrinkle_edges import extract_wrinkle_candidates
from app.settings import settings

logger = logging.getLogger(__name__)


def _mask_png_base64(mask: np.ndarray) -> str | None:
    try:
        ok, buf = cv2.imencode(".png", mask)
        if not ok:
            return None
        return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()
    except Exception:  # pragma: no cover
        return None


def run_inspection(
    *,
    image_bgr: np.ndarray,
    garment: str,
    region_dict: dict | None,
    db: Session,
    saved_path: Path,
    image_filename: str | None,
    source: str = "web",
    use_segmentation: bool = False,
    segmentation_provider: str | None = None,
    use_pose_advanced: bool = True,
    use_illustration_model: bool = True,
    return_debug_overlays: bool = False,
) -> InspectResponse:
    """Run the full inspection pipeline, persist history, and return the result."""
    t0 = time.perf_counter()
    notes: list[str] = []
    issues: list[dict] = []
    overall = 0.0
    result_str = "ok"
    scores: dict[str, float] = {}
    model_scores: dict = {}
    num_candidates = 0
    pose_detected = False
    region_used = False
    seg_debug: dict | None = None
    pose_debug: dict | None = None
    removed_lines: dict[str, int] = {}
    line_filter: dict[str, int] = {}
    overlays: dict | None = None
    models_used = {
        "rule_engine": True,
        "photo_anomaly_model": True,
        "illustration_feedback_model": False,
    }

    try:
        ih, iw = image_bgr.shape[:2]
        region = get_garment_region(image_bgr, region_dict)
        region_used = region.used_selection

        # --- Segmentation (optional) ---
        seg_mask = None
        seg = segment_garment(
            image_bgr,
            region,
            settings,
            use_segmentation=use_segmentation or settings.enable_auto_segmentation,
            provider_override=segmentation_provider,
        )
        seg_debug = seg.to_debug()
        if seg.mask_available:
            seg_mask = seg.mask

        # --- Pose (always estimated; advanced geometry gated by flag) ---
        pose = estimate_pose(image_bgr)
        pose_detected = pose.detected
        joint_contexts = (
            get_joint_contexts(pose, (ih, iw)) if (pose.detected and use_pose_advanced) else []
        )
        pose_debug = {
            "detected": pose.detected,
            "provider": pose.backend,
            "advanced": use_pose_advanced,
            "joint_contexts": [c.to_debug() for c in joint_contexts],
        }

        # --- Wrinkle candidates (masked) ---
        candidates = extract_wrinkle_candidates(
            image_bgr, region, seg_mask=seg_mask, settings=settings
        )
        num_candidates = candidates.count
        line_filter = candidates.line_filter or {}
        removed_lines = {
            k.replace("removed_", ""): v
            for k, v in line_filter.items()
            if k.startswith("removed_")
        }

        # image_shape enables advanced joint features; None -> basic joint rule.
        features = extract_features(
            candidates, pose, image_shape=(ih, iw) if use_pose_advanced else None
        )

        anomaly_score = get_anomaly_model().score(features, garment)

        # --- Illustration feedback model (optional) ---
        illustration_score = None
        if use_illustration_model and is_ready():
            model = get_illustration_model()
            if model is not None:
                illustration_score = model.score(features.to_dict(), garment)
                models_used["illustration_feedback_model"] = illustration_score is not None

        region_context = {
            "bbox": region.as_dict(),
            "image_w": iw,
            "image_h": ih,
            "region_diag": math.hypot(region.w, region.h),
        }
        thresholds = load_thresholds(settings.thresholds_path)
        density_stats, _ = load_density_stats(settings.reference_stats_path)

        payload = integrate_scores(
            features,
            garment,
            pose,
            region_context,
            settings,
            reference_stats=density_stats,
            anomaly_score=anomaly_score,
            thresholds=thresholds,
            illustration_score=illustration_score,
        )
        issues = payload["issues"]
        overall = payload["overall_score"]
        result_str = payload["result"]
        scores = payload["scores"]
        model_scores = payload.get("model_scores", {})

        if not pose.detected:
            notes.append("pose_not_detected: 関節系の判定は低信頼度です。")
        if seg.fallback_used:
            notes.append(f"segmentation_fallback: {seg.reason}")

        # --- Debug overlays (optional, heavier) ---
        if return_debug_overlays:
            overlays = {
                "candidate_lines": [
                    [
                        round(ln.x1 + candidates.offset[0], 1),
                        round(ln.y1 + candidates.offset[1], 1),
                        round(ln.x2 + candidates.offset[0], 1),
                        round(ln.y2 + candidates.offset[1], 1),
                    ]
                    for ln in candidates.lines[:400]
                ],
                "pose_landmarks": {
                    name: [round(lm["x"], 1), round(lm["y"], 1)]
                    for name, lm in pose.landmarks.items()
                }
                if pose.detected
                else {},
            }
            if seg_mask is not None:
                overlays["mask_png"] = _mask_png_base64(seg_mask)
    except Exception as exc:  # pragma: no cover - defensive top-level guard
        logger.exception("Inspection pipeline failed")
        notes.append(f"pipeline_error: {exc}")

    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    debug = DebugInfo(
        pose_detected=pose_detected,
        garment_region_used=region_used,
        num_wrinkle_candidates=num_candidates,
        processing_time_ms=elapsed_ms,
        scores=scores,
        notes=notes,
        segmentation=seg_debug,
        pose=pose_debug,
        model_scores=model_scores,
        models_used=models_used,
        capabilities=capabilities.get_capabilities(settings),
        removed_lines=removed_lines,
        line_filter=line_filter,
        overlays=overlays,
    )

    row = Inspection(
        image_path=str(saved_path),
        image_filename=image_filename or saved_path.name,
        garment_type=garment,
        selected_region_json=json.dumps(region_dict) if region_dict else None,
        overall_score=overall,
        result=result_str,
        issues_json=json.dumps(issues, ensure_ascii=False),
        debug_json=json.dumps(debug.model_dump(exclude={"overlays"}), ensure_ascii=False),
    )
    db.add(row)
    db.commit()
    db.refresh(row)

    return InspectResponse(
        inspection_id=row.id,
        result=result_str,
        overall_score=overall,
        issues=issues,
        debug=debug,
    )
