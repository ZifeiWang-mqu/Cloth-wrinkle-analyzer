"""Pydantic schemas shared across the API.

These mirror the contract documented in the requirements (section 6) and are
the single source of truth for request/response shapes. The TypeScript client
should be kept in sync with these definitions.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, field_validator


# --------------------------------------------------------------------------- #
# Enums
# --------------------------------------------------------------------------- #
class GarmentType(str, Enum):
    shirt = "shirt"
    skirt = "skirt"
    pants = "pants"
    dress = "dress"
    jacket = "jacket"
    unknown = "unknown"


class IssueType(str, Enum):
    gravity_inconsistency = "gravity_inconsistency"
    joint_inconsistency = "joint_inconsistency"
    tension_ambiguity = "tension_ambiguity"
    body_volume_inconsistency = "body_volume_inconsistency"
    density_inconsistency = "density_inconsistency"
    shadow_wrinkle_mismatch = "shadow_wrinkle_mismatch"


class Severity(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"


class InspectResult(str, Enum):
    ok = "ok"
    needs_review = "needs_review"


class FeedbackKind(str, Enum):
    correct = "correct"
    false_positive = "false_positive"
    missed_issue = "missed_issue"
    wrong_location = "wrong_location"
    wrong_reason = "wrong_reason"


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #
class BBox(BaseModel):
    x: float = Field(..., ge=0)
    y: float = Field(..., ge=0)
    w: float = Field(..., gt=0)
    h: float = Field(..., gt=0)


# --------------------------------------------------------------------------- #
# Inspection
# --------------------------------------------------------------------------- #
class Issue(BaseModel):
    id: str
    type: IssueType
    label: str  # 日本語の表示名
    severity: Severity
    bbox: BBox
    confidence: float = Field(..., ge=0.0, le=1.0)
    message: str  # 日本語の説明文


class DebugInfo(BaseModel):
    pose_detected: bool = False
    garment_region_used: bool = False
    num_wrinkle_candidates: int = 0
    processing_time_ms: int = 0
    # Per-requirement raw scores (handy for the frontend debug panel / tuning).
    scores: dict[str, float] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


class InspectResponse(BaseModel):
    inspection_id: str
    result: InspectResult
    overall_score: float = Field(..., ge=0.0, le=1.0)
    issues: list[Issue] = Field(default_factory=list)
    debug: DebugInfo


# --------------------------------------------------------------------------- #
# Feedback
# --------------------------------------------------------------------------- #
class FeedbackRequest(BaseModel):
    inspection_id: str
    issue_id: str | None = None
    feedback: FeedbackKind
    corrected_bbox: BBox | None = None
    corrected_type: str | None = None
    comment: str | None = None

    @field_validator("comment")
    @classmethod
    def _trim_comment(cls, v: str | None) -> str | None:
        if v is None:
            return None
        v = v.strip()
        return v or None


class FeedbackResponse(BaseModel):
    status: str = "saved"
    feedback_id: str
