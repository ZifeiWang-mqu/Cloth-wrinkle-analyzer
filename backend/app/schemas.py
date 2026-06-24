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
    # Raw rule score (0..1), the effective threshold applied, and whether the
    # score crossed it. The frontend can re-filter on `score` independently of
    # the backend judgment (`flagged`).
    score: float = Field(0.0, ge=0.0, le=1.0)
    threshold: float = Field(0.0, ge=0.0, le=1.0)
    flagged: bool = True


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
# Inspection input (base64 / external tools)
# --------------------------------------------------------------------------- #
class InspectBase64Request(BaseModel):
    image_base64: str  # raw base64 or a data URL ("data:image/png;base64,...")
    garment_type: str | None = None
    selected_region: BBox | None = None
    source: str = "external"  # "web" | "photoshop" | "external"


# --------------------------------------------------------------------------- #
# Feedback
# --------------------------------------------------------------------------- #
class FeedbackRequest(BaseModel):
    inspection_id: str
    issue_id: str | None = None
    feedback: FeedbackKind
    # Rich context (all optional) so feedback can later seed a training set.
    image_id: str | None = None
    garment_type: str | None = None
    issue_type: str | None = None
    original_bbox: BBox | None = None
    corrected_bbox: BBox | None = None
    corrected_type: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    severity: str | None = None
    source: str | None = None
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


# --------------------------------------------------------------------------- #
# Model status / inspection detail (external tools)
# --------------------------------------------------------------------------- #
class ModelStatus(BaseModel):
    # Allow field names starting with "model_" (reserved by Pydantic v2).
    model_config = {"protected_namespaces": ()}

    model_loaded: bool
    model_type: str
    model_path: str
    reference_stats_loaded: bool
    reference_stats_path: str
    thresholds_loaded: bool
    thresholds_path: str
    available_garment_models: list[str] = Field(default_factory=list)
    version: str


class StoredFeedback(BaseModel):
    id: str
    issue_id: str | None = None
    feedback: str
    image_id: str | None = None
    garment_type: str | None = None
    issue_type: str | None = None
    corrected_type: str | None = None
    comment: str | None = None
    created_at: str | None = None


class InspectionDetail(BaseModel):
    inspection_id: str
    image_path: str
    image_filename: str
    garment_type: str
    result: str
    overall_score: float
    issues: list[Issue] = Field(default_factory=list)
    debug: DebugInfo | None = None
    feedback: list[StoredFeedback] = Field(default_factory=list)
    created_at: str | None = None
