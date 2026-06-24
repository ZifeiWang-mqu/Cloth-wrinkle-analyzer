"""ORM models for inspection history and user feedback.

JSON-heavy columns are stored as TEXT (SQLite) holding serialized JSON. This
keeps the MVP flexible while we iterate on the issue/debug shapes. Feedback is
always persisted so it can later seed an illustration-specific dataset.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import Float, ForeignKey, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.database import Base


def _uuid() -> str:
    return uuid.uuid4().hex


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Inspection(Base):
    __tablename__ = "inspections"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    image_path: Mapped[str] = mapped_column(String(1024))
    image_filename: Mapped[str] = mapped_column(String(512))
    garment_type: Mapped[str] = mapped_column(String(32), default="unknown")
    selected_region_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    overall_score: Mapped[float] = mapped_column(Float, default=0.0)
    result: Mapped[str] = mapped_column(String(32), default="ok")
    issues_json: Mapped[str] = mapped_column(Text, default="[]")
    debug_json: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[datetime] = mapped_column(default=_utcnow, server_default=func.now())

    feedback: Mapped[list["IssueFeedback"]] = relationship(
        back_populates="inspection",
        cascade="all, delete-orphan",
    )


class IssueFeedback(Base):
    __tablename__ = "issue_feedback"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    inspection_id: Mapped[str] = mapped_column(
        ForeignKey("inspections.id", ondelete="CASCADE"), index=True
    )
    issue_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    feedback: Mapped[str] = mapped_column(String(32))
    # Rich context captured for building an illustration-specific dataset.
    image_id: Mapped[str | None] = mapped_column(String(512), nullable=True)
    garment_type: Mapped[str | None] = mapped_column(String(32), nullable=True)
    issue_type: Mapped[str | None] = mapped_column(String(64), nullable=True)
    original_bbox_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    corrected_bbox_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    corrected_type: Mapped[str | None] = mapped_column(String(64), nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    severity: Mapped[str | None] = mapped_column(String(16), nullable=True)
    source: Mapped[str | None] = mapped_column(String(32), nullable=True)
    comment: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(default=_utcnow, server_default=func.now())

    inspection: Mapped[Inspection] = relationship(back_populates="feedback")
