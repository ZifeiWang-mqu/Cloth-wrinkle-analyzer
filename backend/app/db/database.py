"""SQLite engine / session management (SQLAlchemy 2.0)."""

from __future__ import annotations

import logging
from collections.abc import Iterator

from sqlalchemy import create_engine, inspect as sa_inspect, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.settings import settings

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    pass


# check_same_thread=False is required because FastAPI may use the connection
# across threads. SQLite is fine for the MVP's single-process use case.
engine = create_engine(
    settings.database_url,
    connect_args={"check_same_thread": False},
    future=True,
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


# Additive migrations: column name -> SQLite type. Applied if missing so old
# DBs gain new columns without losing data (no Alembic needed for the MVP).
_ISSUE_FEEDBACK_COLUMNS = {
    "image_id": "VARCHAR(512)",
    "garment_type": "VARCHAR(32)",
    "issue_type": "VARCHAR(64)",
    "original_bbox_json": "TEXT",
    "confidence": "FLOAT",
    "severity": "VARCHAR(16)",
    "source": "VARCHAR(32)",
}


def _apply_additive_migrations() -> None:
    """Add any newly-introduced columns to existing tables (idempotent)."""
    insp = sa_inspect(engine)
    if "issue_feedback" not in insp.get_table_names():
        return
    existing = {c["name"] for c in insp.get_columns("issue_feedback")}
    missing = {k: v for k, v in _ISSUE_FEEDBACK_COLUMNS.items() if k not in existing}
    if not missing:
        return
    with engine.begin() as conn:
        for name, sqltype in missing.items():
            conn.execute(text(f"ALTER TABLE issue_feedback ADD COLUMN {name} {sqltype}"))
            logger.info("Migrated issue_feedback: added column %s", name)


def init_db() -> None:
    """Create tables if they do not exist, then apply additive migrations."""
    settings.ensure_dirs()
    # Import models so they register on Base.metadata before create_all.
    from app.db import models  # noqa: F401

    Base.metadata.create_all(bind=engine)
    _apply_additive_migrations()


def get_db() -> Iterator[Session]:
    """FastAPI dependency yielding a scoped session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
