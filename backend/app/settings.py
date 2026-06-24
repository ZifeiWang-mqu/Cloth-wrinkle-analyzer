"""Application configuration.

All values can be overridden via environment variables (prefix ``WRINKLE_``)
or a local ``.env`` file. Keeping config in one place makes the MVP easy to
tune without touching business logic.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Repository layout:  <repo>/backend/app/settings.py  ->  BASE_DIR = <repo>
BACKEND_DIR = Path(__file__).resolve().parent.parent
REPO_DIR = BACKEND_DIR.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="WRINKLE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- General ---
    app_name: str = "Illustration Wrinkle Inconsistency Detector"
    debug: bool = True

    # --- Storage ---
    # Uploaded images and the SQLite DB live under <repo>/data by default.
    data_dir: Path = REPO_DIR / "data"
    upload_subdir: str = "illustrations_raw"
    models_subdir: str = "models"
    db_filename: str = "wrinkle.db"
    anomaly_model_filename: str = "anomaly_model.json"

    # --- Uploads ---
    allowed_extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp")
    max_upload_bytes: int = 15 * 1024 * 1024  # 15 MB

    # --- CORS (Next.js dev server defaults) ---
    cors_origins: tuple[str, ...] = (
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    )
    # Allow any origin (for Photoshop UXP / external tools). Disables credentials.
    cors_allow_all: bool = False

    # --- Scoring thresholds ---
    # A requirement becomes an "issue" once its score crosses issue_threshold
    # (fallback when thresholds.json is not loaded).
    issue_threshold: float = 0.45
    # Issues with score >= min_report_score are returned to the client (even if
    # below the judgment threshold) so the UI display slider can reveal them.
    min_report_score: float = 0.30
    # overall_score above review_threshold -> result == "needs_review".
    review_threshold: float = 0.4
    severity_medium: float = 0.55
    severity_high: float = 0.75
    # How much the learned anomaly model contributes to overall_score (0..1).
    anomaly_weight: float = 0.15

    @property
    def upload_dir(self) -> Path:
        return self.data_dir / self.upload_subdir

    @property
    def models_dir(self) -> Path:
        return self.data_dir / self.models_subdir

    @property
    def model_path(self) -> Path:
        return self.models_dir / self.anomaly_model_filename

    @property
    def thresholds_path(self) -> Path:
        return BACKEND_DIR / "app" / "config" / "thresholds.json"

    @property
    def reference_stats_path(self) -> Path:
        return self.data_dir / "features" / "reference_stats.json"

    @property
    def db_path(self) -> Path:
        return self.data_dir / self.db_filename

    @property
    def database_url(self) -> str:
        return f"sqlite:///{self.db_path}"

    def ensure_dirs(self) -> None:
        """Create the directories the app writes to (idempotent)."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        (self.data_dir / "outputs").mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
