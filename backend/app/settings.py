"""Application configuration.

All values can be overridden via environment variables (prefix ``WRINKLE_``)
or a local ``.env`` file. Keeping config in one place makes the MVP easy to
tune without touching business logic.
"""

from __future__ import annotations

import os
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

    # --- CORS (comma-separated origins; env: WRINKLE_CORS_ORIGINS) ---
    cors_origins: str = "http://localhost:3000,http://127.0.0.1:3000"
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

    # --- Segmentation (SAM / OpenCV fallback) ---
    # Whether auto-segmentation runs by default (per-request flag overrides).
    enable_auto_segmentation: bool = False
    segmentation_provider: str = "opencv"  # "none" | "sam" | "opencv"
    sam_checkpoint_path: str = ""  # absolute or relative; empty -> data/models/sam/
    sam_model_type: str = "vit_b"
    sam_device: str = "cpu"  # "cpu" | "cuda" (falls back to cpu if no GPU)
    sam_lazy_load: bool = True  # load SAM on first use instead of at startup
    sam_download_url: str = ""  # optional checkpoint download URL
    sam_auto_download: bool = False  # download checkpoint at runtime if missing
    segmentation_fallback: str = "manual_bbox"  # "manual_bbox" | "whole_image"
    segmentation_min_area_ratio: float = 0.02
    segmentation_max_area_ratio: float = 0.85

    # --- Hand inspection (MediaPipe HandLandmarker; Tasks API) ---
    # Path to hand_landmarker.task (env: WRINKLE_HAND_MODEL_PATH).
    # Empty -> auto-discover under data/models/hand/. Model is git-ignored.
    hand_model_path: str = ""

    # --- AI visual review (OpenAI; OPTIONAL, server-side only) ---
    # Key: WRINKLE_OPENAI_API_KEY, falling back to plain OPENAI_API_KEY.
    # Never exposed to the frontend; never logged.
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    ai_review_max_image_px: int = 768  # crop is resized so max side <= this
    ai_review_timeout_s: float = 60.0
    # Drop lines whose midpoint is within this fraction of the diagonal of the
    # mask boundary (likely garment outline, not a wrinkle). 0 disables.
    mask_boundary_margin_ratio: float = 0.015

    # --- Illustration feedback model ---
    illustration_model_filename: str = "illustration_feedback_model.joblib"
    illustration_metrics_filename: str = "illustration_feedback_model_metrics.json"
    # Weight of the illustration model in the final score (only when ready).
    illustration_model_weight: float = 0.25
    min_feedback_train_samples: int = 30

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

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
    def sam_dir(self) -> Path:
        return self.models_dir / "sam"

    @property
    def resolved_sam_checkpoint(self) -> Path | None:
        """Resolve the SAM checkpoint path, or None if not configured/found."""
        if self.sam_checkpoint_path:
            p = Path(self.sam_checkpoint_path)
            if not p.is_absolute():
                p = REPO_DIR / p
            return p
        # Fall back to the first *.pth under data/models/sam/.
        if self.sam_dir.exists():
            for pth in sorted(self.sam_dir.glob("*.pth")):
                return pth
        return None

    @property
    def hand_models_dir(self) -> Path:
        return self.models_dir / "hand"

    @property
    def resolved_hand_model(self) -> Path | None:
        """Resolve the MediaPipe HandLandmarker ``.task`` model, or None.

        Order: explicit ``WRINKLE_HAND_MODEL_PATH`` (if the file exists) ->
        ``data/models/hand/hand_landmarker.task`` -> first ``*.task`` in that
        directory. The model file is never committed to git (data/models is
        ignored); see README for the download location.
        """
        if self.hand_model_path:
            p = Path(self.hand_model_path)
            if not p.is_absolute():
                p = REPO_DIR / p
            return p if p.exists() else None
        default = self.hand_models_dir / "hand_landmarker.task"
        if default.exists():
            return default
        if self.hand_models_dir.exists():
            for task in sorted(self.hand_models_dir.glob("*.task")):
                return task
        return None

    @property
    def resolved_openai_key(self) -> str:
        """WRINKLE_OPENAI_API_KEY (env or .env, captured at process start),
        falling back to the conventional OPENAI_API_KEY. Whitespace-stripped —
        a trailing newline in .env must not produce a 401."""
        key = (self.openai_api_key or "").strip()
        if key:
            return key
        return (os.getenv("OPENAI_API_KEY") or "").strip()

    def openai_key_diagnostics(self) -> dict:
        """Safe key diagnostics: source, short prefix, length — NEVER the key."""
        wrinkle = (self.openai_api_key or "").strip()
        fallback = (os.getenv("OPENAI_API_KEY") or "").strip()
        key = wrinkle or fallback
        if wrinkle:
            source = "WRINKLE_OPENAI_API_KEY"
        elif fallback:
            source = "OPENAI_API_KEY"
        else:
            source = "none"
        return {
            "key_source": source,
            "key_prefix": key[:8] if key else None,
            "key_length": len(key),
            # True when a WRINKLE_ key is shadowing a DIFFERENT OPENAI_API_KEY —
            # the classic "shell test works but the app 401s" situation.
            "shadows_openai_api_key": bool(wrinkle and fallback and wrinkle != fallback),
        }

    @property
    def illustration_model_path(self) -> Path:
        return self.models_dir / self.illustration_model_filename

    @property
    def illustration_metrics_path(self) -> Path:
        return self.models_dir / self.illustration_metrics_filename

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
