"""Anomaly model interface + a simple learned model.

Two implementations ship here:

* :class:`DummyAnomalyModel` — deterministic heuristic fallback (no training).
* :class:`MahalanobisAnomalyModel` — a genuinely *learned* one-class model. It
  fits a multivariate Gaussian to the structural features of natural (photo)
  wrinkles and scores how far a new patch deviates (Mahalanobis distance,
  calibrated to 0..1). It is trained by ``ml/train_anomaly_model.py`` and
  persisted as numpy-only JSON, so the backend loads it without extra deps.

This keeps the "swap the model later" promise: a future DINOv2 + PatchCore /
Anomalib model only needs to subclass :class:`BaseAnomalyModel` and be returned
from :func:`get_anomaly_model`.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from app.services.feature_extraction import Features
from app.settings import settings

logger = logging.getLogger(__name__)

# Scale-invariant features only — absolute counts/lengths depend on region size
# and would make a full-image inspection look anomalous vs. small photo patches.
FEATURE_KEYS: tuple[str, ...] = (
    "edge_density",
    "line_count_density",
    "orientation_dispersion",
    "convergence_strength",
    "gradient_coherence",
    "local_contrast",
)

GLOBAL_KEY = "_global"


def _clamp01(v: float) -> float:
    return float(max(0.0, min(1.0, v)))


class BaseAnomalyModel(ABC):
    name: str = "base"

    @abstractmethod
    def score(self, features: Features, garment_type: str = "unknown") -> float:
        """Return an anomaly score in [0, 1] (0 = normal, 1 = anomalous)."""

    def load(self) -> None:  # pragma: no cover - dummy has nothing to load
        return None


class DummyAnomalyModel(BaseAnomalyModel):
    """Heuristic stand-in used when no trained model is present."""

    name = "dummy"

    def score(self, features: Features, garment_type: str = "unknown") -> float:
        disorder = features.orientation_dispersion  # 0..1
        incoherent = 1.0 - features.gradient_coherence  # 0..1
        raw = 0.6 * disorder + 0.4 * incoherent
        if features.num_lines < 3:
            raw *= 0.3  # too little line work to judge
        return _clamp01(raw)


class _GaussianFit:
    """Per-garment fitted Gaussian + distance calibration."""

    def __init__(
        self,
        mean: np.ndarray,
        std: np.ndarray,
        cov_inv: np.ndarray,
        p50: float,
        p95: float,
    ) -> None:
        self.mean = mean
        self.std = std
        self.cov_inv = cov_inv
        self.p50 = p50
        self.p95 = p95

    def distance(self, vec: np.ndarray) -> float:
        z = (vec - self.mean) / self.std
        d2 = float(z @ self.cov_inv @ z)
        return float(np.sqrt(max(0.0, d2)))

    def score(self, vec: np.ndarray) -> float:
        d = self.distance(vec)
        denom = (self.p95 - self.p50) or 1e-6
        # ~0 at the training median, ~1 at the 95th percentile and beyond.
        return _clamp01((d - self.p50) / denom)

    @classmethod
    def from_dict(cls, d: dict) -> "_GaussianFit":
        return cls(
            mean=np.asarray(d["mean"], dtype=np.float64),
            std=np.asarray(d["std"], dtype=np.float64),
            cov_inv=np.asarray(d["cov_inv"], dtype=np.float64),
            p50=float(d["p50"]),
            p95=float(d["p95"]),
        )


class MahalanobisAnomalyModel(BaseAnomalyModel):
    """Learned one-class model (multivariate Gaussian per garment)."""

    name = "mahalanobis"

    def __init__(self, feature_keys: tuple[str, ...], fits: dict[str, _GaussianFit]):
        self.feature_keys = feature_keys
        self.fits = fits

    def _vector(self, features: Features) -> np.ndarray:
        return np.asarray(
            [float(getattr(features, k, 0.0)) for k in self.feature_keys],
            dtype=np.float64,
        )

    def score(self, features: Features, garment_type: str = "unknown") -> float:
        if features.num_lines < 3:
            return 0.0  # not enough evidence
        fit = self.fits.get(garment_type) or self.fits.get(GLOBAL_KEY)
        if fit is None:
            return 0.0
        return fit.score(self._vector(features))

    @classmethod
    def from_file(cls, path: Path) -> "MahalanobisAnomalyModel":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        keys = tuple(data.get("feature_keys", FEATURE_KEYS))
        fits = {
            garment: _GaussianFit.from_dict(fit)
            for garment, fit in data.get("fits", {}).items()
        }
        if not fits:
            raise ValueError("model file has no fits")
        return cls(feature_keys=keys, fits=fits)


_MODEL: BaseAnomalyModel | None = None


def get_anomaly_model() -> BaseAnomalyModel:
    """Return the best available model (cached).

    Loads the trained Mahalanobis model from ``settings.model_path`` if present
    and valid; otherwise falls back to the dummy heuristic.
    """
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    path = settings.model_path
    if path.exists():
        try:
            _MODEL = MahalanobisAnomalyModel.from_file(path)
            logger.info("Loaded trained anomaly model: %s", path)
            return _MODEL
        except Exception as exc:
            logger.warning("Failed to load anomaly model (%s); using dummy.", exc)

    _MODEL = DummyAnomalyModel()
    return _MODEL


def reset_anomaly_model_cache() -> None:
    """Clear the cached model (useful after retraining or in tests)."""
    global _MODEL
    _MODEL = None
