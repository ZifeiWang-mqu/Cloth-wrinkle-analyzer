"""Illustration-specific feedback model (optional).

A lightweight scikit-learn classifier trained from user feedback
(valid_issue vs false_issue). Optional: if scikit-learn/joblib or the model
file are missing, everything returns "not ready" and inference falls back.

The feature vector is shared between training (``ml/train_illustration_feedback_model.py``)
and inference via :func:`feature_row` so columns stay aligned.
"""

from __future__ import annotations

import json
import logging

from app.settings import settings

logger = logging.getLogger(__name__)

# Numeric features (all scale-invariant or bounded) pulled from Features.to_dict().
ILLU_NUMERIC_KEYS: tuple[str, ...] = (
    "edge_density",
    "line_count_density",
    "orientation_dispersion",
    "convergence_strength",
    "gradient_coherence",
    "local_contrast",
    "vertical_fraction",
    "horizontal_fraction",
    "angle_diff_from_gravity",
)
ILLU_GARMENTS: tuple[str, ...] = ("shirt", "skirt", "pants", "dress", "jacket", "unknown")


def feature_row(features_dict: dict, garment: str) -> dict[str, float]:
    """Build a flat, consistent feature row (numeric + garment one-hot)."""
    row: dict[str, float] = {}
    for k in ILLU_NUMERIC_KEYS:
        v = features_dict.get(k, 0.0)
        try:
            row[k] = float(v) if v is not None else 0.0
        except (TypeError, ValueError):
            row[k] = 0.0
    g = garment if garment in ILLU_GARMENTS else "unknown"
    for cand in ILLU_GARMENTS:
        row[f"garment_{cand}"] = 1.0 if cand == g else 0.0
    return row


def column_order() -> list[str]:
    return list(ILLU_NUMERIC_KEYS) + [f"garment_{g}" for g in ILLU_GARMENTS]


class IllustrationModel:
    def __init__(self, model, columns: list[str], metrics: dict | None):
        self._model = model
        self._columns = columns
        self.metrics = metrics or {}

    def score(self, features_dict: dict, garment: str) -> float | None:
        try:
            row = feature_row(features_dict, garment)
            vec = [[row.get(c, 0.0) for c in self._columns]]
            if hasattr(self._model, "predict_proba"):
                proba = self._model.predict_proba(vec)[0]
                # Probability of the positive class (valid_issue == 1).
                classes = list(getattr(self._model, "classes_", [0, 1]))
                idx = classes.index(1) if 1 in classes else len(proba) - 1
                return float(proba[idx])
            pred = self._model.predict(vec)[0]
            return float(pred)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Illustration model scoring failed: %s", exc)
            return None


_MODEL: IllustrationModel | None = None
_TRIED = False


def _load_metrics() -> dict:
    try:
        if settings.illustration_metrics_path.exists():
            return json.loads(settings.illustration_metrics_path.read_text("utf-8"))
    except Exception:  # pragma: no cover
        pass
    return {}


def get_illustration_model() -> IllustrationModel | None:
    """Load the trained model if available (cached). None if unavailable."""
    global _MODEL, _TRIED
    if _MODEL is not None:
        return _MODEL
    if _TRIED:
        return None
    _TRIED = True
    path = settings.illustration_model_path
    if not path.exists():
        return None
    try:
        import joblib  # type: ignore

        bundle = joblib.load(path)
        model = bundle.get("model") if isinstance(bundle, dict) else bundle
        columns = (
            bundle.get("columns", column_order())
            if isinstance(bundle, dict)
            else column_order()
        )
        metrics = bundle.get("metrics") if isinstance(bundle, dict) else None
        if metrics is None:
            metrics = _load_metrics()
        _MODEL = IllustrationModel(model, columns, metrics)
        logger.info("Loaded illustration feedback model: %s", path)
        return _MODEL
    except Exception as exc:
        logger.warning("Failed to load illustration model (%s).", exc)
        return None


def reset_cache() -> None:
    global _MODEL, _TRIED
    _MODEL = None
    _TRIED = False


def is_ready() -> bool:
    """True if a model is loaded AND trained on enough samples."""
    m = get_illustration_model()
    if m is None:
        return False
    n = int(m.metrics.get("training_samples", 0))
    return n >= settings.min_feedback_train_samples


def describe() -> dict:
    """Status summary for /api/model/status."""
    metrics = _load_metrics()
    m = get_illustration_model()
    loaded = m is not None
    return {
        "loaded": loaded,
        "ready": is_ready(),
        "path": str(settings.illustration_model_path),
        "training_samples": int(metrics.get("training_samples", 0)),
        "positive_samples": int(metrics.get("positive_samples", 0)),
        "negative_samples": int(metrics.get("negative_samples", 0)),
        "metrics": metrics.get("metrics", {}),
    }
