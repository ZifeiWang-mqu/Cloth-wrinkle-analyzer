"""Tests for the anomaly models (dummy heuristic + learned Mahalanobis)."""

from __future__ import annotations

import json

import numpy as np

from app.services.anomaly_model import (
    FEATURE_KEYS,
    GLOBAL_KEY,
    DummyAnomalyModel,
    MahalanobisAnomalyModel,
    _GaussianFit,
)
from app.services.feature_extraction import Features


def _features(value: float, num_lines: int = 10) -> Features:
    kw = {k: value for k in FEATURE_KEYS}
    return Features(num_lines=num_lines, **kw)


def test_dummy_score_in_range():
    m = DummyAnomalyModel()
    for v in (0.0, 0.5, 1.0):
        s = m.score(_features(v))
        assert 0.0 <= s <= 1.0


def _identity_model() -> MahalanobisAnomalyModel:
    k = len(FEATURE_KEYS)
    fit = _GaussianFit(
        mean=np.zeros(k),
        std=np.ones(k),
        cov_inv=np.eye(k),
        p50=1.0,
        p95=3.0,
    )
    return MahalanobisAnomalyModel(FEATURE_KEYS, {GLOBAL_KEY: fit})


def test_mahalanobis_low_at_mean_high_for_outlier():
    model = _identity_model()
    near = model.score(_features(0.0))  # at the fitted mean
    far = model.score(_features(5.0))  # far outlier
    assert near == 0.0
    assert far >= 0.9
    assert near < far


def test_mahalanobis_requires_enough_lines():
    model = _identity_model()
    assert model.score(_features(5.0, num_lines=1)) == 0.0


def test_mahalanobis_prefers_garment_fit_then_global():
    k = len(FEATURE_KEYS)
    base = dict(mean=np.zeros(k), std=np.ones(k), cov_inv=np.eye(k))
    fits = {
        GLOBAL_KEY: _GaussianFit(p50=1.0, p95=3.0, **base),
        "skirt": _GaussianFit(p50=10.0, p95=20.0, **base),  # very tolerant
    }
    model = MahalanobisAnomalyModel(FEATURE_KEYS, fits)
    # The same input scores lower under the tolerant garment-specific fit.
    assert model.score(_features(3.0), "skirt") < model.score(_features(3.0), "pants")


def test_from_file_roundtrip(tmp_path):
    k = len(FEATURE_KEYS)
    payload = {
        "feature_keys": list(FEATURE_KEYS),
        "fits": {
            GLOBAL_KEY: {
                "mean": [0.0] * k,
                "std": [1.0] * k,
                "cov_inv": np.eye(k).tolist(),
                "p50": 1.0,
                "p95": 3.0,
            }
        },
    }
    path = tmp_path / "anomaly_model.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    model = MahalanobisAnomalyModel.from_file(path)
    assert 0.0 <= model.score(_features(0.0)) <= 1.0
    assert model.score(_features(6.0)) >= 0.9
