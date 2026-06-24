"""Tests for threshold loading and effective-threshold logic."""

from __future__ import annotations

import json

from app.services.thresholds import Thresholds, load_thresholds, reset_cache
from app.settings import settings


def test_shipped_thresholds_load():
    th = load_thresholds(settings.thresholds_path)
    assert th.loaded is True
    # Effective threshold is the mean of issue-type and garment thresholds.
    eff = th.effective("gravity_inconsistency", "skirt")
    assert 0.0 <= eff <= 1.0


def test_effective_falls_back_to_global():
    th = Thresholds(global_threshold=0.8, issue_type={}, garment_type={}, loaded=True)
    assert th.effective("anything", "whatever") == 0.8


def test_missing_file_defaults(tmp_path):
    reset_cache()
    missing = tmp_path / "nope.json"
    th = load_thresholds(missing)
    assert th.loaded is False
    assert th.effective("x", "y") == th.global_threshold
    reset_cache()


def test_effective_is_mean(tmp_path):
    reset_cache()
    p = tmp_path / "thresholds.json"
    p.write_text(
        json.dumps(
            {
                "global": 0.7,
                "issue_type": {"gravity_inconsistency": 0.6},
                "garment_type": {"skirt": 0.8},
            }
        ),
        encoding="utf-8",
    )
    th = load_thresholds(p)
    assert abs(th.effective("gravity_inconsistency", "skirt") - 0.7) < 1e-9
    reset_cache()
