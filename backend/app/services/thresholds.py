"""Per-issue-type / per-garment judgment thresholds.

Loaded from ``backend/app/config/thresholds.json``. These are the *backend*
judgment thresholds (used to decide whether an issue is ``flagged`` and whether
the result is ``needs_review``). They are intentionally separate from the
frontend *display* threshold (a UI slider), per requirement §7.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class Thresholds:
    global_threshold: float = 0.75
    issue_type: dict[str, float] = field(default_factory=dict)
    garment_type: dict[str, float] = field(default_factory=dict)
    loaded: bool = False
    path: str = ""

    def effective(self, issue_type: str, garment_type: str) -> float:
        """Effective judgment threshold = mean of the issue-type and garment
        thresholds (each falling back to the global value)."""
        t = self.issue_type.get(issue_type, self.global_threshold)
        g = self.garment_type.get(garment_type, self.global_threshold)
        return (t + g) / 2.0

    def to_dict(self) -> dict:
        return {
            "global": self.global_threshold,
            "issue_type": self.issue_type,
            "garment_type": self.garment_type,
        }


_CACHE: dict[str, Thresholds] = {}


def load_thresholds(path: Path) -> Thresholds:
    key = str(path)
    if key in _CACHE:
        return _CACHE[key]

    th = Thresholds(path=key)
    try:
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            th.global_threshold = float(data.get("global", 0.75))
            th.issue_type = {k: float(v) for k, v in data.get("issue_type", {}).items()}
            th.garment_type = {
                k: float(v) for k, v in data.get("garment_type", {}).items()
            }
            th.loaded = True
            logger.info("Loaded thresholds: %s", path)
        else:
            logger.info("thresholds.json not found (%s); using defaults.", path)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Failed to load thresholds (%s); using defaults.", exc)

    _CACHE[key] = th
    return th


def reset_cache() -> None:
    _CACHE.clear()
