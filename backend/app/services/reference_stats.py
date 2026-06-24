"""Loader for photo-derived reference statistics.

``data/features/reference_stats.json`` is produced by
``ml/extract_photo_features.py`` and contains per-garment mean/std of
structural metrics. Here we expose the ``line_count_density`` stats in the shape
the density rule expects: ``{garment: (mean, std)}`` (requirement §8.1).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_CACHE: dict[str, tuple[dict[str, tuple[float, float]] | None, bool]] = {}


def load_density_stats(
    path: Path,
) -> tuple[dict[str, tuple[float, float]] | None, bool]:
    """Return ``({garment: (mean, std)}, loaded)``. ``None`` if unavailable."""
    key = str(path)
    if key in _CACHE:
        return _CACHE[key]

    stats: dict[str, tuple[float, float]] | None = None
    loaded = False
    try:
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            per_garment = data.get("per_garment", {})
            built: dict[str, tuple[float, float]] = {}
            for garment, metrics in per_garment.items():
                lcd = metrics.get("line_count_density")
                if lcd and "mean" in lcd and "std" in lcd:
                    built[str(garment)] = (float(lcd["mean"]), float(lcd["std"]))
            if built:
                stats = built
                loaded = True
                logger.info("Loaded reference stats: %s", path)
        else:
            logger.info("reference_stats.json not found (%s).", path)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Failed to load reference stats (%s).", exc)

    _CACHE[key] = (stats, loaded)
    return stats, loaded


def reset_cache() -> None:
    _CACHE.clear()
