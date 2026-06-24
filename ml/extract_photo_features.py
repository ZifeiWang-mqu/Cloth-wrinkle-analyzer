#!/usr/bin/env python3
"""Extract natural-wrinkle reference features from real clothing photos.

Reads photos under ``data/photos_raw/<garment>/`` (garment type is inferred from
the folder name), splits each image's garment region into patches, and computes
*structural* features per patch (edge density, dominant line angle, line count,
gradient orientation, contrast, …). Output is written to a parquet (or CSV)
file, plus an aggregated ``reference_stats.json`` that the backend's
``score_line_density`` rule can consume.

Key choices (see requirements section 9 + 15):
  * Structural features only — we never learn photo colour/texture, because of
    the large photo↔illustration domain gap.
  * The patch feature computation **reuses the backend pipeline**
    (``wrinkle_edges`` + ``feature_extraction``) so photo stats are directly
    comparable to what the inspection API computes on illustrations.
  * Train/test split is done at the **image** level so patches from one photo
    never leak across the split.
  * The extractor is pluggable (``--extractor``); a DINOv2 hook is stubbed for a
    later phase.

Usage:
    python ml/extract_photo_features.py                 # uses configs/default.yaml
    python ml/extract_photo_features.py --photos-root data/photos_raw --patch-size 160
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("extract_photo_features")

REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = REPO_ROOT / "backend"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
MASK_EXTS = (".png", ".jpg", ".jpeg", ".webp")

DEFAULT_CONFIG: dict[str, Any] = {
    "photos_root": "data/photos_raw",
    "masks_root": "data/photos_masks",
    "output_path": "data/features/photo_reference_features.parquet",
    "reference_stats_path": "data/features/reference_stats.json",
    "patch_size": 128,
    "stride": 128,
    "min_patch_coverage": 0.5,
    "min_lines_per_patch": 0,
    "extractor": "classic",
    "test_ratio": 0.2,
    "seed": 42,
}


# --------------------------------------------------------------------------- #
# Feature extractors (pluggable)
# --------------------------------------------------------------------------- #
class BaseFeatureExtractor(ABC):
    name = "base"

    @abstractmethod
    def extract(self, image_bgr: np.ndarray, region: tuple[int, int, int, int]) -> dict:
        """Return a flat dict of features for the patch ``region`` (x, y, w, h)."""


class ClassicCVExtractor(BaseFeatureExtractor):
    """Reuses the backend's classic-CV pipeline for parity with inspection."""

    name = "classic"

    def __init__(self) -> None:
        if str(BACKEND_DIR) not in sys.path:
            sys.path.insert(0, str(BACKEND_DIR))
        try:
            from app.services.feature_extraction import extract_features
            from app.services.pose import PoseResult
            from app.services.segmentation import GarmentRegion
            from app.services.wrinkle_edges import extract_wrinkle_candidates
        except Exception as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "backend のサービスを読み込めませんでした。"
                "backend/requirements.txt を同じ環境にインストールしてください。"
                f" 詳細: {exc}"
            ) from exc

        self._GarmentRegion = GarmentRegion
        self._extract_candidates = extract_wrinkle_candidates
        self._extract_features = extract_features
        self._PoseResult = PoseResult

    def extract(self, image_bgr: np.ndarray, region: tuple[int, int, int, int]) -> dict:
        x, y, w, h = region
        gr = self._GarmentRegion(x, y, w, h, used_selection=True, source="patch")
        candidates = self._extract_candidates(image_bgr, gr)
        feats = self._extract_features(candidates, self._PoseResult())
        return {
            # Columns requested in requirements section 9:
            "edge_density": feats.edge_density,
            "dominant_line_angle": feats.dominant_line_angle,
            "line_count": feats.num_lines,
            "line_length_sum": feats.line_length_sum,
            "gradient_orientation": feats.gradient_orientation,
            "local_contrast": feats.local_contrast,
            # Extra structural features (useful for stats / future models):
            "line_count_density": feats.line_count_density,
            "orientation_dispersion": feats.orientation_dispersion,
            "convergence_strength": feats.convergence_strength,
            "gradient_coherence": feats.gradient_coherence,
        }


class DinoV2Extractor(BaseFeatureExtractor):  # pragma: no cover - future phase
    name = "dino"

    def extract(self, image_bgr: np.ndarray, region: tuple[int, int, int, int]) -> dict:
        raise NotImplementedError(
            "DINOv2 抽出は将来フェーズ用のプレースホルダーです（--extractor classic を使用）。"
        )


def get_extractor(name: str) -> BaseFeatureExtractor:
    if name == "classic":
        return ClassicCVExtractor()
    if name == "dino":
        return DinoV2Extractor()
    raise ValueError(f"未知の extractor: {name}")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def load_config(config_path: Path | None) -> dict:
    cfg = dict(DEFAULT_CONFIG)
    if config_path and config_path.exists():
        try:
            import yaml  # type: ignore

            with config_path.open("r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f) or {}
            cfg.update({k: v for k, v in loaded.items() if v is not None})
            logger.info("設定を読み込みました: %s", config_path)
        except Exception as exc:
            logger.warning("設定の読み込みに失敗（既定値を使用）: %s", exc)
    return cfg


def resolve(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (REPO_ROOT / p)


def discover_images(photos_root: Path) -> list[Path]:
    if not photos_root.exists():
        return []
    return sorted(
        p for p in photos_root.rglob("*") if p.suffix.lower() in IMAGE_EXTS
    )


def garment_from_path(image_path: Path, photos_root: Path) -> str:
    try:
        rel = image_path.relative_to(photos_root)
        return rel.parts[0] if len(rel.parts) > 1 else "unknown"
    except ValueError:
        return "unknown"


def find_mask(image_path: Path, photos_root: Path, masks_root: Path) -> Path | None:
    if not masks_root.exists():
        return None
    rel = image_path.relative_to(photos_root)
    stem_rel = rel.with_suffix("")
    for ext in MASK_EXTS:
        cand = masks_root / stem_rel.with_suffix(ext)
        if cand.exists():
            return cand
    return None


def region_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return x0, y0, (x1 - x0 + 1), (y1 - y0 + 1)


def iter_patches(
    region: tuple[int, int, int, int], patch: int, stride: int
) -> list[tuple[int, int, int, int]]:
    rx, ry, rw, rh = region
    if rw <= patch or rh <= patch:
        return [(rx, ry, rw, rh)]  # region smaller than a patch -> single patch
    patches = []
    for y in range(ry, ry + rh - patch + 1, stride):
        for x in range(rx, rx + rw - patch + 1, stride):
            patches.append((x, y, patch, patch))
    return patches or [(rx, ry, rw, rh)]


def split_for_image(image_id: str, test_ratio: float, seed: int) -> str:
    digest = hashlib.md5(f"{seed}:{image_id}".encode()).hexdigest()
    bucket = int(digest[:8], 16) % 100
    return "test" if bucket < int(test_ratio * 100) else "train"


def save_table(rows: list[dict], output_path: Path) -> Path:
    import pandas as pd  # imported lazily so --help works without pandas

    df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(output_path, index=False)
        return output_path
    except Exception as exc:  # pyarrow/fastparquet missing
        csv_path = output_path.with_suffix(".csv")
        logger.warning("parquet 保存に失敗、CSV にフォールバック (%s): %s", csv_path.name, exc)
        df.to_csv(csv_path, index=False)
        return csv_path


def compute_reference_stats(rows: list[dict]) -> dict:
    import pandas as pd

    df = pd.DataFrame(rows)
    stats: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "num_images": int(df["image_id"].nunique()) if not df.empty else 0,
        "num_patches": int(len(df)),
        "metrics": ["line_count_density", "edge_density"],
        "per_garment": {},
    }
    if df.empty:
        return stats
    train = df[df["split"] == "train"]
    source = train if not train.empty else df
    for garment, grp in source.groupby("garment_type"):
        per_metric = {}
        for metric in ("line_count_density", "edge_density"):
            vals = grp[metric].astype(float)
            per_metric[metric] = {
                "mean": float(vals.mean()),
                "std": float(vals.std(ddof=0)),
                "n": int(vals.count()),
            }
        stats["per_garment"][str(garment)] = per_metric
    return stats


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=REPO_ROOT / "ml/configs/default.yaml")
    ap.add_argument("--photos-root")
    ap.add_argument("--masks-root")
    ap.add_argument("--output-path")
    ap.add_argument("--reference-stats-path")
    ap.add_argument("--patch-size", type=int)
    ap.add_argument("--stride", type=int)
    ap.add_argument("--min-patch-coverage", type=float)
    ap.add_argument("--min-lines-per-patch", type=int)
    ap.add_argument("--extractor")
    ap.add_argument("--test-ratio", type=float)
    ap.add_argument("--seed", type=int)
    return ap.parse_args()


def merge_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    mapping = {
        "photos_root": args.photos_root,
        "masks_root": args.masks_root,
        "output_path": args.output_path,
        "reference_stats_path": args.reference_stats_path,
        "patch_size": args.patch_size,
        "stride": args.stride,
        "min_patch_coverage": args.min_patch_coverage,
        "min_lines_per_patch": args.min_lines_per_patch,
        "extractor": args.extractor,
        "test_ratio": args.test_ratio,
        "seed": args.seed,
    }
    for k, v in mapping.items():
        if v is not None:
            cfg[k] = v
    return cfg


def main() -> int:
    args = parse_args()
    cfg = merge_overrides(load_config(args.config), args)

    photos_root = resolve(cfg["photos_root"])
    masks_root = resolve(cfg["masks_root"])
    output_path = resolve(cfg["output_path"])
    stats_path = resolve(cfg["reference_stats_path"])
    patch = int(cfg["patch_size"])
    stride = int(cfg["stride"])
    min_cov = float(cfg["min_patch_coverage"])
    min_lines = int(cfg["min_lines_per_patch"])

    images = discover_images(photos_root)
    if not images:
        logger.error(
            "画像が見つかりません: %s\n"
            "data/photos_raw/<garment>/ に写真を配置してください "
            "(garment = shirt/skirt/pants/dress/jacket など)。",
            photos_root,
        )
        return 1

    logger.info("画像 %d 枚を処理します（extractor=%s）", len(images), cfg["extractor"])
    extractor = get_extractor(str(cfg["extractor"]))

    rows: list[dict] = []
    processed = 0
    for image_path in images:
        image_id = str(image_path.relative_to(photos_root)).replace("\\", "/")
        garment = garment_from_path(image_path, photos_root)
        split = split_for_image(image_id, float(cfg["test_ratio"]), int(cfg["seed"]))

        img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if img is None:
            logger.warning("読み込み失敗、スキップ: %s", image_path)
            continue
        ih, iw = img.shape[:2]

        mask = None
        mask_path = find_mask(image_path, photos_root, masks_root)
        if mask_path is not None:
            m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if m is not None:
                if m.shape[:2] != (ih, iw):
                    m = cv2.resize(m, (iw, ih), interpolation=cv2.INTER_NEAREST)
                mask = (m > 127).astype(np.uint8)

        region = region_from_mask(mask) if mask is not None else (0, 0, iw, ih)
        if region is None:
            region = (0, 0, iw, ih)

        for px, py, pw, ph in iter_patches(region, patch, stride):
            # Coverage filter (only when a mask is available).
            if mask is not None:
                sub = mask[py : py + ph, px : px + pw]
                if sub.size == 0 or float(sub.mean()) < min_cov:
                    continue
            try:
                feats = extractor.extract(img, (px, py, pw, ph))
            except Exception as exc:  # pragma: no cover - defensive per-patch
                logger.warning("特徴抽出に失敗 (%s): %s", image_id, exc)
                continue
            if feats.get("line_count", 0) < min_lines:
                continue
            rows.append(
                {
                    "image_id": image_id,
                    "garment_type": garment,
                    "split": split,
                    "patch_x": px,
                    "patch_y": py,
                    "patch_w": pw,
                    "patch_h": ph,
                    **feats,
                }
            )
        processed += 1

    if not rows:
        logger.error("特徴量を1件も抽出できませんでした。画像/マスクを確認してください。")
        return 1

    saved = save_table(rows, output_path)
    stats = compute_reference_stats(rows)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    n_train = sum(1 for r in rows if r["split"] == "train")
    logger.info("完了: 画像 %d 枚 / パッチ %d 件", processed, len(rows))
    logger.info("  train=%d, test=%d", n_train, len(rows) - n_train)
    logger.info("  features -> %s", saved)
    logger.info("  reference_stats -> %s", stats_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
