#!/usr/bin/env python3
"""Train a simple one-class anomaly model from photo reference features.

This is the "proof of concept" learned model: for each garment type (and a
global fallback) we fit a multivariate Gaussian over scale-invariant structural
features of *natural* wrinkles, then calibrate the Mahalanobis distance so that
typical training patches score ~0 and outliers score ~1.

Pipeline:
    ml/extract_photo_features.py   ->  photo_reference_features.parquet
    ml/train_anomaly_model.py      ->  data/models/anomaly_model.json
The backend auto-loads that JSON (numpy only — no sklearn/torch needed) via
``app.services.anomaly_model.get_anomaly_model``.

Usage:
    python ml/train_anomaly_model.py
    python ml/train_anomaly_model.py --min-samples 40 --reg 1e-2
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("train_anomaly_model")

REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = REPO_ROOT / "backend"

# Reuse the exact feature set the backend scores with (keeps train/infer aligned).
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
try:
    from app.services.anomaly_model import FEATURE_KEYS, GLOBAL_KEY
except Exception:  # pragma: no cover - allows running without backend on path
    FEATURE_KEYS = (
        "edge_density",
        "line_count_density",
        "orientation_dispersion",
        "convergence_strength",
        "gradient_coherence",
        "local_contrast",
    )
    GLOBAL_KEY = "_global"


def load_features(path: Path):
    import pandas as pd

    if path.exists():
        if path.suffix == ".parquet":
            try:
                return pd.read_parquet(path)
            except Exception as exc:
                logger.warning("parquet 読み込み失敗 (%s)、CSV を試します", exc)
        if path.suffix == ".csv":
            return pd.read_csv(path)
    # Try sibling CSV if parquet missing (extract script's fallback output).
    csv = path.with_suffix(".csv")
    if csv.exists():
        return pd.read_csv(csv)
    raise FileNotFoundError(
        f"特徴量ファイルが見つかりません: {path} (または {csv}). "
        "先に ml/extract_photo_features.py を実行してください。"
    )


def fit_gaussian(x: np.ndarray, reg: float) -> dict | None:
    """Fit a standardized multivariate Gaussian and calibrate distances."""
    n, k = x.shape
    if n <= k:
        return None
    mean = x.mean(axis=0)
    std = x.std(axis=0, ddof=0)
    std = np.where(std < 1e-9, 1.0, std)  # avoid div-by-zero on constant features

    z = (x - mean) / std
    cov = np.cov(z, rowvar=False)
    cov = np.atleast_2d(cov)
    cov += reg * np.eye(k)  # regularize -> always invertible
    cov_inv = np.linalg.inv(cov)

    d = np.sqrt(np.maximum(0.0, np.einsum("ij,jk,ik->i", z, cov_inv, z)))
    p50 = float(np.percentile(d, 50))
    p95 = float(np.percentile(d, 95))
    if p95 <= p50:
        p95 = p50 + 1e-3
    return {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "cov_inv": cov_inv.tolist(),
        "p50": p50,
        "p95": p95,
        "n": int(n),
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--features",
        type=Path,
        default=REPO_ROOT / "data/features/photo_reference_features.parquet",
    )
    ap.add_argument(
        "--output", type=Path, default=REPO_ROOT / "data/models/anomaly_model.json"
    )
    ap.add_argument(
        "--min-samples",
        type=int,
        default=30,
        help="この件数未満の garment は個別モデルを作らず global で代替",
    )
    ap.add_argument("--reg", type=float, default=1e-3, help="共分散の正則化係数")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    df = load_features(args.features)

    missing = [k for k in FEATURE_KEYS if k not in df.columns]
    if missing:
        logger.error("特徴量に必要な列がありません: %s", missing)
        return 1

    if "split" in df.columns:
        train = df[df["split"] == "train"]
        if train.empty:
            logger.warning("train 分割が空のため全データで学習します。")
            train = df
    else:
        train = df

    fits: dict[str, dict] = {}

    # Global fallback (all garments together).
    x_all = train[list(FEATURE_KEYS)].to_numpy(dtype=np.float64)
    global_fit = fit_gaussian(x_all, args.reg)
    if global_fit is None:
        logger.error(
            "学習データが少なすぎます（%d 件、特徴 %d 次元）。"
            "最低でも %d 件以上のパッチが必要です。",
            len(x_all),
            len(FEATURE_KEYS),
            len(FEATURE_KEYS) + 1,
        )
        return 1
    fits[GLOBAL_KEY] = global_fit

    # Per-garment models.
    if "garment_type" in train.columns:
        for garment, grp in train.groupby("garment_type"):
            x = grp[list(FEATURE_KEYS)].to_numpy(dtype=np.float64)
            if len(x) < args.min_samples:
                logger.info(
                    "  %-8s : %d 件 (< %d) -> global で代替",
                    garment,
                    len(x),
                    args.min_samples,
                )
                continue
            fit = fit_gaussian(x, args.reg)
            if fit is not None:
                fits[str(garment)] = fit
                logger.info("  %-8s : %d 件で学習", garment, len(x))

    model = {
        "model_type": "mahalanobis_gaussian_v1",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "feature_keys": list(FEATURE_KEYS),
        "reg": args.reg,
        "fits": fits,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(model, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    logger.info("完了: %d 個のモデル（global 含む）を保存 -> %s", len(fits), args.output)
    logger.info("バックエンドは次回起動時にこのモデルを自動で読み込みます。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
