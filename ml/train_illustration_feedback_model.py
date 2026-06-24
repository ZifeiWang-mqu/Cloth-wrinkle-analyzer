#!/usr/bin/env python3
"""Train the illustration-specific feedback classifier (scikit-learn).

Reads features.csv produced by ml/build_illustration_feedback_dataset.py and
trains a lightweight binary classifier (valid_issue vs false_issue). Saves a
joblib bundle + a metrics JSON that the backend reads for /api/model/status.

CLI:
  python ml/train_illustration_feedback_model.py \
    --features data/illustration_feedback_dataset/features.csv \
    --output-model data/models/illustration_feedback_model.joblib \
    --output-metrics data/models/illustration_feedback_model_metrics.json \
    --min-samples 30 --model-type logreg
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("train_illustration_feedback_model")

REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = REPO_ROOT / "backend"

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
try:
    from app.services.illustration_model import column_order
except Exception:  # pragma: no cover
    column_order = None


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--features",
        type=Path,
        default=REPO_ROOT / "data/illustration_feedback_dataset/features.csv",
    )
    ap.add_argument(
        "--output-model", type=Path, default=REPO_ROOT / "data/models/illustration_feedback_model.joblib"
    )
    ap.add_argument(
        "--output-metrics",
        type=Path,
        default=REPO_ROOT / "data/models/illustration_feedback_model_metrics.json",
    )
    ap.add_argument("--min-samples", type=int, default=30)
    ap.add_argument("--model-type", choices=["logreg", "rf"], default="logreg")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    if not args.features.exists():
        logger.error(
            "features.csv がありません: %s\n"
            "先に ml/build_illustration_feedback_dataset.py を実行してください。",
            args.features,
        )
        return 1

    try:
        import joblib
        import pandas as pd
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import precision_recall_fscore_support
        from sklearn.model_selection import train_test_split
    except Exception as exc:
        logger.error("scikit-learn / joblib が必要です。pip install scikit-learn joblib （%s）", exc)
        return 1

    df = pd.read_csv(args.features)
    if "label" not in df.columns:
        logger.error("features.csv に label 列がありません。")
        return 1

    cols = column_order() if column_order else [
        c for c in df.columns if c not in ("sample_id", "garment_type", "label")
    ]
    cols = [c for c in cols if c in df.columns]
    X = df[cols].fillna(0.0).to_numpy()
    y = df["label"].astype(int).to_numpy()

    n = len(y)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n < args.min_samples:
        logger.error("サンプル数が不足しています (%d < %d)。学習を中止します。", n, args.min_samples)
        return 1
    if n_pos < 3 or n_neg < 3:
        logger.error(
            "クラスのいずれかが不足しています (positive=%d, negative=%d)。両方に十分なデータが必要です。",
            n_pos,
            n_neg,
        )
        return 1

    # Holdout metrics, then refit on all data.
    metrics = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    try:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )
        clf_eval = _make_model(args.model_type, LogisticRegression, RandomForestClassifier)
        clf_eval.fit(X_tr, y_tr)
        pred = clf_eval.predict(X_te)
        p, r, f1, _ = precision_recall_fscore_support(
            y_te, pred, average="binary", zero_division=0
        )
        metrics = {"precision": float(p), "recall": float(r), "f1": float(f1)}
    except Exception as exc:
        logger.warning("ホールドアウト評価に失敗（メトリクスは0）: %s", exc)

    model = _make_model(args.model_type, LogisticRegression, RandomForestClassifier)
    model.fit(X, y)

    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    metrics_payload = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "model_type": args.model_type,
        "training_samples": n,
        "positive_samples": n_pos,
        "negative_samples": n_neg,
        "metrics": metrics,
        "columns": cols,
    }
    joblib.dump({"model": model, "columns": cols, "metrics": metrics_payload}, args.output_model)
    args.output_metrics.write_text(
        json.dumps(metrics_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    logger.info(
        "完了: n=%d (pos=%d, neg=%d) P=%.3f R=%.3f F1=%.3f",
        n,
        n_pos,
        n_neg,
        metrics["precision"],
        metrics["recall"],
        metrics["f1"],
    )
    logger.info("  -> %s", args.output_model)
    logger.info("  -> %s", args.output_metrics)
    return 0


def _make_model(kind, LogisticRegression, RandomForestClassifier):
    if kind == "rf":
        return RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
    return LogisticRegression(max_iter=1000, class_weight="balanced")


if __name__ == "__main__":
    raise SystemExit(main())
