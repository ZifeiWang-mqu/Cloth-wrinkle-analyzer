#!/usr/bin/env python3
"""Build an illustration-specific training dataset from stored feedback.

Reads the SQLite DB (inspections + issue_feedback), crops the bbox patch for
each feedback row from the original inspection image, assigns a binary label
(valid_issue / false_issue), extracts the SAME structural features the backend
uses, and writes metadata + features + patch images.

Outputs:
  data/illustration_feedback_dataset/metadata.csv
  data/illustration_feedback_dataset/features.csv
  data/illustration_feedback_dataset/patches/positive/*.png
  data/illustration_feedback_dataset/patches/negative/*.png

CLI:
  python ml/build_illustration_feedback_dataset.py \
    --db data/wrinkle.db \
    --images-root data/illustrations_raw \
    --output-root data/illustration_feedback_dataset
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("build_illustration_feedback_dataset")

REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = REPO_ROOT / "backend"

POSITIVE = {"correct", "missed_issue", "wrong_location", "wrong_reason", "corrected_location"}
NEGATIVE = {"false_positive"}


def _load_backend():
    if str(BACKEND_DIR) not in sys.path:
        sys.path.insert(0, str(BACKEND_DIR))
    from app.services.feature_extraction import extract_features
    from app.services.illustration_model import column_order, feature_row
    from app.services.pose import PoseResult
    from app.services.segmentation import GarmentRegion
    from app.services.wrinkle_edges import extract_wrinkle_candidates

    return {
        "extract_features": extract_features,
        "feature_row": feature_row,
        "column_order": column_order,
        "PoseResult": PoseResult,
        "GarmentRegion": GarmentRegion,
        "extract_wrinkle_candidates": extract_wrinkle_candidates,
    }


def _bbox_from_json(s: str | None) -> dict | None:
    if not s:
        return None
    try:
        d = json.loads(s)
        return {"x": float(d["x"]), "y": float(d["y"]), "w": float(d["w"]), "h": float(d["h"])}
    except Exception:
        return None


def _bbox_from_issue(issues_json: str | None, issue_id: str | None) -> dict | None:
    if not issues_json or not issue_id:
        return None
    try:
        for issue in json.loads(issues_json):
            if issue.get("id") == issue_id:
                b = issue.get("bbox")
                if b:
                    return {"x": float(b["x"]), "y": float(b["y"]), "w": float(b["w"]), "h": float(b["h"])}
    except Exception:
        return None
    return None


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", type=Path, default=REPO_ROOT / "data/wrinkle.db")
    ap.add_argument("--images-root", type=Path, default=REPO_ROOT / "data/illustrations_raw")
    ap.add_argument(
        "--output-root", type=Path, default=REPO_ROOT / "data/illustration_feedback_dataset"
    )
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    if not args.db.exists():
        logger.error("DB が見つかりません: %s （先に検査を実行してください）", args.db)
        return 1

    try:
        be = _load_backend()
    except Exception as exc:
        logger.error("backend サービスを読み込めません: %s", exc)
        return 1

    conn = sqlite3.connect(str(args.db))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT f.*, i.image_path AS insp_image_path, i.issues_json AS insp_issues_json,
                   i.garment_type AS insp_garment
            FROM issue_feedback f
            JOIN inspections i ON i.id = f.inspection_id
            ORDER BY f.created_at
            """
        ).fetchall()
    except sqlite3.OperationalError as exc:
        logger.error("DB スキーマが想定と異なります: %s", exc)
        return 1
    finally:
        conn.close()

    if not rows:
        logger.error("feedback がありません。UI でフィードバックを保存してから再実行してください。")
        return 1

    out = args.output_root
    (out / "patches" / "positive").mkdir(parents=True, exist_ok=True)
    (out / "patches" / "negative").mkdir(parents=True, exist_ok=True)
    columns = be["column_order"]()

    meta_rows: list[dict] = []
    feat_rows: list[dict] = []
    skipped = 0

    for r in rows:
        feedback = (r["feedback"] or "").strip()
        if feedback in POSITIVE:
            label = "valid_issue"
        elif feedback in NEGATIVE:
            label = "false_issue"
        else:
            skipped += 1
            continue

        bbox = (
            _bbox_from_json(r["corrected_bbox_json"])
            or _bbox_from_json(r["original_bbox_json"])
            or _bbox_from_issue(r["insp_issues_json"], r["issue_id"])
        )
        image_path = r["insp_image_path"]
        if not image_path or not Path(image_path).exists():
            logger.warning("画像が見つからずスキップ: %s", image_path)
            skipped += 1
            continue
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None:
            skipped += 1
            continue
        ih, iw = img.shape[:2]
        if bbox is None:
            bbox = {"x": 0, "y": 0, "w": iw, "h": ih}  # whole image fallback

        x = int(max(0, min(bbox["x"], iw - 1)))
        y = int(max(0, min(bbox["y"], ih - 1)))
        w = int(max(1, min(bbox["w"], iw - x)))
        h = int(max(1, min(bbox["h"], ih - y)))

        garment = (r["garment_type"] or r["insp_garment"] or "unknown")
        region = be["GarmentRegion"](x, y, w, h, used_selection=True, source="patch")
        candidates = be["extract_wrinkle_candidates"](img, region)
        feats = be["extract_features"](candidates, be["PoseResult"]())
        row_feats = be["feature_row"](feats.to_dict(), garment)

        sample_id = r["id"]
        sub = "positive" if label == "valid_issue" else "negative"
        patch = region.crop(img)
        patch_path = out / "patches" / sub / f"{sample_id}.png"
        try:
            cv2.imwrite(str(patch_path), patch)
        except Exception:
            patch_path = None

        meta_rows.append(
            {
                "sample_id": sample_id,
                "image_id": r["image_id"] or Path(image_path).name,
                "source_image_path": image_path,
                "feedback_id": r["id"],
                "issue_id": r["issue_id"],
                "garment_type": garment,
                "issue_type": r["issue_type"] or r["corrected_type"] or "",
                "label": label,
                "bbox_x": x,
                "bbox_y": y,
                "bbox_w": w,
                "bbox_h": h,
                "corrected": 1 if r["corrected_bbox_json"] else 0,
                "source_feedback": feedback,
                "patch_path": str(patch_path) if patch_path else "",
                "created_at": r["created_at"],
            }
        )
        feat_rows.append({**row_feats, "label": 1 if label == "valid_issue" else 0,
                          "sample_id": sample_id, "garment_type": garment})

    if not feat_rows:
        logger.error("有効な学習サンプルを作成できませんでした（画像/ bbox を確認）。skip=%d", skipped)
        return 1

    import pandas as pd

    pd.DataFrame(meta_rows).to_csv(out / "metadata.csv", index=False)
    feat_cols = ["sample_id", "garment_type"] + columns + ["label"]
    pd.DataFrame(feat_rows).reindex(columns=feat_cols).to_csv(out / "features.csv", index=False)

    n_pos = sum(1 for f in feat_rows if f["label"] == 1)
    logger.info(
        "完了: サンプル %d (positive=%d, negative=%d), skip=%d",
        len(feat_rows),
        n_pos,
        len(feat_rows) - n_pos,
        skipped,
    )
    logger.info("  -> %s", out / "metadata.csv")
    logger.info("  -> %s", out / "features.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
