#!/usr/bin/env python3
"""Evaluate the current inspection model against annotated illustrations.

Compares the backend's *flagged* predictions to human annotations using IoU
matching, and reports Precision / Recall / F1 overall and broken down by
issue_type and garment_type, plus a confidence-band false-positive rate and
explicit FP / FN / wrong_location / wrong_reason lists.

Reuses the backend inspection pipeline directly (no HTTP server needed).

CLI:
    python ml/evaluate_illustration_results.py \
        --images-root data/illustrations_raw \
        --annotations data/annotations/illustration_feedback.csv \
        --output-md reports/evaluation_report.md \
        --output-json reports/evaluation_metrics.json \
        --iou-threshold 0.3

Annotation CSV columns:
    image_id,garment_type,issue_type,bbox_x,bbox_y,bbox_w,bbox_h,label,comment
Labels: true_positive | false_positive | false_negative | corrected_location | wrong_reason
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cv2

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("evaluate_illustration_results")

REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = REPO_ROOT / "backend"
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")

# Annotation labels that represent a REAL issue (ground-truth positives).
GT_LABELS = {"true_positive", "false_negative", "corrected_location", "wrong_reason"}
CONFIDENCE_BANDS = [(0.0, 0.5), (0.5, 0.7), (0.7, 0.85), (0.85, 1.01)]


# --------------------------------------------------------------------------- #
# Backend pipeline (imported lazily)
# --------------------------------------------------------------------------- #
def _load_backend():
    if str(BACKEND_DIR) not in sys.path:
        sys.path.insert(0, str(BACKEND_DIR))
    from app.services.anomaly_model import get_anomaly_model
    from app.services.feature_extraction import extract_features
    from app.services.pose import estimate_pose
    from app.services.reference_stats import load_density_stats
    from app.services.region_geometry import RegionGeometry
    from app.services.rule_engine import integrate_scores
    from app.services.segmentation import get_garment_region
    from app.services.thresholds import load_thresholds
    from app.services.wrinkle_edges import extract_wrinkle_candidates
    from app.settings import settings

    return {
        "get_anomaly_model": get_anomaly_model,
        "extract_features": extract_features,
        "estimate_pose": estimate_pose,
        "load_density_stats": load_density_stats,
        "integrate_scores": integrate_scores,
        "get_garment_region": get_garment_region,
        "load_thresholds": load_thresholds,
        "extract_wrinkle_candidates": extract_wrinkle_candidates,
        "RegionGeometry": RegionGeometry,
        "settings": settings,
    }


def inspect_image(be: dict, image, garment: str) -> list[dict]:
    """Run the pipeline on a BGR image; return flagged issues (type/bbox/conf)."""
    import math

    region = be["get_garment_region"](image, None)
    pose = be["estimate_pose"](image)
    candidates = be["extract_wrinkle_candidates"](image, region)
    features = be["extract_features"](candidates, pose)
    anomaly = be["get_anomaly_model"]().score(features, garment)
    ih, iw = image.shape[:2]
    region_geometry = be["RegionGeometry"](
        bbox=region.as_dict(),
        image_w=iw,
        image_h=ih,
        region_diag=math.hypot(region.w, region.h),
    )
    settings = be["settings"]
    thresholds = be["load_thresholds"](settings.thresholds_path)
    density_stats, _ = be["load_density_stats"](settings.reference_stats_path)
    payload = be["integrate_scores"](
        features,
        garment,
        pose,
        region_geometry,
        settings,
        reference_stats=density_stats,
        anomaly_score=anomaly,
        thresholds=thresholds,
    )
    return [i for i in payload["issues"] if i.get("flagged")]


# --------------------------------------------------------------------------- #
# Geometry / metrics
# --------------------------------------------------------------------------- #
def iou(a: dict, b: dict) -> float:
    ax2, ay2 = a["x"] + a["w"], a["y"] + a["h"]
    bx2, by2 = b["x"] + b["w"], b["y"] + b["h"]
    ix1, iy1 = max(a["x"], b["x"]), max(a["y"], b["y"])
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = a["w"] * a["h"] + b["w"] * b["h"] - inter
    return inter / union if union > 0 else 0.0


def prf(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def match(preds: list[dict], gts: list[dict], iou_thr: float, type_sensitive: bool):
    """Greedy IoU matching. Returns (tp_pairs, fp_preds, fn_gts)."""
    used_gt: set[int] = set()
    tp_pairs, fp_preds = [], []
    for p in preds:
        best_j, best_iou = -1, 0.0
        for j, g in enumerate(gts):
            if j in used_gt:
                continue
            if type_sensitive and p["type"] != g["issue_type"]:
                continue
            v = iou(p["bbox"], g["bbox"])
            if v >= iou_thr and v > best_iou:
                best_iou, best_j = v, j
        if best_j >= 0:
            used_gt.add(best_j)
            tp_pairs.append((p, gts[best_j]))
        else:
            fp_preds.append(p)
    fn_gts = [g for j, g in enumerate(gts) if j not in used_gt]
    return tp_pairs, fp_preds, fn_gts


# --------------------------------------------------------------------------- #
# Annotations
# --------------------------------------------------------------------------- #
def load_annotations(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                rows.append(
                    {
                        "image_id": (r.get("image_id") or "").strip(),
                        "garment_type": (r.get("garment_type") or "unknown").strip(),
                        "issue_type": (r.get("issue_type") or "").strip(),
                        "bbox": {
                            "x": float(r.get("bbox_x") or 0),
                            "y": float(r.get("bbox_y") or 0),
                            "w": float(r.get("bbox_w") or 0) or 1.0,
                            "h": float(r.get("bbox_h") or 0) or 1.0,
                        },
                        "label": (r.get("label") or "").strip(),
                        "comment": (r.get("comment") or "").strip(),
                    }
                )
            except (ValueError, TypeError) as exc:
                logger.warning("行をスキップ（数値変換エラー）: %s (%s)", r, exc)
    return rows


def find_image(images_root: Path, image_id: str) -> Path | None:
    direct = images_root / image_id
    if direct.exists():
        return direct
    stem = Path(image_id).stem
    for ext in IMAGE_EXTS:
        cand = images_root / f"{stem}{ext}"
        if cand.exists():
            return cand
    matches = list(images_root.rglob(image_id))
    return matches[0] if matches else None


def write_template(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "image_id,garment_type,issue_type,bbox_x,bbox_y,bbox_w,bbox_h,label,comment\n"
        "eval_001.png,shirt,joint_inconsistency,120,220,80,60,true_positive,肘の皺方向が不自然\n",
        encoding="utf-8",
    )


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def _fmt_metrics_table(title: str, rows: dict[str, dict]) -> str:
    lines = [f"### {title}", "", "| key | precision | recall | f1 | tp | fp | fn |", "|---|---|---|---|---|---|---|"]
    for key, m in sorted(rows.items()):
        lines.append(
            f"| {key} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} "
            f"| {m['tp']} | {m['fp']} | {m['fn']} |"
        )
    lines.append("")
    return "\n".join(lines)


def build_report(metrics: dict, iou_thr: float) -> str:
    o = metrics["overall"]
    out = [
        "# Illustration Evaluation Report",
        "",
        f"- generated_at: {metrics['generated_at']}",
        f"- iou_threshold: {iou_thr}",
        f"- images_evaluated: {metrics['images_evaluated']}",
        f"- images_missing: {metrics['images_missing']}",
        "",
        "## Overall",
        "",
        f"- **Precision**: {o['precision']:.3f}",
        f"- **Recall**: {o['recall']:.3f}",
        f"- **F1**: {o['f1']:.3f}",
        f"- TP={o['tp']} FP={o['fp']} FN={o['fn']}",
        "",
        _fmt_metrics_table("By issue_type", metrics["by_issue_type"]),
        _fmt_metrics_table("By garment_type", metrics["by_garment_type"]),
        "## False-positive rate by confidence band",
        "",
        "| band | predictions | false_positives | fp_rate |",
        "|---|---|---|---|",
    ]
    for band in metrics["confidence_bands"]:
        out.append(
            f"| {band['range']} | {band['predictions']} | {band['false_positives']} "
            f"| {band['fp_rate']:.3f} |"
        )
    out.append("")

    def _list_section(title: str, items: list[dict]) -> str:
        lines = [f"## {title} ({len(items)})", ""]
        if not items:
            lines.append("_none_\n")
            return "\n".join(lines)
        for it in items[:100]:
            lines.append(
                f"- `{it.get('image_id','?')}` {it.get('issue_type') or it.get('type','')} "
                f"bbox={it.get('bbox')} {('— ' + it['comment']) if it.get('comment') else ''}"
            )
        lines.append("")
        return "\n".join(lines)

    out.append(_list_section("False positives", metrics["false_positives"]))
    out.append(_list_section("False negatives", metrics["false_negatives"]))
    out.append(_list_section("Wrong location (annotated)", metrics["wrong_location"]))
    out.append(_list_section("Wrong reason (annotated)", metrics["wrong_reason"]))
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--images-root", type=Path, default=REPO_ROOT / "data/illustrations_raw")
    ap.add_argument("--annotations", type=Path, default=REPO_ROOT / "data/annotations/illustration_feedback.csv")
    ap.add_argument("--output-md", type=Path, default=REPO_ROOT / "reports/evaluation_report.md")
    ap.add_argument("--output-json", type=Path, default=REPO_ROOT / "reports/evaluation_metrics.json")
    ap.add_argument("--iou-threshold", type=float, default=0.3)
    return ap.parse_args()


def main() -> int:
    args = parse_args()

    if not args.annotations.exists():
        template = args.annotations.with_suffix(".csv")
        logger.error("アノテーションCSVがありません: %s", args.annotations)
        logger.error("テンプレートを作成します。記入後に再実行してください。")
        try:
            write_template(template)
            logger.error("テンプレート作成: %s", template)
        except Exception as exc:  # pragma: no cover
            logger.error("テンプレート作成に失敗: %s", exc)
        return 1

    if not args.images_root.exists() or not any(args.images_root.rglob("*")):
        logger.error(
            "評価画像がありません: %s\n"
            "data/illustrations_raw/ に評価用イラストを置いてください。",
            args.images_root,
        )
        return 1

    annotations = load_annotations(args.annotations)
    if not annotations:
        logger.error("アノテーションが空です。CSV を記入してください: %s", args.annotations)
        return 1

    try:
        be = _load_backend()
    except Exception as exc:
        logger.error("backend サービスを読み込めません（依存をインストール済みか確認）: %s", exc)
        return 1

    by_image: dict[str, list[dict]] = defaultdict(list)
    for row in annotations:
        by_image[row["image_id"]].append(row)

    # Accumulators
    overall = {"tp": 0, "fp": 0, "fn": 0}
    by_type: dict[str, dict] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    by_garment: dict[str, dict] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    band_counts = [{"predictions": 0, "false_positives": 0} for _ in CONFIDENCE_BANDS]
    fp_list, fn_list = [], []
    images_evaluated = 0
    images_missing = 0

    for image_id, rows in by_image.items():
        img_path = find_image(args.images_root, image_id)
        if img_path is None:
            images_missing += 1
            logger.warning("画像が見つかりません: %s", image_id)
            continue
        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image is None:
            images_missing += 1
            logger.warning("画像を読み込めません: %s", img_path)
            continue

        garment = rows[0].get("garment_type", "unknown") or "unknown"
        gts = [r for r in rows if r["label"] in GT_LABELS]
        try:
            preds = inspect_image(be, image, garment)
        except Exception as exc:  # never let one image break the whole run
            logger.warning("検査に失敗、スキップ: %s (%s)", image_id, exc)
            continue
        images_evaluated += 1

        # Overall (type-sensitive) matching.
        tp_pairs, fps, fns = match(preds, gts, args.iou_threshold, type_sensitive=True)
        overall["tp"] += len(tp_pairs)
        overall["fp"] += len(fps)
        overall["fn"] += len(fns)
        for p in fps:
            fp_list.append({"image_id": image_id, "type": p["type"], "bbox": p["bbox"]})
        for g in fns:
            fn_list.append(
                {"image_id": image_id, "issue_type": g["issue_type"], "bbox": g["bbox"], "comment": g.get("comment", "")}
            )

        # Per garment.
        by_garment[garment]["tp"] += len(tp_pairs)
        by_garment[garment]["fp"] += len(fps)
        by_garment[garment]["fn"] += len(fns)

        # Per issue_type (compute independently per type for clean P/R).
        types = {p["type"] for p in preds} | {g["issue_type"] for g in gts}
        for t in types:
            p_t = [p for p in preds if p["type"] == t]
            g_t = [g for g in gts if g["issue_type"] == t]
            tpp, fpp, fnn = match(p_t, g_t, args.iou_threshold, type_sensitive=True)
            by_type[t]["tp"] += len(tpp)
            by_type[t]["fp"] += len(fpp)
            by_type[t]["fn"] += len(fnn)

        # Confidence bands (over all predictions; FP from the overall matching).
        fp_ids = {id(p) for p in fps}
        for p in preds:
            conf = float(p.get("confidence", 0.0))
            for bi, (lo, hi) in enumerate(CONFIDENCE_BANDS):
                if lo <= conf < hi:
                    band_counts[bi]["predictions"] += 1
                    if id(p) in fp_ids:
                        band_counts[bi]["false_positives"] += 1
                    break

    metrics = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "iou_threshold": args.iou_threshold,
        "images_evaluated": images_evaluated,
        "images_missing": images_missing,
        "overall": prf(overall["tp"], overall["fp"], overall["fn"]),
        "by_issue_type": {t: prf(v["tp"], v["fp"], v["fn"]) for t, v in by_type.items()},
        "by_garment_type": {g: prf(v["tp"], v["fp"], v["fn"]) for g, v in by_garment.items()},
        "confidence_bands": [
            {
                "range": f"{lo:.2f}-{hi if hi <= 1 else 1.0:.2f}",
                "predictions": band_counts[i]["predictions"],
                "false_positives": band_counts[i]["false_positives"],
                "fp_rate": (
                    band_counts[i]["false_positives"] / band_counts[i]["predictions"]
                    if band_counts[i]["predictions"]
                    else 0.0
                ),
            }
            for i, (lo, hi) in enumerate(CONFIDENCE_BANDS)
        ],
        "false_positives": fp_list,
        "false_negatives": fn_list,
        "wrong_location": [
            {"image_id": r["image_id"], "issue_type": r["issue_type"], "bbox": r["bbox"], "comment": r["comment"]}
            for r in annotations
            if r["label"] == "corrected_location"
        ],
        "wrong_reason": [
            {"image_id": r["image_id"], "issue_type": r["issue_type"], "bbox": r["bbox"], "comment": r["comment"]}
            for r in annotations
            if r["label"] == "wrong_reason"
        ],
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(build_report(metrics, args.iou_threshold), encoding="utf-8")

    o = metrics["overall"]
    logger.info(
        "評価完了: 画像 %d 枚 (欠落 %d) / P=%.3f R=%.3f F1=%.3f",
        images_evaluated,
        images_missing,
        o["precision"],
        o["recall"],
        o["f1"],
    )
    logger.info("  -> %s", args.output_md)
    logger.info("  -> %s", args.output_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
