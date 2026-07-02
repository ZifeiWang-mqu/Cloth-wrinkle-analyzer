"""Review-memory layer: per-inspection reviews -> retrievable case memory.

Phase 1: store the user's verdict plus a TRUSTWORTHY snapshot copied from the
persisted ``Inspection`` row (never from client-supplied debug data).
Phase 2: turn each review into a compact deterministic English summary.
Phase 3: retrieve similar past cases via local hashed-BoW embeddings + cosine.

No external API calls anywhere in this module; GPT explanation is a later
layer on top of the retrieval provided here.
"""

from __future__ import annotations

import json
import logging

from sqlalchemy.orm import Session

from app.db.models import Inspection, ReviewFeedback, ReviewMemory
from app.schemas import ReviewRequest
from app.services.embeddings import cosine_similarity, get_embedder

logger = logging.getLogger(__name__)

MODE_HAND = "hand"
MODE_WRINKLE = "wrinkle"


def infer_mode(inspection: Inspection) -> str:
    return MODE_HAND if inspection.garment_type == "hand" else MODE_WRINKLE


def _load_json(text: str | None, default):
    try:
        return json.loads(text) if text else default
    except (json.JSONDecodeError, TypeError):
        return default


def build_snapshot(inspection: Inspection) -> dict:
    """Parse the persisted inspection row into a snapshot dict (server data only)."""
    issues = _load_json(inspection.issues_json, [])
    debug = _load_json(inspection.debug_json, {})
    return {
        "mode": infer_mode(inspection),
        "garment_type": inspection.garment_type,
        "result": inspection.result,
        "overall_score": float(inspection.overall_score or 0.0),
        "issues": issues,
        "scores": debug.get("scores", {}) or {},
        "debug": debug,
        "selected_region": bool(inspection.selected_region_json),
    }


# --------------------------------------------------------------------------- #
# Phase 2 — deterministic case summaries
# --------------------------------------------------------------------------- #
def _flagged_types(issues: list[dict]) -> list[str]:
    return [i.get("type", "?") for i in issues if i.get("flagged")]


def _fmt_scores(scores: dict, top_n: int = 3) -> str:
    ranked = sorted(
        ((k, v) for k, v in scores.items() if isinstance(v, (int, float))),
        key=lambda kv: kv[1],
        reverse=True,
    )[:top_n]
    return ", ".join(f"{k}={v:.2f}" for k, v in ranked)


def _hand_body(snapshot: dict) -> list[str]:
    parts: list[str] = []
    hand = (snapshot.get("debug") or {}).get("hand") or {}
    region_used = bool(snapshot.get("selected_region"))
    if hand.get("detected"):
        confs = hand.get("confidences") or []
        conf_txt = (
            f" with {', '.join(f'{c:.3f}' for c in confs)} confidence" if confs else ""
        )
        parts.append(
            f"Detector ({hand.get('backend', 'unknown')}) found "
            f"{hand.get('num_hands', 0)} hand(s){conf_txt}."
        )
    elif region_used:
        parts.append(
            "The detector found no hand despite a user-selected region "
            f"(note: {hand.get('note') or 'unknown'})."
        )
    else:
        parts.append(
            f"The detector found no hand (note: {hand.get('note') or 'unknown'})."
        )

    scores = snapshot.get("scores") or {}
    flagged = _flagged_types(snapshot.get("issues") or [])
    if flagged:
        parts.append(f"Flagged rule issues: {', '.join(sorted(set(flagged)))}.")
    elif scores and all(v == 0 for v in scores.values()):
        parts.append("All landmark rules returned 0.0.")
    elif scores:
        parts.append(f"Top rule scores: {_fmt_scores(scores)}.")

    if region_used and hand.get("detected"):
        parts.append("A user-selected region hint was used.")
    parts.append(
        f"App result: {snapshot.get('result')} "
        f"(overall {snapshot.get('overall_score', 0.0):.2f})."
    )
    return parts


def _wrinkle_body(snapshot: dict) -> list[str]:
    parts: list[str] = [f"Garment type: {snapshot.get('garment_type', 'unknown')}."]
    flagged = _flagged_types(snapshot.get("issues") or [])
    if flagged:
        parts.append(f"Flagged issues: {', '.join(sorted(set(flagged)))}.")
    else:
        parts.append("No issues were flagged.")
    scores = snapshot.get("scores") or {}
    if scores:
        parts.append(f"Top rule scores: {_fmt_scores(scores)}.")
    if snapshot.get("selected_region"):
        parts.append("A user-selected garment region was used.")
    parts.append(
        f"App result: {snapshot.get('result')} "
        f"(overall {snapshot.get('overall_score', 0.0):.2f})."
    )
    return parts


def build_case_summary(
    snapshot: dict,
    verdict: str | None = None,
    corrected_issue_type: str | None = None,
    comment: str | None = None,
) -> str:
    """Deterministic English summary of one case.

    With ``verdict`` -> a review memory summary; without -> a neutral query
    summary for the same inspection (used at search time so query and stored
    summaries share vocabulary).
    """
    mode = snapshot.get("mode", MODE_WRINKLE)
    label = "Hand" if mode == MODE_HAND else "Wrinkle"
    parts: list[str] = []
    if verdict is not None:
        head = f"{label} {verdict}"
        if corrected_issue_type:
            head += f": user reported {corrected_issue_type}"
        parts.append(head + ".")
    else:
        parts.append(f"{label} inspection.")

    parts.extend(_hand_body(snapshot) if mode == MODE_HAND else _wrinkle_body(snapshot))

    # Interpretation hint for the classic landmark-normalization failure.
    if (
        mode == MODE_HAND
        and verdict == "false_negative"
        and (snapshot.get("debug") or {}).get("hand", {}).get("detected")
        and (snapshot.get("scores") or {})
        and all(v == 0 for v in (snapshot.get("scores") or {}).values())
    ):
        parts.append(
            "This suggests a landmark-normalization failure where visual "
            "contour anomalies are not represented in the 21-point skeleton."
        )

    if comment:
        parts.append(f'User comment: "{comment}"')
    return " ".join(parts)


# --------------------------------------------------------------------------- #
# Phase 1 — save review (+ Phase 2 memory row)
# --------------------------------------------------------------------------- #
def save_review(
    db: Session, inspection: Inspection, payload: ReviewRequest
) -> tuple[ReviewFeedback, ReviewMemory]:
    snapshot = build_snapshot(inspection)
    mode = snapshot["mode"]
    summary = build_case_summary(
        snapshot,
        verdict=payload.user_verdict.value,
        corrected_issue_type=payload.corrected_issue_type,
        comment=payload.user_comment,
    )

    review = ReviewFeedback(
        inspection_id=inspection.id,
        mode=mode,
        user_verdict=payload.user_verdict.value,
        corrected_issue_type=payload.corrected_issue_type,
        user_comment=payload.user_comment,
        selected_region_json=inspection.selected_region_json,
        include_debug_snapshot=payload.include_debug_snapshot,
        include_image_crop=payload.include_image_crop,
        # Flag-only in this phase: reference the already-stored image path,
        # never create crop files.
        image_path_ref=inspection.image_path if payload.include_image_crop else None,
        app_result=snapshot["result"],
        overall_score=snapshot["overall_score"],
        issues_json=json.dumps(snapshot["issues"], ensure_ascii=False),
        scores_json=json.dumps(snapshot["scores"], ensure_ascii=False),
        debug_snapshot_json=(
            json.dumps(snapshot["debug"], ensure_ascii=False)
            if payload.include_debug_snapshot
            else None
        ),
    )
    db.add(review)
    db.flush()  # assign review.id before the memory row

    flagged = _flagged_types(snapshot["issues"])
    memory = ReviewMemory(
        feedback_id=review.id,
        mode=mode,
        verdict=payload.user_verdict.value,
        issue_type=payload.corrected_issue_type or (flagged[0] if flagged else None),
        summary_text=summary,
        embedding_json=json.dumps(get_embedder().embed(summary)),
    )
    db.add(memory)
    db.commit()
    db.refresh(review)
    db.refresh(memory)
    return review, memory


# --------------------------------------------------------------------------- #
# Phase 3 — minimal similar-case retrieval
# --------------------------------------------------------------------------- #
def search_memory(
    db: Session,
    query_text: str,
    mode: str | None = None,
    verdict: str | None = None,
    top_k: int = 5,
) -> list[dict]:
    """Rank stored memories against ``query_text`` by cosine similarity."""
    q = db.query(ReviewMemory)
    if mode:
        q = q.filter(ReviewMemory.mode == mode)
    if verdict:
        q = q.filter(ReviewMemory.verdict == verdict)
    rows = q.all()
    if not rows:
        return []

    embedder = get_embedder()
    query_vec = embedder.embed(query_text)
    scored: list[tuple[float, ReviewMemory]] = []
    for row in rows:
        vec = _load_json(row.embedding_json, None)
        if not vec:  # legacy/absent embedding -> embed the summary on the fly
            vec = embedder.embed(row.summary_text)
        scored.append((cosine_similarity(query_vec, vec), row))
    scored.sort(key=lambda t: t[0], reverse=True)

    return [
        {
            "memory_id": row.id,
            "feedback_id": row.feedback_id,
            "mode": row.mode,
            "verdict": row.verdict,
            "issue_type": row.issue_type,
            "summary_text": row.summary_text,
            "similarity": round(score, 4),
        }
        for score, row in scored[:top_k]
    ]
