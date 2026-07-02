"use client";

import { useState } from "react";

import { saveReview } from "@/lib/api";
import {
  HAND_REVIEW_LABELS,
  HAND_TYPES,
  type InspectMode,
  TYPE_LABELS,
  type UserVerdict,
  VERDICT_OPTIONS,
  WRINKLE_TYPES,
} from "@/lib/types";

// Detector rule types usable as corrections (informational statuses excluded).
const HAND_RULE_TYPES = HAND_TYPES.filter(
  (t) => t !== "low_confidence_hand" && t !== "hand_detection_failed",
);

interface Props {
  inspectionId: string;
  mode: InspectMode;
}

/**
 * Per-INSPECTION review ("この判定は正しいですか？") feeding the review-memory
 * layer via POST /api/review. Separate from the per-issue FeedbackButtons.
 * The parent keys this component by inspection_id so state resets per result.
 */
export default function ReviewPanel({ inspectionId, mode }: Props) {
  const [verdict, setVerdict] = useState<UserVerdict | null>(null);
  const [correctedType, setCorrectedType] = useState<string>("");
  const [comment, setComment] = useState("");
  const [includeSnapshot, setIncludeSnapshot] = useState(true);
  const [includeCrop, setIncludeCrop] = useState(false);
  const [status, setStatus] = useState<"idle" | "saving" | "saved" | "error">(
    "idle",
  );
  const [summary, setSummary] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Hand mode offers the broader review-only labels first (users must be able
  // to describe problems the detector cannot represent yet), then rule types.
  const correctionOptions: { value: string; label: string }[] =
    mode === "hand"
      ? [
          ...HAND_REVIEW_LABELS,
          ...HAND_RULE_TYPES.map((t) => ({ value: t, label: TYPE_LABELS[t] })),
        ]
      : WRINKLE_TYPES.map((t) => ({ value: t, label: TYPE_LABELS[t] }));
  const showTypeSelect =
    verdict === "false_positive" || verdict === "false_negative";

  async function submit() {
    if (!verdict || status === "saving") return;
    setStatus("saving");
    setError(null);
    try {
      const res = await saveReview({
        inspection_id: inspectionId,
        user_verdict: verdict,
        corrected_issue_type: showTypeSelect && correctedType ? correctedType : null,
        user_comment: comment || null,
        include_debug_snapshot: includeSnapshot,
        include_image_crop: includeCrop,
      });
      setSummary(res.summary_text);
      setStatus("saved");
    } catch (e) {
      setStatus("error");
      setError(e instanceof Error ? e.message : "保存に失敗しました。");
    }
  }

  if (status === "saved") {
    return (
      <div className="review-panel">
        <div className="review-title">この判定は正しいですか？</div>
        <p className="feedback-saved">レビューを保存しました ✓（今後の類似事例検索に利用されます）</p>
        {summary && <p className="review-summary hint">{summary}</p>}
        <button
          type="button"
          className="chip"
          onClick={() => {
            setStatus("idle");
            setSummary(null);
          }}
        >
          修正して再送信
        </button>
      </div>
    );
  }

  return (
    <div className="review-panel">
      <div className="review-title">この判定は正しいですか？</div>
      <div className="feedback">
        {VERDICT_OPTIONS.map((opt) => (
          <button
            key={opt.value}
            type="button"
            className={`chip${verdict === opt.value ? " selected" : ""}`}
            onClick={() => setVerdict(opt.value)}
            disabled={status === "saving"}
          >
            {opt.label}
          </button>
        ))}
      </div>

      {verdict && (
        <>
          {showTypeSelect && (
            <>
              <label className="field" htmlFor="review-corrected-type">
                該当するissueタイプ（任意）
              </label>
              <select
                id="review-corrected-type"
                value={correctedType}
                onChange={(e) => setCorrectedType(e.target.value)}
                disabled={status === "saving"}
              >
                <option value="">指定なし</option>
                {correctionOptions.map((o) => (
                  <option key={o.value} value={o.value}>
                    {o.label}
                  </option>
                ))}
              </select>
            </>
          )}

          <textarea
            placeholder="コメント（任意）"
            value={comment}
            onChange={(e) => setComment(e.target.value)}
            style={{ marginTop: 8 }}
            disabled={status === "saving"}
          />

          <label className="opt">
            <input
              type="checkbox"
              checked={includeSnapshot}
              onChange={(e) => setIncludeSnapshot(e.target.checked)}
              disabled={status === "saving"}
            />
            デバッグ情報を保存する（推奨）
          </label>
          <label className="opt">
            <input
              type="checkbox"
              checked={includeCrop}
              onChange={(e) => setIncludeCrop(e.target.checked)}
              disabled={status === "saving"}
            />
            画像参照を保存する
          </label>

          <div style={{ marginTop: 8 }}>
            <button
              type="button"
              className="btn small"
              onClick={submit}
              disabled={status === "saving"}
            >
              {status === "saving" ? "保存中…" : "レビューを送信"}
            </button>
          </div>
        </>
      )}

      {error && <div className="error">{error}</div>}
    </div>
  );
}
