"use client";

import { useState } from "react";

import {
  type BBox,
  ISSUE_COLORS,
  type IssueType,
  TYPE_LABELS,
} from "@/lib/types";

interface Props {
  addMode: boolean;
  draftBox: BBox | null;
  onToggleAddMode: () => void;
  onSubmit: (issueType: IssueType, comment: string) => Promise<void>;
  onClearDraft: () => void;
}

const TYPES = Object.keys(TYPE_LABELS) as IssueType[];

/**
 * Lets the user report a missed issue: enable add-mode, drag a box on the
 * image, pick the issue type, then submit as `missed_issue` feedback.
 */
export default function MissedIssueForm({
  addMode,
  draftBox,
  onToggleAddMode,
  onSubmit,
  onClearDraft,
}: Props) {
  const [issueType, setIssueType] = useState<IssueType>("joint_inconsistency");
  const [comment, setComment] = useState("");
  const [status, setStatus] = useState<"idle" | "saving" | "saved" | "error">(
    "idle",
  );
  const [err, setErr] = useState<string | null>(null);

  async function submit() {
    if (!draftBox) return;
    setStatus("saving");
    setErr(null);
    try {
      await onSubmit(issueType, comment);
      setStatus("saved");
      setComment("");
    } catch (e) {
      setStatus("error");
      setErr(e instanceof Error ? e.message : "送信に失敗しました。");
    }
  }

  return (
    <div className="missed-form">
      <div className="row">
        <span className="label">見逃しの追加</span>
        <button
          type="button"
          className={`btn small ${addMode ? "" : "secondary"}`}
          onClick={onToggleAddMode}
        >
          {addMode ? "追加モード: ON" : "追加モード: OFF"}
        </button>
      </div>

      {addMode && (
        <>
          <p className="hint">
            画像上をドラッグして、AIが見逃した不整合の範囲を囲んでください。
          </p>
          <label className="field" htmlFor="missed-type">
            種類
          </label>
          <select
            id="missed-type"
            value={issueType}
            onChange={(e) => setIssueType(e.target.value as IssueType)}
          >
            {TYPES.map((t) => (
              <option key={t} value={t}>
                {TYPE_LABELS[t]}
              </option>
            ))}
          </select>

          <div className="draft-status">
            {draftBox ? (
              <span>
                <span
                  className="dot"
                  style={{ background: ISSUE_COLORS[issueType] }}
                />
                範囲 {Math.round(draftBox.w)}×{Math.round(draftBox.h)}px{" "}
                <button type="button" className="chip" onClick={onClearDraft}>
                  クリア
                </button>
              </span>
            ) : (
              <span className="hint">まだ範囲が指定されていません。</span>
            )}
          </div>

          <textarea
            placeholder="コメント（任意）"
            value={comment}
            onChange={(e) => setComment(e.target.value)}
            style={{ marginTop: 6 }}
          />
          <div style={{ marginTop: 6 }}>
            <button
              type="button"
              className="btn small"
              onClick={submit}
              disabled={!draftBox || status === "saving"}
            >
              {status === "saving" ? "送信中…" : "見逃しとして送信"}
            </button>
            {status === "saved" && (
              <span className="feedback-saved"> 保存しました ✓</span>
            )}
          </div>
          {err && <div className="error">{err}</div>}
        </>
      )}
    </div>
  );
}
