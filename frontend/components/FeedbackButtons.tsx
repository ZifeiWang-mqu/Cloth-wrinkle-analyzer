"use client";

import { useState } from "react";

import { FEEDBACK_OPTIONS, type FeedbackKind } from "@/lib/types";

interface Props {
  issueId: string | null;
  onSubmit: (
    issueId: string | null,
    kind: FeedbackKind,
    comment: string,
  ) => Promise<void>;
}

export default function FeedbackButtons({ issueId, onSubmit }: Props) {
  const [kind, setKind] = useState<FeedbackKind | null>(null);
  const [comment, setComment] = useState("");
  const [status, setStatus] = useState<"idle" | "saving" | "saved" | "error">(
    "idle",
  );
  const [err, setErr] = useState<string | null>(null);

  async function submit() {
    if (!kind) return;
    setStatus("saving");
    setErr(null);
    try {
      await onSubmit(issueId, kind, comment);
      setStatus("saved");
    } catch (e) {
      setStatus("error");
      setErr(e instanceof Error ? e.message : "送信に失敗しました。");
    }
  }

  return (
    <div>
      <div className="feedback">
        {FEEDBACK_OPTIONS.map((opt) => (
          <button
            key={opt.value}
            type="button"
            className={`chip${kind === opt.value ? " selected" : ""}`}
            onClick={() => {
              setKind(opt.value);
              setStatus("idle");
            }}
          >
            {opt.label}
          </button>
        ))}
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
          disabled={!kind || status === "saving"}
        >
          {status === "saving" ? "送信中…" : "フィードバック送信"}
        </button>
        {status === "saved" && (
          <span className="feedback-saved"> 保存しました ✓</span>
        )}
      </div>
      {err && <div className="error">{err}</div>}
    </div>
  );
}
