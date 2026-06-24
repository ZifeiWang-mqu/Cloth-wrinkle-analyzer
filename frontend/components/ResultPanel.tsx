"use client";

import {
  type FeedbackKind,
  type InspectResponse,
  TYPE_LABELS,
} from "@/lib/types";
import FeedbackButtons from "./FeedbackButtons";

// Includes the non-issue "anomaly_model" key that appears in debug.scores.
const SCORE_LABELS: Record<string, string> = {
  ...TYPE_LABELS,
  anomaly_model: "異常スコア（学習モデル）",
};

interface Props {
  result: InspectResponse;
  onFeedback: (
    issueId: string | null,
    kind: FeedbackKind,
    comment: string,
  ) => Promise<void>;
}

const SEV_CLASS: Record<string, string> = {
  high: "sev-high",
  medium: "sev-medium",
  low: "sev-low",
};

const SEV_LABEL: Record<string, string> = {
  high: "高",
  medium: "中",
  low: "低",
};

export default function ResultPanel({ result, onFeedback }: Props) {
  const review = result.result === "needs_review";
  const scoreEntries = Object.entries(result.debug.scores);

  return (
    <div className="panel">
      <h2>検査結果</h2>

      <div className="score-banner">
        <span className="score-pill">{result.overall_score.toFixed(2)}</span>
        <div>
          <span className={`badge ${review ? "review" : "ok"}`}>
            {review ? "要確認" : "問題なし"}
          </span>
          <div className="hint">総合スコア（1.0 に近いほど要確認）</div>
        </div>
      </div>

      {result.issues.length === 0 ? (
        <p className="hint">閾値を超える不整合は検出されませんでした。</p>
      ) : (
        result.issues.map((issue) => (
          <div
            key={issue.id}
            className={`issue-card ${SEV_CLASS[issue.severity] ?? "sev-low"}`}
          >
            <div className="row">
              <span className="label">{issue.label}</span>
              <span className="meta">
                重大度 {SEV_LABEL[issue.severity] ?? issue.severity} ・ 信頼度{" "}
                {(issue.confidence * 100).toFixed(0)}%
              </span>
            </div>
            <p className="message">{issue.message}</p>
            <FeedbackButtons issueId={issue.id} onSubmit={onFeedback} />
          </div>
        ))
      )}

      <h3 style={{ marginTop: 18, fontSize: 14 }}>見逃しの報告</h3>
      <p className="hint">検出されなかった不整合があればこちらから報告できます。</p>
      <FeedbackButtons issueId={null} onSubmit={onFeedback} />

      <details className="debug">
        <summary>デバッグ情報</summary>
        <div className="scorebars" style={{ marginTop: 8 }}>
          {scoreEntries.map(([type, score]) => (
            <div className="bar-row" key={type}>
              <span className="bar-name">{SCORE_LABELS[type] ?? type}</span>
              <span className="bar-track">
                <span
                  className="bar-fill"
                  style={{ width: `${Math.round(score * 100)}%` }}
                />
              </span>
              <span className="bar-val">{score.toFixed(2)}</span>
            </div>
          ))}
        </div>
        <pre>
          {JSON.stringify(
            {
              inspection_id: result.inspection_id,
              pose_detected: result.debug.pose_detected,
              garment_region_used: result.debug.garment_region_used,
              num_wrinkle_candidates: result.debug.num_wrinkle_candidates,
              processing_time_ms: result.debug.processing_time_ms,
              notes: result.debug.notes,
            },
            null,
            2,
          )}
        </pre>
      </details>
    </div>
  );
}
