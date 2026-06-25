"use client";

import {
  type FeedbackKind,
  type InspectResponse,
  ISSUE_COLORS,
  type Issue,
  SEVERITY_LABELS,
  TYPE_LABELS,
} from "@/lib/types";
import FeedbackButtons from "./FeedbackButtons";

const SCORE_LABELS: Record<string, string> = {
  ...TYPE_LABELS,
  anomaly_model: "異常スコア（学習モデル）",
};

interface Props {
  result: InspectResponse;
  issues: Issue[]; // filtered + sorted (visible)
  selectedIssueId: string | null;
  onSelectIssue: (id: string | null) => void;
  onFeedback: (
    issueId: string | null,
    kind: FeedbackKind,
    comment: string,
  ) => Promise<void>;
}

export default function ResultPanel({
  result,
  issues,
  selectedIssueId,
  onSelectIssue,
  onFeedback,
}: Props) {
  const review = result.result === "needs_review";
  const scoreEntries = Object.entries(result.debug.scores);
  const selected = result.issues.find((i) => i.id === selectedIssueId) || null;

  return (
    <div className="panel">
      <h2>検査結果</h2>

      <div className="score-banner">
        <span className="score-pill">{result.overall_score.toFixed(2)}</span>
        <div>
          <span className={`badge ${review ? "review" : "ok"}`}>
            {review ? "要確認の可能性あり" : "目立つ問題なし"}
          </span>
          <div className="hint">総合スコア（1.0 に近いほど要確認）</div>
        </div>
      </div>

      {/* Score breakdown (rule / anomaly / illustration / final) */}
      {result.debug.model_scores && (
        <div className="score-breakdown">
          {(
            [
              ["rule_score", "ルール"],
              ["anomaly_score", "写真異常"],
              ["illustration_model_score", "専用モデル"],
              ["final_score", "最終"],
            ] as [string, string][]
          ).map(([k, label]) => {
            const v = result.debug.model_scores?.[k];
            return (
              <div className="sb-row" key={k}>
                <span className="sb-label">{label}</span>
                <span className="sb-val">{v == null ? "—" : v.toFixed(2)}</span>
              </div>
            );
          })}
        </div>
      )}

      {/* Selected issue detail */}
      {selected && (
        <div
          className="selected-issue"
          style={{ borderColor: ISSUE_COLORS[selected.type] }}
        >
          <div className="row">
            <span className="label" style={{ color: ISSUE_COLORS[selected.type] }}>
              {selected.label}
            </span>
            <button type="button" className="chip" onClick={() => onSelectIssue(null)}>
              閉じる
            </button>
          </div>
          <div className="meta">
            重大度 {SEVERITY_LABELS[selected.severity]} ・ 信頼度{" "}
            {(selected.confidence * 100).toFixed(0)}% ・ スコア{" "}
            {selected.score.toFixed(2)}（閾値 {selected.threshold.toFixed(2)}）
            {selected.flagged ? "" : " ・ 参考"}
          </div>
          <p className="message">{selected.message}</p>
          <FeedbackButtons issueId={selected.id} onSubmit={onFeedback} />
        </div>
      )}

      {/* Issue list */}
      {issues.length === 0 ? (
        <p className="hint">表示条件に合う検出はありません（閾値やフィルタを調整してください）。</p>
      ) : (
        <div className="issue-list">
          {issues.map((issue, i) => {
            const color = ISSUE_COLORS[issue.type];
            const active = issue.id === selectedIssueId;
            return (
              <button
                key={issue.id}
                type="button"
                className={`issue-row${active ? " active" : ""}`}
                style={{ borderLeftColor: color }}
                onClick={() => onSelectIssue(issue.id)}
              >
                <span className="issue-row-main">
                  <span className="issue-row-num" style={{ background: color }}>
                    {i + 1}
                  </span>
                  <span className="label">{issue.label}</span>
                  {!issue.flagged && <span className="ref-tag">参考</span>}
                </span>
                <span className="meta">
                  {SEVERITY_LABELS[issue.severity]} ・{" "}
                  {(issue.confidence * 100).toFixed(0)}%
                </span>
              </button>
            );
          })}
        </div>
      )}

      <details className="debug">
        <summary>デバッグ情報</summary>
        {result.debug.segmentation && (
          <p className="hint" style={{ marginTop: 6 }}>
            服領域抽出: {result.debug.segmentation.provider}
            {result.debug.segmentation.fallback_used ? "（フォールバック）" : ""} ・ 面積比{" "}
            {(result.debug.segmentation.mask_area_ratio * 100).toFixed(0)}%
            {result.debug.segmentation.device
              ? ` ・ ${result.debug.segmentation.device}`
              : ""}
            {result.debug.segmentation.last_error
              ? ` ・ ${result.debug.segmentation.last_error}`
              : ""}
          </p>
        )}
        {result.debug.line_filter &&
          (result.debug.line_filter.raw_lines ?? 0) > 0 && (
            <p className="hint">
              線フィルタ: 検出 {result.debug.line_filter.raw_lines} → 採用{" "}
              {result.debug.line_filter.kept_lines}（除外 mask{" "}
              {result.debug.line_filter.removed_outside_mask ?? 0}, 境界{" "}
              {result.debug.line_filter.removed_near_boundary ?? 0}, 端{" "}
              {result.debug.line_filter.removed_touches_edge ?? 0}, 長{" "}
              {result.debug.line_filter.removed_too_long ?? 0}, 短{" "}
              {result.debug.line_filter.removed_too_short ?? 0}）
            </p>
          )}
        {result.debug.pose && (
          <p className="hint">
            姿勢: {result.debug.pose.detected ? result.debug.pose.provider : "未検出"}
            {result.debug.pose.joint_contexts?.length
              ? ` ・ 関節 ${result.debug.pose.joint_contexts
                  .map((j) => `${j.joint_name}(${j.angle_degrees.toFixed(0)}°)`)
                  .join(", ")}`
              : ""}
          </p>
        )}
        <div className="scorebars" style={{ marginTop: 8 }}>
          {scoreEntries.map(([type, score]) => (
            <div className="bar-row" key={type}>
              <span className="bar-name">{SCORE_LABELS[type] ?? type}</span>
              <span className="bar-track">
                <span className="bar-fill" style={{ width: `${Math.round(score * 100)}%` }} />
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
