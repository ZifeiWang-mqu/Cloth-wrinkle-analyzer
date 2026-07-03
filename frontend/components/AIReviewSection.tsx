"use client";

import { useEffect, useState } from "react";

import { getAIReviewCapability, requestAIReview } from "@/lib/api";
import { type AIReviewResponse, VISUAL_PROBLEM_LABELS } from "@/lib/types";

interface Props {
  inspectionId: string;
}

/**
 * Prominent "AIによる視覚レビュー" card. Calls the backend endpoint
 * POST /api/inspection/ai-review (the OpenAI key stays server-side; the
 * frontend never talks to OpenAI). Hidden button becomes a disabled notice
 * when the backend reports the capability as unavailable.
 */
export default function AIReviewSection({ inspectionId }: Props) {
  const [available, setAvailable] = useState<boolean | null>(null);
  const [status, setStatus] = useState<"idle" | "loading" | "done" | "error">(
    "idle",
  );
  const [result, setResult] = useState<AIReviewResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    getAIReviewCapability().then((c) => {
      if (active) setAvailable(c.available);
    });
    return () => {
      active = false;
    };
  }, []);

  async function run() {
    if (status === "loading") return;
    setStatus("loading");
    setError(null);
    try {
      const res = await requestAIReview(inspectionId, "ja");
      setResult(res);
      setStatus("done");
    } catch (e) {
      setStatus("error");
      setError(e instanceof Error ? e.message : "AIレビューに失敗しました。");
    }
  }

  return (
    <div className="ai-review">
      <div className="ai-review-title">AIによる視覚レビュー</div>
      <p className="hint">
        現在の検出結果と画像をもとに、AIが手や皺の見た目を補助的にレビューします。
        これは補助的な確認であり、確定診断ではありません。
      </p>

      {available === false ? (
        <button type="button" className="btn secondary" disabled>
          AIレビューは現在利用できません
        </button>
      ) : (
        <button
          type="button"
          className="btn"
          onClick={run}
          disabled={status === "loading" || available === null}
        >
          {status === "loading"
            ? "AIが画像を確認しています…"
            : result
              ? "もう一度AIに見てもらう"
              : "AIに詳しく見てもらう"}
        </button>
      )}

      {error && <div className="error">{error}</div>}

      {result && status === "done" && (
        <div className="ai-review-result">
          <div className="ai-section">
            <div className="ai-section-title">概要</div>
            <p>{result.summary}</p>
          </div>

          <div className="ai-section">
            <div className="ai-section-title">検出された可能性のある問題</div>
            {result.detected_visual_problems.length === 0 ? (
              <p className="hint">AIは明確な問題を検出しませんでした。</p>
            ) : (
              <ul className="ai-problems">
                {result.detected_visual_problems.map((p, i) => (
                  <li key={i}>
                    <span className="ai-problem-head">
                      {VISUAL_PROBLEM_LABELS[p.type] ?? p.type} /{" "}
                      {(p.confidence * 100).toFixed(0)}%
                    </span>
                    <div className="ai-problem-desc">{p.description}</div>
                  </li>
                ))}
              </ul>
            )}
          </div>

          <div className="ai-section">
            <div className="ai-section-title">既存検出器との比較</div>
            <p>{result.detector_comparison}</p>
          </div>

          <div className="ai-section">
            <div className="ai-section-title">おすすめの確認・修正ポイント</div>
            <p>{result.recommended_action}</p>
          </div>

          <div className="ai-section">
            <div className="ai-section-title">注意点</div>
            <p>{result.limitations}</p>
          </div>

          <p className="hint">
            使用モデル: {result.meta.model} ・ 解析範囲: {result.meta.crop.source}
          </p>
        </div>
      )}
    </div>
  );
}
