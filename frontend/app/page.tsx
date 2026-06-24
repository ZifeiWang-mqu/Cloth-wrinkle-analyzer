"use client";

import { useEffect, useState } from "react";

import ImageUploader from "@/components/ImageUploader";
import RegionSelector from "@/components/RegionSelector";
import ResultPanel from "@/components/ResultPanel";
import { inspectWrinkle, sendFeedback } from "@/lib/api";
import {
  type BBox,
  type FeedbackKind,
  GARMENT_OPTIONS,
  type GarmentType,
  type InspectResponse,
} from "@/lib/types";

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [garmentType, setGarmentType] = useState<GarmentType>("unknown");
  const [region, setRegion] = useState<BBox | null>(null);
  const [result, setResult] = useState<InspectResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Clean up the object URL when it changes / unmounts.
  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  function handleFile(f: File) {
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setFile(f);
    setPreviewUrl(URL.createObjectURL(f));
    setRegion(null);
    setResult(null);
    setError(null);
  }

  async function handleInspect() {
    if (!file) return;
    setLoading(true);
    setError(null);
    try {
      const res = await inspectWrinkle(file, garmentType, region);
      setResult(res);
    } catch (e) {
      setError(e instanceof Error ? e.message : "検査に失敗しました。");
    } finally {
      setLoading(false);
    }
  }

  function handleReset() {
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setFile(null);
    setPreviewUrl(null);
    setRegion(null);
    setResult(null);
    setError(null);
  }

  async function handleFeedback(
    issueId: string | null,
    kind: FeedbackKind,
    comment: string,
  ) {
    if (!result) return;
    await sendFeedback({
      inspection_id: result.inspection_id,
      issue_id: issueId,
      feedback: kind,
      comment: comment || null,
    });
  }

  return (
    <main className="app">
      <header className="app-header">
        <div>
          <h1>イラスト皺チェッカー</h1>
          <div className="subtitle">
            衣服の皺が重力・関節・張力・立体・密度・陰影と整合しているか検査します。
          </div>
        </div>
      </header>

      <div className="layout">
        {/* --- Controls --- */}
        <section className="panel">
          <h2>1. 画像と設定</h2>
          <ImageUploader onFile={handleFile} disabled={loading} />

          <label className="field" htmlFor="garment">
            服の種類
          </label>
          <select
            id="garment"
            value={garmentType}
            onChange={(e) => setGarmentType(e.target.value as GarmentType)}
            disabled={loading}
          >
            {GARMENT_OPTIONS.map((o) => (
              <option key={o.value} value={o.value}>
                {o.label}
              </option>
            ))}
          </select>

          <p className="hint" style={{ marginTop: 12 }}>
            {file
              ? "プレビュー上でドラッグして服領域を選択できます（任意）。"
              : "画像を選択すると服領域を指定できます。"}
          </p>
          {region && (
            <p className="hint">
              選択領域: x{Math.round(region.x)} y{Math.round(region.y)} ・{" "}
              {Math.round(region.w)}×{Math.round(region.h)}px{" "}
              <button
                type="button"
                className="chip"
                onClick={() => setRegion(null)}
              >
                クリア
              </button>
            </p>
          )}

          <div style={{ marginTop: 14, display: "grid", gap: 8 }}>
            <button
              className="btn"
              onClick={handleInspect}
              disabled={!file || loading}
            >
              {loading ? "検査中…" : result ? "再検査する" : "検査する"}
            </button>
            <button
              className="btn secondary"
              onClick={handleReset}
              disabled={loading || !file}
            >
              リセット
            </button>
          </div>

          {error && <div className="error">{error}</div>}
        </section>

        {/* --- Image stage --- */}
        <section className="panel">
          <h2>2. プレビュー / 検出</h2>
          {previewUrl ? (
            <RegionSelector
              imageUrl={previewUrl}
              region={region}
              issues={result?.issues ?? []}
              interactive={!loading}
              onRegionChange={setRegion}
            />
          ) : (
            <div className="empty">ここに画像が表示されます。</div>
          )}
        </section>

        {/* --- Result --- */}
        {result ? (
          <ResultPanel result={result} onFeedback={handleFeedback} />
        ) : (
          <section className="panel">
            <h2>3. 検査結果</h2>
            <div className="empty">
              検査するとここに結果と問題箇所が表示されます。
            </div>
          </section>
        )}
      </div>
    </main>
  );
}
