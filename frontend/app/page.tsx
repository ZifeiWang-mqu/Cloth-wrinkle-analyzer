"use client";

import { useEffect, useMemo, useState } from "react";

import ImageUploader from "@/components/ImageUploader";
import InspectionOptions from "@/components/InspectionOptions";
import IssueControls, {
  type SeverityFilter,
  type SortBy,
} from "@/components/IssueControls";
import MissedIssueForm from "@/components/MissedIssueForm";
import RegionSelector from "@/components/RegionSelector";
import ResultPanel from "@/components/ResultPanel";
import { getModelStatus, inspectWrinkle, sendFeedback } from "@/lib/api";
import {
  type BBox,
  type FeedbackKind,
  GARMENT_OPTIONS,
  type GarmentType,
  type InspectOptions,
  type InspectResponse,
  type IssueType,
  type ModelStatus,
  type RegionPolygon,
  TYPE_LABELS,
} from "@/lib/types";

const ALL_TYPES = Object.keys(TYPE_LABELS) as IssueType[];

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [garmentType, setGarmentType] = useState<GarmentType>("unknown");
  const [region, setRegion] = useState<RegionPolygon | null>(null);
  const [result, setResult] = useState<InspectResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [selectedIssueId, setSelectedIssueId] = useState<string | null>(null);
  const [visibleTypes, setVisibleTypes] = useState<Set<IssueType>>(
    new Set(ALL_TYPES),
  );
  const [severityFilter, setSeverityFilter] = useState<SeverityFilter>("all");
  const [sortBy, setSortBy] = useState<SortBy>("score");
  const [displayThreshold, setDisplayThreshold] = useState(0.4);

  const [regionEditMode, setRegionEditMode] = useState(true);
  const [addMode, setAddMode] = useState(false);
  const [draftIssueBox, setDraftIssueBox] = useState<BBox | null>(null);

  const [modelStatus, setModelStatus] = useState<ModelStatus | null>(null);

  const [options, setOptions] = useState<InspectOptions>({
    use_segmentation: true, // SAM-MVP default (falls back if SAM unavailable)
    use_pose_advanced: true,
    use_illustration_model: true,
    return_debug_overlays: false,
  });
  const [showMask, setShowMask] = useState(true);
  const [showPose, setShowPose] = useState(true);
  const [showLines, setShowLines] = useState(false);

  useEffect(() => {
    getModelStatus()
      .then(setModelStatus)
      .catch(() => setModelStatus(null));
  }, []);

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
    setSelectedIssueId(null);
    setDraftIssueBox(null);
    setAddMode(false);
    setRegionEditMode(true);
  }

  async function handleInspect() {
    if (!file) return;
    setLoading(true);
    setError(null);
    try {
      const res = await inspectWrinkle(file, garmentType, region, options);
      setResult(res);
      setSelectedIssueId(null);
      setRegionEditMode(false);
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
    setSelectedIssueId(null);
    setDraftIssueBox(null);
    setAddMode(false);
    setRegionEditMode(true);
  }

  function toggleOverlay(which: "mask" | "pose" | "lines") {
    if (which === "mask") setShowMask((v) => !v);
    else if (which === "pose") setShowPose((v) => !v);
    else setShowLines((v) => !v);
  }

  function toggleType(t: IssueType) {
    setVisibleTypes((prev) => {
      const next = new Set(prev);
      if (next.has(t)) next.delete(t);
      else next.add(t);
      return next;
    });
  }

  async function handleIssueFeedback(
    issueId: string | null,
    kind: FeedbackKind,
    comment: string,
  ) {
    if (!result) return;
    const issue = result.issues.find((i) => i.id === issueId) || null;
    await sendFeedback({
      inspection_id: result.inspection_id,
      issue_id: issueId,
      feedback: kind,
      image_id: file?.name ?? null,
      garment_type: garmentType,
      issue_type: issue?.type ?? null,
      original_bbox: issue?.bbox ?? null,
      confidence: issue?.confidence ?? null,
      severity: issue?.severity ?? null,
      source: "web",
      comment: comment || null,
    });
  }

  async function handleMissedSubmit(issueType: IssueType, comment: string) {
    if (!result || !draftIssueBox) return;
    await sendFeedback({
      inspection_id: result.inspection_id,
      feedback: "missed_issue",
      image_id: file?.name ?? null,
      garment_type: garmentType,
      issue_type: issueType,
      corrected_bbox: draftIssueBox,
      source: "web",
      comment: comment || null,
    });
    setDraftIssueBox(null);
  }

  const drawMode: "region" | "issue" | "off" = addMode
    ? "issue"
    : regionEditMode
      ? "region"
      : "off";

  const filteredIssues = useMemo(() => {
    if (!result) return [];
    const arr = result.issues.filter(
      (i) =>
        visibleTypes.has(i.type) &&
        i.score >= displayThreshold &&
        (severityFilter === "all" || i.severity === severityFilter),
    );
    arr.sort((a, b) =>
      sortBy === "score" ? b.score - a.score : b.confidence - a.confidence,
    );
    return arr;
  }, [result, visibleTypes, displayThreshold, severityFilter, sortBy]);

  return (
    <main className="app">
      <header className="app-header">
        <div>
          <h1>イラスト皺チェッカー</h1>
          <div className="subtitle">
            衣服の皺が重力・関節・張力・立体・密度・陰影と整合しているか検査します（結果は可能性の提示です）。
          </div>
        </div>
        {modelStatus && (
          <div className="model-status" title={modelStatus.model_path}>
            モデル: {modelStatus.model_loaded ? modelStatus.model_type : "簡易(未学習)"} ・ 閾値
            {modelStatus.thresholds_loaded ? "✓" : "×"} ・ 参照統計
            {modelStatus.reference_stats_loaded ? "✓" : "×"}
          </div>
        )}
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

          <label className="field">検査オプション</label>
          <InspectionOptions
            options={options}
            onChange={setOptions}
            status={modelStatus}
            hasOverlays={!!result?.debug.overlays}
            showMask={showMask}
            showPose={showPose}
            showLines={showLines}
            onToggleOverlay={toggleOverlay}
            disabled={loading}
          />

          {file && (
            <div className="region-tools">
              <button
                type="button"
                className={`btn small ${regionEditMode ? "" : "secondary"}`}
                onClick={() => {
                  setRegionEditMode((v) => !v);
                  setAddMode(false);
                }}
              >
                {regionEditMode ? "服領域: 選択中" : "服領域を選び直す"}
              </button>
              {region && (
                <span className="hint">
                  なげなわ {region.points.length} 点{" "}
                  <button type="button" className="chip" onClick={() => setRegion(null)}>
                    クリア
                  </button>
                </span>
              )}
            </div>
          )}

          <div style={{ marginTop: 14, display: "grid", gap: 8 }}>
            <button className="btn" onClick={handleInspect} disabled={!file || loading}>
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

          {result && (
            <>
              <hr className="sep" />
              <h2>2. 表示フィルタ</h2>
              <IssueControls
                allTypes={ALL_TYPES}
                visibleTypes={visibleTypes}
                onToggleType={toggleType}
                severityFilter={severityFilter}
                onSeverityChange={setSeverityFilter}
                sortBy={sortBy}
                onSortChange={setSortBy}
                displayThreshold={displayThreshold}
                onThresholdChange={setDisplayThreshold}
              />
              <hr className="sep" />
              <MissedIssueForm
                addMode={addMode}
                draftBox={draftIssueBox}
                onToggleAddMode={() => {
                  setAddMode((v) => !v);
                  setRegionEditMode(false);
                }}
                onSubmit={handleMissedSubmit}
                onClearDraft={() => setDraftIssueBox(null)}
              />
            </>
          )}
        </section>

        {/* --- Image stage --- */}
        <section className="panel">
          <h2>プレビュー / 検出</h2>
          {previewUrl ? (
            <RegionSelector
              imageUrl={previewUrl}
              drawMode={loading ? "off" : drawMode}
              region={region}
              draftIssueBox={draftIssueBox}
              issues={filteredIssues}
              selectedIssueId={selectedIssueId}
              onRegionChange={setRegion}
              onIssueBoxChange={setDraftIssueBox}
              onSelectIssue={setSelectedIssueId}
              overlays={result?.debug.overlays}
              showMask={showMask}
              showPose={showPose}
              showLines={showLines}
            />
          ) : (
            <div className="empty">ここに画像が表示されます。</div>
          )}
          {previewUrl && (
            <p className="hint" style={{ marginTop: 8 }}>
              {addMode
                ? "ドラッグで見逃し範囲を指定 → 左の「見逃しとして送信」"
                : regionEditMode
                  ? "なぞって服領域を囲みます（ロープツール・任意。指を離すと確定）。"
                  : "検出枠をクリックすると詳細が右に表示されます。"}
            </p>
          )}
        </section>

        {/* --- Result --- */}
        {result ? (
          <ResultPanel
            result={result}
            issues={filteredIssues}
            selectedIssueId={selectedIssueId}
            onSelectIssue={setSelectedIssueId}
            onFeedback={handleIssueFeedback}
          />
        ) : (
          <section className="panel">
            <h2>検査結果</h2>
            <div className="empty">検査するとここに結果と問題箇所が表示されます。</div>
          </section>
        )}
      </div>
    </main>
  );
}
