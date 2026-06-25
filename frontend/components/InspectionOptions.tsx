"use client";

import type { InspectOptions, ModelStatus } from "@/lib/types";

interface Props {
  options: InspectOptions;
  onChange: (next: InspectOptions) => void;
  status: ModelStatus | null;
  hasOverlays: boolean;
  showMask: boolean;
  showPose: boolean;
  showLines: boolean;
  onToggleOverlay: (which: "mask" | "pose" | "lines") => void;
  disabled?: boolean;
}

export default function InspectionOptions({
  options,
  onChange,
  status,
  hasOverlays,
  showMask,
  showPose,
  showLines,
  onToggleOverlay,
  disabled,
}: Props) {
  const set = (k: keyof InspectOptions, v: boolean) =>
    onChange({ ...options, [k]: v });

  const seg = status?.segmentation;
  const segLabel = status
    ? status.sam_available && status.sam_checkpoint_present
      ? "服領域補助抽出（SAM）"
      : "服領域補助抽出（簡易）"
    : "服領域補助抽出";
  const segStatus = seg
    ? seg.provider === "sam" && seg.sam_available && seg.checkpoint_exists
      ? `SAMが選択範囲内の服領域抽出を補助（${seg.device}${seg.sam_loaded ? "・loaded" : "・lazy"}）`
      : seg.sam_available && !seg.checkpoint_exists
        ? "SAMチェックポイント未配置のため、手動/簡易領域を使用します"
        : "服領域の補助抽出が利用できないため、手動で選択した領域を使用します"
    : "";
  const poseNote = status && !status.mediapipe_available ? "（MediaPipe未導入→簡易）" : "";
  const illu = status?.illustration_feedback_model;
  const illuReady = !!illu?.ready;
  const illuNote = illu
    ? illuReady
      ? `（学習済み: ${illu.training_samples}件）`
      : `（未学習: ${illu.training_samples}件 / 学習に十分なデータが必要）`
    : "";

  return (
    <div className="controls">
      <label className="opt">
        <input
          type="checkbox"
          checked={options.use_segmentation}
          disabled={disabled}
          onChange={(e) => set("use_segmentation", e.target.checked)}
        />
        {segLabel}
      </label>
      {options.use_segmentation && (
        <div className="hint" style={{ marginLeft: 24 }}>
          SAMは服専用の分類器ではありません。服のおおよその範囲を選択すると、その範囲内でマスクを補助します。
        </div>
      )}
      {options.use_segmentation && segStatus && (
        <div className="hint" style={{ marginLeft: 24 }}>
          {segStatus}
          {seg?.last_error ? ` ・ ${seg.last_error}` : ""}
        </div>
      )}
      <label className="opt">
        <input
          type="checkbox"
          checked={options.use_pose_advanced}
          disabled={disabled}
          onChange={(e) => set("use_pose_advanced", e.target.checked)}
        />
        高度な関節判定 {poseNote}
      </label>
      <label className="opt">
        <input
          type="checkbox"
          checked={options.use_illustration_model}
          disabled={disabled || !illuReady}
          onChange={(e) => set("use_illustration_model", e.target.checked)}
        />
        イラスト専用モデル {illuNote}
      </label>
      <label className="opt">
        <input
          type="checkbox"
          checked={options.return_debug_overlays}
          disabled={disabled}
          onChange={(e) => set("return_debug_overlays", e.target.checked)}
        />
        debugオーバーレイを取得
      </label>

      {hasOverlays && (
        <div className="overlay-toggles">
          <span className="control-label">表示:</span>
          <button
            type="button"
            className={`chip${showMask ? " selected" : ""}`}
            onClick={() => onToggleOverlay("mask")}
          >
            服マスク
          </button>
          <button
            type="button"
            className={`chip${showPose ? " selected" : ""}`}
            onClick={() => onToggleOverlay("pose")}
          >
            関節
          </button>
          <button
            type="button"
            className={`chip${showLines ? " selected" : ""}`}
            onClick={() => onToggleOverlay("lines")}
          >
            皺候補線
          </button>
        </div>
      )}
    </div>
  );
}
