// API types — kept in sync with backend/app/schemas.py (the contract source).

export type GarmentType =
  | "shirt"
  | "skirt"
  | "pants"
  | "dress"
  | "jacket"
  | "unknown";

export type WrinkleIssueType =
  | "gravity_inconsistency"
  | "joint_inconsistency"
  | "tension_ambiguity"
  | "body_volume_inconsistency"
  | "density_inconsistency"
  | "shadow_wrinkle_mismatch";

export type HandIssueType =
  | "joint_angle_anomaly"
  | "finger_length_anomaly"
  | "thumb_position_anomaly"
  | "wrist_connection_anomaly"
  | "finger_overlap_anomaly"
  | "low_confidence_hand"
  | "hand_detection_failed";

export type IssueType = WrinkleIssueType | HandIssueType;

// 検査モード（服の皺 / 手）
export type InspectMode = "wrinkle" | "hand";

export const WRINKLE_TYPES: WrinkleIssueType[] = [
  "gravity_inconsistency",
  "joint_inconsistency",
  "tension_ambiguity",
  "body_volume_inconsistency",
  "density_inconsistency",
  "shadow_wrinkle_mismatch",
];

export const HAND_TYPES: HandIssueType[] = [
  "joint_angle_anomaly",
  "finger_length_anomaly",
  "thumb_position_anomaly",
  "wrist_connection_anomaly",
  "finger_overlap_anomaly",
  "low_confidence_hand",
  "hand_detection_failed",
];

// 参考表示（スコア閾値を無視して常に表示する情報系タイプ）
export const INFORMATIONAL_TYPES: IssueType[] = [
  "low_confidence_hand",
  "hand_detection_failed",
];

export type Severity = "low" | "medium" | "high";

export interface BBox {
  x: number;
  y: number;
  w: number;
  h: number;
}

// Freehand lasso ("rope") selection: polygon vertices in natural-image px.
export interface RegionPolygon {
  points: [number, number][];
}

export interface EvidenceBox {
  x: number;
  y: number;
  w: number;
  h: number;
  score?: number | null;
  reason?: string | null;
  source?: string | null;
  fallback_broad_bbox?: boolean;
}

export interface Issue {
  id: string;
  type: IssueType;
  category?: "wrinkle" | "hand" | string; // 検査ファミリー（backend既定は"wrinkle"）
  label: string; // 表示名（皺=日本語 / 手=英語MVP）
  severity: Severity;
  bbox: BBox;
  confidence: number;
  message: string; // 日本語説明文
  score: number; // 生スコア 0..1
  threshold: number; // 判定に使われた閾値
  flagged: boolean; // バックエンド判定で検出扱いか
  evidence_boxes?: EvidenceBox[]; // 局所的な根拠ボックス（優先表示）
}

export interface Overlays {
  candidate_lines?: number[][]; // [x1,y1,x2,y2] full-image coords
  pose_landmarks?: Record<string, [number, number]>;
  hand_landmarks?: [number, number][][]; // per hand: 21 [x,y] full-image px
  mask_png?: string; // data URL
}

export interface DebugInfo {
  pose_detected: boolean;
  garment_region_used: boolean;
  num_wrinkle_candidates: number;
  processing_time_ms: number;
  scores: Record<string, number>;
  notes: string[];
  segmentation?: {
    enabled: boolean;
    provider: string;
    mask_available: boolean;
    mask_area_ratio: number;
    fallback_used: boolean;
    reason: string | null;
    device?: string | null;
    model_type?: string | null;
    last_error?: string | null;
  } | null;
  pose?: {
    detected: boolean;
    provider: string;
    advanced?: boolean;
    joint_contexts?: {
      joint_name: string;
      joint_type: string;
      angle_degrees: number;
      bend_strength: number;
      confidence: number;
    }[];
  } | null;
  model_scores?: Record<string, number | null>;
  models_used?: Record<string, boolean>;
  capabilities?: Record<string, unknown> | null;
  removed_lines?: Record<string, number>;
  line_filter?: Record<string, number>;
  overlays?: Overlays | null;
  // 手検査のデバッグ（/api/inspect-hand のみ）
  hand?: {
    detected: boolean;
    backend: string;
    num_hands: number;
    confidences: number[];
    note: string | null;
    features?: Record<string, unknown>[];
    rule_details?: { hand: number; rules: Record<string, unknown>[] }[];
  } | null;
}

export interface InspectOptions {
  use_segmentation: boolean;
  use_pose_advanced: boolean;
  use_illustration_model: boolean;
  return_debug_overlays: boolean;
}

export interface InspectResponse {
  inspection_id: string;
  result: "ok" | "needs_review";
  overall_score: number;
  issues: Issue[];
  debug: DebugInfo;
}

export type FeedbackKind =
  | "correct"
  | "false_positive"
  | "missed_issue"
  | "wrong_location"
  | "wrong_reason";

export interface FeedbackRequest {
  inspection_id: string;
  issue_id?: string | null;
  feedback: FeedbackKind;
  image_id?: string | null;
  garment_type?: string | null;
  issue_type?: string | null;
  original_bbox?: BBox | null;
  corrected_bbox?: BBox | null;
  corrected_type?: string | null;
  confidence?: number | null;
  severity?: string | null;
  source?: string | null;
  comment?: string | null;
}

export interface FeedbackResponse {
  status: string;
  feedback_id: string;
}

export interface ModelStatus {
  model_loaded: boolean;
  model_type: string;
  model_path: string;
  reference_stats_loaded: boolean;
  reference_stats_path: string;
  thresholds_loaded: boolean;
  thresholds_path: string;
  available_garment_models: string[];
  sam_available: boolean;
  sam_checkpoint_present: boolean;
  mediapipe_available: boolean;
  segmentation?: {
    enabled: boolean;
    provider: string;
    sam_available: boolean;
    sam_loaded: boolean;
    checkpoint_path: string | null;
    checkpoint_exists: boolean;
    model_type: string;
    device: string;
    fallback_provider: string;
    last_error: string | null;
  };
  illustration_feedback_model: {
    loaded: boolean;
    ready: boolean;
    training_samples: number;
    positive_samples: number;
    negative_samples: number;
    metrics?: { precision?: number; recall?: number; f1?: number };
  };
  version: string;
}

// issue_type ごとの色（皺=既存パレット / 手=ピンク・シアン系で別ファミリー）
export const ISSUE_COLORS: Record<IssueType, string> = {
  gravity_inconsistency: "#4f8cff", // blue
  joint_inconsistency: "#ff4d4f", // red
  tension_ambiguity: "#ff9f43", // orange
  body_volume_inconsistency: "#a66bff", // purple
  density_inconsistency: "#36c08a", // green
  shadow_wrinkle_mismatch: "#9aa3b2", // gray
  // hand family
  joint_angle_anomaly: "#ff5da2", // pink
  finger_length_anomaly: "#ff8fab", // light pink
  thumb_position_anomaly: "#e254ff", // magenta
  wrist_connection_anomaly: "#00c2d7", // cyan
  finger_overlap_anomaly: "#ff7096", // rose
  low_confidence_hand: "#8899aa", // muted blue-gray (informational)
  hand_detection_failed: "#7a8699", // gray (informational)
};

export const GARMENT_OPTIONS: { value: GarmentType; label: string }[] = [
  { value: "unknown", label: "不明 / 自動" },
  { value: "shirt", label: "シャツ" },
  { value: "skirt", label: "スカート" },
  { value: "pants", label: "パンツ" },
  { value: "dress", label: "ワンピース" },
  { value: "jacket", label: "ジャケット" },
];

export const TYPE_LABELS: Record<IssueType, string> = {
  gravity_inconsistency: "重力と皺の矛盾",
  joint_inconsistency: "関節と皺の矛盾",
  tension_ambiguity: "張力点が不明瞭",
  body_volume_inconsistency: "体の立体と皺の矛盾",
  density_inconsistency: "皺の密度が不自然",
  shadow_wrinkle_mismatch: "陰影と皺の不一致",
  // hand family (English for MVP; Japanese localisation later)
  joint_angle_anomaly: "Impossible joint angle",
  finger_length_anomaly: "Unnatural finger length",
  thumb_position_anomaly: "Thumb on wrong side",
  wrist_connection_anomaly: "Implausible wrist connection",
  finger_overlap_anomaly: "Fingers appear merged",
  low_confidence_hand: "Low-confidence hand detection",
  hand_detection_failed: "Hand detection unavailable",
};

export const FEEDBACK_OPTIONS: { value: FeedbackKind; label: string }[] = [
  { value: "correct", label: "正しい" },
  { value: "false_positive", label: "誤検出" },
  { value: "wrong_location", label: "位置が違う" },
  { value: "wrong_reason", label: "理由が違う" },
  { value: "missed_issue", label: "見逃しあり" },
];

// Per-issue feedback (missed_issue is handled separately by the add-issue form).
export const ISSUE_FEEDBACK_OPTIONS: { value: FeedbackKind; label: string }[] = [
  { value: "correct", label: "正しい" },
  { value: "false_positive", label: "誤検出" },
  { value: "wrong_location", label: "位置が違う" },
  { value: "wrong_reason", label: "理由が違う" },
];

export const SEVERITY_LABELS: Record<Severity, string> = {
  high: "高",
  medium: "中",
  low: "低",
};
