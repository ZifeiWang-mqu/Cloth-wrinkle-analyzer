// API types — kept in sync with backend/app/schemas.py (the contract source).

export type GarmentType =
  | "shirt"
  | "skirt"
  | "pants"
  | "dress"
  | "jacket"
  | "unknown";

export type IssueType =
  | "gravity_inconsistency"
  | "joint_inconsistency"
  | "tension_ambiguity"
  | "body_volume_inconsistency"
  | "density_inconsistency"
  | "shadow_wrinkle_mismatch";

export type Severity = "low" | "medium" | "high";

export interface BBox {
  x: number;
  y: number;
  w: number;
  h: number;
}

export interface Issue {
  id: string;
  type: IssueType;
  label: string; // 日本語表示名
  severity: Severity;
  bbox: BBox;
  confidence: number;
  message: string; // 日本語説明文
}

export interface DebugInfo {
  pose_detected: boolean;
  garment_region_used: boolean;
  num_wrinkle_candidates: number;
  processing_time_ms: number;
  scores: Record<string, number>;
  notes: string[];
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
  corrected_bbox?: BBox | null;
  corrected_type?: string | null;
  comment?: string | null;
}

export interface FeedbackResponse {
  status: string;
  feedback_id: string;
}

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
};

export const FEEDBACK_OPTIONS: { value: FeedbackKind; label: string }[] = [
  { value: "correct", label: "正しい" },
  { value: "false_positive", label: "誤検出" },
  { value: "wrong_location", label: "位置が違う" },
  { value: "wrong_reason", label: "理由が違う" },
  { value: "missed_issue", label: "見逃しあり" },
];
