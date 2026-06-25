"use client";

import {
  ISSUE_COLORS,
  type Issue,
  type IssueType,
  TYPE_LABELS,
} from "@/lib/types";

interface Props {
  issues: Issue[]; // currently visible issues
}

/**
 * Small legend mapping each detected issue_type to its highlight color, with a
 * per-type count. Only shows types that are actually present. Color-blind-safe
 * because every swatch is paired with a text label.
 */
export default function IssueLegend({ issues }: Props) {
  if (!issues.length) return null;

  const counts = new Map<IssueType, number>();
  for (const issue of issues) {
    counts.set(issue.type, (counts.get(issue.type) ?? 0) + 1);
  }
  const types = Array.from(counts.keys());

  return (
    <div className="legend">
      <span className="legend-title">凡例（{issues.length}件）</span>
      {types.map((t) => (
        <span className="legend-item" key={t}>
          <span className="legend-swatch" style={{ background: ISSUE_COLORS[t] }} />
          {TYPE_LABELS[t]}（{counts.get(t)}）
        </span>
      ))}
    </div>
  );
}
