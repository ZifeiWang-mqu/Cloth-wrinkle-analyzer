"use client";

import type { CSSProperties } from "react";

import { ISSUE_COLORS, type Issue } from "@/lib/types";

interface Props {
  issues: Issue[]; // already filtered to what's visible
  scale: number; // displayed px per natural px
  selectedId: string | null;
  onSelect: (id: string) => void;
}

/**
 * Draws issue bounding boxes over the image, colored by issue_type. Boxes are
 * clickable (select) and the selected one is emphasised. Unflagged issues
 * (below the backend judgment threshold) are drawn dashed.
 */
export default function HeatmapOverlay({
  issues,
  scale,
  selectedId,
  onSelect,
}: Props) {
  return (
    <>
      {issues.map((issue) => {
        const { x, y, w, h } = issue.bbox;
        const color = ISSUE_COLORS[issue.type] ?? "#9aa3b2";
        const selected = issue.id === selectedId;
        const style: CSSProperties = {
          left: x * scale,
          top: y * scale,
          width: w * scale,
          height: h * scale,
          borderColor: color,
          borderStyle: issue.flagged ? "solid" : "dashed",
          borderWidth: selected ? 3 : 2,
          background: selected ? `${color}26` : "transparent",
          boxShadow: selected ? `0 0 0 2px ${color}66` : "none",
          zIndex: selected ? 5 : 2,
        };
        return (
          <div
            key={issue.id}
            className="issue-box"
            style={style}
            title={`${issue.label}（信頼度 ${(issue.confidence * 100).toFixed(0)}%）\n${issue.message}`}
            onPointerDown={(e) => e.stopPropagation()}
            onClick={(e) => {
              e.stopPropagation();
              onSelect(issue.id);
            }}
          >
            <span className="tag" style={{ background: color }}>
              {issue.label}
            </span>
          </div>
        );
      })}
    </>
  );
}
