"use client";

import type { CSSProperties } from "react";

import type { Issue } from "@/lib/types";

interface Props {
  issues: Issue[];
  scale: number; // displayed px per natural px
}

const SEV_CLASS: Record<string, string> = {
  high: "sev-high",
  medium: "sev-medium",
  low: "sev-low",
};

/**
 * Draws detected issue bounding boxes over the image. Boxes are positioned in
 * the natural image frame and scaled to the displayed size, so they stay
 * aligned regardless of how the image is resized in the layout.
 */
export default function HeatmapOverlay({ issues, scale }: Props) {
  return (
    <>
      {issues.map((issue) => {
        const { x, y, w, h } = issue.bbox;
        const style: CSSProperties = {
          left: x * scale,
          top: y * scale,
          width: w * scale,
          height: h * scale,
        };
        return (
          <div
            key={issue.id}
            className={`issue-box ${SEV_CLASS[issue.severity] ?? "sev-low"}`}
            style={style}
            title={`${issue.label}（信頼度 ${(issue.confidence * 100).toFixed(0)}%）\n${issue.message}`}
          >
            <span className="tag">{issue.label}</span>
          </div>
        );
      })}
    </>
  );
}
