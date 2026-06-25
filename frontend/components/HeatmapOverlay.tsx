"use client";

import type { CSSProperties } from "react";

import { type EvidenceBox, ISSUE_COLORS, type Issue } from "@/lib/types";

interface Props {
  issues: Issue[]; // already filtered to what's visible (order = list order)
  scale: number; // displayed px per natural px
  selectedId: string | null;
  onSelect: (id: string) => void;
  showLabels?: boolean;
  showPrecise?: boolean; // local evidence boxes (default)
  showBroad?: boolean; // original broad issue bbox (debug/compare)
}

const FILL_OPACITY: Record<string, number> = {
  low: 0.18,
  medium: 0.24,
  high: 0.3,
};

function hexToRgba(hex: string, alpha: number): string {
  const h = hex.replace("#", "");
  const full = h.length === 3 ? h.split("").map((c) => c + c).join("") : h;
  const r = parseInt(full.slice(0, 2), 16);
  const g = parseInt(full.slice(2, 4), 16);
  const b = parseInt(full.slice(4, 6), 16);
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

type RenderBox = {
  issue: Issue;
  issueNo: number; // 1-based issue index (matches the list)
  sub: number; // 1-based sub-index within the issue, or 0 if single
  box: { x: number; y: number; w: number; h: number };
  kind: "evidence" | "fallback" | "broad";
};

/**
 * Renders precise local evidence boxes per issue (preferred), falling back to
 * the broad bbox only when no evidence exists. The original broad bbox can be
 * shown separately for debug/comparison. Color = issue_type; fill/border encode
 * severity; fallback/broad boxes are dashed + low opacity. Number badges match
 * the result list (with sub-index when an issue has multiple boxes).
 */
export default function HeatmapOverlay({
  issues,
  scale,
  selectedId,
  onSelect,
  showLabels = true,
  showPrecise = true,
  showBroad = false,
}: Props) {
  const renderBoxes: RenderBox[] = [];
  issues.forEach((issue, i) => {
    const issueNo = i + 1;
    const ev: EvidenceBox[] = issue.evidence_boxes ?? [];
    const precise = ev.filter((b) => !b.fallback_broad_bbox);
    const fallback = ev.filter((b) => b.fallback_broad_bbox);

    if (showPrecise) {
      const list = precise.length ? precise : fallback;
      const kind: RenderBox["kind"] = precise.length ? "evidence" : "fallback";
      list.forEach((b, j) =>
        renderBoxes.push({
          issue,
          issueNo,
          sub: list.length > 1 ? j + 1 : 0,
          box: b,
          kind,
        }),
      );
    }
    if (showBroad) {
      renderBoxes.push({
        issue,
        issueNo,
        sub: 0,
        box: issue.bbox,
        kind: "broad",
      });
    }
  });

  return (
    <>
      {renderBoxes.map((rb, idx) => {
        const { issue, box, kind } = rb;
        const color = ISSUE_COLORS[issue.type] ?? "#9aa3b2";
        const selected = issue.id === selectedId;
        const broadish = kind !== "evidence";
        const baseAlpha = broadish ? 0.1 : FILL_OPACITY[issue.severity] ?? 0.22;
        const fillAlpha = Math.min(0.5, baseAlpha + (selected ? 0.12 : 0));
        const borderW = issue.severity === "high" && !broadish ? 3 : 2;
        const numText = rb.sub ? `${rb.issueNo}.${rb.sub}` : `${rb.issueNo}`;
        const style: CSSProperties = {
          left: box.x * scale,
          top: box.y * scale,
          width: box.w * scale,
          height: box.h * scale,
          borderColor: color,
          borderStyle: broadish || !issue.flagged ? "dashed" : "solid",
          borderWidth: selected ? borderW + 1 : borderW,
          background: hexToRgba(color, fillAlpha),
          boxShadow: selected ? `0 0 0 2px ${hexToRgba(color, 0.5)}` : "none",
          zIndex: selected ? 6 : broadish ? 2 : 3,
        };
        return (
          <div
            key={`${issue.id}-${kind}-${idx}`}
            className="issue-box"
            style={style}
            title={`${numText}. ${issue.label}（${issue.severity}）\n${box && (box as EvidenceBox).reason ? (box as EvidenceBox).reason : issue.message}`}
            onPointerDown={(e) => e.stopPropagation()}
            onClick={(e) => {
              e.stopPropagation();
              onSelect(issue.id);
            }}
          >
            {!broadish && (
              <span className="issue-num" style={{ background: color }}>
                {numText}
              </span>
            )}
            {showLabels && !broadish && rb.sub <= 1 && (
              <span className="tag" style={{ background: color }}>
                {issue.label}
              </span>
            )}
          </div>
        );
      })}
    </>
  );
}
