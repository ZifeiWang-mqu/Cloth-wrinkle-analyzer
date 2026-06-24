"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type { PointerEvent as ReactPointerEvent } from "react";

import type { BBox, Issue, Overlays, RegionPolygon } from "@/lib/types";
import HeatmapOverlay from "./HeatmapOverlay";

// Pose skeleton bone connections (landmark name pairs).
const BONES: [string, string][] = [
  ["left_shoulder", "right_shoulder"],
  ["left_shoulder", "left_elbow"],
  ["left_elbow", "left_wrist"],
  ["right_shoulder", "right_elbow"],
  ["right_elbow", "right_wrist"],
  ["left_hip", "right_hip"],
  ["left_shoulder", "left_hip"],
  ["right_shoulder", "right_hip"],
  ["left_hip", "left_knee"],
  ["left_knee", "left_ankle"],
  ["right_hip", "right_knee"],
  ["right_knee", "right_ankle"],
];

type DrawMode = "region" | "issue" | "off";
type Pt = { x: number; y: number };

interface Props {
  imageUrl: string;
  drawMode: DrawMode;
  region: RegionPolygon | null; // natural-image coords (lasso)
  draftIssueBox: BBox | null; // natural-image coords (missed issue rect)
  issues: Issue[];
  selectedIssueId: string | null;
  onRegionChange: (region: RegionPolygon | null) => void;
  onIssueBoxChange: (box: BBox | null) => void;
  onSelectIssue: (id: string | null) => void;
  overlays?: Overlays | null;
  showMask?: boolean;
  showPose?: boolean;
  showLines?: boolean;
}

interface Rect {
  x: number;
  y: number;
  w: number;
  h: number;
}

function polygonArea(pts: Pt[]): number {
  let a = 0;
  for (let i = 0; i < pts.length; i++) {
    const j = (i + 1) % pts.length;
    a += pts[i].x * pts[j].y - pts[j].x * pts[i].y;
  }
  return Math.abs(a) / 2;
}

/**
 * Image stage. In "region" mode a freehand lasso (rope) traces a polygon; in
 * "issue" mode a drag makes a rectangle (manual missed issue). Coordinates are
 * captured in displayed px and converted to the natural image frame. Detected
 * issues are overlaid (clickable) at the same scale.
 */
export default function RegionSelector({
  imageUrl,
  drawMode,
  region,
  draftIssueBox,
  issues,
  selectedIssueId,
  onRegionChange,
  onIssueBoxChange,
  onSelectIssue,
  overlays,
  showMask,
  showPose,
  showLines,
}: Props) {
  const imgRef = useRef<HTMLImageElement>(null);
  const [natural, setNatural] = useState({ w: 0, h: 0 });
  const [display, setDisplay] = useState({ w: 0, h: 0 });
  const [lasso, setLasso] = useState<Pt[] | null>(null); // live lasso (display px)
  const [dragRect, setDragRect] = useState<Rect | null>(null); // live issue rect
  const dragging = useRef(false);
  const start = useRef<Pt | null>(null);

  const measure = useCallback(() => {
    const img = imgRef.current;
    if (!img) return;
    setDisplay({ w: img.clientWidth, h: img.clientHeight });
    if (img.naturalWidth) setNatural({ w: img.naturalWidth, h: img.naturalHeight });
  }, []);

  useEffect(() => {
    measure();
    const ro = new ResizeObserver(measure);
    if (imgRef.current) ro.observe(imgRef.current);
    window.addEventListener("resize", measure);
    return () => {
      ro.disconnect();
      window.removeEventListener("resize", measure);
    };
  }, [measure, imageUrl]);

  const scale = natural.w > 0 ? display.w / natural.w : 1;

  function localPoint(e: ReactPointerEvent): Pt {
    const img = imgRef.current!;
    const rect = img.getBoundingClientRect();
    return {
      x: Math.min(Math.max(0, e.clientX - rect.left), rect.width),
      y: Math.min(Math.max(0, e.clientY - rect.top), rect.height),
    };
  }

  function onPointerDown(e: ReactPointerEvent) {
    if (drawMode === "off") {
      onSelectIssue(null);
      return;
    }
    (e.target as Element).setPointerCapture?.(e.pointerId);
    dragging.current = true;
    const p = localPoint(e);
    start.current = p;
    if (drawMode === "region") setLasso([p]);
    else setDragRect({ x: p.x, y: p.y, w: 0, h: 0 });
  }

  function onPointerMove(e: ReactPointerEvent) {
    if (!dragging.current) return;
    const p = localPoint(e);
    if (drawMode === "region") {
      setLasso((prev) => {
        if (!prev) return [p];
        const last = prev[prev.length - 1];
        if (Math.hypot(p.x - last.x, p.y - last.y) < 3) return prev;
        return [...prev, p];
      });
    } else if (drawMode === "issue" && start.current) {
      const s = start.current;
      setDragRect({
        x: Math.min(s.x, p.x),
        y: Math.min(s.y, p.y),
        w: Math.abs(p.x - s.x),
        h: Math.abs(p.y - s.y),
      });
    }
  }

  function onPointerUp() {
    if (!dragging.current) return;
    dragging.current = false;
    const mode = drawMode;
    start.current = null;

    if (mode === "region") {
      const pts = lasso;
      setLasso(null);
      if (!pts || pts.length < 3 || scale === 0 || polygonArea(pts) < 100) {
        onRegionChange(null);
        return;
      }
      onRegionChange({
        points: pts.map((p) => [p.x / scale, p.y / scale] as [number, number]),
      });
      return;
    }

    if (mode === "issue") {
      const r = dragRect;
      setDragRect(null);
      if (!r || r.w < 5 || r.h < 5 || scale === 0) return;
      onIssueBoxChange({
        x: r.x / scale,
        y: r.y / scale,
        w: r.w / scale,
        h: r.h / scale,
      });
    }
  }

  // Committed region polygon -> display points.
  const regionDisplay: Pt[] | null =
    lasso ??
    (region && region.points.length >= 3
      ? region.points.map(([x, y]) => ({ x: x * scale, y: y * scale }))
      : null);
  const regionIsLive = lasso !== null;

  const issueDraft: Rect | null =
    dragRect ??
    (draftIssueBox
      ? {
          x: draftIssueBox.x * scale,
          y: draftIssueBox.y * scale,
          w: draftIssueBox.w * scale,
          h: draftIssueBox.h * scale,
        }
      : null);

  const ptsStr = (pts: Pt[]) => pts.map((p) => `${p.x},${p.y}`).join(" ");

  return (
    <div
      className="image-stage"
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
      style={{ cursor: drawMode === "off" ? "default" : "crosshair" }}
    >
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img ref={imgRef} src={imageUrl} alt="検査対象" onLoad={measure} draggable={false} />

      {regionDisplay && (
        <svg
          className="lasso-svg"
          width={display.w}
          height={display.h}
          style={{ position: "absolute", left: 0, top: 0, pointerEvents: "none" }}
        >
          {regionIsLive ? (
            <polyline
              points={ptsStr(regionDisplay)}
              fill="rgba(79,140,255,0.12)"
              stroke="var(--accent)"
              strokeWidth={2}
              strokeDasharray="5 4"
            />
          ) : (
            <polygon
              points={ptsStr(regionDisplay)}
              fill="rgba(79,140,255,0.15)"
              stroke="var(--accent)"
              strokeWidth={2}
            />
          )}
        </svg>
      )}

      {issueDraft && (
        <div
          className="draft-issue-box"
          style={{ left: issueDraft.x, top: issueDraft.y, width: issueDraft.w, height: issueDraft.h }}
        >
          <span className="tag">追加する見逃し</span>
        </div>
      )}

      {/* Debug overlays (segmentation mask, pose skeleton, candidate lines) */}
      {showMask && overlays?.mask_png && (
        // eslint-disable-next-line @next/next/no-img-element
        <img
          src={overlays.mask_png}
          alt="服マスク"
          style={{
            position: "absolute",
            left: 0,
            top: 0,
            width: display.w,
            height: display.h,
            opacity: 0.3,
            pointerEvents: "none",
            mixBlendMode: "screen",
          }}
        />
      )}
      {(showPose || showLines) && (
        <svg
          width={display.w}
          height={display.h}
          style={{ position: "absolute", left: 0, top: 0, pointerEvents: "none" }}
        >
          {showLines &&
            overlays?.candidate_lines?.map((l, i) => (
              <line
                key={`cl-${i}`}
                x1={l[0] * scale}
                y1={l[1] * scale}
                x2={l[2] * scale}
                y2={l[3] * scale}
                stroke="#36c08a"
                strokeWidth={1}
                opacity={0.7}
              />
            ))}
          {showPose &&
            overlays?.pose_landmarks &&
            BONES.map(([a, b], i) => {
              const pa = overlays.pose_landmarks?.[a];
              const pb = overlays.pose_landmarks?.[b];
              if (!pa || !pb) return null;
              return (
                <line
                  key={`bone-${i}`}
                  x1={pa[0] * scale}
                  y1={pa[1] * scale}
                  x2={pb[0] * scale}
                  y2={pb[1] * scale}
                  stroke="#ff4d4f"
                  strokeWidth={2}
                  opacity={0.85}
                />
              );
            })}
        </svg>
      )}

      <HeatmapOverlay
        issues={issues}
        scale={scale}
        selectedId={selectedIssueId}
        onSelect={onSelectIssue}
      />
    </div>
  );
}
