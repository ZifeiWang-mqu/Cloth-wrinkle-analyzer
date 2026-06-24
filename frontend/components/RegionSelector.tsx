"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type { PointerEvent as ReactPointerEvent } from "react";

import type { BBox, Issue } from "@/lib/types";
import HeatmapOverlay from "./HeatmapOverlay";

type DrawMode = "region" | "issue" | "off";

interface Props {
  imageUrl: string;
  drawMode: DrawMode;
  region: BBox | null; // natural-image coords
  draftIssueBox: BBox | null; // natural-image coords (manual missed issue)
  issues: Issue[]; // visible issues to overlay
  selectedIssueId: string | null;
  onRegionChange: (region: BBox | null) => void;
  onIssueBoxChange: (box: BBox | null) => void;
  onSelectIssue: (id: string | null) => void;
}

interface Rect {
  x: number;
  y: number;
  w: number;
  h: number;
}

/**
 * Image stage. A drag draws either the garment region or a missed-issue box
 * (per `drawMode`), captured in displayed px and converted to the natural
 * image frame. Detected issues are overlaid (clickable) at the same scale.
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
}: Props) {
  const imgRef = useRef<HTMLImageElement>(null);
  const [natural, setNatural] = useState({ w: 0, h: 0 });
  const [display, setDisplay] = useState({ w: 0, h: 0 });
  const [dragRect, setDragRect] = useState<Rect | null>(null);
  const dragging = useRef(false);
  const start = useRef<{ x: number; y: number } | null>(null);

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

  function localPoint(e: ReactPointerEvent): { x: number; y: number } {
    const img = imgRef.current!;
    const rect = img.getBoundingClientRect();
    const x = Math.min(Math.max(0, e.clientX - rect.left), rect.width);
    const y = Math.min(Math.max(0, e.clientY - rect.top), rect.height);
    return { x, y };
  }

  function onPointerDown(e: ReactPointerEvent) {
    if (drawMode === "off") {
      onSelectIssue(null); // click empty area to deselect
      return;
    }
    (e.target as Element).setPointerCapture?.(e.pointerId);
    dragging.current = true;
    const p = localPoint(e);
    start.current = p;
    setDragRect({ x: p.x, y: p.y, w: 0, h: 0 });
  }

  function onPointerMove(e: ReactPointerEvent) {
    if (!dragging.current || !start.current) return;
    const p = localPoint(e);
    const s = start.current;
    setDragRect({
      x: Math.min(s.x, p.x),
      y: Math.min(s.y, p.y),
      w: Math.abs(p.x - s.x),
      h: Math.abs(p.y - s.y),
    });
  }

  function onPointerUp() {
    if (!dragging.current) return;
    dragging.current = false;
    const r = dragRect;
    const mode = drawMode;
    start.current = null;
    setDragRect(null);
    if (!r || r.w < 5 || r.h < 5 || scale === 0) {
      if (mode === "region") onRegionChange(null);
      return;
    }
    const natbox: BBox = {
      x: r.x / scale,
      y: r.y / scale,
      w: r.w / scale,
      h: r.h / scale,
    };
    if (mode === "region") onRegionChange(natbox);
    else if (mode === "issue") onIssueBoxChange(natbox);
  }

  const toDisplay = (b: BBox): Rect => ({
    x: b.x * scale,
    y: b.y * scale,
    w: b.w * scale,
    h: b.h * scale,
  });

  const regionBox = dragRect && drawMode === "region" ? dragRect : region ? toDisplay(region) : null;
  const issueDraft = dragRect && drawMode === "issue" ? dragRect : draftIssueBox ? toDisplay(draftIssueBox) : null;

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

      {regionBox && (
        <div
          className="selection-box"
          style={{ left: regionBox.x, top: regionBox.y, width: regionBox.w, height: regionBox.h }}
        />
      )}

      {issueDraft && (
        <div
          className="draft-issue-box"
          style={{ left: issueDraft.x, top: issueDraft.y, width: issueDraft.w, height: issueDraft.h }}
        >
          <span className="tag">追加する見逃し</span>
        </div>
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
