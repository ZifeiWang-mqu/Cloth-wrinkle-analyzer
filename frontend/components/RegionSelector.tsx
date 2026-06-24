"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type { PointerEvent as ReactPointerEvent } from "react";

import type { BBox, Issue } from "@/lib/types";
import HeatmapOverlay from "./HeatmapOverlay";

interface Props {
  imageUrl: string;
  region: BBox | null; // natural-image coords
  issues: Issue[];
  interactive: boolean;
  onRegionChange: (region: BBox | null) => void;
}

interface Rect {
  x: number;
  y: number;
  w: number;
  h: number;
}

/**
 * Displays the uploaded image and lets the user drag a rectangle to select the
 * garment region. Selection is captured in displayed pixels and converted to
 * the natural image frame before being reported upward. Detected issue boxes
 * are overlaid via HeatmapOverlay using the same scale.
 */
export default function RegionSelector({
  imageUrl,
  region,
  issues,
  interactive,
  onRegionChange,
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
    if (img.naturalWidth) {
      setNatural({ w: img.naturalWidth, h: img.naturalHeight });
    }
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
    if (!interactive) return;
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
    start.current = null;
    if (!r || r.w < 5 || r.h < 5 || scale === 0) {
      setDragRect(null);
      onRegionChange(null);
      return;
    }
    // Convert displayed px -> natural image px, then let the committed region
    // (rendered as region * scale) drive the box so it stays resize-safe.
    onRegionChange({
      x: r.x / scale,
      y: r.y / scale,
      w: r.w / scale,
      h: r.h / scale,
    });
    setDragRect(null);
  }

  // The box to render: live drag, else the committed region (natural->display).
  const shownBox: Rect | null = dragRect
    ? dragRect
    : region
      ? {
          x: region.x * scale,
          y: region.y * scale,
          w: region.w * scale,
          h: region.h * scale,
        }
      : null;

  return (
    <div
      className="image-stage"
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
      style={{ cursor: interactive ? "crosshair" : "default" }}
    >
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img ref={imgRef} src={imageUrl} alt="検査対象" onLoad={measure} draggable={false} />

      {shownBox && (
        <div
          className="selection-box"
          style={{
            left: shownBox.x,
            top: shownBox.y,
            width: shownBox.w,
            height: shownBox.h,
          }}
        />
      )}

      <HeatmapOverlay issues={issues} scale={scale} />
    </div>
  );
}
