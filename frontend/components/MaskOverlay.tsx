"use client";

interface Props {
  src: string; // data URL (PNG) of the segmentation mask, full-image frame
  width: number; // displayed px
  height: number;
  opacity?: number;
}

/**
 * Semi-transparent garment-mask overlay, scaled to the displayed image.
 * White = garment region kept for analysis.
 */
export default function MaskOverlay({ src, width, height, opacity = 0.3 }: Props) {
  return (
    // eslint-disable-next-line @next/next/no-img-element
    <img
      src={src}
      alt="服マスク"
      style={{
        position: "absolute",
        left: 0,
        top: 0,
        width,
        height,
        opacity,
        pointerEvents: "none",
        mixBlendMode: "screen",
      }}
    />
  );
}
