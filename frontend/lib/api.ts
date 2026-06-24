// Thin client for the FastAPI backend.

import type {
  BBox,
  FeedbackRequest,
  FeedbackResponse,
  GarmentType,
  InspectResponse,
} from "./types";

export const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";

async function parseError(res: Response): Promise<string> {
  try {
    const data = await res.json();
    if (typeof data?.detail === "string") return data.detail;
    return JSON.stringify(data);
  } catch {
    return `${res.status} ${res.statusText}`;
  }
}

export async function inspectWrinkle(
  file: File,
  garmentType: GarmentType,
  region: BBox | null,
): Promise<InspectResponse> {
  const fd = new FormData();
  fd.append("image", file);
  fd.append("garment_type", garmentType);
  if (region) {
    // Backend expects integer-friendly pixel coords in the original image frame.
    const r: BBox = {
      x: Math.round(region.x),
      y: Math.round(region.y),
      w: Math.round(region.w),
      h: Math.round(region.h),
    };
    fd.append("selected_region", JSON.stringify(r));
  }

  const res = await fetch(`${API_BASE}/api/inspect-wrinkle`, {
    method: "POST",
    body: fd,
  });
  if (!res.ok) throw new Error(await parseError(res));
  return (await res.json()) as InspectResponse;
}

export async function sendFeedback(
  payload: FeedbackRequest,
): Promise<FeedbackResponse> {
  const res = await fetch(`${API_BASE}/api/feedback`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error(await parseError(res));
  return (await res.json()) as FeedbackResponse;
}
