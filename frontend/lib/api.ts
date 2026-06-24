// Thin client for the FastAPI backend.

import type {
  FeedbackRequest,
  FeedbackResponse,
  GarmentType,
  InspectResponse,
  ModelStatus,
  RegionPolygon,
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
  region: RegionPolygon | null,
): Promise<InspectResponse> {
  const fd = new FormData();
  fd.append("image", file);
  fd.append("garment_type", garmentType);
  if (region && region.points.length >= 3) {
    // Lasso polygon in original-image pixel coords.
    const payload = {
      points: region.points.map(
        ([x, y]) => [Math.round(x), Math.round(y)] as [number, number],
      ),
    };
    fd.append("selected_region", JSON.stringify(payload));
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

export async function getModelStatus(): Promise<ModelStatus> {
  const res = await fetch(`${API_BASE}/api/model/status`);
  if (!res.ok) throw new Error(await parseError(res));
  return (await res.json()) as ModelStatus;
}
