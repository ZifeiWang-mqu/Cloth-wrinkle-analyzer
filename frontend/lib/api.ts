// Thin client for the FastAPI backend.

import type {
  FeedbackRequest,
  FeedbackResponse,
  GarmentType,
  InspectOptions,
  InspectResponse,
  ModelStatus,
  RegionPolygon,
  ReviewRequest,
  ReviewResponse,
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

function appendRegion(fd: FormData, region: RegionPolygon | null): void {
  if (region && region.points.length >= 3) {
    // Lasso polygon in original-image pixel coords.
    const payload = {
      points: region.points.map(
        ([x, y]) => [Math.round(x), Math.round(y)] as [number, number],
      ),
    };
    fd.append("selected_region", JSON.stringify(payload));
  }
}

export async function inspectWrinkle(
  file: File,
  garmentType: GarmentType,
  region: RegionPolygon | null,
  options?: Partial<InspectOptions>,
): Promise<InspectResponse> {
  const fd = new FormData();
  fd.append("image", file);
  fd.append("garment_type", garmentType);
  appendRegion(fd, region);
  if (options) {
    if (options.use_segmentation !== undefined)
      fd.append("use_segmentation", String(options.use_segmentation));
    if (options.use_pose_advanced !== undefined)
      fd.append("use_pose_advanced", String(options.use_pose_advanced));
    if (options.use_illustration_model !== undefined)
      fd.append("use_illustration_model", String(options.use_illustration_model));
    if (options.return_debug_overlays !== undefined)
      fd.append("return_debug_overlays", String(options.return_debug_overlays));
  }

  const res = await fetch(`${API_BASE}/api/inspect-wrinkle`, {
    method: "POST",
    body: fd,
  });
  if (!res.ok) throw new Error(await parseError(res));
  return (await res.json()) as InspectResponse;
}

export async function inspectHand(
  file: File,
  region: RegionPolygon | null,
): Promise<InspectResponse> {
  const fd = new FormData();
  fd.append("image", file);
  appendRegion(fd, region); // lasso doubles as the hand-detection hint
  // Landmark overlays are tiny (21 points/hand) — always request them so the
  // skeleton can be visually compared against the drawn hand.
  fd.append("return_debug_overlays", "true");
  const res = await fetch(`${API_BASE}/api/inspect-hand`, {
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

export async function saveReview(payload: ReviewRequest): Promise<ReviewResponse> {
  const res = await fetch(`${API_BASE}/api/review`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error(await parseError(res));
  return (await res.json()) as ReviewResponse;
}

export async function getModelStatus(): Promise<ModelStatus> {
  const res = await fetch(`${API_BASE}/api/model/status`);
  if (!res.ok) throw new Error(await parseError(res));
  return (await res.json()) as ModelStatus;
}
