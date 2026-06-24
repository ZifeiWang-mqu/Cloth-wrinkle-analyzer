"""Safe upload handling and image decoding.

Security notes (requirement 16.8): uploaded filenames are never trusted. We
store files under a generated UUID name, keep only a whitelisted extension, and
enforce a max byte size. Decoding is done in-memory with OpenCV.
"""

from __future__ import annotations

import base64
import binascii
import uuid
from pathlib import Path

import cv2
import numpy as np

from app.settings import Settings


class UploadError(Exception):
    """Raised for invalid uploads (mapped to HTTP 400 by the route)."""


def _safe_extension(filename: str | None, allowed: tuple[str, ...]) -> str:
    ext = Path(filename or "").suffix.lower()
    if ext == ".jpeg":
        ext = ".jpg"
    if ext not in allowed:
        raise UploadError(
            f"対応していない拡張子です: '{ext or '(なし)'}'. 対応: {', '.join(allowed)}"
        )
    return ext


def decode_image(raw: bytes) -> np.ndarray:
    """Decode bytes into a BGR ndarray. Raises UploadError on failure."""
    if not raw:
        raise UploadError("空のファイルです。")
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise UploadError("画像をデコードできませんでした。jpg/png/webp を確認してください。")
    return img


def save_upload(
    raw: bytes, original_filename: str | None, settings: Settings
) -> tuple[Path, str, np.ndarray]:
    """Validate, decode, and persist an upload.

    Returns ``(saved_path, safe_display_name, image_bgr)``.
    """
    if len(raw) > settings.max_upload_bytes:
        raise UploadError(
            f"ファイルが大きすぎます（上限 {settings.max_upload_bytes // (1024 * 1024)}MB）。"
        )
    ext = _safe_extension(original_filename, settings.allowed_extensions)
    image = decode_image(raw)  # decode first so we never store junk

    settings.ensure_dirs()
    safe_name = f"{uuid.uuid4().hex}{ext}"
    saved_path = settings.upload_dir / safe_name
    saved_path.write_bytes(raw)
    return saved_path, safe_name, image


def decode_base64(data: str) -> tuple[bytes, str]:
    """Decode a base64 string (raw or data URL) into ``(bytes, extension)``."""
    if not data or not data.strip():
        raise UploadError("image_base64 が空です。")
    s = data.strip()
    ext = ".png"
    if s.startswith("data:"):
        header, _, b64 = s.partition(",")
        low = header.lower()
        if "image/jpeg" in low or "image/jpg" in low:
            ext = ".jpg"
        elif "image/webp" in low:
            ext = ".webp"
        elif "image/png" in low:
            ext = ".png"
        s = b64
    try:
        raw = base64.b64decode(s, validate=False)
    except (binascii.Error, ValueError) as exc:
        raise UploadError(f"base64 をデコードできませんでした: {exc}")
    if not raw:
        raise UploadError("base64 のデコード結果が空です。")
    return raw, ext


def persist_bytes(raw: bytes, settings: Settings, ext: str = ".png") -> Path:
    """Persist already-decoded image bytes under a UUID name. Validates size."""
    if len(raw) > settings.max_upload_bytes:
        raise UploadError(
            f"ファイルが大きすぎます（上限 {settings.max_upload_bytes // (1024 * 1024)}MB）。"
        )
    if ext not in settings.allowed_extensions:
        ext = ".png"
    settings.ensure_dirs()
    safe_name = f"{uuid.uuid4().hex}{ext}"
    saved_path = settings.upload_dir / safe_name
    saved_path.write_bytes(raw)
    return saved_path
