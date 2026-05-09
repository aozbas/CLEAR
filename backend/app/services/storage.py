from uuid import uuid4

from ..config import supabase

BUCKET = "scan-images"


def _detect_image_type(image_bytes: bytes) -> tuple[str, str]:
    """Return (extension, content_type) by inspecting the magic bytes."""
    if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png", "image/png"
    if image_bytes.startswith(b"\xff\xd8\xff"):
        return "jpg", "image/jpeg"
    raise ValueError("Unsupported image format (only PNG and JPEG allowed)")


def upload_scan_image(image_bytes: bytes, user_id: str) -> str:
    ext, content_type = _detect_image_type(image_bytes)
    path = f"{user_id}/{uuid4()}.{ext}"
    supabase.storage.from_(BUCKET).upload(
        path, image_bytes, {"content-type": content_type}
    )
    return path
