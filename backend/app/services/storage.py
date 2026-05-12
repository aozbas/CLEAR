from uuid import uuid4

from ..config import supabase

BUCKET = "scan-images"


class UnsupportedImageFormatError(ValueError):
    """Raised when uploaded bytes are not a supported scan image type."""


def _detect_image_type(image_bytes: bytes) -> tuple[str, str]:
    """Return (extension, content_type) by inspecting the magic bytes."""
    if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png", "image/png"
    if image_bytes.startswith(b"\xff\xd8\xff"):
        return "jpg", "image/jpeg"
    raise UnsupportedImageFormatError("Unsupported image format (only PNG and JPEG allowed)")


def upload_scan_image(image_bytes: bytes, user_id: str) -> str:
    ext, content_type = _detect_image_type(image_bytes)
    path = f"{user_id}/{uuid4()}.{ext}"
    supabase.storage.from_(BUCKET).upload(
        path, image_bytes, {"content-type": content_type}
    )
    return path


def create_signed_image_url(path: str, expires_in: int = 60 * 60) -> str:
    response = supabase.storage.from_(BUCKET).create_signed_url(path, expires_in)
    if isinstance(response, dict):
        signed_url = (
            response.get("signedURL")
            or response.get("signedUrl")
            or response.get("signed_url")
        )
        if signed_url:
            return signed_url

    signed_url = (
        getattr(response, "signedUrl", None)
        or getattr(response, "signedURL", None)
        or getattr(response, "signed_url", None)
    )
    if not signed_url:
        raise RuntimeError(f"Supabase did not return a signed URL for {path}")
    return signed_url
