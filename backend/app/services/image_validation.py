import warnings
from io import BytesIO

from PIL import Image, UnidentifiedImageError
from starlette.requests import ClientDisconnect, Request

ALLOWED_IMAGE_TYPES = {
    "image/jpeg": "JPEG",
    "image/png": "PNG",
}


class ImageRequestError(ValueError):
    """Base class for safe image-request failures."""


class ImageTooLargeError(ImageRequestError):
    """Raised when an image exceeds the configured byte or pixel limit."""


class UnsupportedImageTypeError(ImageRequestError):
    """Raised when the declared or detected image type is unsupported."""


class MalformedImageError(ImageRequestError):
    """Raised when submitted bytes are not a complete decodable image."""


def normalized_content_type(value: str | None) -> str:
    return (value or "").partition(";")[0].strip().lower()


async def read_validated_image_body(
    request: Request,
    *,
    max_bytes: int,
    max_pixels: int,
) -> bytes:
    content_type = normalized_content_type(request.headers.get("content-type"))
    expected_format = ALLOWED_IMAGE_TYPES.get(content_type)
    if expected_format is None:
        raise UnsupportedImageTypeError("Submit a JPEG or PNG image.")

    content_length = request.headers.get("content-length")
    if content_length:
        try:
            declared_bytes = int(content_length)
        except ValueError as exc:
            raise ImageRequestError("The Content-Length header is invalid.") from exc
        if declared_bytes < 1:
            raise MalformedImageError("The image body is empty.")
        if declared_bytes > max_bytes:
            raise ImageTooLargeError("The image exceeds the upload size limit.")

    body = bytearray()
    try:
        async for chunk in request.stream():
            if not chunk:
                continue
            if len(body) + len(chunk) > max_bytes:
                raise ImageTooLargeError("The image exceeds the upload size limit.")
            body.extend(chunk)
    except ClientDisconnect as exc:
        raise ImageRequestError("The image upload was interrupted.") from exc

    if not body:
        raise MalformedImageError("The image body is empty.")

    image_bytes = bytes(body)
    validate_image_bytes(image_bytes, expected_format=expected_format, max_pixels=max_pixels)
    return image_bytes


def validate_image_bytes(image_bytes: bytes, *, expected_format: str, max_pixels: int) -> None:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(image_bytes)) as image:
                if image.format != expected_format:
                    raise UnsupportedImageTypeError(
                        "The image contents do not match the declared JPEG or PNG type."
                    )
                width, height = image.size
                if width < 1 or height < 1:
                    raise MalformedImageError("The image dimensions are invalid.")
                if width * height > max_pixels:
                    raise ImageTooLargeError("The image exceeds the pixel limit.")
                image.load()
    except (UnsupportedImageTypeError, ImageTooLargeError, MalformedImageError):
        raise
    except Image.DecompressionBombWarning as exc:
        raise ImageTooLargeError("The image exceeds the pixel limit.") from exc
    except Image.DecompressionBombError as exc:
        raise ImageTooLargeError("The image exceeds the pixel limit.") from exc
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        raise MalformedImageError("The image is malformed or incomplete.") from exc
