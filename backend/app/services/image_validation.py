from __future__ import annotations

import warnings
from io import BytesIO
from typing import TYPE_CHECKING

from PIL import Image, ImageStat, UnidentifiedImageError

if TYPE_CHECKING:
    from starlette.requests import Request

ALLOWED_IMAGE_TYPES = {
    "image/jpeg": "JPEG",
    "image/png": "PNG",
}
MIN_IMAGE_SIDE = 64
MIN_LUMINANCE_MEAN = 3.0
MAX_LUMINANCE_MEAN = 252.0
MIN_LUMINANCE_STANDARD_DEVIATION = 1.0
QUALITY_THUMBNAIL_SIZE = (64, 64)


class ImageRequestError(ValueError):
    """Base class for safe image-request failures."""


class ImageTooLargeError(ImageRequestError):
    """Raised when an image exceeds the configured byte or pixel limit."""


class UnsupportedImageTypeError(ImageRequestError):
    """Raised when the declared or detected image type is unsupported."""


class MalformedImageError(ImageRequestError):
    """Raised when submitted bytes are not a complete decodable image."""


class PoorImageQualityError(ImageRequestError):
    """Raised when a valid image has too little visible information for the demo."""


def normalized_content_type(value: str | None) -> str:
    return (value or "").partition(";")[0].strip().lower()


async def read_validated_image_body(
    request: Request,
    *,
    max_bytes: int,
    max_pixels: int,
) -> bytes:
    from starlette.requests import ClientDisconnect

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
                validate_image_quality(image)
    except (
        UnsupportedImageTypeError,
        ImageTooLargeError,
        MalformedImageError,
        PoorImageQualityError,
    ):
        raise
    except Image.DecompressionBombWarning as exc:
        raise ImageTooLargeError("The image exceeds the pixel limit.") from exc
    except Image.DecompressionBombError as exc:
        raise ImageTooLargeError("The image exceeds the pixel limit.") from exc
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        raise MalformedImageError("The image is malformed or incomplete.") from exc


def validate_image_quality(image: Image.Image) -> None:
    width, height = image.size
    if min(width, height) < MIN_IMAGE_SIDE:
        raise PoorImageQualityError("The image is too small for this experiment.")

    luminance = image.convert("L")
    luminance.thumbnail(QUALITY_THUMBNAIL_SIZE, Image.Resampling.BILINEAR)
    statistics = ImageStat.Stat(luminance)
    mean = float(statistics.mean[0])
    standard_deviation = float(statistics.stddev[0])
    if (
        mean <= MIN_LUMINANCE_MEAN
        or mean >= MAX_LUMINANCE_MEAN
        or standard_deviation <= MIN_LUMINANCE_STANDARD_DEVIATION
    ):
        raise PoorImageQualityError(
            "The image does not contain enough visible detail for this experiment."
        )
