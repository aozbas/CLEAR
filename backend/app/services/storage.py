from uuid import uuid4

from ..config import hosted_database

BUCKET = "scan-images"


def upload_scan_image(image_bytes: bytes, user_id: str) -> str:
    path = f"{user_id}/{uuid4()}.jpg"
    hosted_database.storage.from_(BUCKET).upload(
        path, image_bytes, {"content-type": "image/jpeg"}
    )
    return path
