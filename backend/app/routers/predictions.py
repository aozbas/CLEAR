import logging
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from ..config import hosted_database
from ..dependencies import get_current_user_id
from ..services.inference import InvalidImageError, predict_lesion
from ..services.storage import (
    UnsupportedImageFormatError,
    create_signed_image_url,
    upload_scan_image,
)

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("")
async def create_prediction(
    image: Annotated[UploadFile, File()],
    user_id: Annotated[str, Depends(get_current_user_id)],
):
    data = await image.read()
    try:
        result = predict_lesion(data)
        image_path = upload_scan_image(data, user_id)
        insert_response = (
            hosted_database.table("scans")
            .insert(
                {
                    "user_id": user_id,
                    "image_url": image_path,
                    "prediction": result["label"],
                    "confidence": result["confidence"],
                }
            )
            .execute()
        )
        scan = insert_response.data[0]
        signed_image_url = create_signed_image_url(image_path)
    except (InvalidImageError, UnsupportedImageFormatError) as exc:
        raise HTTPException(status_code=415, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        logger.exception("Prediction model checkpoint is missing")
        raise HTTPException(
            status_code=503, detail="Classification model is not available."
        ) from exc
    except Exception as exc:
        logger.exception("Classification request failed")
        raise HTTPException(
            status_code=500, detail="Classification could not be completed."
        ) from exc

    return {
        "label": result["label"],
        "confidence": result["confidence"],
        "image_url": image_path,
        "signed_image_url": signed_image_url,
        "scan_id": scan["id"],
    }
