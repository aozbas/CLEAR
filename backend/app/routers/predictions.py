import logging
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from ..config import settings, supabase
from ..dependencies import get_current_user_id
from ..services.inference import InvalidImageError, predict_lesion
from ..services.storage import UnsupportedImageFormatError

router = APIRouter()
logger = logging.getLogger(__name__)


def prediction_response(
    result: dict,
    *,
    image_url: str | None,
    signed_image_url: str | None,
    scan_id: str | None,
    saved: bool,
    should_retry: bool,
    message: str | None,
) -> dict:
    return {
        "label": result["label"],
        "confidence": result["confidence"],
        "image_url": image_url,
        "signed_image_url": signed_image_url,
        "scan_id": scan_id,
        "saved": saved,
        "should_retry": should_retry,
        "message": message,
        "model_version": settings.model_version,
    }


def low_confidence_response(result: dict) -> dict:
    return prediction_response(
        result,
        image_url=None,
        signed_image_url=None,
        scan_id=None,
        saved=False,
        should_retry=True,
        message="Image unclear - try again.",
    )


@router.post("/demo")
async def create_demo_prediction(image: Annotated[UploadFile, File()]):
    data = await image.read()
    try:
        result = predict_lesion(data)
        if result["confidence"] < settings.min_prediction_confidence:
            return low_confidence_response(result)

        return prediction_response(
            result,
            image_url=None,
            signed_image_url=None,
            scan_id=None,
            saved=False,
            should_retry=False,
            message="Demo result only. No photo or result was saved.",
        )
    except (InvalidImageError, UnsupportedImageFormatError) as exc:
        raise HTTPException(status_code=415, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        logger.exception("Prediction model checkpoint is missing")
        raise HTTPException(
            status_code=503, detail="Classification model is not available."
        ) from exc
    except Exception as exc:
        logger.exception("Demo classification request failed")
        raise HTTPException(
            status_code=500, detail="Classification could not be completed."
        ) from exc


@router.post("")
async def create_prediction(
    image: Annotated[UploadFile, File()],
    user_id: Annotated[str, Depends(get_current_user_id)],
):
    data = await image.read()
    try:
        result = predict_lesion(data)
        if result["confidence"] < settings.min_prediction_confidence:
            return low_confidence_response(result)

        insert_response = (
            supabase.table("scans")
            .insert(
                {
                    "user_id": user_id,
                    "image_url": None,
                    "prediction": result["label"],
                    "confidence": result["confidence"],
                    "model_version": settings.model_version,
                }
            )
            .execute()
        )
        scan = insert_response.data[0]
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

    return prediction_response(
        result,
        image_url=None,
        signed_image_url=None,
        scan_id=scan["id"],
        saved=True,
        should_retry=False,
        message="Saved to history. Photo was not saved.",
    )
