import logging
from collections.abc import Mapping
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Request, status

from ..config import settings
from ..models.predictions import ExperimentalClassificationResponse, PredictionOutcome
from ..services.image_validation import (
    ImageRequestError,
    ImageTooLargeError,
    MalformedImageError,
    PoorImageQualityError,
    UnsupportedImageTypeError,
    read_validated_image_body,
)
from ..services.prediction_runtime import (
    PredictionBusyError,
    PredictionCallable,
    PredictionCapacity,
    PredictionTimeoutError,
)

router = APIRouter()
logger = logging.getLogger(__name__)
prediction_capacity = PredictionCapacity(settings.max_concurrent_predictions)

RAW_IMAGE_REQUEST = {
    "requestBody": {
        "required": True,
        "content": {
            "image/jpeg": {"schema": {"type": "string", "format": "binary"}},
            "image/png": {"schema": {"type": "string", "format": "binary"}},
        },
    }
}


def get_predictor() -> PredictionCallable:
    # Keep ML imports out of application startup and API-only test collection. The
    # production adapter is imported only when a real prediction request reaches it.
    from ..services.inference import predict_lesion

    return predict_lesion


def _validated_prediction(result: Mapping[str, Any]) -> tuple[str, float]:
    label = result.get("label")
    score = result.get("confidence")
    if not isinstance(label, str) or not label:
        raise RuntimeError("Predictor returned an invalid label")
    if not isinstance(score, (int, float)) or isinstance(score, bool):
        raise RuntimeError("Predictor returned an invalid score")
    numeric_score = float(score)
    if not 0.0 <= numeric_score <= 1.0:
        raise RuntimeError("Predictor returned an out-of-range score")
    return label, numeric_score


def _abstention(
    outcome: PredictionOutcome,
    message: str,
) -> ExperimentalClassificationResponse:
    return ExperimentalClassificationResponse(
        outcome=outcome,
        label=None,
        model_score=None,
        should_retry=True,
        message=message,
        model_version=settings.model_version,
    )


@router.post(
    "/demo",
    response_model=ExperimentalClassificationResponse,
    summary="Return one transient experimental classification",
    description=(
        "Accepts one raw JPEG or PNG request body. CLEAR processes it transiently and returns "
        "one experimental classification without storing the image or result. This is not a "
        "diagnosis."
    ),
    openapi_extra=RAW_IMAGE_REQUEST,
)
async def create_demo_prediction(
    request: Request,
    predictor: Annotated[PredictionCallable, Depends(get_predictor)],
) -> ExperimentalClassificationResponse:
    try:
        image_bytes = await read_validated_image_body(
            request,
            max_bytes=settings.max_upload_bytes,
            max_pixels=settings.max_image_pixels,
        )
        raw_result = await prediction_capacity.run(
            predictor,
            image_bytes,
            queue_timeout_seconds=settings.prediction_queue_timeout_seconds,
            prediction_timeout_seconds=settings.prediction_timeout_seconds,
        )
        label, score = _validated_prediction(raw_result)
    except PoorImageQualityError:
        return _abstention(
            "poor_image_quality",
            (
                "No result is shown because the image does not contain enough visible detail. "
                "Try a clear, close-up photo of one visible skin spot."
            ),
        )
    except ImageTooLargeError as exc:
        raise HTTPException(status_code=status.HTTP_413_CONTENT_TOO_LARGE, detail=str(exc)) from exc
    except (UnsupportedImageTypeError, MalformedImageError) as exc:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail=str(exc)
        ) from exc
    except ImageRequestError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except PredictionBusyError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The experimental classifier is busy. Try again shortly.",
            headers={"Retry-After": "2"},
        ) from exc
    except PredictionTimeoutError as exc:
        logger.warning("Experimental classification timed out")
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="The experimental classification timed out. Try again.",
        ) from exc
    except FileNotFoundError as exc:
        logger.error("Prediction model checkpoint is missing")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The experimental classifier is unavailable.",
        ) from exc
    except Exception as exc:
        logger.error("Experimental classification request failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The experimental classification could not be completed.",
        ) from exc

    if score < settings.min_prediction_confidence:
        return _abstention(
            "classifier_uncertain",
            (
                "No result is shown because the experimental classifier was uncertain. "
                "Try a clear, close-up photo of one visible skin spot."
            ),
        )

    return ExperimentalClassificationResponse(
        outcome="classification_available",
        label=label,
        model_score=score,
        should_retry=False,
        message="Experimental result only. CLEAR does not save the image or result.",
        model_version=settings.model_version,
    )
