from fastapi import APIRouter, UploadFile, File, Depends

from ..dependencies import get_current_user_id
from ..services.inference import predict_lesion

router = APIRouter()


@router.post("")
async def create_prediction(
    image: UploadFile = File(...),
    user_id: str = Depends(get_current_user_id),
):
    data = await image.read()
    result = predict_lesion(data)
    # TODO (Phase 1): upload image, persist scan row for user_id
    return result
