from fastapi import APIRouter, UploadFile, File

from ..services.inference import predict_lesion

router = APIRouter()


@router.post("")
async def create_prediction(image: UploadFile = File(...)):
    data = await image.read()
    result = predict_lesion(data)
    # TODO: persist to scans table via Supabase
    return result
