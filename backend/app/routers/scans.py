import logging

from fastapi import APIRouter, Depends, HTTPException

from ..config import supabase
from ..dependencies import get_current_user_id
from ..services.storage import create_signed_image_url

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("")
def list_scans(user_id: str = Depends(get_current_user_id)):
    try:
        response = supabase.table("scans").select(
            "id,image_url,prediction,confidence,created_at"
        ).eq("user_id", user_id).order("created_at", desc=True).execute()
    except Exception as exc:
        logger.exception("Could not fetch scans")
        raise HTTPException(status_code=500, detail="Could not fetch scans.") from exc

    scans = []
    for scan in response.data:
        signed_image_url = None
        try:
            signed_image_url = create_signed_image_url(scan["image_url"])
        except Exception:
            logger.warning("Could not sign scan image URL", exc_info=True)
            # History can still render labels/confidence if signing one image fails.
            pass

        scans.append({
            "id": scan["id"],
            "image_url": scan["image_url"],
            "signed_image_url": signed_image_url,
            "label": scan["prediction"],
            "confidence": scan["confidence"],
            "created_at": scan["created_at"],
        })

    return {"scans": scans}
