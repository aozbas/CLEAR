from fastapi import APIRouter, Depends

from ..dependencies import get_current_user_id

router = APIRouter()


@router.get("")
def list_scans(user_id: str = Depends(get_current_user_id)):
    # TODO (Phase 1): fetch scans for user_id from Supabase
    return {"scans": []}
