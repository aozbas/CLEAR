from fastapi import APIRouter

router = APIRouter()


@router.get("")
def list_scans():
    # TODO: fetch scan history for current user
    return {"scans": []}
