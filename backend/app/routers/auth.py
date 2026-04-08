from fastapi import APIRouter

router = APIRouter()


@router.post("/register")
def register():
    # TODO: proxy to Hosted database auth
    return {"todo": "register"}


@router.post("/login")
def login():
    # TODO: proxy to Hosted database auth
    return {"todo": "login"}
