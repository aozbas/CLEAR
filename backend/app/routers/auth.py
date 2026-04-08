from fastapi import APIRouter

router = APIRouter()


@router.post("/register")
def register():
    # TODO: proxy to Supabase auth
    return {"todo": "register"}


@router.post("/login")
def login():
    # TODO: proxy to Supabase auth
    return {"todo": "login"}
