from fastapi import Header, HTTPException

from .config import supabase


async def get_current_user_id(authorization: str = Header(...)) -> str:
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header")
    token = authorization.removeprefix("Bearer ")
    response = supabase.auth.get_user(token)
    if response.user is None:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    return response.user.id
