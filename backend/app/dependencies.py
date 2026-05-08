from fastapi import Header, HTTPException

from .config import supabase


async def get_current_user_id(authorization: str | None = Header(default=None)) -> str:
    # Use Header(default=None) so a missing header yields 401, not FastAPI's
    # default 422 for required-header validation errors.
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid authorization header")
    token = authorization.removeprefix("Bearer ")
    try:
        response = supabase.auth.get_user(token)
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    if response.user is None:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    return response.user.id
