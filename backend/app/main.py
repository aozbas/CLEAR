from fastapi import FastAPI

from .routers import auth, predictions, scans

app = FastAPI(title="Skin Lesion API")

app.include_router(auth.router, prefix="/auth", tags=["auth"])
app.include_router(predictions.router, prefix="/predictions", tags=["predictions"])
app.include_router(scans.router, prefix="/scans", tags=["scans"])


@app.get("/health")
def health():
    return {"status": "ok"}
