from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routers import predictions, scans

app = FastAPI(title="CLEAR API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8081", "http://localhost:19006"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Auth is handled directly by the mobile app via the Supabase JS client.
# The backend validates the JWT on each request but does not proxy auth calls.
app.include_router(predictions.router, prefix="/predictions", tags=["predictions"])
app.include_router(scans.router, prefix="/scans", tags=["scans"])


@app.get("/health")
def health():
    return {"status": "ok"}
