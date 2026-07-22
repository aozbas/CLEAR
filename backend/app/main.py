from fastapi import FastAPI, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware

from .config import settings
from .models.predictions import HealthResponse, ReadinessResponse
from .routers import predictions

app = FastAPI(
    title="CLEAR Experimental Classification API",
    description=(
        "A privacy-first, stateless educational demo. It accepts one image and returns one "
        "experimental classification. It is not a medical device and does not provide diagnoses."
    ),
    version="0.1.0",
)

app.add_middleware(TrustedHostMiddleware, allowed_hosts=settings.parsed_allowed_hosts)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.parsed_cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)


@app.middleware("http")
async def privacy_headers(request, call_next):
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store"
    response.headers["Pragma"] = "no-cache"
    response.headers["Strict-Transport-Security"] = "max-age=31536000"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    return response


app.include_router(predictions.router, prefix="/predictions", tags=["experimental classification"])


@app.get("/health", response_model=HealthResponse, summary="Liveness check")
def health() -> HealthResponse:
    return HealthResponse()


@app.get(
    "/ready",
    response_model=ReadinessResponse,
    summary="Static checkpoint readiness check",
    description="Checks only that the configured checkpoint path exists; it does not load a model.",
)
def readiness(response: Response) -> ReadinessResponse:
    checkpoint_present = settings.resolved_model_path.is_file()
    if not checkpoint_present:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return ReadinessResponse(
        status="ready" if checkpoint_present else "not_ready",
        model_checkpoint_present=checkpoint_present,
    )
