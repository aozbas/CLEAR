from typing import Literal

from pydantic import BaseModel, Field


class ExperimentalClassificationResponse(BaseModel):
    """One transient research output; never a diagnosis or saved scan record."""

    result_type: Literal["experimental_classification"] = "experimental_classification"
    label: str | None = Field(
        description="Model category, omitted when the display threshold is not met."
    )
    model_score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Uncalibrated top-class model score; not a probability of disease.",
    )
    should_retry: bool
    message: str
    model_version: str


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"


class ReadinessResponse(BaseModel):
    status: Literal["ready", "not_ready"]
    model_checkpoint_present: bool
