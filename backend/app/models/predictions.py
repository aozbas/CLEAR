from typing import Literal

from pydantic import BaseModel, Field, model_validator

PredictionOutcome = Literal[
    "classification_available",
    "classifier_uncertain",
    "poor_image_quality",
    "unsupported_image",
]


class ExperimentalClassificationResponse(BaseModel):
    """One transient research output; never a diagnosis or saved scan record."""

    result_type: Literal["experimental_classification"] = "experimental_classification"
    outcome: PredictionOutcome
    label: str | None = Field(
        description="Model category, present only when an experimental classification is shown."
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

    @model_validator(mode="after")
    def validate_outcome_fields(self) -> "ExperimentalClassificationResponse":
        classification_available = self.outcome == "classification_available"
        if classification_available:
            if self.label is None or self.model_score is None or self.should_retry:
                raise ValueError(
                    "A displayed classification requires a label, score, and no retry."
                )
        elif self.label is not None or self.model_score is not None or not self.should_retry:
            raise ValueError("An abstention must omit the label and score and allow a retry.")
        return self


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"


class ReadinessResponse(BaseModel):
    status: Literal["ready", "not_ready"]
    model_checkpoint_present: bool
