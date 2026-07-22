from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=BACKEND_DIR / ".env", extra="ignore")

    model_path: str = "ml/models/lesion_classifier_ham10000.pt"
    model_version: str = "ham10000-resnet18-baseline-2026-05-18"
    min_prediction_confidence: float = Field(default=0.60, ge=0.0, le=1.0)
    max_upload_bytes: int = Field(default=8 * 1024 * 1024, ge=1)
    max_image_pixels: int = Field(default=25_000_000, ge=1)
    prediction_timeout_seconds: float = Field(default=25.0, gt=0.0)
    prediction_queue_timeout_seconds: float = Field(default=0.25, gt=0.0)
    max_concurrent_predictions: int = Field(default=1, ge=1, le=8)
    cors_origins: str = "http://localhost:8081,http://localhost:19006"
    allowed_hosts: str = "localhost,127.0.0.1,testserver"

    @field_validator("cors_origins", "allowed_hosts")
    @classmethod
    def reject_wildcards_and_empty_lists(cls, value: str) -> str:
        entries = [entry.strip() for entry in value.split(",") if entry.strip()]
        if not entries:
            raise ValueError("At least one explicit entry is required")
        if any("*" in entry for entry in entries):
            raise ValueError("Wildcard hosts and origins are not allowed")
        return value

    @property
    def resolved_model_path(self) -> Path:
        path = Path(self.model_path)
        if path.is_absolute():
            return path
        if self.model_path.startswith(".."):
            return (BACKEND_DIR / path).resolve()
        return (PROJECT_ROOT / path).resolve()

    @property
    def parsed_cors_origins(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def parsed_allowed_hosts(self) -> list[str]:
        return [host.strip() for host in self.allowed_hosts.split(",") if host.strip()]


settings = Settings()
