from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict
from supabase import create_client, Client

BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=BACKEND_DIR / ".env", extra="ignore")

    supabase_url: str
    supabase_service_role_key: str
    model_path: str = "ml/models/lesion_classifier_binary.pt"

    @property
    def resolved_model_path(self) -> Path:
        path = Path(self.model_path)
        if path.is_absolute():
            return path
        if self.model_path.startswith(".."):
            return (BACKEND_DIR / path).resolve()
        return (PROJECT_ROOT / path).resolve()


settings = Settings()
supabase: Client = create_client(settings.supabase_url, settings.supabase_service_role_key)
