from pydantic_settings import BaseSettings, SettingsConfigDict
from supabase import create_client, Client


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    supabase_url: str
    supabase_service_role_key: str
    model_path: str = "../ml/models/lesion_classifier_binary.pt"


settings = Settings()
supabase: Client = create_client(settings.supabase_url, settings.supabase_service_role_key)
