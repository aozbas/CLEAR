from pydantic_settings import BaseSettings
from supabase import create_client, Client


class Settings(BaseSettings):
    supabase_url: str = ""
    supabase_key: str = ""
    model_path: str = "ml/models/lesion_classifier_binary.pt"

    class Config:
        env_file = ".env"


settings = Settings()
supabase: Client = create_client(settings.supabase_url, settings.supabase_key)
