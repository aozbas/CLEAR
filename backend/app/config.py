from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    supabase_url: str = ""
    supabase_key: str = ""
    model_path: str = "ml/models/lesion_classifier.pt"

    class Config:
        env_file = ".env"


settings = Settings()
