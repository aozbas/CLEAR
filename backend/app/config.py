from pydantic_settings import BaseSettings
from hosted_database import create_client, Client


class Settings(BaseSettings):
    hosted_database_url: str = ""
    hosted_database_key: str = ""
    model_path: str = "ml/models/lesion_classifier_binary.pt"

    class Config:
        env_file = ".env"


settings = Settings()
hosted_database: Client = create_client(settings.hosted_database_url, settings.hosted_database_key)
