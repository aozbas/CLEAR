from pydantic_settings import BaseSettings, SettingsConfigDict
from hosted_database import create_client, Client


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    hosted_database_url: str
    hosted_database_service_role_key: str
    model_path: str = "../ml/models/lesion_classifier_binary.pt"


settings = Settings()
hosted_database: Client = create_client(settings.hosted_database_url, settings.hosted_database_service_role_key)
