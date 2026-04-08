from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    hosted_database_url: str = ""
    hosted_database_key: str = ""
    model_path: str = "ml/models/lesion_classifier.pt"

    class Config:
        env_file = ".env"


settings = Settings()
