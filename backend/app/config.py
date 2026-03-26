from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    DATABASE_URL: str
    DATABASE_URL_SYNC: str
    MISTRAL_API_KEY: str = ""

    model_config = {"env_file": ".env"}


settings = Settings()
