from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    DATABASE_URL: str = "sqlite+aiosqlite:////data/barter.db"
    MISTRAL_API_KEY: str = ""
    SEMANTIC_URL: str = "http://localhost:8002"
    AUDIO_URL: str = "http://localhost:8001"
    WARNING_URL: str = "http://localhost:8003"
    RESOURCE_URL: str = "http://resource_agent:8004"

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()
