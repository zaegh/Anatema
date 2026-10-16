import os

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    DOC_URL: str | None = None
    REDOC_URL: str | None = None
    # POSTGRESQL
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    POSTGRES_DB: str
    POSTGRES_HOST: str
    POSTGRES_PORT: int = Field(default=5432)
    # AUTH
    OPEN_WEBUI_SECRET_KEY: str
    # CALDAV
    CALDAV_BEARER_TOKEN: str
    # HOME ASSISTANT
    HAOS_TOKEN: str
    HOME_ASSISTANT_URL: str
    # MODELS
    OLLAMA_BASE_URL: str
    OLLAMA_KEEP_ALIVE: int = Field(default=900)
    MISTRAL_API_KEY: str
    # SUBAGENT FLAG
    USE_CALDAV_SUBAGENT: bool = True
    USE_EMAIL_READER_SUBAGENT: bool = False
    USE_EMAIL_SENDER_SUBAGENT: bool = False
    USE_KNOWLEDGE_WRITER_SUBAGENT: bool = False
    USE_KNOWLEDGE_READER_SUBAGENT: bool = False
    USE_SEARCH_SUBAGENT: bool = False
    USE_WEATHER_SUBAGENT: bool = False
    USE_HAOS_SUBAGENT: bool = True

    model_config = SettingsConfigDict(
        env_file=".env" if os.path.exists(".env") else None,
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
