"""
Central settings, loaded from environment variables (.env locally).

TRD mapping: this replaces AWS Secrets Manager for local/demo purposes.
In production, DATABASE_URL would point at the managed Postgres instance
and secrets would be injected by the hosting platform instead of a .env file.
"""
import os
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BACKEND_DIR / ".env"), extra="ignore", protected_namespaces=()
    )

    database_url: str = f"sqlite:///{BACKEND_DIR / 'cartiq.db'}"
    cors_origins: str = "http://localhost:5500,http://127.0.0.1:5500"
    model_path: str = str(BACKEND_DIR / "ml" / "model.json")
    model_meta_path: str = str(BACKEND_DIR / "ml" / "model_meta.json")

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


settings = Settings()
