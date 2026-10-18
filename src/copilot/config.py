from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="COPILOT_", env_file=".env", extra="ignore")

    database_url: str = "postgresql://copilot:copilot@localhost:5433/copilot"
    openai_api_key: str = ""
    github_token: str = ""
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536
    budget_usd: float = 8.0
    demo_repo: str = ""
    since_months: int = 12


@lru_cache
def get_settings() -> Settings:
    return Settings()
