from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # API Keys
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    
    # Upstash Redis
    upstash_redis_url: str = ""
    upstash_redis_token: str = ""
    
    # App settings
    app_name: str = "RAG API"
    debug: bool = False
    
    class Config:
        env_file = ".env"


@lru_cache()
def get_settings() -> Settings:
    """Cached settings instance."""
    return Settings()
