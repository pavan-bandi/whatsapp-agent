from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional
import os


class Settings(BaseSettings):
    # AI Providers
    anthropic_api_key: Optional[str] = Field(None, env="ANTHROPIC_API_KEY")
    openai_api_key: Optional[str] = Field(None, env="OPENAI_API_KEY")

    # Default models
    default_model_provider: str = Field("anthropic", env="DEFAULT_MODEL_PROVIDER")
    default_anthropic_model: str = Field("claude-sonnet-4-20250514", env="DEFAULT_ANTHROPIC_MODEL")
    default_openai_model: str = Field("gpt-4o", env="DEFAULT_OPENAI_MODEL")

    # Weaviate
    weaviate_url: str = Field("http://localhost:8080", env="WEAVIATE_URL")
    weaviate_api_key: Optional[str] = Field(None, env="WEAVIATE_API_KEY")

    # RAG
    chunk_size: int = Field(50, env="CHUNK_SIZE")
    chunk_overlap: int = Field(10, env="CHUNK_OVERLAP")
    max_retrieved_chunks: int = Field(8, env="MAX_RETRIEVED_CHUNKS")
    embedding_model: str = Field("text-embedding-3-small", env="EMBEDDING_MODEL")

    # App
    max_file_size_mb: int = Field(50, env="MAX_FILE_SIZE_MB")
    cors_origins: str = Field("http://localhost:5173,http://localhost:3000", env="CORS_ORIGINS")
    log_level: str = Field("INFO", env="LOG_LEVEL")

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",")]

    @property
    def has_anthropic(self) -> bool:
        return bool(self.anthropic_api_key)

    @property
    def has_openai(self) -> bool:
        return bool(self.openai_api_key)

    @property
    def available_providers(self) -> list[str]:
        providers = []
        if self.has_anthropic:
            providers.append("anthropic")
        if self.has_openai:
            providers.append("openai")
        return providers

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


settings = Settings()
