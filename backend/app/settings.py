from pydantic_settings import BaseSettings
from typing import Literal

class Settings(BaseSettings):
    VECTOR_DB: Literal["QDRANT"] = "QDRANT"
    OLLAMA_HOST: str = "http://host.docker.internal:11434"
    MODEL_NAME: str = "gpt-oss:20b"
    QDRANT_HOST: str = "qdrant"
    QDRANT_PORT: int = 6333
    CACHE_ENABLED: bool = True
    CACHE_TTL: int = 300             # saniye
    CACHE_MAX_ITEMS: int = 1024
    
    class Config:
        env_file = ".env"

settings = Settings()
assert settings.VECTOR_DB == "QDRANT", "VECTOR_DB yalnızca 'QDRANT' olabilir."
