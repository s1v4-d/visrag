from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv
from pydantic_settings import BaseSettings
from pydantic import Field

# Load secrets from env/secrets/secrets.env if present (non-fatal if missing)
load_dotenv(dotenv_path=Path("env/secrets/secrets.env"), override=False)

class Settings(BaseSettings):
    model_config = {
        "env_prefix": "",
        "case_sensitive": False,
    }

    # service
    service_port: int = Field(default=int(os.environ.get("SERVICE_PORT", 8080)))

    # OpenAI
    openai_api_key: str | None = os.environ.get("OPENAI_API_KEY")
    openai_vision_model: str = os.environ.get("OPENAI_VISION_MODEL", "gpt-4o-mini")
    openai_text_model: str = os.environ.get("OPENAI_TEXT_MODEL", "gpt-4o-mini")
    embedding_model: str = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")

    # Qdrant
    qdrant_url: str = os.environ.get("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key: str | None = os.environ.get("QDRANT_API_KEY")
    qdrant_collection: str = os.environ.get("QDRANT_COLLECTION", "vlmrag")

    # Embedding dimensions
    text_dim: int = int(os.environ.get("TEXT_EMBED_DIM", 1536))
    image_dim: int = int(os.environ.get("IMAGE_EMBED_DIM", 512))  # ViT-B/32 default

    # Files
    data_dir: str = os.environ.get("DATA_DIR", "/app/data")


settings = Settings()