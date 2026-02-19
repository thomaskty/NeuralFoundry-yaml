from pydantic_settings import BaseSettings
from pydantic import ConfigDict


class Settings(BaseSettings):
    # API Keys
    OPENAI_API_KEY: str  # Read from system env or .env

    # Database
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    POSTGRES_DB: str
    POSTGRES_HOST: str
    POSTGRES_PORT: int

    # Embedding Configuration
    EMBEDDING_MODEL: str = "text-embedding-3-small"  # NEW: OpenAI embedding model
    EMBEDDING_DIMENSIONS: int = 1536  # NEW: OpenAI embedding dimensions

    # RAG Configuration - Chat History (Hybrid Approach)
    RECENT_MESSAGE_WINDOW: int = 20
    OLDER_MESSAGE_RETRIEVAL: int = 20
    CHAT_HISTORY_THRESHOLD: float = 0.10

    # RAG Configuration - Knowledge Base
    KB_CHUNK_THRESHOLD: float = 0.1
    MAX_KB_CHUNKS_PER_KB: int = 20
    KB_CHUNK_SIZE: int = 800  # NEW: Max characters per chunk
    KB_CHUNK_OVERLAP: int = 100  # NEW: Overlap between chunks

    # Token Limits (approximate word counts)
    MAX_RECENT_MESSAGES_TOKENS: int = 800
    MAX_OLDER_MESSAGES_TOKENS: int = 300
    MAX_KB_CONTEXT_TOKENS: int = 1500

    # LLM Settings
    DEFAULT_LLM_MODEL: str = "gpt-4o-mini"  # CHANGED: OpenAI model
    LLM_TEMPERATURE: float = 0.7
    LLM_MAX_TOKENS: int = 1000

    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8"
    )


settings = Settings()
