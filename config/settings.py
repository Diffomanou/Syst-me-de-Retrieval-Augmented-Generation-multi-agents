"""
config/settings.py
──────────────────
Configuration centralisée — version 100% GRATUITE
  LLM      : Groq (gratuit — https://console.groq.com)
  Embeddings: HuggingFace sentence-transformers (local, gratuit)
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Groq (LLM gratuit)
    groq_api_key: str

    # Neo4j
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password123"

    # Application
    app_env: str = "development"
    app_port: int = 8000
    log_level: str = "INFO"

    # Modèle LLM Groq (gratuit)
    llm_model: str = "llama-3.1-8b-instant"

    # Modèle d'embeddings HuggingFace (local, gratuit)
    # paraphrase-multilingual = supporte le français
    embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2"

    # ChromaDB
    chroma_persist_dir: str = "./chroma_db"
    chroma_collection_name: str = "documents"


settings = Settings()
