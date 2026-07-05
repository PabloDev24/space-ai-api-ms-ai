from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Proveedor activo: gemini | groq | openai
    proveedor: str = "gemini"

    # API Keys (solo la del proveedor activo es requerida)
    gemini_api_key: str = ""
    groq_api_key: str = ""
    openai_api_key: str = ""

    # Modelo (si no se especifica se usa el default del proveedor)
    modelo: str = ""
    modelo_gemini: str = "gemini-2.5-flash"  # compatibilidad con .env existente

    # Embeddings: local | openai | gemini
    embedding_provider: str = "local"
    embedding_model: str = ""  # usa el default del proveedor si está vacío

    # Logging
    log_level: str = "INFO"

    # RAG
    chroma_path: str = "chroma_db"
    collection_name: str = "documentos"
    n_resultados: int = 8

    # Docling PDF parsing
    docling_enable_ocr: bool = True
    docling_enable_tables: bool = True
    docling_ocr_langs: str = "es,en"
    docling_ocr_use_gpu: bool = False
    docling_num_threads: int = 4
    docling_chunk_size: int = 512

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
