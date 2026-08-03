from pydantic_settings import BaseSettings, SettingsConfigDict

# Prefijos de código de grupo (carrera) usados en los horarios. Un número
# suelto como "905" es ambiguo entre grupos de distintas carreras (IMT905,
# ISP905, IDGS905...), así que tanto la ingesta (para leer el código completo
# del nombre de archivo) como la búsqueda (para detectar el grupo en la
# pregunta) deben usar el código con prefijo, no solo los dígitos.
PREFIJOS_GRUPO = (
    "AUT",
    "GAST",
    "IDGS",
    "IEVN",
    "IMT",
    "ISMF",
    "ISP",
    "LDGR",
    "LGA",
    "LGCH",
    "LGDT",
    "LIMI",
    "LNM",
    "LTM",
    "LTU",
    "MOP",
)


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

    # Timeout duro para la llamada al LLM. Sin esto, un proveedor que se cuelga
    # (red lenta, rate limit sin respuesta de error) deja la petición colgada
    # indefinidamente y, al ser un solo worker, tumba /ask y /health para todos
    # los demás usuarios hasta reiniciar el proceso.
    llm_timeout_seconds: float = 30.0

    # Embeddings: local | openai | gemini
    embedding_provider: str = "local"
    embedding_model: str = ""  # usa el default del proveedor si está vacío

    # Logging
    log_level: str = "INFO"

    # RAG
    chroma_path: str = "chroma_db"
    collection_name: str = "documentos"
    n_resultados: int = 8

    # Archivo del PDF original en Azure Blob (trazabilidad / re-index / descarga).
    # Si blob_connection_string está vacío, el archivado se omite (no rompe la ingesta).
    blob_connection_string: str = ""
    blob_container: str = "documents"

    # Auth del endpoint /ingest. Si está vacío, el endpoint queda abierto (solo dev/demo);
    # si tiene valor, se exige la cabecera X-API-Key con ese valor.
    ingest_api_key: str = ""

    # Reranker cross-encoder (reordena los candidatos densos; mejora ranking y calibración)
    rerank_enabled: bool = True
    rerank_model: str = "jinaai/jina-reranker-v2-base-multilingual"
    rerank_candidatos: int = 20  # pool denso a recuperar antes de rerankear
    rerank_top_n: int = 5  # cuántos quedan tras rerankear

    # Docling PDF parsing
    docling_enable_ocr: bool = True
    docling_enable_tables: bool = True
    docling_ocr_langs: str = "es,en"
    docling_ocr_use_gpu: bool = False
    docling_num_threads: int = 4
    docling_chunk_size: int = 512

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
