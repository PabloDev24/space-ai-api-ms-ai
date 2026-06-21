from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    gemini_api_key: str
    chroma_path: str = "chroma_db"
    collection_name: str = "documentos"
    n_resultados: int = 5
    modelo_gemini: str = "gemini-2.5-flash"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
