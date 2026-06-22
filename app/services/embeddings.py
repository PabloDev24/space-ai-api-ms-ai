from langchain_core.embeddings import Embeddings

from app.config import settings

_MODELOS_DEFAULT = {
    "local": "BAAI/bge-small-en-v1.5",  # ONNX, sin PyTorch, multilingüe
    "openai": "text-embedding-3-small",
    "gemini": "models/text-embedding-004",
}


def get_embeddings() -> Embeddings:
    """
    Retorna el cliente de embeddings configurado según EMBEDDING_PROVIDER.
    El mismo cliente debe usarse tanto en el ingest como en las consultas
    para que los vectores sean comparables.
    """
    provider = settings.embedding_provider.lower()
    model = settings.embedding_model or _MODELOS_DEFAULT.get(provider, "")

    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model=model, api_key=settings.openai_api_key)

    if provider == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        return GoogleGenerativeAIEmbeddings(
            model=model,
            google_api_key=settings.gemini_api_key,
        )

    # local (default): fastembed — ONNX, sin PyTorch, ~30MB por modelo
    from langchain_community.embeddings import FastEmbedEmbeddings

    return FastEmbedEmbeddings(model_name=model)
