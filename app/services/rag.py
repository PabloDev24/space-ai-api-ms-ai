import re

from langchain_chroma import Chroma

from app.config import settings
from app.services.embeddings import get_embeddings

_PATRON_GRUPO = re.compile(r"\b(?:IDGS|grupo\s*)?(\d{3,4})\b", re.IGNORECASE)


def _detectar_grupo(pregunta: str) -> str | None:
    match = _PATRON_GRUPO.search(pregunta)
    return match.group(1) if match else None


def _load_vectorstore() -> Chroma | None:
    import os

    if not os.path.isdir(settings.chroma_path):
        print(
            f"[Aviso] No se encontró '{settings.chroma_path}'. "
            "Indexa documentos con scripts/ingest.py primero."
        )
        return None
    try:
        vs = Chroma(
            collection_name=settings.collection_name,
            embedding_function=get_embeddings(),
            persist_directory=settings.chroma_path,
        )
        count = vs._collection.count()
        print(f"Base de conocimiento cargada: {count} chunks disponibles.")
        return vs
    except Exception as e:
        print(f"[Aviso] No se pudo cargar la colección: {e}")
        return None


_vectorstore = _load_vectorstore()


def buscar_contexto(pregunta: str) -> list[str]:
    if _vectorstore is None:
        return []

    grupo = _detectar_grupo(pregunta)
    filtro = {"grupo": grupo} if grupo else None

    try:
        docs = _vectorstore.similarity_search(
            pregunta,
            k=settings.n_resultados,
            filter=filtro,
        )
        # Si el filtro por grupo no devuelve resultados, busca sin filtro
        if not docs and filtro:
            docs = _vectorstore.similarity_search(pregunta, k=settings.n_resultados)
    except Exception:
        docs = _vectorstore.similarity_search(pregunta, k=settings.n_resultados)

    return [doc.page_content for doc in docs]


def collection_info() -> dict:
    if _vectorstore is None:
        return {"cargada": False, "chunks": 0}
    try:
        return {"cargada": True, "chunks": _vectorstore._collection.count()}
    except Exception:
        return {"cargada": False, "chunks": 0}
