import os
import chromadb
from app.config import settings


def _load_collection():
    if not os.path.isdir(settings.chroma_path):
        print(f"[Aviso] No se encontró '{settings.chroma_path}'. El servicio funcionará "
              "sin contexto de documentos hasta que se indexen PDFs.")
        return None

    client = chromadb.PersistentClient(path=settings.chroma_path)
    try:
        col = client.get_collection(name=settings.collection_name)
        print(f"Base de conocimiento cargada: {col.count()} chunks disponibles.")
        return col
    except Exception:
        print(f"[Aviso] La colección '{settings.collection_name}' no existe todavía.")
        return None


_collection = _load_collection()


def buscar_contexto(pregunta: str) -> list[str]:
    if _collection is None:
        return []
    resultados = _collection.query(
        query_texts=[pregunta],
        n_results=settings.n_resultados,
    )
    return resultados.get("documents", [[]])[0]


def collection_info() -> dict:
    return {
        "cargada": _collection is not None,
        "chunks": _collection.count() if _collection is not None else 0,
    }
