import logging
import re

from langchain_chroma import Chroma

from app.config import settings
from app.services import reranker
from app.services.embeddings import get_embeddings

logger = logging.getLogger(__name__)

_PATRON_GRUPO = re.compile(r"\b(?:IDGS|grupo\s*)?(\d{3,4})\b", re.IGNORECASE)


def _detectar_grupo(pregunta: str) -> str | None:
    match = _PATRON_GRUPO.search(pregunta)
    return match.group(1) if match else None


def _load_vectorstore() -> Chroma | None:
    """
    Construye el vectorstore. No exige que el índice ya exista: Chroma con
    persist_directory hace get_or_create de la colección y crea el directorio,
    así que una colección vacía es válida (count 0, búsqueda devuelve []). Esto
    evita el bug de cold-start donde, si al arrancar no había índice, el objeto
    quedaba en None para siempre y las consultas nunca veían lo que se ingestara
    después sin reiniciar el proceso.
    """
    try:
        vs = Chroma(
            collection_name=settings.collection_name,
            embedding_function=get_embeddings(),
            persist_directory=settings.chroma_path,
        )
        logger.info(
            "Base de conocimiento cargada: %d chunks disponibles.", vs._collection.count()
        )
        return vs
    except Exception as e:
        logger.warning("No se pudo cargar la colección: %s", e)
        return None


# Intento eager al importar (single-thread, sin carrera). Si falla, _get_vectorstore
# reintenta perezosamente en cada consulta hasta lograr construirlo y luego cachea.
_vectorstore = _load_vectorstore()


def _get_vectorstore() -> Chroma | None:
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = _load_vectorstore()
    return _vectorstore


def _k_recuperacion() -> int:
    """Cuántos candidatos densos recuperar: un pool amplio si hay reranker, si no el top final."""
    return settings.rerank_candidatos if settings.rerank_enabled else settings.n_resultados


def buscar_contexto(pregunta: str) -> list[str]:
    return [contenido for contenido, _, _ in buscar_contexto_con_metadata(pregunta)]


def buscar_contexto_con_metadata(pregunta: str) -> list[tuple[str, dict, float]]:
    """
    Recupera contexto conservando metadata (fuente/pagina/seccion) y score por chunk.

    Recupera un pool denso amplio y lo reordena con el reranker cross-encoder, que
    afina el ranking y calibra mejor el score. Es la única ruta de recuperación:
    buscar_contexto delega aquí.
    """
    vs = _get_vectorstore()
    if vs is None:
        return []

    grupo = _detectar_grupo(pregunta)
    filtro = {"grupo": grupo} if grupo else None
    k = _k_recuperacion()

    try:
        resultados = vs.similarity_search_with_relevance_scores(pregunta, k=k, filter=filtro)
        if not resultados and filtro:
            resultados = vs.similarity_search_with_relevance_scores(pregunta, k=k)
    except Exception:
        resultados = vs.similarity_search_with_relevance_scores(pregunta, k=k)

    candidatos = [
        (doc.page_content, doc.metadata, max(0.0, min(1.0, score))) for doc, score in resultados
    ]
    return reranker.rerank(pregunta, candidatos)


def collection_info() -> dict:
    vs = _get_vectorstore()
    if vs is None:
        return {"cargada": False, "chunks": 0}
    try:
        return {"cargada": True, "chunks": vs._collection.count()}
    except Exception:
        return {"cargada": False, "chunks": 0}
