import logging
import re

from langchain_chroma import Chroma

from app.config import PREFIJOS_GRUPO, settings
from app.services import reranker
from app.services.embeddings import get_embeddings

logger = logging.getLogger(__name__)

_PATRON_GRUPO = re.compile(r"\b(?:IDGS|grupo\s*)?(\d{3,4})\b", re.IGNORECASE)

# "901" solo es ambiguo entre grupos de distintas carreras (IDGS901, LGA901...):
# si la pregunta trae el prefijo, hay que usarlo completo como filtro.
_PATRON_GRUPO_CON_PREFIJO = re.compile(
    r"\b(" + "|".join(PREFIJOS_GRUPO) + r")\s?(\d{3,4})([a-z]{0,6})\b",
    re.IGNORECASE,
)


def _detectar_grupo(pregunta: str) -> str | None:
    match = _PATRON_GRUPO_CON_PREFIJO.search(pregunta)
    if match:
        prefijo, digitos, sufijo = match.groups()
        return f"{prefijo.upper()}{digitos}{sufijo.upper()}"
    match = _PATRON_GRUPO.search(pregunta)
    return match.group(1) if match else None


def _filtro_grupo(pregunta: str) -> dict | None:
    """
    Si la pregunta trae prefijo (LGA901...), el código es inequívoco: igualdad
    exacta. Si solo trae dígitos sueltos ("grupo 901"), son ambiguos entre
    carreras (IDGS901, LGA901, LGDT901...), así que se filtra por cualquier
    variante conocida con esos dígitos en vez de perder el filtro por completo.
    """
    grupo = _detectar_grupo(pregunta)
    if not grupo:
        return None
    if grupo.isdigit():
        candidatos = [grupo] + [f"{p}{grupo}" for p in PREFIJOS_GRUPO]
        return {"grupo": {"$in": candidatos}}
    return {"grupo": grupo}


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
        logger.info("Base de conocimiento cargada: %d chunks disponibles.", vs._collection.count())
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

    filtro = _filtro_grupo(pregunta)
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
