"""
Reranker cross-encoder para la fase de recuperación del RAG.

La búsqueda densa recupera un pool amplio de candidatos barato pero con ranking
grueso. El cross-encoder mira pregunta y documento juntos y reordena con mucha más
precisión, quedándose con los mejores. Su score, además, calibra mejor la confianza
que la distancia densa.

Espejo del patrón de embeddings.py: factory perezoso gated por settings, tolerante
a fallos para no romper la demo (si el rerank falla, se devuelve el orden denso).
"""

import logging
import math

from app.config import settings

logger = logging.getLogger(__name__)

# Tipo de un candidato recuperado: (contenido, metadata, score).
Candidato = tuple[str, dict, float]

_encoder = None


def _sigmoide(x: float) -> float:
    """Mapea el score crudo del cross-encoder (logit) a (0, 1)."""
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0


def _get_encoder():
    """Carga perezosa del cross-encoder; se reutiliza entre consultas."""
    global _encoder
    if _encoder is None:
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        _encoder = TextCrossEncoder(model_name=settings.rerank_model)
    return _encoder


def rerank(pregunta: str, candidatos: list[Candidato]) -> list[Candidato]:
    """
    Reordena los candidatos por relevancia cruzada pregunta↔documento y trunca a
    settings.rerank_top_n. El score devuelto es el del reranker normalizado a (0,1).

    - Si el reranker está desactivado, es passthrough (recorte al top_n en orden denso).
    - Si el rerank falla, se loguea y se devuelve el orden denso (estabilidad de demo).
    """
    top_n = settings.rerank_top_n
    if not candidatos:
        return []
    if not settings.rerank_enabled:
        return candidatos[:top_n]

    try:
        textos = [contenido for contenido, _, _ in candidatos]
        scores = list(_get_encoder().rerank(pregunta, textos))
    except Exception as exc:
        logger.warning("Reranker falló, se usa el orden denso: %s", exc)
        return candidatos[:top_n]

    reordenados = [
        (contenido, metadata, _sigmoide(score))
        for (contenido, metadata, _), score in zip(candidatos, scores, strict=False)
    ]
    reordenados.sort(key=lambda c: c[2], reverse=True)
    return reordenados[:top_n]
