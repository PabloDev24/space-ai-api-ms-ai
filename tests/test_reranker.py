"""
Tests del reranker cross-encoder.

Se mockea el encoder para no descargar el modelo (1.11GB) ni depender de red:
lo que se verifica es la lógica de reordenar/truncar/gating, no el modelo en sí.
"""

from app.services import reranker


class _FakeEncoder:
    """Devuelve un score por documento según un mapa contenido->score."""

    def __init__(self, scores_por_texto):
        self._scores = scores_por_texto

    def rerank(self, query, documents, **kwargs):
        return [self._scores[d] for d in documents]


def _candidatos():
    # Orden denso inicial: el más relevante ("B") llega en segundo lugar a propósito.
    return [
        ("doc A poco relevante", {"fuente": "a"}, 0.4),
        ("doc B muy relevante", {"fuente": "b"}, 0.3),
        ("doc C irrelevante", {"fuente": "c"}, 0.2),
    ]


def test_reranker_sube_el_mas_relevante_y_trunca(monkeypatch):
    monkeypatch.setattr(reranker.settings, "rerank_enabled", True)
    monkeypatch.setattr(reranker.settings, "rerank_top_n", 2)
    # B recibe el score crudo más alto del cross-encoder.
    fake = _FakeEncoder(
        {"doc A poco relevante": -2.0, "doc B muy relevante": 5.0, "doc C irrelevante": -6.0}
    )
    monkeypatch.setattr(reranker, "_get_encoder", lambda: fake)

    salida = reranker.rerank("¿cuál es el relevante?", _candidatos())

    assert len(salida) == 2  # truncado a top_n
    assert salida[0][1]["fuente"] == "b"  # B ascendió al primer lugar
    assert 0.0 <= salida[0][2] <= 1.0  # score normalizado con sigmoide
    assert salida[0][2] > salida[1][2]  # orden descendente por score


def test_reranker_desactivado_es_passthrough(monkeypatch):
    monkeypatch.setattr(reranker.settings, "rerank_enabled", False)
    monkeypatch.setattr(reranker.settings, "rerank_top_n", 2)

    salida = reranker.rerank("x", _candidatos())

    assert [c[1]["fuente"] for c in salida] == ["a", "b"]  # orden denso original, truncado


def test_reranker_tolera_fallo_del_encoder(monkeypatch):
    monkeypatch.setattr(reranker.settings, "rerank_enabled", True)
    monkeypatch.setattr(reranker.settings, "rerank_top_n", 2)

    def _boom():
        raise RuntimeError("modelo no disponible")

    monkeypatch.setattr(reranker, "_get_encoder", _boom)

    salida = reranker.rerank("x", _candidatos())

    # No rompe: cae al orden denso truncado.
    assert [c[1]["fuente"] for c in salida] == ["a", "b"]


def test_reranker_lista_vacia():
    assert reranker.rerank("x", []) == []
