"""
Regresión del bug de frescura del índice (cold-start).

Antes del fix: si al importar `rag` el directorio de Chroma no existía, el
vectorstore quedaba en None permanentemente y las consultas nunca veían los
documentos ingestados después, aun estando en el mismo proceso. Este test
reproduce ese escenario y verifica que ya no ocurre.
"""

import importlib

from langchain_chroma import Chroma
from langchain_core.documents import Document


def test_cold_start_ve_documentos_ingestados_sin_reiniciar(tmp_path, monkeypatch):
    chroma_dir = tmp_path / "chroma_inexistente"
    assert not chroma_dir.exists()  # cold-start: el índice no existe al arrancar

    # Recarga el módulo con la config apuntando al dir vacío, simulando el boot.
    from app import config

    monkeypatch.setattr(config.settings, "chroma_path", str(chroma_dir))
    from app.services import embeddings, rag

    monkeypatch.setattr(embeddings.settings, "chroma_path", str(chroma_dir))
    rag = importlib.reload(rag)

    # Con índice vacío, la búsqueda no rompe y devuelve lista vacía.
    assert rag.buscar_contexto("¿dónde está la entrada?") == []

    # Se ingesta un documento por una instancia Chroma aparte (como indexar_pdf),
    # en el mismo proceso, SIN reiniciar.
    vs_ingest = Chroma(
        collection_name=config.settings.collection_name,
        embedding_function=embeddings.get_embeddings(),
        persist_directory=str(chroma_dir),
    )
    vs_ingest.add_documents(
        [
            Document(
                page_content="La entrada principal del campus está al norte.",
                metadata={"fuente": "mapa"},
            )
        ]
    )

    # El fix: la consulta ahora SÍ ve el documento recién ingestado.
    resultados = rag.buscar_contexto("¿dónde está la entrada principal?")
    assert resultados, "cold-start: la consulta no vio el documento ingestado sin reiniciar"
    assert "entrada principal" in resultados[0].lower()
