"""
Tests de la linealización fila→frase de tablas de datos no-horario.

- Unit: la función pura sobre una matriz mock (determinista, sin PDF ni modelos).
- Integración: ingesta el fixture real (tests/fixtures/tabla_estadia.pdf) y comprueba
  que una fila concreta se recupera con precisión (recuperación densa, sin reranker
  para que el test sea rápido).
"""

from pathlib import Path

from langchain_chroma import Chroma

from app.services import ingest

FIXTURE = Path(__file__).parent / "fixtures" / "tabla_estadia.pdf"


def test_tabla_generica_una_frase_por_fila():
    matriz = [
        ["Modalidad", "Costo", "Documento a entregar"],
        ["Presencial", "$1,500", "Carta de aceptacion"],
        ["Internacional", "$3,000", "Visa y seguro medico"],
    ]
    frases = ingest._tabla_a_frases_genericas(matriz, "cuotas.pdf", seccion="Estadias")

    assert len(frases) == 2  # una por fila de datos
    assert "Modalidad: Internacional" in frases[1]
    assert "Costo: $3,000" in frases[1]
    assert "Documento a entregar: Visa y seguro medico" in frases[1]
    assert frases[1].startswith("En cuotas.pdf, sección Estadias:")


def test_tabla_generica_salta_horario():
    # Una cuadrícula de horario no debe linealizarse por esta ruta.
    matriz = [
        ["Horas", "Lunes", "Martes"],
        ["07:00-08:00", "Mate", "Fisica"],
    ]
    assert ingest._tabla_a_frases_genericas(matriz, "x.pdf") == []


def test_tabla_generica_salta_celdas_vacias():
    matriz = [["A", "B", "C"], ["1", "-", "3"]]
    frases = ingest._tabla_a_frases_genericas(matriz, "x.pdf")
    assert frases == ["En x.pdf: A: 1; C: 3."]  # la celda "-" (vacía) se omite


def test_integracion_fixture_recupera_fila_precisa(tmp_path, monkeypatch):
    assert FIXTURE.exists(), "falta el fixture tabla_estadia.pdf"

    chunks = ingest.procesar_pdf(str(FIXTURE), "tabla_estadia.pdf")
    # Debe haber una frase por fila con la fila 'Internacional' completa.
    internacional = [c for c in chunks if "Internacional" in c.page_content]
    assert internacional, "no se generó una frase para la fila Internacional"
    assert any("$3,000" in c.page_content for c in internacional)

    # Recuperación densa (sin reranker, para que el test sea rápido) sobre índice temporal.
    from app.services import embeddings

    vs = Chroma(
        collection_name="test_tablas",
        embedding_function=embeddings.get_embeddings(),
        persist_directory=str(tmp_path),
    )
    vs.add_documents(chunks)
    docs = vs.similarity_search(
        "cuánto cuesta la modalidad internacional y qué documento entrego", k=1
    )

    assert docs, "sin resultados"
    top = docs[0].page_content
    assert "Internacional" in top and "$3,000" in top and "Visa" in top
