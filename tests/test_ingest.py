from types import SimpleNamespace

from langchain_core.documents import Document

from app.services import ingest


class _FakeValues(list):
    def tolist(self):
        return list(self)


class _FakeDataFrame:
    def __init__(self, columns, rows):
        self.columns = columns
        self.values = _FakeValues(rows)

    def fillna(self, _value):
        return self


class _FakeTable:
    def __init__(self, columns, rows):
        self._df = _FakeDataFrame(columns, rows)

    def export_to_dataframe(self, doc):
        return self._df


class _FakeDoc:
    def __init__(self, tables=None, markdown="# Titulo\n\nContenido institucional."):
        self.tables = tables or []
        self._markdown = markdown

    def export_to_markdown(self):
        return self._markdown


class _FakeChunker:
    def __init__(self, chunks):
        self._chunks = chunks

    def chunk(self, dl_doc):
        return self._chunks

    def contextualize(self, chunk):
        return chunk.text


def test_extraer_grupo_desde_nombre_archivo():
    assert ingest._extraer_grupo("IDGS901.pdf") == "901"
    assert ingest._extraer_grupo("reglamento_general.pdf") == ""


def test_horario_ignora_celdas_vacias_con_guiones():
    tabla = [
        ["Horas", "Lunes", "Martes"],
        ["08:00-08:50", "-\n-", "- -"],
    ]

    assert ingest._horario_a_frases(tabla, "901") == []


def test_procesar_pdf_usa_horario_extraido_por_docling(monkeypatch):
    table = _FakeTable(
        ["Horas", "Lunes", "Martes"],
        [["08:00-08:50", "Algebra\nD, DM\nIPM", "-"]],
    )
    conv_res = SimpleNamespace(document=_FakeDoc(tables=[table]))
    monkeypatch.setattr(ingest, "_convertir_con_docling", lambda _ruta: conv_res)

    monkeypatch.setattr(ingest, "_procesar_horario", lambda *_args: [])

    docs = ingest.procesar_pdf("horario.pdf", "IDGS901.pdf")

    # Aditivo: una frase por franja + una frase agregada por día.
    assert len(docs) == 2
    contenidos = [d.page_content for d in docs]
    assert any(c.startswith("Grupo 901. Lunes 08:00-08:50") for c in contenidos)  # por franja
    assert any("Clases del Lunes:" in c for c in contenidos)  # agregada por día
    for d in docs:
        assert d.metadata["tipo"] == "horario"
        assert d.metadata["parser"] == "docling"


def test_procesar_pdf_prefiere_horario_pdfplumber_mas_rico(monkeypatch):
    table = _FakeTable(
        ["Horas", "Lunes"],
        [["08:00-08:50", "Algebra\nD, DM\nIPM"]],
    )
    conv_res = SimpleNamespace(document=_FakeDoc(tables=[table]))
    pdfplumber_docs = [
        Document(
            page_content=(
                "Grupo 901. Lunes 08:00-08:50: Algebra "
                "(edificio D, aula DM; profesor Isabel Perez Mora (IPM))."
            ),
            metadata={
                "fuente": "IDGS901.pdf",
                "grupo": "901",
                "tipo": "horario",
                "parser": "pdfplumber",
            },
        )
    ]
    monkeypatch.setattr(ingest, "_convertir_con_docling", lambda _ruta: conv_res)
    monkeypatch.setattr(ingest, "_procesar_horario", lambda *_args: pdfplumber_docs)

    docs = ingest.procesar_pdf("horario.pdf", "IDGS901.pdf")

    assert docs == pdfplumber_docs


def test_procesar_pdf_horario_caida_a_pdfplumber(monkeypatch):
    conv_res = SimpleNamespace(document=_FakeDoc(tables=[]))
    expected = [
        Document(
            page_content="Grupo 901. Lunes 08:00-08:50: Algebra.",
            metadata={"tipo": "horario", "parser": "pdfplumber"},
        )
    ]
    monkeypatch.setattr(ingest, "_convertir_con_docling", lambda _ruta: conv_res)
    monkeypatch.setattr(ingest, "_procesar_horario", lambda *_args: expected)

    docs = ingest.procesar_pdf("horario.pdf", "IDGS901.pdf")

    assert docs == expected


def test_chunks_docling_agregan_metadata_y_contexto_de_tabla(monkeypatch):
    prov = SimpleNamespace(page_no=3)
    item = SimpleNamespace(label="table", prov=[prov])
    chunk = SimpleNamespace(
        text="| Carrera | Duracion |\n|---|---|\n| TSU | 6 cuatrimestres |",
        meta=SimpleNamespace(doc_items=[item], headings=["Oferta Academica"]),
    )
    conv_res = SimpleNamespace(document=_FakeDoc())
    monkeypatch.setattr(ingest, "_crear_docling_chunker", lambda: _FakeChunker([chunk]))

    docs = ingest._chunks_desde_docling(conv_res, "IngenieriasUTL.pdf", "")

    assert len(docs) == 1
    assert docs[0].metadata["fuente"] == "IngenieriasUTL.pdf"
    assert docs[0].metadata["pagina"] == 3
    assert docs[0].metadata["seccion"] == "Oferta Academica"
    assert docs[0].metadata["tipo"] == "tabla"
    assert docs[0].metadata["parser"] == "docling"
    assert docs[0].page_content.startswith(
        "Tabla del documento IngenieriasUTL.pdf, seccion Oferta Academica."
    )


def test_fallback_markdown_desde_docling_genera_chunks():
    texto = "# Servicios Escolares\n\n" + "Tramites universitarios. " * 220
    conv_res = SimpleNamespace(document=_FakeDoc(markdown=texto))

    docs = ingest._chunks_markdown_desde_docling(conv_res, "Servicios.pdf", "")

    assert docs
    assert all(doc.metadata["parser"] == "docling" for doc in docs)
    assert all(doc.metadata["tipo"] == "texto" for doc in docs)
