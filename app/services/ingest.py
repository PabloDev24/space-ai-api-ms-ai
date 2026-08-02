"""
Servicio de indexación de documentos PDF para RAG.

Reutiliza la lógica de scripts/ingest.py como módulo importable
para que el endpoint HTTP pueda invocarla sin duplicar código.
"""

import logging
import re
from typing import Any

import pdfplumber
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings
from app.services.embeddings import get_embeddings

logger = logging.getLogger(__name__)

# Días de la semana que identifican una tabla de horario. Sin acentos para
# coincidir con el texto tal como lo extrae pdfplumber del PDF.
DIAS_SEMANA = ("lunes", "martes", "miercoles", "jueves", "viernes", "sabado", "domingo")

# Celdas vacías de la cuadrícula de horario (guiones sueltos en distintas formas).
_CELDA_VACIA = {"", "-", "-\n-", "--"}

# Medidos en TOKENS para Docling HybridChunker. El fallback Markdown usa una
# equivalencia aproximada en caracteres para no depender de descargas de tokenizer.
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
FALLBACK_CHAR_CHUNK_SIZE = 2400
FALLBACK_CHAR_CHUNK_OVERLAP = 250

_LABEL_TIPOS = {
    "table": "tabla",
    "list_item": "lista",
    "ordered_list": "lista",
    "unordered_list": "lista",
    "picture": "imagen",
    "chart": "imagen",
    "document_index": "lista",
}


def _extraer_grupo(nombre_archivo: str) -> str:
    match = re.search(r"(\d{3,4})", nombre_archivo)
    return match.group(1) if match else ""


def _docling_ocr_langs() -> list[str]:
    langs = [lang.strip() for lang in settings.docling_ocr_langs.split(",")]
    return [lang for lang in langs if lang] or ["es"]


def _crear_docling_converter():
    try:
        from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import (
            EasyOcrOptions,
            PdfPipelineOptions,
            TableStructureOptions,
        )
        from docling.document_converter import DocumentConverter, PdfFormatOption
    except ImportError as exc:
        raise RuntimeError(
            "Docling no está instalado. Ejecuta `pip install -r requirements.txt` "
            "para habilitar el nuevo parser de PDFs."
        ) from exc

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = settings.docling_enable_ocr
    pipeline_options.do_table_structure = settings.docling_enable_tables
    pipeline_options.table_structure_options = TableStructureOptions(do_cell_matching=True)
    pipeline_options.accelerator_options = AcceleratorOptions(
        num_threads=settings.docling_num_threads,
        device=AcceleratorDevice.AUTO,
    )

    pipeline_options.ocr_options = EasyOcrOptions(
        lang=_docling_ocr_langs(),
        use_gpu=True if settings.docling_ocr_use_gpu else None,
    )

    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
        }
    )


def _convertir_con_docling(ruta: str):
    return _crear_docling_converter().convert(source=ruta)


def _crear_docling_chunker():
    try:
        from docling.chunking import HybridChunker
    except ImportError as exc:
        raise RuntimeError(
            "Docling HybridChunker no está disponible. Reinstala con "
            "`pip install -r requirements.txt`."
        ) from exc

    return HybridChunker(merge_peers=True)


def _valor_label(label: Any) -> str:
    raw = getattr(label, "value", None) or getattr(label, "name", None) or str(label)
    return str(raw).replace("DocItemLabel.", "").lower()


def _page_no(prov: Any) -> int | None:
    page = getattr(prov, "page_no", None)
    if isinstance(page, int):
        return page
    return None


def _metadata_desde_chunk(chunk: Any, nombre: str, grupo: str) -> dict[str, Any]:
    meta = getattr(chunk, "meta", None)
    doc_items = list(getattr(meta, "doc_items", []) or [])
    labels = [_valor_label(getattr(item, "label", "")) for item in doc_items]
    paginas = sorted(
        {
            page
            for item in doc_items
            for prov in getattr(item, "prov", []) or []
            if (page := _page_no(prov)) is not None
        }
    )

    tipo = "texto"
    for label in labels:
        if label in _LABEL_TIPOS:
            tipo = _LABEL_TIPOS[label]
            break
    if "mapa" in nombre.lower() and tipo == "imagen":
        tipo = "mapa"

    headings = [str(h).strip() for h in getattr(meta, "headings", []) or [] if str(h).strip()]

    metadata: dict[str, Any] = {
        "fuente": nombre,
        "grupo": grupo,
        "tipo": tipo,
        "parser": "docling",
    }
    if paginas:
        metadata["pagina"] = paginas[0] if len(paginas) == 1 else ",".join(map(str, paginas))
    if headings:
        metadata["seccion"] = " > ".join(headings[-3:])
    if labels:
        metadata["labels"] = ",".join(sorted(set(labels)))
    return metadata


def _contexto_tabla(metadata: dict[str, Any], nombre: str) -> str:
    seccion = metadata.get("seccion")
    if seccion:
        return f"Tabla del documento {nombre}, seccion {seccion}."
    return f"Tabla del documento {nombre}."


def _chunks_desde_docling(conv_res: Any, nombre: str, grupo: str) -> list[Document]:
    chunker = _crear_docling_chunker()
    docs: list[Document] = []

    for chunk in chunker.chunk(dl_doc=conv_res.document):
        texto = chunker.contextualize(chunk=chunk).strip()
        if not texto:
            continue
        metadata = _metadata_desde_chunk(chunk, nombre, grupo)
        if metadata["tipo"] == "tabla":
            texto = f"{_contexto_tabla(metadata, nombre)}\n\n{texto}"
        docs.append(Document(page_content=texto, metadata=metadata))

    return docs


def _tabla_a_frases_genericas(matriz: list, nombre: str, seccion: str = "") -> list[str]:
    """
    Convierte una tabla de datos (no horario) en una frase por fila, con pares
    cabecera:valor. Cada fila queda recuperable de forma independiente, en vez de
    embeber toda la tabla como un bloque que pierde la relación fila↔columna.
    Devuelve [] si la matriz es una cuadrícula de horario (tiene su propia ruta).
    """
    if not matriz or len(matriz) < 2:
        return []
    cabecera = [str(c).strip() for c in matriz[0]]
    if _es_cabecera_horario(cabecera):
        return []

    pref = f"En {nombre}" + (f", sección {seccion}" if seccion else "")
    frases: list[str] = []
    for fila in matriz[1:]:
        pares = [
            f"{cabecera[i]}: {str(valor).strip()}"
            for i, valor in enumerate(fila)
            if i < len(cabecera) and cabecera[i] and not _es_celda_vacia(str(valor))
        ]
        if pares:
            frases.append(f"{pref}: {'; '.join(pares)}.")
    return frases


def _procesar_tablas_genericas(conv_res: Any, nombre: str, grupo: str) -> list[Document]:
    docs: list[Document] = []
    for table in getattr(conv_res.document, "tables", []) or []:
        try:
            matriz = _tabla_docling_a_matriz(table, conv_res.document)
        except Exception as exc:
            logger.warning("No se pudo convertir una tabla de Docling: %s", exc)
            continue
        try:
            seccion = (table.caption_text(conv_res.document) or "").strip()
        except Exception:
            seccion = ""
        docs.extend(
            Document(
                page_content=frase,
                metadata={"fuente": nombre, "grupo": grupo, "tipo": "tabla", "parser": "docling"},
            )
            for frase in _tabla_a_frases_genericas(matriz, nombre, seccion)
        )
    return docs


def _chunks_markdown_desde_docling(conv_res: Any, nombre: str, grupo: str) -> list[Document]:
    md_texto = conv_res.document.export_to_markdown()
    if not md_texto.strip():
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max(settings.docling_chunk_size * 4, FALLBACK_CHAR_CHUNK_SIZE),
        chunk_overlap=FALLBACK_CHAR_CHUNK_OVERLAP,
        separators=["\n## ", "\n# ", "\n\n", "\n", ". ", " ", ""],
    )
    docs = splitter.create_documents(
        [md_texto],
        metadatas=[{"fuente": nombre, "grupo": grupo, "tipo": "texto", "parser": "docling"}],
    )
    for idx, doc in enumerate(docs):
        doc.metadata["chunk"] = idx
    return docs


def _tabla_docling_a_matriz(table: Any, dl_doc: Any) -> list[list[str]]:
    df = table.export_to_dataframe(doc=dl_doc)
    headers = [str(col).strip() for col in df.columns]
    filas = [
        [str(value).strip() if value is not None else "" for value in row]
        for row in df.fillna("").values.tolist()
    ]
    if _es_cabecera_horario(headers):
        return [headers, *filas]
    if filas and _es_cabecera_horario(filas[0]):
        return filas
    return [headers, *filas]


def _tablas_desde_docling(conv_res: Any) -> list[list[list[str]]]:
    tablas = []
    for table in getattr(conv_res.document, "tables", []) or []:
        try:
            matriz = _tabla_docling_a_matriz(table, conv_res.document)
        except Exception as exc:
            logger.warning("No se pudo convertir una tabla de Docling: %s", exc)
            continue
        if matriz:
            tablas.append(matriz)
    return tablas


def _procesar_horario_docling(conv_res: Any, nombre: str, grupo: str) -> list[Document]:
    tablas = _tablas_desde_docling(conv_res)
    if not tablas:
        return []

    frases: list[str] = []
    docentes = _extraer_docentes(tablas)
    for tabla in tablas:
        frases.extend(_horario_a_frases(tabla, grupo, docentes))
        frases.extend(_horario_por_dia_a_frases(tabla, grupo, docentes))

    return [
        Document(
            page_content=frase,
            metadata={"fuente": nombre, "grupo": grupo, "tipo": "horario", "parser": "docling"},
        )
        for frase in frases
    ]


def _promedio_largo_texto(docs: list[Document]) -> float:
    if not docs:
        return 0
    return sum(len(doc.page_content) for doc in docs) / len(docs)


def _horario_pdfplumber_es_mas_rico(
    docs_pdfplumber: list[Document],
    docs_docling: list[Document],
) -> bool:
    if not docs_pdfplumber:
        return False
    if not docs_docling:
        return True

    tiene_docentes_pdf = any("profesor" in doc.page_content.lower() for doc in docs_pdfplumber)
    tiene_docentes_docling = any("profesor" in doc.page_content.lower() for doc in docs_docling)
    if tiene_docentes_pdf and not tiene_docentes_docling:
        return True

    if len(docs_pdfplumber) >= max(1, int(len(docs_docling) * 0.75)):
        return _promedio_largo_texto(docs_pdfplumber) > _promedio_largo_texto(docs_docling) * 1.15

    return False


def _es_cabecera_horario(fila: list) -> bool:
    """True si la fila contiene al menos dos días de la semana (cabecera de horario)."""
    texto = " ".join(str(c).lower() for c in fila if c)
    return sum(1 for dia in DIAS_SEMANA if dia in texto) >= 2


def _es_celda_vacia(valor: str) -> bool:
    texto = re.sub(r"\s+", "", str(valor or ""))
    return not texto or texto in _CELDA_VACIA or set(texto) == {"-"}


def _normalizar_materia(texto: str) -> str:
    """Normaliza el nombre de una materia para emparejar horario y tabla de docentes."""
    return re.sub(r"\s+", " ", str(texto).replace("\n", " ")).strip().lower()


def _extraer_docentes(tablas: list) -> dict[str, str]:
    """
    Construye un mapa {materia normalizada -> nombre completo del docente}.

    El horario solo lleva las siglas del profesor en cada celda (ej. "IPM"),
    pero el PDF incluye una tabla aparte (sección de exámenes) que lista cada
    docente con su nombre completo y la asignatura que imparte. El puente
    fiable entre ambas es la asignatura, no la sigla.
    """
    mapa: dict[str, str] = {}
    for tabla in tablas:
        if not tabla or not tabla[0]:
            continue
        cabecera = [str(c).strip().lower() if c else "" for c in tabla[0]]
        if "docente" not in cabecera or "asignatura" not in cabecera:
            continue
        idx_doc = cabecera.index("docente")
        idx_asig = cabecera.index("asignatura")
        for fila in tabla[1:]:
            if idx_doc >= len(fila) or idx_asig >= len(fila):
                continue
            nombre = str(fila[idx_doc] or "").replace("\n", " ").strip()
            materia = _normalizar_materia(fila[idx_asig] or "")
            if nombre and nombre != "-" and materia:
                mapa[materia] = nombre
    return mapa


def _formatear_celda(valor: str, docentes: dict[str, str] | None = None) -> str:
    """
    Convierte una celda de horario en una frase legible.

    El PDF apila en una celda, en este orden fijo (separado por saltos de línea):
        1. Materia (puede ocupar varias líneas)
        2. Edificio y aula (ej. "D, DM" = edificio D, aula DM)
        3. Nomenclatura del profesor (ej. "IPM"); si no hay profesor, es "-"

    Ejemplo: "Desarrollo WEB Integral\\nD, DM\\nIPM"
        → "Desarrollo WEB Integral (edificio D, aula DM; profesor Ismael Perez Mena (IPM))"

    Si `docentes` mapea materia→nombre completo, se resuelve el nombre del
    profesor a partir de la materia (las siglas del horario son ambiguas).

    El campo del profesor SÍ se conserva aunque sea "-" para no desplazar las
    posiciones: la última línea siempre es el profesor, la penúltima el aula.
    """
    if _es_celda_vacia(valor):
        return ""

    lineas = [ln.strip() for ln in str(valor).split("\n") if ln.strip()]
    if not lineas:
        return ""
    if len(lineas) == 1:
        return lineas[0] if not _es_celda_vacia(lineas[0]) else ""

    # Orden fijo: [...materia, edificio_aula, profesor]
    siglas_profesor = lineas[-1]
    edificio_aula = lineas[-2]
    materia = " ".join(lineas[:-2]) if len(lineas) > 2 else lineas[0]
    if _es_celda_vacia(materia):
        return ""

    detalle = []
    if edificio_aula and edificio_aula != "-":
        # "D, DM" → edificio D, aula DM
        partes_ubicacion = [p.strip() for p in edificio_aula.split(",")]
        if len(partes_ubicacion) == 2:
            detalle.append(f"edificio {partes_ubicacion[0]}, aula {partes_ubicacion[1]}")
        else:
            detalle.append(f"aula {edificio_aula}")
    if siglas_profesor and siglas_profesor != "-":
        nombre = (docentes or {}).get(_normalizar_materia(materia))
        if nombre:
            detalle.append(f"profesor {nombre} ({siglas_profesor})")
        else:
            detalle.append(f"profesor {siglas_profesor}")

    frase = materia.strip()
    if detalle:
        frase += f" ({'; '.join(detalle)})"
    return frase


def _horario_a_frases(tabla: list, grupo: str, docentes: dict[str, str] | None = None) -> list[str]:
    """
    Convierte una tabla de horario (extraída por pdfplumber) en frases naturales,
    una por clase. Evita que el LLM tenga que alinear columnas de una cuadrícula.
    """
    if not tabla or not _es_cabecera_horario(tabla[0]):
        return []

    cabecera = [str(c).strip() if c else "" for c in tabla[0]]
    # Índices de columnas que son días de la semana
    cols_dia = {i: cabecera[i] for i, c in enumerate(cabecera) if c.lower() in DIAS_SEMANA}
    # Columna de horas (la que contiene rangos tipo HH:MM-HH:MM)
    idx_horas = next(
        (i for i, c in enumerate(cabecera) if c.lower() in ("horas", "hora")),
        None,
    )

    frases = []
    pref = f"Grupo {grupo}. " if grupo else ""
    for fila in tabla[1:]:
        if not fila or _es_cabecera_horario(fila):
            continue
        horas = ""
        if idx_horas is not None and idx_horas < len(fila):
            horas = str(fila[idx_horas]).strip()
        if not re.search(r"\d{1,2}:\d{2}", horas):
            continue  # fila sin franja horaria válida (cabeceras intermedias, etc.)

        for idx, dia in cols_dia.items():
            if idx >= len(fila):
                continue
            valor = str(fila[idx]).strip() if fila[idx] else ""
            if _es_celda_vacia(valor):
                continue
            clase = _formatear_celda(valor, docentes)
            if clase:
                frases.append(f"{pref}{dia} {horas}: {clase}.")
    return frases


def _horario_por_dia_a_frases(
    tabla: list, grupo: str, docentes: dict[str, str] | None = None
) -> list[str]:
    """
    Una frase por (grupo, día) que agrega TODAS las clases de ese día.

    Complementa a `_horario_a_frases` (una por franja): las consultas del tipo
    "¿qué clases tengo el sábado?" necesitan cobertura completa del día, que los
    chunks por franja no dan porque varios slots casi idénticos de una misma materia
    copan el top-k y dejan fuera a las demás.
    """
    if not tabla or not _es_cabecera_horario(tabla[0]):
        return []

    cabecera = [str(c).strip() if c else "" for c in tabla[0]]
    cols_dia = {i: cabecera[i] for i, c in enumerate(cabecera) if c.lower() in DIAS_SEMANA}
    idx_horas = next(
        (i for i, c in enumerate(cabecera) if c.lower() in ("horas", "hora")),
        None,
    )

    por_dia: dict[str, list[str]] = {dia: [] for dia in cols_dia.values()}
    for fila in tabla[1:]:
        if not fila or _es_cabecera_horario(fila):
            continue
        horas = ""
        if idx_horas is not None and idx_horas < len(fila):
            horas = str(fila[idx_horas]).strip()
        if not re.search(r"\d{1,2}:\d{2}", horas):
            continue
        for idx, dia in cols_dia.items():
            if idx >= len(fila):
                continue
            valor = str(fila[idx]).strip() if fila[idx] else ""
            if _es_celda_vacia(valor):
                continue
            clase = _formatear_celda(valor, docentes)
            if clase:
                por_dia[dia].append(f"{horas} {clase}")

    pref = f"Grupo {grupo}. " if grupo else ""
    return [
        f"{pref}Clases del {dia}: {'; '.join(clases)}."
        for dia, clases in por_dia.items()
        if clases
    ]


def _procesar_horario(ruta: str, nombre: str, grupo: str) -> list[Document]:
    """
    Extrae tablas de horario con pdfplumber y las convierte en frases naturales.

    pdfplumber lee la cuadrícula con columnas alineadas (a diferencia de la
    conversión a Markdown, que desalinea las celdas). Cada clase se vuelve un
    Document independiente para que la recuperación semántica sea precisa.
    Devuelve [] si el PDF no contiene una tabla de horario.
    """
    frases: list[str] = []
    with pdfplumber.open(ruta) as pdf:
        tablas = [tabla for pagina in pdf.pages for tabla in pagina.extract_tables()]

    # La tabla de docentes mapea cada materia a su profesor con nombre completo,
    # que luego resolvemos al formatear las siglas de cada clase del horario.
    docentes = _extraer_docentes(tablas)
    for tabla in tablas:
        frases.extend(_horario_a_frases(tabla, grupo, docentes))
        frases.extend(_horario_por_dia_a_frases(tabla, grupo, docentes))

    return [
        Document(
            page_content=frase,
            metadata={"fuente": nombre, "grupo": grupo, "tipo": "horario", "parser": "pdfplumber"},
        )
        for frase in frases
    ]


def procesar_pdf(ruta: str, nombre: str) -> list[Document]:
    grupo = _extraer_grupo(nombre)

    conv_res = _convertir_con_docling(ruta)

    # Docling es la ruta principal. Los horarios se extraen primero desde sus
    # tablas; si no logra reconstruir la cuadrícula, se usa el fallback probado.
    docs_horario = _procesar_horario_docling(conv_res, nombre, grupo)
    if docs_horario:
        docs_horario_pdfplumber = _procesar_horario(ruta, nombre, grupo)
        if _horario_pdfplumber_es_mas_rico(docs_horario_pdfplumber, docs_horario):
            return docs_horario_pdfplumber
        return docs_horario

    docs_horario = _procesar_horario(ruta, nombre, grupo)
    if docs_horario:
        return docs_horario

    try:
        # Aditivo: se conserva el chunk de tabla de Docling (buen contexto de sección) y
        # se añaden frases por fila para que cada fila de una tabla de datos sea
        # recuperable con precisión. La duplicación es menor y el reranker la resuelve.
        chunks = _chunks_desde_docling(conv_res, nombre, grupo)
        chunks += _procesar_tablas_genericas(conv_res, nombre, grupo)
    except Exception:
        chunks = _chunks_markdown_desde_docling(conv_res, nombre, grupo)

    return chunks


def indexar_pdf(ruta_archivo: str, nombre_archivo: str) -> int:
    """
    Procesa un PDF y guarda sus chunks en ChromaDB.

    Elimina chunks previos del mismo archivo antes de reinsertar,
    garantizando idempotencia ante re-subidas del mismo documento.

    Returns:
        Número de chunks indexados.
    """
    chunks = procesar_pdf(ruta_archivo, nombre_archivo)
    if not chunks:
        return 0

    vectorstore = Chroma(
        collection_name=settings.collection_name,
        embedding_function=get_embeddings(),
        persist_directory=settings.chroma_path,
    )

    try:
        vectorstore.delete(where={"fuente": nombre_archivo})
    except Exception as exc:
        logger.debug("No se pudieron borrar chunks previos de %s: %s", nombre_archivo, exc)

    vectorstore.add_documents(chunks)
    return len(chunks)
