"""
Servicio de indexación de documentos PDF para RAG.

Reutiliza la lógica de scripts/ingest.py como módulo importable
para que el endpoint HTTP pueda invocarla sin duplicar código.
"""

import re

import pdfplumber
import pymupdf4llm
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from app.config import settings
from app.services.embeddings import get_embeddings

# Días de la semana que identifican una tabla de horario. Sin acentos para
# coincidir con el texto tal como lo extrae pdfplumber del PDF.
DIAS_SEMANA = ("lunes", "martes", "miercoles", "jueves", "viernes", "sabado", "domingo")

# Celdas vacías de la cuadrícula de horario (guiones sueltos en distintas formas).
_CELDA_VACIA = {"", "-", "-\n-", "--"}

# Medidos en TOKENS (no caracteres) para respetar el límite de la ventana del LLM.
# 512 tokens es lo bastante grande para contener una sección o tabla de horario
# completa sin partirla, soportando preguntas y respuestas amplias.
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50

HEADERS_TO_SPLIT = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
]


def _extraer_grupo(nombre_archivo: str) -> str:
    match = re.search(r"(\d{3,4})", nombre_archivo)
    return match.group(1) if match else ""


def _es_cabecera_horario(fila: list) -> bool:
    """True si la fila contiene al menos dos días de la semana (cabecera de horario)."""
    texto = " ".join(str(c).lower() for c in fila if c)
    return sum(1 for dia in DIAS_SEMANA if dia in texto) >= 2


def _formatear_celda(valor: str) -> str:
    """
    Convierte una celda de horario en una frase legible.

    El PDF apila en una celda, en este orden fijo (separado por saltos de línea):
        1. Materia (puede ocupar varias líneas)
        2. Edificio y aula (ej. "D, DM" = edificio D, aula DM)
        3. Nomenclatura del profesor (ej. "IPM"); si no hay profesor, es "-"

    Ejemplo: "Desarrollo WEB Integral\\nD, DM\\nIPM"
        → "Desarrollo WEB Integral (edificio D, aula DM; profesor IPM)"

    El campo del profesor SÍ se conserva aunque sea "-" para no desplazar las
    posiciones: la última línea siempre es el profesor, la penúltima el aula.
    """
    lineas = [ln.strip() for ln in str(valor).split("\n") if ln.strip()]
    if not lineas:
        return ""
    if len(lineas) == 1:
        return lineas[0] if lineas[0] != "-" else ""

    # Orden fijo: [...materia, edificio_aula, profesor]
    profesor = lineas[-1]
    edificio_aula = lineas[-2]
    materia = " ".join(lineas[:-2]) if len(lineas) > 2 else lineas[0]

    detalle = []
    if edificio_aula and edificio_aula != "-":
        # "D, DM" → edificio D, aula DM
        partes_ubicacion = [p.strip() for p in edificio_aula.split(",")]
        if len(partes_ubicacion) == 2:
            detalle.append(f"edificio {partes_ubicacion[0]}, aula {partes_ubicacion[1]}")
        else:
            detalle.append(f"aula {edificio_aula}")
    if profesor and profesor != "-":
        detalle.append(f"profesor {profesor}")

    frase = materia.strip()
    if detalle:
        frase += f" ({'; '.join(detalle)})"
    return frase


def _horario_a_frases(tabla: list, grupo: str) -> list[str]:
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
            if valor in _CELDA_VACIA:
                continue
            clase = _formatear_celda(valor)
            if clase:
                frases.append(f"{pref}{dia} {horas}: {clase}.")
    return frases


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
        for pagina in pdf.pages:
            for tabla in pagina.extract_tables():
                frases.extend(_horario_a_frases(tabla, grupo))

    return [
        Document(page_content=frase, metadata={"fuente": nombre, "grupo": grupo, "tipo": "horario"})
        for frase in frases
    ]


def procesar_pdf(ruta: str, nombre: str) -> list[Document]:
    grupo = _extraer_grupo(nombre)

    # Los horarios son cuadrículas que pymupdf4llm desalinea: se procesan aparte
    # con pdfplumber, convirtiendo cada clase en una frase legible.
    docs_horario = _procesar_horario(ruta, nombre, grupo)
    if docs_horario:
        return docs_horario

    md_texto = pymupdf4llm.to_markdown(ruta)
    if not md_texto.strip():
        return []

    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS_TO_SPLIT,
        strip_headers=False,
    )
    docs_por_seccion = md_splitter.split_text(md_texto)

    # Conteo por tokens (tiktoken) para no rebasar la ventana del LLM.
    char_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = char_splitter.split_documents(docs_por_seccion)

    for chunk in chunks:
        chunk.metadata.update({"fuente": nombre, "grupo": grupo})

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

    # Colección nueva: si no hay nada que borrar, ignoramos el error.
    try:
        vectorstore.delete(where={"fuente": nombre_archivo})
    except Exception:  # noqa: S110  # nosec
        pass

    vectorstore.add_documents(chunks)
    return len(chunks)
