"""
Servicio de indexación de documentos PDF para RAG.

Reutiliza la lógica de scripts/ingest.py como módulo importable
para que el endpoint HTTP pueda invocarla sin duplicar código.
"""

import re

import pymupdf4llm
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from app.config import settings
from app.services.embeddings import get_embeddings

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


def procesar_pdf(ruta: str, nombre: str) -> list[Document]:
    grupo = _extraer_grupo(nombre)

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
