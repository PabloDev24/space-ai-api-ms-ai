"""
Indexador de documentos para RAG con LangChain + ChromaDB.

Uso:
    python scripts/ingest.py docs/

Soporta cualquier PDF sin importar su estructura. Usa pymupdf4llm para
convertir cada PDF a Markdown inteligente (preserva tablas, columnas y
secciones) y luego LangChain para fragmentar por estructura semántica.
"""

import os
import re
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

import pymupdf4llm
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

# Importar embeddings después de load_dotenv para que las variables de entorno estén disponibles
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.config import settings
from app.services.embeddings import get_embeddings

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# Encabezados Markdown que definen límites de contexto
HEADERS_TO_SPLIT = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
]


def _extraer_grupo(nombre_archivo: str) -> str:
    """Extrae el identificador numérico de grupo del nombre del archivo."""
    match = re.search(r'(\d{3,4})', nombre_archivo)
    return match.group(1) if match else ""


def procesar_pdf(ruta: str, nombre: str) -> list[Document]:
    """
    Convierte un PDF a chunks de LangChain Documents.

    pymupdf4llm convierte el PDF completo a Markdown preservando:
    - Tablas como tablas Markdown (con | columnas | y --- separadores)
    - Encabezados con # según jerarquía real del documento
    - Texto en columnas correctamente reordenado

    Luego LangChain fragmenta primero por encabezados (contexto semántico)
    y después por tamaño de caracteres si el fragmento es muy largo.
    """
    grupo = _extraer_grupo(nombre)

    # 1. PDF → Markdown inteligente
    md_texto = pymupdf4llm.to_markdown(ruta)
    if not md_texto.strip():
        return []

    # 2. Fragmentar por encabezados (mantiene contexto de sección)
    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS_TO_SPLIT,
        strip_headers=False,  # mantiene el encabezado dentro del chunk
    )
    docs_por_seccion = md_splitter.split_text(md_texto)

    # 3. Fragmentar secciones largas por caracteres con solapamiento
    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = char_splitter.split_documents(docs_por_seccion)

    # 4. Añadir metadatos de origen a cada chunk
    for chunk in chunks:
        chunk.metadata.update({
            "fuente": nombre,
            "grupo": grupo,
        })

    return chunks


def main():
    if len(sys.argv) < 2:
        print("Uso: python scripts/ingest.py <carpeta_con_pdfs>")
        sys.exit(1)

    carpeta = sys.argv[1]
    if not os.path.isdir(carpeta):
        print(f"Error: '{carpeta}' no es una carpeta válida.")
        sys.exit(1)

    pdfs = [f for f in os.listdir(carpeta) if f.lower().endswith(".pdf")]
    if not pdfs:
        print(f"No se encontraron PDFs en '{carpeta}'.")
        sys.exit(1)

    print(f"Se encontraron {len(pdfs)} PDF(s).")
    print(f"Proveedor de embeddings: {settings.embedding_provider}")

    embeddings = get_embeddings()

    # Cargar o crear el vectorstore (persiste automáticamente)
    vectorstore = Chroma(
        collection_name=settings.collection_name,
        embedding_function=embeddings,
        persist_directory=settings.chroma_path,
    )

    total = 0
    for nombre in pdfs:
        ruta = os.path.join(carpeta, nombre)
        print(f"\nProcesando: {nombre}")

        chunks = procesar_pdf(ruta, nombre)
        if not chunks:
            print("  -> Sin contenido extraíble. Saltando.")
            continue

        print(f"  -> {len(chunks)} chunks generados.")

        # Eliminar chunks anteriores del mismo archivo antes de reinsertar
        try:
            vectorstore.delete(where={"fuente": nombre})
        except Exception:
            pass  # colección nueva, no hay nada que borrar

        vectorstore.add_documents(chunks)
        total += len(chunks)

    print(f"\nListo. {total} chunks indexados en '{settings.chroma_path}'.")
    print(f"Total en la colección: {vectorstore._collection.count()} chunks.")


if __name__ == "__main__":
    main()
