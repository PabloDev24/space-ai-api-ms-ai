"""
Indexador de documentos para RAG con LangChain + ChromaDB.

Uso:
    python scripts/ingest.py docs/

Soporta documentos universitarios variados. Usa Docling como parser principal
layout-aware con OCR, tablas, columnas y secciones; luego genera chunks
estructurales para ChromaDB.
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from langchain_chroma import Chroma

# Importar después de load_dotenv para que las variables de entorno estén disponibles
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.config import settings
from app.services.embeddings import get_embeddings

# El pipeline de procesamiento vive en app.services.ingest (única fuente de
# verdad). Este script solo orquesta el recorrido por lote de una carpeta.
from app.services.ingest import procesar_pdf


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
        except Exception:  # nosec B110
            pass

        vectorstore.add_documents(chunks)
        total += len(chunks)

    print(f"\nListo. {total} chunks indexados en '{settings.chroma_path}'.")
    print(f"Total en la colección: {vectorstore._collection.count()} chunks.")


if __name__ == "__main__":
    main()
