"""
Indexador de PDFs para RAG con ChromaDB.

Uso:
    python scripts/ingest.py docs/

Procesa todos los PDFs de la carpeta indicada, los divide en chunks
y los guarda en la base ChromaDB definida en .env (CHROMA_PATH).
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

import chromadb
from pypdf import PdfReader

CHROMA_PATH = os.getenv("CHROMA_PATH", "chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "documentos")
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def extraer_texto_pdf(ruta: str) -> str:
    lector = PdfReader(ruta)
    texto = ""
    for pagina in lector.pages:
        contenido = pagina.extract_text()
        if contenido:
            texto += contenido + "\n"
    return texto


def dividir_en_chunks(texto: str) -> list[str]:
    chunks, inicio = [], 0
    while inicio < len(texto):
        chunk = texto[inicio: inicio + CHUNK_SIZE].strip()
        if chunk:
            chunks.append(chunk)
        inicio += CHUNK_SIZE - CHUNK_OVERLAP
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

    cliente = chromadb.PersistentClient(path=CHROMA_PATH)
    coleccion = cliente.get_or_create_collection(name=COLLECTION_NAME)
    total = 0

    for nombre in pdfs:
        ruta = os.path.join(carpeta, nombre)
        print(f"\nProcesando: {nombre}")
        texto = extraer_texto_pdf(ruta)
        if not texto.strip():
            print("  -> Sin texto extraíble (¿PDF escaneado?). Saltando.")
            continue

        chunks = dividir_en_chunks(texto)
        print(f"  -> {len(chunks)} chunks generados.")

        coleccion.upsert(
            documents=chunks,
            ids=[f"{nombre}_chunk_{i}" for i in range(len(chunks))],
            metadatas=[{"fuente": nombre, "chunk": i} for i in range(len(chunks))],
        )
        total += len(chunks)

    print(f"\nListo. {total} chunks indexados en '{CHROMA_PATH}'.")


if __name__ == "__main__":
    main()
