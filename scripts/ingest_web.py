"""
Indexa en ChromaDB las páginas scrapeadas por scrape_utl_web.py
(docs/_web/paginas.json).

Quita el boilerplate de nav/footer (idéntico en todas las páginas, detectado
por frecuencia) y trocea el resto en chunks de ~700 caracteres. Chunks más
chicos que el fallback genérico de texto plano (FALLBACK_CHAR_CHUNK_SIZE en
app/services/ingest.py, 2400): el reranker cross-encoder procesa cada par
(pregunta, chunk) y su costo escala con el largo del chunk — en el tier
Basic de Azure (CPU compartida) chunks de 2400 caracteres hacían que
preguntas con contenido web tardaran 30-50s+ en responder; con ~700 baja a
5-10s.

Uso:
    python scripts/scrape_utl_web.py   # primero, genera docs/_web/paginas.json
    python scripts/ingest_web.py
"""

import json
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings
from app.services.embeddings import get_embeddings

WEB_DIR = Path(__file__).resolve().parent.parent / "docs" / "_web"
WEB_CHAR_CHUNK_SIZE = 700
WEB_CHAR_CHUNK_OVERLAP = 80

# Líneas de nav/footer idénticas en todas las páginas (calculado por
# frecuencia >=70% sobre las páginas scrapeadas). No aportan nada al chunk y
# repetirlas en decenas de chunks infla el peso de texto genérico.
BOILERPLATE = {
    "CONOCE LA UTL",
    "ACTIVIDADES CULTURALES",
    "(477) 7 10 00 20",
    "Protocolos de Salud para el Estudiantado",
    "Aviso de Privacidad Simplificado | Aviso de Privacidad Integral",
    "Puedes conocer nuestro aviso de privacidad aquí",
    "Transparencia",
    "CAMPUS LEÓN",
    "Noticias",
    "INTERNACIONALIZACIÓN",
    "SOMOS UTL",
    "CAMPUS ACÁMBARO",
    "CONTÁCTANOS",
    "TIENDA SOMOS LEONES",
    "Calendario Escolar 2026 - 2027",
    "CP. 37670 León, Gto. Mex.",
    "CERRAR",
    "TRANSPARENCIA",
    "ACTIVIDADES DEPORTIVAS",
    "difusion@utleon.edu.mx",
    "¿Quieres Trabajar en la UTL?",
    "Calendario Escolar 2025 - 2026",
    "Blvd. Universidad Tecnológica #225 Col. San Carlos",
    "Protocolo de Atención a Casos de Violencia, Acoso y Hostigamiento Escolar",
    "SITO",
    "ENLACES DE INTERÉS",
}

# Páginas de "misión/mensaje/generalidades": prosa genérica sobre "la UTL"
# que compite en embedding contra preguntas puntuales que un PDF fuente ya
# responde mejor y con más precisión (ej. "qué ingenierías ofrece" perdía
# contra el mensaje del rector). Se excluyen de la ingesta.
EXCLUIR_FUENTES = {
    "https://www.utleon.edu.mx/",
    "https://www.utleon.edu.mx/mensaje-rector",
    "https://www.utleon.edu.mx/modelo-flexible-leon",
    "https://www.utleon.edu.mx/modelo-flexible-acambaro",
}


def limpiar(texto: str) -> str:
    lineas = [ln for ln in texto.splitlines() if ln.strip() and ln.strip() not in BOILERPLATE]
    return "\n".join(lineas).strip()


def main():
    paginas = json.loads((WEB_DIR / "paginas.json").read_text(encoding="utf-8"))
    paginas = [p for p in paginas if p["url"] not in EXCLUIR_FUENTES]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=WEB_CHAR_CHUNK_SIZE,
        chunk_overlap=WEB_CHAR_CHUNK_OVERLAP,
    )

    docs: list[Document] = []
    fuentes: set[str] = set()
    for p in paginas:
        limpio = limpiar(p["texto"])
        if len(limpio) < 80:  # página sin contenido propio (solo boilerplate)
            continue
        fuentes.add(p["url"])
        for texto in splitter.split_text(limpio):
            docs.append(
                Document(
                    page_content=texto,
                    metadata={"fuente": p["url"], "titulo": p["titulo"], "tipo": "web"},
                )
            )

    print(f"{len(paginas)} páginas -> {len(fuentes)} con contenido -> {len(docs)} chunks")

    vectorstore = Chroma(
        collection_name=settings.collection_name,
        embedding_function=get_embeddings(),
        persist_directory=settings.chroma_path,
    )
    for fuente in fuentes | EXCLUIR_FUENTES:
        try:
            vectorstore.delete(where={"fuente": fuente})
        except Exception:  # nosec B110
            pass
    vectorstore.add_documents(docs)
    print(f"Total en la colección: {vectorstore._collection.count()} chunks.")


if __name__ == "__main__":
    main()
