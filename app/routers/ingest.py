import logging
import os
import tempfile
from urllib.parse import urlparse

from fastapi import APIRouter, Depends, Header, HTTPException, UploadFile
from pydantic import BaseModel

from app.config import settings
from app.services.blob import subir_pdf
from app.services.ingest import eliminar_documento, indexar_pdf, indexar_url, listar_documentos

logger = logging.getLogger(__name__)

router = APIRouter(tags=["ingest"])

ALLOWED_CONTENT_TYPES = {"application/pdf"}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB


def verificar_api_key(x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> None:
    """Exige X-API-Key solo si INGEST_API_KEY está configurada.

    Sin configurar (dev/local/demo) el endpoint queda abierto, pero se avisa en el log
    para que no pase inadvertido en un despliegue real.
    """
    esperado = settings.ingest_api_key
    if not esperado:
        logger.warning("INGEST_API_KEY sin configurar; /ingest queda abierto (solo dev/demo).")
        return
    if x_api_key != esperado:
        raise HTTPException(status_code=401, detail="API key inválida o ausente.")


class IngestResponse(BaseModel):
    archivo: str
    chunks_indexados: int
    mensaje: str
    archivado_en: str | None = None


class DocumentoIndexado(BaseModel):
    archivo: str
    chunks: int
    tipos: list[str]


class IngestUrlRequest(BaseModel):
    url: str
    # Nombre con el que aparece la fuente en las citas y en el panel. Si no llega,
    # se deriva del host y la ruta para que el usuario reconozca de dónde salió.
    nombre: str | None = None


class EliminarResponse(BaseModel):
    archivo: str
    chunks_eliminados: int


def _nombre_desde_url(url: str) -> str:
    partes = urlparse(url)
    ruta = (partes.path or "").strip("/")
    if ruta:
        return f"{partes.netloc}/{ruta}"
    return partes.netloc or url


# Sin auth, igual que /ask: el único caller es el backend .NET por red interna,
# no un navegador. No expone contenido, solo nombres de archivo y conteos.
@router.get("/documents", response_model=list[DocumentoIndexado])
async def listar_documentos_endpoint():
    return listar_documentos()


@router.post("/ingest", response_model=IngestResponse, dependencies=[Depends(verificar_api_key)])
async def ingest_documento(archivo: UploadFile):
    if archivo.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Tipo de archivo no soportado: '{archivo.content_type}'. Solo se aceptan PDFs.",
        )

    nombre = os.path.basename(archivo.filename or "documento.pdf")
    if not nombre.lower().endswith(".pdf"):
        raise HTTPException(status_code=415, detail="El archivo debe tener extensión .pdf")

    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_path = tmp.name
            size = 0
            while chunk := await archivo.read(1024 * 64):
                size += len(chunk)
                if size > MAX_FILE_SIZE:
                    limite_mb = MAX_FILE_SIZE // (1024 * 1024)
                    raise HTTPException(
                        status_code=413,
                        detail=f"El archivo supera el límite de {limite_mb} MB.",
                    )
                tmp.write(chunk)

        chunks_indexados = indexar_pdf(tmp_path, nombre)

        if chunks_indexados == 0:
            raise HTTPException(
                status_code=422,
                detail="No se pudo extraer contenido del PDF. Verifica que no esté vacío o protegido.",  # noqa: E501
            )

        # Archiva el PDF original (durable) antes de borrar el temporal. No bloqueante:
        # si el Blob no está configurado o falla, devuelve None y la ingesta ya está hecha.
        archivado_en = subir_pdf(tmp_path, nombre)

        return IngestResponse(
            archivo=nombre,
            chunks_indexados=chunks_indexados,
            mensaje=f"Documento indexado correctamente con {chunks_indexados} fragmentos.",
            archivado_en=archivado_en,
        )
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


@router.post(
    "/ingest-url", response_model=IngestResponse, dependencies=[Depends(verificar_api_key)]
)
async def ingest_url(peticion: IngestUrlRequest):
    """Indexa el contenido de una página web como una fuente más del RAG."""
    url = peticion.url.strip()
    partes = urlparse(url)

    # El converter aceptaría rutas locales y esquemas como file://. Este endpoint lo
    # invoca el backend, así que una URL no validada permitiría leer el disco del
    # servidor o alcanzar servicios internos que no están expuestos a la red.
    if partes.scheme not in ("http", "https") or not partes.netloc:
        raise HTTPException(
            status_code=422,
            detail="La URL debe ser absoluta y usar http o https.",
        )

    nombre = (peticion.nombre or "").strip() or _nombre_desde_url(url)

    try:
        chunks_indexados = indexar_url(url, nombre)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Falló la ingesta de la URL %s", url)
        raise HTTPException(
            status_code=502,
            detail=f"No se pudo obtener o procesar la página: {exc}",
        ) from exc

    if chunks_indexados == 0:
        raise HTTPException(
            status_code=422,
            detail="No se extrajo contenido de la página. Puede requerir JavaScript o estar vacía.",
        )

    return IngestResponse(
        archivo=nombre,
        chunks_indexados=chunks_indexados,
        mensaje=f"Página indexada correctamente con {chunks_indexados} fragmentos.",
        archivado_en=None,
    )


@router.delete(
    "/documents/{archivo:path}",
    response_model=EliminarResponse,
    dependencies=[Depends(verificar_api_key)],
)
async def eliminar_documento_endpoint(archivo: str):
    """
    Retira una fuente del índice vectorial.

    Sin esto, borrar un documento desde el panel lo quita del catálogo pero el RAG
    lo sigue recuperando y citando en sus respuestas.
    """
    eliminados = eliminar_documento(archivo)
    if eliminados == 0:
        raise HTTPException(
            status_code=404,
            detail=f"No hay chunks indexados para '{archivo}'.",
        )

    return EliminarResponse(archivo=archivo, chunks_eliminados=eliminados)
