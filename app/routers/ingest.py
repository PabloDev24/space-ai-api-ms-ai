import logging
import os
import tempfile

from fastapi import APIRouter, Depends, Header, HTTPException, UploadFile
from pydantic import BaseModel

from app.config import settings
from app.services.blob import subir_pdf
from app.services.ingest import indexar_pdf

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
