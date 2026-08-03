"""
Archivado del PDF original en Azure Blob Storage.

Guarda el documento tal cual se subió (fuente durable para re-index, descarga y
trazabilidad) — separado del índice vectorial de Chroma, que solo tiene los chunks.

Tolerante a fallos y opcional: si no hay `blob_connection_string` configurada, o si
la subida falla, la ingesta NO se rompe (devuelve None y registra el motivo). El PDF
ya quedó indexado; el archivado es un extra, no un requisito del flujo.
"""

import logging
from functools import lru_cache

from app.config import settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_container_client():
    """Cliente del contenedor Blob, o None si no está configurado.

    El import de azure-storage-blob es perezoso: entornos sin la dependencia o sin
    connection string (tests, dev local) no la necesitan.
    """
    if not settings.blob_connection_string:
        return None
    from azure.storage.blob import BlobServiceClient

    servicio = BlobServiceClient.from_connection_string(settings.blob_connection_string)
    return servicio.get_container_client(settings.blob_container)


def subir_pdf(ruta_local: str, nombre: str) -> str | None:
    """Sube el PDF original al contenedor Blob y devuelve su URL, o None.

    Idempotente: `overwrite=True` — re-subir el mismo nombre reemplaza el blob, igual
    que la re-indexación reemplaza sus chunks (borra por `fuente` antes de insertar).
    """
    cliente = _get_container_client()
    if cliente is None:
        logger.info("Blob deshabilitado (sin BLOB_CONNECTION_STRING); no se archiva %s", nombre)
        return None

    try:
        try:
            from azure.storage.blob import ContentSettings

            content_settings = ContentSettings(content_type="application/pdf")
        except Exception:
            content_settings = None

        with open(ruta_local, "rb") as f:
            blob = cliente.upload_blob(
                name=nombre,
                data=f,
                overwrite=True,
                content_settings=content_settings,
            )
        logger.info("PDF archivado en Blob: %s", nombre)
        return blob.url
    except Exception as exc:
        # No rompe la ingesta: el PDF ya está indexado, el archivado es adicional.
        logger.warning("No se pudo archivar %s en Blob: %s", nombre, exc)
        return None
