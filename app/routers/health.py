from fastapi import APIRouter
from app.services.rag import collection_info

router = APIRouter(tags=["status"])


@router.get("/health")
def health_check():
    info = collection_info()
    return {
        "estado": "ok",
        "base_de_conocimiento_cargada": info["cargada"],
        "chunks_disponibles": info["chunks"],
    }
