from fastapi import APIRouter

from app.services.rag import buscar_contexto, collection_info

router = APIRouter(tags=["status"])


@router.get("/health")
def health_check():
    info = collection_info()
    return {
        "estado": "ok",
        "base_de_conocimiento_cargada": info["cargada"],
        "chunks_disponibles": info["chunks"],
    }


@router.get("/debug/rag")
def debug_rag(pregunta: str):
    chunks = buscar_contexto(pregunta)
    return {"total": len(chunks), "chunks": chunks}
