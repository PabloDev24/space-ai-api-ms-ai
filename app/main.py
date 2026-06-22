from fastapi import FastAPI

from app.routers import chat, health, ingest

app = FastAPI(
    title="Punto de Información Inteligente - UT León",
    description="API de consulta conversacional con RAG sobre información institucional.",
    version="1.0.0",
)

app.include_router(health.router)
app.include_router(chat.router)
app.include_router(ingest.router)
