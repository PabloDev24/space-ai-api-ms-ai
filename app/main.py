import logging

from fastapi import FastAPI

from app.config import settings
from app.routers import ask, chat, health, ingest

# Sin esto, el root logger de Python queda en WARNING sin formato — los
# logger.info(...) que ya existen (ver app/services/ingest.py) nunca se ven.
logging.basicConfig(
    level=settings.log_level.upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

app = FastAPI(
    title="Punto de Información Inteligente - UT León",
    description="API de consulta conversacional con RAG sobre información institucional.",
    version="1.0.0",
)

app.include_router(health.router)
app.include_router(chat.router)
app.include_router(ask.router)
app.include_router(ingest.router)
