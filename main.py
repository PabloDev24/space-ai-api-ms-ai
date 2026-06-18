from fastapi import FastAPI

app = FastAPI(
    title="Space AI RAG API",
    description="Microservicio de RAG para procesamiento de documentos y preguntas",
    version="1.0.0",
)


@app.get("/health", tags=["status"])
async def health():
    return {"status": "ok"}
