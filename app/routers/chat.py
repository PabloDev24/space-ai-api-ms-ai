from datetime import datetime
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.rag import buscar_contexto
from app.services.llm import generate

DIAS = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]

router = APIRouter(tags=["chat"])


class ChatRequest(BaseModel):
    pregunta: str


class ChatResponse(BaseModel):
    respuesta: str


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    pregunta = request.pregunta.strip()
    if not pregunta:
        raise HTTPException(status_code=400, detail="La pregunta no puede estar vacía.")

    documentos = buscar_contexto(pregunta)

    ahora = datetime.now()
    contexto_fecha = (
        f"Fecha y hora actual: {ahora.strftime('%A %d de %B de %Y, %H:%M')} "
        f"({DIAS[ahora.weekday()]})"
    )

    if documentos:
        contexto = "\n\n".join(documentos)
        prompt = (
            f"{contexto_fecha}\n\n"
            f"Información relacionada:\n{contexto}\n\n"
            f"Pregunta de la persona: {pregunta}"
        )
    else:
        prompt = f"{contexto_fecha}\n\nPregunta de la persona: {pregunta}"

    try:
        return ChatResponse(respuesta=generate(prompt))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Error al consultar el LLM: {e}")
