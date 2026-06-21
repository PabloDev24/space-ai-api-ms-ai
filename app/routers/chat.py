from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.rag import buscar_contexto
from app.services.llm import model

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

    if documentos:
        contexto = "\n\n".join(documentos)
        prompt = f"Información relacionada:\n{contexto}\n\nPregunta de la persona: {pregunta}"
    else:
        prompt = pregunta

    try:
        respuesta = model.generate_content(prompt)
        return ChatResponse(respuesta=respuesta.text)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Error al consultar Gemini: {e}")
