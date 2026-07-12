from datetime import datetime

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from app.services.llm import generate
from app.services.rag import buscar_contexto_con_metadata

DIAS = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]
SIN_CONTEXTO = (
    "No encontré información suficiente en los documentos del campus "
    "para responder con certeza."
)

router = APIRouter(tags=["ask"])


def _parse_page(pagina: object) -> int | None:
    """La metadata 'pagina' a veces es un rango tipo '17,18' — se toma la primera."""
    if pagina is None:
        return None
    try:
        return int(str(pagina).split(",")[0].strip())
    except (ValueError, IndexError):
        return None


class UserContext(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    user_id: str = Field(alias="userId")
    name: str
    career: str


class AskRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    question: str
    user_context: UserContext = Field(alias="userContext")
    source: str


class Source(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    title: str
    page: int | None = None
    chunk_id: str = Field(alias="chunkId")


class AskResponse(BaseModel):
    answer: str
    confidence: float
    sources: list[Source]


@router.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    pregunta = request.question.strip()
    if not pregunta:
        raise HTTPException(status_code=400, detail="La pregunta no puede estar vacía.")

    resultados = buscar_contexto_con_metadata(pregunta)

    if not resultados:
        return AskResponse(answer=SIN_CONTEXTO, confidence=0.0, sources=[])

    ahora = datetime.now()
    contexto_fecha = (
        f"Fecha y hora actual: {ahora.strftime('%A %d de %B de %Y, %H:%M')} "
        f"({DIAS[ahora.weekday()]})"
    )
    contexto = "\n\n".join(contenido for contenido, _, _ in resultados)
    prompt = (
        f"{contexto_fecha}\n\n"
        f"Información relacionada:\n{contexto}\n\n"
        f"Pregunta de {request.user_context.name} ({request.user_context.career}): {pregunta}"
    )

    try:
        respuesta = generate(prompt)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Error al consultar el LLM: {e}") from e

    mejor_score = max(score for _, _, score in resultados)
    sources = [
        Source(
            title=metadata.get("fuente", "documento"),
            page=_parse_page(metadata.get("pagina")),
            chunk_id=f"{metadata.get('fuente', 'doc')}-{metadata.get('pagina', 0)}-{i}",
        )
        for i, (_, metadata, _) in enumerate(resultados)
    ]

    return AskResponse(answer=respuesta, confidence=mejor_score, sources=sources)
