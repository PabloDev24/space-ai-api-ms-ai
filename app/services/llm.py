from collections.abc import Callable

from app.config import settings

_INSTRUCCION_SISTEMA = """
Eres el asistente virtual del punto de información de la Universidad
Tecnológica de León (UT León). Ayudas a estudiantes nuevos, estudiantes
actuales, visitantes o cualquier persona que se encuentre en el campus,
respondiendo preguntas sobre ubicaciones, cómo llegar a edificios,
carreras, procesos administrativos y cualquier información relacionada
con la universidad.

Reglas de estilo para tus respuestas:
- Habla de forma natural, cálida y directa, como lo haría una persona
  real que trabaja en el módulo de informes de la universidad.
- NUNCA uses frases como "según el contexto", "de acuerdo a la
  información proporcionada", "en base a los documentos" o similares.
  El usuario no debe notar que estás usando información recuperada de
  documentos; debe sentir que simplemente sabes la respuesta.
- Sé breve y claro. Evita relleno innecesario o repetir la pregunta.
- SIEMPRE que recibas información de contexto, úsala directamente para
  responder. La información de contexto es confiable y actualizada.
- Cuando recibas una tabla de horario, léela con cuidado: las columnas
  son los días de la semana (Lunes a Sábado) y cada fila es un bloque
  horario. Cada celda puede contener materia, modalidad, aula y
  abreviatura del docente. Usa la fecha y hora actual para identificar
  qué día es "hoy" y responde con las materias de esa columna.
- Si NO se te da contexto relevante, responde con tu conocimiento
  general de forma útil, sin mencionar documentos.
- Si de verdad no sabes la respuesta, dilo con honestidad y sugiere
  que la persona pregunte en el módulo de informes o control escolar.
""".strip()

_MODELOS_DEFAULT = {
    "gemini": "gemini-2.5-flash",
    "groq": "llama-3.3-70b-versatile",
    "openai": "gpt-4o-mini",
}

_GROQ_BASE_URL = "https://api.groq.com/openai/v1"


def _build_gemini() -> Callable[[str], str]:
    import google.generativeai as genai

    modelo = settings.modelo or settings.modelo_gemini
    genai.configure(api_key=settings.gemini_api_key)
    model = genai.GenerativeModel(modelo, system_instruction=_INSTRUCCION_SISTEMA)

    def generate(prompt: str) -> str:
        return model.generate_content(
            prompt, request_options={"timeout": settings.llm_timeout_seconds}
        ).text

    return generate


def _build_openai_compatible(api_key: str, base_url: str | None) -> Callable[[str], str]:
    from openai import OpenAI

    modelo = settings.modelo or _MODELOS_DEFAULT.get(settings.proveedor, "gpt-4o-mini")
    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=settings.llm_timeout_seconds,
        max_retries=1,
    )

    def generate(prompt: str) -> str:
        response = client.chat.completions.create(
            model=modelo,
            messages=[
                {"role": "system", "content": _INSTRUCCION_SISTEMA},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content

    return generate


def _init() -> Callable[[str], str]:
    proveedor = settings.proveedor.lower()

    if proveedor == "gemini":
        return _build_gemini()
    if proveedor == "groq":
        return _build_openai_compatible(settings.groq_api_key, _GROQ_BASE_URL)
    if proveedor == "openai":
        return _build_openai_compatible(settings.openai_api_key, None)

    raise ValueError(
        f"Proveedor '{proveedor}' no reconocido. Opciones válidas: gemini, groq, openai"
    )


generate = _init()
