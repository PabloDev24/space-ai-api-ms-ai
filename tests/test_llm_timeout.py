"""
Regresión: sin timeout, un proveedor colgado deja /ask esperando para siempre
y, con un solo worker, tumba /health para todos los demás usuarios hasta
reiniciar el proceso. El cliente del LLM debe llevar un timeout duro.
"""

from app.config import settings
from app.services import llm


def test_cliente_openai_compatible_lleva_timeout(monkeypatch):
    capturado = {}

    class _FakeOpenAI:
        def __init__(self, **kwargs):
            capturado.update(kwargs)

    monkeypatch.setattr("openai.OpenAI", _FakeOpenAI)
    llm._build_openai_compatible("clave-falsa", "https://ejemplo.invalido/v1")

    assert capturado.get("timeout") == settings.llm_timeout_seconds
    assert capturado.get("max_retries") == 1
