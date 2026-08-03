"""
Tests de autenticación del endpoint /ingest (cabecera X-API-Key).

Se mockean indexar_pdf y subir_pdf: aquí se prueba el gate de auth, no la ingesta ni
el Blob reales. Con INGEST_API_KEY configurada, falta/valor incorrecto → 401; correcto
→ 200. Sin configurar, el endpoint queda abierto (dev/demo).
"""

from fastapi.testclient import TestClient

from app.main import app
from app.routers import ingest as ingest_router

client = TestClient(app)

_PDF = ("x.pdf", b"%PDF-1.4 fake", "application/pdf")


def _mock_pipeline(monkeypatch):
    monkeypatch.setattr(ingest_router, "indexar_pdf", lambda *_a: 3)
    monkeypatch.setattr(ingest_router, "subir_pdf", lambda *_a: None)


def test_rechaza_sin_api_key_cuando_esta_configurada(monkeypatch):
    monkeypatch.setattr(ingest_router.settings, "ingest_api_key", "secreto")
    _mock_pipeline(monkeypatch)

    resp = client.post("/ingest", files={"archivo": _PDF})
    assert resp.status_code == 401


def test_rechaza_api_key_incorrecta(monkeypatch):
    monkeypatch.setattr(ingest_router.settings, "ingest_api_key", "secreto")
    _mock_pipeline(monkeypatch)

    resp = client.post("/ingest", files={"archivo": _PDF}, headers={"X-API-Key": "malo"})
    assert resp.status_code == 401


def test_acepta_api_key_correcta(monkeypatch):
    monkeypatch.setattr(ingest_router.settings, "ingest_api_key", "secreto")
    _mock_pipeline(monkeypatch)

    resp = client.post("/ingest", files={"archivo": _PDF}, headers={"X-API-Key": "secreto"})
    assert resp.status_code == 200
    assert resp.json()["chunks_indexados"] == 3


def test_abierto_sin_api_key_configurada(monkeypatch):
    monkeypatch.setattr(ingest_router.settings, "ingest_api_key", "")
    _mock_pipeline(monkeypatch)

    resp = client.post("/ingest", files={"archivo": _PDF})
    assert resp.status_code == 200
