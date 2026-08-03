"""
Tests del archivado del PDF en Blob (app.services.blob).

Sin red ni azure-storage-blob real: se mockea el cliente del contenedor. Verifica que
sin connection string el archivado se omite (None) y que con cliente sube con overwrite
y devuelve la URL. El cliente está cacheado (lru_cache) → se limpia entre tests.
"""

from app.services import blob


def _reset_cache():
    blob._get_container_client.cache_clear()


def test_deshabilitado_sin_connection_string(monkeypatch, tmp_path):
    monkeypatch.setattr(blob.settings, "blob_connection_string", "")
    _reset_cache()

    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")

    assert blob.subir_pdf(str(pdf), "x.pdf") is None


def test_sube_con_overwrite_y_devuelve_url(monkeypatch, tmp_path):
    subidas = {}

    class _FakeBlob:
        url = "https://acct.blob.core.windows.net/documents/x.pdf"

    class _FakeContainer:
        def upload_blob(self, name, data, overwrite, content_settings):
            subidas["name"] = name
            subidas["overwrite"] = overwrite
            data.read()  # consume el stream como el SDK real
            return _FakeBlob()

    monkeypatch.setattr(blob, "_get_container_client", lambda: _FakeContainer())

    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")

    url = blob.subir_pdf(str(pdf), "x.pdf")

    assert url == "https://acct.blob.core.windows.net/documents/x.pdf"
    assert subidas == {"name": "x.pdf", "overwrite": True}


def test_fallo_de_subida_no_rompe_devuelve_none(monkeypatch, tmp_path):
    class _ExplotaContainer:
        def upload_blob(self, **_kwargs):
            raise RuntimeError("red caída")

    monkeypatch.setattr(blob, "_get_container_client", lambda: _ExplotaContainer())

    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")

    # La ingesta no debe romperse por un fallo de archivado.
    assert blob.subir_pdf(str(pdf), "x.pdf") is None
