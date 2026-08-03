"""
Scraping puntual y respetuoso de utleon.edu.mx para un set curado de páginas
informativas (oferta educativa, admisiones, becas, servicios, etc.).

El sitio es una SPA en Vue.js sin robots.txt con reglas explícitas: el
contenido se renderiza por JS, así que un simple `requests.get`/curl solo
trae el shell vacío del index. Usa Playwright (navegador headless) para
esperar el render y extraer el texto real.

Requiere, una sola vez, además de requirements.txt:
    pip install playwright
    playwright install chromium

No se agrega playwright a requirements.txt: el servicio en producción nunca
lo usa (solo esta herramienta de mantenimiento offline), así que no tiene
sentido cargar un navegador headless (~300MB) en la imagen de Docker.

1 request por página, 1.5s de pausa entre cada una — no es un crawl del
sitio completo, es un set curado de ~60 páginas relevantes para un
asistente de campus estudiantil (se excluyen adrede páginas de gobierno
corporativo/transparencia, de bajo valor para preguntas de estudiantes).

Uso:
    python scripts/scrape_utl_web.py
"""

import json
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

BASE = "https://www.utleon.edu.mx/"
OUT_DIR = Path(__file__).resolve().parent.parent / "docs" / "_web"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT_DIR / "paginas.json"

RUTAS = [
    "historia",
    "filosofia",
    "mensaje-rector",
    "directorio",
    "campus-central",
    "campus-ii",
    "campus-acambaro",
    "oferta-educativa-ingenierias-leon",
    "oferta-educativa-licenciaturas-leon",
    "oferta-educativa-ingenierias-acambaro",
    "oferta-educativa-licenciaturas-acambaro",
    "admisiones-leon",
    "admisiones-acambaro",
    "admisiones-continuidad",
    "becas",
    "becas-acambaro",
    "estadias",
    "estadias-empresariales",
    "estadias-especiales",
    "extraordinario",
    "extraordinario-acambaro",
    "recuperacion",
    "recuperacion-acambaro",
    "reinscripcion",
    "reinscripcion-acambaro",
    "reingreso",
    "reingreso-acambaro",
    "baja-temporal",
    "baja-temporal-acambaro",
    "baja-definitiva",
    "baja-definitiva-acambaro",
    "titulacion-tsu",
    "titulacion-tsu-acambaro",
    "titulacion-ing-lic",
    "titulacion-ing-lic-acambaro",
    "tutoreo",
    "biblioteca",
    "laboratorios",
    "centros-computo",
    "aula-virtual",
    "atencion-tecnopedagogica",
    "psicopedagogico",
    "salud-integral",
    "constancias",
    "constancias-acambaro",
    "tramite-credencial",
    "bolsa-de-trabajo",
    "bolsa-trabajo-egresado",
    "bolsa-trabajo-egresado-acambaro",
    "microcredenciales",
    "delfin",
    "incubacion",
    "educacion-dual",
    "procuraduria-derechos-universitarios-genero",
    "comite-atencion-violencia",
    "protocolo-seguridad",
    "igualdad",
    "utl-datos",
]


def main():
    paginas = []
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        for ruta in RUTAS:
            url = BASE + ruta
            try:
                page.goto(url, wait_until="networkidle", timeout=20000)
                paginas.append(
                    {"url": url, "titulo": page.title(), "texto": page.inner_text("body")}
                )
                print(f"OK  {url}  ({len(paginas[-1]['texto'])} chars)")
            except Exception as e:
                print(f"ERR {url}: {e}")
            time.sleep(1.5)
        browser.close()

    OUT_FILE.write_text(json.dumps(paginas, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n{len(paginas)} páginas guardadas en {OUT_FILE}")


if __name__ == "__main__":
    main()
