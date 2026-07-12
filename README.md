# Space AI · ms-ai 🤖

![Python](https://img.shields.io/badge/Python%203.14-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-FF6B35?style=for-the-badge&logo=databricks&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)

Microservicio de inteligencia artificial del ecosistema **Space AI**, diseñado como el **Punto de Información Inteligente** de la Universidad Tecnológica de León. Responde preguntas institucionales de forma natural mediante una arquitectura RAG sobre documentos PDF.

---

## 👨‍💻 Desarrollador y Equipo

- **Desarrollador Principal:** Daniel Ojeda
- **Equipo de Desarrollo:** Lattice Systems

---

## 🚀 Tecnologías y Arquitectura

Este microservicio implementa una **arquitectura RAG (Retrieval-Augmented Generation)** con proveedor de LLM y embeddings intercambiables via variables de entorno:

- **Framework:** FastAPI con Python 3.14.
- **LLM:** Multi-proveedor — Groq, OpenAI o Google Gemini (configurable con `PROVEEDOR`).
- **Embeddings:** Multi-proveedor — local (fastembed/ONNX), OpenAI o Gemini (configurable con `EMBEDDING_PROVIDER`).
- **Pipeline RAG:** Docling + LangChain para normalizar PDFs universitarios variados a chunks estructurales, con fragmentación por tokens (~512 tokens) y metadata de origen/sección/tipo.
- **Parseo de PDFs:** Docling como parser principal layout-aware con OCR, tablas, listas, títulos y párrafos; `pdfplumber` queda como fallback especializado para horarios en cuadrícula.
- **Base de conocimiento:** ChromaDB como vector store persistente.
- **Configuración:** Pydantic Settings v2 con soporte `.env`.
- **Contenedores:** Docker & Docker Compose.

---

## 🛠️ Instalación y Configuración

### 1. Prerrequisitos

| Herramienta | Versión mínima | Descarga |
|---|---|---|
| **Docker Desktop** | 4.x o superior | [docker.com](https://www.docker.com/) |
| **Python** | 3.14 | [python.org](https://www.python.org/) |
| **Git** | última versión | [git-scm.com](https://git-scm.com/) |

---

### 2. Variables de entorno

Copia el archivo de ejemplo y configura según el proveedor que vayas a usar:

```bash
cp .env.example .env
```

**LLM:**

| Variable | Requerida | Default | Descripción |
|---|---|---|---|
| `PROVEEDOR` | ✅ | `groq` | Proveedor LLM: `groq` \| `openai` \| `gemini` |
| `GROQ_API_KEY` | Si `PROVEEDOR=groq` | — | API Key de [console.groq.com](https://console.groq.com) |
| `OPENAI_API_KEY` | Si `PROVEEDOR=openai` | — | API Key de OpenAI |
| `GEMINI_API_KEY` | Si `PROVEEDOR=gemini` | — | API Key de Google AI Studio |
| `MODELO` | ❌ | auto | Sobreescribe el modelo default del proveedor |

**Embeddings:**

| Variable | Requerida | Default | Descripción |
|---|---|---|---|
| `EMBEDDING_PROVIDER` | ❌ | `local` | `local` \| `openai` \| `gemini` |
| `EMBEDDING_MODEL` | ❌ | auto | Sobreescribe el modelo default de embeddings |

**RAG:**

| Variable | Requerida | Default | Descripción |
|---|---|---|---|
| `CHROMA_PATH` | ❌ | `chroma_db` | Ruta de la base vectorial persistente |
| `COLLECTION_NAME` | ❌ | `documentos` | Nombre de la colección en ChromaDB |
| `N_RESULTADOS` | ❌ | `8` | Chunks recuperados por consulta |

**Parsing/OCR de PDFs:**

| Variable | Requerida | Default | Descripción |
|---|---|---|---|
| `DOCLING_ENABLE_OCR` | ❌ | `true` | Activa OCR automático para PDFs escaneados |
| `DOCLING_ENABLE_TABLES` | ❌ | `true` | Activa reconocimiento estructural de tablas |
| `DOCLING_OCR_LANGS` | ❌ | `es,en` | Idiomas para OCR, separados por coma |
| `DOCLING_OCR_USE_GPU` | ❌ | `false` | Permite usar GPU en OCR si el entorno lo soporta |
| `DOCLING_NUM_THREADS` | ❌ | `4` | Hilos usados por Docling |
| `DOCLING_CHUNK_SIZE` | ❌ | `512` | Tamaño objetivo de chunks para HybridChunker |

> ℹ️ Para pruebas gratuitas usa `PROVEEDOR=groq` (free tier en [console.groq.com](https://console.groq.com)) y `EMBEDDING_PROVIDER=local` (corre en tu máquina, sin API key).

---

### 3. Método A: Docker (Recomendado)

```bash
# Levantar el servicio
docker compose up -d

# Verificar que está corriendo
curl http://localhost:8000/health
```

---

### 4. Método B: Ejecución local

```bash
# 1. Crear entorno virtual e instalar dependencias
python -m venv .venv
source .venv/bin/activate      # Linux/Mac
.venv\Scripts\activate         # Windows

pip install -r requirements.txt

# 2. Levantar la API
uvicorn main:app --reload
```

---

### 5. Verificar el estado

Abre **[http://localhost:8000/docs](http://localhost:8000/docs)** para explorar la documentación interactiva (Swagger UI).

---

## 📚 Indexar Documentos (RAG)

El microservicio enriquece sus respuestas con PDFs institucionales. Hay dos formas de indexar:

**Opción A · Por lote (CLI):** coloca los archivos en `docs/` e indexa toda la carpeta:

```bash
python scripts/ingest.py docs/
```

**Opción B · Subida individual (endpoint):** sube un PDF al vuelo vía `POST /ingest`:

```bash
curl -X POST http://localhost:8000/ingest \
  -F "archivo=@docs/IDGS901.pdf;type=application/pdf"
```

El pipeline usa Docling como parser principal para documentos universitarios variados: carreras, mapas, reglamentos, trámites, servicios, convocatorias, calendarios, listas, tablas y PDFs escaneados. La salida se normaliza a chunks con metadata de origen, sección, página y tipo de contenido.

- **Documentos generales** (carreras, servicios escolares, planes de estudio, reglamentos, mapas, etc.): se convierten con Docling, se fragmentan por estructura con `HybridChunker` y conservan metadata como `fuente`, `pagina`, `seccion`, `tipo` y `parser`.
- **Tablas**: se mantienen como chunks atómicos cuando es posible y se enriquecen con una frase de contexto para mejorar la recuperación semántica.
- **Horarios de grupo** (cuadrículas con días de la semana): primero se intentan reconstruir con las tablas de Docling; si no se puede, entra el fallback con `pdfplumber`, que convierte cada clase en una frase natural (ej. *"Lunes 17:10-18:00: Desarrollo WEB Integral (edificio D, aula DM; profesor IPM)"*). Así el LLM responde con precisión sin tener que interpretar una cuadrícula.

En todos los casos los vectores se guardan en ChromaDB con metadatos de origen (`fuente`, `grupo`, `tipo`; cuando Docling lo provee también `pagina`, `seccion` y `parser`). La re-subida de un mismo archivo es idempotente: reemplaza sus chunks anteriores. La respuesta del `/health` muestra cuántos chunks están disponibles.

> ⚠️ Si cambias `EMBEDDING_PROVIDER` o actualizas el pipeline de parsing/chunking, debes borrar `chroma_db/` y re-indexar — los vectores y chunks anteriores no son comparables.

> ℹ️ La primera ejecución de Docling/EasyOCR puede tardar más porque descarga modelos locales. Después queda cacheado en la máquina o imagen correspondiente.

---

## 📦 Comandos Disponibles

| Comando | Descripción |
|---|---|
| `docker compose up -d` | Inicia el servicio en segundo plano |
| `docker compose logs -f api` | Ver logs de la API en tiempo real |
| `docker compose down` | Detiene y elimina los contenedores |
| `uvicorn main:app --reload` | Levanta la API en modo desarrollo con hot-reload |
| `python scripts/ingest.py <carpeta>` | Indexa por lote los PDFs de una carpeta |
| `curl -X POST http://localhost:8000/ingest -F "archivo=@<ruta.pdf>"` | Sube e indexa un PDF individual |
| `curl http://localhost:8000/health` | Verifica estado y chunks disponibles |

---

## 🌐 Endpoints

| Método | Ruta | Descripción |
|---|---|---|
| `GET` | `/health` | Estado del servicio y chunks disponibles |
| `POST` | `/chat` | Enviar una pregunta al asistente (contrato público, no el interno que consume el backend) |
| `POST` | `/ingest` | Subir un PDF e indexarlo en la base vectorial (`multipart/form-data`, campo `archivo`) |
| `GET` | `/debug/rag?pregunta=...` | Ver qué chunks recupera el RAG para una pregunta |
| `GET` | `/docs` | Documentación interactiva Swagger |

> ⚠️ **`POST /ask`** — el contrato interno que consume `space-ai-api-core` (`docs/03_API_MINIMUM_CONTRACTS.txt` §11: `question`/`userContext`/`source` → `answer`/`confidence`/`sources`) **no está disponible en `main`** todavía. Existe implementado en la branch `feat/knowledge-rag` (`app/routers/ask.py`) sin mergear. Hasta que se mergee, el backend .NET no tiene un endpoint real que llamar para ese contrato.

**Ejemplo de consulta:**

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"pregunta": "¿Qué carreras hay en la universidad?"}'
```

---

## 🗂️ Estructura del Proyecto

```
space-ai-api-ms-ai/
├── main.py                    → Entrypoint uvicorn (invocado por Dockerfile y README)
├── app/
│   ├── main.py                → Inicialización de FastAPI y registro de routers
│   ├── config.py              → Configuración vía Pydantic Settings (.env)
│   ├── routers/
│   │   ├── chat.py            → Endpoint POST /chat (consulta al asistente)
│   │   ├── ingest.py          → Endpoint POST /ingest (subida e indexado de PDFs)
│   │   └── health.py          → Endpoints GET /health y GET /debug/rag
│   └── services/
│       ├── llm.py             → Factory multi-proveedor (Groq / OpenAI / Gemini)
│       ├── embeddings.py      → Factory de embeddings (local / OpenAI / Gemini)
│       ├── ingest.py          → Pipeline de indexado (PDF → Docling → chunks → ChromaDB)
│       └── rag.py             → Búsqueda semántica con filtro por grupo
├── scripts/
│   └── ingest.py              → CLI de indexado por lote (mismo pipeline que el servicio)
├── tests/                     → pytest (ver sección de calidad más abajo)
├── docs/                      → PDFs institucionales (gitignored, agregar manualmente)
├── chroma_db/                 → Base vectorial persistente (gitignored)
├── .github/workflows/         → CI: pytest + ruff + bandit en cada push/PR
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml             → Config de ruff/bandit/pytest
├── requirements.txt
└── .env.example
```

---

## ✅ Calidad y CI

En cada push/PR (`.github/workflows/lint-security.yml`) corren automáticamente:

| Herramienta | Qué revisa |
|---|---|
| `pytest` | Suite de pruebas en `tests/` |
| `ruff check` | Linting |
| `ruff format --check` | Formato |
| `bandit` | Análisis estático de seguridad |

Localmente:
```bash
pytest
ruff check .
ruff format --check .
bandit -r app/
```

---

*Desarrollado con innovación y dedicación por Lattice Systems.*
