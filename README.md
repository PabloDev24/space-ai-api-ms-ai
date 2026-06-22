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
- **Pipeline RAG:** LangChain para chunking inteligente por estructura de documento, con fragmentación por tokens (~512 tokens, 50 de solapamiento) para respetar la ventana del LLM.
- **Parseo de PDFs:** `pymupdf4llm` convierte cualquier PDF a Markdown preservando tablas y secciones.
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
| `PROVEEDOR` | ✅ | `gemini` | Proveedor LLM: `groq` \| `openai` \| `gemini` |
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

En ambos casos el pipeline convierte cada PDF a Markdown inteligente, lo fragmenta por secciones y por tamaño (~512 tokens con 50 de solapamiento, medido con `tiktoken` para no rebasar la ventana del LLM) y guarda los vectores en ChromaDB con metadatos de origen (`fuente`, `grupo`). La re-subida de un mismo archivo es idempotente: reemplaza sus chunks anteriores. La respuesta del `/health` muestra cuántos chunks están disponibles.

> ⚠️ Si cambias `EMBEDDING_PROVIDER`, debes borrar `chroma_db/` y re-indexar — los vectores de distintos modelos no son comparables.

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
| `POST` | `/chat` | Enviar una pregunta al asistente |
| `POST` | `/ingest` | Subir un PDF e indexarlo en la base vectorial (`multipart/form-data`, campo `archivo`) |
| `GET` | `/debug/rag?pregunta=...` | Ver qué chunks recupera el RAG para una pregunta |
| `GET` | `/docs` | Documentación interactiva Swagger |

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
│       ├── ingest.py          → Pipeline de indexado (PDF → Markdown → chunks → ChromaDB)
│       └── rag.py             → Búsqueda semántica con filtro por grupo
├── scripts/
│   └── ingest.py              → CLI de indexado por lote (mismo pipeline que el servicio)
├── docs/                      → PDFs institucionales (gitignored, agregar manualmente)
├── chroma_db/                 → Base vectorial persistente (gitignored)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

---

*Desarrollado con innovación y dedicación por Lattice Systems.*
