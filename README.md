# Space AI · ms-ai 🤖

![Python](https://img.shields.io/badge/Python%203.14-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Google Gemini](https://img.shields.io/badge/Gemini%202.5%20Flash-4285F4?style=for-the-badge&logo=google&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-FF6B35?style=for-the-badge&logo=databricks&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)

Microservicio de inteligencia artificial del ecosistema **Space AI**, diseñado como el **Punto de Información Inteligente** de la Universidad Tecnológica de León. Responde preguntas institucionales de forma natural mediante una arquitectura RAG sobre documentos PDF.

---

## 👨‍💻 Desarrollador y Equipo

- **Desarrollador Principal:** Daniel Ojeda
- **Equipo de Desarrollo:** Lattice Systems

---

## 🚀 Tecnologías y Arquitectura

Este microservicio implementa una **arquitectura RAG (Retrieval-Augmented Generation)** que combina búsqueda semántica sobre documentos con un modelo de lenguaje de última generación:

- **Framework:** FastAPI 0.136 con Python 3.14.
- **LLM:** Google Gemini 2.5 Flash vía `google-generativeai`.
- **Base de conocimiento:** ChromaDB como vector store persistente.
- **Ingesta:** Pipeline PDF → chunks → embeddings con `pypdf`.
- **Configuración:** Pydantic Settings v2 con soporte `.env`.
- **Contenedores:** Docker & Docker Compose.
- **Monitoreo:** Sentry SDK integrado.

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

Copia el archivo de ejemplo y configura tu API Key de Gemini:

```bash
cp .env.example .env
```

| Variable | Requerida | Default | Descripción |
|---|---|---|---|
| `GEMINI_API_KEY` | ✅ | — | API Key de Google AI Studio |
| `CHROMA_PATH` | ❌ | `chroma_db` | Ruta de la base vectorial persistente |
| `COLLECTION_NAME` | ❌ | `documentos` | Nombre de la colección en ChromaDB |
| `N_RESULTADOS` | ❌ | `5` | Chunks recuperados por consulta |
| `MODELO_GEMINI` | ❌ | `gemini-2.5-flash` | Modelo Gemini a usar |

---

### 3. Método A: Docker (Recomendado)

La forma más rápida para levantar el servicio con la base de conocimiento persistente:

```bash
# Levantar el servicio
docker compose up -d

# Verificar que está corriendo
curl http://localhost:8000/health
```

---

### 4. Método B: Ejecución local

Ideal para desarrollo activo y depuración:

```bash
# 1. Crear entorno virtual e instalar dependencias
python -m venv .venv
source .venv/bin/activate      # Linux/Mac
.venv\Scripts\activate         # Windows

pip install -r requirements.txt

# 2. Levantar la API
uvicorn main:app --reload
```

> ℹ️ **Nota:** En ambos métodos, sin documentos indexados el servicio responde usando solo el conocimiento del modelo Gemini, sin contexto institucional. Consulta la sección [Indexar documentos](#-indexar-documentos-rag) para cargar tu base de conocimiento.

---

### 5. Verificar el estado

Abre **[http://localhost:8000/docs](http://localhost:8000/docs)** para explorar la documentación interactiva de la API (Swagger UI).

---

## 📚 Indexar Documentos (RAG)

El microservicio enriquece sus respuestas con información de documentos PDF institucionales. Para indexarlos:

```bash
# Coloca tus PDFs en una carpeta (ej. docs/) y ejecuta:
python scripts/ingest.py docs/
```

El script extrae el texto de cada PDF, lo divide en chunks con solapamiento y los almacena en ChromaDB. La respuesta del `/health` muestra cuántos chunks están disponibles.

---

## 📦 Comandos Disponibles

| Comando | Descripción |
|---|---|
| `docker compose up -d` | Inicia el servicio en segundo plano |
| `docker compose logs -f api` | Ver logs de la API en tiempo real |
| `docker compose down` | Detiene y elimina los contenedores |
| `uvicorn main:app --reload` | Levanta la API en modo desarrollo con hot-reload |
| `python scripts/ingest.py <carpeta>` | Indexa PDFs en la base de conocimiento |
| `curl http://localhost:8000/health` | Verifica estado y chunks disponibles |

---

## 🌐 Endpoints

| Método | Ruta | Descripción |
|---|---|---|
| `GET` | `/health` | Estado del servicio y base de conocimiento |
| `POST` | `/chat` | Enviar una pregunta al asistente |
| `GET` | `/docs` | Documentación interactiva Swagger |

**Ejemplo de consulta:**

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"pregunta": "¿Dónde está el edificio de rectoría?"}'
```

---

## 🗂️ Estructura del Proyecto

```
space-ai-api-ms-ai/
├── app/
│   ├── main.py             → Inicialización de FastAPI y registro de routers
│   ├── config.py           → Configuración vía Pydantic Settings (.env)
│   ├── routers/
│   │   ├── chat.py         → Endpoint POST /chat (consulta al asistente)
│   │   └── health.py       → Endpoint GET /health (estado del servicio)
│   └── services/
│       ├── llm.py          → Cliente Gemini con instrucción de sistema
│       └── rag.py          → Búsqueda semántica en ChromaDB
├── scripts/
│   └── ingest.py           → Pipeline de indexación PDF → ChromaDB
├── docs/                   → Carpeta de PDFs institucionales (gitignored)
├── chroma_db/              → Base vectorial persistente (gitignored)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

---

*Desarrollado con innovación y dedicación por Lattice Systems.*
