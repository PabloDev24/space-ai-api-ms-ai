# RAG — Log de mejoras y decisiones

Registro de las mejoras al microservicio RAG y el porqué de cada decisión. La regla de
trabajo es **medir antes de asumir**: cada cambio se valida con el harness de evaluación
(`eval/`, `scripts/evaluate.py`) comparando antes/después, no a ojo.

Rama de trabajo: `feat/rag-eval-and-freshness`.

## Roadmap

| # | Pieza | Estado | Resultado |
|---|---|---|---|
| 1 | Harness de evaluación | ✅ | Golden set + `evaluate.py`; línea base congelada |
| 2 | Embedding multilingüe | ↩️ revertido | Regresó retrieval; se mantiene `bge-small-en` |
| 3 | Frescura del índice (cold-start) | ✅ | Índice queryable tras ingesta sin reiniciar |
| 4 | Reranker cross-encoder | ✅ | pass 14/15→15/15; confianza calibrada |
| 5 | Híbrida BM25 | ❌ descartada | Sin ganancia medible; reranker ya cubre exactas |
| 6 | Tablas de datos fila→frase | ✅ | Filas recuperables; harness 15/15 |
| 8 | Endurecer el harness | ✅ | 29 preguntas; desatura la métrica (24/25) |
| 7 | Evaluar `multilingual-e5-large` | ⏳ pendiente | Ya medible tras #8 |
| 5 | Híbrida BM25 | ⏳ reconsiderar | Ya medible tras #8 |

## Detalle por pieza

### 1. Harness de evaluación
Golden set de preguntas reales del campus (`eval/golden_set.jsonl`) + `scripts/evaluate.py`
que mide recuperación (source hit-rate, keyword recall) offline y determinista, con
`--e2e` opcional que llama al LLM. Permite comparar cualquier cambio con datos.
**Línea base:** pass 14/15 (93%), source_hit@3 100%.

### 2. Embedding multilingüe (revertido)
Se probó `paraphrase-multilingual-MiniLM-L12-v2` porque los docs son español y el default
`bge-small-en` es inglés. **Regresó:** source_hit 100%→87%, pass 93%→73%.
**Causa raíz:** el problema era el TIPO de modelo, no el idioma — MiniLM-paraphrase es de
similitud simétrica, no de retrieval; `bge-small-en`, aunque inglés, es retrieval-tuned y
rankea mejor en este corpus chico y distintivo. Se revirtió. Ganó calibración pero perdió
ranking, que es lo que importa. Ver [candidato pendiente #7].

### 3. Frescura del índice (cold-start)
Bug: `rag.py` cargaba el vectorstore una vez al importar y quedaba `None` para siempre si
`chroma_db/` no existía al arrancar → documentos ingestados después eran invisibles hasta
reiniciar. Fix: no exigir que el índice exista (colección vacía es válida) + getter
perezoso con reintento. Test de regresión reproduce el escenario de demo en vivo.

### 4. Reranker cross-encoder
Recupera un pool denso amplio (20) y lo reordena con `jina-reranker-v2-base-multilingual`
(fastembed, sin dep nueva), quedándose con el top 5. **Medido:** pass 14/15→15/15
(resuelve `est-politicas`), y confianza de preguntas fuera de alcance 0.55→<0.12 (el
`confidence` de `/ask` pasa a ser señal fiable). **Latencia:** ~630 ms/query extra en CPU
(modelo caliente); aceptable porque el LLM domina el end-to-end. Configurable y con
rollback trivial (`rerank_enabled=False`). Ante fallo, cae al orden denso (no rompe).

### 5. Híbrida BM25 (descartada)
Se evaluó antes de construir: consultas adversariales de tokens exactos (siglas de
profesor, códigos de aula) ya se recuperan al top-1 con el reranker (el cross-encoder hace
match léxico fuerte). Harness en 15/15. Añadir `rank_bm25` (dep nueva) + índice BM25 +
staleness, para cero ganancia medible, choca con demo-first y "no deps sin justificación".
**Reconsiderar solo si el harness endurecido muestra un fallo real.**

### 6. Tablas de datos fila→frase
Las tablas no-horario se embebían como un bloque coarse que perdía la relación fila↔columna.
Ahora cada fila de una tabla de datos se emite también como una frase ("col: valor; …"),
recuperable de forma independiente. **Aditivo** (no reemplaza el chunk de Docling): el
primer intento que sí reemplazaba regresó `est-politicas` (las "tablas" de PODEPE04 son
boilerplate de control cuyo título de sección hacía falta para recuperar) — el harness lo
cazó y se pasó a aditivo. **Medido:** harness 15/15, keyword_recall 93%→100%; test de
integración con fixture real recupera la fila objetivo en top-1. Fixture generado local;
sin dep nueva al proyecto.

### 8. Endurecer el harness
El golden set directo llegó a 15/15 con el reranker → saturado, no discriminaba, así que
ningún cambio de embedding/retrieval era medible. Se amplió a **29 preguntas**
(parafraseadas, multi-restricción, cross-documento, distractoras, negativas plausibles) y
`evaluate.py` acepta `fuente_esperada` como lista (cross-doc). **Resultado:** 24/25 (96%),
con un fallo de diagnóstico útil: `hard-901-sabado-materias` (pregunta multi-respuesta;
el top-5 no cubre la segunda materia → debilidad de cobertura/diversidad). Negativas
calibradas 0.07–0.28. Ahora e5 y BM25 son evaluables.

### 7. Evaluar `multilingual-e5-large` (pendiente, ya medible)
Candidato correcto para el embedding: multilingüe Y retrieval-tuned (2.24 GB, exige
prefijos `query:`/`passage:` → wrapper de código). Adoptar solo si mejora los casos
difíciles del harness endurecido sin regresar los fáciles ni añadir latencia inaceptable.

## Cómo medir un cambio

```bash
.venv/bin/python scripts/evaluate.py --k 5 --json eval/despues.json   # antes y después
.venv/bin/python -m pytest -q
.venv/bin/ruff check app/ tests/ scripts/
```
Al cambiar embedding o el chunking, **re-indexar**: `rm -rf chroma_db/ && python scripts/ingest.py docs/`.
