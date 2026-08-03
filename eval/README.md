# Evaluación del RAG (golden set)

Mide objetivamente la calidad del microservicio para poder comparar cambios
(embedding, reranker, chunking) con datos, no a ojo.

## Qué hay

- `golden_set.jsonl` — preguntas reales del campus con la fuente y las keywords
  que la recuperación debe traer. Una pregunta por línea (JSON).
- `../scripts/evaluate.py` — corre las preguntas contra el índice actual y reporta
  métricas de recuperación (gratis, determinista) y, con `--e2e`, la respuesta del LLM.

## Uso

Desde la raíz del microservicio, con el venv:

```bash
.venv/bin/python scripts/evaluate.py            # solo recuperación (gratis)
.venv/bin/python scripts/evaluate.py --k 3      # profundidad realista
.venv/bin/python scripts/evaluate.py --e2e      # además llama al LLM (consume API)
.venv/bin/python scripts/evaluate.py --json baseline.json   # vuelca crudos
```

Sale con código 1 si algún caso puntuado falla (sirve para CI).

## Métricas

- `source_hit@k` — la fuente correcta está entre los k chunks recuperados.
- `keyword_hit` — todas las keywords esperadas aparecen en el contexto (sin acentos, minúsculas).
- `pass` = source_hit AND keyword_hit.
- Las preguntas `fuera_alcance` no se puntúan; se reporta su `score_max` para vigilar
  la calibración de confianza (idealmente bajo). En `--e2e` se valida que remitan
  al módulo de informes en vez de inventar.

## Cómo comparar un cambio

1. `evaluate.py --json antes.json` sobre el índice actual.
2. Aplica el cambio (ej. embedding multilingüe) y **re-indexa** (`scripts/ingest.py docs/`).
3. `evaluate.py --json despues.json` y compara los resúmenes.

## Líneas base

### Denso puro, sin reranker (2026-08, `bge-small-en`, 224 chunks) — histórica
- `pass` 14/15 (93%), `source_hit@3` 15/15 (100%), `keyword_recall` 93%.
- Falla: `est-politicas` (fuente correcta, frase exacta fuera de top-k).
- Calibración pobre: fuera de alcance puntuaba ~0.53–0.56.

### Con reranker + tablas fila→frase, harness endurecido (2026-08, 239 chunks) — actual
Set ampliado a 29 preguntas (parafraseadas, multi-restricción, cross-documento,
distractoras, negativas plausibles). Comando: `evaluate.py --k 5`.
- `pass` **24/25 (96%)**, `source_hit@5` 25/25 (100%), `keyword_recall` 98%.
- Fallo de diagnóstico: `hard-901-sabado-materias` — en una pregunta multi-respuesta el
  top-5 se llena con slots de una sola materia y no cubre la segunda (debilidad de
  **cobertura/diversidad** en recuperación). Es la señal medible para juzgar e5/BM25.
- Calibración sana: negativas puntúan 0.07–0.28 (bien por debajo del rango relevante).

## Al añadir documentos nuevos

Agrega preguntas verificables al `golden_set.jsonl` para cubrirlos. El valor del
harness crece con la cobertura del corpus real.
