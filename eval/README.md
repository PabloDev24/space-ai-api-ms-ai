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

## Línea base (2026-08, embedding local `bge-small-en`, 224 chunks)

- `pass` 14/15 (93%), `source_hit@3` 15/15 (100%), `keyword_recall` 93%.
- Falla: `est-politicas` (fuente correcta, frase exacta fuera de top-k).
- Alerta de calibración: preguntas fuera de alcance puntúan ~0.53–0.56 de relevancia
  (el embedding inglés no separa bien relevante de ruido; mejora esperada con multilingüe).

## Al añadir documentos nuevos

Agrega preguntas verificables al `golden_set.jsonl` para cubrirlos. El valor del
harness crece con la cobertura del corpus real.
