"""
Evaluación del RAG contra un set dorado de preguntas (eval/golden_set.jsonl).

Mide la calidad SIN llamar al LLM por defecto: solo la fase de recuperación,
que es determinista y gratuita. Así puedes comparar objetivamente un cambio
(ej. embedding multilingüe, reranker) corriendo el mismo comando antes y después.

Métricas de recuperación (preguntas con fuente esperada):
  - source_hit@k : la fuente correcta aparece entre los k chunks recuperados.
  - keyword_hit  : TODAS las keywords esperadas aparecen en el contexto recuperado.
  - keyword_recall: fracción de keywords encontradas (parcial).
  - pass = source_hit AND keyword_hit.

Preguntas 'fuera_alcance' (espera_sin_contexto=true) no se puntúan en recuperación
(el índice siempre devuelve algo); se reporta su score máximo para inspección, y
en modo --e2e se valida que la respuesta remita al módulo de informes.

Uso:
    python scripts/evaluate.py                 # solo recuperación (gratis)
    python scripts/evaluate.py --e2e           # además llama al LLM (consume API)
    python scripts/evaluate.py --k 8           # override de top-k
    python scripts/evaluate.py --json out.json # vuelca resultados crudos

Correr SIEMPRE desde la raíz del microservicio (usa el chroma_db de ese cwd).
"""

from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

# Permite ejecutar como `python scripts/evaluate.py` desde la raíz del proyecto.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.rag import buscar_contexto_con_metadata  # noqa: E402

GOLDEN_PATH = Path(__file__).resolve().parent.parent / "eval" / "golden_set.jsonl"


def _norm(texto: str) -> str:
    """Minúsculas sin acentos, para comparar keywords de forma robusta."""
    nfkd = unicodedata.normalize("NFKD", texto.lower())
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def _cargar_golden(path: Path) -> list[dict]:
    if not path.exists():
        sys.exit(f"No existe el set dorado: {path}")
    casos = []
    for linea in path.read_text(encoding="utf-8").splitlines():
        linea = linea.strip()
        if linea:
            casos.append(json.loads(linea))
    return casos


def _evaluar_caso(caso: dict, k: int) -> dict:
    resultados = buscar_contexto_con_metadata(caso["pregunta"])
    resultados = resultados[:k]

    contexto = _norm("\n".join(c for c, _, _ in resultados))
    fuentes = [m.get("fuente", "") for _, m, _ in resultados]
    max_score = max((s for _, _, s in resultados), default=0.0)

    espera_sin_contexto = caso.get("espera_sin_contexto", False)
    keywords = caso.get("keywords", [])
    kw_encontradas = [kw for kw in keywords if _norm(kw) in contexto]
    keyword_recall = len(kw_encontradas) / len(keywords) if keywords else 1.0

    fuente_esperada = caso.get("fuente_esperada")
    source_hit = fuente_esperada in fuentes if fuente_esperada else None
    keyword_hit = len(kw_encontradas) == len(keywords)

    if espera_sin_contexto:
        # No se puntúa en recuperación; solo se reporta el score máximo.
        passed = None
    else:
        passed = bool(source_hit) and keyword_hit

    return {
        "id": caso["id"],
        "categoria": caso["categoria"],
        "pregunta": caso["pregunta"],
        "source_hit": source_hit,
        "keyword_hit": keyword_hit,
        "keyword_recall": round(keyword_recall, 2),
        "kw_faltantes": [kw for kw in keywords if kw not in kw_encontradas],
        "max_score": round(max_score, 3),
        "fuentes_top": fuentes,
        "espera_sin_contexto": espera_sin_contexto,
        "passed": passed,
    }


def _evaluar_e2e(caso: dict, resultado: dict) -> dict:
    """Llama al LLM y valida la respuesta. Solo en modo --e2e."""
    from app.services.llm import generate

    contexto_docs = buscar_contexto_con_metadata(caso["pregunta"])
    contexto = "\n\n".join(c for c, _, _ in contexto_docs)
    prompt = f"Información relacionada:\n{contexto}\n\nPregunta de la persona: {caso['pregunta']}"
    try:
        respuesta = generate(prompt)
    except Exception as exc:  # noqa: BLE001
        resultado["e2e_error"] = str(exc)
        resultado["e2e_passed"] = False
        return resultado

    resp_norm = _norm(respuesta)
    if caso.get("espera_sin_contexto"):
        # Debe remitir a informes / control escolar, no inventar.
        resultado["e2e_passed"] = any(t in resp_norm for t in ("informes", "control escolar"))
    else:
        resultado["e2e_passed"] = all(_norm(kw) in resp_norm for kw in caso.get("keywords", []))
    resultado["respuesta"] = respuesta.strip()[:200]
    return resultado


def main() -> int:
    parser = argparse.ArgumentParser(description="Evalúa el RAG contra el set dorado.")
    parser.add_argument("--k", type=int, default=8, help="top-k de recuperación (default 8)")
    parser.add_argument("--e2e", action="store_true", help="además llama al LLM (consume API)")
    parser.add_argument("--json", type=str, default="", help="vuelca resultados crudos a un archivo")
    args = parser.parse_args()

    casos = _cargar_golden(GOLDEN_PATH)
    resultados = []
    for caso in casos:
        r = _evaluar_caso(caso, args.k)
        if args.e2e:
            r = _evaluar_e2e(caso, r)
        resultados.append(r)

    # --- Reporte por caso ---
    print(f"\n=== Evaluación RAG  (k={args.k}, casos={len(casos)}) ===\n")
    for r in resultados:
        if r["passed"] is None:
            marca = "·"  # no puntuado (fuera de alcance)
        else:
            marca = "PASS" if r["passed"] else "FAIL"
        extra = ""
        if r["passed"] is False:
            faltan = []
            if r["source_hit"] is False:
                faltan.append("fuente")
            if not r["keyword_hit"]:
                faltan.append(f"kw={r['kw_faltantes']}")
            extra = "  <- falta: " + ", ".join(faltan)
        elif r["passed"] is None:
            extra = f"  score_max={r['max_score']}"
        print(f"  [{marca:>4}] {r['id']:<28} {r['categoria']:<14}{extra}")
        if args.e2e and "e2e_passed" in r:
            e2e = "PASS" if r["e2e_passed"] else "FAIL"
            print(f"         e2e[{e2e}] {r.get('respuesta', r.get('e2e_error', ''))}")

    # --- Agregados ---
    puntuados = [r for r in resultados if r["passed"] is not None]
    passed = sum(1 for r in puntuados if r["passed"])
    total = len(puntuados)
    src = sum(1 for r in puntuados if r["source_hit"])
    kw = sum(1 for r in puntuados if r["keyword_hit"])
    recall_prom = sum(r["keyword_recall"] for r in puntuados) / total if total else 0.0

    print("\n=== Resumen recuperación ===")
    print(f"  pass (fuente+keywords) : {passed}/{total}  ({passed / total:.0%})" if total else "  sin casos")
    print(f"  source_hit@{args.k:<12}: {src}/{total}  ({src / total:.0%})" if total else "")
    print(f"  keyword_hit (todas)    : {kw}/{total}  ({kw / total:.0%})" if total else "")
    print(f"  keyword_recall (prom)  : {recall_prom:.0%}")

    # Por categoría
    por_cat: dict[str, list] = defaultdict(list)
    for r in puntuados:
        por_cat[r["categoria"]].append(r)
    print("\n  Por categoría:")
    for cat, rs in sorted(por_cat.items()):
        p = sum(1 for r in rs if r["passed"])
        print(f"    {cat:<14} {p}/{len(rs)}")

    if args.e2e:
        e2e_casos = [r for r in resultados if "e2e_passed" in r]
        e2e_pass = sum(1 for r in e2e_casos if r["e2e_passed"])
        print(f"\n=== Resumen e2e (LLM) === {e2e_pass}/{len(e2e_casos)}  ({e2e_pass / len(e2e_casos):.0%})")

    if args.json:
        Path(args.json).write_text(json.dumps(resultados, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nResultados crudos -> {args.json}")

    # Código de salida: 0 si todo pasa, 1 si hay fallos (útil para CI).
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
