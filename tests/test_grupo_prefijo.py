"""
Regresión del bug de colisión de código de grupo.

"901" a secas es ambiguo entre grupos de carreras distintas (IDGS901, LGA901,
LGDT901...). La ingesta debe conservar el prefijo en la metadata para no
cruzar documentos, pero NO debe repetirlo en el texto embebido de cada frase
de horario (cambia el embedding y desestabiliza el ranking fino ya ajustado
entre chunks por franja y por día). La búsqueda, del lado de la pregunta,
debe usar el código completo si la pregunta lo trae, y una lista de
candidatos si solo trae dígitos sueltos.
"""

from app.services import ingest, rag


def test_extraer_grupo_conserva_prefijo_para_metadata():
    assert ingest._extraer_grupo("IDGS901.pdf") == "IDGS901"
    assert ingest._extraer_grupo("Aulas Edificio B UTL.pdf") == ""


def test_etiqueta_grupo_para_texto_usa_solo_digitos():
    assert ingest._etiqueta_grupo("IDGS901") == "901"
    assert ingest._etiqueta_grupo("901") == "901"


def test_horario_a_frases_no_repite_prefijo_en_el_texto():
    tabla = [
        ["Horas", "Lunes", "Sabado"],
        ["08:00-08:50", "Algebra\nD, DM\nIPM", "-"],
    ]
    frases = ingest._horario_a_frases(tabla, "IDGS901")
    assert frases[0].startswith("Grupo 901.")
    assert "IDGS" not in frases[0]


def test_detectar_grupo_con_prefijo_es_exacto():
    assert rag._detectar_grupo("¿en qué aula está el grupo LGA 901?") == "LGA901"


def test_filtro_grupo_bare_incluye_todos_los_prefijos_candidatos():
    filtro = rag._filtro_grupo("¿qué clase tiene el grupo 901 el sábado?")
    candidatos = filtro["grupo"]["$in"]
    assert "901" in candidatos
    assert "IDGS901" in candidatos
    assert "LGA901" in candidatos


def test_filtro_grupo_con_prefijo_es_igualdad_exacta():
    assert rag._filtro_grupo("¿en qué aula está el grupo LGA 901?") == {"grupo": "LGA901"}
