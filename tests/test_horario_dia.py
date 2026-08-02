"""
Tests de la agregación de horario por día (_horario_por_dia_a_frases).

Determinista, sin PDF ni modelos: valida que cada día produce una frase con todas
sus clases y que los días/celdas vacíos se omiten.
"""

from app.services import ingest


def _tabla_mock():
    return [
        ["Horas", "Lunes", "Sabado"],
        ["08:00-09:00", "Mate", "Fisica"],
        ["09:00-10:00", "-", "Quimica"],
    ]


def test_una_frase_por_dia_con_todas_las_clases():
    frases = ingest._horario_por_dia_a_frases(_tabla_mock(), "901")

    # Un día = una frase; ambos días tienen clases.
    assert len(frases) == 2
    por_dia = {f.split("Clases del ")[1].split(":")[0]: f for f in frases}

    # El sábado agrega sus DOS materias en una sola frase (cobertura completa).
    assert "Fisica" in por_dia["Sabado"] and "Quimica" in por_dia["Sabado"]
    assert por_dia["Sabado"].startswith("Grupo 901. Clases del Sabado:")

    # El lunes tiene Mate; la celda vacía ("-") de las 09:00 se omite.
    assert "Mate" in por_dia["Lunes"]
    assert "09:00-10:00" not in por_dia["Lunes"]


def test_salta_dia_sin_clases():
    tabla = [
        ["Horas", "Lunes", "Sabado"],
        ["08:00-09:00", "-", "Fisica"],
    ]
    frases = ingest._horario_por_dia_a_frases(tabla, "901")
    assert len(frases) == 1  # solo sábado; el lunes vacío no genera frase
    assert "Sabado" in frases[0]


def test_tabla_no_horario_devuelve_vacio():
    tabla = [["Modalidad", "Costo"], ["Presencial", "$1,500"]]
    assert ingest._horario_por_dia_a_frases(tabla, "901") == []
