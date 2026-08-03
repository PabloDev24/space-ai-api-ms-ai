"""
Convierte documentos .md transcritos (fotos/capturas) a PDF para trazabilidad
y para que scripts/ingest.py los pueda procesar como cualquier otro PDF.

`multi_cell()` de fpdf2 no regresa el cursor X al margen izquierdo cuando el
texto cabe en una sola línea (solo lo hace al envolver). Bullets consecutivos
sin línea en blanco entre ellos heredan la posición X del bullet anterior y
quedan con casi nada de ancho disponible: el texto se trunca a un puñado de
caracteres por línea. Se detectó al revisar un PDF generado así — perdía
contenido real que ya estaba indexado en el RAG. Por eso cada línea hace
`set_x(l_margin)` explícito después de escribirse.

Cada archivo se genera además en un subproceso Python nuevo (uno por PDF)
por aislamiento simple, aunque el bug real era el de arriba, no el de
reusar `add_font()` entre instancias.

Uso:
    python scripts/md_a_pdf.py <carpeta_con_md> <carpeta_salida> <mapa.json>

mapa.json: {"archivo.md": "Nombre de salida.pdf", ...}
"""

import subprocess  # nosec B404 -- aislar cada conversión en su proceso, ver docstring
import sys
from pathlib import Path

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

_WORKER = """
import sys
from fpdf import FPDF

md_path, pdf_path, font, font_b = sys.argv[1:5]

pdf = FPDF()
pdf.set_auto_page_break(auto=True, margin=15)
pdf.add_font("dejavu", "", font)
pdf.add_font("dejavu", "B", font_b)
pdf.add_page()
pdf.set_font("dejavu", size=11)
epw = pdf.w - pdf.l_margin - pdf.r_margin

for raw in open(md_path, encoding="utf-8").read().splitlines():
    line = raw.rstrip()
    if not line:
        pdf.ln(3)
        continue
    if line.startswith("### "):
        pdf.set_font("dejavu", "B", 11)
        pdf.multi_cell(epw, 6, line[4:])
        pdf.set_font("dejavu", size=11)
    elif line.startswith("## "):
        pdf.ln(1)
        pdf.set_font("dejavu", "B", 13)
        pdf.multi_cell(epw, 7, line[3:])
        pdf.set_font("dejavu", size=11)
    elif line.startswith("# "):
        pdf.set_font("dejavu", "B", 15)
        pdf.multi_cell(epw, 8, line[2:])
        pdf.set_font("dejavu", size=11)
    else:
        pdf.multi_cell(epw, 6, line)
    pdf.set_x(pdf.l_margin)

pdf.output(pdf_path)
"""


def convertir(md_path: Path, pdf_path: Path) -> None:
    # Lista fija (sin shell); los argumentos son rutas locales de
    # docs/_fotos/_transcrito, no entrada de usuario/red.
    subprocess.run(  # nosec B603
        [sys.executable, "-c", _WORKER, str(md_path), str(pdf_path), FONT, FONT_B],
        check=True,
    )


def main():
    import json

    if len(sys.argv) != 4:
        print("Uso: python scripts/md_a_pdf.py <carpeta_md> <carpeta_salida> <mapa.json>")
        sys.exit(1)

    src, out, mapa_path = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    mapa = json.loads(mapa_path.read_text(encoding="utf-8"))

    for md_name, pdf_name in mapa.items():
        convertir(src / md_name, out / pdf_name)
        print("OK", pdf_name)


if __name__ == "__main__":
    main()
