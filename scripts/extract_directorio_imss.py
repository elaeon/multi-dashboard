"""Extract IMSS facility directory from PDF to CSV.

Source PDF: data/imss/directorio_imss.pdf (1,284 pages, ~8,000+ facilities)

Column x-boundaries (PDF points): Nombre <157, Tipo 157-242, Dirección 243-464, Horario 465+

Tipo subtype words (CLÍNICA, HOSPITAL, etc.) wrap to x<157 on lines where the
previous row had no x<157 content — classified as Tipo, not Nombre.
"""

import argparse
import csv
import re
from pathlib import Path

import pdfplumber

PDF_PATH = Path("data/imss/directorio_imss.pdf")
DEFAULT_OUT = Path("data/imss/directorio_imss.csv")
HEADER = ("Nombre de la Unidad", "Tipo de Unidad", "Dirección", "Horario", "Estado")

_CP_RE = re.compile(r"C\.P\.\s*\d{5},\s*(.+)$")

Y_DATA_START = 165   # skip title + column headers
X_TIPO = 157
X_DIR = 243
X_HORARIO = 465
ROW_Y_TOLERANCE = 6   # words within 6pt vertically → same row
RECORD_GAP = 22        # gap >22pt between rows → new record

# Word tuples: (x0, top, text)
_X = 0
_TOP = 1
_TEXT = 2


def parse_args():
    p = argparse.ArgumentParser(description="Extract IMSS facility directory from PDF to CSV")
    p.add_argument("-o", "--output", type=Path, default=DEFAULT_OUT,
                   help=f"Output CSV path (default: {DEFAULT_OUT})")
    p.add_argument("--from-page", type=int, default=1, metavar="N",
                   help="First page to extract, 1-indexed (default: 1)")
    p.add_argument("--to-page", type=int, default=None, metavar="N",
                   help="Last page to extract, 1-indexed inclusive (default: last page)")
    p.add_argument("--append", action="store_true",
                   help="Append rows to existing file; skip header if file already has content")
    return p.parse_args()


def _group_into_rows(words):
    """Yield rows of words; words must be pre-sorted by (top, x0)."""
    current_row = []
    for w in words:
        if current_row and w[_TOP] - current_row[0][_TOP] > ROW_Y_TOLERANCE:
            yield current_row
            current_row = []
        current_row.append(w)
    if current_row:
        yield current_row


def _group_into_records(rows):
    """Yield records (lists of rows) split on y-gaps > RECORD_GAP."""
    current_record = []
    for row in rows:
        if current_record and row[0][_TOP] - current_record[-1][0][_TOP] > RECORD_GAP:
            yield current_record
            current_record = []
        current_record.append(row)
    if current_record:
        yield current_record


def _extract_columns(record_rows):
    nombre, tipo, direccion, horario = [], [], [], []
    prev_had_col_a = False

    for row_words in record_rows:
        row_has_col_a = any(w[_X] < X_TIPO for w in row_words)

        # Words at x<157 go to Nombre if this is the start of Nombre or Nombre
        # was active on the previous row; otherwise they are a Tipo subtype wrap.
        col_a_target = None
        if row_has_col_a:
            col_a_target = nombre if (not nombre or prev_had_col_a) else tipo

        for w in sorted(row_words, key=lambda w: w[_X]):
            x = w[_X]
            if x >= X_HORARIO:
                horario.append(w[_TEXT])
            elif x >= X_DIR:
                direccion.append(w[_TEXT])
            elif x >= X_TIPO:
                tipo.append(w[_TEXT])
            else:
                col_a_target.append(w[_TEXT])

        prev_had_col_a = row_has_col_a

    dir_str = " ".join(direccion)
    m = _CP_RE.search(dir_str)
    estado = m.group(1).strip() if m else ""
    return " ".join(nombre), " ".join(tipo), dir_str, " ".join(horario), estado


def extract_page(page):
    # crop() restricts layout analysis to data area, skipping the repeated header zone.
    # Sort once here so _group_into_rows receives words in (top, x0) order.
    cropped = page.crop((0, Y_DATA_START, page.width, page.height))
    words = sorted(
        ((w["x0"], w["top"], w["text"]) for w in cropped.extract_words()),
        key=lambda w: (w[_TOP], w[_X]),
    )
    for record_rows in _group_into_records(_group_into_rows(words)):
        row = _extract_columns(record_rows)
        if row[0]:  # skip records with empty Nombre (parsing artifacts)
            yield row


def main():
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    write_header = not (args.append and args.output.exists() and args.output.stat().st_size > 0)
    mode = "a" if args.append else "w"

    total = 0
    with pdfplumber.open(PDF_PATH) as pdf:
        pages = pdf.pages[args.from_page - 1 : args.to_page]
        n = len(pages)
        with args.output.open(mode, newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            if write_header:
                writer.writerow(HEADER)
            for i, page in enumerate(pages, args.from_page):
                for row in extract_page(page):
                    writer.writerow(row)
                    total += 1
                if i % 100 == 0:
                    print(f"  page {i}/{args.from_page - 1 + n}  ({total:,} rows)")

    print(f"Done: {total:,} rows → {args.output}")


if __name__ == "__main__":
    main()
