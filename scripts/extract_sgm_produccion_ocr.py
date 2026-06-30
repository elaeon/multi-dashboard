"""
Extrae tablas de producción/volumen minero por entidad federativa del Anuario SGM
en PDFs escaneados (sin capa de texto), usando OCR con Tesseract.

Uso:
  uv run scripts/extract_sgm_produccion_ocr.py \\
      --pdf data/sgm/raw/Anuario_1999.pdf \\
      --out data/sgm/produccion_minera_entidad_1995_1999.csv \\
      --start 31 --end 62 \\
      --years 1995 1996 1997 1998 1999 \\
      --unidad-val "Pesos corrientes"

Prerequisitos:
  sudo apt install tesseract-ocr tesseract-ocr-spa
  uv add pytesseract

Columnas de salida: estado, tabla, categoria, producto, unidad, año, valor
"""

import argparse
import re
import subprocess
import tempfile
from pathlib import Path

import pandas as pd
import pytesseract
from PIL import Image
from pytesseract import Output

# ── Lista ordenada de los 32 estados del capítulo 6 (orden alfabético) ────────
# El estado se asigna por índice de página (más robusto que OCR del nombre,
# que falla en ~30% de páginas: "COAHUILA"→"CHIAPAS", "$.19.-NUEVO LEÓN", etc.)

ESTADOS_1999 = [
    "Aguascalientes", "Baja California", "Baja California Sur", "Campeche",
    "Chiapas", "Chihuahua", "Coahuila", "Colima", "Distrito Federal",
    "Durango", "Guanajuato", "Guerrero", "Hidalgo", "Jalisco",
    "México", "Michoacán", "Morelos", "Nayarit", "Nuevo León",
    "Oaxaca", "Puebla", "Querétaro", "Quintana Roo", "San Luis Potosí",
    "Sinaloa", "Sonora", "Tabasco", "Tamaulipas", "Tlaxcala",
    "Veracruz", "Yucatán", "Zacatecas",
]

# ── Expresiones regulares ─────────────────────────────────────────────────────

# Línea de encabezado de sección ("6.1.-...", "6,15.--..." etc.) — solo para saltar la línea
SECCION_RE = re.compile(r"^[6$][.,]\d+", re.I)

# Notas al pie y artefactos OCR
SKIP_RE = re.compile(
    r"^(Anuario|p/[\s]|\d+/[\s]|Fuente|Nota|Gobierno|N/D\s|N\.D\.\s|\*+|[—_]{3,}|FUENTE)",
    re.I,
)

# Línea de unidad bajo el título de tabla
UNIT_LINE_RE = re.compile(r"^\((Toneladas|[Pp]esos|Miles de pesos|kilogramos)", re.I)

# Tokens OCR que son ruido puro (solo caracteres no alfanuméricos)
_NOISE_RE = re.compile(r"^[^\w\d]+$")


# ── Helpers de parseo ─────────────────────────────────────────────────────────

def clean_product(name: str) -> tuple[str, str | None]:
    """Limpia marcadores de nota y extrae unidad inline."""
    name = name.strip()
    m = re.search(r"\(([^)]+)\)\s*$", name)
    unit = m.group(1).strip() if m else None
    name = re.sub(r"\s*\([^)]+\)\s*$", "", name).strip()
    name = re.sub(r"\*+$", "", name).strip()
    name = re.sub(r"[:]+\s*$", "", name).strip()
    name = re.sub(r"\s+\d+\s*$", "", name).strip()
    name = re.sub(r"\s+\d+/$", "", name).strip()
    # Quitar comillas OCR pegadas al inicio del nombre ("119 → 119)
    name = re.sub(r'^["\'"]+', "", name).strip()
    name = re.sub(r"^Metalicos$", "Metálicos", name, flags=re.I)
    name = re.sub(r"^No [Mm]etalicos$", "No metálicos", name, flags=re.I)
    return name, unit


def parse_num(tokens: list[str]) -> float | None:
    """Une tokens del mismo campo y parsea como float.

    El Anuario 1999 usa espacio como separador de miles ('550 045,65')
    y a veces coma como decimal (OCR confunde punto y coma).
    """
    s = "".join(tokens).replace(" ", "")
    if not s or s == "-" or s.lower() in ("nd", "n.d.", "n/d"):
        return None
    # Coma como separador decimal al final: "550045,65" → "550045.65"
    s = re.sub(r",(\d{1,2})$", r".\1", s)
    # Eliminar comas restantes (separador de miles)
    s = s.replace(",", "")
    # Limpiar prefijo OCR no numérico (ej. '"119' → '119')
    s = re.sub(r"^[^\d\-]+", "", s)
    try:
        return float(s)
    except ValueError:
        return None


def detect_year_header(
    line: list[dict], years: list[str]
) -> tuple[dict[str, float], float, float] | tuple[None, None, None]:
    """Detecta fila de encabezado buscando ≥2 años esperados en la línea."""
    cols: dict[str, float] = {}
    min_x0 = float("inf")
    for w in line:
        m = re.match(r"^(\d{4})", w["text"])
        if m and m.group(1) in years:
            yr = m.group(1)
            cols[yr] = (w["x0"] + w["x1"]) / 2
            min_x0 = min(min_x0, w["x0"])
    if len(cols) < 2:
        return None, None, None
    label_x1 = max(
        (w["x1"] for w in line if w["x0"] < min_x0 and not re.match(r"^\d{4}", w["text"])),
        default=0.0,
    )
    return cols, min_x0, label_x1


def assign_col(word: dict, year_cols: dict[str, float], prod_end: float) -> str | None:
    """Asigna una palabra al año cuyo centro de columna sea más cercano."""
    wx = (word["x0"] + word["x1"]) / 2
    if wx <= prod_end:
        return None
    best, best_d = None, float("inf")
    for yr, cx in year_cols.items():
        d = abs(wx - cx)
        if d < best_d:
            best, best_d = yr, d
    # Tolerancia 120px (columnas anchas en imágenes a 200 DPI)
    return best if best_d < 120 else None


def parse_data_row(
    line: list[dict], year_cols: dict[str, float], prod_end: float
) -> tuple[str, dict[str, list[str]]] | None:
    """Separa el nombre del producto de los valores por columna año."""
    prod_parts: list[str] = []
    col_tokens: dict[str, list[str]] = {yr: [] for yr in year_cols}
    for w in line:
        col = assign_col(w, year_cols, prod_end)
        if col is None:
            prod_parts.append(w["text"])
        else:
            col_tokens[col].append(w["text"])
    if not prod_parts:
        return None
    return " ".join(prod_parts), col_tokens


# ── OCR ───────────────────────────────────────────────────────────────────────

def render_page_lines(pdf_path: Path, page_idx: int, dpi: int = 200) -> list[list[dict]]:
    """Renderiza una página del PDF escaneado y devuelve líneas de palabras con coords.

    Usa pdftoppm para renderizar y pytesseract.image_to_data() para OCR.
    Agrupa palabras usando el agrupamiento interno de Tesseract
    (block_num, par_num, line_num) en vez de y-buckets manuales —
    esto maneja correctamente la variación vertical dentro de una línea impresa.

    Devuelve lista de líneas; cada línea es una lista de dicts {x0, x1, top, text}.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        out_prefix = Path(tmpdir) / "page"
        subprocess.run(
            [
                "pdftoppm",
                "-r", str(dpi),
                "-png",
                "-f", str(page_idx + 1),
                "-l", str(page_idx + 1),
                str(pdf_path),
                str(out_prefix),
            ],
            check=True,
            capture_output=True,
        )
        pngs = sorted(Path(tmpdir).glob("*.png"))
        if not pngs:
            return []
        img = Image.open(pngs[0])

        d = pytesseract.image_to_data(
            img,
            lang="spa",
            output_type=Output.DICT,
            config="--psm 6",
        )

    # Agrupar por línea interna de Tesseract
    line_buckets: dict[tuple, list[dict]] = {}
    for i in range(len(d["text"])):
        text = d["text"][i].strip()
        conf = d["conf"][i]
        if not text:
            continue
        if isinstance(conf, str):
            conf = int(conf)
        if conf < 20:
            continue
        if _NOISE_RE.match(text):
            continue
        key = (d["block_num"][i], d["par_num"][i], d["line_num"][i])
        line_buckets.setdefault(key, []).append({
            "x0": d["left"][i],
            "x1": d["left"][i] + d["width"][i],
            "top": d["top"][i],
            "text": text,
        })

    # Ordenar líneas por y mínimo; palabras dentro de cada línea por x
    lines = []
    for key in sorted(line_buckets, key=lambda k: min(w["top"] for w in line_buckets[k])):
        lines.append(sorted(line_buckets[key], key=lambda w: w["x0"]))
    return lines


# ── CLI y main ────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extrae tablas de producción minera del Anuario SGM (PDF escaneado, via OCR)"
    )
    p.add_argument("--pdf", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--start", type=int, required=True, help="Página inicio 0-indexed")
    p.add_argument("--end", type=int, required=True, help="Página fin 0-indexed (inclusive)")
    p.add_argument("--years", nargs="+", required=True, help="Años a extraer, e.g. 1995 1996 ...")
    p.add_argument("--unidad-val", default="Pesos corrientes",
                   help="Unidad de la tabla de valor (default: 'Pesos corrientes')")
    p.add_argument("--dpi", type=int, default=200, help="DPI para renderizar páginas (default: 200)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    years: list[str] = args.years
    years_set = set(years)

    rows: list[dict] = []
    current_unidad_base: str | None = None

    total_pages = args.end - args.start + 1
    for page_idx in range(args.start, args.end + 1):
        page_num = page_idx - args.start + 1
        print(f"  Página {page_num}/{total_pages} (0-idx {page_idx}) ...", end="\r", flush=True)

        # Estado determinado por índice de página (robusto contra errores OCR en el nombre)
        state_idx = page_idx - args.start
        current_estado = ESTADOS_1999[state_idx] if state_idx < len(ESTADOS_1999) else None
        current_tabla: str | None = None
        current_categoria: str | None = None
        year_cols: dict[str, float] = {}
        prod_end: float = 150.0
        pending_product: str | None = None
        pending_unit: str | None = None

        lines = render_page_lines(args.pdf, page_idx, dpi=args.dpi)

        # Pre-escaneo: detectar year_cols desde cualquier encabezado de años de la página.
        # Esto garantiza que los datos del VOLUMEN se extraigan incluso cuando su propio
        # encabezado "PRODUCTOS 1995..." está ilegible en el OCR (Guerrero, Nuevo León).
        for scan_line in lines:
            detected_pre, yr_min_x0_pre, label_x1_pre = detect_year_header(scan_line, years_set)
            if detected_pre:
                year_cols = detected_pre
                prod_end = min(label_x1_pre + 10, yr_min_x0_pre - 5)
                break

        for line in lines:
            if not line:
                continue
            text = " ".join(w["text"] for w in line).strip()

            if SKIP_RE.match(text) or UNIT_LINE_RE.match(text):
                continue

            # Saltarse líneas de encabezado de sección (estado ya asignado por índice)
            if SECCION_RE.match(text):
                continue

            # Ancla tabla valor — NO limpiar year_cols (misma columna en ambas tablas;
            # evita perder datos cuando el header de la tabla valor está ilegible en OCR).
            if re.search(r"Valor d[eé] la [Pp]roducci[oó]n", text, re.I):
                current_tabla = "valor"
                current_unidad_base = args.unidad_val
                current_categoria = None
                pending_product = None
                continue

            # Ancla tabla volumen/producción (sin limpiar year_cols)
            if (re.search(r"(Volumen de la |)Producci[oó]n [Mm]inera", text, re.I)
                    and "Valor" not in text
                    and "entidad" not in text.lower()
                    and "y Valor" not in text):
                current_tabla = "produccion"
                current_unidad_base = "Toneladas"
                current_categoria = None
                pending_product = None
                continue

            # Encabezado de años — actualizar year_cols si se detecta uno explícito
            detected, yr_min_x0, label_x1 = detect_year_header(line, years_set)
            if detected:
                year_cols = detected
                prod_end = min(label_x1 + 10, yr_min_x0 - 5)
                pending_product = None
                continue

            if not (current_estado and current_tabla and year_cols):
                continue

            result = parse_data_row(line, year_cols, prod_end)
            if not result:
                continue
            prod_raw, col_tokens = result
            prod_raw = prod_raw.strip()

            has_values = any(parse_num(ts) is not None for ts in col_tokens.values())
            has_name = bool(prod_raw)

            # Línea solo con valores (sin nombre): intentar asociar al producto pendiente
            if not has_name and has_values and pending_product is not None:
                prod_clean = pending_product
                unidad = pending_unit if pending_unit else current_unidad_base
                for yr in years:
                    v = parse_num(col_tokens.get(yr, []))
                    if v is not None:
                        rows.append({
                            "estado": current_estado,
                            "tabla": current_tabla,
                            "categoria": current_categoria or "",
                            "producto": prod_clean,
                            "unidad": unidad,
                            "año": int(yr),
                            "valor": v,
                        })
                # No limpiar pending_product: puede haber más valores en líneas siguientes
                continue

            if not has_name:
                continue

            pending_product = None
            pending_unit = None

            # Categoría
            cat_m = re.match(
                r"^(Met[aá]licos|No met[aá]licos|Agregados p[eé]treos)",
                prod_raw, re.I,
            )
            if cat_m:
                cat_label = cat_m.group(1)
                if re.search(r"no met", cat_label, re.I):
                    current_categoria = "No metálicos"
                elif re.search("agregados", cat_label, re.I):
                    current_categoria = "No metálicos"
                else:
                    current_categoria = "Metálicos"
                if not has_values:
                    continue

            prod_clean, unit_override = clean_product(prod_raw)
            if not prod_clean:
                continue
            # Filtrar artefactos OCR: nombres que empiezan con dígito (footnotes),
            # muy cortos, o con puntuación de apertura
            if len(prod_clean) <= 1:
                continue
            if prod_clean[0].isdigit():
                continue
            if prod_clean[0] in "({[":
                continue

            unidad = unit_override if unit_override else current_unidad_base

            if not has_values:
                # Guardar para la siguiente línea que tenga valores
                pending_product = prod_clean
                pending_unit = unit_override
                continue

            for yr in years:
                rows.append({
                    "estado": current_estado,
                    "tabla": current_tabla,
                    "categoria": current_categoria or "",
                    "producto": prod_clean,
                    "unidad": unidad,
                    "año": int(yr),
                    "valor": parse_num(col_tokens.get(yr, [])),
                })

    print()
    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False, encoding="utf-8")

    print(f"Guardado: {args.out}")
    print(f"Filas totales: {len(df):,}")
    print(f"Estados únicos: {df['estado'].nunique()}")
    print(f"Tablas: {sorted(df['tabla'].unique())}")
    print()
    print("Productos únicos por tabla/categoría:")
    print(df.groupby(["tabla", "categoria"])["producto"].nunique().to_string())


if __name__ == "__main__":
    main()
