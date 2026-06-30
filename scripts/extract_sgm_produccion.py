"""
Extrae tablas de producción/volumen minero por entidad federativa del Anuario SGM.

Uso:
  uv run scripts/extract_sgm_produccion.py \\
      --pdf data/sgm/raw/Anuario_2024_Edicion_2025.pdf \\
      --out data/sgm/produccion_minera_entidad_2020_2024.csv \\
      --start 51 --end 96 --years 2020 2021 2022 2023 2024

  uv run scripts/extract_sgm_produccion.py \\
      --pdf data/sgm/raw/Anuario_2019_Edicion_2020.pdf \\
      --out data/sgm/produccion_minera_entidad_2015_2019.csv \\
      --start 58 --end 88 --years 2015 2016 2017 2018 2019 \\
      --unidad-val "Pesos corrientes"

Columnas de salida: estado, tabla, categoria, producto, unidad, año, valor
"""

import argparse
import re
from pathlib import Path

import pandas as pd
import pdfplumber

# "6.1 Aguascalientes" (2024) y "6.1. Aguascalientes" (2019)
SECCION_RE = re.compile(r"^6\.(\d+)\.?\s+(.+)")

# Líneas de notas: asteriscos (2024) y números ordinales (2019), p.ej. "1/ Mineral..."
SKIP_RE = re.compile(r"^(p/[\s]|\d+/[\s]|Fuente|Nota|Gobierno|N/D\s|\*+)", re.I)

# Línea de unidad bajo el título de tabla: "(Toneladas)", "(Pesos Corrientes)", etc.
UNIT_LINE_RE = re.compile(r"^\((Toneladas|[Pp]esos|Miles de pesos|kilogramos)", re.I)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extrae tablas de producción minera del Anuario SGM")
    p.add_argument("--pdf", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--start", type=int, required=True, help="Página inicio 0-indexed")
    p.add_argument("--end", type=int, required=True, help="Página fin 0-indexed (inclusive)")
    p.add_argument("--years", nargs="+", required=True, help="Años a extraer, e.g. 2020 2021 ...")
    p.add_argument("--unidad-val", default="Miles de pesos corrientes",
                   help="Unidad de la tabla de valor (default: 'Miles de pesos corrientes')")
    return p.parse_args()


def clean_product(name: str) -> tuple[str, str | None]:
    """Limpia marcadores de nota y extrae unidad inline (e.g. 'Oro (kg)' → 'Oro', 'kg')."""
    name = name.strip()
    m = re.search(r"\(([^)]+)\)\s*$", name)
    unit = m.group(1).strip() if m else None
    name = re.sub(r"\s*\([^)]+\)\s*$", "", name).strip()
    name = re.sub(r"\*+$", "", name).strip()
    name = re.sub(r"[:]+\s*$", "", name).strip()
    name = re.sub(r"\s+\d+\s*$", "", name).strip()   # refs superíndice: "Metálicos 6" → "Metálicos"
    name = re.sub(r"\s+\d+/$", "", name).strip()     # notas numéricas: "Arena 1/" → "Arena"
    name = re.sub(r"^Metalicos$", "Metálicos", name, flags=re.I)
    name = re.sub(r"^No [Mm]etalicos$", "No metálicos", name, flags=re.I)
    return name, unit


def parse_num(tokens: list[str]) -> float | None:
    """Une tokens del mismo campo y parsea como float.

    Maneja:
    - Espacio como separador de miles (2024): ['45', '577.00'] → 45577.0
    - Coma como separador de miles (2019): ['2,120,154.68'] → 2120154.68
    """
    s = "".join(tokens).replace(" ", "").replace(",", "")
    if not s or s == "-" or s.lower() in ("nd", "n.d.", "n/d"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def group_by_line(words: list[dict], y_tol: float = 4) -> list[list[dict]]:
    """Agrupa palabras por posición y (líneas), devuelve cada línea ordenada por x."""
    buckets: dict[int, list[dict]] = {}
    for w in words:
        k = int(w["top"] / y_tol)
        buckets.setdefault(k, []).append(w)
    return [sorted(ws, key=lambda w: w["x0"]) for _, ws in sorted(buckets.items())]


def detect_year_header(line: list[dict], years: list[str]) -> dict[str, float] | None:
    """Detecta fila de encabezado buscando ≥2 años esperados en la línea.

    No requiere 'Productos/Años' en la misma línea, lo que resuelve el caso
    del Anuario 2019 donde el encabezado se divide en dos líneas (y_tol las separa).
    """
    cols = {}
    for w in line:
        if w["text"] in years:
            cols[w["text"]] = (w["x0"] + w["x1"]) / 2
    return cols if len(cols) >= 2 else None


def assign_col(word: dict, year_cols: dict[str, float], prod_end: float) -> str | None:
    """Asigna una palabra al año cuyo centro de columna sea más cercano (< 60px)."""
    wx = (word["x0"] + word["x1"]) / 2
    if wx <= prod_end:
        return None
    best, best_d = None, float("inf")
    for yr, cx in year_cols.items():
        d = abs(wx - cx)
        if d < best_d:
            best, best_d = yr, d
    return best if best_d < 60 else None


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


def main() -> None:
    args = parse_args()
    years: list[str] = args.years

    rows: list[dict] = []
    current_estado: str | None = None
    current_tabla: str | None = None
    current_unidad_base: str | None = None
    current_categoria: str | None = None
    year_cols: dict[str, float] = {}
    prod_end: float = 150.0

    with pdfplumber.open(args.pdf) as pdf:
        for page_idx in range(args.start, args.end + 1):
            page = pdf.pages[page_idx]
            words = page.extract_words(x_tolerance=3, y_tolerance=3)
            lines = group_by_line(words)

            for line in lines:
                if not line:
                    continue
                text = " ".join(w["text"] for w in line).strip()

                # Notas al pie y líneas de unidad
                if SKIP_RE.match(text) or UNIT_LINE_RE.match(text):
                    continue

                # Encabezado de sección estatal: "6.X [.] Nombre del Estado"
                m = SECCION_RE.match(text)
                if m:
                    state_raw = m.group(2).strip()
                    state_name = re.split(r"\s{3,}", state_raw)[0].strip()
                    current_estado = state_name
                    current_tabla = None
                    current_categoria = None
                    year_cols = {}
                    continue

                # Ancla tabla valor (primero, para que "Valor de..." no active la de volumen)
                if re.search(r"Valor de la [Pp]roducci[oó]n", text, re.I):
                    current_tabla = "valor"
                    current_unidad_base = args.unidad_val
                    current_categoria = None
                    year_cols = {}
                    continue

                # Ancla tabla volumen/producción
                # Excluye: "Valor de...", "...por entidad federativa" (título de capítulo),
                # "Volumen y Valor..." (título de capítulo en edición 2019)
                if (re.search(r"(Volumen de la |)Producci[oó]n [Mm]inera", text, re.I)
                        and "Valor" not in text
                        and "entidad" not in text.lower()
                        and "y Valor" not in text):
                    current_tabla = "produccion"
                    current_unidad_base = "Toneladas"
                    current_categoria = None
                    year_cols = {}
                    continue

                # Fila de encabezado de años (funciona aunque esté en línea separada)
                detected = detect_year_header(line, years)
                if detected:
                    year_cols = detected
                    yr_words = [w for w in line if w["text"] in years]
                    if yr_words:
                        prod_end = min(w["x0"] for w in yr_words) - 5
                    continue

                # Sin contexto suficiente todavía
                if not (current_estado and current_tabla and year_cols):
                    continue

                # Parsear como fila de datos
                result = parse_data_row(line, year_cols, prod_end)
                if not result:
                    continue
                prod_raw, col_tokens = result
                prod_raw = prod_raw.strip()
                if not prod_raw:
                    continue

                # Detección de categoría — acepta variantes con/sin acento
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
                    if not any(parse_num(ts) is not None for ts in col_tokens.values()):
                        continue

                prod_clean, unit_override = clean_product(prod_raw)
                if not prod_clean:
                    continue

                unidad = unit_override if unit_override else current_unidad_base

                for yr in years:
                    rows.append(
                        {
                            "estado": current_estado,
                            "tabla": current_tabla,
                            "categoria": current_categoria or "",
                            "producto": prod_clean,
                            "unidad": unidad,
                            "año": int(yr),
                            "valor": parse_num(col_tokens.get(yr, [])),
                        }
                    )

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
