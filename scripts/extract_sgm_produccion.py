"""
Extrae tablas de producción minera por entidad federativa (pp. 52-97)
del Anuario Estadístico de la Minería Mexicana 2024 (SGM).

Salida: data/sgm/produccion_minera_entidad_2020_2024.csv
Columnas: estado, tabla, categoria, producto, unidad, año, valor
"""

import re
from pathlib import Path

import pandas as pd
import pdfplumber

PDF = Path("data/sgm/raw/Anuario_2024_Edicion_2025.pdf")
OUT = Path("data/sgm/produccion_minera_entidad_2020_2024.csv")

START_PAGE = 51  # 0-indexed (página 52 del PDF)
END_PAGE = 96    # 0-indexed (página 97 del PDF)

YEARS = ["2020", "2021", "2022", "2023", "2024"]

SECCION_RE = re.compile(r"^6\.(\d+)\s+(.+)")
SKIP_RE = re.compile(r"^(p/[\s]|Fuente|Nota|\*+)", re.I)
UNIT_LINE_RE = re.compile(r"^\((Toneladas|Miles de pesos|kilogramos)", re.I)


def clean_product(name: str) -> tuple[str, str | None]:
    """Limpia marcadores de nota y extrae unidad inline (e.g. 'Oro (kg)' → 'Oro', 'kg')."""
    name = name.strip()
    m = re.search(r"\(([^)]+)\)\s*$", name)
    unit = m.group(1).strip() if m else None
    name = re.sub(r"\s*\([^)]+\)\s*$", "", name).strip()
    name = re.sub(r"\*+$", "", name).strip()
    name = re.sub(r"[:]+\s*$", "", name).strip()
    name = re.sub(r"\s+\d+\s*$", "", name).strip()  # footnote refs: "Metálicos 6" → "Metálicos"
    # Normalizar variantes sin acento
    name = re.sub(r"^Metalicos$", "Metálicos", name, flags=re.I)
    name = re.sub(r"^No metalicos$", "No metálicos", name, flags=re.I)
    return name, unit


def parse_num(tokens: list[str]) -> float | None:
    """Une tokens del mismo campo (e.g. ['45', '577.00'] → 45577.0)."""
    s = "".join(tokens).replace(" ", "").replace(",", ".")
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


def detect_year_header(line: list[dict]) -> dict[str, float] | None:
    """Si la línea es 'Productos/Años | 2020 | ... | 2024', devuelve {año: x_centro}."""
    text = " ".join(w["text"] for w in line)
    if not re.search(r"Productos", text, re.I):
        return None
    cols = {}
    for w in line:
        if re.match(r"^202[0-4]$", w["text"]):
            cols[w["text"]] = (w["x0"] + w["x1"]) / 2
    return cols or None


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
    rows: list[dict] = []

    current_estado: str | None = None
    current_tabla: str | None = None
    current_unidad_base: str | None = None
    current_categoria: str | None = None
    year_cols: dict[str, float] = {}
    prod_end: float = 150.0

    with pdfplumber.open(PDF) as pdf:
        for page_idx in range(START_PAGE, END_PAGE + 1):
            page = pdf.pages[page_idx]
            words = page.extract_words(x_tolerance=3, y_tolerance=3)
            lines = group_by_line(words)

            for line in lines:
                if not line:
                    continue
                text = " ".join(w["text"] for w in line).strip()

                # Notas al pie, fuentes, líneas de unidad como "(Toneladas)"
                if SKIP_RE.match(text) or UNIT_LINE_RE.match(text):
                    continue

                # Encabezado de sección estatal: "6.X Nombre del Estado"
                m = SECCION_RE.match(text)
                if m:
                    state_raw = m.group(2).strip()
                    # Por si el título de tabla aparece en la misma línea
                    state_name = re.split(r"\s{3,}", state_raw)[0].strip()
                    current_estado = state_name
                    current_tabla = None
                    current_categoria = None
                    year_cols = {}
                    continue

                # Anclas de tipo de tabla (match exacto para evitar falsos positivos
                # como encabezados de capítulo "Producción minera por entidad...")
                if re.fullmatch(r"Valor de la producci[oó]n minera", text, re.I):
                    current_tabla = "valor"
                    current_unidad_base = "Miles de pesos corrientes"
                    current_categoria = None
                    year_cols = {}
                    continue
                if re.fullmatch(r"Producci[oó]n minera", text, re.I):
                    current_tabla = "produccion"
                    current_unidad_base = "Toneladas"
                    current_categoria = None
                    year_cols = {}
                    continue

                # Fila de encabezado de años: "Productos/Años | 2020 | ... | 2024"
                detected = detect_year_header(line)
                if detected:
                    year_cols = detected
                    yr_words = [w for w in line if re.match(r"^202[0-4]$", w["text"])]
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
                        # El sub-bloque Agregados pétreos está bajo No metálicos;
                        # emitir esta fila con su propia etiqueta pero dejar
                        # current_categoria = "No metálicos" para los sub-items.
                        current_categoria = "No metálicos"
                    else:
                        current_categoria = "Metálicos"
                    # Emitir sólo si la fila tiene al menos un valor (subtotales en tabla=valor)
                    if not any(parse_num(ts) is not None for ts in col_tokens.values()):
                        continue

                prod_clean, unit_override = clean_product(prod_raw)
                if not prod_clean:
                    continue

                unidad = unit_override if unit_override else current_unidad_base

                for yr in YEARS:
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
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False, encoding="utf-8")

    print(f"Guardado: {OUT}")
    print(f"Filas totales: {len(df):,}")
    print(f"Estados únicos: {df['estado'].nunique()}")
    print(f"Tablas: {sorted(df['tabla'].unique())}")
    print()
    print("Productos únicos por tabla/categoría:")
    print(df.groupby(["tabla", "categoria"])["producto"].nunique().to_string())
    print()

    # Spot-checks
    checks = [
        ("Aguascalientes", "produccion", "Plata", 2020, 45577.0),
        ("Sonora", "produccion", "Oro", 2024, 38360.3),
    ]
    for estado, tabla, prod, yr, expected in checks:
        val = df[
            (df["estado"] == estado)
            & (df["tabla"] == tabla)
            & (df["producto"] == prod)
            & (df["año"] == yr)
        ]["valor"].values
        status = "OK" if val.size and abs(val[0] - expected) < 1 else "REVISAR"
        print(f"[{status}] {estado} {prod} {yr}: {val} (esperado ~{expected})")


if __name__ == "__main__":
    main()
