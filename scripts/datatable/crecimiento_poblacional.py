"""
Tasa de crecimiento poblacional por entidad, censos generales 1895–2020.

Fuente: data/inegi/ccpv/poblacion/poblacion_historica.csv — serie histórica de
población total por entidad en los 14 censos generales de México. Reporta, censo
por censo: población, cambio absoluto y % respecto al censo anterior, y la tasa
de crecimiento media anual (TCMA) — el estándar demográfico de INEGI/CONAPO para
comparar intercensales de duración distinta (los censos de esta serie van de 5 a
11 años de diferencia entre sí, así que un % de cambio simple no es comparable
de un período a otro).

Run: uv run python scripts/datatable/crecimiento_poblacional.py --entidad Jalisco
"""

import argparse
import csv
import sys
from pathlib import Path

import polars as pl

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "scripts" / "centralismo"))
from comun import NOMBRE, normalizar_estado

CSV_PATH = RAIZ / "data" / "inegi" / "ccpv" / "poblacion" / "poblacion_historica.csv"
DIR = RAIZ / "dashboard_data"
CENSOS = [1895, 1900, 1910, 1921, 1930, 1940, 1950, 1960, 1970, 1980, 1990, 2000, 2010, 2020]


def _entidad(valor):
    """Acepta clave (1-32) o nombre en cualquier régimen ('cdmx', 'Distrito Federal')."""
    if valor.isdigit() and 1 <= int(valor) <= 32:
        return int(valor)
    cve = normalizar_estado(valor)
    if cve is None:
        raise argparse.ArgumentTypeError(
            f"entidad no reconocida: {valor!r} (usa el nombre o la clave 1-32)"
        )
    return cve


def _cve(etiqueta):
    """Nombre de la fuente → CVE_ENT 1-32, o None para las pseudo-filas (Islas
    Marías, la fila combinada de Baja California pre-1930)."""
    return normalizar_estado(etiqueta.split(" (")[0])  # 'Nayarit (Territorio de Tepic)'


def _valor(s):
    """Celda → int, o None si estaba vacía o llevaba el centinela '-'. El
    .replace(' ', '') limpia un residuo de separador de miles con espacio que
    sobrevivió en al menos una celda (Aguascalientes 1940)."""
    s = s.strip().replace(" ", "")
    return None if s in ("", "-") else int(s)


def leer_poblacion_historica() -> pl.DataFrame:
    filas = []
    with open(CSV_PATH, encoding="utf-8", newline="") as f:
        lector = csv.reader(f)
        next(lector)  # encabezado: etiquetas censo_N/poblacion_N genéricas (y una duplicada)
        for row in lector:
            cve = _cve(row[0])
            if cve is None:
                continue
            resto = row[1:]
            for año_str, val_str in zip(resto[0::2], resto[1::2]):
                filas.append({"cve_ent": cve, "entidad": NOMBRE[cve],
                             "censo": int(año_str), "poblacion": _valor(val_str)})

    df = pl.DataFrame(filas, schema={"cve_ent": pl.Int16, "entidad": pl.Utf8,
                                     "censo": pl.Int16, "poblacion": pl.Int64})
    assert df["cve_ent"].n_unique() == 32, f"{df['cve_ent'].n_unique()} entidades, se esperaban 32"
    assert set(df["censo"].unique().to_list()) == set(CENSOS)
    return df


def calcular(df, cve):
    """Serie de una entidad con cambio absoluto, % y TCMA respecto al censo
    anterior CON DATO (no necesariamente el renglón inmediato anterior — algunas
    entidades tienen huecos, p. ej. Quintana Roo no tiene censo propio en 1895 ni
    1900)."""
    salida = []
    censo_prev = pob_prev = None
    for r in df.filter(pl.col("cve_ent") == cve).sort("censo").iter_rows(named=True):
        censo, pob = r["censo"], r["poblacion"]
        fila = {"censo": censo, "poblacion": pob, "años": None,
                "cambio_absoluto": None, "cambio_pct": None, "tcma_pct": None}
        if pob is not None and pob_prev is not None:
            años = censo - censo_prev
            fila["años"] = años
            fila["cambio_absoluto"] = pob - pob_prev
            fila["cambio_pct"] = (pob - pob_prev) / pob_prev * 100
            fila["tcma_pct"] = ((pob / pob_prev) ** (1 / años) - 1) * 100
        salida.append(fila)
        if pob is not None:
            censo_prev, pob_prev = censo, pob
    return pl.DataFrame(salida, schema={
        "censo": pl.Int16, "poblacion": pl.Int64, "años": pl.Int16,
        "cambio_absoluto": pl.Int64, "cambio_pct": pl.Float64, "tcma_pct": pl.Float64,
    })


def _fmt(v, decimales=0, signo=False, sufijo=""):
    if v is None:
        return "—"
    return f"{v:{'+' if signo else ''},.{decimales}f}{sufijo}"


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--entidad", type=_entidad, required=True,
                   help="Nombre o clave INEGI (1-32) de la entidad federativa")
    p.add_argument("--guardar", action="store_true",
                   help="Escribe el resultado a dashboard_data/crecimiento_<entidad>.parquet")
    a = p.parse_args()

    df = leer_poblacion_historica()
    tabla = calcular(df, a.entidad)

    print(f"\n{'═' * 68}")
    print(f"  {NOMBRE[a.entidad]} — crecimiento poblacional, 1895–2020")
    print("═" * 68)
    print(f"  {'Censo':<7} {'Población':>12} {'Δ Población':>14} {'Δ % (total)':>13}"
          f" {'Años':>6} {'TCMA % anual':>13}")
    for r in tabla.iter_rows(named=True):
        print(f"  {r['censo']:<7} {_fmt(r['poblacion']):>12}"
              f" {_fmt(r['cambio_absoluto'], signo=True):>14}"
              f" {_fmt(r['cambio_pct'], 1, signo=True, sufijo='%'):>13}"
              f" {_fmt(r['años']):>6}"
              f" {_fmt(r['tcma_pct'], 2, signo=True, sufijo='%'):>13}")
    print()

    if a.guardar:
        salida = tabla.with_columns(
            pl.lit(a.entidad).cast(pl.Int16).alias("cve_ent"),
            pl.lit(NOMBRE[a.entidad]).alias("entidad"),
        ).select(["cve_ent", "entidad", "censo", "poblacion", "años",
                  "cambio_absoluto", "cambio_pct", "tcma_pct"])
        ruta = DIR / f"crecimiento_{NOMBRE[a.entidad].lower().replace(' ', '_')}.parquet"
        salida.write_parquet(ruta)
        print(f"Saved → {ruta}  ({salida.height:,} filas)\n")


if __name__ == "__main__":
    main()
