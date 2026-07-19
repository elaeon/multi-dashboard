"""
Compara el valor de producción de un mineral (SGM) contra el PIB del sector
minero y el PIB total de un estado (INEGI PIBE), año por año.

Herramienta de diagnóstico de una sola ejecución: no escribe ninguna tabla
intermedia, todo se calcula en memoria a partir de las fuentes ya existentes
en dashboard_data/ y data/inegi/pibe/tabulados/.

Uso:
    uv run python scripts/analisis_producto_vs_pib_mineria.py \\
        --producto "Agregados pétreos" --estado Chiapas \\
        --anio-inicio 2020 --anio-fin 2024
"""

import argparse

import polars as pl

PRODUCCION_PARQUET = "dashboard_data/produccion_minera_clean.parquet"
PIBE_DIR = "data/inegi/pibe/tabulados"

# Mismo mapeo que scripts/prepare_pibe.py: SGM usa nombres cortos, el manifest
# de PIBE usa los nombres largos oficiales para estos 3 estados.
PIBE_NAME_MAP = {
    "Coahuila": "Coahuila de Zaragoza",
    "Michoacán": "Michoacán de Ocampo",
    "Veracruz": "Veracruz de Ignacio de la Llave",
}

FAMILY2_BLOCK5_START = 230  # "Millones de pesos" (corrientes) — ver DATA_OVERVIEW.md


def find_pibe_file(estado: str) -> str:
    manifest = pl.read_csv(f"{PIBE_DIR}/manifest.csv")
    pibe_name = PIBE_NAME_MAP.get(estado, estado)
    match = manifest.filter(pl.col("titulo").str.ends_with(f"/ {pibe_name}"))
    if match.is_empty():
        estados_validos = sorted(
            manifest.filter(pl.col("numero").is_between(49, 80))["titulo"]
            .str.split("/ ").list.last().to_list()
        )
        raise SystemExit(
            f"Estado '{estado}' no encontrado en {PIBE_DIR}/manifest.csv.\n"
            f"Estados válidos: {', '.join(estados_validos)}"
        )
    numero = match["numero"][0]
    return f"{PIBE_DIR}/PIBE_{numero}.xlsx"


def load_pib_rows(estado: str) -> pl.DataFrame:
    """Filas 'Actividad económica total' y '21 - Minería', bloque nominal (corrientes)."""
    path = find_pibe_file(estado)
    df = pl.read_excel(path, sheet_name="Tabulado", read_options={"header_row": 4})
    block = df.slice(FAMILY2_BLOCK5_START, 46).with_columns(
        pl.col("Concepto").str.strip_chars().alias("Concepto")
    )
    rows = block.filter(pl.col("Concepto").is_in(["Actividad económica total", "21 - Minería"]))
    year_cols = [c for c in df.columns if c not in ("Concepto",)]
    long = rows.unpivot(index="Concepto", on=year_cols, variable_name="año_raw", value_name="valor").with_columns(
        pl.col("año_raw").str.strip_chars("R").cast(pl.Int32).alias("año")
    )
    pivoted = long.pivot(on="Concepto", index="año", values="valor").rename({
        "Actividad económica total": "pib_total_mdp",
        "21 - Minería": "pib_mineria_mdp",
    })
    return pivoted


def load_producto_serie(producto: str, estado: str, anio_inicio: int, anio_fin: int) -> pl.DataFrame:
    df = pl.read_parquet(PRODUCCION_PARQUET)

    if producto not in df["producto"].unique().to_list():
        productos_validos = sorted(df["producto"].unique().to_list())
        raise SystemExit(
            f"Producto '{producto}' no encontrado en {PRODUCCION_PARQUET}.\n"
            f"Productos válidos: {', '.join(productos_validos)}"
        )
    if estado not in df["estado"].unique().to_list():
        estados_validos = sorted(df["estado"].unique().to_list())
        raise SystemExit(
            f"Estado '{estado}' no encontrado en {PRODUCCION_PARQUET}.\n"
            f"Estados válidos: {', '.join(estados_validos)}"
        )

    serie = (
        df.filter(
            (pl.col("producto") == producto) & (pl.col("estado") == estado)
            & (pl.col("año").is_between(anio_inicio, anio_fin))
        )
        .group_by("año").agg((pl.col("valor_pesos").sum() / 1e6).alias("valor_producto_mdp"))
    )
    # Completa los años sin dato con null en vez de omitirlos de la tabla.
    todos_los_anios = pl.DataFrame({"año": list(range(anio_inicio, anio_fin + 1))}, schema={"año": pl.Int32})
    return todos_los_anios.join(serie, on="año", how="left").sort("año")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--producto", required=True, help='Nombre exacto del producto, p.ej. "Agregados pétreos"')
    parser.add_argument("--estado", required=True, help='Nombre corto del estado, p.ej. "Chiapas"')
    parser.add_argument("--anio-inicio", type=int, required=True)
    parser.add_argument("--anio-fin", type=int, required=True)
    args = parser.parse_args()

    producto_serie = load_producto_serie(args.producto, args.estado, args.anio_inicio, args.anio_fin)
    pib = load_pib_rows(args.estado)

    tabla = (
        producto_serie.join(pib, on="año", how="left")
        .with_columns(
            (pl.col("valor_producto_mdp") / pl.col("pib_mineria_mdp")).round(2).alias("ratio_vs_mineria"),
            (pl.col("valor_producto_mdp") / pl.col("pib_total_mdp") * 100).round(2).alias("ratio_vs_pib_pct"),
        )
        .select("año", "valor_producto_mdp", "pib_mineria_mdp", "ratio_vs_mineria", "pib_total_mdp", "ratio_vs_pib_pct")
        .sort("año")
        .with_columns(
            pl.col("año").cast(pl.Utf8),
            pl.col("valor_producto_mdp").round(1),
            pl.col("pib_mineria_mdp").round(1),
            pl.col("pib_total_mdp").round(1),
        )
    )

    print(f"\n{args.producto} — {args.estado} ({args.anio_inicio}-{args.anio_fin})")
    print("Valores en millones de pesos (Mdp), pesos corrientes.\n")
    with pl.Config(tbl_rows=-1, tbl_cols=-1, thousands_separator=","):
        print(tabla)


if __name__ == "__main__":
    main()
