"""
Prepara la tabla limpia para el dashboard de Producción Minera (SGM).

Lee los 6 CSV anuales de data/sgm/produccion_minera/, conserva solo las filas
de valor monetario (tabla == "valor"), aplica los filtros de limpieza
documentados en data/sgm/produccion_minera/DATA_OVERVIEW.md (filas de subtotal,
reescalado de unidad, exclusión de "Agregados pétreos", años provisionales) y
guarda el resultado en dashboard_data/produccion_minera_clean.parquet.
"""

from pathlib import Path

import polars as pl

DATA_DIR = "data/sgm/produccion_minera"
OUT_DIR = "dashboard_data"

# Filas "Total" / "Metálicos" / "No metálicos" (y variantes OCR) que duplican el
# valor de sus componentes aunque tengan categoria no nula (ver DATA_OVERVIEW.md
# Key Insight #2).
AGGREGATE_PRODUCTO_RE = r"(?i)^\s*(met[áa]licos|no\s+met[áa]licos|total)"

# "Agregados pétreos" (grava/arena de construcción) tiene valores erráticos e
# implausibles en todo el dataset (no solo Hidalgo/2024): el propio DATA_OVERVIEW.md
# Key Insight #1 recomienda removerlo por completo — "Remove the `agregados` product
# and Hidalgo drops out of the top 8 entirely." Se excluye del análisis principal;
# se conserva por separado para la pestaña explicativa del artefacto.
AGREGADOS_RE = r"(?i)^agregados\s+p[ée]treos$"

# Ruido OCR / texto de encabezado filtrado a producto en los archivos 2000+.
JUNK_PRODUCTOS = [
    "Anua rio Estadístico",
    "Servicio Geológico",
    "Servicio",
    "S/i Sin información.",
]

# El archivo 1995_1999 tiene OCR tan ruidoso (DATA_OVERVIEW.md gotcha #2) que no
# solo corrompe nombres de producto (~197 variantes vs ~55 canónicas) sino también
# magnitudes de valor: p.ej. Sonora/"Cobra" (OCR de "Cobre")/1997 = $544.9 mil
# millones, 34% del total 1995-1999 por sí solo — un artefacto tan severo como el
# de Hidalgo/Agregados. Se excluye el periodo completo en vez de perseguir cada
# fila corrupta individualmente; no cambia el hallazgo de concentración estatal
# (verificado: top-5 = 50.2% con 1995-2024, 51.0% con 2000-2024).
MIN_YEAR = 2000


def load_raw() -> pl.DataFrame:
    files = sorted(Path(DATA_DIR).glob("produccion_minera_entidad_*.csv"))
    frames = [
        pl.read_csv(f, encoding="utf8", null_values=["", "NA"], infer_schema_length=10000)
        for f in files
    ]
    return pl.concat(frames, how="vertical")


def main() -> None:
    df = load_raw()

    valor = df.filter(pl.col("tabla") == "valor")

    clean = (
        valor
        .filter(pl.col("año") >= MIN_YEAR)
        .filter(pl.col("categoria").is_not_null())
        .filter(~pl.col("producto").str.contains(AGGREGATE_PRODUCTO_RE))
        .filter(~pl.col("producto").is_in(JUNK_PRODUCTOS))
        .filter(pl.col("valor").is_not_null())
        .filter(pl.col("valor") >= 0)  # drops the single -1.0 sentinel (SLP/Mármol/2002)
    )

    clean = clean.with_columns(
        pl.when(pl.col("unidad") == "Miles de pesos corrientes")
        .then(pl.col("valor") * 1000)
        .otherwise(pl.col("valor"))
        .alias("valor_pesos"),
        pl.col("producto").str.contains(AGREGADOS_RE).alias("is_agregados"),
        pl.col("año").is_in([2023, 2024]).alias("provisional"),
    ).select(
        "estado", "categoria", "producto", "año", "valor_pesos", "is_agregados", "provisional"
    )

    # ── anclas de validación ────────────────────────────────────────────────
    n_dropped_null_categoria = valor.filter(pl.col("categoria").is_null()).height
    assert n_dropped_null_categoria == 950, f"null-categoria drop mismatch: {n_dropped_null_categoria}"

    hidalgo_2024 = clean.filter(
        (pl.col("estado") == "Hidalgo") & (pl.col("producto") == "Agregados pétreos") & (pl.col("año") == 2024)
    )["valor_pesos"][0]
    assert abs(hidalgo_2024 - 15.568e12) < 0.01e12, f"Hidalgo 2024 Agregados pétreos mismatch: {hidalgo_2024}"

    top5_share = (
        clean.filter(~pl.col("is_agregados"))
        .group_by("estado").agg(pl.col("valor_pesos").sum())
        .sort("valor_pesos", descending=True)
    )
    total = top5_share["valor_pesos"].sum()
    top5_pct = float(top5_share.head(5)["valor_pesos"].sum() / total * 100)
    assert 45 <= top5_pct <= 55, f"top-5 concentration out of expected range: {top5_pct:.1f}%"

    leader = top5_share.row(0, named=True)["estado"]
    assert leader == "Sonora", f"expected Sonora as top state, got {leader}"

    out_path = f"{OUT_DIR}/produccion_minera_clean.parquet"
    clean.write_parquet(out_path)
    print(f"{out_path}: {clean.height} rows, {len(clean.columns)} cols")
    print(f"Top-5 state concentration (ex-agregados): {top5_pct:.1f}%, leader: {leader}")
    print("Todas las anclas de validación coinciden con DATA_OVERVIEW.md.")


if __name__ == "__main__":
    main()
