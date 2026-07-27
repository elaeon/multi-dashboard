"""
Presupuesto por Ramo (PEF) con clasificación institucional: Secretaría, Entidad
de Control Directo (paraestatales mayores), Ramo General (fondos/transferencias)
u Órgano Autónomo/Poder.

Clasificación hardcodeada por ID_RAMO (estructura oficial SHCP, estable) — todo
Ramo no listado en AUTONOMOS/GENERALES/CONTROL_DIRECTO se asume "Secretaría".

Source: data/presupuesto_federacion/presupuesto/egresos_federacion/PEF_2026.xlsx
Output: dashboard_data/presupuesto_por_ramo.parquet
Run: uv run python scripts/prepare_ramos_sector_presupuesto.py
"""

from pathlib import Path

import polars as pl

PEF_PATH = Path("data/presupuesto_federacion/presupuesto/egresos_federacion/PEF_2026.xlsx")
OUT_PATH = Path("dashboard_data/presupuesto_por_ramo.parquet")

AUTONOMOS = {1, 3, 22, 32, 35, 40, 41, 43, 44, 49}
GENERALES = {19, 24, 28, 30, 33, 34}
CONTROL_DIRECTO = {50, 51, 52, 53}


def categoria(id_ramo: int) -> str:
    if id_ramo in AUTONOMOS:
        return "Órgano Autónomo/Poder"
    if id_ramo in GENERALES:
        return "Ramo General"
    if id_ramo in CONTROL_DIRECTO:
        return "Entidad de Control Directo"
    return "Secretaría"


def main():
    raw = pl.read_excel(PEF_PATH)
    col_monto = next(c for c in raw.columns if "MONTO" in c)

    ramos = (
        raw.group_by("ID_RAMO", "DESC_RAMO")
        .agg(pl.sum(col_monto).alias("MONTO"))
        .with_columns(pl.col("ID_RAMO").map_elements(categoria, return_dtype=pl.Utf8).alias("CATEGORIA"))
        .sort("MONTO", descending=True)
    )
    total = ramos["MONTO"].sum()
    ramos = ramos.with_columns((pl.col("MONTO") / total * 100).alias("PCT_TOTAL"))

    assert ramos["CATEGORIA"].null_count() == 0
    assert abs(ramos["MONTO"].sum() - total) < 1

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ramos.write_parquet(OUT_PATH)
    print(f"Guardado → {OUT_PATH} ({ramos.height} ramos)\n")

    print(f"{'Ramo':<50} {'Categoría':<26} {'Monto (M pesos)':>16} {'%':>6}")
    for r in ramos.iter_rows(named=True):
        print(f"{r['DESC_RAMO']:<50} {r['CATEGORIA']:<26} {r['MONTO']/1e6:>16,.0f} {r['PCT_TOTAL']:>5.1f}%")

    print("\nSubtotal por categoría:")
    subt = ramos.group_by("CATEGORIA").agg(pl.sum("MONTO"), pl.sum("PCT_TOTAL")).sort("MONTO", descending=True)
    for r in subt.iter_rows(named=True):
        print(f"  {r['CATEGORIA']:<26} {r['MONTO']/1e6:>16,.0f} M pesos  ({r['PCT_TOTAL']:.1f}%)")


if __name__ == "__main__":
    main()
