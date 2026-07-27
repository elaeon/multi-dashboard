"""
Ranking de dependencia fiscal estatal (EFIPEM) para un año dado.

Lista las 32 entidades ordenadas por dependencia = transferencias federales
(Participaciones + Aportaciones) / (transferencias + ingresos propios).
Reutiliza cargar_efipem() de cap4_dependencia_efipem.py.

Run: uv run python scripts/datatable/ranking_dependencia_efipem.py [--año 2024]
"""

import argparse
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "centralismo"))
from cap4_dependencia_efipem import AÑOS, PROPIOS, TRANSFER, cargar_efipem
from comun import NOMBRE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--año", type=int, default=2024, choices=AÑOS,
                        help=f"Año a desplegar ({min(AÑOS)}-{max(AÑOS)}, default 2024)")
    año = parser.parse_args().año

    ef = cargar_efipem()
    panel = (
        ef.with_columns(
            pl.when(pl.col("DESCRIPCION_CATEGORIA").is_in(PROPIOS)).then(pl.lit("propios"))
            .when(pl.col("DESCRIPCION_CATEGORIA").is_in(TRANSFER)).then(pl.lit("transfer"))
            .otherwise(pl.lit("otro")).alias("grupo")
        )
        .filter(pl.col("grupo") != "otro")
        .group_by("ANIO", "CVE_ENT", "grupo")
        .agg(pl.sum("VALOR"))
        .pivot(values="VALOR", index=["ANIO", "CVE_ENT"], on="grupo")
        .fill_null(0)
        .with_columns((pl.col("transfer") / (pl.col("transfer") + pl.col("propios"))).alias("dependencia"))
        .rename({"ANIO": "año", "CVE_ENT": "cve_ent"})
    )
    assert panel.filter(pl.col("año") == año)["cve_ent"].n_unique() == 32
    assert panel["dependencia"].is_between(0, 1).all()

    d = (
        panel.filter(pl.col("año") == año)
        .with_columns(pl.col("cve_ent").replace_strict(NOMBRE).alias("estado"))
        .sort("dependencia", descending=True)
    )
    nal = panel.filter(pl.col("año") == año).select(
        (pl.sum("transfer") / (pl.sum("transfer") + pl.sum("propios"))).alias("dependencia")
    )[0, "dependencia"]

    print(f"Nacional {año}: {nal*100:.1f}%\n")
    print(f"{'#':>3}  {'Entidad':<20} {'Dependencia %':>14}")
    for i, r in enumerate(d.iter_rows(named=True), start=1):
        print(f"{i:>3}  {r['estado']:<20} {r['dependencia']*100:>13.1f}%")


if __name__ == "__main__":
    main()
