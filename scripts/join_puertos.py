# /// script
# requires-python = ">=3.11"
# dependencies = ["polars"]
# ///
"""
Join arribos_pasajeros, estadistica_buques, and estadistica_contenedores
into a single CSV using (puerto, mes, anio) as shared keys and
(puerto, mes, anio, trafico) as the key between buques and contenedores.

Usage:
    uv run scripts/join_puertos.py <arribos> <buques> <contenedores> <output>
"""

import sys
import polars as pl


def main():
    if len(sys.argv) != 5:
        print("Usage: uv run join_puertos.py <arribos_pasajeros> <estadistica_buques> <estadistica_contenedores> <output>")
        sys.exit(1)

    path_arribos, path_buques, path_contenedores, path_output = sys.argv[1:]

    arribos = pl.read_csv(path_arribos)
    buques = pl.read_csv(path_buques)
    contenedores = pl.read_csv(path_contenedores)

    # Normalize mes/anio to Int64 across all files (arribos has them as floats)
    for col in ("mes", "anio"):
        arribos = arribos.with_columns(pl.col(col).cast(pl.Int64))
        buques = buques.with_columns(pl.col(col).cast(pl.Int64))
        contenedores = contenedores.with_columns(pl.col(col).cast(pl.Int64))

    # Pivot arribos_pasajeros: Arribos → buques, Pasajeros → pasajeros
    arribos = arribos.pivot(
        on="arr_pas",
        index=["puerto", "mes", "anio"],
        values="unidades",
    ).rename({"Arribos": "buques", "Pasajeros": "pasajeros"}).with_columns(
        pl.lit("pasajeros").alias("tipo_carga")
    )

    # Aggregate contenedores before join and set tipo_carga = "Contenerizada"
    contenedores = contenedores.group_by(
        ["puerto", "mes", "anio", "trafico", "movimiento"]
    ).agg(
        pl.col("cajas").sum(),
        pl.col("toneladas").sum(),
        pl.col("teus").sum(),
    ).with_columns(pl.lit("Contenerizada").alias("tipo_carga"))

    # Join buques + contenedores on (puerto, mes, anio, trafico, tipo_carga)
    buques_contenedores = buques.join(
        contenedores,
        on=["puerto", "mes", "anio", "trafico", "tipo_carga"],
        how="full",
        coalesce=True,
    ).with_columns(pl.lit(None).cast(pl.Int64).alias("pasajeros"))

    # Concat: arribos rows get null for all buques/contenedores-specific columns
    result = pl.concat([buques_contenedores, arribos], how="diagonal")

    result = result.unique().sort(["puerto", "anio", "mes"])
    result.write_csv(path_output)
    print(f"Wrote {result.height} rows x {result.width} columns → {path_output}")


if __name__ == "__main__":
    main()
