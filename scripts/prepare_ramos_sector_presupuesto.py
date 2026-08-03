"""
Presupuesto por Ramo (PEF) con clasificación institucional: Secretaría, Entidad
de Control Directo (paraestatales mayores), Órgano Autónomo/Poder, Estados
(Participaciones + Aportaciones + otras transferencias) u Otros compromisos
federales (deuda pública, aportaciones a seguridad social federal, adefas).

"Estados" se identifica por el capítulo de gasto oficial (COG) "Participaciones
y aportaciones" (capítulo 8000) en vez de una lista fija de Ramos — así incluye
también la porción de Ramo 25 (Previsiones y Aportaciones para Educación
Básica) que SHCP ya clasifica como aportación, no solo Ramo 28/33. El resto de
la clasificación sigue siendo por ID_RAMO (estructura oficial SHCP, estable) —
todo Ramo no listado en AUTONOMOS/CONTROL_DIRECTO/OTROS_FEDERALES y sin gasto
en capítulo 8000 se asume "Secretaría". Un mismo Ramo puede repartirse entre
dos categorías (p. ej. Ramo 25 aparece en "Estados" y en "Secretaría").

Source: data/presupuesto_federacion/presupuesto/egresos_federacion/ — el
nombre de archivo varía por año (PEF_{año}.xlsx, PEF{año}_AC01.xlsx,
pef_{año}.xlsx, pef_ac01_{año}.xlsx); se prueban esos patrones en orden.
Solo verificado para años con el esquema de columnas 2018+ (ID_RAMO,
DESC_CAPITULO, MONTO_*); años más viejos (2008-2017, formato AC01) pueden
tener un esquema distinto y no están soportados.

Por default solo imprime la tabla en terminal — no escribe nada a disco. Pasa
--save para guardar el parquet en dashboard_data/.

Output (con --save): dashboard_data/presupuesto_por_ramo_{año}.parquet
Run: uv run python scripts/prepare_ramos_sector_presupuesto.py --year 2026 --save

Con --by-ramo ID_RAMO, en vez de la tabla ramo×categoría, imprime (y con
--save guarda) el desglose de ESE Ramo por Programa presupuestario (Pp) —
p. ej. --by-ramo 33 separa FONE/FASSA/FAIS/FORTAMUN/etc. dentro de Aportaciones.
Run: uv run python scripts/prepare_ramos_sector_presupuesto.py --year 2026 --by-ramo 33 --save
"""

import argparse
from pathlib import Path

import polars as pl

PEF_DIR = Path("data/presupuesto_federacion/presupuesto/egresos_federacion")
OUT_DIR = Path("dashboard_data")

AUTONOMOS = {1, 3, 22, 32, 35, 40, 41, 43, 44, 49}
CONTROL_DIRECTO = {50, 51, 52, 53}
OTROS_FEDERALES = {19, 24, 30, 34}  # Aportaciones Seg. Social, Deuda Pública, Adefas, Ahorradores
CAPITULO_ESTADOS = "participaciones y aportaciones"  # comparar en minúsculas: mayúsculas varían por año (2018: "Participaciones y Aportaciones")


def encontrar_pef(year: int) -> Path:
    candidatos = [
        f"PEF_{year}.xlsx",
        f"PEF{year}_AC01.xlsx",
        f"pef_{year}.xlsx",
        f"pef_ac01_{year}.xlsx",
    ]
    for nombre in candidatos:
        ruta = PEF_DIR / nombre
        if ruta.exists():
            return ruta
    raise FileNotFoundError(f"No se encontró el PEF de {year} en {PEF_DIR} (probé: {candidatos})")


def categoria(desc_capitulo: str, id_ramo: int) -> str:
    if desc_capitulo.strip().lower() == CAPITULO_ESTADOS:
        return "Estados (Aportaciones+Participaciones+otras transf.)"
    if id_ramo in AUTONOMOS:
        return "Órgano Autónomo/Poder"
    if id_ramo in CONTROL_DIRECTO:
        return "Entidad de Control Directo"
    if id_ramo in OTROS_FEDERALES:
        return "Otros compromisos federales (deuda, seg. social, adefas)"
    return "Secretaría"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--by-ramo", type=int, default=None, metavar="ID_RAMO", help="Desglosa este Ramo por Programa presupuestario (Pp) en vez de la tabla ramo×categoría")
    parser.add_argument("--save", action="store_true", help="Guardar el resultado como parquet en dashboard_data/ (por default no se guarda, solo se imprime)")
    args = parser.parse_args()
    year = args.year

    pef_path = encontrar_pef(year)
    out_path = OUT_DIR / f"presupuesto_por_ramo_{year}.parquet"

    raw = pl.read_excel(pef_path)
    try:
        col_monto = next(c for c in raw.columns if "MONTO" in c)
    except StopIteration:
        raise ValueError(
            f"{pef_path.name} no tiene columna MONTO_* (columnas: {raw.columns}). "
            "Los PEF 2008-2017 en formato AC01 no siguen este esquema y no están soportados."
        )

    ramos = (
        raw.with_columns(
            pl.struct(["DESC_CAPITULO", "ID_RAMO"])
            .map_elements(lambda r: categoria(r["DESC_CAPITULO"], r["ID_RAMO"]), return_dtype=pl.Utf8)
            .alias("CATEGORIA")
        )
        .group_by("ID_RAMO", "DESC_RAMO", "CATEGORIA")
        .agg(pl.sum(col_monto).alias("MONTO"))
        .sort("MONTO", descending=True)
    )
    total = ramos["MONTO"].sum()
    ramos = ramos.with_columns(
        (pl.col("MONTO") / total * 100).alias("PCT_TOTAL"),
        pl.lit(year).alias("AÑO"),
    )

    assert ramos["CATEGORIA"].null_count() == 0
    assert abs(ramos["MONTO"].sum() - total) < 1

    if args.by_ramo is not None:
        id_ramo = args.by_ramo
        sub = raw.filter(pl.col("ID_RAMO") == id_ramo)
        if sub.height == 0:
            raise ValueError(f"ID_RAMO {id_ramo} no existe en {pef_path.name}")
        desc_ramo = sub["DESC_RAMO"][0]

        out_path = OUT_DIR / f"presupuesto_ramo_{id_ramo}_{year}.parquet"
        pp = (
            sub.group_by("ID_PP", "DESC_PP")
            .agg(pl.sum(col_monto).alias("MONTO"))
            .sort("MONTO", descending=True)
        )
        total_ramo = pp["MONTO"].sum()
        pp = pp.with_columns(
            (pl.col("MONTO") / total_ramo * 100).alias("PCT_TOTAL"),
            pl.lit(id_ramo).alias("ID_RAMO"),
            pl.lit(desc_ramo).alias("DESC_RAMO"),
            pl.lit(year).alias("AÑO"),
        )

        print(f"Fuente: {pef_path}")
        if args.save:
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            pp.write_parquet(out_path)
            print(f"Guardado → {out_path} ({pp.height} programas presupuestarios)\n")
        else:
            print("(no guardado — pasa --save para escribir el parquet)\n")

        print(f"Ramo {id_ramo} — {desc_ramo}\n")
        print(f"{'Pp':<60} {'Monto (M pesos)':>16} {'%':>6}")
        for r in pp.iter_rows(named=True):
            print(f"{r['DESC_PP']:<60} {r['MONTO']/1e6:>16,.0f} {r['PCT_TOTAL']:>5.1f}%")
        print(f"{'TOTAL':<60} {total_ramo/1e6:>16,.0f} {100.0:>5.1f}%")
        return

    print(f"Fuente: {pef_path}")
    if args.save:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        ramos.write_parquet(out_path)
        print(f"Guardado → {out_path} ({ramos.height} filas ramo×categoría)\n")
    else:
        print("(no guardado — pasa --save para escribir el parquet)\n")

    print(f"{'ID':>4} {'Ramo':<50} {'Categoría':<58} {'Monto (M pesos)':>16} {'%':>6}")
    for r in ramos.iter_rows(named=True):
        print(f"{r['ID_RAMO']:>4} {r['DESC_RAMO']:<50} {r['CATEGORIA']:<58} {r['MONTO']/1e6:>16,.0f} {r['PCT_TOTAL']:>5.1f}%")

    print("\nSubtotal por categoría:")
    subt = ramos.group_by("CATEGORIA").agg(pl.sum("MONTO"), pl.sum("PCT_TOTAL")).sort("MONTO", descending=True)
    for r in subt.iter_rows(named=True):
        print(f"  {r['CATEGORIA']:<58} {r['MONTO']/1e6:>16,.0f} M pesos  ({r['PCT_TOTAL']:.1f}%)")


if __name__ == "__main__":
    main()
