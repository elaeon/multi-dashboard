"""
Consolida los 15 xlsx de Cuenta Pública (2011-2025) en un parquet agregado:

  informe_data/cp_estado_ramo.parquet
  grano: ciclo × id_ramo × cve_ent  (cve_ent 1-32; 33=extranjero, 34=no distribuible,
         0=sin geografía [2011]) con monto_aprobado y monto_ejercido.

Maneja la evolución de esquema documentada en DATA_OVERVIEW.md (Known Issues 1-6):
renombres de CICLO, columna de ejercido (4 nombres), rename masivo 2025, 2011 sin geografía.

Run: uv run python scripts/centralismo/preparar_cuenta_publica.py
"""

from pathlib import Path

import fastexcel
import polars as pl

RAIZ = Path(__file__).resolve().parents[2]
DIR_CP = RAIZ / "data/presupuesto_federacion/cuenta_publica/cuenta_publica"
SALIDA = RAIZ / "informe_data/cp_estado_ramo.parquet"

ARCHIVOS = {
    2011: "Cuenta_Publica_2011.xlsx",
    2012: "Cuenta_Publica_2012.xlsx",
    2013: "cuenta_publica_2013_ra_ecd.xlsx",
    2014: "cuenta_publica_2014_ra_ecd.xlsx",
    2015: "cuenta_publica_2015_ra_ecd_epe.xlsx",
    2016: "cuenta_publica_2016_gf_ecd_epe.xlsx",
    2017: "cuenta_publica_2017_gf_ecd_epe.xlsx",
    2018: "cuenta_publica_2018_gf_ecd_epe.xlsx",
    2019: "cuenta_publica_2019_gf_ecd_epe.xlsx",
    2020: "cuenta_publica_2020_gf_ecd_epe.xlsx",
    2021: "cuenta_publica_2021_gf_ecd_epe.xlsx",
    2022: "cuenta_publica_2022_gf_ecd_epe.xlsx",
    2023: "cuenta_publica_2023_gf_ecd_epe.xlsx",
    2024: "cuenta_publica_2024_gf_ecd_epe.xlsx",
    2025: "cuenta_publica_2025_gf_ecd_epe 1.xlsx",
}

CAND_RAMO = ["ID_RAMO", "R", "RAMO", "Ramo"]
CAND_ENT = ["ID_ENTIDAD_FEDERATIVA", "ENTIDAD_FEDERATIVA"]  # 2025: ENTIDAD_FEDERATIVA es el ID
CAND_EJERCIDO = ["MONTO_EJERCIDO", "MONTO_EJERCICIO", "EJERCICIO", "Ejercicio"]
CAND_APROBADO = ["MONTO_APROBADO"]


def elegir(cols: list[str], candidatos: list[str], contexto: str) -> str | None:
    for c in candidatos:
        if c in cols:
            return c
    print(f"  [warn] sin columna {candidatos[0]} en {contexto}")
    return None


def cargar_año(ciclo: int, nombre: str) -> pl.DataFrame:
    lector = fastexcel.read_excel(DIR_CP / nombre)
    cols = [c.name for c in lector.load_sheet(0, n_rows=0).available_columns()]

    c_ramo = elegir(cols, CAND_RAMO, nombre)
    c_ent = elegir(cols, CAND_ENT, nombre)
    c_ej = elegir(cols, CAND_EJERCIDO, nombre)
    c_ap = elegir(cols, CAND_APROBADO, nombre)
    if c_ramo is None or c_ej is None:
        raise ValueError(f"{nombre}: esquema inesperado: {cols[:12]}")

    usadas = [c for c in (c_ramo, c_ent, c_ap, c_ej) if c]
    df = lector.load_sheet(0, use_columns=usadas,
                           dtypes={c: "float" for c in (c_ap, c_ej)}).to_polars()

    # 2025: ENTIDAD_FEDERATIVA puede venir como str numérico; a veces el ID quedó textual
    exprs = [
        pl.col(c_ramo).cast(pl.Int64, strict=False).alias("id_ramo"),
        (pl.col(c_ent).cast(pl.Int64, strict=False) if c_ent else pl.lit(0, pl.Int64))
        .fill_null(0).alias("cve_ent"),
        pl.col(c_ap).cast(pl.Float64, strict=False).fill_null(0).alias("monto_aprobado"),
        pl.col(c_ej).cast(pl.Float64, strict=False).fill_null(0).alias("monto_ejercido"),
    ]
    out = (
        df.select(exprs)
        .group_by("id_ramo", "cve_ent")
        .agg(pl.sum("monto_aprobado"), pl.sum("monto_ejercido"))
        .with_columns(pl.lit(ciclo).alias("ciclo"))
    )
    tot = out["monto_ejercido"].sum() / 1e12
    print(f"  {ciclo}: {df.height:,} filas → {out.height} grupos; ejercido {tot:.2f} T MXN")
    return out


def main():
    partes = []
    for ciclo, nombre in ARCHIVOS.items():
        print(f"[{ciclo}] {nombre}")
        partes.append(cargar_año(ciclo, nombre))

    cp = pl.concat(partes).select(
        "ciclo", "id_ramo", "cve_ent", "monto_aprobado", "monto_ejercido"
    )

    # sanity: total 2024 ≈ 10.5 T MXN (DATA_OVERVIEW Key Insight #4)
    t24 = cp.filter(pl.col("ciclo") == 2024)["monto_ejercido"].sum() / 1e12
    assert 9.0 < t24 < 12.0, f"total 2024 fuera de rango: {t24:.2f} T"
    # sanity: entidades 1-32 presentes desde 2012
    ents = cp.filter((pl.col("ciclo") == 2024) & pl.col("cve_ent").is_between(1, 32))
    assert ents["cve_ent"].n_unique() == 32

    cp.write_parquet(SALIDA)
    print(f"\nOK → {SALIDA.relative_to(RAIZ)} ({cp.height:,} filas; 2024: {t24:.2f} T MXN)")


if __name__ == "__main__":
    main()
