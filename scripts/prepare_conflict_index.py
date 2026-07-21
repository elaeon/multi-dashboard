"""
Índice de Conflicto Municipal.

Índice A (2015-2024, cobertura completa ~2,500 municipios): 4 pilares
construidos ÚNICAMENTE a partir de SESNSP Incidencia Delictiva Fuero Común
(única fuente con grano municipal y sin huecos temporales), normalizados por
percentil dentro de cada año y promediados en un score compuesto 0-100.
Pilares adaptados de los 4 de ACLED (Deadliness/Danger/Fragmentation/
Diffusion) a lo que existe a nivel municipio — "difusión espacial" no aplica
dentro de un solo municipio, se sustituye por persistencia temporal:

    letalidad    = homicidio doloso / población x 100,000      (Deadliness)
    dano_civil   = (feminicidio + secuestro) / población x 1e5 (Danger)
    coercion     = extorsión / población x 100,000             (proxy)
    persistencia = % de meses del año con >=1 caso              (Diffusion)

Sub-índice B (2007-2018, solo municipios con evento OCVED nombrado):
fragmentación de actores criminales, tabla SEPARADA — nunca fusionada ni
comparada con el Índice A. Reusa informe_data/grupos_panel_ocved.parquet
(ver informe_data/README_grupos_criminales.md: presencia != dominio, y B no
es comparable con años posteriores a 2018).

Este índice no es un dato de dashboard (no alimenta ninguna página de
dashboard/): la carpeta de salida se especifica por línea de comandos.

Run: uv run python scripts/prepare_conflict_index.py --output-dir <ruta>
"""
import argparse
import io
import zipfile
from pathlib import Path

import polars as pl

RAIZ = Path(__file__).resolve().parent.parent

INCIDENCIA = (RAIZ / "data/incidencia_delictiva/incidencia_fuero_comun/"
              "incidencia_delictiva_fuero_comun.parquet")
CONAPO_ZIP = RAIZ / "data/conapo/proyecciones_poblacion/00_Republica_mexicana.zip"
CONAPO_INNER = "00_Republica_mexicana/3_Indicadores_Dem_00_RM.xlsx"
GRUPOS_OCVED = RAIZ / "informe_data/grupos_panel_ocved.parquet"

MESES = ["Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio",
          "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"]

YEAR_MIN, YEAR_MAX = 2015, 2024

# Peso igual por pilar — supuesto documentado y ajustable, no un valor derivado.
PESOS = {"pct_letalidad": 0.25, "pct_dano_civil": 0.25,
         "pct_coercion": 0.25, "pct_persistencia": 0.25}

NIVELES = ["Bajo", "Medio", "Alto", "Extremo"]


def cargar_poblacion() -> pl.DataFrame:
    with zipfile.ZipFile(CONAPO_ZIP) as z, z.open(CONAPO_INNER) as f:
        buf = io.BytesIO(f.read())
    raw = pl.read_excel(buf)
    return (raw.filter(pl.col("AÑO").is_between(YEAR_MIN, YEAR_MAX))
            .select(pl.col("CLAVE").cast(pl.Int64).alias("cve_mun"),
                    pl.col("AÑO").alias("anio"),
                    pl.col("POB_MIT_MUN").alias("poblacion")))


def cargar_catalogo_municipios() -> pl.DataFrame:
    """cve_mun -> nombre de municipio/entidad, para anotar ambas tablas."""
    return (pl.scan_parquet(INCIDENCIA)
            .select(pl.col("Cve. Municipio").cast(pl.Int64).alias("cve_mun"),
                    pl.col("Municipio").alias("municipio"),
                    pl.col("Entidad").alias("entidad"))
            .unique(subset="cve_mun")
            .collect())


def construir_indice_base() -> pl.DataFrame:
    lf = pl.scan_parquet(INCIDENCIA).filter(pl.col("Año").is_between(YEAR_MIN, YEAR_MAX))

    componente = (
        pl.when(pl.col("Subtipo de delito") == "Homicidio doloso").then(pl.lit("homicidio"))
        .when(pl.col("Subtipo de delito") == "Feminicidio").then(pl.lit("dano_civil"))
        .when(pl.col("Tipo de delito") == "Secuestro").then(pl.lit("dano_civil"))
        .when(pl.col("Tipo de delito") == "Extorsión").then(pl.lit("coercion"))
        .otherwise(None)
        .alias("componente")
    )

    largo = (
        lf.with_columns(componente)
        .filter(pl.col("componente").is_not_null())
        .group_by(["Año", "Cve. Municipio", "componente"])
        .agg([pl.col(m).sum().alias(m) for m in MESES])
        .collect()
        .unpivot(on=MESES, index=["Año", "Cve. Municipio", "componente"],
                 variable_name="mes", value_name="casos")
    )

    anual = (
        largo.group_by(["Año", "Cve. Municipio", "componente"])
        .agg(pl.col("casos").sum().alias("casos_anual"))
        .pivot(on="componente", index=["Año", "Cve. Municipio"], values="casos_anual")
        .fill_null(0)
        .rename({"Año": "anio", "Cve. Municipio": "cve_mun"})
    )

    persistencia = (
        largo.group_by(["Año", "Cve. Municipio", "mes"])
        .agg(pl.col("casos").sum().alias("casos_mes"))
        .with_columns((pl.col("casos_mes") > 0).alias("activo"))
        .group_by(["Año", "Cve. Municipio"])
        .agg((pl.col("activo").sum() / 12).alias("persistencia_meses"))
        .rename({"Año": "anio", "Cve. Municipio": "cve_mun"})
    )

    return anual.join(persistencia, on=["anio", "cve_mun"], how="left")


def calcular_indice(base: pl.DataFrame, poblacion: pl.DataFrame,
                     catalogo: pl.DataFrame) -> pl.DataFrame:
    df = (base.join(poblacion, on=["cve_mun", "anio"], how="inner")
          .join(catalogo, on="cve_mun", how="left")
          .with_columns(
              pl.col("cve_mun").cast(pl.Int64).__floordiv__(1000).alias("cve_ent"),
              (pl.col("homicidio") / pl.col("poblacion") * 1e5).alias("homicidio_rate"),
              (pl.col("dano_civil") / pl.col("poblacion") * 1e5).alias("dano_civil_rate"),
              (pl.col("coercion") / pl.col("poblacion") * 1e5).alias("extorsion_rate"),
          ))

    for raw_col, pct_col in [("homicidio_rate", "pct_letalidad"),
                              ("dano_civil_rate", "pct_dano_civil"),
                              ("extorsion_rate", "pct_coercion"),
                              ("persistencia_meses", "pct_persistencia")]:
        df = df.with_columns(
            (pl.col(raw_col).rank(method="average").over("anio")
             / pl.len().over("anio") * 100).alias(pct_col))

    df = df.with_columns(
        sum(pl.col(c) * w for c, w in PESOS.items()).alias("indice_conflicto")
    ).with_columns(
        pl.col("indice_conflicto").qcut([0.25, 0.5, 0.75], labels=NIVELES)
        .over("anio").alias("nivel")
    )

    return df.select(
        "cve_ent", "cve_mun", "municipio", "entidad", "anio", "poblacion",
        "homicidio_rate", "dano_civil_rate", "extorsion_rate", "persistencia_meses",
        "pct_letalidad", "pct_dano_civil", "pct_coercion", "pct_persistencia",
        "indice_conflicto", "nivel",
    ).sort(["anio", "indice_conflicto"], descending=[False, True])


def construir_subindice_fragmentacion(catalogo: pl.DataFrame) -> pl.DataFrame:
    df = pl.read_parquet(GRUPOS_OCVED)
    base = (df.group_by(["cve_ent", "cve_mun", "anio"])
            .agg(pl.col("n_grupos_mun_anio").first().alias("n_grupos"),
                 pl.col("eventos_totales").first().alias("eventos"),
                 pl.col("control_proxy").first()))

    base = base.with_columns(
        (pl.col("n_grupos").rank(method="average").over("anio")
         / pl.len().over("anio") * 100).alias("pct_fragmentacion")
    ).with_columns(pl.col("cve_mun").cast(pl.Int64))

    return (base.join(catalogo, on="cve_mun", how="left")
            .select("cve_ent", "cve_mun", "municipio", "anio", "n_grupos",
                    "control_proxy", "eventos", "pct_fragmentacion")
            .sort(["anio", "pct_fragmentacion"], descending=[False, True]))


def validar(indice: pl.DataFrame, ultimo_anio: int, poblacion: pl.DataFrame) -> None:
    cobertura = (indice.filter(pl.col("anio") == ultimo_anio)["poblacion"].sum()
                 / poblacion.filter(pl.col("anio") == ultimo_anio)["poblacion"].sum())
    print(f"\nCobertura poblacional {ultimo_anio}: {cobertura * 100:.1f}%")
    assert cobertura > 0.90, "cobertura poblacional por debajo del umbral esperado"

    print(f"\nDistribución de niveles ({ultimo_anio}):")
    print(indice.filter(pl.col("anio") == ultimo_anio)["nivel"].value_counts().sort("nivel"))

    print(f"\nTop 10 municipios por índice de conflicto ({ultimo_anio}):")
    print(indice.filter(pl.col("anio") == ultimo_anio)
          .head(10)
          .select("municipio", "entidad", "homicidio_rate", "indice_conflicto", "nivel"))

    print(f"\nPromedio de índice por entidad ({ultimo_anio}), top 10:")
    print(indice.filter(pl.col("anio") == ultimo_anio)
          .group_by("entidad")
          .agg(pl.col("indice_conflicto").mean().alias("indice_promedio"))
          .sort("indice_promedio", descending=True)
          .head(10))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path,
                         help="Carpeta donde escribir los parquet de salida "
                              "(este índice no es dato de dashboard, no tiene default).")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Cargando población municipal (CONAPO)...")
    poblacion = cargar_poblacion()

    print("Cargando catálogo de municipios...")
    catalogo = cargar_catalogo_municipios()

    print(f"Construyendo Índice A ({YEAR_MIN}-{YEAR_MAX})...")
    base = construir_indice_base()
    indice = calcular_indice(base, poblacion, catalogo)

    out_a = args.output_dir / "conflict_index_municipio_anio.parquet"
    indice.write_parquet(out_a)
    print(f"Índice A: {indice.height:,} filas -> {out_a}")

    print("\nConstruyendo Sub-índice B (fragmentación de actores, 2007-2018)...")
    fragmentacion = construir_subindice_fragmentacion(catalogo)
    out_b = args.output_dir / "conflict_index_fragmentacion_2007_2018.parquet"
    fragmentacion.write_parquet(out_b)
    print(f"Sub-índice B: {fragmentacion.height:,} filas -> {out_b}")

    validar(indice, YEAR_MAX, poblacion)


if __name__ == "__main__":
    main()
