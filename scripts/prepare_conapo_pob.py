"""
Extracts state-year population data from CONAPO proyecciones ZIP.
Output: data/conapo/proyecciones_poblacion/pob_estado_año.parquet (1,632 rows)
Run: uv run python scripts/prepare_conapo_pob.py
"""
import io
import zipfile
from pathlib import Path

import polars as pl

ZIP_PATH = Path("data/conapo/proyecciones_poblacion/00_Republica_mexicana.zip")
INNER    = "00_Republica_mexicana/3_Indicadores_Dem_00_RM.xlsx"
OUT_PATH = Path("data/conapo/proyecciones_poblacion/pob_estado_año.parquet")

CONAPO_TO_ESTADO = {
    "Aguascalientes":                  "Aguascalientes",
    "Baja California":                 "Baja California",
    "Baja California Sur":             "Baja California Sur",
    "Campeche":                        "Campeche",
    "Chiapas":                         "Chiapas",
    "Chihuahua":                       "Chihuahua",
    "Ciudad de México":                "Ciudad De Mexico",
    "Coahuila de Zaragoza":            "Coahuila",
    "Colima":                          "Colima",
    "Durango":                         "Durango",
    "Guanajuato":                      "Guanajuato",
    "Guerrero":                        "Guerrero",
    "Hidalgo":                         "Hidalgo",
    "Jalisco":                         "Jalisco",
    "México":                          "Mexico",
    "Michoacán de Ocampo":             "Michoacan",
    "Morelos":                         "Morelos",
    "Nayarit":                         "Nayarit",
    "Nuevo León":                      "Nuevo Leon",
    "Oaxaca":                          "Oaxaca",
    "Puebla":                          "Puebla",
    "Querétaro":                       "Queretaro",
    "Quintana Roo":                    "Quintana Roo",
    "San Luis Potosí":                 "San Luis Potosi",
    "Sinaloa":                         "Sinaloa",
    "Sonora":                          "Sonora",
    "Tabasco":                         "Tabasco",
    "Tamaulipas":                      "Tamaulipas",
    "Tlaxcala":                        "Tlaxcala",
    "Veracruz de Ignacio de la Llave": "Veracruz",
    "Yucatán":                         "Yucatan",
    "Zacatecas":                       "Zacatecas",
}

print(f"Reading {INNER} from ZIP...")
with zipfile.ZipFile(ZIP_PATH) as z:
    with z.open(INNER) as f:
        buf = io.BytesIO(f.read())

raw = pl.read_excel(buf)
print(f"Raw rows: {raw.height:,}, columns: {raw.width}")

state_year = (
    raw
    .group_by(["NOM_ENT", "AÑO"])
    .agg([
        pl.first("POB_MIT_ENT").alias("pob_total"),
        pl.col("POB_15_64").sum().alias("pob_15_64"),
        (pl.col("EDAD_MED").cast(pl.Float64) * pl.col("POB_MIT_MUN")).sum().alias("_edad_wtd"),
        pl.col("POB_MIT_MUN").sum().alias("_pob_mun"),
        pl.col("POB_00_14").sum().alias("_pob_00_14"),
        pl.col("POB_65_MAS").sum().alias("_pob_65_mas"),
        pl.col("POB_60_MAS").sum().alias("_pob_60_mas"),
    ])
    .with_columns([
        (pl.col("_edad_wtd") / pl.col("_pob_mun")).round(1).alias("edad_med"),
        ((pl.col("_pob_00_14") + pl.col("_pob_65_mas")) / pl.col("pob_15_64") * 100).round(2).alias("raz_dep"),
        (pl.col("_pob_60_mas") / pl.col("_pob_00_14") * 100).round(2).alias("ind_env_60"),
    ])
    .drop(["_edad_wtd", "_pob_mun", "_pob_00_14", "_pob_65_mas", "_pob_60_mas"])
    .with_columns([
        pl.col("NOM_ENT").replace(CONAPO_TO_ESTADO).alias("estado"),
        pl.col("AÑO").cast(pl.Int32).alias("año"),
    ])
    .select(["estado", "año", "pob_total", "pob_15_64", "edad_med", "raz_dep", "ind_env_60"])
    .sort(["estado", "año"])
)

print(f"State-year rows: {state_year.height:,}  (expected 1,632 = 32 × 51)")

unmapped = state_year.filter(~pl.col("estado").is_in(list(CONAPO_TO_ESTADO.values())))
if unmapped.height > 0:
    print(f"WARNING: {unmapped.height} unmapped state names: {unmapped['estado'].unique().to_list()}")

state_year.write_parquet(OUT_PATH)
print(f"Saved → {OUT_PATH}")
