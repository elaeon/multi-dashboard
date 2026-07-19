"""
Extracts municipality-level population (2017-2025) from the CONAPO
proyecciones ZIP, replacing the deleted data/conapo/estados_municipios.csv.

Source: data/conapo/proyecciones_poblacion/00_Republica_mexicana.zip
        -> 3_Indicadores_Dem_00_RM.xlsx (grain: municipality x year,
           POB_MIT_MUN already sums both sexes -> renamed to POB_TOTAL)

Shared by dashboard/internal_migration_flow.py (2017-2024) and
dashboard/imss_directorio.py (2024, 2025).

Output: dashboard_data/conapo_pob_municipal.parquet
Run: uv run python scripts/prepare_conapo_pob_municipal.py
"""
import io
import zipfile
from pathlib import Path

import polars as pl

ZIP_PATH = Path("data/conapo/proyecciones_poblacion/00_Republica_mexicana.zip")
INNER    = "00_Republica_mexicana/3_Indicadores_Dem_00_RM.xlsx"
OUT_PATH = Path("dashboard_data/conapo_pob_municipal.parquet")

YEAR_MIN, YEAR_MAX = 2017, 2025

_STATE_RENAME = {
    "Michoacán de Ocampo":             "Michoacán",
    "Coahuila de Zaragoza":            "Coahuila",
    "Veracruz de Ignacio de la Llave": "Veracruz",
}

print(f"Reading {INNER} from ZIP...")
with zipfile.ZipFile(ZIP_PATH) as z:
    with z.open(INNER) as f:
        buf = io.BytesIO(f.read())

raw = pl.read_excel(buf)
print(f"Raw rows: {raw.height:,}, columns: {raw.width}")

pob_municipal = (
    raw
    .filter(pl.col("AÑO").is_between(YEAR_MIN, YEAR_MAX))
    .select(["CLAVE", "CLAVE_ENT", "NOM_ENT", "NOM_MUN", "AÑO", "POB_MIT_MUN"])
    .rename({"POB_MIT_MUN": "POB_TOTAL"})
    .with_columns(pl.col("NOM_ENT").replace(_STATE_RENAME))
    .sort(["CLAVE", "AÑO"])
)

n_years = YEAR_MAX - YEAR_MIN + 1
print(f"Municipal-year rows: {pob_municipal.height:,} (expected ~{2475 * n_years:,} = 2,475 municipios × {n_years} años)")

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
pob_municipal.write_parquet(OUT_PATH)
print(f"Saved → {OUT_PATH}")
