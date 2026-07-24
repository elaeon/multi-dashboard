"""
Builds pre-aggregated tables for the "Extranjeros" tabs of dashboard/sectur_hoteleria.py
from the SECTUR DataTur foreign-arrivals-by-country-of-residence census.

Source: data/sectur/extranjeros_residencia/BD_Residencia.zip -> BD_Residencia.xlsx
        (sheet BDUPM_Res, 432,583 rows; grain: Año x MesNum x Aeropuerto x Pais x Sexo)
Output: dashboard_data/sectur_extranjeros_mensual.parquet     (Año, MesNum, Región, Pais)
        dashboard_data/sectur_extranjeros_aeropuerto.parquet  (Año, Aeropuerto, Pais)
Run: uv run python scripts/prepare_sectur_extranjeros_residencia.py
"""
import zipfile
from pathlib import Path

import polars as pl

ZIP_PATH   = Path("data/sectur/extranjeros_residencia/BD_Residencia.zip")
INNER      = "BD_Residencia.xlsx"
OUT_MENSUAL     = Path("dashboard_data/sectur_extranjeros_mensual.parquet")
OUT_AEROPUERTO  = Path("dashboard_data/sectur_extranjeros_aeropuerto.parquet")

print(f"Reading {INNER} from {ZIP_PATH}...")
with zipfile.ZipFile(ZIP_PATH) as z:
    with z.open(INNER) as f:
        raw = pl.read_excel(f.read(), sheet_id=1)

print(f"Raw rows: {raw.height:,}")
raw = raw.drop(["Origen", "Fecha"])

mensual = (
    raw.group_by(["Año", "MesNum", "Región", "Pais"])
    .agg(pl.col("Valor").sum())
    .sort(["Año", "MesNum"])
)

aeropuerto = (
    raw.filter(pl.col("Aeropuerto") != "No especificado")
    .group_by(["Año", "Aeropuerto", "Pais"])
    .agg(pl.col("Valor").sum())
    .sort(["Año", "Aeropuerto"])
)

print(f"Mensual rows: {mensual.height:,}, Aeropuerto rows: {aeropuerto.height:,}")

# ── Validation against DATA_OVERVIEW.md Key Insights ────────────────────────
by_year = mensual.group_by("Año").agg(pl.col("Valor").sum()).sort("Año")
totals = {row["Año"]: row["Valor"] for row in by_year.iter_rows(named=True)}
print(f"2019: {totals[2019]:,} (expected ~18.46M)")
print(f"2020: {totals[2020]:,} (expected ~7.70M)")
print(f"2024: {totals[2024]:,} (expected ~20.50M)")
print(f"2025: {totals[2025]:,} (expected ~20.60M)")

usa_share = mensual.filter(pl.col("Pais") == "Estados Unidos")["Valor"].sum() / mensual["Valor"].sum() * 100
print(f"USA share: {usa_share:.1f}% (expected ~63.2%)")

cancun_share = (
    aeropuerto.filter(pl.col("Aeropuerto").str.starts_with("Cancún"))["Valor"].sum()
    / aeropuerto["Valor"].sum() * 100
)
print(f"Cancún share (attributed): {cancun_share:.1f}% (expected ~44.7%)")

por_region_año = (
    mensual.filter(pl.col("Región") != "No especificado")
    .group_by(["Región", "Año"]).agg(pl.col("Valor").sum())
)
base_2019 = por_region_año.filter(pl.col("Año") == 2019).select("Región", pl.col("Valor").alias("v2019"))
val_2024 = por_region_año.filter(pl.col("Año") == 2024).select("Región", pl.col("Valor").alias("v2024"))
recovery = base_2019.join(val_2024, on="Región").with_columns((pl.col("v2024") / pl.col("v2019") * 100).alias("idx2024"))
above_2019 = recovery.filter(pl.col("idx2024") > 100)["Región"].to_list()
print(f"Regions above 2019 level in 2024: {above_2019} (expected only América del Norte)")

assert 18.0e6 <= totals[2019] <= 18.9e6
assert 7.4e6 <= totals[2020] <= 8.0e6
assert 20.2e6 <= totals[2024] <= 20.8e6
assert 20.3e6 <= totals[2025] <= 20.9e6
assert 62.0 <= usa_share <= 64.5
assert 43.5 <= cancun_share <= 45.9
assert above_2019 == ["América del Norte"]

OUT_MENSUAL.parent.mkdir(parents=True, exist_ok=True)
mensual.write_parquet(OUT_MENSUAL)
aeropuerto.write_parquet(OUT_AEROPUERTO)
print(f"Saved → {OUT_MENSUAL}")
print(f"Saved → {OUT_AEROPUERTO}")
