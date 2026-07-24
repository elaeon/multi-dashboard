"""
Builds pre-aggregated tables for the "Nacionalidad" tabs of dashboard/sectur_hoteleria.py
from the SECTUR DataTur foreign-arrivals-by-nationality census (companion to the
extranjeros_residencia dataset — same schema/keys, Origen='Nacionalidad' instead of 'Residencia').

Source: data/sectur/extranjeros_nacionalidad/BD_Nacionalidad.xlsx
        (sheet BDUPM_Nac, 511,121 rows; grain: Año x MesNum x Aeropuerto x Pais x Sexo)
Output: dashboard_data/sectur_nacionalidad_mensual.parquet     (Año, MesNum, Región, Pais)
        dashboard_data/sectur_nacionalidad_aeropuerto.parquet  (Año, Aeropuerto, Pais)
Run: uv run python scripts/prepare_sectur_extranjeros_nacionalidad.py
"""
from pathlib import Path

import polars as pl

XLSX_PATH      = Path("data/sectur/extranjeros_nacionalidad/BD_Nacionalidad.xlsx")
OUT_MENSUAL    = Path("dashboard_data/sectur_nacionalidad_mensual.parquet")
OUT_AEROPUERTO = Path("dashboard_data/sectur_nacionalidad_aeropuerto.parquet")

print(f"Reading {XLSX_PATH}...")
raw = pl.read_excel(XLSX_PATH, sheet_id=1)
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
print(f"2019: {totals[2019]:,} (expected ~19.04M)")
print(f"2020: {totals[2020]:,} (expected ~7.94M)")
print(f"2024: {totals[2024]:,} (expected ~22.28M)")
print(f"2025: {totals[2025]:,} (expected ~22.13M)")

usa_share = mensual.filter(pl.col("Pais") == "Estados Unidos")["Valor"].sum() / mensual["Valor"].sum() * 100
print(f"USA share: {usa_share:.1f}% (expected ~60.0%)")

cancun_share = (
    aeropuerto.filter(pl.col("Aeropuerto").str.starts_with("Cancún"))["Valor"].sum()
    / aeropuerto["Valor"].sum() * 100
)
print(f"Cancún share (attributed): {cancun_share:.1f}% (expected ~43.9%)")

por_region_año = (
    mensual.filter(~pl.col("Región").is_in(["No especificado", "Apátrida"]))
    .group_by(["Región", "Año"]).agg(pl.col("Valor").sum())
)
base_2019 = por_region_año.filter(pl.col("Año") == 2019).select("Región", pl.col("Valor").alias("v2019"))
val_2025 = por_region_año.filter(pl.col("Año") == 2025).select("Región", pl.col("Valor").alias("v2025"))
recovery = base_2019.join(val_2025, on="Región").with_columns((pl.col("v2025") / pl.col("v2019") * 100).alias("idx2025"))
above_2019 = recovery.filter(pl.col("idx2025") > 100)["Región"].to_list()
print(f"Regions above 2019 level in 2025: {above_2019} (expected América del Norte, Asia, África — Asia/África only 'edge up' slightly)")
idx_norte = float(recovery.filter(pl.col("Región") == "América del Norte")["idx2025"][0])
print(f"América del Norte index 2025: {idx_norte:.1f} (expected ~131.5, i.e. +31.5%)")

assert 18.6e6 <= totals[2019] <= 19.4e6
assert 7.6e6 <= totals[2020] <= 8.2e6
assert 21.9e6 <= totals[2024] <= 22.6e6
assert 21.7e6 <= totals[2025] <= 22.5e6
assert 58.5 <= usa_share <= 61.5
assert 42.5 <= cancun_share <= 45.3
assert set(above_2019) == {"América del Norte", "Asia", "África"}
assert 129 <= idx_norte <= 134

OUT_MENSUAL.parent.mkdir(parents=True, exist_ok=True)
mensual.write_parquet(OUT_MENSUAL)
aeropuerto.write_parquet(OUT_AEROPUERTO)
print(f"Saved → {OUT_MENSUAL}")
print(f"Saved → {OUT_AEROPUERTO}")
