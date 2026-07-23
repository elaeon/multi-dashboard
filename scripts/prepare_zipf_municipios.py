"""
Builds the rank-size table for dashboard/zipf_municipios.py: municipios ranked
by population within each year (2017-2025), plus the theoretical Zipf reference
(pop[rank=1, year] / rank) computed per year.

Source: dashboard_data/conapo_pob_municipal.parquet
Output: dashboard_data/zipf_municipios.parquet
Run: uv run python scripts/prepare_zipf_municipios.py
"""
from pathlib import Path

import polars as pl

IN_PATH  = Path("dashboard_data/conapo_pob_municipal.parquet")
OUT_PATH = Path("dashboard_data/zipf_municipios.parquet")

YEAR_MIN, YEAR_MAX = 2017, 2025

df = pl.read_parquet(IN_PATH)

d = (
    df.filter(pl.col("AÑO").is_between(YEAR_MIN, YEAR_MAX))
    .select(["CLAVE", "CLAVE_ENT", "NOM_ENT", "NOM_MUN", "AÑO", "POB_TOTAL"])
    .with_columns(
        pl.col("POB_TOTAL").rank(method="ordinal", descending=True).over("AÑO").alias("rank"),
        pl.col("POB_TOTAL").max().over("AÑO").alias("_pob_rank1"),
    )
    .with_columns((pl.col("_pob_rank1") / pl.col("rank")).alias("pob_ideal_zipf"))
    .drop("_pob_rank1")
    .sort(["AÑO", "rank"])
)

n_years = YEAR_MAX - YEAR_MIN + 1
print(f"Rows: {d.height:,} (expected {2475 * n_years:,} = 2,475 municipios × {n_years} años)")

top_2025 = d.filter((pl.col("AÑO") == 2025) & (pl.col("rank") == 1))
print(f"2025 rank 1: {top_2025.item(0, 'NOM_MUN')} ({top_2025.item(0, 'POB_TOTAL'):,})")

assert d.height == 2475 * n_years
assert top_2025.item(0, "NOM_MUN") == "Tijuana"

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
d.write_parquet(OUT_PATH)
print(f"Saved → {OUT_PATH}")
