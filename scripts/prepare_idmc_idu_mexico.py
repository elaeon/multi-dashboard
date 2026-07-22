"""
Filters the IDMC IDU recent-events feed to Mexico-only rows.
Output: data/idmc/idu-data_2026_mex.csv
Run: uv run python scripts/prepare_idmc_idu_mexico.py
"""
from pathlib import Path

import polars as pl

JSON_PATH = Path("data/idmc/idu-data_2026.json")
OUT_PATH  = Path("data/idmc/idu-data_2026_mex.csv")

df = pl.read_json(JSON_PATH)
print(f"Read {df.height} rows from {JSON_PATH}")

mex = df.filter(pl.col("iso3") == "MEX")
print(f"Filtered to {mex.height} Mexico rows")

mex.write_csv(OUT_PATH)
print(f"Wrote {OUT_PATH}")
