import polars as pl
from pathlib import Path

INPUT = Path("data/profeco/casas_empeño/registro_casas_empeno_2014_2025.csv")

df = pl.read_csv(INPUT)
df = df.with_columns(
    pl.col("domicilio")
      .str.split(",")
      .list.last()
      .str.strip_chars()
      .alias("estado")
)
df.write_csv(INPUT)
print(f"Done: {len(df):,} rows, {df['estado'].n_unique()} unique states")
print(df["estado"].value_counts().sort("count", descending=True))
