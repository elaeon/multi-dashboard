import argparse
import glob
from pathlib import Path
import polars as pl

parser = argparse.ArgumentParser(description="Merge incidencia delictiva xlsx files into a parquet file.")
parser.add_argument("input_dir", type=Path, help="Directory containing xlsx files")
parser.add_argument("output", type=Path, help="Output parquet file path")
args = parser.parse_args()

files = sorted(glob.glob(str(args.input_dir / "*.xlsx")))
if not files:
    raise FileNotFoundError(f"No xlsx files found in {args.input_dir}")

MONTHS = ["Enero","Febrero","Marzo","Abril","Mayo","Junio",
          "Julio","Agosto","Septiembre","Octubre","Noviembre","Diciembre"]
SCHEMA = {"Clave_Ent": pl.Int32, "Cve. Municipio": pl.Int32, **{m: pl.Int32 for m in MONTHS}}

frames = []
for path in files:
    df = pl.read_excel(path, sheet_id=1, schema_overrides=SCHEMA)
    frames.append(df)

merged = pl.concat(frames, how="diagonal_relaxed")
merged.write_parquet(args.output)
print(f"Merged {len(files)} files → {args.output} ({len(merged):,} rows)")
