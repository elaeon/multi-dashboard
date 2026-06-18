#!/usr/bin/env python3
"""
Consolidate downloaded FONE ZIP files into one Parquet per table per year.

Input:  data/fone/<year>/Q<n>/<state>/<TableName>_<StateCode>_Trimestre_<QQ>_<YYYY>.zip
Output: data/fone/<year>/<TableName>_<year>.parquet

Usage:
  python scripts/consolidate_fone.py                          # all years, all tables
  python scripts/consolidate_fone.py --year 2025              # one year
  python scripts/consolidate_fone.py --table AnaliticoPlazas  # one table
  python scripts/consolidate_fone.py --year 2024 --table PlazasDocAdmtvasDirec
"""

import argparse
import io
import zipfile
from pathlib import Path

import polars as pl

DATA_DIR = Path("data/fone")

TABLES = [
    "AnaliticoPlazas",
    "PlazasDocAdmtvasDirec",
    "MovimientosPlaza",
    "PersonalLicencias",
]

# Columns that carry a leading apostrophe artifact from Excel export
APOSTROPHE_COLS: dict[str, list[str]] = {
    "AnaliticoPlazas":       ["CATEGORIA"],
    "PlazasDocAdmtvasDirec": ["NIVEL_CATEGORIA"],
    "PersonalLicencias":     ["CLAVE_LICENCIA"],
    "MovimientosPlaza":      [],
}

# Columns known to be 100% null for certain tables — drop them
DROP_NULL_COLS: dict[str, list[str]] = {
    "AnaliticoPlazas": ["MONTO_MENSUAL_ZONA_3"],
}


def detect_separator(sample: bytes) -> str:
    """Guess CSV separator from the first line."""
    first_line = sample.split(b"\n")[0].decode("latin-1", errors="replace")
    return "|" if first_line.count("|") > first_line.count(",") else ","


def read_zip_csv(zip_path: Path) -> pl.DataFrame:
    """Extract and read the single CSV inside a ZIP file."""
    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not names:
            raise ValueError(f"No CSV found in {zip_path}")
        raw = zf.read(names[0])

    sep = detect_separator(raw[:4096])
    return pl.read_csv(
        io.BytesIO(raw),
        encoding="latin1",
        separator=sep,
        infer_schema_length=5000,
        ignore_errors=True,
    )


def clean(df: pl.DataFrame, table: str) -> pl.DataFrame:
    """Apply table-specific cleanups."""
    # Strip apostrophe prefix from key columns
    for col in APOSTROPHE_COLS.get(table, []):
        if col in df.columns:
            df = df.with_columns(pl.col(col).str.strip_chars_start("'"))

    # Strip whitespace from CLAVE_PLAZA (all tables); cast to str first if inferred as numeric
    if "CLAVE_PLAZA" in df.columns:
        if df["CLAVE_PLAZA"].dtype != pl.String:
            df = df.with_columns(pl.col("CLAVE_PLAZA").cast(pl.String))
        df = df.with_columns(pl.col("CLAVE_PLAZA").str.strip_chars())

    # Drop known-null columns
    for col in DROP_NULL_COLS.get(table, []):
        if col in df.columns:
            df = df.drop(col)

    # Cast numeric columns to consistent types
    for col, dtype in [("NUMERO_HORAS", pl.Float32), ("PERCEPCIONES_TRIMESTRALES", pl.Float64)]:
        if col in df.columns and df[col].dtype == pl.String:
            df = df.with_columns(pl.col(col).cast(dtype, strict=False))

    return df


def add_period_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Parse TRIMESTRE ('2025/01') into YEAR and QUARTER int columns."""
    if "TRIMESTRE" not in df.columns:
        return df
    parts = df["TRIMESTRE"].str.split("/")
    return df.with_columns(
        parts.list.get(0).cast(pl.Int16).alias("YEAR"),
        parts.list.get(1).cast(pl.Int8).alias("QUARTER"),
    )


def consolidate(year: int, table: str) -> None:
    # Use *<table>* to also match ZIPs with a random-ID prefix (e.g. "abc123-AnaliticoPlazas_...")
    zip_paths = sorted(DATA_DIR.glob(f"{year}/Q*/*/*{table}*.zip"))
    # Also match accent variant (e.g. AnalíticoPlazas in some 2024 Q1 states)
    if table == "AnaliticoPlazas":
        zip_paths = sorted(set(zip_paths) | set(DATA_DIR.glob(f"{year}/Q*/*/*naliticoPlazas*.zip")))
        zip_paths = sorted(zip_paths)

    if not zip_paths:
        print(f"  [{year}] {table}: no ZIPs found — skipping")
        return

    frames = []
    for zp in zip_paths:
        try:
            df = read_zip_csv(zp)
            df = clean(df, table)
            frames.append(df)
        except Exception as e:
            print(f"  WARN: {zp.name}: {e}")

    if not frames:
        print(f"  [{year}] {table}: all ZIPs failed — skipping")
        return

    combined = pl.concat(frames, how="diagonal_relaxed")
    combined = add_period_columns(combined)

    out_path = DATA_DIR / str(year) / f"{table}_{year}.parquet"
    combined.write_parquet(out_path, compression="zstd")

    rows = len(combined)
    size_mb = out_path.stat().st_size / 1_048_576
    states = combined["ENTIDAD_FEDERATIVA"].n_unique() if "ENTIDAD_FEDERATIVA" in combined.columns else "?"
    quarters = combined["QUARTER"].n_unique() if "QUARTER" in combined.columns else "?"
    print(f"  [{year}] {table}: {rows:,} filas, {states} estados, {quarters} trimestres → {out_path.name} ({size_mb:.1f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Consolidate FONE ZIPs into yearly Parquet files")
    parser.add_argument("--year", type=int, help="Filter to specific year")
    parser.add_argument("--table", choices=TABLES, help="Filter to specific table")
    args = parser.parse_args()

    years = [args.year] if args.year else sorted(
        int(p.name) for p in DATA_DIR.iterdir() if p.is_dir() and p.name.isdigit()
    )
    tables = [args.table] if args.table else TABLES

    for year in years:
        for table in tables:
            consolidate(year, table)

    print("\nDone.")


if __name__ == "__main__":
    main()
