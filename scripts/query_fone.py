#!/usr/bin/env python3
"""
Query raw FONE payroll records by year and CURP.

Usage (run from project root):
    uv run python scripts/query_fone.py --year 2024 --curp GOTL490218HJCMRS05
    uv run python scripts/query_fone.py --year 2024 --curp GOTL490218HJCMRS05 ZAMR630919HNTTNB05
    uv run python scripts/query_fone.py --year 2024 --curp GOTL490218HJCMRS05 --out /tmp/result.csv
    uv run python scripts/query_fone.py --year 2024 --curp GOTL490218HJCMRS05 --agg
"""
import argparse
import glob
import sys

import polars as pl


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Consulta registros de PlazasDocAdmtvasDirec por año y CURP · FONE"
    )
    parser.add_argument("--year", type=int, required=True, help="Año (ej. 2024)")
    parser.add_argument("--curp", nargs="+", required=True, help="Uno o más CURPs")
    parser.add_argument("--agg",  action="store_true", help="Agregar por YEAR, QUARTER, TIPO_PLAZA y sumar percepciones")
    parser.add_argument("--out",  default="-", help="Ruta CSV de salida (default: stdout)")
    args = parser.parse_args()

    paths = sorted(glob.glob("data/fone/*/PlazasDocAdmtvasDirec_*.parquet"))
    if not paths:
        sys.exit("Error: no se encontraron archivos parquet en data/fone/")

    curps = [c.upper().strip() for c in args.curp]

    result = (
        pl.scan_parquet(paths)
        .filter(
            (pl.col("YEAR").cast(pl.Int32) == args.year) &
            pl.col("CURP").is_in(curps)
        )
        .sort(["CURP", "QUARTER"])
        .collect()
    )

    if result.is_empty():
        sys.exit(f"Sin resultados para year={args.year}, CURP(s)={curps}")

    if args.agg:
        result = (
            result
            .group_by(["CURP", "YEAR", "QUARTER", "TIPO_PLAZA"])
            .agg(pl.col("PERCEPCIONES_TRIMESTRALES").sum())
            .sort(["CURP", "YEAR", "QUARTER", "TIPO_PLAZA"])
        )

    print(f"# {len(result)} registros | year={args.year} | CURPs: {', '.join(curps)}", file=sys.stderr)

    if args.out == "-":
        sys.stdout.write(result.write_csv())
    else:
        result.write_csv(args.out)
        print(f"Escrito: {args.out} ({len(result)} filas)", file=sys.stderr)


if __name__ == "__main__":
    main()
