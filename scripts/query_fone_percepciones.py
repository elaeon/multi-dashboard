#!/usr/bin/env python3
"""
Get CURPs in the P99 salary percentile from FONE payroll data.

Usage (run from project root):
    uv run python scripts/query_fone_percepciones.py --year 2024 --estado JALISCO
    uv run python scripts/query_fone_percepciones.py --year 2025 --estado "CIUDAD DE MEXICO" --tipo Docente
    uv run python scripts/query_fone_percepciones.py --year 2024 --estado OAXACA --out /tmp/oaxaca_p99.csv
"""
import argparse
import glob
import sys

import polars as pl


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CURPs en el percentil 99 de percepciones anuales · FONE"
    )
    parser.add_argument("--year",   type=int, required=True, help="Año (ej. 2024)")
    parser.add_argument("--estado", required=True,           help="Entidad federativa (ej. JALISCO)")
    parser.add_argument("--tipo",   default=None,            help="Tipo de plaza (opcional, ej. Docente)")
    parser.add_argument("--out",    default="-",             help="Ruta CSV de salida (default: stdout)")
    args = parser.parse_args()

    paths = sorted(glob.glob("data/fone/*/PlazasDocAdmtvasDirec_*.parquet"))
    if not paths:
        sys.exit("Error: no se encontraron archivos parquet en data/fone/")

    estado = args.estado.upper().strip()

    lf = (
        pl.scan_parquet(paths)
        .filter(
            (pl.col("YEAR").cast(pl.Int32) == args.year) &
            (pl.col("ENTIDAD_FEDERATIVA") == estado)
        )
    )
    if args.tipo:
        lf = lf.filter(pl.col("TIPO_PLAZA") == args.tipo)

    totals = (
        lf.group_by(["CURP", "ENTIDAD_FEDERATIVA", "TIPO_PLAZA"])
        .agg(
            pl.col("NOMBRE").first(),
            pl.col("RFC").first(),
            pl.col("PERCEPCIONES_TRIMESTRALES").sum().alias("PERCEPCIONES_ANUALES"),
        )
        .filter(pl.col("PERCEPCIONES_ANUALES") > 0)
        .collect()
    )

    if totals.is_empty():
        label = f"year={args.year}, estado={estado}"
        if args.tipo:
            label += f", tipo={args.tipo}"
        sys.exit(f"Sin datos para {label}")

    p99 = float(totals["PERCEPCIONES_ANUALES"].quantile(0.99))

    result = (
        totals
        .filter(pl.col("PERCEPCIONES_ANUALES") >= p99)
        .select(["CURP", "NOMBRE", "RFC", "ENTIDAD_FEDERATIVA", "TIPO_PLAZA", "PERCEPCIONES_ANUALES"])
        .sort("PERCEPCIONES_ANUALES", descending=True)
    )

    print(
        f"# P99 = {p99:,.2f} MXN | {len(result)} CURPs | "
        f"year={args.year} estado={estado}" + (f" tipo={args.tipo}" if args.tipo else ""),
        file=sys.stderr,
    )

    if args.out == "-":
        sys.stdout.write(result.write_csv())
    else:
        result.write_csv(args.out)
        print(f"Escrito: {args.out} ({len(result)} filas)", file=sys.stderr)


if __name__ == "__main__":
    main()
