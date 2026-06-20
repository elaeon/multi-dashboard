#!/usr/bin/env python3
"""
Query FONE payroll data by percentile threshold.

Usage (run from project root):
    # Totals by tipo de plaza (--total is default when no -p)
    uv run python scripts/query_fone_percepciones.py -y 2024
    uv run python scripts/query_fone_percepciones.py -y 2024 -e JALISCO

    # Individual CURPs at or above a percentile
    uv run python scripts/query_fone_percepciones.py -y 2024 -p 99
    uv run python scripts/query_fone_percepciones.py -y 2024 -p 99 -e JALISCO
    uv run python scripts/query_fone_percepciones.py -y 2024 -p 90 -e JALISCO --tipo Docente

    # Totals for CURPs at or above a percentile
    uv run python scripts/query_fone_percepciones.py -y 2024 -p 99 --total
    uv run python scripts/query_fone_percepciones.py -y 2024 -p 95 -e JALISCO --total
"""
import argparse
import glob
import sys

import polars as pl

_PCTILE_MAP = {50: 0.50, 90: 0.90, 95: 0.95, 99: 0.99}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Percepciones FONE por percentil · FONE"
    )
    parser.add_argument("-y", "--year",      type=int, required=True,
                        help="Año (ej. 2024)")
    parser.add_argument("-p", "--percentil", type=int, choices=[50, 90, 95, 99], default=None,
                        help="Percentil umbral (50/90/95/99). Sin valor: totales.")
    parser.add_argument("-e", "--estado",    default=None,
                        help="Entidad federativa (ej. JALISCO). Opcional.")
    parser.add_argument("--tipo",            default=None,
                        help="Tipo de plaza (ej. Docente). Opcional.")
    parser.add_argument("--total",           action="store_true",
                        help="Agregar totales por tipo de plaza. Default cuando -p no se especifica.")
    parser.add_argument("--out",             default="-",
                        help="Ruta CSV de salida (default: stdout)")
    args = parser.parse_args()

    paths = sorted(glob.glob("data/fone/*/PlazasDocAdmtvasDirec_*.parquet"))
    if not paths:
        sys.exit("Error: no se encontraron archivos parquet en data/fone/")

    lf = pl.scan_parquet(paths).filter(pl.col("YEAR").cast(pl.Int32) == args.year)
    if args.estado:
        lf = lf.filter(pl.col("ENTIDAD_FEDERATIVA") == args.estado.upper().strip())
    if args.tipo:
        lf = lf.filter(pl.col("TIPO_PLAZA") == args.tipo)

    label = f"year={args.year}"
    if args.estado:
        label += f" estado={args.estado.upper().strip()}"
    if args.tipo:
        label += f" tipo={args.tipo}"

    # --total is implicit when no percentile is given
    total_mode = args.total or args.percentil is None

    # Build per-CURP annual totals
    curp_totals = (
        lf.group_by(["CURP"])
        .agg(
            pl.col("NOMBRE").first(),
            pl.col("RFC").first(),
            pl.col("ENTIDAD_FEDERATIVA").first(),
            pl.col("TIPO_PLAZA").first(),
            pl.col("PERCEPCIONES_TRIMESTRALES").sum().alias("PERCEPCIONES_ANUALES"),
        )
        .filter(pl.col("PERCEPCIONES_ANUALES") > 0)
        .collect()
    )

    if curp_totals.is_empty():
        sys.exit(f"Sin datos para {label}")

    # Apply percentile filter if requested
    if args.percentil is not None:
        threshold = float(curp_totals["PERCEPCIONES_ANUALES"].quantile(_PCTILE_MAP[args.percentil]))
        curp_totals = curp_totals.filter(pl.col("PERCEPCIONES_ANUALES") >= threshold)
        pctile_label = f"P{args.percentil} = {threshold:,.2f} MXN | "
    else:
        threshold = None
        pctile_label = ""

    if total_mode:
        result = (
            curp_totals
            .group_by("TIPO_PLAZA")
            .agg(
                pl.col("CURP").n_unique().alias("N_CURPS"),
                pl.col("PERCEPCIONES_ANUALES").sum().alias("PERCEPCIONES_ANUALES_TOTAL"),
            )
            .sort("PERCEPCIONES_ANUALES_TOTAL", descending=True)
        )
        n_curps = int(result["N_CURPS"].sum())
        total_perc = float(result["PERCEPCIONES_ANUALES_TOTAL"].sum())
        print(f"# {pctile_label}{n_curps:,} CURPs | {total_perc:,.2f} MXN total | {label}", file=sys.stderr)
    else:
        result = (
            curp_totals
            .select(["CURP", "NOMBRE", "RFC", "ENTIDAD_FEDERATIVA", "TIPO_PLAZA", "PERCEPCIONES_ANUALES"])
            .sort("PERCEPCIONES_ANUALES", descending=True)
        )
        print(f"# {pctile_label}{len(result)} CURPs | {label}", file=sys.stderr)

    if args.out == "-":
        sys.stdout.write(result.write_csv())
    else:
        result.write_csv(args.out)
        print(f"Escrito: {args.out} ({len(result)} filas)", file=sys.stderr)


if __name__ == "__main__":
    main()
