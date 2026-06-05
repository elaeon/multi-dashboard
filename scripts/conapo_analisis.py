"""
Análisis demográfico de estados y municipios (CONAPO).

Uso:
    python scripts/conapo_analisis.py crecimiento --nivel estado --desde 1990 --hasta 2020
    python scripts/conapo_analisis.py demografia --nivel estado --año 2020 --sexo AMBOS --grupo POB_65_69
    python scripts/conapo_analisis.py demografia --nivel estado --año 2020 --edad 20-40
    python scripts/conapo_analisis.py demografia --nivel municipio --año 2020 --edad 65-
    python scripts/conapo_analisis.py municipios --año 2020 [--estado Aguascalientes]
"""

import argparse
import sys
from pathlib import Path

import polars as pl

DATA_FILE = Path(__file__).parent.parent / "data" / "conapo" / "estados_municipios.xlsx"
CSV_FILE = DATA_FILE.with_suffix(".csv")

AGE_GROUPS = [
    "POB_00_04", "POB_05_09", "POB_10_14", "POB_15_19",
    "POB_20_24", "POB_25_29", "POB_30_34", "POB_35_39",
    "POB_40_44", "POB_45_49", "POB_50_54", "POB_55_59",
    "POB_60_64", "POB_65_69", "POB_70_74", "POB_75_79",
    "POB_80_84", "POB_85_mm",
]

ALL_POB_COLS = AGE_GROUPS + ["POB_TOTAL"]


def _group_bounds(g: str) -> tuple[int, int]:
    lo, hi = g.removeprefix("POB_").split("_")
    return int(lo), (999 if hi == "mm" else int(hi))


GROUP_BOUNDS = {g: _group_bounds(g) for g in AGE_GROUPS}


def parse_edad_range(s: str) -> tuple[int | None, int | None]:
    """Parse '20-40', '20-', or '-40' into (lo, hi). None means unbounded."""
    if "-" not in s:
        v = int(s)
        return v, v
    left, right = s.split("-", 1)
    return (int(left) if left else None), (int(right) if right else None)


def groups_in_range(lo: int | None, hi: int | None) -> list[str]:
    return [
        g for g, (g_lo, g_hi) in GROUP_BOUNDS.items()
        if (lo is None or g_hi >= lo) and (hi is None or g_lo <= hi)
    ]


def edad_label(lo: int | None, hi: int | None) -> str:
    lo_s = str(lo) if lo is not None else "0"
    hi_s = str(hi) if hi is not None else "mas"
    return f"POB_{lo_s}_{hi_s}"


def show_result(result: pl.DataFrame, title: str, top_n: int) -> None:
    print(f"\n--- {title} ---")
    with pl.Config(tbl_rows=top_n):
        print(result)


def gcols(nivel: str) -> list[str]:
    return ["NOM_ENT"] if nivel == "estado" else ["NOM_MUN", "NOM_ENT"]


def crecimiento(df: pl.LazyFrame, args: argparse.Namespace) -> None:
    cols = gcols(args.nivel)

    if args.edad:
        lo, hi = parse_edad_range(args.edad)
        selected = groups_in_range(lo, hi)
        if not selected:
            print(f"Error: ningún grupo de edad en el rango '{args.edad}'.", file=sys.stderr)
            sys.exit(1)
        agg_col = edad_label(lo, hi)
    else:
        selected = None
        agg_col = "POB_TOTAL"

    def pob_año(año: int, alias: str) -> pl.LazyFrame:
        base = df.filter(pl.col("AÑO") == año)
        if args.sexo != "AMBOS":
            base = base.filter(pl.col("SEXO") == args.sexo)
        if selected:
            base = base.with_columns(pl.sum_horizontal([pl.col(g) for g in selected]).alias(agg_col))
        return base.group_by(cols).agg(pl.col(agg_col).sum().alias(alias))

    result = (
        pob_año(args.desde, "pob_inicio").join(pob_año(args.hasta, "pob_fin"), on=cols)
        .with_columns(
            (pl.col("pob_fin") - pl.col("pob_inicio")).alias("delta"),
            ((pl.col("pob_fin") - pl.col("pob_inicio")) / pl.col("pob_inicio") * 100)
            .round(2).alias("pct_cambio"),
        )
        .sort("delta", descending=(args.orden == "mayor"))
        .head(args.top)
        .collect()
    )

    extras = " | ".join(filter(None, [
        f"edad {args.edad}" if args.edad else None,
        args.sexo.lower() if args.sexo != "AMBOS" else None,
    ]))
    title = f"Crecimiento {args.desde}→{args.hasta} por {args.nivel}" + (f" ({extras})" if extras else "")
    show_result(result, f"{title} (top {args.top}, {args.orden})", args.top)


def demografia(df: pl.LazyFrame, args: argparse.Namespace) -> None:
    cols = gcols(args.nivel)
    filtered = df.filter(pl.col("AÑO") == args.año)
    if args.sexo != "AMBOS":
        filtered = filtered.filter(pl.col("SEXO") == args.sexo)

    if args.grupo:
        age_col = args.grupo
        query = filtered.group_by(cols).agg(pl.col(age_col).sum())
    else:
        lo, hi = parse_edad_range(args.edad)
        selected = groups_in_range(lo, hi)
        if not selected:
            print(f"Error: ningún grupo de edad en el rango '{args.edad}'.", file=sys.stderr)
            sys.exit(1)
        age_col = edad_label(lo, hi)
        query = (
            filtered
            .with_columns(pl.sum_horizontal([pl.col(g) for g in selected]).alias(age_col))
            .group_by(cols)
            .agg(pl.col(age_col).sum())
        )

    result = (
        query
        .sort(age_col, descending=(args.orden == "mayor"))
        .head(args.top)
        .collect()
    )

    sexo_label = args.sexo.lower() if args.sexo != "AMBOS" else "ambos sexos"
    show_result(result, f"{age_col} en {args.año}, {sexo_label}, por {args.nivel} (top {args.top}, {args.orden})", args.top)


def municipios(df: pl.LazyFrame, args: argparse.Namespace) -> None:
    filtered = df.filter(pl.col("AÑO") == args.año)
    if args.estado:
        filtered = filtered.filter(pl.col("NOM_ENT") == args.estado)

    result = (
        filtered.group_by(["NOM_MUN", "NOM_ENT", "CLAVE"])
        .agg([pl.col(c).sum() for c in ALL_POB_COLS])
        .sort(["NOM_ENT", "NOM_MUN"])
        .collect()
    )

    label = args.estado or "todos los estados"
    print(f"\n--- Población por municipio en {args.año} ({label}) ---")

    slug = args.estado.lower().replace(" ", "_") if args.estado else "todos"
    csv_path = DATA_FILE.parent / f"municipios_{args.año}_{slug}.csv"
    result.write_csv(csv_path)
    print(f"Guardado en: {csv_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Análisis demográfico CONAPO")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("crecimiento", help="Crecimiento/decrecimiento poblacional")
    p1.add_argument("--nivel", choices=["estado", "municipio"], required=True)
    p1.add_argument("--desde", type=int, required=True, metavar="AÑO")
    p1.add_argument("--hasta", type=int, required=True, metavar="AÑO")
    p1.add_argument("--sexo", choices=["HOMBRES", "MUJERES", "AMBOS"], default="AMBOS")
    p1.add_argument("--edad", metavar="RANGO", default=None, help="Rango de edad, ej: 20-40, 65-, -18")
    p1.add_argument("--orden", choices=["mayor", "menor"], default="mayor")
    p1.add_argument("--top", type=int, default=10)

    p2 = sub.add_parser("demografia", help="Población por grupo de edad y sexo")
    p2.add_argument("--nivel", choices=["estado", "municipio"], required=True)
    p2.add_argument("--año", type=int, required=True)
    p2.add_argument("--sexo", choices=["HOMBRES", "MUJERES", "AMBOS"], default="AMBOS")
    edad_grp = p2.add_mutually_exclusive_group(required=True)
    edad_grp.add_argument("--grupo", choices=AGE_GROUPS, metavar="GRUPO")
    edad_grp.add_argument("--edad", metavar="RANGO", help="Rango de edad, ej: 20-40, 65-, -18")
    p2.add_argument("--orden", choices=["mayor", "menor"], default="mayor")
    p2.add_argument("--top", type=int, default=10)

    p3 = sub.add_parser("municipios", help="Población de todos los municipios de un año")
    p3.add_argument("--año", type=int, required=True)
    p3.add_argument("--estado", default=None, metavar="NOMBRE")

    return parser


def main() -> None:
    args = build_parser().parse_args()

    if not DATA_FILE.exists():
        print(f"Error: no se encontró {DATA_FILE}", file=sys.stderr)
        sys.exit(1)

    if not CSV_FILE.exists():
        print(f"Convirtiendo {DATA_FILE.name} a CSV...", file=sys.stderr)
        pl.read_excel(DATA_FILE, engine="calamine").write_csv(CSV_FILE)

    df = pl.scan_csv(CSV_FILE)

    {"crecimiento": crecimiento, "demografia": demografia, "municipios": municipios}[args.cmd](df, args)


if __name__ == "__main__":
    main()
