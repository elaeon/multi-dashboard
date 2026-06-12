import argparse
import io
import zipfile
from pathlib import Path

import polars as pl

DATA_DIR = Path("data/inegi")
NAC_DIR = DATA_DIR / "natalidad"
DES_DIR = DATA_DIR / "defunciones"
OUT_DIR = DATA_DIR / "nacimientos_descesos"
GEO_COLS = ["ent_resid", "mun_resid", "tloc_resid"]
OD_COLS = ["ent_resid", "mun_resid", "tloc_resid", "ent_ocurr", "mun_ocurr", "tloc_ocurr"]


def find_zip(directory: Path, year: int) -> Path:
    matches = list(directory.glob(f"*{year}*"))
    if not matches:
        raise FileNotFoundError(f"No zip found for year {year} in {directory}")
    if len(matches) > 1:
        raise ValueError(f"Multiple zips found for year {year} in {directory}: {[m.name for m in matches]}")
    return matches[0]


def read_grouped(zip_path: Path, cols: list[str], count_col: str) -> pl.DataFrame:
    with zipfile.ZipFile(zip_path) as z:
        csv_name = next(
            n for n in z.namelist()
            if n.startswith("conjunto_de_datos/") and n.lower().endswith(".csv")
        )
        with z.open(csv_name) as f:
            first_line = f.readline()
            sep = ";" if b";" in first_line else ","
            data = io.BytesIO(first_line + f.read())

    df = pl.read_csv(data, separator=sep, infer_schema=False)
    df = df.rename({c: c.lower() for c in df.columns})
    return (
        df.select(cols)
        .with_columns(pl.col(cols).str.strip_chars())
        .group_by(cols)
        .len(count_col)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Estadísticas de nacimientos y defunciones por municipio")
    parser.add_argument("year", type=int, help="Año a procesar (ej. 2024)")
    args = parser.parse_args()

    nac_zip = find_zip(NAC_DIR, args.year)
    des_zip = find_zip(DES_DIR, args.year)

    # Residence-only output (original format)
    df_nac = read_grouped(nac_zip, GEO_COLS, "total_nac")
    df_des = read_grouped(des_zip, GEO_COLS, "total_des")
    result = (
        df_nac.join(df_des, on=GEO_COLS)
        .with_columns(pl.lit(args.year).alias("anio"))
        .sort(GEO_COLS)
    )
    OUT_DIR.mkdir(exist_ok=True)
    result.write_csv(OUT_DIR / f"{args.year}.csv")
    print(f"Saved {len(result)} rows to {OUT_DIR}/{args.year}.csv")

    # Origin-destination output (residence vs occurrence)
    df_nac_od = read_grouped(nac_zip, OD_COLS, "total_nac")
    df_des_od = read_grouped(des_zip, OD_COLS, "total_des")
    od = (
        df_nac_od.join(df_des_od, on=OD_COLS, how="full", coalesce=True)
        .with_columns(pl.lit(args.year).alias("anio"))
        .sort(OD_COLS)
    )
    od.write_csv(OUT_DIR / f"od_{args.year}.csv")
    print(f"Saved {len(od)} OD rows to {OUT_DIR}/od_{args.year}.csv")


if __name__ == "__main__":
    main()
