"""ETL: ENOE Indicadores Laborales — agrega 2017-2025 (Q1) en un solo parquet.

Salida: data/inegi/indicadores_laborales/indicadores_laborales.parquet
est==1 solamente (valor puntual); cve_mun==0 son agregados estatales.
"""
import io, zipfile
from pathlib import Path
import polars as pl

DATA_DIR = Path("data/inegi/indicadores_laborales")
YEARS = range(2017, 2026)
BOM = b"\xef\xbb\xbf"


def _read(z, name, encoding="latin1"):
    raw = z.read(name)
    if raw.startswith(BOM):
        raw = raw[3:]
    return pl.read_csv(io.BytesIO(raw), infer_schema=False, encoding=encoding)


def read_year(year):
    zip_path = next(DATA_DIR.glob(f"*{year}*"))
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
        data_name = next(n for n in names if "conjunto_de_datos/" in n and n.endswith(".csv"))

        # Detect 2025 format by first column name
        raw0 = z.read(data_name)
        if raw0.startswith(BOM):
            raw0 = raw0[3:]
        first_col = pl.read_csv(io.BytesIO(raw0), infer_schema=False, n_rows=0).columns[0]
        is_new = first_col.lower() == "cvegeo"

        df = _read(z, data_name)  # latin1 works for both data files
        df = df.rename({c: c.strip().lower() for c in df.columns})

        if is_new:
            df = df.select(["cve_ent", "cve_mun", "est", "pea", "ocupados", "informales"])
            # cve_ent.csv is UTF-8; cve_mun.csv is mixed (latin1 for nom_ent, utf8 for nom_mun)
            ent_cat = _read(z, next(n for n in names if "catalogos/cve_ent.csv" in n), "utf8")
            mun_cat = _read(z, next(n for n in names if "catalogos/cve_mun.csv" in n), "latin1")
            ent_lookup = ent_cat.select(
                pl.col("cve_ent").str.strip_chars().cast(pl.Int32),
                pl.col("nom_ent").str.strip_chars())
            mun_lookup = mun_cat.select(
                pl.col("cve_ent").str.strip_chars().cast(pl.Int32),
                pl.col("cve_mun").str.strip_chars().cast(pl.Int32),
                pl.col("nom_mun").str.strip_chars())
        else:
            df = df.rename({"ent": "cve_ent", "mun": "cve_mun"})
            df = df.select(["cve_ent", "cve_mun", "est", "pea", "ocupados", "informales"])
            ent_cat = _read(z, next(n for n in names if "catalogos/ent.csv" in n))
            mun_cat = _read(z, next(n for n in names if "catalogos/mun.csv" in n))
            ent_lookup = ent_cat.select(
                pl.col("cve").str.strip_chars().cast(pl.Int32).alias("cve_ent"),
                pl.col("descrip").str.strip_chars().alias("nom_ent"))
            mun_lookup = mun_cat.select(
                pl.col("cve_ent").str.strip_chars().cast(pl.Int32),
                pl.col("cve_mun").str.strip_chars().cast(pl.Int32),
                pl.col("descrip").str.strip_chars().alias("nom_mun"))

    df = (df
        .with_columns(
            pl.col("cve_ent").str.strip_chars().cast(pl.Int32),
            pl.col("cve_mun").str.strip_chars().cast(pl.Int32),
            pl.col("est").str.strip_chars().cast(pl.Int32),
            pl.col("pea").str.strip_chars().cast(pl.Float32),
            pl.col("ocupados").str.strip_chars().cast(pl.Float32),
            pl.col("informales").str.strip_chars().cast(pl.Float32))
        .filter(pl.col("est") == 1).drop("est")
        .join(ent_lookup, on="cve_ent", how="left")
        .join(mun_lookup, on=["cve_ent", "cve_mun"], how="left")
        # For state-level rows (cve_mun==0), nom_mun from catalog may be wrong encoding;
        # use nom_ent (correctly read) as the canonical name.
        .with_columns(
            pl.when(pl.col("cve_mun") == 0)
            .then(pl.col("nom_ent"))
            .otherwise(pl.col("nom_mun"))
            .alias("nom_mun"))
        .with_columns(pl.lit(year).cast(pl.Int32).alias("año")))
    return df.select(["año", "cve_ent", "nom_ent", "cve_mun", "nom_mun", "pea", "ocupados", "informales"])


def main():
    frames = [read_year(y) for y in YEARS]
    result = pl.concat(frames)
    out = DATA_DIR / "indicadores_laborales.parquet"
    result.write_parquet(out)
    print(f"Saved {len(result):,} rows → {out}")
    sample = result.filter((pl.col("cve_mun") == 0) & (pl.col("cve_ent").is_between(9, 22)))
    print(sample.head(8))


if __name__ == "__main__":
    main()
