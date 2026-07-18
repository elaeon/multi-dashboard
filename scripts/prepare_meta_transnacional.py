"""
Prepara las tablas derivadas para el dashboard del giro transnacional de
metanfetaminas (dashboard/meta_transnacional.py).

Combina el parquet limpio de desmantelamientos de laboratorios clandestinos
de la DEA (data/dea/desmantelamiento_laboratorios/) con la base de datos
antidrogas de MUCD (data/mucd/antidrogas/), y guarda 4 tablas pequeñas en
dashboard_data/.
"""

import io
import tarfile

import polars as pl

DEA_PARQUET = "data/dea/desmantelamiento_laboratorios/desmantelamiento_laboratorios_clean.parquet"
MUCD_TARGZ = "data/mucd/antidrogas/Base_de_datos_MUCD.tar.gz"
MUCD_CSV_INNER = "Base_de_datos_MUCD/antidrog_1990-2022_062023.csv"
OUT_DIR = "dashboard_data"

STATE_MAP = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR", "California": "CA",
    "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE", "Florida": "FL", "Georgia": "GA",
    "Hawaii": "HI", "Idaho": "ID", "Illinois": "IL", "Indiana": "IN", "Iowa": "IA", "Kansas": "KS",
    "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD", "Massachusetts": "MA",
    "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS", "Missouri": "MO", "Montana": "MT",
    "Nebraska": "NE", "Nevada": "NV", "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM",
    "New York": "NY", "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK",
    "Oregon": "OR", "Pennsylvania": "PA", "Rhode Island": "RI", "South Carolina": "SC",
    "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX", "Utah": "UT", "Vermont": "VT",
    "Virginia": "VA", "Washington": "WA", "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY",
    "District of Columbia": "DC", "Puerto Rico": "PR", "Guam": "GU",
}

RELIABLE_YEARS = [y for y in list(range(2004, 2019)) + list(range(2020, 2024))]
GAP_YEARS = {2019}
PARTIAL_YEARS = {2024, 2025}
TWO_STATE_YEARS = {2000, 2001, 2002, 2003}


def _window_flag(year: int) -> str:
    if year in TWO_STATE_YEARS:
        return "two_state_2000_03"
    if year in GAP_YEARS:
        return "gap_2019"
    if year in PARTIAL_YEARS:
        return "partial_recent"
    return "reliable"


def prepare_dea() -> None:
    df = pl.read_parquet(DEA_PARQUET)
    df = df.filter(~pl.col("state_unresolved"))
    df = df.with_columns(
        pl.col("state").replace(STATE_MAP).alias("state")
    )

    annual = (
        df.group_by("year").agg(pl.len().alias("n_labs")).sort("year")
        .with_columns(
            pl.col("year").map_elements(_window_flag, return_dtype=pl.Utf8).alias("window_flag")
        )
    )
    annual.write_parquet(f"{OUT_DIR}/meta_us_annual.parquet")

    state_year = (
        df.filter(pl.col("year").is_in(RELIABLE_YEARS))
        .group_by(["state", "year"]).agg(pl.len().alias("n_labs"))
        .sort(["year", "state"])
    )
    state_year.write_parquet(f"{OUT_DIR}/meta_us_state_year.parquet")

    print("meta_us_annual:", annual.shape, annual["year"].min(), "-", annual["year"].max())
    print("meta_us_state_year:", state_year.shape)


def prepare_mucd() -> None:
    with tarfile.open(MUCD_TARGZ, "r:gz") as tf:
        f = tf.extractfile(MUCD_CSV_INNER)
        raw = pl.read_csv(io.BytesIO(f.read()), encoding="latin-1", infer_schema_length=100000)

    d = raw.filter(pl.col("año") <= 2021)

    annual = (
        d.group_by("año").agg(pl.col("AseMet_SEDENA").sum().alias("kg_met_sedena"))
        .sort("año")
    )
    annual.write_parquet(f"{OUT_DIR}/meta_mx_annual.parquet")

    state_year = (
        d.filter((pl.col("año") >= 2000) & (pl.col("entidad") != "No Especificada"))
        .group_by(["entidad", "año"]).agg(pl.col("AseMet_SEDENA").sum().alias("kg_met_sedena"))
        .sort(["año", "entidad"])
    )
    state_year.write_parquet(f"{OUT_DIR}/meta_mx_state_year.parquet")

    print("meta_mx_annual:", annual.shape, annual["año"].min(), "-", annual["año"].max())
    print("meta_mx_state_year:", state_year.shape)


if __name__ == "__main__":
    prepare_dea()
    prepare_mucd()
