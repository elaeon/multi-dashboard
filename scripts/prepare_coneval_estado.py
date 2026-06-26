"""
Extrae tasas de pobreza y carencias sociales por estado del CONEVAL (4 ondas: 2016–2022).
Outputs:
  data/coneval/deprivacion_estado_2022.parquet  (32 filas — corte transversal 2022)
  data/coneval/deprivacion_panel.parquet        (128 filas — 32 estados × 4 años)
Run: uv run python scripts/prepare_coneval_estado.py
"""
import io
import zipfile
from pathlib import Path

import polars as pl

CONEVAL_DIR = Path("data/coneval")
OUT_2022    = CONEVAL_DIR / "deprivacion_estado_2022.parquet"
OUT_PANEL   = CONEVAL_DIR / "deprivacion_panel.parquet"

WAVES = [
    (2016, "Python_MMP_2016.zip", "Base final/pobreza16.csv"),
    (2018, "Python_MMP_2018.zip", "Base final/pobreza18.csv"),
    (2020, "Python_MMP_2020.zip", "Base final/pobreza20.csv"),
    (2022, "Python_MMP_2022.zip", "Base final/pobreza22.csv"),
]

ENT_TO_ESTADO = {
    1:  "Aguascalientes",
    2:  "Baja California",
    3:  "Baja California Sur",
    4:  "Campeche",
    5:  "Coahuila",
    6:  "Colima",
    7:  "Chiapas",
    8:  "Chihuahua",
    9:  "Ciudad De Mexico",
    10: "Durango",
    11: "Guanajuato",
    12: "Guerrero",
    13: "Hidalgo",
    14: "Jalisco",
    15: "Mexico",
    16: "Michoacan",
    17: "Morelos",
    18: "Nayarit",
    19: "Nuevo Leon",
    20: "Oaxaca",
    21: "Puebla",
    22: "Queretaro",
    23: "Quintana Roo",
    24: "San Luis Potosi",
    25: "Sinaloa",
    26: "Sonora",
    27: "Tabasco",
    28: "Tamaulipas",
    29: "Tlaxcala",
    30: "Veracruz",
    31: "Yucatan",
    32: "Zacatecas",
}

IC_COLS = ["ic_rezedu", "ic_asalud", "ic_segsoc", "ic_cv", "ic_sbv", "ic_ali"]
ENT_STR = {str(k): v for k, v in ENT_TO_ESTADO.items()}


def aggregate_wave(year: int, zip_name: str, inner_csv: str) -> pl.DataFrame:
    zip_path = CONEVAL_DIR / zip_name
    print(f"  Reading {inner_csv} from {zip_name}...", end=" ", flush=True)
    with zipfile.ZipFile(zip_path) as z:
        with z.open(inner_csv) as f:
            buf = io.BytesIO(f.read())

    raw = pl.read_csv(
        buf,
        columns=["ent", "factor", "carencias3", "pobreza", "i_privacion"] + IC_COLS,
    )
    print(f"{raw.height:,} rows")

    weighted_ic = [(pl.col(c) * pl.col("factor")).alias(f"_{c}_w") for c in IC_COLS]
    sum_ic      = [pl.col(f"_{c}_w").sum() for c in IC_COLS]
    pct_ic      = [(pl.col(f"_{c}_w") / pl.col("_pop") * 100).round(2).alias(f"pct_{c}") for c in IC_COLS]

    return (
        raw
        .with_columns([
            (pl.col("carencias3")  * pl.col("factor")).alias("_c3_w"),
            (pl.col("pobreza")     * pl.col("factor")).alias("_pob_w"),
            (pl.col("i_privacion") * pl.col("factor")).alias("_priv_w"),
            *weighted_ic,
        ])
        .group_by("ent")
        .agg([
            pl.col("factor").sum().alias("_pop"),
            pl.col("_c3_w").sum(),
            pl.col("_pob_w").sum(),
            pl.col("_priv_w").sum(),
            *sum_ic,
        ])
        .with_columns([
            (pl.col("_c3_w")  / pl.col("_pop") * 100).round(2).alias("pct_carencias3"),
            (pl.col("_pob_w") / pl.col("_pop") * 100).round(2).alias("pct_pobreza"),
            (pl.col("_priv_w")/ pl.col("_pop")).round(3).alias("promedio_carencias"),
            *pct_ic,
        ])
        .with_columns([
            pl.col("ent").cast(pl.String).replace(ENT_STR).alias("estado"),
            pl.lit(year).cast(pl.Int32).alias("año"),
        ])
        .select(["estado", "año", "pct_carencias3", "pct_pobreza", "promedio_carencias"]
                + [f"pct_{c}" for c in IC_COLS])
        .sort("estado")
    )


print("Processing CONEVAL waves:")
frames = [aggregate_wave(year, zip_name, inner) for year, zip_name, inner in WAVES]

panel = pl.concat(frames).sort(["estado", "año"])
print(f"\nPanel: {panel.height} rows ({panel.height // 4} states × 4 years)")

# 2022 cross-section (backward-compatible output)
wave_2022 = panel.filter(pl.col("año") == 2022).drop("año")
wave_2022.write_parquet(OUT_2022)
print(f"Saved 2022 cross-section → {OUT_2022}")

panel.write_parquet(OUT_PANEL)
print(f"Saved panel → {OUT_PANEL}")

print("\nTop 5 by carencias3 in 2022:")
print(wave_2022.sort("pct_carencias3", descending=True)
      .select(["estado", "pct_carencias3", "pct_pobreza"]).head(5))
