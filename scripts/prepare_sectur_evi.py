"""
Cleans the SECTUR Encuesta de Viajeros Internacionales (Cuenta de Viajeros)
for the "Viajeros" tabs of dashboard/sectur_hoteleria.py.

Tiny table (3,822 rows) — no pre-aggregation needed, unlike the hotelería
weekly panel or the extranjeros_residencia census. Ship one cleaned long/tidy
parquet; figure factories filter/group in Polars at load time.

Source: data/sectur/encuesta_viajeros_internacionales/BD_CuentaViajeros_descarga.zip
        -> BD_CuentaViajeros_descarga.xlsx, sheet Hoja1
Output: dashboard_data/sectur_evi.parquet
Run: uv run python scripts/prepare_sectur_evi.py
"""
import zipfile
from pathlib import Path

import polars as pl

ZIP_PATH = Path("data/sectur/encuesta_viajeros_internacionales/BD_CuentaViajeros_descarga.zip")
INNER    = "BD_CuentaViajeros_descarga.xlsx"
OUT_PATH = Path("dashboard_data/sectur_evi.parquet")

MONTH_MAP = {
    "Ene": 1, "Feb": 2, "Mar": 3, "Abr": 4, "May": 5, "Jun": 6,
    "Jul": 7, "Ago": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dic": 12,
}

print(f"Reading {INNER} from {ZIP_PATH}...")
with zipfile.ZipFile(ZIP_PATH) as z:
    with z.open(INNER) as f:
        raw = pl.read_excel(f.read(), sheet_id=1)

print(f"Raw rows: {raw.height:,}")

d = (
    raw.drop(["Fuente", "DescripcionNivel01", "DescripcionNivel02"])
    .with_columns(
        pl.col("kano").cast(pl.Int32),
        pl.col("MesText").replace(MONTH_MAP).cast(pl.Int8).alias("mes"),
    )
)

print(f"Rows: {d.height:,} (expected 3,822)")
assert d.height == 3822

# ── Validation against DATA_OVERVIEW.md Statistical Findings ────────────────

def headcount(kano: int) -> float:
    return d.filter(
        (pl.col("Tipo") == "Ingresos") & (pl.col("DescripcionNivel03") == "Número de Viajeros") & (pl.col("kano") == kano)
    )["Valor"].sum()


def ingresos(kano: int) -> float:
    return d.filter(
        (pl.col("Tipo") == "Ingresos") & (pl.col("DescripcionNivel03") == "Ingresos") & (pl.col("kano") == kano)
    )["Valor"].sum()


def egresos(kano: int) -> float:
    return d.filter(
        (pl.col("Tipo") == "Egresos") & (pl.col("DescripcionNivel03") == "Egresos") & (pl.col("kano") == kano)
    )["Valor"].sum()


h2019, h2020, h2025 = headcount(2019), headcount(2020), headcount(2025)
rev2025 = ingresos(2025)
egr2025 = egresos(2025)

print(f"2019 headcount: {h2019/1e6:.2f}M (expected ~97.41M)")
print(f"2020 headcount: {h2020/1e6:.2f}M (expected ~51.13M, {(h2020/h2019-1)*100:.1f}% vs 2019, expected ~-47.5%)")
print(f"2025 headcount: {h2025/1e6:.2f}M (expected ~98.20M)")
print(f"2025 Ingresos: {rev2025/1e9:.2f}B (expected ~$34.99B)")
print(f"2025 Egresos: {egr2025/1e9:.2f}B (expected ~$13.65B)")

assert 96.4e6 <= h2019 <= 98.4e6
assert 50.1e6 <= h2020 <= 52.1e6
assert 97.2e6 <= h2025 <= 99.2e6
assert 34.5e9 <= rev2025 <= 35.5e9
assert 13.4e9 <= egr2025 <= 13.9e9

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
d.write_parquet(OUT_PATH)
print(f"Saved → {OUT_PATH}")
