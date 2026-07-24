"""
Builds a tidy destination-week occupancy panel for dashboard/sectur_hoteleria.py
from the 117 weekly SECTUR DataTur hotelería reports.

Each weekly file reports a rolling 3-year comparison (t-2/t-1/t0), but t0 is
always the file's own week/year, so only the t0 columns are needed per file —
no need to parse the merged 3-row header (see DATA_OVERVIEW.md G1/G3/G7).
Reading with skiprows=9 and no header, the blank column A is dropped by
polars/fastexcel automatically, shifting positions: column_1=centro,
column_4=cuartos_disp (t0), column_11=cuartos_ocup (t0), column_18=ocupacion (t0).

Source: data/sectur/hoteleria/{YEAR}_Semana_{WW}.xlsx (117 files)
Output: dashboard_data/sectur_hoteleria_panel.parquet
Run: uv run python scripts/prepare_sectur_hoteleria.py
"""
import re
from datetime import date
from pathlib import Path

import polars as pl

IN_DIR   = Path("data/sectur/hoteleria")
OUT_PATH = Path("dashboard_data/sectur_hoteleria_panel.parquet")

ROLLUP_LABELS = {
    "Total", "Centros de Playa", "Integralmente Planeados", "Tradicionales",
    "Otros", "Ciudades", "Grandes", "Del Interior", "Fronterizas",
}
LOS_CABOS_SUBROWS = {"CABO SAN LUCAS", "SAN JOSÉ DEL CABO", "ZONA CORREDOR LOS CABOS"}

CATEGORIA_MAP = {
    # Integralmente Planeados + Tradicionales + Otros -> Playa
    "BAHIAS DE HUATULCO": "Playa", "CANCUN": "Playa", "IXTAPA ZIHUATANEJO": "Playa",
    "LORETO": "Playa", "LOS CABOS": "Playa",
    "ACAPULCO": "Playa", "COZUMEL": "Playa", "LA PAZ": "Playa", "MANZANILLO": "Playa",
    "MAZATLAN": "Playa", "PUERTO VALLARTA": "Playa", "VERACRUZ BOCA DEL RIO": "Playa",
    "ISLA MUJERES": "Playa", "NUEVO NAYARIT": "Playa", "RIVIERA MAYA": "Playa",
    "AKUMAL": "Playa", "PLAYA DEL CARMEN": "Playa", "PLAYACAR": "Playa",
    "PUERTO ESCONDIDO": "Playa", "PLAYAS DE ROSARITO": "Playa", "SAN FELIPE": "Playa",
    "TONALÁ- PUERTO ARISTA": "Playa",
    # Grandes + Del Interior -> Ciudad
    "CIUDAD DE MÉXICO": "Ciudad", "GUADALAJARA": "Ciudad", "MONTERREY": "Ciudad",
    "AGUASCALIENTES": "Ciudad", "CAMPECHE": "Ciudad", "CELAYA": "Ciudad",
    "CHIHUAHUA": "Ciudad", "COATZACOALCOS": "Ciudad", "COLIMA": "Ciudad",
    "COMITÁN DE DOMÍNGUEZ": "Ciudad", "CULIACAN": "Ciudad", "DURANGO": "Ciudad",
    "EL FUERTE": "Ciudad", "GUANAJUATO": "Ciudad", "HERMOSILLO": "Ciudad",
    "IRAPUATO": "Ciudad", "LEON": "Ciudad", "LOS MOCHIS": "Ciudad", "MERIDA": "Ciudad",
    "MORELIA": "Ciudad", "OAXACA": "Ciudad", "PACHUCA": "Ciudad", "PALENQUE": "Ciudad",
    "PUEBLA": "Ciudad", "QUERETARO": "Ciudad", "SALAMANCA": "Ciudad",
    "SAN CRISTÓBAL DE LAS CASAS": "Ciudad", "SAN JUAN DE LOS LAGOS": "Ciudad",
    "SAN JUAN DEL RÍO": "Ciudad", "SAN LUIS POTOSI": "Ciudad",
    "SAN MIGUEL DE ALLENDE": "Ciudad", "TAXCO": "Ciudad", "TEQUISQUIAPAN": "Ciudad",
    "TLAXCALA": "Ciudad", "TOLUCA": "Ciudad", "TUXTLA GUTIÉRREZ": "Ciudad",
    "VALLE DE BRAVO": "Ciudad", "VILLAHERMOSA": "Ciudad", "XALAPA": "Ciudad",
    "ZACATECAS": "Ciudad",
    # Fronterizas -> Frontera
    "CIUDAD JUÁREZ": "Frontera", "MEXICALI": "Frontera", "PIEDRAS NEGRAS": "Frontera",
    "TECATE": "Frontera", "TIJUANA": "Frontera",
}

_SUFFIX_RE = re.compile(r"\s*/\s*\d+|\s*\d+\s*/\s*$")


def normalize_centro(raw: str) -> str:
    cleaned = _SUFFIX_RE.sub("", raw).strip()
    base = cleaned.split(",")[0].strip().rstrip(".").strip().upper()
    return base


def parse_file(path: Path) -> pl.DataFrame:
    año, semana = _parse_filename(path)
    fecha = date.fromisocalendar(año, semana, 1)

    raw = pl.read_excel(path, sheet_id=1, has_header=False, read_options={"skip_rows": 9})
    d = raw.select(
        pl.col("column_1").alias("centro_raw"),
        pl.col("column_4").cast(pl.Float64, strict=False).alias("cuartos_disp"),
        pl.col("column_11").cast(pl.Float64, strict=False).alias("cuartos_ocup"),
        pl.col("column_18").cast(pl.Float64, strict=False).alias("ocupacion"),
    ).drop_nulls(["centro_raw", "ocupacion"])

    d = d.filter(
        ~pl.col("centro_raw").str.strip_chars().is_in(ROLLUP_LABELS)
        & ~pl.col("centro_raw").str.strip_chars().is_in(LOS_CABOS_SUBROWS)
    )

    d = d.with_columns(
        pl.col("centro_raw").map_elements(normalize_centro, return_dtype=pl.String).alias("centro"),
        pl.lit(año).alias("año"),
        pl.lit(semana).alias("semana"),
        pl.lit(fecha).alias("fecha"),
    ).drop("centro_raw")

    d = d.with_columns(pl.col("centro").replace(CATEGORIA_MAP).alias("categoria"))
    return d


def _parse_filename(path: Path) -> tuple[int, int]:
    año_str, semana_str = path.stem.split("_Semana_")
    return int(año_str), int(semana_str)


files = sorted(IN_DIR.glob("*_Semana_*.xlsx"))
print(f"Parsing {len(files)} files...")

frames = [parse_file(f) for f in files]
d = pl.concat(frames).sort(["año", "semana", "centro"])

unmapped = d.filter(pl.col("categoria").is_null())["centro"].unique().to_list()
if unmapped:
    raise ValueError(f"Unmapped centro values (add to CATEGORIA_MAP): {unmapped}")

print(f"Rows: {d.height:,} (expected ~7,000-8,200)")
print(f"Unique centros: {d['centro'].n_unique()} (expected 60-70)")

top5_share = (
    d.group_by("centro").agg(pl.col("cuartos_ocup").sum())
    .sort("cuartos_ocup", descending=True)
    .head(5)["cuartos_ocup"].sum() / d["cuartos_ocup"].sum() * 100
)
print(f"Top-5 destinos share of cuartos_ocup: {top5_share:.1f}% (expected ~46%)")

playa_mean = d.filter(pl.col("categoria") == "Playa")["ocupacion"].mean()
ciudad_mean = d.filter(pl.col("categoria") == "Ciudad")["ocupacion"].mean()
print(f"Playa mean ocupación: {playa_mean:.1f}%, Ciudad mean: {ciudad_mean:.1f}%, gap: {playa_mean - ciudad_mean:.1f}pp (expected ~+13pp)")

assert 7000 <= d.height <= 8200, f"Row count {d.height} out of expected range"
assert 60 <= d["centro"].n_unique() <= 70, f"Centro count {d['centro'].n_unique()} out of expected range"
assert 44 <= top5_share <= 48, f"Top-5 share {top5_share:.1f}% out of expected range"
assert 11 <= (playa_mean - ciudad_mean) <= 15, f"Playa-ciudad gap {playa_mean - ciudad_mean:.1f}pp out of expected range"

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
d.write_parquet(OUT_PATH)
print(f"Saved → {OUT_PATH}")
