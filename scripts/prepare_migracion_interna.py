"""
Prepara todas las tablas derivadas del dashboard de Migración Interna Estimada.

Método: Ecuación demográfica de balanza de componentes
  Migración neta(t) = Población(t+1) − Población(t) − Nacimientos(t) + Defunciones(t)

Fuentes:
  • INEGI – nacimientos/defunciones por municipio, 2017-2024
    data/inegi/nacimientos_descesos/{año}.csv
  • CONAPO – población municipal 2017-2025 (ya preparada por
    scripts/prepare_conapo_pob_municipal.py)
    dashboard_data/conapo_pob_municipal.parquet
  • RNPDNO – Registro Nacional de Personas Desaparecidas y No Localizadas
    data/datamx/desaparecidos/desaparecidos.csv
  • SESNSP – Incidencia delictiva del fuero común
    data/incidencia_delictiva/incidencia_fuero_comun/incidencia_delictiva_fuero_comun.parquet
  • INEGI – Indicadores laborales (ENOE, Q1)
    data/inegi/indicadores_laborales/indicadores_laborales.parquet
  • INEGI – ENVIPE panel estatal (ya preparado)
    data/inegi/envipe/envipe_state_panel.parquet
  • CONEVAL – pobreza extrema e ingreso, bienal
    data/coneval/Python_MMP_{año}.zip
  • Banxico – remesas por entidad (CA79)
    data/banxico/remesas/Consulta_20260615-174730348.csv
  • CONAPO – intensidad migratoria México-EE.UU. 2020
    data/conapo/intensidad_migratoria/06_iim_mex_eeuu_2020_municipio.csv

Salida: dashboard_data/migracion_*.parquet (consumidas por
dashboard/internal_migration_flow.py)

Run: uv run python scripts/prepare_migracion_interna.py
"""
import io
import zipfile
from pathlib import Path

import polars as pl

OUT_DIR = Path("dashboard_data")
OUT_DIR.mkdir(parents=True, exist_ok=True)

_STATE_RENAME = {
    "Michoacán de Ocampo":             "Michoacán",
    "Coahuila de Zaragoza":            "Coahuila",
    "Veracruz de Ignacio de la Llave": "Veracruz",
}

# 1) Births and deaths 2017-2024.
#    tloc_resid categories (locality size) are mutually exclusive -> sum all.
#    Exclude mun_resid=999 (unspecified municipality).
_bd_raw = pl.concat([
    pl.read_csv(f"data/inegi/nacimientos_descesos/{yr}.csv")
    for yr in range(2017, 2025)
])
df_bd = (
    _bd_raw
    .filter(pl.col("mun_resid") != 999)
    .group_by(["ent_resid", "mun_resid", "anio"])
    .agg(pl.col("total_nac").sum(), pl.col("total_des").sum())
    .with_columns(
        (pl.col("ent_resid").cast(pl.Utf8).str.zfill(2)
         + pl.col("mun_resid").cast(pl.Utf8).str.zfill(3)).alias("CLAVE")
    )
    .rename({"anio": "año"})
)

# 2) Population 2017-2024: already summed over sex, 5-digit CLAVE.
df_pop = (
    pl.read_parquet("dashboard_data/conapo_pob_municipal.parquet")
    .filter(pl.col("AÑO").is_between(2017, 2024))
    .with_columns(pl.col("CLAVE").cast(pl.Utf8).str.zfill(5))
    .rename({"AÑO": "año"})
)

# pop_next at year t  = population of year t+1 (shift index back by 1)
_pop_next = (
    df_pop.select(["CLAVE", "año", "POB_TOTAL"])
    .with_columns((pl.col("año") - 1).alias("año"))
    .rename({"POB_TOTAL": "pop_next"})
)

# 3) Compute net migration per (municipality, year) for 2017-2023.
df_mig_mun = (
    df_bd.filter(pl.col("año").is_between(2017, 2023))
    .join(
        df_pop.rename({"POB_TOTAL": "pop_t"}).select(
            ["CLAVE", "CLAVE_ENT", "NOM_ENT", "NOM_MUN", "año", "pop_t"]
        ),
        on=["CLAVE", "año"], how="inner",
    )
    .join(_pop_next, on=["CLAVE", "año"], how="inner")
    .with_columns([
        (pl.col("pop_next") - pl.col("pop_t")
         - pl.col("total_nac") + pl.col("total_des")).alias("net_mig"),
        (pl.col("total_nac") - pl.col("total_des")).alias("natural_growth"),
    ])
)

# 4) State-level aggregates (trusted, used for most charts).
df_mig_state = (
    df_mig_mun
    .group_by(["CLAVE_ENT", "NOM_ENT", "año"])
    .agg(
        pl.col("net_mig").sum(),
        pl.col("natural_growth").sum(),
        pl.col("total_nac").sum().alias("births"),
        pl.col("total_des").sum().alias("deaths"),
        pl.col("pop_t").sum(),
    )
    .sort(["NOM_ENT", "año"])
)

# ── Desaparecidos ─────────────────────────────────────────────────────────────
# Source: data/datamx/desaparecidos/desaparecidos.csv (RNPDNO register)
# ~57% of records have a known FECHA_DESAPARICION; the rest are CONFIDENCIAL.
# Only those with parseable dates are used here.

_DES_NORM = {
    "AGUASCALIENTES": "Aguascalientes", "BAJA CALIFORNIA": "Baja California",
    "BAJA CALIFORNIA SUR": "Baja California Sur", "CAMPECHE": "Campeche",
    "CHIAPAS": "Chiapas", "CHIHUAHUA": "Chihuahua",
    "CIUDAD DE MÉXICO": "Ciudad de México", "COAHUILA": "Coahuila",
    "COLIMA": "Colima", "DURANGO": "Durango", "ESTADO DE MÉXICO": "México",
    "GUANAJUATO": "Guanajuato", "GUERRERO": "Guerrero", "HIDALGO": "Hidalgo",
    "JALISCO": "Jalisco", "MICHOACÁN": "Michoacán", "MORELOS": "Morelos",
    "NAYARIT": "Nayarit", "NUEVO LEÓN": "Nuevo León", "OAXACA": "Oaxaca",
    "PUEBLA": "Puebla", "QUERÉTARO": "Querétaro", "QUINTANA ROO": "Quintana Roo",
    "SAN LUIS POTOSÍ": "San Luis Potosí", "SINALOA": "Sinaloa", "SONORA": "Sonora",
    "TABASCO": "Tabasco", "TAMAULIPAS": "Tamaulipas", "TLAXCALA": "Tlaxcala",
    "VERACRUZ": "Veracruz", "YUCATÁN": "Yucatán", "ZACATECAS": "Zacatecas",
}

_df_des_all = (
    pl.read_csv("data/datamx/desaparecidos/desaparecidos.csv", infer_schema_length=5000)
    .filter(pl.col("CVE_ENT").is_between(1, 32))
    .with_columns(pl.col("ENTIDAD").replace(_DES_NORM).alias("NOM_ENT"))
)

# Completeness: share of records with parseable date per state (for chart annotations)
df_des_completeness = (
    _df_des_all
    .group_by("NOM_ENT")
    .agg(
        pl.len().alias("total"),
        pl.col("FECHA_DESAPARICION").str.slice(0, 10)
          .str.to_date("%Y-%m-%d", strict=False).is_not_null().sum().alias("dated"),
    )
    .with_columns(
        (pl.col("dated") / pl.col("total") * 100).alias("completeness_pct")
    )
)

_df_des_raw = (
    _df_des_all
    .with_columns(
        pl.col("FECHA_DESAPARICION").str.slice(0, 10)
          .str.to_date("%Y-%m-%d", strict=False).alias("fecha_des")
    )
    .filter(pl.col("fecha_des").is_not_null())
    .with_columns(
        pl.col("fecha_des").dt.year().cast(pl.Int32).alias("año"),
    )
)

# Pre-aggregate to (NOM_ENT, año) so callbacks just filter this small table
df_des_by_year = (
    _df_des_raw
    .group_by(["NOM_ENT", "año"])
    .agg(pl.len().alias("desaparecidos"))
)

# 5) Adjusted migration: desaparecidos treated as hidden deaths (RNPDNO, date-known only).
#    net_mig_adj = net_mig + desaparecidos
df_mig_state_adj = (
    df_mig_state
    .join(
        df_des_by_year.with_columns(pl.col("año").cast(pl.Int64)),
        on=["NOM_ENT", "año"], how="left",
    )
    .with_columns(
        pl.col("desaparecidos").fill_null(0),
        (pl.col("net_mig") + pl.col("desaparecidos")).alias("net_mig_adj"),
    )
)

# ── Incidencia delictiva ──────────────────────────────────────────────────────
# Source: data/incidencia_delictiva/incidencia_fuero_comun/incidencia_delictiva_fuero_comun.parquet (SESNSP)
# Monthly counts per (state, municipality, crime type). Sum across 12 months = annual total.
# Some months carry −1 values (data corrections) → clip to 0 before summing.

_MONTHS = ["Enero","Febrero","Marzo","Abril","Mayo","Junio",
           "Julio","Agosto","Septiembre","Octubre","Noviembre","Diciembre"]

_lf_crime = (
    pl.scan_parquet("data/incidencia_delictiva/incidencia_fuero_comun/incidencia_delictiva_fuero_comun.parquet")
    .with_columns(
        pl.sum_horizontal(*[pl.col(m).fill_null(0).clip(lower_bound=0) for m in _MONTHS])
        .alias("Casos")
    )
)

_q_total = _lf_crime.group_by(["Año","Clave_Ent"]).agg(pl.col("Casos").sum())
_q_sek   = (_lf_crime.filter(pl.col("Tipo de delito") == "Secuestro")
             .group_by(["Año","Clave_Ent"]).agg(pl.col("Casos").sum().alias("sek")))
_q_hom   = (_lf_crime.filter(pl.col("Tipo de delito") == "Homicidio")
             .group_by(["Año","Clave_Ent"]).agg(pl.col("Casos").sum().alias("hom")))
_q_tipo  = _lf_crime.group_by(["Año","Clave_Ent","Tipo de delito"]).agg(pl.col("Casos").sum())

df_crime_state, df_sek_state, df_hom_state, df_crime_tipo = pl.collect_all(
    [_q_total, _q_sek, _q_hom, _q_tipo]
)

# ── Indicadores laborales (ENOE, Q1) ─────────────────────────────────────────
# Source: scripts/indicadores_laborales.py → data/inegi/indicadores_laborales/indicadores_laborales.parquet
# est==1 (valor puntual). cve_mun==0 = agregado estatal. Years 2017-2025.

df_lab = (
    pl.scan_parquet("data/inegi/indicadores_laborales/indicadores_laborales.parquet")
    .filter((pl.col("cve_mun") == 0) & pl.col("cve_ent").is_between(1, 32))
    .select(["año", "cve_ent", "nom_ent", "pea", "ocupados", "informales"])
    .collect()
)

# ── CONEVAL: pobreza extrema e ingreso ────────────────────────────────────────
# Source: data/coneval/Python_MMP_{YEAR}.zip → Base final/pobreza{YY}.csv
# Biennial survey: 2016, 2018, 2020, 2022.  Streamed without full extraction.
# ic_asalud excluded: methodology break Seguro Popular → INSABI between 2018 and 2020.

_CONEVAL_ZIPS = {
    2016: ("data/coneval/Python_MMP_2016.zip", "Base final/pobreza16.csv"),
    2018: ("data/coneval/Python_MMP_2018.zip", "Base final/pobreza18.csv"),
    2020: ("data/coneval/Python_MMP_2020.zip", "Base final/pobreza20.csv"),
    2022: ("data/coneval/Python_MMP_2022.zip", "Base final/pobreza22.csv"),
}

_coneval_frames = []
for _yr, (_path, _member) in _CONEVAL_ZIPS.items():
    with zipfile.ZipFile(_path) as _z:
        with _z.open(_member) as _f:
            _cdf = pl.read_csv(_f, columns=["ent", "factor", "pobreza_e", "plp"], encoding="utf8")
    _coneval_frames.append(_cdf.with_columns(pl.lit(_yr).alias("año_coneval")))

_coneval_raw = pl.concat(_coneval_frames)

# Weighted state-level rates per biennial year
_df_coneval_bienal = (
    _coneval_raw
    .group_by(["ent", "año_coneval"])
    .agg(
        ((pl.col("pobreza_e") * pl.col("factor")).sum() / pl.col("factor").sum()).alias("pobreza_e"),
        ((pl.col("plp") * pl.col("factor")).sum() / pl.col("factor").sum()).alias("plp"),
    )
    .sort(["ent", "año_coneval"])
)

# Forward-fill biennial survey to each migration year (use most recent CONEVAL year ≤ t)
_CONEVAL_MAP = {2017: 2016, 2018: 2018, 2019: 2018, 2020: 2020, 2021: 2020, 2022: 2022, 2023: 2022}

df_coneval_annual = pl.concat([
    _df_coneval_bienal.filter(pl.col("año_coneval") == _cyr)
    .with_columns(pl.lit(_myr).cast(pl.Int64).alias("año"))
    .drop("año_coneval")
    for _myr, _cyr in _CONEVAL_MAP.items()
])

# ── Remesas Banxico (CA79) ────────────────────────────────────────────────────
# Source: data/banxico/remesas/Consulta_20260615-174730348.csv
# Quarterly state-level remittances (millions USD, 2003-2026). Wide format, 10-row header.
# Aggregate 4 quarters → annual sum.

_BANXICO_NAME_MAP = {
    "Estado de México": "México",
    "Coahuila":         "Coahuila de Zaragoza",
    "Michoacán":        "Michoacán de Ocampo",
    "Veracruz":         "Veracruz de Ignacio de la Llave",
}

with open("data/banxico/remesas/Consulta_20260615-174730348.csv", encoding="latin1") as _f:
    _bnx_lines = _f.readlines()
_bnx_start = next(i for i, l in enumerate(_bnx_lines) if l.startswith('"Título"'))
_raw_bnx = pl.read_csv(io.StringIO("".join(_bnx_lines[_bnx_start:])))
_bnx_date_cols = [c for c in _raw_bnx.columns if c[0].isdigit()]
df_banxico_annual = (
    _raw_bnx
    .filter(pl.col("Título").str.contains("Remesas Familiares,"))
    .with_columns(pl.col("Título").str.split(", ").list.get(1).alias("_state_raw"))
    .filter(pl.col("_state_raw") != "TOTAL")
    .with_columns(pl.col("_state_raw").replace(_BANXICO_NAME_MAP).alias("NOM_ENT"))
    .select(["NOM_ENT"] + _bnx_date_cols)
    .unpivot(index="NOM_ENT", variable_name="fecha", value_name="remesas")
    .with_columns(pl.col("fecha").str.slice(6, 4).cast(pl.Int64).alias("año"))
    .group_by(["NOM_ENT", "año"])
    .agg(pl.col("remesas").sum().alias("remesas_mmusd"))
)

# ── CONAPO intensidad migratoria (emigración hacia EE.UU., 2020) ──────────────
# Source: data/conapo/intensidad_migratoria/06_iim_mex_eeuu_2020_municipio.csv
# Municipal level, census 2020. gim_dp2 grade is cross-year comparable.
# CLAVE join: cve_ent (1-32) + cve_mun (EEEMMM) → 5-digit CLAVE matching df_mig_mun.

_im_2020 = pl.read_csv(
    "data/conapo/intensidad_migratoria/06_iim_mex_eeuu_2020_municipio.csv",
    encoding="utf8-lossy",
)
df_intensidad_mun = (
    _im_2020
    .with_columns(
        (pl.col("cve_ent").cast(pl.Utf8).str.zfill(2) +
         (pl.col("cve_mun") - pl.col("cve_ent") * 1000).cast(pl.Utf8).str.zfill(3)
        ).alias("CLAVE"),
        pl.col("gim_dp2").str.to_lowercase().alias("grade"),
    )
    .select(["CLAVE", "cve_ent", "grade", "viv_rem", "viv_emig", "viv_circ", "viv_ret"])
)

# State-level: % municipalities with grade "alto" or "muy alto" (excluding "nulo")
df_intensidad_state = (
    df_intensidad_mun
    .filter(pl.col("grade") != "nulo")
    .group_by("cve_ent")
    .agg(
        (pl.col("grade").is_in(["alto", "muy alto"]).sum().cast(pl.Float64) /
         pl.col("grade").count() * 100).alias("pct_alta_intensidad"),
        pl.col("grade").count().alias("n_mun"),
    )
    .with_columns(pl.col("cve_ent").cast(pl.Int64))
)

# ── Write outputs ─────────────────────────────────────────────────────────────

_OUTPUTS = {
    "migracion_pob_municipal.parquet":            df_pop,
    "migracion_neta_municipio.parquet":            df_mig_mun,
    "migracion_neta_estado.parquet":               df_mig_state,
    "migracion_neta_estado_adj.parquet":           df_mig_state_adj,
    "migracion_desaparecidos_year.parquet":        df_des_by_year,
    "migracion_desaparecidos_completeness.parquet": df_des_completeness,
    "migracion_crimen_estado.parquet":             df_crime_state,
    "migracion_secuestro_estado.parquet":          df_sek_state,
    "migracion_homicidio_estado.parquet":          df_hom_state,
    "migracion_crimen_tipo_estado.parquet":        df_crime_tipo,
    "migracion_laboral_estado.parquet":            df_lab,
    "migracion_coneval_anual.parquet":             df_coneval_annual,
    "migracion_banxico_anual.parquet":             df_banxico_annual,
    "migracion_intensidad_municipio.parquet":      df_intensidad_mun,
    "migracion_intensidad_estado.parquet":         df_intensidad_state,
}

for _name, _df in _OUTPUTS.items():
    _out = OUT_DIR / _name
    _df.write_parquet(_out)
    print(f"Saved → {_out}  ({_df.height:,} rows)")
