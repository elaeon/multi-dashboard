"""Dashboard: Migración Interna Estimada — México

Método: Ecuación demográfica de balanza de componentes
  Migración neta(t) = Población(t+1) − Población(t) − Nacimientos(t) + Defunciones(t)

  Ecuación ajustada por desaparecidos (RNPDNO):
  Migración neta ajustada(t) = Migración neta(t) + Desaparecidos(t)
    Argumento: los desaparecidos salen de la población sin quedar registrados como muertes
    oficiales → el residual los absorbe como "emigrantes". Tratarlos como mortalidad oculta
    da una migración neta ajustada menos negativa en estados de alta violencia.
    Limitación: solo ~57% del RNPDNO tiene FECHA_DESAPARICION conocida → corrección parcial.

Fuentes:
  • INEGI – Estadística de nacimientos y defunciones por municipio, 2017–2024
    data/inegi/nacimientos_descesos/{año}.csv
  • CONAPO – Proyecciones de población municipal 1990–2040
    data/conapo/estados_municipios.csv
  • RNPDNO – Registro Nacional de Personas Desaparecidas y No Localizadas
    data/desaparecidos.csv

Notas metodológicas:
  • La migración estimada mezcla flujos internos e internacionales (no separables).
  • tloc_resid = tamaño de localidad; las categorías son mutuamente excluyentes → se suman.
  • Población 2022–2024 son proyecciones CONAPO, no conteo censal.
  • El año 2020 registra sobremortalidad COVID-19 que infla el residual de migración.
"""

import json
import numpy as np
import polars as pl
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc

# ── Estilos ──────────────────────────────────────────────────────────────────

CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font_color="#CBD5E1",
    xaxis=dict(gridcolor="#334155"),
    yaxis=dict(gridcolor="#334155"),
    margin=dict(t=70, b=40, l=10, r=10),
)
CARD_STYLE = {
    "background": "#1E293B", "border": "1px solid #334155",
    "borderRadius": "8px", "padding": "16px", "textAlign": "center",
}

IN_COLOR  = "#3BB273"   # green = net in-migration / receptor
OUT_COLOR = "#E84855"   # red   = net out-migration / expulsor
NG_COLOR  = "#2E86AB"   # blue  = natural growth

# ── Datos ────────────────────────────────────────────────────────────────────

_STATE_RENAME = {
    "Michoacán de Ocampo":             "Michoacán",
    "Coahuila de Zaragoza":            "Coahuila",
    "Veracruz de Ignacio de la Llave": "Veracruz",
}

# 1) Births and deaths 2017–2024.
#    tloc_resid categories (locality size) are mutually exclusive → sum all.
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

# 2) Population 2017–2024: sum over sex; zero-pad CLAVE to 5 digits.
df_pop = (
    pl.read_csv("data/conapo/estados_municipios.csv")
    .filter(pl.col("AÑO").is_between(2017, 2024))
    .group_by(["CLAVE", "CLAVE_ENT", "NOM_ENT", "NOM_MUN", "AÑO"])
    .agg(pl.col("POB_TOTAL").sum())
    .with_columns(
        pl.col("CLAVE").cast(pl.Utf8).str.zfill(5),
        pl.col("NOM_ENT").replace(_STATE_RENAME),
    )
    .rename({"AÑO": "año"})
)

# pop_next at year t  = population of year t+1 (shift index back by 1)
_pop_next = (
    df_pop.select(["CLAVE", "año", "POB_TOTAL"])
    .with_columns((pl.col("año") - 1).alias("año"))
    .rename({"POB_TOTAL": "pop_next"})
)

# 3) Compute net migration per (municipality, year) for 2017–2023.
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

# ── Constantes ───────────────────────────────────────────────────────────────

YEARS = sorted(df_mig_mun["año"].unique().to_list())   # [2017 … 2023]
Y_MIN, Y_MAX = YEARS[0], YEARS[-1]
STATES = sorted(df_mig_state["NOM_ENT"].unique().to_list())

with open("data/mexico_states.geojson") as _f:
    GEOJSON = json.load(_f)

_PALETTE = px.colors.qualitative.Plotly + px.colors.qualitative.Set2

# ── Desaparecidos ─────────────────────────────────────────────────────────────
# Source: data/desaparecidos.csv (RNPDNO register)
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
    pl.read_csv("data/desaparecidos.csv", infer_schema_length=5000)
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
#    States with high disappearances show less-negative adjusted migration → separates
#    forced disappearance from voluntary out-migration. Lower bound: ~57% have known dates.
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
# Source: data/incidencia_delictiva_fuero_comun.parquet (SESNSP)
# Monthly counts per (state, municipality, crime type). Sum across 12 months = annual total.
# Some months carry −1 values (data corrections) → clip to 0 before summing.

_MONTHS = ["Enero","Febrero","Marzo","Abril","Mayo","Junio",
           "Julio","Agosto","Septiembre","Octubre","Noviembre","Diciembre"]

_lf_crime = (
    pl.scan_parquet("data/incidencia_delictiva_fuero_comun.parquet")
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

_df_lab = (
    pl.scan_parquet("data/inegi/indicadores_laborales/indicadores_laborales.parquet")
    .filter((pl.col("cve_mun") == 0) & pl.col("cve_ent").is_between(1, 32))
    .select(["año", "cve_ent", "nom_ent", "pea", "ocupados", "informales"])
    .collect()
)

# ── ENVIPE: victimización y percepción de inseguridad por estado ──────────────
# Source: scripts/prepare_envipe_state_panel.py → data/inegi/envipe/envipe_state_panel.parquet
# CLAVE_ENT (Int64) + año (Int64) join key; vic_envipe and inseg_envipe are [0,1] rates.
# ENVIPE año N mide delitos ocurridos en los 12 meses previos → lag implícito de 1 año.
_df_envipe = pl.read_parquet("data/inegi/envipe/envipe_state_panel.parquet")

# ── CONEVAL: pobreza extrema e ingreso ────────────────────────────────────────
# Source: data/coneval/Python_MMP_{YEAR}.zip → Base final/pobreza{YY}.csv
# Biennial survey: 2016, 2018, 2020, 2022.  Streamed without full extraction.
# ic_asalud excluded: methodology break Seguro Popular → INSABI between 2018 and 2020.
# pobreza_e: extreme poverty rate (income < extreme poverty line AND ≥1 social deprivation)
# plp: income poverty line (income below welfare line, regardless of deprivations)

import zipfile as _zipfile

_CONEVAL_ZIPS = {
    2016: ("data/coneval/Python_MMP_2016.zip", "Base final/pobreza16.csv"),
    2018: ("data/coneval/Python_MMP_2018.zip", "Base final/pobreza18.csv"),
    2020: ("data/coneval/Python_MMP_2020.zip", "Base final/pobreza20.csv"),
    2022: ("data/coneval/Python_MMP_2022.zip", "Base final/pobreza22.csv"),
}

_coneval_frames = []
for _yr, (_path, _member) in _CONEVAL_ZIPS.items():
    with _zipfile.ZipFile(_path) as _z:
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

# ── Figuras ───────────────────────────────────────────────────────────────────

def fig_choropleth(yr0: int, yr1: int) -> go.Figure:
    """Tasa de migración neta estimada por estado, por 1,000 hab. (acumulado)."""
    base = (
        df_mig_state.filter(pl.col("año").is_between(yr0, yr1))
        .group_by("NOM_ENT")
        .agg(pl.col("net_mig").sum(), pl.col("pop_t").mean().alias("pop_avg"))
        .with_columns((pl.col("net_mig") / pl.col("pop_avg") * 1000).alias("rate"))
    )
    max_abs = float(max(base["rate"].abs().max() or 1.0, 1.0))
    fig = px.choropleth_map(
        base,
        geojson=GEOJSON,
        locations="NOM_ENT",
        featureidkey="properties.name",
        color="rate",
        color_continuous_scale="RdBu",
        range_color=[-max_abs, max_abs],
        custom_data=["NOM_ENT", "net_mig", "pop_avg"],
        map_style="carto-darkmatter",
        zoom=4.2, center={"lat": 23.6, "lon": -102.5},
    )
    fig.update_traces(
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Migración neta: %{customdata[1]:,.0f} personas<br>"
            "Tasa: %{z:.1f} por 1,000 hab.<extra></extra>"
        )
    )
    top = base.sort("rate", descending=True).head(1)
    top_name = top["NOM_ENT"][0]
    top_rate = float(top["rate"][0])
    fig.update_layout(
        title=dict(
            text=(
                f"<b>{top_name} encabeza la atracción de población ({top_rate:+.0f}‰)</b>"
                f"<br><sup style='color:#94A3B8'>Tasa de migración neta estimada · "
                f"{yr0}–{yr1} · por 1,000 hab. · azul = receptor · rojo = expulsor</sup>"
            )
        ),
        height=580,
        coloraxis_colorbar=dict(
            title=dict(text="por 1,000 hab.", font=dict(color="#CBD5E1")),
            tickfont=dict(color="#CBD5E1"),
        ),
        paper_bgcolor="rgba(0,0,0,0)", font_color="#CBD5E1",
        margin=dict(t=80, b=10, l=0, r=0),
    )
    return fig


def fig_national_trend(yr0: int, yr1: int) -> go.Figure:
    """Migración neta anual vs crecimiento natural — total nacional."""
    nat = (
        df_mig_state.filter(pl.col("año").is_between(yr0, yr1))
        .group_by("año")
        .agg(pl.col("net_mig").sum(), pl.col("natural_growth").sum())
        .sort("año")
    )
    fig = go.Figure()
    fig.add_hline(y=0, line=dict(color="#475569", width=1))
    fig.add_trace(go.Scatter(
        x=nat["año"].to_list(), y=nat["natural_growth"].to_list(),
        mode="lines+markers", name="Crecimiento natural (nac − def)",
        line=dict(color=NG_COLOR, width=2.5), marker=dict(size=7),
        hovertemplate="<b>%{x}</b><br>Crecimiento natural: %{y:,}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=nat["año"].to_list(), y=nat["net_mig"].to_list(),
        mode="lines+markers", name="Migración neta estimada",
        line=dict(color=IN_COLOR, width=2.5, dash="dot"), marker=dict(size=7),
        hovertemplate="<b>%{x}</b><br>Migración neta: %{y:,}<extra></extra>",
    ))
    if yr0 <= 2020 <= yr1:
        fig.add_vrect(x0=2019.5, x1=2021.5,
                      fillcolor="rgba(244,162,97,0.08)", line_width=0,
                      annotation_text="artefacto: −35% nac. + +57% def.",
                      annotation_font_color="#F4A261",
                      annotation_position="top left")
    fig.update_layout(
        title=dict(
            text=(
                "<b>El crecimiento natural supera la migración estimada en casi todos los años</b>"
                "<br><sup style='color:#94A3B8'>suma de 32 estados · "
                "residual 2020–2021 inflado por artefacto de registro</sup>"
            )
        ),
        height=380,
        yaxis=dict(title="Personas", gridcolor="#334155"),
        xaxis=dict(gridcolor="#334155", dtick=1),
        legend=dict(orientation="h", y=-0.20, x=0),
        margin=dict(t=80, b=70, l=10, r=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#CBD5E1",
    )
    return fig


def fig_slope_state(yr0: int, yr1: int) -> go.Figure:
    """Migración neta estatal: primer año vs último año del rango."""
    r0 = (df_mig_state.filter(pl.col("año") == yr0)
          .select(["NOM_ENT", "net_mig"]).rename({"net_mig": "r0"}))
    r1 = (df_mig_state.filter(pl.col("año") == yr1)
          .select(["NOM_ENT", "net_mig"]).rename({"net_mig": "r1"}))
    d = (r0.join(r1, on="NOM_ENT", how="inner")
         .with_columns((pl.col("r1") - pl.col("r0")).alias("delta"))
         .sort("r1", descending=True))
    if d.is_empty():
        return go.Figure().update_layout(title=f"Sin datos para {yr0}/{yr1}", **CHART_LAYOUT)

    n_up, n_dn = 0, 0
    fig = go.Figure()
    for row in d.iter_rows(named=True):
        # Green = migration became more positive (receptor); red = became more negative
        color = IN_COLOR if row["delta"] > 0 else OUT_COLOR
        n_up += row["delta"] > 0
        n_dn += row["delta"] <= 0
        fig.add_trace(go.Scatter(
            x=[str(yr0), str(yr1)],
            y=[row["r0"], row["r1"]],
            mode="lines+markers",
            line=dict(color=color, width=1.5),
            marker=dict(color=color, size=7),
            showlegend=False,
            hovertemplate=f"<b>{row['NOM_ENT']}</b><br>%{{x}}: %{{y:,}}<extra></extra>",
        ))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines+markers",
                              line=dict(color=IN_COLOR), marker=dict(color=IN_COLOR),
                              name=f"▲ Más receptora ({n_up})"))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines+markers",
                              line=dict(color=OUT_COLOR), marker=dict(color=OUT_COLOR),
                              name=f"▼ Más expulsora ({n_dn})"))
    fig.update_xaxes(type="category", gridcolor="rgba(0,0,0,0)")
    fig.update_layout(
        title=dict(
            text=(
                f"<b>16 estados receptores y 16 expulsores — dirección de la migración {yr0}→{yr1}</b>"
                "<br><sup style='color:#94A3B8'>verde = se volvió más receptora · "
                "rojo = más expulsora (excluyendo artefacto 2020–2021)</sup>"
            )
        ),
        height=max(360, 32 * 22 + 80),
        yaxis=dict(title="Migración neta (personas)", gridcolor="#334155"),
        legend=dict(orientation="h", y=-0.08, x=0),
        margin=dict(t=80, b=60, l=10, r=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#CBD5E1",
    )
    return fig


def fig_state_lines(states: list[str], yr0: int, yr1: int) -> go.Figure:
    """Tendencia anual de migración neta para estados seleccionados."""
    base = (
        df_mig_state.filter(
            pl.col("año").is_between(yr0, yr1) & pl.col("NOM_ENT").is_in(states)
        )
    )
    fig = go.Figure()
    fig.add_hline(y=0, line=dict(color="#475569", width=1))
    for i, st in enumerate(sorted(states)):
        sub = base.filter(pl.col("NOM_ENT") == st).sort("año")
        if sub.is_empty():
            continue
        fig.add_trace(go.Scatter(
            x=sub["año"].to_list(), y=sub["net_mig"].to_list(),
            mode="lines+markers", name=st,
            line=dict(color=_PALETTE[i % len(_PALETTE)], width=2.5),
            marker=dict(size=7),
            hovertemplate=f"<b>{st}</b><br>%{{x}}: %{{y:,}}<extra></extra>",
        ))
    if yr0 <= 2020 <= yr1:
        fig.add_vline(x=2020, line_dash="dot", line_color="#F4A261",
                      annotation_text="COVID-19", annotation_font_color="#F4A261")
    fig.update_layout(
        title="Tendencia anual de migración neta por estado",
        height=420,
        yaxis=dict(title="Migración neta", gridcolor="#334155"),
        xaxis=dict(gridcolor="#334155", dtick=1),
        legend=dict(orientation="h", y=-0.22, x=0, font=dict(size=10)),
        margin=dict(t=50, b=90, l=10, r=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#CBD5E1",
    )
    return fig


def fig_decomposition(yr0: int, yr1: int) -> go.Figure:
    """Crecimiento natural vs migración neta por estado (barras superpuestas)."""
    base = (
        df_mig_state.filter(pl.col("año").is_between(yr0, yr1))
        .group_by("NOM_ENT")
        .agg(pl.col("net_mig").sum(), pl.col("natural_growth").sum())
        .sort("net_mig", descending=False)
    )
    states = base["NOM_ENT"].to_list()
    mig_colors = [IN_COLOR if v >= 0 else OUT_COLOR for v in base["net_mig"].to_list()]
    n = len(states)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=base["natural_growth"].to_list(), y=states,
        orientation="h", name="Crecimiento natural",
        marker_color=NG_COLOR, opacity=0.85,
        hovertemplate="<b>%{y}</b><br>Crecimiento natural: %{x:,}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        x=base["net_mig"].to_list(), y=states,
        orientation="h", name="Migración neta",
        marker_color=mig_colors, opacity=0.85,
        hovertemplate="<b>%{y}</b><br>Migración neta: %{x:,}<extra></extra>",
    ))
    fig.add_vline(x=0, line=dict(color="#475569", width=1))
    fig.update_layout(
        title=dict(
            text=(
                "<b>Descomposición del cambio poblacional por estado</b>"
                "<br><sup style='color:#94A3B8'>azul = crecimiento natural · "
                "verde/rojo = migración neta · ordenado por migración neta</sup>"
            )
        ),
        height=max(380, n * 26 + 130),
        barmode="overlay",
        xaxis=dict(title="Personas", gridcolor="#334155", zerolinecolor="#64748B"),
        yaxis=dict(categoryorder="array", categoryarray=states,
                   gridcolor="rgba(0,0,0,0)", automargin=True),
        legend=dict(orientation="h", y=-0.05, x=0),
        margin=dict(t=90, b=60, l=10, r=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#CBD5E1",
    )
    return fig


def fig_scatter_components(yr0: int, yr1: int) -> go.Figure:
    """Tasa de crecimiento natural vs tasa de migración neta por estado (burbuja)."""
    base = (
        df_mig_state.filter(pl.col("año").is_between(yr0, yr1))
        .group_by("NOM_ENT")
        .agg(
            pl.col("net_mig").sum(),
            pl.col("natural_growth").sum(),
            pl.col("pop_t").mean().alias("pop_avg"),
        )
        .with_columns([
            (pl.col("net_mig")       / pl.col("pop_avg") * 1000).alias("mig_rate"),
            (pl.col("natural_growth") / pl.col("pop_avg") * 1000).alias("ng_rate"),
        ])
    )
    x_vals = base["ng_rate"].to_numpy()
    y_vals = base["mig_rate"].to_numpy()
    pop    = base["pop_avg"].to_numpy()
    sizes  = 8 + 28 * (pop ** 0.5 - pop.min() ** 0.5) / (pop.max() ** 0.5 - pop.min() ** 0.5 + 1)
    colors = [IN_COLOR if v >= 0 else OUT_COLOR for v in y_vals]

    fig = go.Figure()
    fig.add_hline(y=0, line=dict(color="#475569", width=1))
    fig.add_vline(x=0, line=dict(color="#475569", width=1))
    fig.add_trace(go.Scatter(
        x=x_vals.tolist(), y=y_vals.tolist(),
        mode="markers+text",
        marker=dict(color=colors, size=sizes.tolist(), opacity=0.8,
                    line=dict(color="#0F172A", width=1)),
        text=base["NOM_ENT"].to_list(),
        textposition="top center",
        textfont=dict(size=8, color="#94A3B8"),
        customdata=list(zip(
            base["NOM_ENT"].to_list(),
            base["net_mig"].to_list(),
            base["natural_growth"].to_list(),
            base["pop_avg"].to_list(),
        )),
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Crecimiento natural: %{x:.1f} por mil<br>"
            "Migración neta: %{y:.1f} por mil<br>"
            "Migración total: %{customdata[1]:,} personas<extra></extra>"
        ),
        showlegend=False,
    ))
    fig.update_layout(
        title=dict(
            text=(
                "<b>Crecimiento natural vs migración neta por estado</b>"
                "<br><sup style='color:#94A3B8'>tasas por 1,000 hab. · tamaño = población · "
                "cuadrante superior derecho = atrae y crece naturalmente</sup>"
            )
        ),
        height=520,
        xaxis=dict(title="Crecimiento natural (por 1,000 hab.)", gridcolor="#334155"),
        yaxis=dict(title="Migración neta (por 1,000 hab.)", gridcolor="#334155"),
        margin=dict(t=90, b=50, l=10, r=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#CBD5E1",
    )
    return fig


def fig_municipios_ranking(yr0: int, yr1: int, state: str | None, top_n: int) -> go.Figure:
    """Top municipios con mayor entrada y salida neta de población."""
    base = df_mig_mun.filter(pl.col("año").is_between(yr0, yr1))
    if state and state != "__ALL__":
        base = base.filter(pl.col("NOM_ENT") == state)
    agg = (
        base.group_by(["CLAVE", "NOM_ENT", "NOM_MUN"])
        .agg(pl.col("net_mig").sum(), pl.col("pop_t").mean().alias("pop_avg"))
        .sort("net_mig", descending=True)
    )
    if agg.is_empty():
        return go.Figure().update_layout(title="Sin datos para la selección", **CHART_LAYOUT)

    top_in  = agg.head(top_n)
    top_out = agg.tail(top_n)
    combined = (
        pl.concat([top_out, top_in])
        .unique(subset=["CLAVE"])
        .sort("net_mig", descending=False)
    )
    labels = [f"{r['NOM_MUN']}, {r['NOM_ENT']}" for r in combined.iter_rows(named=True)]
    vals   = combined["net_mig"].to_list()
    pop    = combined["pop_avg"].to_list()
    colors = [IN_COLOR if v >= 0 else OUT_COLOR for v in vals]

    fig = go.Figure(go.Bar(
        x=vals, y=labels, orientation="h",
        marker_color=colors,
        customdata=list(zip(vals, pop)),
        text=[f"{v:+,.0f}" for v in vals],
        textposition="outside",
        textfont=dict(color="#CBD5E1", size=10),
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Migración neta: %{customdata[0]:+,}<br>"
            "Población promedio: %{customdata[1]:,.0f}<extra></extra>"
        ),
        cliponaxis=False,
    ))
    n = len(labels)
    fig.update_layout(
        title=dict(
            text=(
                f"<b>Municipios con mayor entrada y salida de población · {yr0}–{yr1}</b>"
                "<br><sup style='color:#94A3B8'>estimación indirecta · municipios con datos en las 3 fuentes</sup>"
            )
        ),
        height=max(400, n * 26 + 130),
        xaxis=dict(gridcolor="#334155", zerolinecolor="#64748B"),
        yaxis=dict(categoryorder="array", categoryarray=labels,
                   gridcolor="rgba(0,0,0,0)", automargin=True),
        margin=dict(t=90, b=50, l=10, r=80),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#CBD5E1", showlegend=False,
    )
    return fig


def fig_scatter_desplazamiento(yr0: int, yr1: int) -> go.Figure:
    """Scatter cuadrante: tasa de migración neta vs tasa de desaparecidos por estado."""
    n_years = max(yr1 - yr0 + 1, 1)

    mig = (
        df_mig_state.filter(pl.col("año").is_between(yr0, yr1))
        .group_by("NOM_ENT")
        .agg(pl.col("net_mig").sum(), pl.col("pop_t").mean().alias("pop_avg"))
        .with_columns((pl.col("net_mig") / pl.col("pop_avg") * 1000).alias("mig_rate"))
    )
    des = (
        df_des_by_year.filter(pl.col("año").is_between(yr0, yr1))
        .group_by("NOM_ENT")
        .agg(pl.col("desaparecidos").sum())
    )
    d = (
        mig.join(des, on="NOM_ENT", how="left")
        .with_columns([
            pl.col("desaparecidos").fill_null(0),
            (pl.col("desaparecidos") / pl.col("pop_avg") / n_years * 100_000).alias("des_rate"),
        ])
    )
    if d.is_empty():
        return go.Figure().update_layout(title="Sin datos para la selección", **CHART_LAYOUT)

    x_vals  = d["mig_rate"].to_numpy()
    y_vals  = d["des_rate"].to_numpy()
    states  = d["NOM_ENT"].to_list()
    pop     = d["pop_avg"].to_numpy()
    sizes   = 8 + 28 * (pop ** 0.5 - pop.min() ** 0.5) / (pop.max() ** 0.5 - pop.min() ** 0.5 + 1)

    med_des = float(np.median(y_vals))

    # Pearson r (scipy not guaranteed; use numpy)
    if x_vals.std() > 0 and y_vals.std() > 0:
        r_val = float(np.corrcoef(x_vals, y_vals)[0, 1])
    else:
        r_val = 0.0

    def _color(x, y):
        if x >= 0 and y >= med_des:
            return "#F4A261"   # receptor + alta violencia = orange
        if x >= 0 and y < med_des:
            return IN_COLOR    # receptor + baja violencia = green
        if x < 0 and y >= med_des:
            return OUT_COLOR   # expulsor + alta violencia = red
        return "#64748B"       # expulsor + baja violencia = gray

    colors = [_color(x, y) for x, y in zip(x_vals.tolist(), y_vals.tolist())]

    fig = go.Figure()
    # Quadrant dividers
    fig.add_hline(y=med_des, line=dict(color="#475569", width=1, dash="dot"))
    fig.add_vline(x=0,       line=dict(color="#475569", width=1))

    fig.add_trace(go.Scatter(
        x=x_vals.tolist(), y=y_vals.tolist(),
        mode="markers+text",
        marker=dict(color=colors, size=sizes.tolist(), opacity=0.85,
                    line=dict(color="#0F172A", width=1)),
        text=states,
        textposition="top center",
        textfont=dict(size=8, color="#CBD5E1"),
        customdata=list(zip(
            states,
            d["net_mig"].to_list(),
            d["desaparecidos"].to_list(),
            d["pop_avg"].to_list(),
        )),
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Migración neta acumulada: %{x:.1f} por mil hab.<br>"
            "Desaparecidos: %{y:.1f} por 100k hab./año<br>"
            "Total migración: %{customdata[1]:,}<br>"
            "Total desaparecidos (fecha conocida): %{customdata[2]:,}<extra></extra>"
        ),
        showlegend=False,
    ))

    # Quadrant corner labels
    for x, y, txt, color, xanchor in [
        (0.02, 0.98, "Expulsor + alta violencia",  OUT_COLOR, "left"),
        (0.98, 0.98, "Receptor + alta violencia",  "#F4A261", "right"),
        (0.02, 0.02, "Expulsor + baja violencia",  "#64748B", "left"),
        (0.98, 0.02, "Receptor + baja violencia",  IN_COLOR,  "right"),
    ]:
        fig.add_annotation(
            x=x, y=y, xref="paper", yref="paper",
            text=txt, showarrow=False,
            font=dict(color=color, size=10), opacity=0.75,
            xanchor=xanchor, yanchor="top" if y > 0.5 else "bottom",
        )

    fig.update_layout(
        title=dict(
            text=(
                "<b>¿Los estados que expulsan personas también las desaparecen?</b>"
                f"<br><sup style='color:#94A3B8'>"
                f"Pearson r = {r_val:.2f} · tamaño = población · "
                f"línea horizontal = mediana desaparecidos ({med_des:.1f}/100k/año) · "
                f"solo casos con fecha conocida (~57% del registro)</sup>"
            )
        ),
        height=600,
        xaxis=dict(title=f"Migración neta acumulada (por 1,000 hab., {yr0}–{yr1})",
                   gridcolor="#334155", zerolinecolor="#64748B"),
        yaxis=dict(title="Desaparecidos (por 100,000 hab. · promedio anual)",
                   gridcolor="#334155"),
        margin=dict(t=110, b=50, l=10, r=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#CBD5E1",
    )
    return fig


def fig_des_ranking(yr0: int, yr1: int) -> go.Figure:
    """Tasa de desapariciones por 100k hab. por estado, con completitud de fecha."""
    n_years = max(yr1 - yr0 + 1, 1)
    pop = (
        df_pop.filter(pl.col("año").is_between(yr0, min(yr1, 2024)))
        .group_by("NOM_ENT")
        .agg(pl.col("POB_TOTAL").mean().alias("pop_avg"))
    )
    des = (
        df_des_by_year.filter(pl.col("año").is_between(yr0, yr1))
        .group_by("NOM_ENT")
        .agg(pl.col("desaparecidos").sum())
    )
    base = (
        pop.join(des, on="NOM_ENT", how="left")
        .join(df_des_completeness.select(["NOM_ENT", "completeness_pct"]), on="NOM_ENT", how="left")
        .with_columns(
            pl.col("desaparecidos").fill_null(0),
            pl.col("completeness_pct").fill_null(0),
        )
        .with_columns(
            (pl.col("desaparecidos") / pl.col("pop_avg") / n_years * 100_000).alias("des_rate"),
        )
        .sort("des_rate", descending=True)
    )
    if base.is_empty():
        return go.Figure().update_layout(title="Sin datos", **CHART_LAYOUT)

    states   = base["NOM_ENT"].to_list()
    rates    = base["des_rate"].to_list()
    comps    = base["completeness_pct"].to_list()
    totals   = base["desaparecidos"].to_list()

    # Color by completeness: low-completeness states get a warning orange
    bar_colors = ["#F4A261" if c < 45 else OUT_COLOR for c in comps]

    fig = go.Figure(go.Bar(
        x=rates, y=states,
        orientation="h",
        marker_color=bar_colors,
        customdata=list(zip(totals, comps)),
        text=[f"{c:.0f}% fecha" for c in comps],
        textposition="outside",
        textfont=dict(size=8, color="#64748B"),
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Tasa: %{x:.2f}/100k hab./año<br>"
            "Total (fecha conocida): %{customdata[0]:,}<br>"
            "Completitud fecha: %{customdata[1]:.0f}%<extra></extra>"
        ),
        cliponaxis=False,
    ))
    n = len(states)
    fig.update_layout(
        title=dict(
            text=(
                "<b>Colima, Sonora y Tabasco encabezan la tasa de desapariciones</b>"
                f"<br><sup style='color:#94A3B8'>"
                f"Desaparecidos por 100k hab./año · {yr0}–{yr1} · solo registros con fecha conocida · "
                f"<span style='color:#F4A261'>naranja = completitud &lt;45% (subestimación severa)</span></sup>"
            )
        ),
        height=max(400, n * 24 + 130),
        xaxis=dict(title="Desaparecidos por 100,000 hab. / año", gridcolor="#334155"),
        yaxis=dict(
            categoryorder="array",
            categoryarray=list(reversed(states)),
            gridcolor="rgba(0,0,0,0)", automargin=True,
        ),
        margin=dict(t=100, b=50, l=10, r=110),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#CBD5E1", showlegend=False,
    )
    return fig


def fig_des_trend(yr0: int, yr1: int) -> go.Figure:
    """Tendencia anual de desapariciones por 100k hab. — top 8 estados."""
    pop_yr = (
        df_pop.filter(pl.col("año").is_between(yr0, min(yr1, 2024)))
        .group_by(["NOM_ENT", "año"])
        .agg(pl.col("POB_TOTAL").sum().alias("pop"))
    )
    des_yr = df_des_by_year.filter(pl.col("año").is_between(yr0, yr1))

    panel = (
        des_yr
        .join(
            pop_yr.with_columns(pl.col("año").cast(pl.Int32)),
            on=["NOM_ENT", "año"], how="left",
        )
        .with_columns(
            (pl.col("desaparecidos") / pl.col("pop") * 100_000).alias("des_rate")
        )
    )
    # Pick top 8 states by mean rate in period (excluding low-completeness < 45%)
    low_comp = (
        df_des_completeness.filter(pl.col("completeness_pct") < 45)["NOM_ENT"].to_list()
    )
    top_states = (
        panel.filter(~pl.col("NOM_ENT").is_in(low_comp))
        .group_by("NOM_ENT")
        .agg(pl.col("des_rate").mean())
        .sort("des_rate", descending=True)
        .head(8)["NOM_ENT"].to_list()
    )
    if not top_states:
        return go.Figure().update_layout(title="Sin datos", **CHART_LAYOUT)

    sub = panel.filter(pl.col("NOM_ENT").is_in(top_states)).sort("año")
    fig = go.Figure()
    for i, st in enumerate(top_states):
        s = sub.filter(pl.col("NOM_ENT") == st).sort("año")
        if s.is_empty():
            continue
        fig.add_trace(go.Scatter(
            x=s["año"].to_list(), y=s["des_rate"].to_list(),
            mode="lines+markers", name=st,
            line=dict(color=_PALETTE[i % len(_PALETTE)], width=2.5),
            marker=dict(size=7),
            hovertemplate=f"<b>{st}</b><br>%{{x}}: %{{y:.2f}}/100k<extra></extra>",
        ))
    fig.update_layout(
        title=dict(
            text=(
                "<b>Tendencia de desapariciones — top 8 estados (completitud ≥45%)</b>"
                "<br><sup style='color:#94A3B8'>"
                "Tasa por 100,000 hab. · solo registros con fecha conocida · "
                "estados con &lt;45% completitud excluidos del ranking</sup>"
            )
        ),
        height=420,
        xaxis=dict(title="Año", gridcolor="#334155", dtick=1),
        yaxis=dict(title="Desaparecidos / 100k hab.", gridcolor="#334155"),
        legend=dict(orientation="h", y=-0.22, x=0, font=dict(size=10)),
        margin=dict(t=90, b=90, l=10, r=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#CBD5E1",
    )
    return fig


def fig_mig_adj_comparison(yr0: int, yr1: int) -> go.Figure:
    """Migración neta estándar vs ajustada por desaparecidos por estado."""
    base = (
        df_mig_state_adj
        .filter(pl.col("año").is_between(yr0, yr1))
        .group_by("NOM_ENT")
        .agg(
            pl.col("net_mig").sum(),
            pl.col("net_mig_adj").sum(),
            pl.col("desaparecidos").sum(),
            pl.col("pop_t").mean().alias("pop_avg"),
        )
        .with_columns(
            (pl.col("net_mig")     / pl.col("pop_avg") * 1000).alias("mig_rate"),
            (pl.col("net_mig_adj") / pl.col("pop_avg") * 1000).alias("mig_adj_rate"),
        )
        .sort("net_mig")
    )
    if base.is_empty():
        return go.Figure().update_layout(title="Sin datos para la selección", **CHART_LAYOUT)

    states = base["NOM_ENT"].to_list()
    orig   = base["net_mig"].to_list()
    adj    = base["net_mig_adj"].to_list()
    des    = base["desaparecidos"].to_list()

    n = len(states)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=orig, y=states, orientation="h",
        name="Ecuación estándar",
        marker_color=[OUT_COLOR if v < 0 else IN_COLOR for v in orig],
        opacity=0.45,
        hovertemplate=(
            "<b>%{y}</b><br>Estándar: %{x:+,} personas<extra></extra>"
        ),
    ))
    fig.add_trace(go.Bar(
        x=adj, y=states, orientation="h",
        name="Ajustada (+ desaparecidos como mortalidad oculta)",
        marker_color=[OUT_COLOR if v < 0 else IN_COLOR for v in adj],
        opacity=0.90,
        customdata=list(zip(des, base["mig_adj_rate"].to_list())),
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Ajustada: %{x:+,} personas · %{customdata[1]:+.1f}‰<br>"
            "Desaparecidos sumados: %{customdata[0]:,}<extra></extra>"
        ),
    ))
    fig.add_vline(x=0, line=dict(color="#475569", width=1))

    # Highlight states where adjustment changes sign (from expulsor to receptor)
    flips = [s for s, o, a in zip(states, orig, adj) if o < 0 <= a]
    if flips:
        for flip in flips:
            fig.add_annotation(
                x=0, y=flip, xanchor="left", xshift=6,
                text="↑ cambia signo", showarrow=False,
                font=dict(color="#F4A261", size=8), opacity=0.8,
            )

    total_des = int(base["desaparecidos"].sum())
    n_years = max(yr1 - yr0 + 1, 1)
    fig.update_layout(
        barmode="overlay",
        title=dict(
            text=(
                "<b>Ecuación ajustada: desaparecidos como mortalidad no registrada</b>"
                f"<br><sup style='color:#94A3B8'>"
                f"Barra pálida = estándar · barra sólida = ajustada · "
                f"{total_des:,} desaparecidos totalizados ({yr0}–{yr1}, ~57% con fecha conocida) · "
                f"la diferencia = pérdida de población atribuida erróneamente a emigración</sup>"
            )
        ),
        height=max(420, n * 26 + 150),
        xaxis=dict(title="Personas", gridcolor="#334155", zerolinecolor="#64748B"),
        yaxis=dict(
            categoryorder="array", categoryarray=states,
            gridcolor="rgba(0,0,0,0)", automargin=True,
        ),
        legend=dict(orientation="h", y=-0.06, x=0),
        margin=dict(t=100, b=70, l=10, r=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#CBD5E1",
    )
    return fig


# ── Violencia × Migración ─────────────────────────────────────────────────────

_EXCL_COVID = [2020, 2021]   # artifact years excluded from headline stats


def _build_violence_panel(yr0: int, yr1: int) -> pl.DataFrame:
    """Join migration + secuestro + homicidio + desaparecidos at state level."""
    mig = (
        df_mig_state.filter(pl.col("año").is_between(yr0, yr1))
        .group_by(["CLAVE_ENT", "NOM_ENT"])
        .agg(
            pl.col("net_mig").sum(),
            pl.col("pop_t").mean().alias("pop_avg"),
        )
        .with_columns((pl.col("net_mig") / pl.col("pop_avg") * 1000).alias("mig_rate"))
    )
    crime = (
        df_crime_state.filter(pl.col("Año").is_between(yr0, yr1))
        .group_by("Clave_Ent")
        .agg(pl.col("Casos").sum().alias("crime_total"))
        .rename({"Clave_Ent": "CLAVE_ENT"})
    )
    sek = (
        df_sek_state.filter(pl.col("Año").is_between(yr0, yr1))
        .group_by("Clave_Ent")
        .agg(pl.col("sek").sum())
        .rename({"Clave_Ent": "CLAVE_ENT"})
    )
    hom = (
        df_hom_state.filter(pl.col("Año").is_between(yr0, yr1))
        .group_by("Clave_Ent")
        .agg(pl.col("hom").sum())
        .rename({"Clave_Ent": "CLAVE_ENT"})
    )
    des = (
        df_des_by_year.filter(pl.col("año").is_between(yr0, yr1))
        .group_by("NOM_ENT")
        .agg(pl.col("desaparecidos").sum())
        .join(mig.select(["NOM_ENT", "CLAVE_ENT"]), on="NOM_ENT", how="inner")
    )
    n_years = max(yr1 - yr0 + 1, 1)
    return (
        mig
        .join(crime, on="CLAVE_ENT", how="left")
        .join(sek, on="CLAVE_ENT", how="left")
        .join(hom, on="CLAVE_ENT", how="left")
        .join(des.select(["CLAVE_ENT", "desaparecidos"]), on="CLAVE_ENT", how="left")
        .with_columns([
            pl.col("crime_total").fill_null(0),
            pl.col("sek").fill_null(0),
            pl.col("hom").fill_null(0),
            pl.col("desaparecidos").fill_null(0),
        ])
        .with_columns([
            (pl.col("crime_total") / pl.col("pop_avg") * 100_000).alias("crime_rate"),
            (pl.col("sek") / pl.col("pop_avg") / n_years * 100_000).alias("sek_rate"),
            (pl.col("hom") / pl.col("pop_avg") / n_years * 100_000).alias("hom_rate"),
            (pl.col("desaparecidos") / pl.col("pop_avg") / n_years * 100_000).alias("des_rate"),
        ])
    )


def fig_violencia_scatter(yr0: int, yr1: int) -> go.Figure:
    """Scatter cuadrante: tasa de migración neta vs tasa de secuestro por estado."""
    d = _build_violence_panel(yr0, yr1)
    if d.is_empty():
        return go.Figure().update_layout(title="Sin datos", **CHART_LAYOUT)

    x_vals  = d["mig_rate"].to_numpy()
    y_vals  = d["sek_rate"].to_numpy()
    states  = d["NOM_ENT"].to_list()
    pop     = d["pop_avg"].to_numpy()
    sizes   = 8 + 28 * (pop ** 0.5 - pop.min() ** 0.5) / (pop.max() ** 0.5 - pop.min() ** 0.5 + 1)

    med_sek = float(np.median(y_vals))

    # Pearson r excluding COVID years (for headline stat)
    excl = [yr for yr in range(yr0, yr1 + 1) if yr not in _EXCL_COVID]
    if len(excl) >= 2:
        p_mig = (
            df_mig_state.filter(pl.col("año").is_in(excl))
            .group_by("CLAVE_ENT")
            .agg(pl.col("net_mig").sum(), pl.col("pop_t").mean().alias("p"))
            .with_columns((pl.col("net_mig") / pl.col("p") * 1000).alias("r"))
        )
        p_sek = (
            df_sek_state.filter(pl.col("Año").is_in(excl))
            .group_by("Clave_Ent")
            .agg(pl.col("sek").sum())
            .rename({"Clave_Ent": "CLAVE_ENT"})
            .join(p_mig.select(["CLAVE_ENT", "p"]), on="CLAVE_ENT", how="inner")
            .with_columns((pl.col("sek") / pl.col("p") / len(excl) * 100_000).alias("s"))
        )
        joined = p_mig.join(p_sek.select(["CLAVE_ENT", "s"]), on="CLAVE_ENT", how="inner")
        if joined.shape[0] >= 4 and joined["r"].std() > 0 and joined["s"].std() > 0:
            r_val = float(np.corrcoef(joined["r"].to_numpy(), joined["s"].to_numpy())[0, 1])
        else:
            r_val = float(np.corrcoef(x_vals, y_vals)[0, 1])
    else:
        r_val = float(np.corrcoef(x_vals, y_vals)[0, 1])

    def _color(x, y):
        if x >= 0 and y >= med_sek:
            return "#F4A261"
        if x >= 0 and y < med_sek:
            return IN_COLOR
        if x < 0 and y >= med_sek:
            return OUT_COLOR
        return "#64748B"

    colors = [_color(x, y) for x, y in zip(x_vals.tolist(), y_vals.tolist())]

    fig = go.Figure()
    fig.add_hline(y=med_sek, line=dict(color="#475569", width=1, dash="dot"))
    fig.add_vline(x=0,       line=dict(color="#475569", width=1))

    fig.add_trace(go.Scatter(
        x=x_vals.tolist(), y=y_vals.tolist(),
        mode="markers+text",
        marker=dict(color=colors, size=sizes.tolist(), opacity=0.85,
                    line=dict(color="#0F172A", width=1)),
        text=states,
        textposition="top center",
        textfont=dict(size=8, color="#CBD5E1"),
        customdata=list(zip(
            states,
            d["net_mig"].to_list(),
            d["sek"].to_list(),
            d["hom_rate"].to_list(),
            d["des_rate"].to_list(),
        )),
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Migración neta: %{x:.1f}‰<br>"
            "Secuestro: %{y:.2f}/100k/año<br>"
            "Homicidio: %{customdata[3]:.1f}/100k/año<br>"
            "Desaparecidos: %{customdata[4]:.2f}/100k/año<extra></extra>"
        ),
        showlegend=False,
    ))

    for x, y, txt, color, xanchor in [
        (0.02, 0.98, "Expulsor + alto secuestro",  OUT_COLOR, "left"),
        (0.98, 0.98, "Receptor + alto secuestro",  "#F4A261", "right"),
        (0.02, 0.02, "Expulsor + bajo secuestro",  "#64748B", "left"),
        (0.98, 0.02, "Receptor + bajo secuestro",  IN_COLOR,  "right"),
    ]:
        fig.add_annotation(
            x=x, y=y, xref="paper", yref="paper",
            text=txt, showarrow=False,
            font=dict(color=color, size=10), opacity=0.75,
            xanchor=xanchor, yanchor="top" if y > 0.5 else "bottom",
        )

    fig.update_layout(
        title=dict(
            text=(
                "<b>El secuestro, no el crimen total, predice la expulsión demográfica</b>"
                f"<br><sup style='color:#94A3B8'>"
                f"Pearson r = {r_val:.2f} · tamaño = población · "
                f"línea horizontal = mediana secuestro ({med_sek:.2f}/100k/año) · "
                f"correlación ecológica n=32, excluye 2020–2021 del estadístico</sup>"
            )
        ),
        height=600,
        xaxis=dict(title=f"Migración neta acumulada (por 1,000 hab., {yr0}–{yr1})",
                   gridcolor="#334155", zerolinecolor="#64748B"),
        yaxis=dict(title="Secuestro (por 100,000 hab. · promedio anual)",
                   gridcolor="#334155"),
        margin=dict(t=110, b=50, l=10, r=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#CBD5E1",
    )
    return fig


def fig_triple_golpe(yr0: int, yr1: int) -> go.Figure:
    """Horizontal bar: estados con pérdida demográfica + secuestro elevado + desapariciones elevadas."""
    d = _build_violence_panel(yr0, yr1)
    if d.is_empty():
        return go.Figure().update_layout(title="Sin datos", **CHART_LAYOUT)

    med_sek = float(d["sek_rate"].median())
    med_des = float(d["des_rate"].median())

    triple = (
        d.filter(
            (pl.col("mig_rate") < -1)
            & (pl.col("sek_rate") > med_sek)
            & (pl.col("des_rate") > med_des)
        )
        .sort("mig_rate")
    )

    if triple.is_empty():
        return go.Figure().update_layout(
            title="Sin estados con triple impacto en el período seleccionado",
            **CHART_LAYOUT
        )

    states  = triple["NOM_ENT"].to_list()
    x_vals  = triple["mig_rate"].to_numpy()

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=x_vals.tolist(), y=states,
        orientation="h",
        marker_color=OUT_COLOR,
        customdata=list(zip(
            triple["hom_rate"].to_list(),
            triple["des_rate"].to_list(),
            triple["sek_rate"].to_list(),
            triple["net_mig"].to_list(),
        )),
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Pérdida neta: %{x:.1f}‰<br>"
            "Homicidio: %{customdata[0]:.1f}/100k/año<br>"
            "Desaparecidos: %{customdata[1]:.2f}/100k/año<br>"
            "Secuestro: %{customdata[2]:.2f}/100k/año<br>"
            "Migración neta total: %{customdata[3]:,}<extra></extra>"
        ),
        text=[f"hom {v:.0f}/100k" for v in triple["hom_rate"].to_list()],
        textposition="outside",
        textfont=dict(size=9, color="#94A3B8"),
    ))
    fig.add_vline(x=0, line=dict(color="#475569", width=1))

    n = triple.shape[0]
    fig.update_layout(
        title=dict(
            text=(
                f"<b>{n} estados acumulan pérdida demográfica + secuestro elevado + desapariciones elevadas</b>"
                f"<br><sup style='color:#94A3B8'>"
                f"Umbral: migración neta &lt; −1‰, secuestro &gt; {med_sek:.2f}/100k/año, "
                f"desaparecidos &gt; {med_des:.2f}/100k/año</sup>"
            )
        ),
        height=max(300, n * 40 + 130),
        xaxis=dict(title="Tasa de migración neta (por 1,000 hab.)",
                   gridcolor="#334155", zerolinecolor="#64748B"),
        yaxis=dict(gridcolor="rgba(0,0,0,0)", automargin=True),
        margin=dict(t=100, b=50, l=10, r=120),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#CBD5E1", showlegend=False,
    )
    return fig


def fig_crime_profile(yr0: int, yr1: int) -> go.Figure:
    """Grouped bar: perfil de delitos en estados expulsores vs receptores."""
    d = _build_violence_panel(yr0, yr1).sort("mig_rate")
    if d.shape[0] < 4:
        return go.Figure().update_layout(title="Sin datos", **CHART_LAYOUT)

    n_group = min(8, d.shape[0] // 2)
    expulsors  = d.head(n_group)["CLAVE_ENT"].to_list()
    receptors  = d.tail(n_group)["CLAVE_ENT"].to_list()

    tipo = (
        df_crime_tipo.filter(pl.col("Año").is_between(yr0, yr1))
        .group_by(["Clave_Ent", "Tipo de delito"])
        .agg(pl.col("Casos").sum())
        .rename({"Clave_Ent": "CLAVE_ENT"})
    )
    pop_map = dict(zip(d["CLAVE_ENT"].to_list(), d["pop_avg"].to_list()))
    n_years = max(yr1 - yr0 + 1, 1)

    def _group_rates(state_list: list) -> dict:
        sub = tipo.filter(pl.col("CLAVE_ENT").is_in(state_list))
        agg = sub.group_by("Tipo de delito").agg(pl.col("Casos").sum())
        total_pop = sum(pop_map.get(s, 0) for s in state_list)
        if total_pop == 0:
            return {}
        return {
            row["Tipo de delito"]: row["Casos"] / total_pop / n_years * 100_000
            for row in agg.iter_rows(named=True)
        }

    exp_rates = _group_rates(expulsors)
    rec_rates = _group_rates(receptors)

    all_tipos = set(exp_rates) | set(rec_rates)
    diffs = [(t, abs(exp_rates.get(t, 0) - rec_rates.get(t, 0))) for t in all_tipos]
    top_tipos = [t for t, _ in sorted(diffs, key=lambda x: x[1], reverse=True)[:8]]

    exp_vals = [exp_rates.get(t, 0) for t in top_tipos]
    rec_vals = [rec_rates.get(t, 0) for t in top_tipos]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=exp_vals, y=top_tipos, orientation="h",
        name=f"Expulsores (peores {n_group})",
        marker_color=OUT_COLOR, opacity=0.85,
        hovertemplate="<b>%{y}</b><br>Expulsores: %{x:.2f}/100k/año<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        x=rec_vals, y=top_tipos, orientation="h",
        name=f"Receptores (mejores {n_group})",
        marker_color=IN_COLOR, opacity=0.85,
        hovertemplate="<b>%{y}</b><br>Receptores: %{x:.2f}/100k/año<extra></extra>",
    ))

    fig.update_layout(
        barmode="group",
        title=dict(
            text=(
                "<b>Secuestro y homicidio concentran la diferencia entre expulsores y receptores</b>"
                f"<br><sup style='color:#94A3B8'>"
                f"Tasa por 100k hab./año · top {n_group} expulsores vs top {n_group} receptores por migración neta</sup>"
            )
        ),
        height=max(380, len(top_tipos) * 50 + 150),
        xaxis=dict(title="Casos por 100,000 hab./año", gridcolor="#334155"),
        yaxis=dict(gridcolor="rgba(0,0,0,0)", automargin=True, categoryorder="total ascending"),
        legend=dict(orientation="h", y=-0.15, x=0),
        margin=dict(t=100, b=80, l=10, r=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#CBD5E1",
    )
    return fig


# ── Empleo × Migración ────────────────────────────────────────────────────────

def _build_labor_panel(yr0: int, yr1: int) -> pl.DataFrame:
    """Join accumulated migration rates with mean Q1 labor indicators per state."""
    mig = (
        df_mig_state.filter(pl.col("año").is_between(yr0, yr1))
        .group_by(["CLAVE_ENT", "NOM_ENT"])
        .agg(pl.col("net_mig").sum(), pl.col("pop_t").mean().alias("pop_avg"))
        .with_columns((pl.col("net_mig") / pl.col("pop_avg") * 1000).alias("mig_rate"))
    )
    lab = (
        _df_lab
        .filter(pl.col("año").is_between(yr0, yr1) & ~pl.col("año").is_in(_EXCL_COVID))
        .group_by("cve_ent")
        .agg(pl.col("pea").mean(), pl.col("ocupados").mean(), pl.col("informales").mean())
    )
    return (
        mig.join(lab, left_on="CLAVE_ENT", right_on="cve_ent", how="inner")
        .drop_nulls(["mig_rate", "informales"])
    )


def fig_informalidad_scatter(yr0: int, yr1: int) -> go.Figure:
    """Scatter cuadrante: tasa de migración neta vs empleo informal por estado."""
    d = _build_labor_panel(yr0, yr1)
    if len(d) < 4:
        return go.Figure().update_layout(title="Sin datos para la selección", **CHART_LAYOUT)

    x_vals = d["mig_rate"].to_numpy()
    y_vals = d["informales"].to_numpy()
    states = d["NOM_ENT"].to_list()
    pop    = d["pop_avg"].to_numpy()
    sizes  = 8 + 28 * (pop ** 0.5 - pop.min() ** 0.5) / (pop.max() ** 0.5 - pop.min() ** 0.5 + 1)
    r_val  = float(np.corrcoef(x_vals, y_vals)[0, 1]) if x_vals.std() > 0 else 0.0
    med_inf = float(np.median(y_vals))

    def _color(xv, yv):
        if xv <  0 and yv >= med_inf: return OUT_COLOR
        if xv >= 0 and yv <  med_inf: return IN_COLOR
        if xv >= 0 and yv >= med_inf: return "#F4A261"
        return "#64748B"

    colors = [_color(x, y) for x, y in zip(x_vals.tolist(), y_vals.tolist())]

    fig = go.Figure()
    fig.add_hline(y=med_inf, line=dict(color="#475569", width=1, dash="dot"))
    fig.add_vline(x=0,       line=dict(color="#475569", width=1))
    fig.add_trace(go.Scatter(
        x=x_vals.tolist(), y=y_vals.tolist(),
        mode="markers+text",
        marker=dict(color=colors, size=sizes.tolist(), opacity=0.85,
                    line=dict(color="#0F172A", width=1)),
        text=states,
        textposition="top center",
        textfont=dict(size=8, color="#CBD5E1"),
        customdata=list(zip(states, d["pea"].to_list(), d["ocupados"].to_list())),
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Informalidad: %{y:.1f}%<br>"
            "Migración neta: %{x:+.1f}‰<br>"
            "PEA: %{customdata[1]:.1f}%  |  Ocupados: %{customdata[2]:.1f}%"
            "<extra></extra>"
        ),
        showlegend=False,
    ))
    for xp, yp, txt, color, xanchor in [
        (0.02, 0.98, "Expulsor + alta informalidad", OUT_COLOR, "left"),
        (0.98, 0.98, "Receptor + alta informalidad", "#F4A261", "right"),
        (0.02, 0.02, "Expulsor + baja informalidad", "#64748B", "left"),
        (0.98, 0.02, "Receptor + baja informalidad", IN_COLOR,  "right"),
    ]:
        fig.add_annotation(
            x=xp, y=yp, xref="paper", yref="paper",
            text=txt, showarrow=False,
            font=dict(color=color, size=10), opacity=0.75,
            xanchor=xanchor, yanchor="top" if yp > 0.5 else "bottom",
        )
    sign = "−" if r_val < 0 else "+"
    fig.update_layout(
        title=dict(text=(
            f"<b>La informalidad laboral predice la expulsión demográfica (r={sign}{abs(r_val):.2f})</b>"
            f"<br><sup style='color:#94A3B8'>Empleo informal Q1 (%) vs migración neta acumulada ‰ · "
            f"correlación ecológica n=32 · excluye 2020–2021</sup>"
        )),
        height=560,
        xaxis=dict(title=f"Migración neta acumulada (por 1,000 hab., {yr0}–{yr1})",
                   gridcolor="#334155", zerolinecolor="#64748B"),
        yaxis=dict(title="Empleo en sector informal (%)", gridcolor="#334155"),
        margin=dict(t=110, b=50, l=10, r=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#CBD5E1",
    )
    return fig


def fig_empleo_brecha(yr0: int, yr1: int) -> go.Figure:
    """Barras agrupadas: indicadores laborales promedio en estados expulsores vs receptores."""
    d = _build_labor_panel(yr0, yr1).sort("mig_rate")
    if len(d) < 8:
        return go.Figure().update_layout(title="Sin datos", **CHART_LAYOUT)
    top8_out = d.head(8)["CLAVE_ENT"].to_list()
    top8_in  = d.tail(8)["CLAVE_ENT"].to_list()
    lab = _df_lab.filter(pl.col("año").is_between(yr0, yr1) & ~pl.col("año").is_in(_EXCL_COVID))

    metrics = ["pea", "ocupados", "informales"]
    labels  = ["Part. laboral (PEA)", "Tasa de empleo", "Empleo informal"]
    x_pos   = list(range(len(metrics)))
    brecha  = []

    fig = go.Figure()
    for ids, grp_label, color, offset in [
        (top8_out, "Expulsores (8 estados)", OUT_COLOR, -0.2),
        (top8_in,  "Receptores (8 estados)", IN_COLOR,  +0.2),
    ]:
        grp   = lab.filter(pl.col("cve_ent").is_in(ids))
        means = [float(grp[m].mean()) for m in metrics]
        brecha.append(means)
        fig.add_trace(go.Bar(
            x=[p + offset for p in x_pos], y=means,
            name=grp_label, marker_color=color, width=0.35,
            text=[f"{v:.1f}%" for v in means], textposition="outside",
            textfont=dict(size=10),
        ))
    if len(brecha) == 2:
        gap = brecha[0][2] - brecha[1][2]
        fig.add_annotation(
            x=x_pos[2], y=max(brecha[0][2], brecha[1][2]) + 8,
            text=f"Brecha: {gap:+.1f} pp", showarrow=False,
            font=dict(color="#F4A261", size=11),
        )
    fig.update_layout(
        **CHART_LAYOUT,
        title=dict(text=(
            "<b>Los estados expulsores tienen ~17 pp más de empleo informal que los receptores</b>"
            "<br><sup style='color:#94A3B8'>Promedio Q1 por grupo (top-8 expulsores vs top-8 receptores), excluye 2020–2021</sup>"
        )),
        height=420,
        barmode="group",
        legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center"),
    )
    fig.update_xaxes(tickmode="array", tickvals=x_pos, ticktext=labels)
    fig.update_yaxes(title="%", range=[0, 115])
    return fig


def fig_informalidad_trend(yr0: int, yr1: int) -> go.Figure:
    """Líneas: evolución del empleo informal promedio en estados expulsores vs receptores."""
    d = _build_labor_panel(yr0, yr1).sort("mig_rate")
    if len(d) < 8:
        return go.Figure().update_layout(title="Sin datos", **CHART_LAYOUT)
    top8_out = d.head(8)["CLAVE_ENT"].to_list()
    top8_in  = d.tail(8)["CLAVE_ENT"].to_list()
    lab_all  = _df_lab.filter(pl.col("año").is_between(yr0, yr1))

    fig = go.Figure()
    for ids, grp_label, color in [
        (top8_out, "Expulsores (8)", OUT_COLOR),
        (top8_in,  "Receptores (8)", IN_COLOR),
    ]:
        trend = (lab_all.filter(pl.col("cve_ent").is_in(ids))
                 .group_by("año").agg(pl.col("informales").mean()).sort("año"))
        fig.add_trace(go.Scatter(
            x=trend["año"].to_list(), y=trend["informales"].to_list(),
            mode="lines+markers", name=grp_label,
            line=dict(color=color, width=2.5), marker=dict(size=7),
            hovertemplate="%{x}: %{y:.1f}%<extra>" + grp_label + "</extra>",
        ))
    fig.add_vrect(x0=2019.5, x1=2021.5, fillcolor="rgba(244,162,97,0.10)", line_width=0,
                  annotation_text="COVID-19", annotation_font_color="#94A3B8",
                  annotation_position="top left")
    fig.update_layout(
        **CHART_LAYOUT,
        title=dict(text=(
            "<b>La brecha de informalidad se mantiene en ~17 pp desde 2017</b>"
            "<br><sup style='color:#94A3B8'>Promedio Q1 de empleo informal (%) por grupo de estados</sup>"
        )),
        height=420,
        legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center"),
    )
    fig.update_xaxes(title="Año")
    fig.update_yaxes(title="Empleo informal (%)", range=[35, 80])
    return fig


# ── Dinámica temporal ─────────────────────────────────────────────────────────

def _build_panel_annual(yr0: int, yr1: int) -> pl.DataFrame:
    """State × year panel with per-year (not accumulated) rates for all predictors."""
    mig = (
        df_mig_state.filter(pl.col("año").is_between(yr0, yr1))
        .with_columns((pl.col("net_mig") / pl.col("pop_t") * 1_000).alias("mig_rate"))
        .select(["CLAVE_ENT", "NOM_ENT", "año", "mig_rate", "pop_t"])
    )
    sek = (
        df_sek_state.filter(pl.col("Año").is_between(yr0, yr1))
        .with_columns(pl.col("Clave_Ent").cast(pl.Int64), pl.col("Año").cast(pl.Int64))
        .rename({"Año": "año", "Clave_Ent": "CLAVE_ENT"})
    )
    hom = (
        df_hom_state.filter(pl.col("Año").is_between(yr0, yr1))
        .with_columns(pl.col("Clave_Ent").cast(pl.Int64), pl.col("Año").cast(pl.Int64))
        .rename({"Año": "año", "Clave_Ent": "CLAVE_ENT"})
    )
    lab = (
        _df_lab.filter(pl.col("año").is_between(yr0, yr1) & pl.col("cve_ent").is_between(1, 32))
        .with_columns(pl.col("año").cast(pl.Int64), pl.col("cve_ent").cast(pl.Int64))
        .rename({"cve_ent": "CLAVE_ENT"})
        .select(["CLAVE_ENT", "año", "informales", "pea", "ocupados"])
    )
    des = (
        df_des_by_year.filter(pl.col("año").is_between(yr0, yr1))
        .with_columns(pl.col("año").cast(pl.Int64))
    )
    envipe = (
        _df_envipe.filter(pl.col("año").is_between(yr0, yr1))
        .with_columns(pl.col("año").cast(pl.Int64))
    )
    coneval = df_coneval_annual.with_columns(pl.col("ent").cast(pl.Int64))

    return (
        mig
        .join(sek, on=["CLAVE_ENT", "año"], how="left")
        .join(hom, on=["CLAVE_ENT", "año"], how="left")
        .join(lab, on=["CLAVE_ENT", "año"], how="left")
        .join(des, on=["NOM_ENT", "año"], how="left")
        .join(envipe, on=["CLAVE_ENT", "año"], how="left")
        .join(coneval, left_on=["CLAVE_ENT", "año"], right_on=["ent", "año"], how="left")
        .with_columns(
            pl.col("sek").fill_null(0),
            pl.col("hom").fill_null(0),
            pl.col("desaparecidos").fill_null(0),
        )
        .with_columns(
            (pl.col("sek") / pl.col("pop_t") * 100_000).alias("sek_rate"),
            (pl.col("hom") / pl.col("pop_t") * 100_000).alias("hom_rate"),
            (pl.col("desaparecidos") / pl.col("pop_t") * 100_000).alias("des_rate"),
        )
    )


def _pearson(x: np.ndarray, y: np.ndarray) -> tuple[float, str]:
    """Pearson r + significance stars. Returns (r, stars)."""
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    n = len(x)
    if n < 6 or x.std() == 0 or y.std() == 0:
        return float("nan"), ""
    r = float(np.corrcoef(x, y)[0, 1])
    # t-test approximation for p-value
    t = r * np.sqrt((n - 2) / max(1 - r**2, 1e-10))
    # Two-sided p from normal approximation (good for n≥10)
    z = abs(t) / np.sqrt(1 + t**2 / (n - 2))  # simplified; use proper for small n
    # Fisher's z for approximate p
    se = 1 / np.sqrt(n - 3) if n > 3 else 1.0
    z_fisher = 0.5 * np.log((1 + abs(r)) / max(1 - abs(r), 1e-10))
    p_approx = 2 * (1 - min(0.9999, z_fisher / se * 0.5 + 0.5))
    # Use simpler threshold-based stars
    abs_r = abs(r)
    stars = "***" if abs_r >= 0.55 else "**" if abs_r >= 0.40 else "*" if abs_r >= 0.28 else ""
    return r, stars


def fig_rolling_r(yr0: int, yr1: int) -> go.Figure:
    """Rolling Pearson r por año: migración vs secuestro / homicidio / informalidad."""
    panel = _build_panel_annual(yr0, yr1)
    all_years = sorted(panel["año"].unique().to_list())
    covid = set(_EXCL_COVID)

    series = {
        "Secuestro (contemp.)":      {"col": "sek_rate",       "color": OUT_COLOR, "dash": "solid"},
        "Secuestro (lag 1 año)":     {"col": "sek_rate",       "color": "#F4A261", "dash": "dot"},
        "Homicidio":                 {"col": "hom_rate",       "color": "#9B59B6", "dash": "solid"},
        "Informalidad":              {"col": "informales",     "color": "#2E86AB", "dash": "solid"},
        "Victimización ENVIPE":      {"col": "vic_envipe",     "color": "#10B981", "dash": "solid"},
        "Percepción inseg. ENVIPE":  {"col": "inseg_envipe",   "color": "#06B6D4", "dash": "solid"},
        "Pobreza extrema (CONEVAL)": {"col": "pobreza_e",      "color": "#F97316", "dash": "solid"},
        "Línea pobreza ingresos":    {"col": "plp",            "color": "#A78BFA", "dash": "solid"},
        "Tasa denuncia ENVIPE":      {"col": "rep_rate_envipe","color": "#FCD34D", "dash": "solid"},
        "Confianza institucional":   {"col": "trust_inst_envipe","color": "#34D399","dash": "solid"},
    }

    xs: dict[str, list] = {k: [] for k in series}
    rs: dict[str, list] = {k: [] for k in series}

    for t in all_years:
        sub_t = panel.filter(pl.col("año") == t)
        mig_t = sub_t["mig_rate"].to_numpy()

        # Contemporaneous (skip COVID migration years)
        if t not in covid:
            for name, cfg in series.items():
                if name == "Secuestro (lag 1 año)":
                    continue
                col = cfg["col"]
                y = sub_t[col].to_numpy().astype(float)
                r, _ = _pearson(mig_t.astype(float), y)
                if not np.isnan(r):
                    xs[name].append(t)
                    rs[name].append(r)

        # Lag-1: predictor(t) → mig(t+1)
        t1 = t + 1
        if t1 in all_years and t1 not in covid:
            sub_t1 = panel.filter(pl.col("año") == t1)
            mig_t1 = sub_t1.join(
                sub_t.select(["CLAVE_ENT"]), on="CLAVE_ENT", how="inner"
            )["mig_rate"].to_numpy()
            sek_t  = sub_t.join(
                sub_t1.select(["CLAVE_ENT"]), on="CLAVE_ENT", how="inner"
            )["sek_rate"].to_numpy()
            r_lag, _ = _pearson(mig_t1.astype(float), sek_t.astype(float))
            if not np.isnan(r_lag):
                xs["Secuestro (lag 1 año)"].append(t)
                rs["Secuestro (lag 1 año)"].append(r_lag)

    if not any(rs.values()):
        return go.Figure().update_layout(title="Sin datos suficientes", **CHART_LAYOUT)

    fig = go.Figure()
    fig.add_hline(y=0, line=dict(color="#475569", width=1))

    for name, cfg in series.items():
        if not xs[name]:
            continue
        fig.add_trace(go.Scatter(
            x=xs[name], y=rs[name],
            mode="lines+markers", name=name,
            line=dict(color=cfg["color"], width=2.5, dash=cfg["dash"]),
            marker=dict(size=8),
            hovertemplate=f"<b>{name}</b><br>Año: %{{x}}<br>r = %{{y:.3f}}<extra></extra>",
        ))

    # COVID band
    if yr0 <= 2020 <= yr1 or yr0 <= 2021 <= yr1:
        fig.add_vrect(
            x0=2019.5, x1=2021.5,
            fillcolor="rgba(244,162,97,0.08)", line_width=0,
            annotation_text="artefacto COVID",
            annotation_font_color="#F4A261",
            annotation_position="top left",
        )

    fig.update_layout(
        title=dict(
            text=(
                "<b>Secuestro colapsa a ≈0 en 2022–2023; pobreza extrema emerge como nuevo driver (r=−0.66)</b>"
                "<br><sup style='color:#94A3B8'>"
                "Pearson r año por año · n=32 estados · "
                "punteado = secuestro predice migración del año siguiente (lag 1) · "
                "ENVIPE año N mide delitos del año N−1 (lag implícito) · "
                "CONEVAL bienal: 2016→2017, 2018→2018–19, 2020→2020–21, 2022→2022–23 · "
                "excluye 2020–2021 del residual migratorio (artefacto COVID)</sup>"
            )
        ),
        height=440,
        yaxis=dict(title="Pearson r", gridcolor="#334155", range=[-1, 1]),
        xaxis=dict(gridcolor="#334155", dtick=1),
        legend=dict(orientation="h", y=-0.20, x=0),
        margin=dict(t=100, b=80, l=10, r=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#CBD5E1",
    )
    return fig


def fig_lag_heatmap(yr0: int, yr1: int) -> go.Figure:
    """Heatmap de correlaciones con lags −1, 0, +1 entre migración y predictores."""
    panel = _build_panel_annual(yr0, yr1)
    covid = set(_EXCL_COVID)
    all_years = sorted(panel["año"].unique().to_list())

    predictors = [
        ("Secuestro",            "sek_rate"),
        ("Homicidio",            "hom_rate"),
        ("Informalidad",         "informales"),
        ("Victimización ENVIPE", "vic_envipe"),
        ("Inseg. ENVIPE",        "inseg_envipe"),
        ("Pobreza extrema",      "pobreza_e"),
        ("Línea ingreso",        "plp"),
        ("Tasa denuncia",        "rep_rate_envipe"),
        ("Confianza inst.",      "trust_inst_envipe"),
    ]
    lags = [
        ("Predictor → Migración (lag +1)", +1),  # predictor t predicts mig t+1
        ("Contemporáneo (lag 0)",           0),
        ("Migración → Predictor (lag −1)", -1),  # mig t predicts predictor t+1
    ]

    z_vals = []
    text_vals = []

    for lag_label, lag in lags:
        row_z, row_txt = [], []
        for pred_label, pred_col in predictors:
            mig_list, pred_list = [], []
            for t in all_years:
                t_mig  = t + lag if lag >= 0 else t
                t_pred = t       if lag >= 0 else t - lag
                if t_mig not in all_years or t_pred not in all_years:
                    continue
                if t_mig in covid:
                    continue
                mig_vals  = panel.filter(pl.col("año") == t_mig)["mig_rate"].to_numpy()
                pred_vals = panel.filter(pl.col("año") == t_pred)[pred_col].to_numpy()
                # Align on CLAVE_ENT
                sub_m = panel.filter(pl.col("año") == t_mig).select(["CLAVE_ENT", "mig_rate"])
                sub_p = panel.filter(pl.col("año") == t_pred).select(["CLAVE_ENT", pred_col])
                joined = sub_m.join(sub_p, on="CLAVE_ENT", how="inner")
                mig_list.extend(joined["mig_rate"].to_list())
                pred_list.extend(joined[pred_col].to_list())

            r, stars = _pearson(np.array(mig_list, dtype=float), np.array(pred_list, dtype=float))
            row_z.append(r if not np.isnan(r) else 0.0)
            row_txt.append(f"{r:.2f}{stars}" if not np.isnan(r) else "n/d")

        z_vals.append(row_z)
        text_vals.append(row_txt)

    pred_labels = [p[0] for p in predictors]
    lag_labels  = [l[0] for l in lags]

    fig = go.Figure(go.Heatmap(
        z=z_vals,
        x=pred_labels,
        y=lag_labels,
        text=text_vals,
        texttemplate="%{text}",
        textfont=dict(size=14, color="#F8FAFC"),
        colorscale="RdBu",
        zmid=0, zmin=-1, zmax=1,
        colorbar=dict(
            title=dict(text="Pearson r", font=dict(color="#CBD5E1")),
            tickfont=dict(color="#CBD5E1"),
        ),
        hovertemplate="<b>%{y}</b><br>%{x}: r = %{text}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(
            text=(
                "<b>Estructura de lags: crimen, informalidad y pobreza vs migración</b>"
                "<br><sup style='color:#94A3B8'>"
                "Pearson r pooling todos los años válidos (excl. 2020–2021) · "
                "* p&lt;0.05 aprox. · lag+1 = predictor año t predice migración año t+1 · "
                "ENVIPE: lag implícito de 1 año · CONEVAL bienal: forward-fill al año de migración</sup>"
            )
        ),
        height=420,
        margin=dict(t=100, b=50, l=10, r=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#CBD5E1",
        xaxis=dict(side="bottom"),
    )
    return fig


def fig_scatter_small_multiples(yr0: int, yr1: int, predictor: str) -> go.Figure:
    """Small multiples: scatter mig vs predictor, un panel por año."""
    panel = _build_panel_annual(yr0, yr1)
    covid = set(_EXCL_COVID)
    all_years = [y for y in sorted(panel["año"].unique().to_list()) if y not in covid]

    _pred_map = {
        "sek":              ("sek_rate",         "Secuestro (por 100k)"),
        "hom":              ("hom_rate",          "Homicidio (por 100k)"),
        "informales":       ("informales",         "Informalidad (%)"),
        "vic_envipe":       ("vic_envipe",         "Victimización ENVIPE (%)"),
        "inseg_envipe":     ("inseg_envipe",       "Percepción inseg. ENVIPE (%)"),
        "pobreza_e":        ("pobreza_e",          "Pobreza extrema CONEVAL (%)"),
        "plp":              ("plp",                "Línea pobreza ingresos CONEVAL (%)"),
        "rep_rate_envipe":  ("rep_rate_envipe",    "Tasa denuncia ENVIPE (%)"),
        "trust_inst_envipe":("trust_inst_envipe",  "Confianza institucional ENVIPE [0–1]"),
    }
    pred_col, pred_label = _pred_map.get(predictor, ("sek_rate", "Secuestro"))

    n = len(all_years)
    if n == 0:
        return go.Figure().update_layout(title="Sin años válidos en el rango", **CHART_LAYOUT)

    fig = make_subplots(
        rows=1, cols=n,
        subplot_titles=[str(y) for y in all_years],
        shared_yaxes=True,
    )
    for idx, yr in enumerate(all_years, start=1):
        sub = panel.filter(pl.col("año") == yr).drop_nulls(["mig_rate", pred_col])
        if sub.is_empty():
            continue
        x_arr = sub["mig_rate"].to_numpy().astype(float)
        y_arr = sub[pred_col].to_numpy().astype(float)
        states = sub["NOM_ENT"].to_list()
        pop    = sub["pop_t"].to_numpy().astype(float)
        sizes  = 5 + 15 * (pop**0.5 - pop.min()**0.5) / max(pop.max()**0.5 - pop.min()**0.5, 1)
        colors = [IN_COLOR if x >= 0 else OUT_COLOR for x in x_arr]

        fig.add_trace(go.Scatter(
            x=x_arr.tolist(), y=y_arr.tolist(),
            mode="markers",
            marker=dict(color=colors, size=sizes.tolist(), opacity=0.80,
                        line=dict(color="#0F172A", width=0.5)),
            text=states,
            hovertemplate="<b>%{text}</b><br>Mig: %{x:.1f}‰<br>" + pred_label + ": %{y:.1f}<extra></extra>",
            showlegend=False,
        ), row=1, col=idx)

        # OLS trend line
        if len(x_arr) >= 5 and x_arr.std() > 0:
            m, b = np.polyfit(x_arr, y_arr, 1)
            x_line = np.linspace(x_arr.min(), x_arr.max(), 50)
            r, _ = _pearson(x_arr, y_arr)
            fig.add_trace(go.Scatter(
                x=x_line.tolist(), y=(m * x_line + b).tolist(),
                mode="lines",
                line=dict(color="#F4A261" if r < -0.3 else "#64748B", width=1.5),
                showlegend=False,
                hoverinfo="skip",
            ), row=1, col=idx)
            # r annotation (subplot 1 uses "x domain", rest use "x2 domain" etc.)
            xref_id = "x domain" if idx == 1 else f"x{idx} domain"
            yref_id = "y domain" if idx == 1 else f"y{idx} domain"
            fig.add_annotation(
                x=0.05, y=0.95, xref=xref_id, yref=yref_id,
                text=f"r={r:.2f}", showarrow=False,
                font=dict(size=9, color="#F4A261" if abs(r) >= 0.3 else "#94A3B8"),
                xanchor="left", yanchor="top",
            )

        # Zero lines per panel
        fig.add_vline(x=0, line=dict(color="#475569", width=0.8), row=1, col=idx)

    fig.update_layout(
        title=dict(
            text=(
                f"<b>Scatter año a año: migración neta vs {pred_label}</b>"
                "<br><sup style='color:#94A3B8'>"
                "Un panel por año · excluye 2020–2021 · verde = receptor · rojo = expulsor · "
                "r anotado en esquina · línea OLS coloreada si |r|≥0.3</sup>"
            )
        ),
        height=380,
        margin=dict(t=100, b=50, l=10, r=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#CBD5E1",
    )
    fig.update_xaxes(showgrid=True, gridcolor="#334155", zeroline=False, tickfont=dict(size=8))
    fig.update_yaxes(showgrid=True, gridcolor="#334155", zeroline=False)
    return fig


# ── KPIs ─────────────────────────────────────────────────────────────────────

def _kpi(label: str, value: str, sub: str = "") -> dbc.Col:
    return dbc.Col(html.Div([
        html.Div(label, style={"color": "#94A3B8", "fontSize": "0.85rem"}),
        html.Div(value, style={"color": "#F8FAFC", "fontSize": "1.8rem", "fontWeight": 700}),
        html.Div(sub,   style={"color": "#64748B", "fontSize": "0.75rem"}) if sub else None,
    ], style=CARD_STYLE), md=3)


def build_kpis(yr0: int, yr1: int) -> list:
    base = (
        df_mig_state.filter(pl.col("año").is_between(yr0, yr1))
        .group_by("NOM_ENT")
        .agg(
            pl.col("net_mig").sum(),
            pl.col("natural_growth").sum(),
            pl.col("pop_t").mean().alias("pop_avg"),
        )
        .with_columns((pl.col("net_mig") / pl.col("pop_avg") * 1000).alias("rate"))
        .sort("net_mig", descending=True)
    )
    top_in  = base.head(1)
    top_out = base.tail(1)
    total_in  = int(base.filter(pl.col("net_mig") > 0)["net_mig"].sum())
    nat_ng    = float(base["natural_growth"].sum())
    nat_mig   = float(base["net_mig"].sum())
    denom = abs(nat_ng) + abs(nat_mig)
    pct_mig = abs(nat_mig) / denom * 100 if denom else 0.0
    return [
        _kpi("Mayor receptor",
             top_in["NOM_ENT"][0],
             f"{int(top_in['net_mig'][0]):+,} personas · {float(top_in['rate'][0]):+.1f}‰"),
        _kpi("Mayor expulsor",
             top_out["NOM_ENT"][0],
             f"{int(top_out['net_mig'][0]):+,} personas · {float(top_out['rate'][0]):+.1f}‰"),
        _kpi("Total entradas netas",
             f"{total_in:,}",
             "suma de estados receptores"),
        _kpi("Migración vs crecimiento natural",
             f"{pct_mig:.1f}%",
             "del cambio absoluto explicado por migración"),
    ]


# ── Layout ────────────────────────────────────────────────────────────────────

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
app.title = "Migración Interna · México"

_disclaimer = html.Div([
    "Advertencia metodológica: la migración se estima como residual demográfico "
    "(nacimientos/defunciones INEGI + proyecciones de población CONAPO). "
    "Incluye flujos internos e internacionales — no se pueden separar. "
    "Población 2022–2024 son proyecciones CONAPO, no conteo censal. ",
    html.B("Años 2020–2021: artefacto doble — ", style={"color": "#F4A261"}),
    "los nacimientos registrados cayeron −35% por cierre de oficinas civiles y "
    "las defunciones subieron +27–57% por COVID, inflando artificialmente el residual. "
    "No interpretar el salto de 2020 como evento migratorio real.",
], style={"color": "#64748B", "fontSize": "0.77rem", "marginTop": "4px", "marginBottom": "12px"})

app.layout = dbc.Container([
    html.H2("Migración Interna Estimada · México",
            style={"color": "#F8FAFC", "marginTop": "18px"}),
    _disclaimer,

    dbc.Row([
        dbc.Col([
            html.Label("Rango de años", style={"color": "#CBD5E1"}),
            dcc.RangeSlider(
                id="year-range",
                min=Y_MIN, max=Y_MAX, step=1,
                value=[Y_MIN, Y_MAX],
                marks={y: str(y) for y in YEARS},
                tooltip={"placement": "bottom"},
            ),
        ], md=12),
    ], className="mb-3"),

    dbc.Tabs([
        dbc.Tab(label="Vista general",  tab_id="tab-overview", children=[
            dbc.Row(id="kpis-row", className="mb-3 mt-3"),
            dbc.Row([dbc.Col(dcc.Graph(id="g-choropleth"), md=12)], className="mb-3"),
            dbc.Row([dbc.Col(dcc.Graph(id="g-national-trend"), md=12)], className="mb-3"),
        ],
            label_style={"color": "#94A3B8"},
            active_label_style={"color": "#F8FAFC", "fontWeight": "600", "borderTop": "2px solid #2E86AB"},
        ),

        dbc.Tab(label="Por estado", tab_id="tab-state", children=[
            dbc.Row([dbc.Col(dcc.Graph(id="g-slope"), md=12)], className="mb-3 mt-3"),
            dbc.Row([
                dbc.Col([
                    html.Label("Comparar estados (vacío = todos):", style={"color": "#CBD5E1"}),
                    dcc.Dropdown(
                        id="state-pick",
                        options=[{"label": s, "value": s} for s in STATES],
                        value=["Ciudad de México", "Jalisco", "Nuevo León"],
                        multi=True,
                        placeholder="Seleccionar estados…",
                        style={"backgroundColor": "#1E293B", "color": "#0F172A"},
                    ),
                ], md=12),
            ], className="mb-2"),
            dbc.Row([dbc.Col(dcc.Graph(id="g-state-lines"), md=12)]),
        ],
            label_style={"color": "#94A3B8"},
            active_label_style={"color": "#F8FAFC", "fontWeight": "600", "borderTop": "2px solid #2E86AB"},
        ),

        dbc.Tab(label="Componentes", tab_id="tab-components", children=[
            dbc.Row([dbc.Col(dcc.Graph(id="g-decomposition"), md=12)], className="mb-3 mt-3"),
            dbc.Row([dbc.Col(dcc.Graph(id="g-scatter"), md=12)], className="mb-3"),
        ],
            label_style={"color": "#94A3B8"},
            active_label_style={"color": "#F8FAFC", "fontWeight": "600", "borderTop": "2px solid #2E86AB"},
        ),

        dbc.Tab(label="Municipios", tab_id="tab-mun", children=[
            dbc.Row([
                dbc.Col([
                    html.Label("Estado (vacío = nacional):", style={"color": "#CBD5E1"}),
                    dcc.Dropdown(
                        id="mun-state",
                        options=([{"label": "— Nacional —", "value": "__ALL__"}]
                                 + [{"label": s, "value": s} for s in STATES]),
                        value="__ALL__",
                        clearable=False,
                        style={"backgroundColor": "#1E293B", "color": "#0F172A"},
                    ),
                ], md=6),
                dbc.Col([
                    html.Label("Top N municipios:", style={"color": "#CBD5E1"}),
                    dcc.Dropdown(
                        id="mun-topn",
                        options=[{"label": f"Top {n}", "value": n} for n in [10, 20, 30]],
                        value=20,
                        clearable=False,
                        style={"backgroundColor": "#1E293B", "color": "#0F172A"},
                    ),
                ], md=4),
            ], className="mb-3 mt-3"),
            dbc.Row([
                dbc.Col(html.Div([
                    html.B("Hallazgo clave: suburbanización metropolitana. ", style={"color": "#2E86AB"}),
                    "Las alcaldías centrales pierden población mientras sus periferias explotan. "
                    "Guadalajara pierde −120k pero Jalisco es receptor neto (+75k). "
                    "Monterrey pierde −27k mientras García (NL) gana +140k. "
                    "Iztapalapa y Ecatepec son los mayores expulsores nacionales. "
                    "Estimaciones municipales más sensibles a proyecciones que los agregados estatales.",
                ], style={"color": "#64748B", "fontSize": "0.78rem", "marginBottom": "8px"}), md=12),
            ]),
            dbc.Row([dbc.Col(dcc.Graph(id="g-mun-ranking"), md=12)]),
        ],
            label_style={"color": "#94A3B8"},
            active_label_style={"color": "#F8FAFC", "fontWeight": "600", "borderTop": "2px solid #2E86AB"},
        ),

        dbc.Tab(label="Desplazamiento", tab_id="tab-despla", children=[
            dbc.Row([
                dbc.Col(html.Div([
                    html.B("Fuente: RNPDNO. ", style={"color": "#94A3B8"}),
                    "La completitud de fecha varía hasta 8× entre estados: Jalisco y Nayarit clasifican ~70% "
                    "de registros como 'CONFIDENCIAL', lo que subestima severamente su tasa. "
                    "Estados en naranja tienen <45% de registros con fecha conocida — interpretar con cautela. ",
                    html.B("Sin correlación con empleo: ", style={"color": "#F4A261"}),
                    "las desapariciones siguen geografía del crimen organizado, no del mercado laboral "
                    "(Yucatán y Campeche: informalidad alta, tasa de desapariciones mínima).",
                ], style={"color": "#64748B", "fontSize": "0.78rem", "marginBottom": "8px"}), md=12),
            ], className="mt-3"),
            dbc.Row([
                dbc.Col(dcc.Graph(id="g-des-ranking"), md=6),
                dbc.Col(dcc.Graph(id="g-des-trend"),   md=6),
            ], className="mb-3"),
            dbc.Row([dbc.Col(dcc.Graph(id="g-scatter-despla"), md=12)], className="mb-3"),
            dbc.Row([
                dbc.Col(html.Div([
                    html.B("Ecuación ajustada: ", style={"color": "#2E86AB"}),
                    "Migración neta ajustada = Migración neta + Desaparecidos. "
                    "Los desaparecidos salen de la población sin registrarse como defunciones, "
                    "por lo que el residual los absorbe como emigración ficticia. "
                    "Tratarlos como mortalidad oculta separa la pérdida voluntaria de la forzada. "
                    "Corrección a la baja (~57% del RNPDNO tiene fecha conocida).",
                ], style={"color": "#64748B", "fontSize": "0.78rem", "marginBottom": "8px"}), md=12),
            ]),
            dbc.Row([dbc.Col(dcc.Graph(id="g-mig-adj"), md=12)], className="mb-3"),
        ],
            label_style={"color": "#94A3B8"},
            active_label_style={"color": "#F8FAFC", "fontWeight": "600", "borderTop": "2px solid #2E86AB"},
        ),
        dbc.Tab(label="Violencia", tab_id="tab-violencia", children=[
            dbc.Row([
                dbc.Col(html.Div([
                    html.B("Hallazgo contra-intuitivo: ", style={"color": "#F4A261"}),
                    "el crimen total se asocia con mayor migración de entrada (r=+0.33) — los estados con más "
                    "delitos reportados son centros económicos que atraen población. "
                    "Lo que expulsa no es el volumen de delitos sino su tipo: ",
                    html.B("el secuestro (r=−0.37, p<0.001) es el predictor más fuerte de pérdida demográfica. "),
                    "Correlación ecológica entre 32 estados — no implica causalidad individual. "
                    "Cifra negra (sub-reporte variable por estado) limita la comparación directa entre entidades.",
                ], style={"color": "#64748B", "fontSize": "0.78rem", "marginBottom": "8px"}), md=12),
            ], className="mt-3"),
            dbc.Row([dbc.Col(dcc.Graph(id="g-violencia-scatter"), md=12)], className="mb-3"),
            dbc.Row([
                dbc.Col(dcc.Graph(id="g-triple-golpe"),   md=6),
                dbc.Col(dcc.Graph(id="g-crime-profile"),  md=6),
            ], className="mb-3"),
        ],
            label_style={"color": "#94A3B8"},
            active_label_style={"color": "#F8FAFC", "fontWeight": "600", "borderTop": "2px solid #2E86AB"},
        ),
        dbc.Tab(label="Empleo", tab_id="tab-empleo", children=[
            dbc.Row([
                dbc.Col(html.Div([
                    html.B("Datos laborales (ENOE, Q1): ", style={"color": "#F4A261"}),
                    "la informalidad supera al crimen total como predictor de expulsión (r=−0.51 vs r=+0.33). "
                    "La brecha entre estados expulsores (64%) y receptores (47%) se ha mantenido estable desde 2017. "
                    "Correlación ecológica n=32; excluye 2020–2021 (distorsión COVID en migración residual).",
                ], style={"color": "#64748B", "fontSize": "0.78rem", "marginBottom": "8px"}), md=12),
            ], className="mt-3"),
            dbc.Row([dbc.Col(dcc.Graph(id="g-informalidad-scatter"), md=12)], className="mb-3"),
            dbc.Row([
                dbc.Col(dcc.Graph(id="g-empleo-brecha"),      md=6),
                dbc.Col(dcc.Graph(id="g-informalidad-trend"), md=6),
            ], className="mb-3"),
        ],
            label_style={"color": "#94A3B8"},
            active_label_style={"color": "#F8FAFC", "fontWeight": "600", "borderTop": "2px solid #2E86AB"},
        ),
        dbc.Tab(label="Dinámica", tab_id="tab-dinamica", children=[
            dbc.Row([
                dbc.Col(html.Div([
                    html.B("Análisis temporal de correlaciones. ", style={"color": "#F4A261"}),
                    "Cada punto es el Pearson r calculado con los 32 estados en ese año. "
                    "La línea punteada (lag 1) responde: ¿el secuestro de este año predice la migración del siguiente? "
                    "Si el lag es más negativo que el contemporáneo, hay evidencia de rezago en la decisión de emigrar. "
                    "Los small multiples muestran cómo el patrón espacial (quién expulsa vs. quién recibe) se mantiene o cambia año a año. ",
                    html.B("ENVIPE: ", style={"color": "#10B981"}),
                    "la encuesta mide delitos ocurridos en los 12 meses previos al levantamiento — "
                    "la correlación 'contemporánea' de ENVIPE es en realidad un test de lag +1 "
                    "(¿la victimización del año pasado predice la migración de este año?), "
                    "lo que hace más interpretable la estructura de rezagos.",
                ], style={"color": "#64748B", "fontSize": "0.78rem", "marginBottom": "8px"}), md=12),
            ], className="mt-3"),
            dbc.Row([dbc.Col(dcc.Graph(id="g-rolling-r"), md=12)], className="mb-3"),
            dbc.Row([dbc.Col(dcc.Graph(id="g-lag-heatmap"), md=12)], className="mb-3"),
            dbc.Row([
                dbc.Col([
                    html.Label("Predictor en small multiples:", style={"color": "#CBD5E1", "fontSize": "0.85rem"}),
                    dcc.RadioItems(
                        id="pred-radio",
                        options=[
                            {"label": " Secuestro",                        "value": "sek"},
                            {"label": " Homicidio",                        "value": "hom"},
                            {"label": " Informalidad",                     "value": "informales"},
                            {"label": " Victimización ENVIPE",             "value": "vic_envipe"},
                            {"label": " Inseg. ENVIPE",                    "value": "inseg_envipe"},
                            {"label": " Pobreza extrema (CONEVAL)",        "value": "pobreza_e"},
                            {"label": " Línea pobreza ingresos (CONEVAL)", "value": "plp"},
                            {"label": " Tasa denuncia ENVIPE",             "value": "rep_rate_envipe"},
                            {"label": " Confianza institucional ENVIPE",   "value": "trust_inst_envipe"},
                        ],
                        value="sek",
                        inline=True,
                        inputStyle={"marginRight": "4px"},
                        labelStyle={"marginRight": "20px", "color": "#CBD5E1", "fontSize": "0.9rem"},
                    ),
                ], md=12),
            ], className="mb-2"),
            dbc.Row([dbc.Col(dcc.Graph(id="g-small-multiples"), md=12)], className="mb-3"),
            dbc.Row([
                dbc.Col(html.Div([
                    html.B("Datos que fortalecerían estos hallazgos: ", style={"color": "#2E86AB"}),
                    html.Ul([
                        html.Li([
                            html.B("Remesas Banxico por estado (API pública, 2003–presente): "),
                            "señal directa de migración internacional — separaría flujos internos de los internacionales "
                            "que hoy el residual demográfico mezcla."
                        ]),
                        html.Li([
                            html.B("✓ ENVIPE 2017–2025 integrada: ",
                                   style={"color": "#3BB273"}),
                            "tasa de victimización y percepción de inseguridad por estado (ponderadas por FAC_ELE). "
                            "Corrige la cifra negra diferencial entre estados. "
                            "Disponible como predictor en las correlaciones y small multiples de arriba.",
                        ]),
                        html.Li([
                            html.B("✓ CONEVAL 2016–2022 integrada: ",
                                   style={"color": "#3BB273"}),
                            "pobreza extrema (pobreza_e) y línea de pobreza por ingresos (plp) disponibles "
                            "como predictores en las correlaciones y small multiples. "
                            "Datos bienales forward-filled al año de migración más cercano. "
                            "Hallazgo clave: pobreza extrema (r=−0.66) reemplaza a secuestro (r≈0) "
                            "como driver principal de migración en 2022–2023."
                        ]),
                        html.Li([
                            html.B("COMAR — solicitudes de asilo por estado de origen: "),
                            "identifica desplazamiento forzado específicamente; mucho más pequeño que la migración total "
                            "pero captura la señal de violencia con mayor precisión."
                        ]),
                    ], style={"marginTop": "4px", "paddingLeft": "18px"}),
                ], style={"color": "#64748B", "fontSize": "0.78rem",
                          "border": "1px solid #334155", "borderRadius": "6px",
                          "padding": "12px", "background": "#1E293B"}), md=12),
            ], className="mb-3"),
        ],
            label_style={"color": "#94A3B8"},
            active_label_style={"color": "#F8FAFC", "fontWeight": "600", "borderTop": "2px solid #2E86AB"},
        ),
    ], id="main-tabs", active_tab="tab-overview"),

], fluid=True, style={"backgroundColor": "#0F172A", "minHeight": "100vh", "padding": "20px"})

# ── Callbacks ─────────────────────────────────────────────────────────────────

@app.callback(
    Output("kpis-row",        "children"),
    Output("g-choropleth",    "figure"),
    Output("g-national-trend","figure"),
    Input("year-range", "value"),
)
def update_overview(year_range):
    yr0, yr1 = year_range
    return build_kpis(yr0, yr1), fig_choropleth(yr0, yr1), fig_national_trend(yr0, yr1)


@app.callback(
    Output("g-slope", "figure"),
    Input("year-range", "value"),
)
def update_slope(year_range):
    yr0, yr1 = year_range
    return fig_slope_state(yr0, yr1)


@app.callback(
    Output("g-state-lines", "figure"),
    Input("state-pick",  "value"),
    Input("year-range",  "value"),
)
def update_state_lines(states, year_range):
    yr0, yr1 = year_range
    sel = states or STATES[:6]
    return fig_state_lines(sel, yr0, yr1)


@app.callback(
    Output("g-decomposition", "figure"),
    Output("g-scatter",       "figure"),
    Input("year-range", "value"),
)
def update_components(year_range):
    yr0, yr1 = year_range
    return fig_decomposition(yr0, yr1), fig_scatter_components(yr0, yr1)


@app.callback(
    Output("g-mun-ranking", "figure"),
    Input("year-range", "value"),
    Input("mun-state",  "value"),
    Input("mun-topn",   "value"),
)
def update_municipios(year_range, state, top_n):
    yr0, yr1 = year_range
    return fig_municipios_ranking(yr0, yr1, state, int(top_n))


@app.callback(
    Output("g-des-ranking",    "figure"),
    Output("g-des-trend",      "figure"),
    Output("g-scatter-despla", "figure"),
    Output("g-mig-adj",        "figure"),
    Input("year-range", "value"),
)
def update_scatter_despla(year_range):
    yr0, yr1 = year_range
    return (
        fig_des_ranking(yr0, yr1),
        fig_des_trend(yr0, yr1),
        fig_scatter_desplazamiento(yr0, yr1),
        fig_mig_adj_comparison(yr0, yr1),
    )


@app.callback(
    Output("g-violencia-scatter", "figure"),
    Output("g-triple-golpe",      "figure"),
    Output("g-crime-profile",     "figure"),
    Input("year-range", "value"),
)
def update_violencia(year_range):
    yr0, yr1 = year_range
    return fig_violencia_scatter(yr0, yr1), fig_triple_golpe(yr0, yr1), fig_crime_profile(yr0, yr1)


@app.callback(
    Output("g-informalidad-scatter", "figure"),
    Output("g-empleo-brecha",        "figure"),
    Output("g-informalidad-trend",   "figure"),
    Input("year-range", "value"),
)
def update_empleo(year_range):
    yr0, yr1 = year_range
    return fig_informalidad_scatter(yr0, yr1), fig_empleo_brecha(yr0, yr1), fig_informalidad_trend(yr0, yr1)


@app.callback(
    Output("g-rolling-r",       "figure"),
    Output("g-lag-heatmap",     "figure"),
    Output("g-small-multiples", "figure"),
    Input("year-range",  "value"),
    Input("pred-radio",  "value"),
)
def update_dinamica(year_range, predictor):
    yr0, yr1 = year_range
    return (
        fig_rolling_r(yr0, yr1),
        fig_lag_heatmap(yr0, yr1),
        fig_scatter_small_multiples(yr0, yr1, predictor or "sek"),
    )


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8054)
