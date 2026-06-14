"""Dashboard: Migración Interna Estimada — México

Método: Ecuación demográfica de balanza de componentes
  Migración neta(t) = Población(t+1) − Población(t) − Nacimientos(t) + Defunciones(t)

Fuentes:
  • INEGI – Estadística de nacimientos y defunciones por municipio, 2017–2024
    data/inegi/nacimientos_descesos/{año}.csv
  • CONAPO – Proyecciones de población municipal 1990–2040
    data/conapo/estados_municipios.csv

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

_df_des_raw = (
    pl.read_csv("data/desaparecidos.csv", infer_schema_length=5000)
    .filter(pl.col("CVE_ENT").is_between(1, 32))
    .with_columns(
        pl.col("FECHA_DESAPARICION").str.slice(0, 10)
          .str.to_date("%Y-%m-%d", strict=False).alias("fecha_des")
    )
    .filter(pl.col("fecha_des").is_not_null())
    .with_columns(
        pl.col("fecha_des").dt.year().cast(pl.Int32).alias("año"),
        pl.col("ENTIDAD").replace(_DES_NORM).alias("NOM_ENT"),
    )
)

# Pre-aggregate to (NOM_ENT, año) so callbacks just filter this small table
df_des_by_year = (
    _df_des_raw
    .group_by(["NOM_ENT", "año"])
    .agg(pl.len().alias("desaparecidos"))
)

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
                    html.B("Fuente: ", style={"color": "#94A3B8"}),
                    "RNPDNO — solo ~57% de registros tienen fecha conocida. "
                    "La migración es el residual demográfico de esta herramienta. ",
                    html.B("Paradoja Tamaulipas: ", style={"color": "#F4A261"}),
                    "aparece neutral en migración (−1k) pero registra 1,240 desapariciones en el período — "
                    "los flujos de entrada (empleo fronterizo) y salida (inseguridad + desapariciones) se cancelan.",
                ], style={"color": "#64748B", "fontSize": "0.78rem", "marginBottom": "8px"}), md=12),
            ], className="mt-3"),
            dbc.Row([dbc.Col(dcc.Graph(id="g-scatter-despla"), md=12)]),
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
    Output("g-scatter-despla", "figure"),
    Input("year-range", "value"),
)
def update_scatter_despla(year_range):
    yr0, yr1 = year_range
    return fig_scatter_desplazamiento(yr0, yr1)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8054)
