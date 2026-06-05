"""
Dashboard de envejecimiento poblacional (CONAPO 1990–2040).

Pirámide por estado y año, choropleth de 65+, tendencia de envejecimiento y
brecha regional Norte / Centro / Sur.
"""

import json
from pathlib import Path

import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
from dash import Dash, Input, Output, dcc, html

# ── paths ──────────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent.parent / "data"
CSV_FILE = DATA_DIR / "conapo" / "estados_municipios.csv"
GEO_FILE = DATA_DIR / "mexico_states.geojson"

# ── constants ──────────────────────────────────────────────────────────────────
AGE_GROUPS = [
    "POB_00_04", "POB_05_09", "POB_10_14", "POB_15_19",
    "POB_20_24", "POB_25_29", "POB_30_34", "POB_35_39",
    "POB_40_44", "POB_45_49", "POB_50_54", "POB_55_59",
    "POB_60_64", "POB_65_69", "POB_70_74", "POB_75_79",
    "POB_80_84", "POB_85_mm",
]
AGE_LABELS = [
    "0–4", "5–9", "10–14", "15–19", "20–24", "25–29", "30–34", "35–39",
    "40–44", "45–49", "50–54", "55–59", "60–64", "65–69", "70–74", "75–79",
    "80–84", "85+",
]
GRUPOS_65   = ["POB_65_69", "POB_70_74", "POB_75_79", "POB_80_84", "POB_85_mm"]
GRUPOS_1529 = ["POB_15_19", "POB_20_24", "POB_25_29"]

NAME_MAP = {
    "Coahuila de Zaragoza":            "Coahuila",
    "Michoacán de Ocampo":             "Michoacán",
    "Veracruz de Ignacio de la Llave": "Veracruz",
}

NORTE = {
    "Baja California", "Baja California Sur", "Chihuahua", "Coahuila de Zaragoza",
    "Durango", "Nuevo León", "Sinaloa", "Sonora", "Tamaulipas",
}
SUR = {
    "Campeche", "Chiapas", "Guerrero", "Oaxaca", "Quintana Roo",
    "Tabasco", "Veracruz de Ignacio de la Llave", "Yucatán",
}

YEARS = list(range(1990, 2041, 5))
MARKS = {y: {"label": str(y), "style": {"color": "#94A3B8", "fontSize": "11px"}}
         for y in YEARS}

# ── theme ──────────────────────────────────────────────────────────────────────
CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font_color="#CBD5E1",
)
GRID = "#334155"
CARD_STYLE = {
    "background": "#1E293B", "border": "1px solid #334155",
    "borderRadius": "8px", "padding": "16px", "textAlign": "center",
}

# ── GeoJSON ────────────────────────────────────────────────────────────────────
with open(GEO_FILE) as f:
    GEOJSON = json.load(f)

# ── data (pre-aggregated once at startup via a single scan) ────────────────────
_lf = pl.scan_csv(CSV_FILE)

_q_sexo = (
    _lf.group_by(["NOM_ENT", "AÑO", "SEXO"])
    .agg([pl.col(c).sum() for c in AGE_GROUPS + ["POB_TOTAL"]])
)

_q_estado = (
    _lf.group_by(["NOM_ENT", "AÑO"])
    .agg([pl.col(c).sum() for c in AGE_GROUPS + ["POB_TOTAL"]])
    .with_columns(
        pl.sum_horizontal([pl.col(c) for c in GRUPOS_65]).alias("POB_65_MAS"),
        pl.sum_horizontal([pl.col(c) for c in GRUPOS_1529]).alias("POB_1529"),
    )
    .with_columns(
        (pl.col("POB_65_MAS") / pl.col("POB_TOTAL") * 100).round(1).alias("PCT_65"),
        (pl.col("POB_1529") / pl.col("POB_TOTAL") * 100).round(1).alias("PCT_1529"),
        pl.col("NOM_ENT")
        .replace_strict(list(NAME_MAP.keys()), list(NAME_MAP.values()), default=None)
        .fill_null(pl.col("NOM_ENT"))
        .alias("NOM_GEO"),
        pl.when(pl.col("NOM_ENT").is_in(list(NORTE))).then(pl.lit("Norte"))
          .when(pl.col("NOM_ENT").is_in(list(SUR))).then(pl.lit("Sur"))
          .otherwise(pl.lit("Centro"))
          .alias("REGION"),
    )
)

df_sexo, df_estado = pl.collect_all([_q_sexo, _q_estado])

ESTADOS = sorted(df_estado["NOM_ENT"].unique().to_list())
DEFAULT_ESTADO = "Ciudad de México"


# ── figure factories ───────────────────────────────────────────────────────────

def _human(n: float) -> str:
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    if n >= 1_000:
        return f"{int(n/1_000)}K"
    return str(int(n))


def fig_pyramid(estado: str, año: int) -> go.Figure:
    d = df_sexo.filter((pl.col("NOM_ENT") == estado) & (pl.col("AÑO") == año))

    def age_vals(sexo: str) -> list[int]:
        rows = d.filter(pl.col("SEXO") == sexo)
        if len(rows) == 0:
            return [0] * len(AGE_GROUPS)
        row = rows.row(0, named=True)
        return [row[c] for c in AGE_GROUPS]

    h = age_vals("HOMBRES")
    m = age_vals("MUJERES")
    max_val = max(max(h, default=1), max(m, default=1)) * 1.1
    half = max_val / 2
    tickvals = [-max_val, -half, 0, half, max_val]
    ticktext = [_human(abs(v)) for v in tickvals]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=AGE_LABELS, x=[-v for v in h], name="Hombres",
        orientation="h", marker_color="#2E86AB",
        customdata=h,
        hovertemplate="<b>Hombres %{y}</b>: %{customdata:,.0f}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        y=AGE_LABELS, x=m, name="Mujeres",
        orientation="h", marker_color="#E84855",
        hovertemplate="<b>Mujeres %{y}</b>: %{x:,.0f}<extra></extra>",
    ))
    fig.update_layout(
        **CHART_LAYOUT,
        barmode="relative",
        title=dict(text=f"Pirámide poblacional — {estado} {año}", font=dict(size=13)),
        legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center"),
        height=480,
        margin=dict(t=60, b=20, l=55, r=10),
    )
    fig.update_xaxes(gridcolor=GRID, tickvals=tickvals, ticktext=ticktext,
                     range=[-max_val, max_val])
    fig.update_yaxes(gridcolor="rgba(0,0,0,0)")
    return fig


def fig_choropleth(año: int) -> go.Figure:
    d = df_estado.filter(pl.col("AÑO") == año)
    fig = px.choropleth_map(
        d,
        geojson=GEOJSON,
        locations="NOM_GEO",
        featureidkey="properties.name",
        color="PCT_65",
        color_continuous_scale=[[0, "#2E86AB"], [0.5, "#F4A261"], [1, "#E84855"]],
        range_color=[3, 22],
        hover_name="NOM_ENT",
        custom_data=["PCT_65", "POB_65_MAS", "POB_TOTAL"],
        labels={"PCT_65": "65+ (%)"},
        map_style="carto-darkmatter",
        center={"lat": 23.6, "lon": -102.5},
        zoom=4,
        title=f"Adultos mayores 65+ (%) — {año}",
    )
    fig.update_traces(
        hovertemplate=(
            "<b>%{hovertext}</b><br>"
            "65+: %{customdata[0]:.1f}%<br>"
            "Pob. 65+: %{customdata[1]:,.0f}<br>"
            "Total: %{customdata[2]:,.0f}<extra></extra>"
        )
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", font_color="#CBD5E1",
        margin=dict(t=50, b=0, l=0, r=0),
        height=480,
        coloraxis_colorbar=dict(
            title=dict(text="65+ (%)", font=dict(color="#CBD5E1")),
            tickfont=dict(color="#CBD5E1"),
            tickvals=[3, 8, 13, 18, 22],
        ),
    )
    return fig


def fig_trend(estado_sel: str) -> go.Figure:
    fig = go.Figure()
    for estado in ESTADOS:
        d = df_estado.filter(pl.col("NOM_ENT") == estado).sort("AÑO")
        sel = estado == estado_sel
        fig.add_trace(go.Scatter(
            x=d["AÑO"].to_list(),
            y=d["PCT_65"].to_list(),
            name=estado,
            mode="lines",
            line=dict(color="#2E86AB" if sel else "#334155", width=2.5 if sel else 1),
            opacity=1.0 if sel else 0.55,
            showlegend=sel,
            hovertemplate=f"<b>{estado}</b><br>%{{x}}: %{{y:.1f}}%<extra></extra>",
        ))
    fig.add_vline(
        x=2020, line_dash="dash", line_color="#475569",
        annotation_text="→ proyección", annotation_font_color="#64748B",
        annotation_position="top right",
    )
    fig.update_layout(
        **CHART_LAYOUT,
        title=dict(text=f"Evolución 65+ (%) — todos los estados, <b>{estado_sel}</b> destacado",
                   font=dict(size=13)),
        legend=dict(x=0.02, y=0.98),
        height=400,
        margin=dict(t=50, b=40, l=55, r=10),
    )
    fig.update_xaxes(gridcolor=GRID, dtick=5, title="")
    fig.update_yaxes(gridcolor=GRID, title="% de 65+")
    return fig


def fig_region(año: int) -> go.Figure:
    d = (
        df_estado.filter(pl.col("AÑO") == año)
        .group_by("REGION")
        .agg(
            pl.col("POB_65_MAS").sum(),
            pl.col("POB_TOTAL").sum(),
        )
        .with_columns(
            (pl.col("POB_65_MAS") / pl.col("POB_TOTAL") * 100).round(1).alias("PCT_65"),
        )
        .sort("PCT_65", descending=True)
    )
    COLORS = {"Norte": "#2E86AB", "Centro": "#F4A261", "Sur": "#3BB273"}
    fig = px.bar(
        d, x="REGION", y="PCT_65",
        color="REGION", color_discrete_map=COLORS,
        text="PCT_65",
        labels={"PCT_65": "65+ (%)", "REGION": ""},
        title=f"65+ (%) por región — {año}",
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside",
                      textfont_color="#CBD5E1")
    fig.update_layout(
        **CHART_LAYOUT,
        height=400,
        showlegend=False,
        margin=dict(t=50, b=40, l=55, r=10),
    )
    fig.update_xaxes(gridcolor=GRID, title="")
    fig.update_yaxes(gridcolor=GRID, range=[0, float(d["PCT_65"].max()) * 1.25],
                     title="% de 65+")
    return fig


# ── KPI helpers ────────────────────────────────────────────────────────────────

def _nacional(año: int) -> dict:
    d = df_estado.filter(pl.col("AÑO") == año)
    total = int(d["POB_TOTAL"].sum())
    pob65 = int(d["POB_65_MAS"].sum())
    pob1529 = int(d["POB_1529"].sum())
    return {
        "pct65": pob65 / total * 100,
        "total": total,
        "pct1529": pob1529 / total * 100,
    }


def _estado_pct65(estado: str, año: int) -> float:
    d = df_estado.filter((pl.col("NOM_ENT") == estado) & (pl.col("AÑO") == año))
    return float(d["PCT_65"][0]) if len(d) else 0.0


def _kpi(label: str, value: str, note: str = "") -> list:
    return [
        html.P(label, style={"color": "#94A3B8", "fontSize": "11px",
                              "marginBottom": "2px", "textTransform": "uppercase",
                              "letterSpacing": "0.05em"}),
        html.H4(value, style={"color": "#F8FAFC", "marginBottom": "0"}),
        html.P(note, style={"color": "#64748B", "fontSize": "11px",
                             "marginTop": "2px", "marginBottom": "0"}),
    ]


# ── layout ─────────────────────────────────────────────────────────────────────
app = Dash(__name__, external_stylesheets=[dbc.themes.SLATE])
app.title = "Envejecimiento · CONAPO"

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H3("Envejecimiento Poblacional · México 1990–2040",
                    style={"color": "#F8FAFC", "marginBottom": "2px"}),
            html.P("Fuente: CONAPO. Valores a partir de 2021 son proyecciones.",
                   style={"color": "#64748B", "fontSize": "12px"}),
        ])
    ], style={"paddingTop": "20px", "marginBottom": "12px"}),

    # Controls
    dbc.Row([
        dbc.Col([
            html.Label("Año", style={"color": "#94A3B8", "fontSize": "12px",
                                     "marginBottom": "4px", "display": "block"}),
            dcc.Slider(
                id="slider-año", min=1990, max=2040, step=5, value=2020,
                marks=MARKS, tooltip={"placement": "bottom", "always_visible": False},
            ),
        ], md=8),
        dbc.Col([
            html.Label("Estado", style={"color": "#94A3B8", "fontSize": "12px",
                                        "marginBottom": "4px", "display": "block"}),
            dcc.Dropdown(
                id="dropdown-estado",
                options=[{"label": e, "value": e} for e in ESTADOS],
                value=DEFAULT_ESTADO, clearable=False,
                style={"backgroundColor": "#1E293B", "color": "#0F172A"},
            ),
        ], md=4),
    ], style={"marginBottom": "20px"}),

    # KPI row
    dbc.Row([
        dbc.Col(dbc.Card(id="kpi-65-nac",    style=CARD_STYLE), md=3),
        dbc.Col(dbc.Card(id="kpi-total",     style=CARD_STYLE), md=3),
        dbc.Col(dbc.Card(id="kpi-1529",      style=CARD_STYLE), md=3),
        dbc.Col(dbc.Card(id="kpi-65-estado", style=CARD_STYLE), md=3),
    ], style={"marginBottom": "20px"}),

    # Pyramid + choropleth
    dbc.Row([
        dbc.Col(dcc.Graph(id="chart-pyramid",    config={"displayModeBar": False}), md=6),
        dbc.Col(dcc.Graph(id="chart-choropleth", config={"displayModeBar": False}), md=6),
    ], style={"marginBottom": "16px"}),

    # Trend + region
    dbc.Row([
        dbc.Col(dcc.Graph(id="chart-trend",  config={"displayModeBar": False}), md=8),
        dbc.Col(dcc.Graph(id="chart-region", config={"displayModeBar": False}), md=4),
    ], style={"marginBottom": "40px"}),

], fluid=True, style={"backgroundColor": "#0F172A", "minHeight": "100vh", "padding": "0 24px"})


# ── callback ───────────────────────────────────────────────────────────────────
@app.callback(
    Output("chart-pyramid",    "figure"),
    Output("chart-choropleth", "figure"),
    Output("chart-trend",      "figure"),
    Output("chart-region",     "figure"),
    Output("kpi-65-nac",       "children"),
    Output("kpi-total",        "children"),
    Output("kpi-1529",         "children"),
    Output("kpi-65-estado",    "children"),
    Input("slider-año",      "value"),
    Input("dropdown-estado", "value"),
)
def update_all(año: int, estado: str):
    nac = _nacional(año)
    pct65_est = _estado_pct65(estado, año)
    estado_short = estado[:18] + "…" if len(estado) > 18 else estado

    return (
        fig_pyramid(estado, año),
        fig_choropleth(año),
        fig_trend(estado),
        fig_region(año),
        _kpi("65+ nacional", f"{nac['pct65']:.1f}%"),
        _kpi("Población total", f"{nac['total']/1e6:.1f} M"),
        _kpi("15–29 nacional", f"{nac['pct1529']:.1f}%", "dividendo demográfico"),
        _kpi(f"65+ {estado_short}", f"{pct65_est:.1f}%"),
    )


if __name__ == "__main__":
    app.run(debug=True, port=8055)
