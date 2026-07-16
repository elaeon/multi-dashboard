"""
Dashboard de PIB Estatal (INEGI PIBE, base 2018, 2003-2024).

Seis hallazgos validados en dashboard_data/pibe_*.parquet:
concentración, tendencia de concentración, crecimiento por estado,
colapso petrolero, impacto COVID-19 y terciarización.
"""

import json
from pathlib import Path

import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
from dash import Dash, dcc, html

# ── paths ────────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent.parent / "dashboard_data"
GEO_FILE = Path(__file__).parent.parent / "data" / "mexico_states.geojson"

# ── theme ────────────────────────────────────────────────────────────────────
CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font_color="#CBD5E1",
)
CARD_STYLE = {
    "background": "#1E293B", "border": "1px solid #334155",
    "borderRadius": "8px", "padding": "16px", "textAlign": "center",
}
TAB_STYLE = {"backgroundColor": "#0F172A", "color": "#94A3B8", "borderTop": "none"}
TAB_SEL = {"backgroundColor": "#1E293B", "color": "#F8FAFC",
           "borderTop": "2px solid #2E86AB", "fontWeight": "600"}

FOCUS, CONTEXT, GREEN, RED, ORANGE = "#2E86AB", "#475569", "#3BB273", "#E84855", "#F4A261"

with open(GEO_FILE) as f:
    GEOJSON = json.load(f)

# ── data (pre-built by scripts/prepare_pibe.py) ─────────────────────────────
gdp_long = pl.read_parquet(DATA_DIR / "pibe_gdp_long.parquet")
gini_by_year = pl.read_parquet(DATA_DIR / "pibe_gini_by_year.parquet")
share_by_state_year = pl.read_parquet(DATA_DIR / "pibe_share_by_state_year.parquet")
cagr_by_state = pl.read_parquet(DATA_DIR / "pibe_cagr_by_state.parquet")
oil_share_long = pl.read_parquet(DATA_DIR / "pibe_oil_share_long.parquet")
covid = pl.read_parquet(DATA_DIR / "pibe_covid.parquet")
sector_composition = pl.read_parquet(DATA_DIR / "pibe_sector_composition.parquet")
fossil_sector = pl.read_parquet(DATA_DIR / "pibe_fossil_states_sector.parquet")

NAME_MAP = {
    "Coahuila de Zaragoza": "Coahuila",
    "Michoacán de Ocampo": "Michoacán",
    "Veracruz de Ignacio de la Llave": "Veracruz",
}

FOSSIL_STATES = ["Campeche", "Tabasco", "Veracruz de Ignacio de la Llave", "Tamaulipas", "Chiapas"]


def _nom_geo(col: str) -> pl.Expr:
    return (
        pl.col(col)
        .replace_strict(list(NAME_MAP.keys()), list(NAME_MAP.values()), default=None)
        .fill_null(pl.col(col))
    )


cagr_by_state = cagr_by_state.with_columns(_nom_geo("entidad").alias("NOM_GEO"))

# ── KPI values ───────────────────────────────────────────────────────────────
_top5_2022 = gini_by_year.filter(pl.col("year") == 2022)["top5_share"][0]
_gini_2022 = gini_by_year.filter(pl.col("year") == 2022)["gini"][0]
_cagr_nacional = cagr_by_state.filter(pl.col("entidad") == "Estados Unidos Mexicanos")["cagr_pct"][0]
_cagr_qroo = cagr_by_state.filter(pl.col("entidad") == "Quintana Roo")["cagr_pct"][0]
_cagr_camp = cagr_by_state.filter(pl.col("entidad") == "Campeche")["cagr_pct"][0]
_oil_share_2003 = oil_share_long.filter((pl.col("entidad") == "Estados Unidos Mexicanos") & (pl.col("year") == 2003))["oil_share_pct"][0]
_oil_share_2024 = oil_share_long.filter((pl.col("entidad") == "Estados Unidos Mexicanos") & (pl.col("year") == 2024))["oil_share_pct"][0]
_oil_mdp_2004 = oil_share_long.filter((pl.col("entidad") == "Estados Unidos Mexicanos") & (pl.col("year") == 2004))["oil_mdp"][0]
_oil_mdp_2024 = oil_share_long.filter((pl.col("entidad") == "Estados Unidos Mexicanos") & (pl.col("year") == 2024))["oil_mdp"][0]
_oil_pct_change = (_oil_mdp_2024 - _oil_mdp_2004) / _oil_mdp_2004 * 100
_covid_2020_nat = covid.filter((pl.col("entidad") == "Estados Unidos Mexicanos") & (pl.col("year") == 2020))["var_pct"][0]
_covid_2021_nat = covid.filter((pl.col("entidad") == "Estados Unidos Mexicanos") & (pl.col("year") == 2021))["var_pct"][0]


# ── figure factories ─────────────────────────────────────────────────────────

def fig_concentration_map() -> go.Figure:
    d = share_by_state_year.filter(pl.col("year") == 2022)
    fig = px.choropleth_map(
        d, geojson=GEOJSON, locations="NOM_GEO", featureidkey="properties.name",
        color="share_pct",
        color_continuous_scale=[[0, "#1E293B"], [0.5, "#2E86AB"], [1, "#F4A261"]],
        hover_name="entidad", custom_data=["share_pct"],
        labels={"share_pct": "% del PIB nacional"},
        map_style="carto-darkmatter", center={"lat": 23.6, "lon": -102.5}, zoom=4,
    )
    fig.update_traces(hovertemplate="<b>%{hovertext}</b><br>%{customdata[0]:.1f}% del PIB nacional<extra></extra>")
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", font_color="#CBD5E1",
        title=dict(text="<b>CDMX y 4 estados concentran 44% del PIB nacional</b>"
                        "<br><sup style='color:#94A3B8'>Participación estatal en el PIB, 2022</sup>"),
        margin=dict(t=70, b=0, l=0, r=0), height=580,
        coloraxis_colorbar=dict(title=dict(text="%", font=dict(color="#CBD5E1")), tickfont=dict(color="#CBD5E1")),
    )
    return fig


def fig_concentration_ranking() -> go.Figure:
    d = share_by_state_year.filter(pl.col("year") == 2022).sort("share_pct", descending=True).head(10)
    colors = [FOCUS if i < 5 else CONTEXT for i in range(d.height)]
    fig = go.Figure(go.Bar(
        x=d["share_pct"], y=d["entidad"], orientation="h", marker_color=colors,
        text=[f"{v:.1f}%" for v in d["share_pct"]], textposition="outside",
    ))
    fig.add_vline(x=_top5_2022, line_dash="dot", line_color="#94A3B8",
                  annotation_text=f"top-5: {_top5_2022:.1f}%", annotation_font_color="#94A3B8")
    fig.update_layout(
        **CHART_LAYOUT,
        title=dict(text="<b>Los 10 estados más grandes producen el 63% del PIB</b>"
                        "<br><sup style='color:#94A3B8'>% del PIB nacional, 2022</sup>"),
        yaxis=dict(autorange="reversed", gridcolor="rgba(0,0,0,0)"),
        xaxis=dict(gridcolor="#334155", title="% del PIB nacional"),
        height=420, margin=dict(l=140),
    )
    return fig


def fig_gini_trend() -> go.Figure:
    d = gini_by_year.sort("year")
    fig = go.Figure(go.Scatter(x=d["year"], y=d["gini"], mode="lines+markers", line=dict(color=FOCUS, width=2)))
    fig.add_annotation(x=2003, y=d.filter(pl.col("year") == 2003)["gini"][0], text="0.450", showarrow=True, ay=-30,
                        font=dict(color="#94A3B8"))
    fig.add_annotation(x=2022, y=d.filter(pl.col("year") == 2022)["gini"][0], text="0.440", showarrow=True, ay=-30,
                        font=dict(color="#94A3B8"))
    fig.update_layout(
        **CHART_LAYOUT,
        title=dict(text="<b>Dos décadas sin desconcentración económica</b>"
                        "<br><sup style='color:#94A3B8'>Coeficiente de Gini del PIB estatal, 2003-2024R</sup>"),
        xaxis=dict(gridcolor="#334155"),
        yaxis=dict(gridcolor="#334155", title="Gini"),
        height=380,
    )
    return fig


def fig_cdmx_share_trend() -> go.Figure:
    d = share_by_state_year.filter(pl.col("entidad") == "Ciudad de México").sort("year")
    fig = go.Figure(go.Scatter(x=d["year"], y=d["share_pct"], mode="lines+markers", line=dict(color=FOCUS, width=2)))
    for y in (2003, 2022, 2024):
        row = d.filter(pl.col("year") == y)
        if row.height:
            fig.add_annotation(x=y, y=row["share_pct"][0], text=f"{row['share_pct'][0]:.1f}%",
                                showarrow=True, ay=-25, font=dict(color="#94A3B8"))
    fig.update_layout(
        **CHART_LAYOUT,
        title=dict(text="<b>La participación de CDMX se mantiene prácticamente estable</b>"
                        "<br><sup style='color:#94A3B8'>% del PIB nacional aportado por Ciudad de México</sup>"),
        xaxis=dict(gridcolor="#334155"),
        yaxis=dict(gridcolor="#334155", title="% del PIB nacional"),
        height=380,
    )
    return fig


def fig_growth_ranking() -> go.Figure:
    d = cagr_by_state.filter(pl.col("entidad") != "Estados Unidos Mexicanos").sort("cagr_pct", descending=True)
    top4 = set(d.head(4)["entidad"])
    colors = [
        GREEN if e in top4 else (RED if e == "Campeche" else CONTEXT)
        for e in d["entidad"]
    ]
    fig = go.Figure(go.Bar(
        x=d["cagr_pct"], y=d["entidad"], orientation="h", marker_color=colors,
    ))
    fig.add_vline(x=0, line_color="#64748B")
    fig.add_vline(x=_cagr_nacional, line_dash="dot", line_color="#94A3B8",
                  annotation_text=f"nacional: {_cagr_nacional:.2f}%/año", annotation_font_color="#94A3B8")
    fig.update_layout(
        **CHART_LAYOUT,
        title=dict(text="<b>Campeche es el único estado con crecimiento negativo</b>"
                        "<br><sup style='color:#94A3B8'>CAGR real del PIB estatal, 2003-2022</sup>"),
        yaxis=dict(gridcolor="rgba(0,0,0,0)", tickfont=dict(size=10)),
        xaxis=dict(gridcolor="#334155", title="CAGR (%/año)"),
        height=max(300, d.height * 22 + 80), margin=dict(l=150),
    )
    return fig


def fig_oil_national_trend() -> go.Figure:
    d = oil_share_long.filter(pl.col("entidad") == "Estados Unidos Mexicanos").sort("year")
    fig = go.Figure(go.Scatter(x=d["year"], y=d["oil_mdp"], mode="lines", line=dict(color=RED, width=2), fill="tozeroy",
                                fillcolor="rgba(232,72,85,0.1)"))
    fig.add_vline(x=2004, line_dash="dot", line_color="#94A3B8",
                  annotation_text="pico 2004", annotation_font_color="#94A3B8")
    fig.add_annotation(x=2024, y=d.filter(pl.col("year") == 2024)["oil_mdp"][0],
                        text="−42.3% vs. 2004", showarrow=True, ay=-30, font=dict(color=RED))
    fig.update_layout(
        **CHART_LAYOUT,
        title=dict(text="<b>El PIB petrolero cayó 42% desde su pico de 2004</b>"
                        "<br><sup style='color:#94A3B8'>Minería petrolera, millones de pesos constantes 2018</sup>"),
        xaxis=dict(gridcolor="#334155"),
        yaxis=dict(gridcolor="#334155", title="Millones de pesos"),
        height=380,
    )
    return fig


def fig_oil_state_dependence() -> go.Figure:
    d = oil_share_long.filter(pl.col("entidad").is_in(["Campeche", "Tabasco"])).sort(["entidad", "year"])
    fig = go.Figure()
    for estado, color in [("Campeche", RED), ("Tabasco", ORANGE)]:
        sub = d.filter(pl.col("entidad") == estado)
        fig.add_trace(go.Scatter(x=sub["year"], y=sub["oil_share_pct"], mode="lines", name=estado,
                                  line=dict(color=color, width=2)))
    fig.update_layout(
        **CHART_LAYOUT,
        title=dict(text="<b>Campeche sigue siendo el más dependiente del petróleo, pese a la caída</b>"
                        "<br><sup style='color:#94A3B8'>PIB petrolero como % del PIB estatal propio</sup>"),
        xaxis=dict(gridcolor="#334155"),
        yaxis=dict(gridcolor="#334155", title="% del PIB estatal"),
        legend=dict(orientation="h", y=1.12, x=0),
        height=380,
    )
    return fig


def fig_fossil_dependence_ranking() -> go.Figure:
    d = (
        oil_share_long
        .filter(pl.col("entidad").is_in(FOSSIL_STATES) & (pl.col("year") == 2024))
        .sort("oil_share_pct", descending=True)
    )
    colors = [RED if e in ("Campeche", "Tabasco") else CONTEXT for e in d["entidad"]]
    fig = go.Figure(go.Bar(
        x=d["oil_share_pct"], y=d["entidad"], orientation="h", marker_color=colors,
        text=[f"{v:.1f}%" for v in d["oil_share_pct"]], textposition="outside",
    ))
    fig.update_layout(
        **CHART_LAYOUT,
        title=dict(text="<b>Fuera de Campeche y Tabasco, la exposición petrolera es marginal</b>"
                        "<br><sup style='color:#94A3B8'>PIB petrolero como % del PIB estatal propio, 2024R</sup>"),
        yaxis=dict(autorange="reversed", gridcolor="rgba(0,0,0,0)"),
        xaxis=dict(gridcolor="#334155", title="% del PIB estatal"),
        height=380, margin=dict(l=180),
    )
    return fig


_FOSSIL_CATEGORY_COLORS = {
    "Primario": "#64748B",
    "Secundario sin petróleo": ORANGE,
    "Petróleo": RED,
    "Terciario": FOCUS,
}


def fig_fossil_sector_composition() -> go.Figure:
    d = fossil_sector.filter(pl.col("year") == 2024)
    fig = go.Figure()
    for cat, color in _FOSSIL_CATEGORY_COLORS.items():
        sub = d.filter(pl.col("categoria") == cat).select("entidad", "share_pct")
        sub = sub.join(pl.DataFrame({"entidad": FOSSIL_STATES}), on="entidad", how="right")
        fig.add_trace(go.Bar(
            x=sub["share_pct"], y=sub["entidad"], orientation="h", name=cat, marker_color=color,
            text=[f"{v:.1f}%" for v in sub["share_pct"]], textposition="inside", insidetextanchor="middle",
        ))
    fig.update_layout(
        barmode="stack",
        **CHART_LAYOUT,
        title=dict(text="<b>El petróleo desplaza al sector terciario en Campeche y Tabasco</b>"
                        "<br><sup style='color:#94A3B8'>Composición del PIB estatal, 2024R</sup>"),
        yaxis=dict(autorange="reversed", gridcolor="rgba(0,0,0,0)"),
        xaxis=dict(gridcolor="#334155", title="% del PIB estatal"),
        legend=dict(orientation="h", y=-0.2, x=0),
        height=380, margin=dict(l=180, b=60),
    )
    return fig


def fig_covid_bar() -> go.Figure:
    d = covid.filter((pl.col("entidad") != "Estados Unidos Mexicanos") & (pl.col("year") == 2020)).sort("var_pct")
    hardest = set(d.head(4)["entidad"])
    colors = [RED if e in hardest else (GREEN if e == "Tabasco" else CONTEXT) for e in d["entidad"]]
    fig = go.Figure(go.Bar(x=d["var_pct"], y=d["entidad"], orientation="h", marker_color=colors))
    fig.add_vline(x=_covid_2020_nat, line_dash="dot", line_color="#94A3B8",
                  annotation_text=f"nacional: {_covid_2020_nat:.1f}%", annotation_font_color="#94A3B8")
    fig.update_layout(
        **CHART_LAYOUT,
        title=dict(text="<b>El turismo sufrió el mayor golpe del COVID-19 en 2020</b>"
                        "<br><sup style='color:#94A3B8'>Variación % anual del PIB estatal, 2020</sup>"),
        yaxis=dict(gridcolor="rgba(0,0,0,0)", tickfont=dict(size=10)),
        xaxis=dict(gridcolor="#334155", title="Variación % anual"),
        height=max(300, d.height * 22 + 80), margin=dict(l=150),
    )
    return fig


def fig_covid_rebound_slope() -> go.Figure:
    d2020 = covid.filter((pl.col("entidad") != "Estados Unidos Mexicanos") & (pl.col("year") == 2020)).select("entidad", pl.col("var_pct").alias("v2020"))
    d2021 = covid.filter((pl.col("entidad") != "Estados Unidos Mexicanos") & (pl.col("year") == 2021)).select("entidad", pl.col("var_pct").alias("v2021"))
    d = d2020.join(d2021, on="entidad").sort("v2020")
    fig = go.Figure()
    for row in d.iter_rows(named=True):
        color = RED if row["v2020"] < 0 else GREEN
        fig.add_trace(go.Scatter(
            x=["2020", "2021"], y=[row["v2020"], row["v2021"]], mode="lines+markers",
            line=dict(color=color, width=1.3), marker=dict(color=color, size=6),
            showlegend=False,
            hovertemplate=f"<b>{row['entidad']}</b><br>%{{x}}: %{{y:.1f}}%<extra></extra>",
        ))
    fig.update_xaxes(type="category", gridcolor="rgba(0,0,0,0)")
    fig.update_layout(
        **CHART_LAYOUT,
        title=dict(text="<b>El rebote de 2021 fue proporcional a la caída de 2020</b>"
                        "<br><sup style='color:#94A3B8'>Variación % anual del PIB por estado</sup>"),
        yaxis=dict(gridcolor="#334155", title="Variación % anual"),
        height=420,
    )
    return fig


_SECTOR_COLORS = {
    "Actividades primarias": "#64748B",
    "Actividades secundarias": ORANGE,
    "Actividades terciarias": FOCUS,
}


def fig_sector_composition_bars() -> go.Figure:
    d = sector_composition.filter(
        pl.col("actividad").is_in(list(_SECTOR_COLORS)) & pl.col("year").is_in([2003, 2024])
    )
    fig = go.Figure()
    for cat, color in _SECTOR_COLORS.items():
        sub = d.filter(pl.col("actividad") == cat).sort("year")
        labels = [f"{y}R" if y == 2024 else str(y) for y in sub["year"]]
        fig.add_trace(go.Bar(
            x=sub["share_pct"], y=labels, orientation="h", name=cat.replace("Actividades ", "").capitalize(),
            marker_color=color, text=[f"{v:.1f}%" for v in sub["share_pct"]],
            textposition="inside", insidetextanchor="middle",
        ))
    fig.update_layout(
        barmode="stack",
        **CHART_LAYOUT,
        title=dict(text="<b>Terciarización: los servicios ganaron 5 puntos del PIB</b>"
                        "<br><sup style='color:#94A3B8'>Composición sectorial nacional, 2003 vs. 2024R</sup>"),
        xaxis=dict(range=[0, 100], visible=False),
        legend=dict(orientation="h", y=-0.25, x=0),
        margin=dict(t=60, b=60, l=10, r=10),
        height=300,
    )
    return fig


def fig_sector_trend() -> go.Figure:
    labels = {
        "Actividades primarias": ("Primarias", "#64748B"),
        "Actividades secundarias": ("Secundarias", ORANGE),
        "Actividades terciarias": ("Terciarias", FOCUS),
        "21 - Minería": ("Minería", RED),
        "31-33 - Industrias manufactureras": ("Manufactura", GREEN),
    }
    fig = go.Figure()
    for cat, (name, color) in labels.items():
        sub = sector_composition.filter(pl.col("actividad") == cat).sort("year")
        fig.add_trace(go.Scatter(x=sub["year"], y=sub["share_pct"], mode="lines", name=name,
                                  line=dict(color=color, width=2)))
    fig.update_layout(
        **CHART_LAYOUT,
        title=dict(text="<b>La caída del sector secundario es casi toda minería, no manufactura</b>"
                        "<br><sup style='color:#94A3B8'>% del PIB nacional por sector, 2003-2024R</sup>"),
        xaxis=dict(gridcolor="#334155"),
        yaxis=dict(gridcolor="#334155", title="% del PIB nacional"),
        legend=dict(orientation="h", y=1.15, x=0),
        height=420,
    )
    return fig


# ── app ──────────────────────────────────────────────────────────────────────
app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY], title="PIB Estatal · INEGI")


def kpi(title, value, sub=""):
    return dbc.Col(html.Div([
        html.Div(str(value), style={"fontSize": "2rem", "fontWeight": "700", "color": "#F8FAFC"}),
        html.Div(title, style={"fontSize": "0.85rem", "color": "#94A3B8", "marginTop": "2px"}),
        html.Div(sub, style={"fontSize": "0.75rem", "color": "#64748B"}) if sub else None,
    ], style=CARD_STYLE), md=3)


app.layout = html.Div(style={"backgroundColor": "#0F172A", "minHeight": "100vh", "padding": "24px"}, children=[
    html.H1("PIB Estatal 2003–2024", style={"color": "#F8FAFC", "fontWeight": "700", "marginBottom": "4px"}),
    html.P("INEGI · Producto Interno Bruto por Entidad Federativa, base 2018",
           style={"color": "#94A3B8", "marginBottom": "24px"}),

    dbc.Row([
        kpi(f"{_top5_2022:.1f}%", "Top 5 estados / PIB nacional", f"Gini {_gini_2022:.3f} · reparto igualitario sería 15.6%"),
        kpi(f"{_cagr_nacional:.2f}%/año", "CAGR nacional 2003–2022", f"Quintana Roo +{_cagr_qroo:.2f}% · Campeche {_cagr_camp:.2f}%"),
        kpi(f"{_oil_pct_change:.1f}%", "PIB minero-petrolero 2004→2024R",
            f"{_oil_share_2003:.1f}% del PIB (2003) → {_oil_share_2024:.1f}% (2024R)"),
        kpi(f"{_covid_2020_nat:.2f}%", "Caída del PIB nacional 2020", f"Rebote 2021: {_covid_2021_nat:+.2f}% nacional"),
    ], className="g-2 mb-4"),

    dcc.Tabs(style={"marginBottom": "16px"}, children=[
        dcc.Tab(label="Concentración", style=TAB_STYLE, selected_style=TAB_SEL, children=[
            dbc.Row([
                dbc.Col(dcc.Graph(figure=fig_concentration_map(), config={"displayModeBar": False}), md=7),
                dbc.Col(dcc.Graph(figure=fig_concentration_ranking(), config={"displayModeBar": False}), md=5),
            ], className="mt-3"),
        ]),
        dcc.Tab(label="Sin desconcentración", style=TAB_STYLE, selected_style=TAB_SEL, children=[
            dbc.Row([
                dbc.Col(dcc.Graph(figure=fig_gini_trend(), config={"displayModeBar": False}), md=6),
                dbc.Col(dcc.Graph(figure=fig_cdmx_share_trend(), config={"displayModeBar": False}), md=6),
            ], className="mt-3"),
        ]),
        dcc.Tab(label="Crecimiento", style=TAB_STYLE, selected_style=TAB_SEL, children=[
            dbc.Row([
                dbc.Col(dcc.Graph(figure=fig_growth_ranking(), config={"displayModeBar": False}), md=12),
            ], className="mt-3"),
        ]),
        dcc.Tab(label="Petróleo", style=TAB_STYLE, selected_style=TAB_SEL, children=[
            dbc.Row([
                dbc.Col(dcc.Graph(figure=fig_oil_national_trend(), config={"displayModeBar": False}), md=6),
                dbc.Col(dcc.Graph(figure=fig_oil_state_dependence(), config={"displayModeBar": False}), md=6),
            ], className="mt-3"),
        ]),
        dcc.Tab(label="Estados petroleros", style=TAB_STYLE, selected_style=TAB_SEL, children=[
            dbc.Row([
                dbc.Col(dcc.Graph(figure=fig_fossil_dependence_ranking(), config={"displayModeBar": False}), md=6),
                dbc.Col(dcc.Graph(figure=fig_fossil_sector_composition(), config={"displayModeBar": False}), md=6),
            ], className="mt-3"),
        ]),
        dcc.Tab(label="COVID-2020", style=TAB_STYLE, selected_style=TAB_SEL, children=[
            dbc.Row([
                dbc.Col(dcc.Graph(figure=fig_covid_bar(), config={"displayModeBar": False}), md=6),
                dbc.Col(dcc.Graph(figure=fig_covid_rebound_slope(), config={"displayModeBar": False}), md=6),
            ], className="mt-3"),
        ]),
        dcc.Tab(label="Terciarización", style=TAB_STYLE, selected_style=TAB_SEL, children=[
            dbc.Row([
                dbc.Col(dcc.Graph(figure=fig_sector_composition_bars(), config={"displayModeBar": False}), md=12),
            ], className="mt-3"),
            dbc.Row([
                dbc.Col(dcc.Graph(figure=fig_sector_trend(), config={"displayModeBar": False}), md=12),
            ], className="mt-3"),
        ]),
    ]),
])


if __name__ == "__main__":
    app.run(debug=True, port=8063)
