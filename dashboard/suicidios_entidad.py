"""
Dashboard: Suicidios por Entidad — México (1990–2024)

Filters: año range + entidad dropdown.
Charts: KPI cards, rate trend by gender, annual counts bar,
        choropleth map (latest year), state ranking bar.

Run: uv run python dashboard/suicidios_entidad.py
"""

import json
import polars as pl
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc

# ── data ──────────────────────────────────────────────────────────────────────
df = pl.read_csv("data/suicidios_entidad.csv")

with open("data/mexico_states.geojson") as f:
    GEO = json.load(f)

MIN_YEAR = int(df["AÑO"].min())
MAX_YEAR = int(df["AÑO"].max())
YEAR_MARKS = {y: str(y) for y in range(MIN_YEAR, MAX_YEAR + 1, 5)}

STATE_LIST = sorted(
    df.filter(~pl.col("ENTIDAD").is_in(["Nacional", "Extranjero", "Not specified"]))
    ["ENTIDAD"].unique().to_list()
)
ENTIDAD_OPTIONS = [{"label": "Nacional", "value": "Nacional"}] + [
    {"label": s, "value": s} for s in STATE_LIST
]

# ── theme ─────────────────────────────────────────────────────────────────────
CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font_color="#CBD5E1",
    xaxis=dict(gridcolor="#334155"),
    yaxis=dict(gridcolor="#334155"),
)
CARD_STYLE = {
    "background": "#1E293B", "border": "1px solid #334155",
    "borderRadius": "8px", "padding": "16px", "textAlign": "center",
}

COLOR_MEN   = "#2E86AB"
COLOR_WOMEN = "#F4A261"
COLOR_TOTAL = "#3BB273"

CRISIS_YEARS = {
    2020: ("COVID-19", "#F4A261"),
    2023: ("Récord histórico", "#E84855"),
}


def _add_crisis_markers(fig: go.Figure, years: list[int]) -> None:
    """Add vertical dashed lines for crisis years that fall within the data range."""
    y_min, y_max = min(years), max(years)
    for year, (label, color) in CRISIS_YEARS.items():
        if y_min <= year <= y_max:
            fig.add_vline(x=year, line_width=1, line_dash="dash", line_color=color,
                          opacity=0.7)
            fig.add_annotation(
                x=year, y=0.97, xref="x", yref="paper",
                text=label, showarrow=False, textangle=-90,
                font=dict(size=10, color=color),
                yanchor="top", xanchor="right",
            )


# ── figure factories ──────────────────────────────────────────────────────────
def fig_trend(d: pl.DataFrame) -> go.Figure:
    s = d.sort("AÑO")
    years = s["AÑO"].to_list()

    fig = go.Figure([
        go.Scatter(x=years, y=s["TASA_HOMBRES"].to_list(),
                   name="Hombres", line=dict(color=COLOR_MEN, width=2)),
        go.Scatter(x=years, y=s["TASA_MUJERES"].to_list(),
                   name="Mujeres", line=dict(color=COLOR_WOMEN, width=2)),
        go.Scatter(x=years, y=s["TASA_TOTAL"].to_list(),
                   name="Total", line=dict(color=COLOR_TOTAL, width=2, dash="dot")),
    ])

    _add_crisis_markers(fig, years)

    # Gender convergence annotation
    if len(s) >= 2:
        first, last = s.head(1), s.tail(1)
        h0 = first["TASA_HOMBRES"].item() or 0
        m0 = first["TASA_MUJERES"].item() or 0.01
        h1 = last["TASA_HOMBRES"].item() or 0
        m1 = last["TASA_MUJERES"].item() or 0.01
        y0_yr, y1_yr = first["AÑO"].item(), last["AÑO"].item()
        fig.add_annotation(
            x=0.01, y=0.99, xref="paper", yref="paper",
            text=(f"Brecha H/M: <b>{h1/m1:.1f}x</b> en {y1_yr}"
                  f"<br>(era {h0/m0:.1f}x en {y0_yr})"),
            showarrow=False, align="left",
            bgcolor="#0F172A", bordercolor="#334155", borderwidth=1,
            font=dict(size=11, color="#CBD5E1"),
            xanchor="left", yanchor="top",
        )

    fig.update_layout(
        title="Tasa de suicidio por 100,000 habitantes",
        height=380, yaxis_title="Tasa por 100k hab.",
        legend=dict(orientation="h", y=-0.18, x=0),
        margin=dict(t=40, b=70, l=10, r=10),
        **CHART_LAYOUT,
    )
    return fig


def fig_gender_bar(d: pl.DataFrame) -> go.Figure:
    s = d.sort("AÑO")
    years = s["AÑO"].to_list()

    fig = go.Figure([
        go.Bar(x=years, y=s["HOMBRES"].to_list(),
               name="Hombres", marker_color=COLOR_MEN),
        go.Bar(x=years, y=s["MUJERES"].to_list(),
               name="Mujeres", marker_color=COLOR_WOMEN),
    ])

    _add_crisis_markers(fig, years)

    fig.update_layout(
        barmode="stack",
        title="Casos anuales por sexo",
        height=380, yaxis_title="Número de suicidios",
        legend=dict(orientation="h", y=-0.18, x=0),
        margin=dict(t=40, b=70, l=10, r=10),
        **CHART_LAYOUT,
    )
    return fig


def fig_map(states_d: pl.DataFrame, year: int) -> go.Figure:
    d = (
        states_d.filter(pl.col("AÑO") == year)
        .with_columns(
            pl.col("ENTIDAD").replace({"Estado de México": "México"}).alias("geo_name")
        )
    )
    fig = px.choropleth_map(
        d,
        geojson=GEO,
        locations="geo_name",
        featureidkey="properties.name",
        color="TASA_TOTAL",
        color_continuous_scale="Reds",
        range_color=[0, 20],
        hover_name="ENTIDAD",
        hover_data={"TASA_TOTAL": ":.2f", "TOTAL": True, "geo_name": False},
        labels={"TASA_TOTAL": "Tasa por 100k"},
        title=f"Tasa de suicidio por entidad — {year}",
        map_style="carto-darkmatter",
        center={"lat": 23.6, "lon": -102.5},
        zoom=4,
    )
    fig.update_layout(
        height=560,
        margin=dict(t=40, b=10, l=0, r=0),
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#CBD5E1",
        coloraxis_colorbar=dict(title="Tasa"),
    )
    return fig


def fig_states_rank(states_d: pl.DataFrame) -> go.Figure:
    agg = (
        states_d.group_by("ENTIDAD")
        .agg(pl.col("TASA_TOTAL").mean().alias("tasa_avg"))
        .sort("tasa_avg")
    )
    fig = px.bar(
        agg, x="tasa_avg", y="ENTIDAD", orientation="h",
        color="tasa_avg", color_continuous_scale="Reds", range_color=[0, 20],
        labels={"tasa_avg": "Tasa promedio por 100k", "ENTIDAD": ""},
        title="Entidades por tasa promedio de suicidio",
    )
    fig.update_layout(
        height=max(300, len(agg) * 24 + 80),
        showlegend=False, coloraxis_showscale=False,
        margin=dict(t=40, b=40, l=10, r=10),
        **CHART_LAYOUT,
    )
    return fig


def compute_kpis(d: pl.DataFrame):
    total  = int(d["TOTAL"].sum())
    rate   = round(float(d["TASA_TOTAL"].drop_nulls().mean()), 2)   if d["TASA_TOTAL"].drop_nulls().len() else None
    rate_h = round(float(d["TASA_HOMBRES"].drop_nulls().mean()), 2) if d["TASA_HOMBRES"].drop_nulls().len() else None
    rate_m = round(float(d["TASA_MUJERES"].drop_nulls().mean()), 2) if d["TASA_MUJERES"].drop_nulls().len() else None
    return total, rate, rate_h, rate_m


def state_extremes(states_d: pl.DataFrame):
    """Return (max_state, max_rate, min_state, min_rate) averaged over the period."""
    agg = (
        states_d.group_by("ENTIDAD")
        .agg(pl.col("TASA_TOTAL").mean().alias("tasa"))
        .sort("tasa", descending=True)
    )
    if agg.is_empty():
        return None, None, None, None
    top = agg.head(1)
    bot = agg.tail(1)
    return (
        top["ENTIDAD"].item(), round(top["tasa"].item(), 1),
        bot["ENTIDAD"].item(), round(bot["tasa"].item(), 1),
    )


# ── layout ────────────────────────────────────────────────────────────────────
app = Dash(__name__, external_stylesheets=[dbc.themes.SLATE], title="Suicidios por Entidad")

app.layout = dbc.Container(
    [
        html.H2("Suicidios por Entidad — México", className="text-light mt-3 mb-1"),
        html.P("Fuente: INEGI · Estadísticas de mortalidad", style={"color": "#64748B", "fontSize": "0.85rem", "marginBottom": "24px"}),

        # Filters
        dbc.Row([
            dbc.Col([
                html.Label("Entidad", className="text-secondary small"),
                dcc.Dropdown(
                    id="entidad-dd",
                    options=ENTIDAD_OPTIONS,
                    value="Nacional",
                    clearable=False,
                    style={"backgroundColor": "#1E293B", "color": "#CBD5E1",
                           "border": "1px solid #334155"},
                ),
            ], width=4),
            dbc.Col([
                html.Label("Periodo", className="text-secondary small"),
                dcc.RangeSlider(
                    id="year-range",
                    min=MIN_YEAR, max=MAX_YEAR,
                    value=[MIN_YEAR, MAX_YEAR],
                    marks=YEAR_MARKS,
                    step=1,
                    tooltip={"placement": "bottom", "always_visible": False},
                ),
            ], width=8),
        ], className="mb-4 align-items-center"),

        # KPI cards
        dbc.Row(id="kpi-row", className="mb-3 g-3"),

        # State extremes callout
        dbc.Row(id="extremes-row", className="mb-4 g-3"),

        # Trend + gender bar
        dbc.Row([
            dbc.Col(dcc.Graph(id="trend-chart"), width=7),
            dbc.Col(dcc.Graph(id="gender-bar"),  width=5),
        ], className="mb-4"),

        # Map + state ranking
        dbc.Row([
            dbc.Col(dcc.Graph(id="map-chart"),   width=6),
            dbc.Col(dcc.Graph(id="states-rank"), width=6),
        ]),
    ],
    fluid=True,
    style={"backgroundColor": "#0F172A", "minHeight": "100vh", "padding": "20px"},
)


# ── callback ──────────────────────────────────────────────────────────────────
@app.callback(
    Output("kpi-row",      "children"),
    Output("extremes-row", "children"),
    Output("trend-chart",  "figure"),
    Output("gender-bar",   "figure"),
    Output("map-chart",    "figure"),
    Output("states-rank",  "figure"),
    Input("year-range",   "value"),
    Input("entidad-dd",   "value"),
)
def update_all(year_range, entidad):
    y0, y1 = year_range

    d_entity = df.filter(
        pl.col("AÑO").is_between(y0, y1) & (pl.col("ENTIDAD") == entidad)
    )
    d_states = df.filter(
        pl.col("AÑO").is_between(y0, y1)
        & ~pl.col("ENTIDAD").is_in(["Nacional", "Extranjero", "Not specified"])
    )

    total, rate, rate_h, rate_m = compute_kpis(d_entity)
    max_st, max_rate, min_st, min_rate = state_extremes(d_states)

    def kpi_card(label, value, fmt="{:.2f}"):
        val_str = fmt.format(value) if value is not None else "—"
        return dbc.Col(
            html.Div([
                html.P(label, style={"color": "#94A3B8", "fontSize": "0.78rem", "marginBottom": "4px"}),
                html.H4(val_str, style={"color": "#F8FAFC", "fontWeight": "600", "margin": 0}),
            ], style=CARD_STYLE),
            width=3,
        )

    def extreme_card(label, state, rate, accent):
        return dbc.Col(
            html.Div([
                html.P(label, style={"color": "#94A3B8", "fontSize": "0.78rem", "marginBottom": "4px"}),
                html.Span(state or "—", style={"color": accent, "fontWeight": "700", "fontSize": "1rem"}),
                html.Span(f"  {rate:.1f}/100k" if rate else "", style={"color": "#CBD5E1", "fontSize": "0.9rem"}),
            ], style=CARD_STYLE),
            width=6,
        )

    kpis = [
        kpi_card("Total de suicidios", total, fmt="{:,.0f}"),
        kpi_card("Tasa total (prom.)", rate),
        kpi_card("Tasa hombres (prom.)", rate_h),
        kpi_card("Tasa mujeres (prom.)", rate_m),
    ]
    extremes = [
        extreme_card("Entidad con mayor tasa (prom. periodo)", max_st, max_rate, "#E84855"),
        extreme_card("Entidad con menor tasa (prom. periodo)", min_st, min_rate, "#3BB273"),
    ]

    return (
        kpis,
        extremes,
        fig_trend(d_entity),
        fig_gender_bar(d_entity),
        fig_map(d_states, y1),
        fig_states_rank(d_states),
    )


if __name__ == "__main__":
    app.run(debug=True, port=8058)
