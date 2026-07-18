import json

import numpy as np
import polars as pl
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc

# ── Data loading ───────────────────────────────────────────────────────────────

df_us_annual = pl.read_parquet("dashboard_data/meta_us_annual.parquet")
df_us_state = pl.read_parquet("dashboard_data/meta_us_state_year.parquet")
df_mx_annual = pl.read_parquet("dashboard_data/meta_mx_annual.parquet")
df_mx_state = pl.read_parquet("dashboard_data/meta_mx_state_year.parquet")

with open("data/mexico_states.geojson") as _f:
    MEXICO_GEO = json.load(_f)

NAME_MAP = {
    "Coahuila de Zaragoza": "Coahuila",
    "Michoacán de Ocampo": "Michoacán",
    "Veracruz de Ignacio de la Llave": "Veracruz",
}

US_MIN_YEAR, US_MAX_YEAR = int(df_us_annual["year"].min()), int(df_us_annual["year"].max())
MX_MIN_YEAR, MX_MAX_YEAR = int(df_mx_annual["año"].min()), int(df_mx_annual["año"].max())

WINDOW_COLORS = {
    "reliable": "#2E86AB",
    "gap_2019": "#64748B",
    "partial_recent": "#64748B",
    "two_state_2000_03": "#475569",
}


# ── Derived headline stats (computed from data, not hardcoded) ─────────────────

def _log_linear_cagr(years: list[int], values: list[float]) -> tuple[float, float]:
    years_a = np.array(years, dtype=float)
    vals_a = np.array(values, dtype=float)
    mask = vals_a > 0
    years_a, vals_a = years_a[mask], vals_a[mask]
    log_vals = np.log(vals_a)
    slope, intercept = np.polyfit(years_a, log_vals, 1)
    pred = np.polyval([slope, intercept], years_a)
    ss_res = float(np.sum((log_vals - pred) ** 2))
    ss_tot = float(np.sum((log_vals - np.mean(log_vals)) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return (float(np.exp(slope)) - 1) * 100, r2


_us_reliable = df_us_annual.filter(pl.col("window_flag") == "reliable")
US_CAGR, US_R2 = _log_linear_cagr(_us_reliable["year"].to_list(), _us_reliable["n_labs"].to_list())
_us_peak_row = df_us_annual.sort("n_labs", descending=True)
US_PEAK_YEAR = int(_us_peak_row["year"][0])
US_PEAK_N = int(_us_peak_row["n_labs"][0])

_mx_pre = df_mx_annual.filter(pl.col("año").is_between(2000, 2008))["kg_met_sedena"]
_mx_post = df_mx_annual.filter(pl.col("año").is_between(2009, 2021))["kg_met_sedena"]
MX_PRE_AVG = float(_mx_pre.mean())
MX_POST_AVG = float(_mx_post.mean())
MX_MULTIPLIER = MX_POST_AVG / MX_PRE_AVG if MX_PRE_AVG > 0 else 0.0

# Change-point: first year the series permanently leaves its prior near-zero baseline
# (max-abs-delta picks a later noisy swing instead, since absolute deltas grow with
# the series level — so use a running-max-ratio threshold to find the level shift).
def _level_shift_year(years: list[int], values: list[float], start_year: int, ratio: float = 10.0) -> int:
    running_max = 0.0
    for y, v in zip(years, values):
        if y < start_year:
            running_max = max(running_max, v)
            continue
        if v / max(running_max, 1.0) >= ratio:
            return y
        running_max = max(running_max, v)
    return years[-1]


_mx_sorted = df_mx_annual.sort("año")
MX_BREAK_YEAR = _level_shift_year(_mx_sorted["año"].to_list(), _mx_sorted["kg_met_sedena"].to_list(), start_year=2001)

_us_totals = df_us_state.group_by("state").agg(pl.col("n_labs").sum().alias("total")).sort("total", descending=True)
US_TOP_STATE = _us_totals["state"][0]

_mx_totals_recent = (
    df_mx_state.filter(pl.col("año") >= MX_BREAK_YEAR)
    .group_by("entidad").agg(pl.col("kg_met_sedena").sum().alias("total"))
    .sort("total", descending=True)
)
MX_TOP_STATE = _mx_totals_recent["entidad"][0]
MX_TOP5_SHARE = float(_mx_totals_recent.head(5)["total"].sum() / _mx_totals_recent["total"].sum() * 100)


# ── Theme (reused from dashboard/desaparecidos.py) ──────────────────────────────

CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font_color="#CBD5E1",
    xaxis=dict(gridcolor="#334155"),
    yaxis=dict(gridcolor="#334155"),
    margin=dict(t=40, b=40, l=10, r=10),
)
CARD_STYLE = {
    "background": "#1E293B", "border": "1px solid #334155",
    "borderRadius": "8px", "padding": "16px", "textAlign": "center",
}
TAB_STYLE = {"backgroundColor": "#0F172A", "color": "#94A3B8", "borderTop": "none",
             "borderBottom": "1px solid #334155", "padding": "10px 18px"}
TAB_SEL = {"backgroundColor": "#1E293B", "color": "#F8FAFC",
           "borderTop": "2px solid #2E86AB", "borderBottom": "none",
           "fontWeight": "600", "padding": "10px 18px"}


def kpi_card(title: str, value: str, color: str = "#CBD5E1") -> dbc.Col:
    return dbc.Col(
        html.Div([
            html.P(title, style={"color": "#94A3B8", "fontSize": "12px", "margin": 0}),
            html.H3(value, style={"color": color, "margin": "4px 0 0"}),
        ], style=CARD_STYLE),
        xs=12, sm=6, md=3,
    )


# ── Figure factories ─────────────────────────────────────────────────────────────

def fig_headline(us: pl.DataFrame, mx: pl.DataFrame) -> go.Figure:
    us_f = us.filter(pl.col("year") <= 2023).sort("year")
    mx_f = mx.filter(pl.col("año") <= 2021).sort("año")

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
        subplot_titles=[
            "Laboratorios clandestinos desmantelados (EU)",
            "Metanfetamina decomisada por SEDENA (México)",
        ],
    )

    fig.add_trace(go.Scatter(
        x=us_f["year"], y=us_f["n_labs"], mode="lines",
        line=dict(color="#475569", width=1, dash="dot"),
        hoverinfo="skip", showlegend=False,
    ), row=1, col=1)
    reliable = us_f.filter(pl.col("window_flag") == "reliable")
    fig.add_trace(go.Scatter(
        x=reliable["year"], y=reliable["n_labs"], mode="lines+markers",
        line=dict(color="#2E86AB", width=2), marker=dict(size=5),
        showlegend=False,
        hovertemplate="<b>%{x}</b><br>%{y:,} laboratorios<extra></extra>",
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=mx_f["año"], y=mx_f["kg_met_sedena"], marker_color="#E84855",
        showlegend=False,
        hovertemplate="<b>%{x}</b><br>%{y:,.0f} kg<extra></extra>",
    ), row=2, col=1)

    fig.add_vrect(
        x0=2005, x1=2009, fillcolor="rgba(244,162,97,0.14)", line_width=0,
        annotation_text="Ley CMEA en EU / colapso de precursores",
        annotation_font_color="#94A3B8", annotation_position="top left",
        row=1, col=1,
    )
    fig.add_vline(
        x=MX_BREAK_YEAR, line_dash="dot", line_color="#94A3B8",
        annotation_text=f"{MX_BREAK_YEAR}: quiebre en decomisos MX",
        annotation_font_color="#94A3B8", annotation_position="top left",
        row=2, col=1,
    )

    fig.update_xaxes(gridcolor="#334155", range=[2000, 2023])
    fig.update_yaxes(gridcolor="#334155", title_text="Laboratorios/año", row=1, col=1)
    fig.update_yaxes(gridcolor="#334155", title_text="kg/año", row=2, col=1)
    fig.update_layout(
        height=620, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#CBD5E1", showlegend=False,
        margin=dict(t=60, b=30, l=10, r=10),
    )
    for ann in fig.layout.annotations[:2]:
        ann.font = dict(color="#CBD5E1", size=14)
    return fig


def fig_us_trend(d: pl.DataFrame) -> go.Figure:
    d = d.sort("year")
    if len(d) == 0:
        return go.Figure()
    colors = [WINDOW_COLORS.get(w, "#2E86AB") for w in d["window_flag"].to_list()]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=d["year"], y=d["n_labs"], marker_color=colors, name="Laboratorios",
        hovertemplate="<b>%{x}</b><br>%{y:,} laboratorios<extra></extra>",
    ))
    reliable = d.filter(pl.col("window_flag") == "reliable")
    if len(reliable) >= 2:
        cagr, _ = _log_linear_cagr(reliable["year"].to_list(), reliable["n_labs"].to_list())
        years = reliable["year"].to_list()
        slope, intercept = np.polyfit(years, np.log(reliable["n_labs"].to_list()), 1)
        trend_y = np.exp(np.polyval([slope, intercept], years))
        fig.add_trace(go.Scatter(
            x=years, y=trend_y, mode="lines",
            line=dict(color="#F4A261", dash="dash", width=2),
            name=f"Tendencia: {cagr:.1f}%/año", hoverinfo="skip",
        ))
    if d["year"].min() <= 2019 <= d["year"].max():
        fig.add_vline(x=2019, line_dash="dot", line_color="#94A3B8",
                      annotation_text="hueco de reporte", annotation_font_color="#94A3B8")
    fig.update_layout(
        title="Desmantelamientos de laboratorios clandestinos por año",
        height=400, showlegend=True,
        legend=dict(orientation="h", y=1.15, x=0),
        xaxis=dict(gridcolor="#334155"),
        yaxis=dict(gridcolor="#334155", title="Laboratorios"),
        **{k: v for k, v in CHART_LAYOUT.items() if k not in ("xaxis", "yaxis", "margin")},
        margin=dict(t=70, b=40, l=10, r=10),
    )
    return fig


def fig_us_map(d: pl.DataFrame) -> go.Figure:
    agg = d.group_by("state").agg(pl.col("n_labs").sum().alias("n_labs"))
    if len(agg) == 0:
        return go.Figure()
    fig = px.choropleth(
        agg, locations="state", locationmode="USA-states", scope="usa",
        color="n_labs", color_continuous_scale="YlOrRd", hover_name="state",
    )
    fig.update_traces(hovertemplate="<b>%{location}</b><br>Laboratorios: %{z:,}<extra></extra>")
    fig.update_coloraxes(colorbar=dict(
        title=dict(text="Laboratorios", font=dict(color="#CBD5E1")),
        tickfont=dict(color="#CBD5E1"),
    ))
    fig.update_layout(
        title="Laboratorios desmantelados por estado",
        height=520, margin=dict(l=0, r=0, t=40, b=0),
        paper_bgcolor="rgba(0,0,0,0)", font_color="#CBD5E1",
        geo=dict(bgcolor="rgba(0,0,0,0)", lakecolor="#0F172A",
                 landcolor="#1E293B", subunitcolor="#334155"),
    )
    return fig


def fig_us_ranking(d: pl.DataFrame) -> go.Figure:
    agg = d.group_by("state").agg(pl.col("n_labs").sum().alias("n_labs")).sort("n_labs", descending=True).head(10)
    if len(agg) == 0:
        return go.Figure()
    states = agg["state"].to_list()
    ns = agg["n_labs"].to_list()
    total = int(d["n_labs"].sum())
    colors = ["#2E86AB" if s == states[0] else "#475569" for s in states]
    fig = go.Figure(go.Bar(
        x=ns[::-1], y=states[::-1], orientation="h", marker_color=colors[::-1],
        text=[f"{n:,} ({n / total * 100:.1f}%)" for n in ns[::-1]], textposition="outside",
        hovertemplate="<b>%{y}</b><br>%{x:,} laboratorios<extra></extra>",
    ))
    fig.update_layout(
        title=f"{states[0]} lidera — no Arkansas ni Oklahoma",
        height=max(320, len(states) * 28 + 100),
        xaxis=dict(gridcolor="#334155"),
        yaxis=dict(gridcolor="rgba(0,0,0,0)"),
        showlegend=False,
        **{k: v for k, v in CHART_LAYOUT.items() if k not in ("xaxis", "yaxis", "margin")},
        margin=dict(t=50, b=40, l=10, r=110),
    )
    return fig


def fig_mx_trend(d: pl.DataFrame) -> go.Figure:
    d = d.sort("año")
    if len(d) == 0:
        return go.Figure()
    fig = go.Figure(go.Bar(
        x=d["año"], y=d["kg_met_sedena"], marker_color="#E84855",
        hovertemplate="<b>%{x}</b><br>%{y:,.0f} kg<extra></extra>",
    ))
    if d["año"].min() <= MX_BREAK_YEAR <= d["año"].max():
        fig.add_vline(
            x=MX_BREAK_YEAR, line_dash="dot", line_color="#94A3B8",
            annotation_text=f"{MX_BREAK_YEAR}: quiebre — colapso de producción en EU",
            annotation_font_color="#94A3B8", annotation_position="top left",
        )
    fig.update_layout(
        title="Metanfetamina decomisada por SEDENA, por año",
        height=400,
        xaxis=dict(gridcolor="#334155"),
        yaxis=dict(gridcolor="#334155", title="kg/año"),
        showlegend=False,
        **{k: v for k, v in CHART_LAYOUT.items() if k not in ("xaxis", "yaxis", "margin")},
        margin=dict(t=50, b=40, l=10, r=10),
    )
    return fig


def fig_mx_map(d: pl.DataFrame) -> go.Figure:
    agg = d.group_by("entidad").agg(pl.col("kg_met_sedena").sum().alias("kg"))
    if len(agg) == 0:
        return go.Figure()
    state_data = pl.DataFrame({
        "entidad": [NAME_MAP.get(e, e) for e in agg["entidad"].to_list()],
        "kg": agg["kg"].to_list(),
    })
    fig = px.choropleth_map(
        state_data, geojson=MEXICO_GEO, locations="entidad", color="kg",
        featureidkey="properties.name", color_continuous_scale="YlOrRd",
        zoom=4.0, center={"lat": 23.6, "lon": -102.5}, opacity=0.85,
        hover_name="entidad", map_style="carto-darkmatter",
    )
    fig.update_traces(hovertemplate="<b>%{hovertext}</b><br>%{z:,.0f} kg<extra></extra>")
    fig.update_coloraxes(colorbar=dict(
        title=dict(text="kg", font=dict(color="#CBD5E1")),
        tickfont=dict(color="#CBD5E1"),
    ))
    fig.update_layout(
        title="Metanfetamina decomisada por estado (SEDENA)",
        height=520, margin=dict(l=0, r=0, t=40, b=0),
        paper_bgcolor="rgba(0,0,0,0)", font_color="#CBD5E1",
    )
    return fig


def fig_mx_ranking(d: pl.DataFrame) -> go.Figure:
    agg = d.group_by("entidad").agg(pl.col("kg_met_sedena").sum().alias("kg")).sort("kg", descending=True).head(10)
    if len(agg) == 0:
        return go.Figure()
    states = agg["entidad"].to_list()
    kgs = agg["kg"].to_list()
    total = float(d["kg_met_sedena"].sum())
    top5_share = float(agg.head(5)["kg"].sum() / total * 100) if total else 0.0
    colors = ["#E84855" if s == states[0] else "#475569" for s in states]
    fig = go.Figure(go.Bar(
        x=kgs[::-1], y=states[::-1], orientation="h", marker_color=colors[::-1],
        text=[f"{k:,.0f} kg ({k / total * 100:.1f}%)" if total else "" for k in kgs[::-1]],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>%{x:,.0f} kg<extra></extra>",
    ))
    fig.update_layout(
        title=f"{states[0]}: los 5 estados top concentran {top5_share:.1f}%",
        height=max(320, len(states) * 28 + 100),
        xaxis=dict(gridcolor="#334155"),
        yaxis=dict(gridcolor="rgba(0,0,0,0)"),
        showlegend=False,
        **{k: v for k, v in CHART_LAYOUT.items() if k not in ("xaxis", "yaxis", "margin")},
        margin=dict(t=50, b=40, l=10, r=120),
    )
    return fig


# ── Layout ────────────────────────────────────────────────────────────────────

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP],
           title="El giro transnacional de la metanfetamina")

US_YEAR_MARKS = {y: str(y) for y in range(US_MIN_YEAR, US_MAX_YEAR + 1, 5)}
US_YEAR_MARKS[US_MAX_YEAR] = str(US_MAX_YEAR)
MX_YEAR_MARKS = {y: str(y) for y in range(MX_MIN_YEAR, MX_MAX_YEAR + 1, 5)}
MX_YEAR_MARKS[MX_MAX_YEAR] = str(MX_MAX_YEAR)

CAVEAT_BOX = html.Div(
    html.P([
        html.Strong("Nota: ", style={"color": "#F4A261"}),
        html.Span(
            "Este panel muestra una coincidencia temporal entre dos series de datos, no una prueba "
            "de causalidad. El colapso de laboratorios en EU (controles de precursores desde 2005–2006) "
            "y el salto de decomisos de metanfetamina en México ocurren en la misma ventana, pero otros "
            "factores (rutas de tráfico, políticas de cada país) también influyen. Las cifras de SEDENA "
            "en este archivo son un reprocesamiento de MUCD y difieren de las cifras oficiales del gobierno "
            "en ~1.77× (mediana) — no se deben citar como tonelaje oficial. 2019 (EU) y 2022 (México) están "
            "incompletos y se excluyen de las tendencias; 2000–2003 (EU) solo tiene 2 estados reportando.",
            style={"color": "#94A3B8", "fontSize": "13px"},
        ),
    ], style={"margin": 0}),
    style={"background": "#1E293B", "border": "1px solid #334155",
           "borderRadius": "8px", "padding": "12px 16px", "marginTop": "20px"},
)

app.layout = html.Div(
    style={"backgroundColor": "#0F172A", "minHeight": "100vh", "padding": "24px"},
    children=[
        html.H2("El giro transnacional de la metanfetamina",
                style={"color": "#F8FAFC", "fontWeight": "700", "marginBottom": "4px"}),
        html.P(
            "Desmantelamiento de laboratorios clandestinos en EU (DEA) vs. decomisos de SEDENA en "
            "México (MUCD), 2000–2023",
            style={"color": "#94A3B8", "marginBottom": "24px"},
        ),

        dcc.Tabs(children=[

            dcc.Tab(label="Panorama", style=TAB_STYLE, selected_style=TAB_SEL, children=[
                html.Div(style={"paddingTop": "20px"}, children=[
                    dbc.Row(className="g-3", style={"marginBottom": "20px"}, children=[
                        kpi_card(f"Pico de laboratorios en EU ({US_PEAK_YEAR})", f"{US_PEAK_N:,}", "#2E86AB"),
                        kpi_card("Caída de laboratorios en EU (2004–2023)", f"{US_CAGR:.1f}%/año", "#2E86AB"),
                        kpi_card("Salto en decomisos de meta en México", f"{MX_MULTIPLIER:.0f}×", "#E84855"),
                        kpi_card("Año de quiebre en México", f"{MX_BREAK_YEAR}", "#E84855"),
                    ]),
                    dbc.Row(dbc.Col(dcc.Graph(figure=fig_headline(df_us_annual, df_mx_annual)), md=12)),
                    CAVEAT_BOX,
                ]),
            ]),

            dcc.Tab(label="Estados Unidos", style=TAB_STYLE, selected_style=TAB_SEL, children=[
                html.Div(style={"paddingTop": "20px"}, children=[
                    dbc.Row(className="mb-4 align-items-center", children=[
                        dbc.Col([
                            html.Label("Rango de años", style={"color": "#94A3B8", "fontSize": "12px"}),
                            dcc.RangeSlider(
                                id="us-year-range",
                                min=US_MIN_YEAR, max=US_MAX_YEAR, value=[2004, 2023],
                                marks=US_YEAR_MARKS, step=1,
                                tooltip={"placement": "bottom", "always_visible": False},
                            ),
                        ], md=8),
                    ]),
                    dbc.Row(className="g-3 mb-3", children=[dbc.Col(dcc.Graph(id="graph-us-trend"), md=12)]),
                    dbc.Row(className="g-3 mb-3", children=[dbc.Col(dcc.Graph(id="graph-us-map"), md=12)]),
                    dbc.Row(className="g-3", children=[dbc.Col(dcc.Graph(id="graph-us-ranking"), md=12)]),
                ]),
            ]),

            dcc.Tab(label="México", style=TAB_STYLE, selected_style=TAB_SEL, children=[
                html.Div(style={"paddingTop": "20px"}, children=[
                    dbc.Row(className="mb-4 align-items-center", children=[
                        dbc.Col([
                            html.Label("Rango de años", style={"color": "#94A3B8", "fontSize": "12px"}),
                            dcc.RangeSlider(
                                id="mx-year-range",
                                min=MX_MIN_YEAR, max=MX_MAX_YEAR, value=[2000, MX_MAX_YEAR],
                                marks=MX_YEAR_MARKS, step=1,
                                tooltip={"placement": "bottom", "always_visible": False},
                            ),
                        ], md=8),
                    ]),
                    dbc.Row(className="g-3 mb-3", children=[dbc.Col(dcc.Graph(id="graph-mx-trend"), md=12)]),
                    dbc.Row(className="g-3 mb-3", children=[dbc.Col(dcc.Graph(id="graph-mx-map"), md=12)]),
                    dbc.Row(className="g-3", children=[dbc.Col(dcc.Graph(id="graph-mx-ranking"), md=12)]),
                ]),
            ]),
        ]),
    ],
)


# ── Callbacks ─────────────────────────────────────────────────────────────────

@app.callback(
    Output("graph-us-trend", "figure"),
    Output("graph-us-map", "figure"),
    Output("graph-us-ranking", "figure"),
    Input("us-year-range", "value"),
)
def update_us(year_range):
    y0, y1 = year_range
    d_annual = df_us_annual.filter(pl.col("year").is_between(y0, y1))
    d_state = df_us_state.filter(pl.col("year").is_between(y0, y1))
    return fig_us_trend(d_annual), fig_us_map(d_state), fig_us_ranking(d_state)


@app.callback(
    Output("graph-mx-trend", "figure"),
    Output("graph-mx-map", "figure"),
    Output("graph-mx-ranking", "figure"),
    Input("mx-year-range", "value"),
)
def update_mx(year_range):
    y0, y1 = year_range
    d_annual = df_mx_annual.filter(pl.col("año").is_between(y0, y1))
    d_state = df_mx_state.filter(pl.col("año").is_between(y0, y1))
    return fig_mx_trend(d_annual), fig_mx_map(d_state), fig_mx_ranking(d_state)


if __name__ == "__main__":
    app.run(debug=True, port=8067)
