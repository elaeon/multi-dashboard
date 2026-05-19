import polars as pl
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc

# ── Data loading ──────────────────────────────────────────────────────────────

_RAW = pl.read_csv("data/temperaturas_lluvias.csv")

df = _RAW.with_columns(
    pl.col("PERIODO").str.slice(0, 4).cast(pl.Int32).alias("anio"),
    pl.col("PERIODO").str.slice(5, 2).cast(pl.Int32).alias("mes"),
)

# Separate Nacional (CVE_ENT=0) from states
ESTADOS = sorted(
    df.filter(pl.col("CVE_ENT") != 0)["ENTIDAD"].unique().to_list()
)

MES_NOMBRES = ["Ene", "Feb", "Mar", "Abr", "May", "Jun",
               "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]

# Precompute baseline mean (1985–1994) per entity for warming delta
_baseline = (
    df.filter(pl.col("anio").is_between(1985, 1994))
    .group_by("ENTIDAD")
    .agg(pl.col("MEDIA").mean().alias("baseline_media"))
)

# ── Theme ─────────────────────────────────────────────────────────────────────

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
TAB_SEL   = {"backgroundColor": "#1E293B", "color": "#F8FAFC",
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


# ── Figure factories ──────────────────────────────────────────────────────────

def fig_temp_trend(d: pl.DataFrame) -> go.Figure:
    anual = (
        d.filter(pl.col("anio").is_between(1985, 2025))
        .group_by("anio")
        .agg(
            pl.col("MINIMA").mean().round(2),
            pl.col("MEDIA").mean().round(2),
            pl.col("MAXIMA").mean().round(2),
        )
        .sort("anio")
    )
    years = anual["anio"].to_list()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=years, y=anual["MAXIMA"].to_list(),
        mode="lines", name="Máxima",
        line=dict(color="#E84855", width=1.5),
        fill="tonexty", fillcolor="rgba(232,72,85,0.07)",
    ))
    fig.add_trace(go.Scatter(
        x=years, y=anual["MEDIA"].to_list(),
        mode="lines+markers", name="Media",
        line=dict(color="#F4A261", width=2.5),
        marker=dict(size=4),
    ))
    fig.add_trace(go.Scatter(
        x=years, y=anual["MINIMA"].to_list(),
        mode="lines", name="Mínima",
        line=dict(color="#2E86AB", width=1.5),
    ))
    fig.update_layout(
        title="Temperatura anual (1985–2025)",
        height=380,
        xaxis=dict(gridcolor="#334155", title="Año", dtick=5),
        yaxis=dict(gridcolor="#334155", ticksuffix="°C"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font_color="#94A3B8",
                    orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        **{k: v for k, v in CHART_LAYOUT.items() if k not in ("xaxis", "yaxis", "margin")},
        margin=dict(t=50, b=40, l=10, r=10),
    )
    return fig


def fig_temp_seasonality(d: pl.DataFrame) -> go.Figure:
    seasonal = (
        d.group_by("mes")
        .agg(
            pl.col("MINIMA").mean().round(1),
            pl.col("MEDIA").mean().round(1),
            pl.col("MAXIMA").mean().round(1),
        )
        .sort("mes")
    )
    meses = [MES_NOMBRES[m - 1] for m in seasonal["mes"].to_list()]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=meses, y=seasonal["MAXIMA"].to_list(),
        name="Máxima", marker_color="#E84855", opacity=0.7,
    ))
    fig.add_trace(go.Bar(
        x=meses, y=seasonal["MEDIA"].to_list(),
        name="Media", marker_color="#F4A261",
    ))
    fig.add_trace(go.Bar(
        x=meses, y=seasonal["MINIMA"].to_list(),
        name="Mínima", marker_color="#2E86AB", opacity=0.7,
    ))
    fig.update_layout(
        barmode="group",
        title="Temperatura promedio por mes",
        height=360,
        xaxis=dict(gridcolor="#334155"),
        yaxis=dict(gridcolor="#334155", ticksuffix="°C"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font_color="#94A3B8",
                    orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        **{k: v for k, v in CHART_LAYOUT.items() if k not in ("xaxis", "yaxis", "margin")},
        margin=dict(t=50, b=40, l=10, r=10),
    )
    return fig


def fig_warming_delta() -> go.Figure:
    """Static chart — shows all states, not filtered."""
    recent = (
        df.filter(pl.col("CVE_ENT") != 0)
        .filter(pl.col("anio").is_between(2015, 2024))
        .group_by("ENTIDAD")
        .agg(pl.col("MEDIA").mean().alias("recent"))
    )
    result = (
        _baseline.filter(pl.col("ENTIDAD") != "Nacional")
        .join(recent, on="ENTIDAD")
        .with_columns((pl.col("recent") - pl.col("baseline_media")).round(2).alias("delta"))
        .sort("delta", descending=True)
    )
    deltas = result["delta"].to_list()
    estados_list = result["ENTIDAD"].to_list()
    colors = ["#E84855" if d >= 2 else "#F4A261" if d >= 1 else "#3BB273" for d in deltas]
    fig = go.Figure(go.Bar(
        x=deltas, y=estados_list, orientation="h",
        marker_color=colors,
        text=[f"+{d:.2f}°C" for d in deltas],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Δ temperatura: +%{x:.2f}°C<extra></extra>",
    ))
    fig.update_layout(
        title="Calentamiento por estado: promedio 2015–2024 vs 1985–1994",
        height=max(340, len(estados_list) * 28 + 80),
        xaxis=dict(range=[0, 4.2], gridcolor="#334155", ticksuffix="°C",
                   title="Δ Temperatura media (°C)"),
        yaxis=dict(gridcolor="rgba(0,0,0,0)", autorange="reversed"),
        **{k: v for k, v in CHART_LAYOUT.items() if k not in ("xaxis", "yaxis", "margin")},
        margin=dict(t=40, b=40, l=10, r=80),
    )
    return fig


def fig_prec_trend(d: pl.DataFrame) -> go.Figure:
    anual = (
        d.filter(pl.col("anio").is_between(1985, 2025))
        .group_by("anio")
        .agg(pl.col("PRECIPITACION").sum().round(1).alias("prec"))
        .sort("anio")
    )
    years = anual["anio"].to_list()
    precs = anual["prec"].to_list()
    avg   = sum(precs) / len(precs)
    colors = ["#2E86AB" if p >= avg else "#F4A261" for p in precs]
    fig = go.Figure()
    fig.add_hrect(y0=avg - 30, y1=avg + 30,
                  fillcolor="rgba(59,178,115,0.06)", line_width=0)
    fig.add_hline(y=avg, line=dict(color="#3BB273", width=1.5, dash="dash"),
                  annotation_text=f"Promedio {avg:.0f}mm",
                  annotation_font_color="#3BB273",
                  annotation_position="top right")
    fig.add_trace(go.Bar(
        x=years, y=precs,
        marker_color=colors,
        hovertemplate="<b>%{x}</b><br>Precipitación: %{y:.0f}mm<extra></extra>",
    ))
    fig.update_layout(
        title="Precipitación anual total (1985–2025)",
        height=380,
        xaxis=dict(gridcolor="#334155", title="Año", dtick=5),
        yaxis=dict(gridcolor="#334155", ticksuffix="mm"),
        showlegend=False,
        **{k: v for k, v in CHART_LAYOUT.items() if k not in ("xaxis", "yaxis", "margin")},
        margin=dict(t=50, b=40, l=10, r=10),
    )
    return fig


def fig_prec_seasonality(d: pl.DataFrame) -> go.Figure:
    seasonal = (
        d.group_by("mes")
        .agg(pl.col("PRECIPITACION").mean().round(1))
        .sort("mes")
    )
    meses = [MES_NOMBRES[m - 1] for m in seasonal["mes"].to_list()]
    precs = seasonal["PRECIPITACION"].to_list()
    max_p = max(precs)
    colors = [
        f"rgba(46,134,171,{0.4 + 0.6 * (p / max_p):.2f})" for p in precs
    ]
    fig = go.Figure(go.Bar(
        x=meses, y=precs,
        marker_color=colors,
        text=[f"{p:.0f}" for p in precs],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Precipitación: %{y:.1f}mm<extra></extra>",
    ))
    fig.update_layout(
        title="Precipitación mensual promedio",
        height=360,
        xaxis=dict(gridcolor="#334155"),
        yaxis=dict(gridcolor="#334155", ticksuffix="mm"),
        showlegend=False,
        **{k: v for k, v in CHART_LAYOUT.items() if k not in ("xaxis", "yaxis", "margin")},
        margin=dict(t=40, b=40, l=10, r=10),
    )
    return fig


def fig_prec_state_ranking() -> go.Figure:
    """Static chart — all states ranked by avg monthly precipitation."""
    agg = (
        df.filter(pl.col("CVE_ENT") != 0)
        .group_by("ENTIDAD")
        .agg(pl.col("PRECIPITACION").mean().round(1).alias("avg"))
        .sort("avg", descending=True)
    )
    states = agg["ENTIDAD"].to_list()
    avgs   = agg["avg"].to_list()
    max_p  = max(avgs)
    colors = [f"rgba(46,134,171,{0.35 + 0.65 * (p / max_p):.2f})" for p in avgs]
    fig = go.Figure(go.Bar(
        x=avgs, y=states, orientation="h",
        marker_color=colors,
        text=[f"{p:.0f}mm" for p in avgs],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Precipitación media: %{x:.1f}mm/mes<extra></extra>",
    ))
    fig.update_layout(
        title="Precipitación media mensual por estado",
        height=max(340, len(states) * 28 + 80),
        xaxis=dict(gridcolor="#334155", ticksuffix="mm"),
        yaxis=dict(gridcolor="rgba(0,0,0,0)", autorange="reversed"),
        **{k: v for k, v in CHART_LAYOUT.items() if k not in ("xaxis", "yaxis", "margin")},
        margin=dict(t=40, b=40, l=10, r=80),
    )
    return fig


def compute_kpis(d: pl.DataFrame, entidad: str):
    recent = d.filter(pl.col("anio").is_between(2015, 2025))
    media_r = float(recent["MEDIA"].mean())
    maxima_r = float(recent["MAXIMA"].mean())
    prec_anual = float(d.filter(pl.col("anio").is_between(1985, 2025))
                       .group_by("anio").agg(pl.col("PRECIPITACION").sum())["PRECIPITACION"].mean())
    base = _baseline.filter(pl.col("ENTIDAD") == entidad)
    if len(base) > 0:
        delta = media_r - float(base["baseline_media"][0])
    else:
        delta = float("nan")
    return media_r, maxima_r, prec_anual, delta


# ── Pre-render static charts ──────────────────────────────────────────────────

_fig_warming = fig_warming_delta()
_fig_prec_ranking = fig_prec_state_ranking()


# ── Layout ────────────────────────────────────────────────────────────────────

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP],
           title="Temperaturas y Lluvias México")

estado_options = (
    [{"label": "Nacional", "value": "Nacional"}]
    + [{"label": e, "value": e} for e in ESTADOS]
)

app.layout = html.Div(
    style={"backgroundColor": "#0F172A", "minHeight": "100vh", "padding": "24px"},
    children=[
        html.H2("Temperaturas y Precipitaciones — México 1985–2025",
                style={"color": "#F8FAFC", "fontWeight": "700", "marginBottom": "4px"}),
        html.P("Temperatura mínima, media y máxima, y precipitación mensual por estado",
               style={"color": "#94A3B8", "marginBottom": "24px"}),

        # Filter
        dbc.Row(style={"marginBottom": "20px"}, children=[
            dbc.Col([
                html.Label("Estado / Región", style={"color": "#94A3B8", "fontSize": "12px"}),
                dcc.Dropdown(
                    id="estado-filter",
                    options=estado_options,
                    value="Nacional",
                    clearable=False,
                    style={"backgroundColor": "#1E293B", "color": "#CBD5E1",
                           "border": "1px solid #334155"},
                ),
            ], md=4),
        ]),

        # KPI row
        dbc.Row(id="kpi-row", className="g-3", style={"marginBottom": "20px"}),

        # Tabs
        dcc.Tabs(style={"marginBottom": "0"}, children=[

            dcc.Tab(label="Temperatura", style=TAB_STYLE, selected_style=TAB_SEL, children=[
                html.Div(style={"paddingTop": "20px"}, children=[
                    dbc.Row(className="g-3", style={"marginBottom": "20px"}, children=[
                        dbc.Col(dcc.Graph(id="graph-temp-trend"), md=7),
                        dbc.Col(dcc.Graph(id="graph-temp-seasonal"), md=5),
                    ]),
                    dbc.Row(className="g-3", children=[
                        dbc.Col(dcc.Graph(id="graph-warming-delta"), md=12),
                    ]),
                ]),
            ]),

            dcc.Tab(label="Precipitación", style=TAB_STYLE, selected_style=TAB_SEL, children=[
                html.Div(style={"paddingTop": "20px"}, children=[
                    dbc.Row(className="g-3", style={"marginBottom": "20px"}, children=[
                        dbc.Col(dcc.Graph(id="graph-prec-trend"), md=7),
                        dbc.Col(dcc.Graph(id="graph-prec-seasonal"), md=5),
                    ]),
                    dbc.Row(className="g-3", children=[
                        dbc.Col(dcc.Graph(id="graph-prec-ranking"), md=12),
                    ]),
                ]),
            ]),
        ]),
    ],
)


# ── Callbacks ─────────────────────────────────────────────────────────────────

@app.callback(
    Output("kpi-row", "children"),
    Output("graph-temp-trend", "figure"),
    Output("graph-temp-seasonal", "figure"),
    Output("graph-prec-trend", "figure"),
    Output("graph-prec-seasonal", "figure"),
    Input("estado-filter", "value"),
)
def update_all(estado: str):
    d = df.filter(pl.col("ENTIDAD") == estado)
    media, maxima, prec_anual, delta = compute_kpis(d, estado)
    delta_sign = "+" if delta >= 0 else ""
    kpis = [
        kpi_card("Temp. media (2015–2025)", f"{media:.1f}°C", "#F4A261"),
        kpi_card("Temp. máxima (2015–2025)", f"{maxima:.1f}°C", "#E84855"),
        kpi_card("Precipitación anual promedio", f"{prec_anual:.0f}mm", "#2E86AB"),
        kpi_card("Calentamiento vs 1985–1994", f"{delta_sign}{delta:.2f}°C",
                 "#E84855" if delta >= 2 else "#F4A261" if delta >= 1 else "#3BB273"),
    ]
    return (
        kpis,
        fig_temp_trend(d),
        fig_temp_seasonality(d),
        fig_prec_trend(d),
        fig_prec_seasonality(d),
    )


@app.callback(
    Output("graph-warming-delta", "figure"),
    Output("graph-prec-ranking", "figure"),
    Input("estado-filter", "value"),
)
def update_static(_):
    return _fig_warming, _fig_prec_ranking


if __name__ == "__main__":
    app.run(debug=True, port=8056)
