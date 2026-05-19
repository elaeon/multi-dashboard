import polars as pl
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc

# ── Data loading & cleaning ───────────────────────────────────────────────────

_RAW = pl.read_csv(
    "data/desaparecidos.csv",
    null_values=["CONFIDENCIAL"],
    infer_schema_length=10000,
)

df = _RAW.with_columns(
    pl.col("FECHA_DESAPARICION").str.slice(0, 4).cast(pl.Int32, strict=False).alias("anio_desap"),
    pl.col("FECHA_DESAPARICION").str.slice(5, 2).cast(pl.Int32, strict=False).alias("mes_desap"),
    pl.col("FECHA_NACIMIENTO").str.slice(0, 4).cast(pl.Int32, strict=False).alias("anio_nac"),
)

# Filter to plausible year range
df = df.with_columns(
    pl.when(pl.col("anio_desap").is_between(1990, 2025))
    .then(pl.col("anio_desap")).otherwise(None).alias("anio_desap")
)

ESTADOS = sorted(df["ENTIDAD"].drop_nulls().unique().to_list())
MES_NOMBRES = ["Ene", "Feb", "Mar", "Abr", "May", "Jun",
               "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]
AGE_LABELS  = ["0–12", "13–17", "18–29", "30–44", "45–59", "60+"]
AGE_RANGES  = [(0, 12), (13, 17), (18, 29), (30, 44), (45, 59), (60, 100)]

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

def fig_trend(d: pl.DataFrame) -> go.Figure:
    anual = (
        d.filter(pl.col("anio_desap").is_not_null())
        .group_by("anio_desap").agg(pl.len().alias("n"))
        .sort("anio_desap")
    )
    if len(anual) == 0:
        return go.Figure()
    years = anual["anio_desap"].to_list()
    ns    = anual["n"].to_list()
    fig = go.Figure(go.Bar(
        x=years, y=ns,
        marker_color="#E84855",
        hovertemplate="<b>%{x}</b><br>Casos: %{y:,}<extra></extra>",
    ))
    fig.update_layout(
        title="Personas desaparecidas por año",
        height=380,
        xaxis=dict(gridcolor="#334155", title="Año"),
        yaxis=dict(gridcolor="#334155", title="Casos registrados"),
        showlegend=False,
        **{k: v for k, v in CHART_LAYOUT.items() if k not in ("xaxis", "yaxis", "margin")},
        margin=dict(t=40, b=40, l=10, r=10),
    )
    return fig


def fig_monthly(d: pl.DataFrame) -> go.Figure:
    monthly = (
        d.filter(pl.col("mes_desap").is_not_null() & pl.col("anio_desap").is_between(2010, 2025))
        .group_by("mes_desap").agg(pl.len().alias("n"))
        .sort("mes_desap")
    )
    if len(monthly) == 0:
        return go.Figure()
    meses = [MES_NOMBRES[m - 1] for m in monthly["mes_desap"].to_list()]
    ns    = monthly["n"].to_list()
    avg   = sum(ns) / len(ns)
    colors = ["#E84855" if n > avg else "#F4A261" for n in ns]
    fig = go.Figure(go.Bar(
        x=meses, y=ns,
        marker_color=colors,
        hovertemplate="<b>%{x}</b><br>Casos: %{y:,}<extra></extra>",
    ))
    fig.update_layout(
        title="Distribución mensual (2010–2025)",
        height=360,
        xaxis=dict(gridcolor="#334155"),
        yaxis=dict(gridcolor="#334155"),
        showlegend=False,
        **{k: v for k, v in CHART_LAYOUT.items() if k not in ("xaxis", "yaxis", "margin")},
        margin=dict(t=40, b=40, l=10, r=10),
    )
    return fig


def fig_sex(d: pl.DataFrame) -> go.Figure:
    counts = (
        d.filter(pl.col("SEXO").is_in(["HOMBRE", "MUJER"]))
        .group_by("SEXO").agg(pl.len().alias("n"))
        .sort("SEXO")
    )
    if len(counts) == 0:
        return go.Figure()
    fig = go.Figure(go.Pie(
        labels=counts["SEXO"].to_list(),
        values=counts["n"].to_list(),
        hole=0.5,
        textinfo="label+percent",
        marker_colors=["#2E86AB", "#F4A261"],
    ))
    fig.update_layout(
        title="Distribución por sexo",
        height=360,
        showlegend=False,
        **CHART_LAYOUT,
    )
    return fig


def fig_age(d: pl.DataFrame) -> go.Figure:
    with_age = d.filter(
        pl.col("anio_nac").is_not_null() &
        pl.col("anio_desap").is_not_null() &
        pl.col("anio_nac").is_between(1930, 2020)
    ).with_columns(
        (pl.col("anio_desap") - pl.col("anio_nac")).alias("edad")
    ).filter(pl.col("edad").is_between(0, 100))

    if len(with_age) == 0:
        return go.Figure()

    counts = []
    for lo, hi in AGE_RANGES:
        n = len(with_age.filter(pl.col("edad").is_between(lo, hi)))
        counts.append(n)

    total = sum(counts)
    pcts = [c / total * 100 if total > 0 else 0 for c in counts]

    fig = go.Figure(go.Bar(
        x=AGE_LABELS, y=pcts,
        marker_color="#F4A261",
        text=[f"{p:.1f}%" for p in pcts],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>%{y:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        title="Distribución por grupo de edad",
        height=360,
        xaxis=dict(gridcolor="#334155", title="Edad al desaparecer"),
        yaxis=dict(gridcolor="#334155", ticksuffix="%", range=[0, max(pcts) * 1.2 + 2]),
        showlegend=False,
        **{k: v for k, v in CHART_LAYOUT.items() if k not in ("xaxis", "yaxis", "margin")},
        margin=dict(t=40, b=40, l=10, r=10),
    )
    return fig


def fig_state_ranking() -> go.Figure:
    """Static — always shows all states."""
    agg = (
        df.filter(pl.col("ENTIDAD").is_not_null())
        .group_by("ENTIDAD").agg(pl.len().alias("n"))
        .sort("n", descending=True)
    )
    states = agg["ENTIDAD"].to_list()
    ns     = agg["n"].to_list()
    max_n  = max(ns)
    colors = [
        f"rgba(232,72,85,{0.35 + 0.65 * (n / max_n):.2f})" for n in ns
    ]
    fig = go.Figure(go.Bar(
        x=ns, y=states, orientation="h",
        marker_color=colors,
        text=[f"{n:,}" for n in ns],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Casos: %{x:,}<extra></extra>",
    ))
    fig.update_layout(
        title="Total de personas desaparecidas por estado",
        height=max(340, len(states) * 28 + 80),
        xaxis=dict(gridcolor="#334155"),
        yaxis=dict(gridcolor="rgba(0,0,0,0)", autorange="reversed"),
        **{k: v for k, v in CHART_LAYOUT.items() if k not in ("xaxis", "yaxis", "margin")},
        margin=dict(t=40, b=40, l=10, r=100),
    )
    return fig


def compute_kpis(d: pl.DataFrame):
    n_total = len(d)
    with_sex = d.filter(pl.col("SEXO").is_in(["HOMBRE", "MUJER"]))
    n_sex = len(with_sex)
    pct_hombre = float((with_sex["SEXO"] == "HOMBRE").sum() / n_sex * 100) if n_sex > 0 else 0
    pct_mujer  = float((with_sex["SEXO"] == "MUJER").sum()  / n_sex * 100) if n_sex > 0 else 0
    anual = (d.filter(pl.col("anio_desap").is_not_null())
             .group_by("anio_desap").agg(pl.len().alias("n")))
    peak_year = int(anual.sort("n", descending=True)["anio_desap"][0]) if len(anual) > 0 else 0
    peak_n    = int(anual.sort("n", descending=True)["n"][0]) if len(anual) > 0 else 0
    return n_total, pct_hombre, pct_mujer, peak_year, peak_n


# ── Pre-render static chart ───────────────────────────────────────────────────

_fig_state_ranking = fig_state_ranking()

# ── Layout ────────────────────────────────────────────────────────────────────

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP],
           title="Personas Desaparecidas México")

estado_options = (
    [{"label": "Todo el país", "value": "__all__"}]
    + [{"label": e.title(), "value": e} for e in ESTADOS]
)

app.layout = html.Div(
    style={"backgroundColor": "#0F172A", "minHeight": "100vh", "padding": "24px"},
    children=[
        html.H2("Personas Desaparecidas — México",
                style={"color": "#F8FAFC", "fontWeight": "700", "marginBottom": "4px"}),
        html.P("Registro Nacional de Personas Desaparecidas y No Localizadas · 133,887 casos",
               style={"color": "#94A3B8", "marginBottom": "24px"}),

        # Filter
        dbc.Row(style={"marginBottom": "20px"}, children=[
            dbc.Col([
                html.Label("Estado", style={"color": "#94A3B8", "fontSize": "12px"}),
                dcc.Dropdown(
                    id="estado-filter",
                    options=estado_options,
                    value="__all__",
                    clearable=False,
                    style={"backgroundColor": "#1E293B", "color": "#CBD5E1",
                           "border": "1px solid #334155"},
                ),
            ], md=4),
        ]),

        # KPI row
        dbc.Row(id="kpi-row", className="g-3", style={"marginBottom": "20px"}),

        # Data note
        html.Div(
            html.P([
                html.Strong("Nota: ", style={"color": "#F4A261"}),
                html.Span(
                    "El 43% de los registros tienen fecha confidencial y no aparecen en las "
                    "gráficas de tendencia. Los totales de KPIs incluyen todos los registros.",
                    style={"color": "#94A3B8", "fontSize": "13px"},
                ),
            ], style={"margin": 0}),
            style={"background": "#1E293B", "border": "1px solid #334155",
                   "borderRadius": "8px", "padding": "12px 16px", "marginBottom": "20px"},
        ),

        # Tabs
        dcc.Tabs(style={"marginBottom": "0"}, children=[

            dcc.Tab(label="Tendencia", style=TAB_STYLE, selected_style=TAB_SEL, children=[
                html.Div(style={"paddingTop": "20px"}, children=[
                    dbc.Row(className="g-3", children=[
                        dbc.Col(dcc.Graph(id="graph-trend"), md=8),
                        dbc.Col(dcc.Graph(id="graph-monthly"), md=4),
                    ]),
                ]),
            ]),

            dcc.Tab(label="Perfil de Víctimas", style=TAB_STYLE, selected_style=TAB_SEL, children=[
                html.Div(style={"paddingTop": "20px"}, children=[
                    dbc.Row(className="g-3", children=[
                        dbc.Col(dcc.Graph(id="graph-sex"), md=4),
                        dbc.Col(dcc.Graph(id="graph-age"), md=8),
                    ]),
                ]),
            ]),

            dcc.Tab(label="Por Estado", style=TAB_STYLE, selected_style=TAB_SEL, children=[
                html.Div(style={"paddingTop": "20px"}, children=[
                    dbc.Row(className="g-3", children=[
                        dbc.Col(dcc.Graph(id="graph-state-ranking"), md=12),
                    ]),
                ]),
            ]),
        ]),
    ],
)


# ── Callbacks ─────────────────────────────────────────────────────────────────

@app.callback(
    Output("kpi-row", "children"),
    Output("graph-trend", "figure"),
    Output("graph-monthly", "figure"),
    Output("graph-sex", "figure"),
    Output("graph-age", "figure"),
    Input("estado-filter", "value"),
)
def update_all(estado: str):
    d = df if estado == "__all__" else df.filter(pl.col("ENTIDAD") == estado)
    n_total, pct_h, pct_m, peak_year, peak_n = compute_kpis(d)
    kpis = [
        kpi_card("Total registrados", f"{n_total:,}"),
        kpi_card("Hombres (con dato)", f"{pct_h:.1f}%", "#2E86AB"),
        kpi_card("Mujeres (con dato)", f"{pct_m:.1f}%", "#F4A261"),
        kpi_card("Año pico", f"{peak_year} ({peak_n:,})", "#E84855"),
    ]
    return kpis, fig_trend(d), fig_monthly(d), fig_sex(d), fig_age(d)


@app.callback(
    Output("graph-state-ranking", "figure"),
    Input("estado-filter", "value"),
)
def update_static(_):
    return _fig_state_ranking


if __name__ == "__main__":
    app.run(debug=True, port=8057)
