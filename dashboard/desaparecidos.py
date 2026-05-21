import json
import polars as pl
import plotly.express as px
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
    pl.when(pl.col("SEXO") == "HOMBRE").then(pl.lit("Hombre"))
    .when(pl.col("SEXO") == "MUJER").then(pl.lit("Mujer"))
    .otherwise(pl.lit("Desconocido")).alias("sexo_cat"),
)

df = df.with_columns(
    pl.when(pl.col("anio_desap").is_between(1990, 2025))
    .then(pl.col("anio_desap")).otherwise(None).alias("anio_desap")
)

ESTADOS = sorted(df["ENTIDAD"].drop_nulls().unique().to_list())

with open("data/mexico_states.geojson") as _f:
    MEXICO_GEO = json.load(_f)
MES_NOMBRES = ["Ene", "Feb", "Mar", "Abr", "May", "Jun",
               "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]
AGE_LABELS  = ["0–12", "13–17", "18–29", "30–44", "45–59", "60+"]
AGE_RANGES  = [(0, 12), (13, 17), (18, 29), (30, 44), (45, 59), (60, 100)]

SEXO_COLORS = {"Hombre": "#2E86AB", "Mujer": "#F4A261", "Desconocido": "#64748B"}
SEXO_ORDER  = ["Hombre", "Mujer", "Desconocido"]

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
        .group_by(["anio_desap", "sexo_cat"]).agg(pl.len().alias("n"))
        .sort("anio_desap")
    )
    if len(anual) == 0:
        return go.Figure()
    years = sorted(anual["anio_desap"].unique().to_list())
    fig = go.Figure()
    for sexo in SEXO_ORDER:
        subset = anual.filter(pl.col("sexo_cat") == sexo)
        yr_n = dict(zip(subset["anio_desap"].to_list(), subset["n"].to_list()))
        ns = [yr_n.get(y, 0) for y in years]
        if sum(ns) == 0:
            continue
        fig.add_trace(go.Bar(
            x=years, y=ns, name=sexo,
            marker_color=SEXO_COLORS[sexo],
            hovertemplate=f"<b>%{{x}}</b> · {sexo}<br>Casos: %{{y:,}}<extra></extra>",
        ))
    fig.update_layout(
        title="Personas desaparecidas por año",
        barmode="stack",
        height=380,
        xaxis=dict(gridcolor="#334155", title="Año"),
        yaxis=dict(gridcolor="#334155", title="Casos registrados"),
        legend=dict(orientation="h", y=-0.18, x=0),
        showlegend=True,
        **{k: v for k, v in CHART_LAYOUT.items() if k not in ("xaxis", "yaxis", "margin")},
        margin=dict(t=40, b=70, l=10, r=10),
    )
    return fig


def fig_monthly(d: pl.DataFrame) -> go.Figure:
    monthly = (
        d.filter(pl.col("mes_desap").is_not_null() & pl.col("anio_desap").is_between(2010, 2025))
        .group_by(["mes_desap", "sexo_cat"]).agg(pl.len().alias("n"))
        .sort("mes_desap")
    )
    if len(monthly) == 0:
        return go.Figure()
    meses_idx = sorted(monthly["mes_desap"].unique().to_list())
    meses_lbl = [MES_NOMBRES[m - 1] for m in meses_idx]
    fig = go.Figure()
    for sexo in SEXO_ORDER:
        subset = monthly.filter(pl.col("sexo_cat") == sexo)
        m_n = dict(zip(subset["mes_desap"].to_list(), subset["n"].to_list()))
        ns = [m_n.get(m, 0) for m in meses_idx]
        if sum(ns) == 0:
            continue
        fig.add_trace(go.Bar(
            x=meses_lbl, y=ns, name=sexo,
            marker_color=SEXO_COLORS[sexo],
            hovertemplate=f"<b>%{{x}}</b> · {sexo}<br>Casos: %{{y:,}}<extra></extra>",
        ))
    fig.update_layout(
        title="Distribución mensual (2010–2025)",
        barmode="stack",
        height=360,
        xaxis=dict(gridcolor="#334155"),
        yaxis=dict(gridcolor="#334155"),
        legend=dict(orientation="h", y=1.1, x=0),
        showlegend=True,
        **{k: v for k, v in CHART_LAYOUT.items() if k not in ("xaxis", "yaxis", "margin")},
        margin=dict(t=60, b=40, l=10, r=10),
    )
    return fig


def fig_sex(d: pl.DataFrame) -> go.Figure:
    counts = (
        d.group_by("sexo_cat").agg(pl.len().alias("n"))
        .sort("n", descending=True)
    )
    if len(counts) == 0:
        return go.Figure()
    total = int(counts["n"].sum())
    labels = counts["sexo_cat"].to_list()
    ns     = counts["n"].to_list()
    pcts   = [n / total * 100 for n in ns]
    colors = [SEXO_COLORS.get(s, "#64748B") for s in labels]
    fig = go.Figure(go.Bar(
        x=ns, y=labels, orientation="h",
        marker_color=colors,
        text=[f"{p:.1f}%  ({n:,})" for p, n in zip(pcts, ns)],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Casos: %{x:,}<extra></extra>",
    ))
    fig.update_layout(
        title="Distribución por sexo",
        height=360,
        xaxis=dict(gridcolor="#334155"),
        yaxis=dict(gridcolor="rgba(0,0,0,0)"),
        showlegend=False,
        **{k: v for k, v in CHART_LAYOUT.items() if k not in ("xaxis", "yaxis", "margin")},
        margin=dict(t=40, b=40, l=10, r=150),
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

    total = len(with_age)
    fig = go.Figure()
    for sexo in SEXO_ORDER:
        subset = with_age.filter(pl.col("sexo_cat") == sexo)
        pcts = [
            len(subset.filter(pl.col("edad").is_between(lo, hi))) / total * 100
            for lo, hi in AGE_RANGES
        ]
        if sum(pcts) == 0:
            continue
        fig.add_trace(go.Bar(
            x=AGE_LABELS, y=pcts, name=sexo,
            marker_color=SEXO_COLORS[sexo],
            hovertemplate=f"<b>%{{x}}</b> · {sexo}<br>%{{y:.1f}}%<extra></extra>",
        ))
    fig.update_layout(
        title="Distribución por grupo de edad",
        barmode="stack",
        height=360,
        xaxis=dict(gridcolor="#334155", title="Edad al desaparecer"),
        yaxis=dict(gridcolor="#334155", ticksuffix="%"),
        legend=dict(orientation="h", y=1.1, x=0),
        showlegend=True,
        **{k: v for k, v in CHART_LAYOUT.items() if k not in ("xaxis", "yaxis", "margin")},
        margin=dict(t=60, b=40, l=10, r=10),
    )
    return fig


def fig_mapa_desaparecidos(d: pl.DataFrame) -> go.Figure:
    agg = (
        d.filter(
            pl.col("ENTIDAD").is_not_null() & (pl.col("ENTIDAD") != "SE DESCONOCE")
        )
        .group_by("ENTIDAD").agg(pl.len().alias("n"))
    )
    if len(agg) == 0:
        return go.Figure()
    # Normalize uppercase ENTIDAD to title case; "ESTADO DE MÉXICO" → "México" (GeoJSON key)
    def _norm(s: str) -> str:
        t = s.title()
        return "México" if t == "Estado De México" else t
    state_data = pl.DataFrame({
        "ENTIDAD": [_norm(s) for s in agg["ENTIDAD"].to_list()],
        "n":       agg["n"].to_list(),
    })
    fig = px.choropleth_map(
        state_data,
        geojson=MEXICO_GEO,
        locations="ENTIDAD",
        color="n",
        featureidkey="properties.name",
        color_continuous_scale="YlOrRd",
        zoom=4.0,
        center={"lat": 23.6, "lon": -102.5},
        opacity=0.85,
        hover_name="ENTIDAD",
        title="Personas desaparecidas por estado",
        map_style="carto-darkmatter",
    )
    fig.update_traces(
        hovertemplate="<b>%{hovertext}</b><br>Casos: %{z:,}<extra></extra>"
    )
    fig.update_coloraxes(
        colorbar=dict(
            title=dict(text="Casos", font=dict(color="#CBD5E1")),
            tickfont=dict(color="#CBD5E1"),
        )
    )
    fig.update_layout(
        height=520,
        margin=dict(l=0, r=0, t=40, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#CBD5E1",
    )
    return fig


def fig_state_ranking(d: pl.DataFrame) -> go.Figure:
    agg = (
        d.filter(pl.col("ENTIDAD").is_not_null())
        .group_by(["ENTIDAD", "sexo_cat"]).agg(pl.len().alias("n"))
    )
    if len(agg) == 0:
        return go.Figure()
    totals = (
        agg.group_by("ENTIDAD").agg(pl.col("n").sum().alias("total"))
        .sort("total", descending=True)
    )
    states = totals["ENTIDAD"].to_list()
    fig = go.Figure()
    for sexo in SEXO_ORDER:
        subset = agg.filter(pl.col("sexo_cat") == sexo)
        st_n = dict(zip(subset["ENTIDAD"].to_list(), subset["n"].to_list()))
        ns = [st_n.get(s, 0) for s in states]
        if sum(ns) == 0:
            continue
        fig.add_trace(go.Bar(
            x=ns, y=states, orientation="h", name=sexo,
            marker_color=SEXO_COLORS[sexo],
            hovertemplate=f"<b>%{{y}}</b> · {sexo}<br>Casos: %{{x:,}}<extra></extra>",
        ))
    fig.update_layout(
        title="Total de personas desaparecidas por estado",
        barmode="stack",
        height=max(340, len(states) * 28 + 80),
        xaxis=dict(gridcolor="#334155"),
        yaxis=dict(gridcolor="rgba(0,0,0,0)", autorange="reversed"),
        legend=dict(orientation="h", y=-0.05, x=0),
        showlegend=True,
        **{k: v for k, v in CHART_LAYOUT.items() if k not in ("xaxis", "yaxis", "margin")},
        margin=dict(t=40, b=80, l=10, r=100),
    )
    return fig


def compute_kpis(d: pl.DataFrame):
    n_total  = len(d)
    n_hombre = int((d["sexo_cat"] == "Hombre").sum())
    n_mujer  = int((d["sexo_cat"] == "Mujer").sum())
    pct_hombre = n_hombre / n_total * 100 if n_total > 0 else 0
    pct_mujer  = n_mujer  / n_total * 100 if n_total > 0 else 0
    anual = d.filter(pl.col("anio_desap").is_not_null()).group_by("anio_desap").agg(pl.len().alias("n"))
    peak_year = int(anual.sort("n", descending=True)["anio_desap"][0]) if len(anual) > 0 else 0
    peak_n    = int(anual.sort("n", descending=True)["n"][0]) if len(anual) > 0 else 0
    return n_total, pct_hombre, pct_mujer, peak_year, peak_n


# ── Layout ────────────────────────────────────────────────────────────────────

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP],
           title="Personas Desaparecidas México")

estado_options = (
    [{"label": "Todo el país", "value": "__all__"}]
    + [{"label": e.title(), "value": e} for e in ESTADOS]
)
sexo_options = [
    {"label": "Todos", "value": "__all__"},
    {"label": "Hombre",      "value": "Hombre"},
    {"label": "Mujer",       "value": "Mujer"},
    {"label": "Desconocido", "value": "Desconocido"},
]

app.layout = html.Div(
    style={"backgroundColor": "#0F172A", "minHeight": "100vh", "padding": "24px"},
    children=[
        html.H2("Personas Desaparecidas — México",
                style={"color": "#F8FAFC", "fontWeight": "700", "marginBottom": "4px"}),
        html.P("Registro Nacional de Personas Desaparecidas y No Localizadas · 133,887 casos",
               style={"color": "#94A3B8", "marginBottom": "24px"}),

        # Filters
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
            dbc.Col([
                html.Label("Sexo", style={"color": "#94A3B8", "fontSize": "12px"}),
                dcc.Dropdown(
                    id="sexo-filter",
                    options=sexo_options,
                    value="__all__",
                    clearable=False,
                    style={"backgroundColor": "#1E293B", "color": "#CBD5E1",
                           "border": "1px solid #334155"},
                ),
            ], md=3),
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
                    dbc.Row(className="g-3", style={"marginBottom": "8px"}, children=[
                        dbc.Col(dcc.Graph(id="graph-mapa"), md=12),
                    ]),
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
    Input("sexo-filter", "value"),
)
def update_all(estado: str, sexo: str):
    d = df if estado == "__all__" else df.filter(pl.col("ENTIDAD") == estado)
    if sexo != "__all__":
        d = d.filter(pl.col("sexo_cat") == sexo)
    n_total, pct_h, pct_m, peak_year, peak_n = compute_kpis(d)
    kpis = [
        kpi_card("Total registrados", f"{n_total:,}"),
        kpi_card("Hombres", f"{pct_h:.1f}%", "#2E86AB"),
        kpi_card("Mujeres", f"{pct_m:.1f}%", "#F4A261"),
        kpi_card("Año pico", f"{peak_year} ({peak_n:,})", "#E84855"),
    ]
    return kpis, fig_trend(d), fig_monthly(d), fig_sex(d), fig_age(d)


@app.callback(
    Output("graph-mapa", "figure"),
    Output("graph-state-ranking", "figure"),
    Input("estado-filter", "value"),
    Input("sexo-filter", "value"),
)
def update_state_ranking(_estado: str, sexo: str):
    d = df if sexo == "__all__" else df.filter(pl.col("sexo_cat") == sexo)
    return fig_mapa_desaparecidos(d), fig_state_ranking(d)


if __name__ == "__main__":
    app.run(debug=True, port=8057)
