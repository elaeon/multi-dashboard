import polars as pl
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc
from collections import Counter

# ── Data loading ──────────────────────────────────────────────────────────────

_RAW = pl.read_excel(
    "data/entorno_alimentario_escuelas.xlsx",
    sheet_name="1252 europeo occidental windows",
)

# Short aliases for the long survey question columns
Q_COLS = {
    "refrescos":  "¿Se venden refrescos con azúcar (no light)?",
    "bebidas_az": "¿Se venden otras bebidas envasadas con azúcar como jugos o aguas saborizadas?",
    "chatarra":   "¿Se vende comida chatarra de lunes a jueves (por ejemplo, frituras, dulces, galletas, helados)?*",
    "frutas":     "¿Se venden frutas y verduras todos los días (por ejemplo, manzana, zanahoria, naranja, sandía, pepino)?",
    "cereales":   "¿Se venden cereales integrales todos los días (por ejemplo, avena, amaranto, palomitas)?",
    "semillas":   "¿Se venden semillas todos los días (por ejemplo, cacahuates, almendras, habas, chicharos secos)?",
    "bebederos":  "¿Hay bebederos o dispensadores de agua funcionando?",
    "comite":     "¿Hay un comité que vigila la prohibición de la venta de comida chatarra y bebidas azucaradas?\xa0",
    "logos":      "¿Hay logos o nombres de marcas de comida chatarra y/o bebidas azucaradas dentro de la escuela (por ejemplo, en la tienda escolar, canchas, patios, eventos y/o torneos)?",
    "puestos":    "Afuera de la escuela, ¿hay puestos ambulantes que venden comida chatarra y/o bebidas azucaradas?",
    "practicas":  "¿Consideras que la escuela ha promovido prácticas positivas para tener una alimentación saludable durante la jornada escolar?\xa0",
}
Q_LABELS = {
    "refrescos":  "Venden refrescos con azúcar",
    "bebidas_az": "Venden otras bebidas azucaradas",
    "chatarra":   "Venden comida chatarra (L-J)",
    "frutas":     "Venden frutas y verduras",
    "cereales":   "Venden cereales integrales",
    "semillas":   "Venden semillas",
    "bebederos":  "Bebederos de agua funcionando",
    "comite":     "Comité de vigilancia activo",
    "logos":      "Logos de marcas chatarra",
    "puestos":    "Puestos ambulantes afuera",
    "practicas":  "Promueve alimentación saludable",
}

SELLER_COL = "¿Quién vende los alimentos y bebidas dentro de la escuela? (señalar todas las que apliquen)\xa0"
VALID_CICLOS = ["2014-2015", "2015-2016", "2016-2017", "2017-2018",
                "2018-2019", "2022-2023", "2023-2024", "2024-2025"]
MAIN_ROLES = {"Madre de familia", "Padre de familia", "Personal docente",
              "Familiar", "Alumno", "Alumna", "Otro"}

# Rename long question columns to short keys, keep only useful columns
df = _RAW.rename({v: k for k, v in Q_COLS.items()})

# Normalize am (respondent role)
df = df.with_columns(
    pl.col("am").map_elements(
        lambda x: x if x in MAIN_ROLES else ("Otro" if isinstance(x, str) and x.strip() else None),
        return_dtype=pl.String,
    ).alias("rol")
)

# Keep only valid binary values {0, 1} per question — store as float for mean()
for k in Q_COLS:
    df = df.with_columns(
        pl.when(pl.col(k).is_in([0, 1])).then(pl.col(k).cast(pl.Float64)).otherwise(None).alias(k)
    )

# Normalize ciclo_escolar
df = df.with_columns(
    pl.when(pl.col("ciclo_escolar").is_in(VALID_CICLOS))
    .then(pl.col("ciclo_escolar"))
    .otherwise(None)
    .alias("ciclo_escolar")
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

# ── Helpers ───────────────────────────────────────────────────────────────────

def pct_yes(d: pl.DataFrame, col: str) -> float:
    valid = d[col].drop_nulls()
    return float(valid.mean() * 100) if len(valid) > 0 else 0.0


def kpi_card(title: str, value: str, color: str = "#CBD5E1") -> dbc.Col:
    return dbc.Col(
        html.Div([
            html.P(title, style={"color": "#94A3B8", "fontSize": "12px", "margin": 0}),
            html.H3(value, style={"color": color, "margin": "4px 0 0"}),
        ], style=CARD_STYLE),
        xs=12, sm=6, md=3,
    )


# ── Figure factories ──────────────────────────────────────────────────────────

def _add_covid_vline(fig: go.Figure) -> None:
    # The data jumps from 2018-2019 (index 4) to 2022-2023 (index 5) — x=4.5 sits in that gap
    fig.add_vline(
        x=4.5,
        line=dict(color="#64748B", width=1.5, dash="dash"),
    )
    fig.add_annotation(
        xref="x", yref="paper",
        x=4.5, y=0.98,
        text="Pandemia<br>COVID-19",
        showarrow=False,
        font=dict(color="#94A3B8", size=11),
        bgcolor="rgba(15,23,42,0.75)",
        borderpad=4,
        xanchor="left",
    )


def fig_panorama(d: pl.DataFrame) -> go.Figure:
    rows = [(Q_LABELS[k], pct_yes(d, k)) for k in Q_COLS]
    rows.sort(key=lambda x: x[1])
    labels, vals = zip(*rows)
    colors = ["#E84855" if v > 50 else "#3BB273" for v in vals]
    fig = go.Figure(go.Bar(
        x=list(vals), y=list(labels), orientation="h",
        marker_color=colors,
        text=[f"{v:.1f}%" for v in vals], textposition="outside",
    ))
    fig.update_layout(
        title="Panorama general — % de respuestas Sí",
        height=max(300, len(Q_COLS) * 38 + 80),
        xaxis=dict(range=[0, 105], gridcolor="#334155", ticksuffix="%"),
        yaxis=dict(gridcolor="rgba(0,0,0,0)"),
        **{k: v for k, v in CHART_LAYOUT.items() if k not in ("xaxis", "yaxis")},
    )
    return fig


def fig_trend(d: pl.DataFrame) -> go.Figure:
    trend_keys = ["chatarra", "bebederos", "frutas", "comite"]
    trend_colors = {"chatarra": "#E84855", "bebederos": "#2E86AB",
                    "frutas": "#3BB273", "comite": "#F4A261"}
    d_valid = d.filter(pl.col("ciclo_escolar").is_not_null())
    agg = (
        d_valid.group_by("ciclo_escolar")
        .agg([pl.col(k).mean().alias(k) for k in trend_keys])
        .sort("ciclo_escolar")
    )
    fig = go.Figure()
    for k in trend_keys:
        y = (agg[k].to_list())
        y_pct = [v * 100 if v is not None else None for v in y]
        fig.add_trace(go.Scatter(
            x=agg["ciclo_escolar"].to_list(), y=y_pct,
            mode="lines+markers", name=Q_LABELS[k],
            line=dict(color=trend_colors[k], width=2),
        ))
    fig.update_layout(
        title="Evolución por ciclo escolar",
        height=380,
        xaxis=dict(gridcolor="#334155"),
        yaxis=dict(gridcolor="#334155", ticksuffix="%", range=[0, 105]),
        legend=dict(bgcolor="rgba(0,0,0,0)", font_color="#94A3B8"),
        **{k: v for k, v in CHART_LAYOUT.items() if k not in ("xaxis", "yaxis")},
    )
    _add_covid_vline(fig)
    return fig


def fig_states(d: pl.DataFrame) -> go.Figure:
    # Top states by junk food rate (min 30 valid responses)
    valid = d.filter(pl.col("chatarra").is_not_null() & pl.col("state").str.len_chars().gt(3))
    agg = (
        valid.group_by("state")
        .agg(pl.col("chatarra").mean().alias("pct"), pl.col("chatarra").count().alias("n"))
        .filter(pl.col("n") >= 30)
        .sort("pct")
        .head(20)
    )
    agg = agg.with_columns((pl.col("pct") * 100).alias("pct"))
    pcts = agg["pct"].to_list()
    states = agg["state"].to_list()
    colors = [
        f"rgb({int(255 * min(1, max(0, (p - 50) / 50)))},{int(255 * min(1, max(0, 1 - (p - 50) / 50)))},80)"
        for p in pcts
    ]
    fig = go.Figure(go.Bar(
        x=pcts, y=states, orientation="h",
        marker_color=colors,
        text=[f"{p:.1f}%" for p in pcts], textposition="outside",
    ))
    fig.update_layout(
        title="% Escuelas con venta de comida chatarra por estado",
        height=max(340, 20 * 28 + 80),
        xaxis=dict(range=[0, 105], gridcolor="#334155", ticksuffix="%"),
        yaxis=dict(gridcolor="rgba(0,0,0,0)"),
        **{k: v for k, v in CHART_LAYOUT.items() if k not in ("xaxis", "yaxis")},
    )
    return fig


def fig_sellers(d: pl.DataFrame) -> go.Figure:
    c: Counter = Counter()
    for v in d[SELLER_COL].drop_nulls().to_list():
        if isinstance(v, str) and v.strip():
            for part in v.split(","):
                p = part.strip()
                if p and not p.lstrip("-").replace(".", "").isdigit() and len(p) > 2:
                    c[p] += 1
    if not c:
        return go.Figure()
    items = sorted(c.items(), key=lambda x: x[1])[-8:]
    labels, vals = zip(*items)
    fig = go.Figure(go.Bar(
        x=list(vals), y=list(labels), orientation="h",
        marker_color="#2E86AB",
        text=[f"{v:,}" for v in vals], textposition="outside",
    ))
    fig.update_layout(
        title="¿Quién vende alimentos dentro de la escuela?",
        height=max(280, len(items) * 38 + 80),
        xaxis=dict(gridcolor="#334155"),
        yaxis=dict(gridcolor="rgba(0,0,0,0)"),
        **{k: v for k, v in CHART_LAYOUT.items() if k not in ("xaxis", "yaxis")},
    )
    return fig


def fig_crisis_trend(d: pl.DataFrame) -> go.Figure:
    d_valid = d.filter(pl.col("ciclo_escolar").is_not_null())

    # bebederos and comite per ciclo
    agg_bc = (
        d_valid.group_by("ciclo_escolar")
        .agg(
            pl.col("bebederos").mean().alias("bebederos"),
            pl.col("comite").mean().alias("comite"),
        )
        .sort("ciclo_escolar")
    )

    # triple problem: chatarra=1 AND bebederos=0 AND comite=0
    agg_triple = (
        d_valid.filter(
            pl.col("chatarra").is_not_null() &
            pl.col("bebederos").is_not_null() &
            pl.col("comite").is_not_null()
        )
        .group_by("ciclo_escolar")
        .agg(
            ((pl.col("chatarra") == 1) & (pl.col("bebederos") == 0) & (pl.col("comite") == 0))
            .mean().alias("triple")
        )
        .sort("ciclo_escolar")
    )

    agg = agg_bc.join(agg_triple, on="ciclo_escolar", how="left").sort("ciclo_escolar")
    ciclos = agg["ciclo_escolar"].to_list()

    def to_pct(col):
        return [round(v * 100, 1) if v is not None else None for v in agg[col].to_list()]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ciclos, y=to_pct("triple"),
        mode="lines+markers", name="Triple problema (chatarra + sin agua + sin comité)",
        line=dict(color="#E84855", width=3),
        fill="tozeroy", fillcolor="rgba(232,72,85,0.07)",
    ))
    fig.add_trace(go.Scatter(
        x=ciclos, y=to_pct("bebederos"),
        mode="lines+markers", name="Bebederos de agua funcionando",
        line=dict(color="#2E86AB", width=2, dash="dot"),
    ))
    fig.add_trace(go.Scatter(
        x=ciclos, y=to_pct("comite"),
        mode="lines+markers", name="Comité de vigilancia activo",
        line=dict(color="#F4A261", width=2, dash="dot"),
    ))
    fig.update_layout(
        title="Crisis estructural: triple problema, agua y vigilancia por ciclo escolar",
        height=460,
        xaxis=dict(gridcolor="#334155"),
        yaxis=dict(gridcolor="#334155", ticksuffix="%", range=[0, 105]),
        legend=dict(bgcolor="rgba(0,0,0,0)", font_color="#94A3B8", orientation="h",
                    yanchor="top", y=-0.18, xanchor="center", x=0.5),
        margin=dict(t=50, b=100, l=10, r=10),
        **{k: v for k, v in CHART_LAYOUT.items() if k not in ("xaxis", "yaxis", "margin")},
    )
    _add_covid_vline(fig)
    return fig


def fig_roles(d: pl.DataFrame) -> go.Figure:
    counts = d.filter(pl.col("rol").is_not_null())["rol"].value_counts().sort("count", descending=True)
    fig = go.Figure(go.Pie(
        labels=counts["rol"].to_list(),
        values=counts["count"].to_list(),
        hole=0.5,
        textinfo="label+percent",
        marker_colors=["#2E86AB", "#3BB273", "#F4A261", "#E84855", "#94A3B8", "#CBD5E1", "#64748B"],
    ))
    fig.update_layout(
        title="Rol del encuestado",
        height=360,
        showlegend=False,
        **CHART_LAYOUT,
    )
    return fig


# ── Layout ────────────────────────────────────────────────────────────────────

TAB_STYLE = {"backgroundColor": "#0F172A", "color": "#94A3B8", "borderTop": "none",
             "borderBottom": "1px solid #334155", "padding": "10px 18px"}
TAB_SEL   = {"backgroundColor": "#1E293B", "color": "#F8FAFC",
             "borderTop": "2px solid #2E86AB", "borderBottom": "none",
             "fontWeight": "600", "padding": "10px 18px"}

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], title="Entorno Alimentario Escuelas")

ciclo_options = [{"label": c, "value": c} for c in VALID_CICLOS]
rol_options = [{"label": r, "value": r} for r in sorted(MAIN_ROLES)] + [{"label": "Todos", "value": "all"}]

app.layout = html.Div(style={"backgroundColor": "#0F172A", "minHeight": "100vh", "padding": "24px"}, children=[
    html.H2("Entorno Alimentario en Escuelas de México",
            style={"color": "#F8FAFC", "fontWeight": "700", "marginBottom": "4px"}),
    html.P("Encuesta sobre hábitos de venta de alimentos y bebidas en escuelas",
           style={"color": "#94A3B8", "marginBottom": "24px"}),

    # Filters
    dbc.Row(style={"marginBottom": "20px"}, children=[
        dbc.Col([
            html.Label("Ciclo escolar", style={"color": "#94A3B8", "fontSize": "12px"}),
            dcc.Dropdown(
                id="ciclo-filter",
                options=[{"label": "Todos", "value": "all"}] + ciclo_options,
                value="all", clearable=False,
                style={"backgroundColor": "#1E293B", "color": "#CBD5E1", "border": "1px solid #334155"},
            ),
        ], md=4),
        dbc.Col([
            html.Label("Rol del encuestado", style={"color": "#94A3B8", "fontSize": "12px"}),
            dcc.Dropdown(
                id="rol-filter",
                options=rol_options,
                value="all", clearable=False,
                style={"backgroundColor": "#1E293B", "color": "#CBD5E1", "border": "1px solid #334155"},
            ),
        ], md=4),
    ]),

    # KPI row
    dbc.Row(id="kpi-row", className="g-3", style={"marginBottom": "20px"}),

    # Tabs
    dcc.Tabs(style={"marginBottom": "0"}, children=[

        dcc.Tab(label="Panorama General", style=TAB_STYLE, selected_style=TAB_SEL, children=[
            html.Div(style={"paddingTop": "20px"}, children=[
                dbc.Row(className="g-3", style={"marginBottom": "20px"}, children=[
                    dbc.Col(dcc.Graph(id="graph-panorama"), md=6),
                    dbc.Col(dcc.Graph(id="graph-trend"), md=6),
                ]),
                dbc.Row(className="g-3", children=[
                    dbc.Col(dcc.Graph(id="graph-states"), md=6),
                    dbc.Col([
                        dcc.Graph(id="graph-sellers"),
                        dcc.Graph(id="graph-roles"),
                    ], md=6),
                ]),
            ]),
        ]),

        dcc.Tab(label="Crisis Estructural", style=TAB_STYLE, selected_style=TAB_SEL, children=[
            html.Div(style={"paddingTop": "20px"}, children=[
                html.Div(style={
                    "background": "#1E293B", "border": "1px solid #E84855",
                    "borderRadius": "8px", "padding": "14px 18px", "marginBottom": "20px",
                }, children=[
                    html.P([
                        html.Strong("Tres indicadores que muestran el deterioro del entorno alimentario escolar. ",
                                    style={"color": "#F8FAFC"}),
                        html.Span(
                            "El triple problema agrupa escuelas con venta de comida chatarra, "
                            "sin bebederos de agua funcionando y sin comité de vigilancia activo — simultáneamente. "
                            "Responde al filtro de rol del encuestado; siempre muestra todos los ciclos.",
                            style={"color": "#94A3B8", "fontSize": "13px"},
                        ),
                    ], style={"margin": 0}),
                ]),
                dbc.Row(className="g-3", children=[
                    dbc.Col(dcc.Graph(id="graph-crisis"), md=12),
                ]),
            ]),
        ]),

    ]),
])


# ── Callback ──────────────────────────────────────────────────────────────────

@app.callback(
    Output("kpi-row", "children"),
    Output("graph-panorama", "figure"),
    Output("graph-trend", "figure"),
    Output("graph-states", "figure"),
    Output("graph-sellers", "figure"),
    Output("graph-roles", "figure"),
    Input("ciclo-filter", "value"),
    Input("rol-filter", "value"),
)
def update_all(ciclo: str, rol: str):
    d = df
    if ciclo != "all":
        d = d.filter(pl.col("ciclo_escolar") == ciclo)
    if rol != "all":
        d = d.filter(pl.col("rol") == rol)

    n = len(d)
    kpis = [
        kpi_card("Total de respuestas", f"{n:,}", "#CBD5E1"),
        kpi_card("Venden refrescos con azúcar", f"{pct_yes(d, 'refrescos'):.1f}%", "#E84855"),
        kpi_card("Venden comida chatarra", f"{pct_yes(d, 'chatarra'):.1f}%", "#E84855"),
        kpi_card("Bebederos funcionando", f"{pct_yes(d, 'bebederos'):.1f}%", "#3BB273"),
        kpi_card("Comité de vigilancia activo", f"{pct_yes(d, 'comite'):.1f}%", "#F4A261"),
        kpi_card("Promueve alimentación saludable", f"{pct_yes(d, 'practicas'):.1f}%", "#2E86AB"),
    ]
    return (
        kpis,
        fig_panorama(d),
        fig_trend(d),
        fig_states(d),
        fig_sellers(d),
        fig_roles(d),
    )


@app.callback(
    Output("graph-crisis", "figure"),
    Input("rol-filter", "value"),
)
def update_crisis(rol: str):
    d = df if rol == "all" else df.filter(pl.col("rol") == rol)
    return fig_crisis_trend(d)


if __name__ == "__main__":
    app.run(debug=True, port=8054)
