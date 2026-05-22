import polars as pl
import plotly.graph_objects as go
import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc

# ── Data ──────────────────────────────────────────────────────────────────────
NUM_COLS = ["buques", "entrada", "salida", "cajas", "toneladas", "teus", "pasajeros"]

df = pl.read_csv("data/puertos.csv", null_values=[""]).with_columns(
    [pl.col("mes").cast(pl.Int32), pl.col("anio").cast(pl.Int32)]
    + [pl.col(c).cast(pl.Float64) for c in NUM_COLS]
)

ALL_TRAFICOS = sorted(df["trafico"].drop_nulls().unique().to_list())

# ── Theme ─────────────────────────────────────────────────────────────────────
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
CARGO_COLORS = {
    "Agricola":     "#3BB273",
    "Contenerizada":"#2E86AB",
    "Mineral":      "#94A3B8",
    "Otros F.":     "#F4A261",
    "Petroleo":     "#E84855",
    "Suelta":       "#9B59B6",
    "pasajeros":    "#22D3EE",
}
TOP_N = 15


def _fmt_num(v: float) -> str:
    if v >= 1_000_000:
        return f"{v / 1_000_000:.1f}M"
    if v >= 1_000:
        return f"{v / 1_000:.1f}K"
    return f"{v:.0f}"


# ── Figure factories ──────────────────────────────────────────────────────────

def fig_top_buques(d: pl.DataFrame) -> go.Figure:
    agg = (
        d.filter(pl.col("tipo_carga") != "pasajeros")
        .group_by("puerto").agg(pl.col("buques").sum().alias("total"))
        .sort("total", descending=True).head(TOP_N)
        .sort("total")
    )
    fig = go.Figure(go.Bar(
        x=agg["total"].to_list(), y=agg["puerto"].to_list(),
        orientation="h", marker_color="#2E86AB",
        text=[_fmt_num(v) for v in agg["total"].to_list()],
        textposition="outside",
    ))
    fig.update_layout(
        title="Top puertos por buques", height=max(300, TOP_N * 28 + 80),
        margin=dict(l=10, r=60, t=40, b=20), **CHART_LAYOUT,
    )
    return fig


def fig_tipo_carga(d: pl.DataFrame) -> go.Figure:
    cargo = d.filter(pl.col("tipo_carga") != "pasajeros")
    top_ports = (
        cargo.group_by("puerto").agg(pl.col("buques").sum())
        .sort("buques", descending=True).head(TOP_N)["puerto"].to_list()
    )
    agg = (
        cargo.filter(pl.col("puerto").is_in(top_ports))
        .group_by(["puerto", "tipo_carga"]).agg(pl.col("buques").sum().alias("total"))
    )
    totals = agg.group_by("puerto").agg(pl.col("total").sum().alias("grand"))
    agg = (
        agg.join(totals, on="puerto")
        .with_columns((pl.col("total") / pl.col("grand") * 100).round(1).alias("pct"))
        .join(
            totals.sort("grand", descending=True)
            .with_row_index("rank"),
            on="puerto",
        )
        .sort("rank", descending=True)
    )
    fig = go.Figure()
    for tipo in ["Agricola", "Contenerizada", "Mineral", "Otros F.", "Petroleo", "Suelta"]:
        sub = agg.filter(pl.col("tipo_carga") == tipo)
        if sub.is_empty():
            continue
        ports = sub["puerto"].to_list()
        pcts = sub["pct"].to_list()
        counts = sub["total"].to_list()
        fig.add_trace(go.Bar(
            x=pcts, y=ports, orientation="h", name=tipo,
            marker_color=CARGO_COLORS.get(tipo, "#888"),
            text=[f"{p:.0f}%" for p in pcts], textposition="inside", insidetextanchor="middle",
            customdata=counts,
            hovertemplate=f"<b>{tipo}</b>: %{{x:.1f}}%  (n=%{{customdata[0]:,.0f}})<extra></extra>",
        ))
    fig.update_layout(
        barmode="stack", title="Tipo de carga por puerto (% buques)",
        height=max(300, TOP_N * 28 + 80),
        xaxis=dict(range=[0, 100], visible=False, gridcolor="#334155"),
        yaxis=dict(gridcolor="#334155"),
        margin=dict(l=10, r=10, t=40, b=20),
        legend=dict(orientation="h", y=-0.12, x=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#CBD5E1",
    )
    return fig


def fig_top_toneladas(d: pl.DataFrame) -> go.Figure:
    agg = (
        d.filter(pl.col("tipo_carga") != "pasajeros")
        .group_by("puerto").agg((pl.col("entrada").sum() / 1_000).alias("miles_ton"))
        .sort("miles_ton", descending=True).head(TOP_N)
        .sort("miles_ton")
    )
    fig = go.Figure(go.Bar(
        x=agg["miles_ton"].to_list(), y=agg["puerto"].to_list(),
        orientation="h", marker_color="#3BB273",
        text=[f"{v:,.0f}K" for v in agg["miles_ton"].to_list()],
        textposition="outside",
    ))
    fig.update_layout(
        title="Top puertos por toneladas de entrada (miles)", height=max(300, TOP_N * 28 + 80),
        margin=dict(l=10, r=80, t=40, b=20), **CHART_LAYOUT,
    )
    return fig


def fig_top_pasajeros(d: pl.DataFrame) -> go.Figure:
    agg = (
        d.filter(pl.col("tipo_carga") == "pasajeros")
        .group_by("puerto").agg(pl.col("pasajeros").sum().alias("total"))
        .drop_nulls("total")
        .sort("total", descending=True).head(TOP_N)
        .sort("total")
    )
    fig = go.Figure(go.Bar(
        x=agg["total"].to_list(), y=agg["puerto"].to_list(),
        orientation="h", marker_color="#22D3EE",
        text=[_fmt_num(v) for v in agg["total"].to_list()],
        textposition="outside",
    ))
    fig.update_layout(
        title="Top puertos por pasajeros (cruceros)", height=max(300, TOP_N * 28 + 80),
        margin=dict(l=10, r=60, t=40, b=20), **CHART_LAYOUT,
    )
    return fig


def fig_top_teus(d: pl.DataFrame) -> go.Figure:
    agg = (
        d.filter(pl.col("tipo_carga") == "Contenerizada")
        .group_by("puerto").agg(pl.col("teus").sum().alias("total"))
        .drop_nulls("total")
        .sort("total", descending=True).head(TOP_N)
        .sort("total")
    )
    fig = go.Figure(go.Bar(
        x=agg["total"].to_list(), y=agg["puerto"].to_list(),
        orientation="h", marker_color="#F4A261",
        text=[_fmt_num(v) for v in agg["total"].to_list()],
        textposition="outside",
    ))
    fig.update_layout(
        title="Top puertos por TEUs (carga contenerizada)", height=max(300, TOP_N * 28 + 80),
        margin=dict(l=10, r=60, t=40, b=20), **CHART_LAYOUT,
    )
    return fig


def compute_kpis(d: pl.DataFrame):
    d_cargo = d.filter(pl.col("tipo_carga") != "pasajeros")
    d_pas = d.filter(pl.col("tipo_carga") == "pasajeros")
    return (
        d["puerto"].n_unique(),
        int(d_cargo["buques"].drop_nulls().sum()),
        int(d_cargo["entrada"].drop_nulls().sum()),
        int(d_pas["pasajeros"].drop_nulls().sum()),
    )


# ── Layout ────────────────────────────────────────────────────────────────────
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SLATE])
app.title = "Puertos México"

FILTER_STYLE = {"background": "#1E293B", "border": "1px solid #334155", "color": "#CBD5E1"}

app.layout = html.Div(style={"backgroundColor": "#0F172A", "minHeight": "100vh", "padding": "24px"}, children=[
    html.H2("Puertos Marítimos de México", style={"color": "#F8FAFC", "marginBottom": "4px"}),
    html.P("Dic 2025 – Ene 2026 · Fuente: SCT / API Puertos", style={"color": "#64748B", "marginBottom": "20px"}),

    # Filters
    dbc.Row([
        dbc.Col([
            html.Label("Periodo", style={"color": "#94A3B8", "fontSize": "12px"}),
            dcc.Dropdown(
                id="dd-periodo",
                options=[
                    {"label": "Ambos periodos", "value": "todos"},
                    {"label": "Dic 2025", "value": "dic2025"},
                    {"label": "Ene 2026", "value": "ene2026"},
                ],
                value="todos", clearable=False,
                style=FILTER_STYLE,
            ),
        ], width=3),
        dbc.Col([
            html.Label("Tráfico (carga)", style={"color": "#94A3B8", "fontSize": "12px"}),
            dcc.Dropdown(
                id="dd-trafico",
                options=[{"label": "Todos", "value": "todos"}]
                        + [{"label": t, "value": t} for t in ALL_TRAFICOS],
                value="todos", clearable=False,
                style=FILTER_STYLE,
            ),
        ], width=3),
    ], style={"marginBottom": "24px"}),

    # KPI cards
    dbc.Row([
        dbc.Col(html.Div([html.H4(id="kpi-puertos", style={"color": "#F8FAFC", "margin": 0}),
                          html.P("Puertos activos", style={"color": "#94A3B8", "margin": 0})],
                         style=CARD_STYLE)),
        dbc.Col(html.Div([html.H4(id="kpi-buques", style={"color": "#2E86AB", "margin": 0}),
                          html.P("Buques (carga)", style={"color": "#94A3B8", "margin": 0})],
                         style=CARD_STYLE)),
        dbc.Col(html.Div([html.H4(id="kpi-toneladas", style={"color": "#3BB273", "margin": 0}),
                          html.P("Toneladas entrada", style={"color": "#94A3B8", "margin": 0})],
                         style=CARD_STYLE)),
        dbc.Col(html.Div([html.H4(id="kpi-pasajeros", style={"color": "#22D3EE", "margin": 0}),
                          html.P("Pasajeros (cruceros)", style={"color": "#94A3B8", "margin": 0})],
                         style=CARD_STYLE)),
    ], style={"marginBottom": "24px"}),

    # Row 1: buques + tipo carga
    dbc.Row([
        dbc.Col(dcc.Graph(id="g-buques"), width=6),
        dbc.Col(dcc.Graph(id="g-tipo-carga"), width=6),
    ], style={"marginBottom": "16px"}),

    # Row 2: toneladas + pasajeros
    dbc.Row([
        dbc.Col(dcc.Graph(id="g-toneladas"), width=6),
        dbc.Col(dcc.Graph(id="g-pasajeros"), width=6),
    ], style={"marginBottom": "16px"}),

    # Row 3: TEUs
    dbc.Row([
        dbc.Col(dcc.Graph(id="g-teus"), width=12),
    ]),
])


# ── Callback ──────────────────────────────────────────────────────────────────
@app.callback(
    Output("kpi-puertos", "children"),
    Output("kpi-buques", "children"),
    Output("kpi-toneladas", "children"),
    Output("kpi-pasajeros", "children"),
    Output("g-buques", "figure"),
    Output("g-tipo-carga", "figure"),
    Output("g-toneladas", "figure"),
    Output("g-pasajeros", "figure"),
    Output("g-teus", "figure"),
    Input("dd-periodo", "value"),
    Input("dd-trafico", "value"),
)
def update(periodo, trafico):
    d = df
    if periodo == "dic2025":
        d = d.filter((pl.col("anio") == 2025) & (pl.col("mes") == 12))
    elif periodo == "ene2026":
        d = d.filter((pl.col("anio") == 2026) & (pl.col("mes") == 1))

    d_cargo = d.filter(pl.col("tipo_carga") != "pasajeros")
    if trafico != "todos":
        d_cargo = d_cargo.filter(pl.col("trafico") == trafico)

    # Rebuild d with filtered cargo + unfiltered pasajeros for KPIs
    d_for_kpis = pl.concat([d_cargo, d.filter(pl.col("tipo_carga") == "pasajeros")], how="diagonal")

    n_puertos, n_buques, n_ton, n_pas = compute_kpis(d_for_kpis)

    return (
        str(n_puertos),
        _fmt_num(n_buques),
        _fmt_num(n_ton),
        _fmt_num(n_pas),
        fig_top_buques(d_cargo),
        fig_tipo_carga(d_cargo),
        fig_top_toneladas(d_cargo),
        fig_top_pasajeros(d),
        fig_top_teus(d_cargo),
    )


if __name__ == "__main__":
    app.run(debug=True, port=8059)
