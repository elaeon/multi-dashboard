"""
Dashboard: Becas Bienestar (CNBBBJ) — 4to trimestre 2025
Niveles: básica, media superior, superior
"""

import polars as pl
import plotly.graph_objects as go
import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc

# ── Estilos ─────────────────────────────────────────────────────────────────

CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font_color="#CBD5E1",
    margin=dict(t=40, b=40, l=10, r=10),
)
CARD_STYLE = {
    "background": "#1E293B", "border": "1px solid #334155",
    "borderRadius": "8px", "padding": "16px", "textAlign": "center",
}
TAB_STYLE = {"backgroundColor": "#0F172A", "color": "#94A3B8", "borderTop": "none"}
TAB_SEL   = {"backgroundColor": "#1E293B", "color": "#F8FAFC",
             "borderTop": "2px solid #2E86AB", "fontWeight": "600"}

NIVEL_LABELS = {
    "todos": "Todos los niveles",
    "basica": "Educación Básica",
    "media_superior": "Media Superior",
    "superior": "Superior",
}
NIVEL_COLORS = {
    "basica": "#2E86AB",
    "media_superior": "#3BB273",
    "superior": "#F4A261",
}

# ── Pre-agregación en un solo scan ───────────────────────────────────────────

lf = (
    pl.scan_parquet("data/becas.parquet")
    .with_columns([
        pl.col("BECA").cast(pl.Int64, strict=False).alias("beca_int"),
        pl.col("FECHA_ALTA").str.slice(6, 4).alias("anio_alta"),
    ])
)

q_estado = (
    lf.group_by(["NOM_EDO", "particion_1"])
    .agg([
        (pl.col("beca_int") > 0).sum().alias("becarios"),
        pl.col("beca_int").filter(pl.col("beca_int") > 0).sum().alias("monto_total"),
        (pl.col("beca_int") == 0).sum().alias("pendientes"),
    ])
)

q_anio = (
    lf.filter(pl.col("beca_int") > 0)
    .filter(pl.col("anio_alta").is_not_null())
    .group_by(["anio_alta", "particion_1"])
    .agg(pl.len().alias("becarios"))
)

#agg_estado, agg_anio = pl.collect_all([q_estado, q_anio])

agg_estado = q_estado.with_columns(
    (pl.col("pendientes") / (pl.col("pendientes") + pl.col("becarios")) * 100)
    .alias("pct_pendiente")
)

# ── Figura factories ─────────────────────────────────────────────────────────

def fig_ranking(d: pl.DataFrame) -> go.Figure:
    top = d.sort("becarios", descending=True).head(20)
    n = len(top)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=top["becarios"].to_list(),
        y=top["NOM_EDO"].to_list(),
        orientation="h",
        marker_color="#2E86AB",
        hovertemplate="<b>%{y}</b><br>Becarios: %{x:,}<extra></extra>",
    ))
    fig.update_layout(
        **CHART_LAYOUT,
        height=max(300, n * 26 + 80),
        yaxis=dict(autorange="reversed", gridcolor="#334155"),
        xaxis=dict(gridcolor="#334155"),
        title=dict(text="Becarios por estado (con pago)", font_size=13),
    )
    return fig


def fig_monto(d: pl.DataFrame) -> go.Figure:
    top = d.sort("monto_total", descending=True).head(20)
    n = len(top)
    monto_m = (top["monto_total"] / 1_000_000).to_list()
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=monto_m,
        y=top["NOM_EDO"].to_list(),
        orientation="h",
        marker_color="#3BB273",
        hovertemplate="<b>%{y}</b><br>Monto: $%{x:,.1f} M<extra></extra>",
    ))
    fig.update_layout(
        **CHART_LAYOUT,
        height=max(300, n * 26 + 80),
        yaxis=dict(autorange="reversed", gridcolor="#334155"),
        xaxis=dict(gridcolor="#334155", title="Millones de pesos"),
        title=dict(text="Monto total distribuido por estado", font_size=13),
    )
    return fig


def fig_timeline(d: pl.DataFrame, nivel: str) -> go.Figure:
    fig = go.Figure()
    if nivel == "todos":
        for nv, color in NIVEL_COLORS.items():
            sub = d.filter(pl.col("particion_1") == nv).sort("anio_alta")
            if len(sub) == 0:
                continue
            fig.add_trace(go.Scatter(
                x=sub["anio_alta"].to_list(),
                y=sub["becarios"].to_list(),
                mode="lines+markers",
                name=NIVEL_LABELS[nv],
                line=dict(color=color, width=2),
                marker=dict(size=6),
                hovertemplate=f"<b>{NIVEL_LABELS[nv]}</b><br>%{{x}}: %{{y:,}}<extra></extra>",
            ))
    else:
        sub = d.filter(pl.col("particion_1") == nivel).sort("anio_alta")
        color = NIVEL_COLORS.get(nivel, "#2E86AB")
        fig.add_trace(go.Scatter(
            x=sub["anio_alta"].to_list(),
            y=sub["becarios"].to_list(),
            mode="lines+markers",
            name=NIVEL_LABELS.get(nivel, nivel),
            line=dict(color=color, width=2),
            marker=dict(size=6),
            hovertemplate="<b>%{x}</b>: %{y:,} becarios<extra></extra>",
        ))
    fig.update_layout(
        **CHART_LAYOUT,
        height=360,
        legend=dict(orientation="h", y=1.1, x=0),
        xaxis=dict(type="category", gridcolor="#334155"),
        yaxis=dict(gridcolor="#334155"),
        title=dict(text="Becarios por año de alta al programa", font_size=13),
    )
    return fig


def fig_pendientes(d: pl.DataFrame) -> go.Figure:
    """% becas con monto pendiente (BECA=0) por estado."""
    sub = d.filter(pl.col("pendientes") > 0).sort("pct_pendiente", descending=True).head(32)
    n = len(sub)
    if n == 0:
        fig = go.Figure()
        fig.update_layout(**CHART_LAYOUT, height=200,
                          title=dict(text="Sin becas pendientes en este nivel", font_size=13))
        return fig

    colors = ["#E84855" if p > 50 else "#F4A261" if p > 25 else "#3BB273"
              for p in sub["pct_pendiente"].to_list()]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=sub["pct_pendiente"].to_list(),
        y=sub["NOM_EDO"].to_list(),
        orientation="h",
        marker_color=colors,
        customdata=list(zip(sub["pendientes"].to_list(), sub["becarios"].to_list())),
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Pendientes: %{customdata[0]:,}<br>"
            "Con pago: %{customdata[1]:,}<br>"
            "%{x:.1f}% pendiente<extra></extra>"
        ),
    ))
    fig.update_layout(
        **CHART_LAYOUT,
        height=max(300, n * 26 + 80),
        yaxis=dict(autorange="reversed", gridcolor="#334155"),
        xaxis=dict(gridcolor="#334155", title="% con monto en 0"),
        title=dict(text="% becas con pago pendiente (BECA=0) por estado", font_size=13),
    )
    return fig


# ── KPIs ─────────────────────────────────────────────────────────────────────

def kpi_cards(d: pl.DataFrame, nivel: str) -> list:
    total_becarios = int(d["becarios"].sum())
    total_monto = int(d["monto_total"].sum())
    total_pendientes = int(d["pendientes"].sum())
    total_registros = total_becarios + total_pendientes
    pct_pend = total_pendientes / total_registros * 100 if total_registros > 0 else 0

    def card(label, value, color="#CBD5E1"):
        return dbc.Col(html.Div([
            html.P(label, style={"color": "#64748B", "margin": "0", "fontSize": "12px"}),
            html.H4(value, style={"color": color, "margin": "4px 0 0 0"}),
        ], style=CARD_STYLE))

    cards = [
        card("Becarios con pago", f"{total_becarios:,}", "#2E86AB"),
        card("Monto total distribuido", f"${total_monto / 1e9:.2f} mil mill.", "#3BB273"),
        card("Becas con monto pendiente", f"{total_pendientes:,}  ({pct_pend:.1f}%)",
             "#E84855" if pct_pend > 30 else "#F4A261"),
    ]
    return cards


# ── Layout ───────────────────────────────────────────────────────────────────

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SLATE])
app.title = "Becas Bienestar CNBBBJ"

app.layout = dbc.Container(fluid=True, style={"backgroundColor": "#0F172A", "minHeight": "100vh", "padding": "24px"}, children=[
    html.H2("Becas Bienestar — CNBBBJ", style={"color": "#F8FAFC", "marginBottom": "4px"}),
    html.P("4to trimestre 2025 · 24.6 millones de registros", style={"color": "#64748B", "marginBottom": "20px"}),

    dbc.Row([
        dbc.Col([
            html.Label("Nivel educativo", style={"color": "#94A3B8", "fontSize": "12px"}),
            dcc.Dropdown(
                id="nivel-dropdown",
                options=[{"label": v, "value": k} for k, v in NIVEL_LABELS.items()],
                value="todos",
                clearable=False,
                style={"backgroundColor": "#1E293B", "color": "#F8FAFC", "border": "1px solid #334155"},
            ),
        ], md=4),
    ], className="mb-3"),

    dbc.Row(id="kpi-row", className="mb-3 g-3"),

    dbc.Tabs(style={"marginBottom": "16px"}, children=[
        dbc.Tab(label="Becarios por estado", tab_style=TAB_STYLE, active_tab_style=TAB_SEL, children=[
            dbc.Row([
                dbc.Col(dcc.Graph(id="graph-ranking"), md=6),
                dbc.Col(dcc.Graph(id="graph-monto"), md=6),
            ], className="g-2"),
        ]),
        dbc.Tab(label="Evolución temporal", tab_style=TAB_STYLE, active_tab_style=TAB_SEL, children=[
            dcc.Graph(id="graph-timeline"),
        ]),
        dbc.Tab(label="Becas pendientes", tab_style=TAB_STYLE, active_tab_style=TAB_SEL, children=[
            dcc.Graph(id="graph-pendientes"),
        ]),
    ]),
])


# ── Callback ─────────────────────────────────────────────────────────────────

@app.callback(
    Output("kpi-row", "children"),
    Output("graph-ranking", "figure"),
    Output("graph-monto", "figure"),
    Output("graph-timeline", "figure"),
    Output("graph-pendientes", "figure"),
    Input("nivel-dropdown", "value"),
)
def update_all(nivel):
    if nivel == "todos":
        d = (
            agg_estado
            .group_by("NOM_EDO")
            .agg([
                pl.col("becarios").sum(),
                pl.col("monto_total").sum(),
                pl.col("pendientes").sum(),
            ])
            .with_columns(
                (pl.col("pendientes") / (pl.col("pendientes") + pl.col("becarios")) * 100)
                .alias("pct_pendiente")
            )
        ).collect()
        d_anio = q_anio.collect()
    else:
        d = agg_estado.filter(pl.col("particion_1") == nivel).collect()
        d_anio = q_anio.filter(pl.col("particion_1") == nivel).collect()

    return (
        kpi_cards(d, nivel),
        fig_ranking(d),
        fig_monto(d),
        fig_timeline(d_anio, nivel),
        fig_pendientes(d),
    )


if __name__ == "__main__":
    app.run(debug=True, port=8060)
