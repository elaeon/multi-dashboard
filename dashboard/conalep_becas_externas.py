"""
Dashboard: Becas Externas CONALEP (2024–2025)

Filters: entidad dropdown + tipo_beca dropdown.
Pre-aggregates 3 summary frames at startup via a single parquet scan.

Run: uv run python dashboard/conalep_becas_externas.py
"""

import polars as pl
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc

# ── data ──────────────────────────────────────────────────────────────────────
def _fix_enc(x: str) -> str:
    if x and "Ã" in x:
        try:
            return x.encode("latin-1").decode("utf-8")
        except (UnicodeDecodeError, UnicodeEncodeError):
            return x
    return x


lf = (
    pl.scan_parquet("data/conalep_becas_externas.parquet")
    .with_columns(
        pl.col("plantel").map_elements(_fix_enc, return_dtype=pl.String),
        pl.col("monto_pesos").fill_null(0.0),
    )
)

q_ent = lf.group_by(["entidad", "tipo_beca", "anio"]).agg(
    pl.col("monto_pesos").sum().alias("monto"),
    pl.len().alias("n_becas"),
)
q_plantel = lf.group_by(["plantel", "entidad", "tipo_beca", "anio"]).agg(
    pl.col("monto_pesos").sum().alias("monto"),
    pl.len().alias("n_becas"),
)
q_otorg = lf.group_by(["otorgante", "entidad", "tipo_beca", "anio"]).agg(
    pl.col("monto_pesos").sum().alias("monto"),
    pl.len().alias("n_becas"),
)

agg_ent, agg_plantel, agg_otorg = pl.collect_all([q_ent, q_plantel, q_otorg])

ENTIDADES = sorted(agg_ent["entidad"].unique().to_list())
ENTIDAD_OPTIONS = [{"label": "Todas las entidades", "value": "__all__"}] + [
    {"label": e, "value": e} for e in ENTIDADES
]
TIPO_OPTIONS = [
    {"label": "Todos los tipos",  "value": "__all__"},
    {"label": "Beca directa",     "value": "Beca directa"},
    {"label": "Beca Indirecta",   "value": "Beca Indirecta"},
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

COLOR_MAIN   = "#2E86AB"
COLOR_DIRECT = "#3BB273"
COLOR_INDIR  = "#F4A261"
COLOR_MUTED  = "#475569"
TIPO_COLORS  = {"Beca directa": COLOR_DIRECT, "Beca Indirecta": COLOR_INDIR}


# ── filter helper ─────────────────────────────────────────────────────────────
def _apply(d: pl.DataFrame, ent: str, tipo: str) -> pl.DataFrame:
    if tipo != "__all__":
        d = d.filter(pl.col("tipo_beca") == tipo)
    if ent != "__all__":
        d = d.filter(pl.col("entidad") == ent)
    return d


# ── figure factories ──────────────────────────────────────────────────────────
def fig_entidad_bar(d_eg: pl.DataFrame, selected: str) -> go.Figure:
    """Monto total por entidad — all states, highlight selected."""
    agg = (
        d_eg.group_by("entidad").agg(pl.col("monto").sum()).sort("monto")
    )
    colors = [
        COLOR_MAIN if (selected == "__all__" or e == selected) else COLOR_MUTED
        for e in agg["entidad"].to_list()
    ]
    fig = go.Figure(go.Bar(
        x=agg["monto"].to_list(), y=agg["entidad"].to_list(),
        orientation="h", marker_color=colors,
        hovertemplate="<b>%{y}</b><br>$%{x:,.0f}<extra></extra>",
    ))
    # Finding 1: show top-state concentration
    if len(agg) > 1:
        total = float(agg["monto"].sum())
        top   = agg.tail(1)
        pct   = float(top["monto"].item()) / total * 100 if total else 0
        fig.add_annotation(
            x=0.98, y=0.02, xref="paper", yref="paper",
            text=f"<b>{top['entidad'].item()}</b>: {pct:.0f}% del total",
            showarrow=False, align="right",
            bgcolor="#0F172A", bordercolor="#334155", borderwidth=1,
            font=dict(size=11, color="#E84855"),
            xanchor="right", yanchor="bottom",
        )
    fig.update_layout(
        title="Monto total por entidad",
        height=max(320, len(agg) * 22 + 80),
        xaxis_title="Monto (pesos)",
        margin=dict(t=40, b=40, l=10, r=10),
        **CHART_LAYOUT,
    )
    return fig


def fig_tipo_bar(d: pl.DataFrame) -> go.Figure:
    agg = d.group_by("tipo_beca").agg(pl.col("monto").sum(), pl.col("n_becas").sum()).sort("tipo_beca")
    if agg.is_empty():
        return go.Figure().update_layout(title="Sin datos", height=240, **CHART_LAYOUT)
    total_m = float(agg["monto"].sum())
    total_b = int(agg["n_becas"].sum())
    fig = go.Figure()
    for row in agg.iter_rows(named=True):
        pct_m = row["monto"] / total_m * 100 if total_m else 0
        pct_b = row["n_becas"] / total_b * 100 if total_b else 0
        color = TIPO_COLORS.get(row["tipo_beca"], COLOR_MAIN)
        fig.add_trace(go.Bar(
            x=[pct_m], y=["Monto"], orientation="h", name=row["tipo_beca"],
            marker_color=color,
            text=f"{pct_m:.1f}%<br>${row['monto']/1e6:.1f}M",
            textposition="inside", insidetextanchor="middle",
            hovertemplate=f"<b>{row['tipo_beca']}</b><br>{pct_m:.1f}% del monto<br>${row['monto']:,.0f}<extra></extra>",
        ))
        fig.add_trace(go.Bar(
            x=[pct_b], y=["Becas"], orientation="h", name=row["tipo_beca"],
            marker_color=color, showlegend=False,
            text=f"{pct_b:.1f}%<br>{row['n_becas']:,}",
            textposition="inside", insidetextanchor="middle",
            hovertemplate=f"<b>{row['tipo_beca']}</b><br>{pct_b:.1f}% de becas<br>{row['n_becas']:,} becas<extra></extra>",
        ))
    fig.update_layout(
        barmode="stack",
        title="Distribución por tipo de beca",
        height=240,
        xaxis=dict(range=[0, 100], visible=False),
        yaxis=dict(gridcolor="rgba(0,0,0,0)"),
        legend=dict(orientation="h", y=-0.25, x=0, bgcolor="rgba(0,0,0,0)"),
        margin=dict(t=40, b=70, l=10, r=10),
        paper_bgcolor="rgba(0,0,0,0)", font_color="#CBD5E1",
    )
    return fig


def fig_avg_monto_tipo(d: pl.DataFrame) -> go.Figure:
    agg = (
        d.group_by("tipo_beca")
        .agg(pl.col("monto").sum(), pl.col("n_becas").sum())
        .with_columns((pl.col("monto") / pl.col("n_becas")).alias("avg_monto"))
        .sort("avg_monto")
    )
    if agg.is_empty():
        return go.Figure().update_layout(title="Sin datos", height=220, **CHART_LAYOUT)
    colors = [TIPO_COLORS.get(t, COLOR_MAIN) for t in agg["tipo_beca"].to_list()]
    fig = go.Figure(go.Bar(
        x=agg["avg_monto"].to_list(), y=agg["tipo_beca"].to_list(),
        orientation="h", marker_color=colors,
        hovertemplate="<b>%{y}</b><br>$%{x:,.0f} promedio<extra></extra>",
    ))
    fig.update_layout(
        title="Monto promedio por tipo de beca",
        height=220,
        xaxis_title="Monto promedio (pesos)",
        margin=dict(t=40, b=40, l=10, r=10),
        **CHART_LAYOUT,
    )
    return fig


def fig_yoy_bar(d_eg: pl.DataFrame) -> go.Figure:
    by_ent_yr = d_eg.group_by(["entidad", "anio"]).agg(pl.col("monto").sum())
    yr24 = by_ent_yr.filter(pl.col("anio") == 2024).rename({"monto": "m24"}).drop("anio")
    yr25 = by_ent_yr.filter(pl.col("anio") == 2025).rename({"monto": "m25"}).drop("anio")
    joined = yr24.join(yr25, on="entidad", how="inner")
    if joined.is_empty():
        return go.Figure().update_layout(title="Sin datos", height=320, **CHART_LAYOUT)
    joined = joined.with_columns(
        ((pl.col("m25") - pl.col("m24")) / pl.col("m24") * 100).alias("pct")
    ).sort("pct")
    colors = ["#E84855" if v < 0 else "#3BB273" for v in joined["pct"].to_list()]
    fig = go.Figure(go.Bar(
        x=joined["pct"].to_list(), y=joined["entidad"].to_list(),
        orientation="h", marker_color=colors,
        hovertemplate="<b>%{y}</b><br>%{x:+.1f}%<extra></extra>",
    ))
    fig.add_vline(x=0, line_width=1, line_color="#64748B")
    fig.update_layout(
        title="Cambio YoY por entidad (2024 → 2025)",
        height=max(320, len(joined) * 22 + 80),
        xaxis_title="Cambio % en monto",
        margin=dict(t=40, b=40, l=10, r=10),
        **CHART_LAYOUT,
    )
    return fig


def fig_anio_bar(d: pl.DataFrame) -> go.Figure:
    agg = (
        d.group_by(["anio", "tipo_beca"])
        .agg(pl.col("monto").sum())
        .sort("anio")
    )
    fig = go.Figure()
    for tipo in ["Beca directa", "Beca Indirecta"]:
        sub = agg.filter(pl.col("tipo_beca") == tipo)
        if sub.is_empty():
            continue
        fig.add_trace(go.Bar(
            x=sub["anio"].cast(pl.String).to_list(),
            y=sub["monto"].to_list(),
            name=tipo, marker_color=TIPO_COLORS[tipo],
            hovertemplate="<b>%{x}</b><br>$%{y:,.0f}<extra></extra>",
        ))
    fig.update_layout(
        barmode="group",
        title="Monto 2024 vs 2025",
        height=320,
        yaxis_title="Monto (pesos)",
        legend=dict(orientation="h", y=-0.22, x=0),
        margin=dict(t=40, b=70, l=10, r=10),
        **CHART_LAYOUT,
    )
    return fig


YEAR_COLORS = {2024: COLOR_MAIN, 2025: COLOR_INDIR}


def fig_top_otorg(d: pl.DataFrame, n: int = 20) -> go.Figure:
    top_names = (
        d.group_by("otorgante").agg(pl.col("monto").sum())
        .sort("monto", descending=True).head(n)
        ["otorgante"].to_list()
    )
    if not top_names:
        return go.Figure().update_layout(title="Sin datos", height=320, **CHART_LAYOUT)
    d_top = d.filter(pl.col("otorgante").is_in(top_names))
    ordered = (
        d_top.group_by("otorgante").agg(pl.col("monto").sum())
        .sort("monto")["otorgante"].to_list()
    )
    fig = go.Figure()
    for yr in sorted(d_top["anio"].unique().to_list()):
        sub = d_top.filter(pl.col("anio") == yr).group_by("otorgante").agg(pl.col("monto").sum())
        val_map = dict(zip(sub["otorgante"].to_list(), sub["monto"].to_list()))
        fig.add_trace(go.Bar(
            x=[val_map.get(o, 0) for o in ordered], y=ordered,
            orientation="h", name=str(yr),
            marker_color=YEAR_COLORS.get(yr, COLOR_MUTED),
            hovertemplate=f"<b>%{{y}}</b> ({yr})<br>$%{{x:,.0f}}<extra></extra>",
        ))
    fig.update_layout(
        barmode="stack",
        title=f"Top {n} otorgantes por monto",
        height=max(320, n * 26 + 80),
        xaxis_title="Monto (pesos)",
        legend=dict(orientation="h", y=-0.12, x=0),
        margin=dict(t=40, b=80, l=10, r=10),
        **CHART_LAYOUT,
    )
    return fig


def fig_top_plantel(d: pl.DataFrame, n: int = 20) -> go.Figure:
    top_n = (
        d.group_by(["plantel", "entidad"]).agg(pl.col("monto").sum())
        .sort("monto", descending=True).head(n).sort("monto")
    )
    if top_n.is_empty():
        return go.Figure().update_layout(title="Sin datos", height=320, **CHART_LAYOUT)
    ordered_keys   = [(r["plantel"], r["entidad"]) for r in top_n.iter_rows(named=True)]
    ordered_labels = [f"{p} ({e})" for p, e in ordered_keys]
    d_top = d.join(top_n.select(["plantel", "entidad"]), on=["plantel", "entidad"])
    fig = go.Figure()
    for yr in sorted(d_top["anio"].unique().to_list()):
        sub = (
            d_top.filter(pl.col("anio") == yr)
            .group_by(["plantel", "entidad"]).agg(pl.col("monto").sum())
        )
        val_map = {(r["plantel"], r["entidad"]): r["monto"] for r in sub.iter_rows(named=True)}
        fig.add_trace(go.Bar(
            x=[val_map.get(k, 0) for k in ordered_keys], y=ordered_labels,
            orientation="h", name=str(yr),
            marker_color=YEAR_COLORS.get(yr, COLOR_MUTED),
            hovertemplate=f"<b>%{{y}}</b> ({yr})<br>$%{{x:,.0f}}<extra></extra>",
        ))
    fig.update_layout(
        barmode="stack",
        title=f"Top {n} planteles por monto",
        height=max(320, n * 26 + 80),
        xaxis_title="Monto (pesos)",
        legend=dict(orientation="h", y=-0.05, x=0),
        margin=dict(t=40, b=50, l=10, r=10),
        **CHART_LAYOUT,
    )
    return fig


def compute_kpis(d_ent: pl.DataFrame, d_plantel: pl.DataFrame):
    total_becas = int(d_ent["n_becas"].sum())
    monto_total = float(d_ent["monto"].sum())
    monto_avg   = monto_total / total_becas if total_becas else 0.0
    n_planteles = int(d_plantel.select(pl.col("plantel").n_unique()).item()) if len(d_plantel) else 0
    return total_becas, monto_total, monto_avg, n_planteles


# ── layout ────────────────────────────────────────────────────────────────────
app = Dash(__name__, external_stylesheets=[dbc.themes.SLATE],
           title="Becas Externas CONALEP")

app.layout = dbc.Container(
    [
        html.H2("Becas Externas CONALEP", className="text-light mt-3 mb-1"),
        html.P("Transparencia CONALEP · 2024–2025",
               style={"color": "#64748B", "fontSize": "0.85rem", "marginBottom": "24px"}),

        dbc.Row([
            dbc.Col([
                html.Label("Entidad", className="text-secondary small"),
                dcc.Dropdown(
                    id="ent-dd", options=ENTIDAD_OPTIONS, value="__all__",
                    clearable=False,
                    style={"backgroundColor": "#1E293B", "color": "#CBD5E1",
                           "border": "1px solid #334155"},
                ),
            ], width=5),
            dbc.Col([
                html.Label("Tipo de beca", className="text-secondary small"),
                dcc.Dropdown(
                    id="tipo-dd", options=TIPO_OPTIONS, value="__all__",
                    clearable=False,
                    style={"backgroundColor": "#1E293B", "color": "#CBD5E1",
                           "border": "1px solid #334155"},
                ),
            ], width=4),
        ], className="mb-4 align-items-end"),

        dbc.Row(id="kpi-row", className="mb-4 g-3"),

        dbc.Row([
            dbc.Col(dcc.Graph(id="entidad-bar"), width=8),
            dbc.Col([
                dcc.Graph(id="tipo-donut"),
                dcc.Graph(id="anio-bar"),
            ], width=4),
        ], className="mb-4"),

        dbc.Row([
            dbc.Col(dcc.Graph(id="avg-monto-tipo"), width=4),
            dbc.Col(dcc.Graph(id="yoy-bar"),         width=8),
        ], className="mb-4"),

        dbc.Row([
            dbc.Col(dcc.Graph(id="top-otorg"),   width=6),
            dbc.Col(dcc.Graph(id="top-plantel"), width=6),
        ]),
    ],
    fluid=True,
    style={"backgroundColor": "#0F172A", "minHeight": "100vh", "padding": "20px"},
)


# ── callback ──────────────────────────────────────────────────────────────────
@app.callback(
    Output("kpi-row",        "children"),
    Output("entidad-bar",    "figure"),
    Output("tipo-donut",     "figure"),
    Output("anio-bar",       "figure"),
    Output("avg-monto-tipo", "figure"),
    Output("yoy-bar",        "figure"),
    Output("top-otorg",      "figure"),
    Output("top-plantel",    "figure"),
    Input("ent-dd",  "value"),
    Input("tipo-dd", "value"),
)
def update_all(ent, tipo):
    # entidad bar always shows all states (tipo filter only)
    d_eg = _apply(agg_ent, "__all__", tipo)
    # detail frames: both filters
    d_e  = _apply(agg_ent,    ent, tipo)
    d_p  = _apply(agg_plantel, ent, tipo)
    d_o  = _apply(agg_otorg,   ent, tipo)

    total_becas, monto_total, monto_avg, n_planteles = compute_kpis(d_e, d_p)

    def kpi_card(label, value, fmt="{:,.0f}", suffix=""):
        return dbc.Col(
            html.Div([
                html.P(label, style={"color": "#94A3B8", "fontSize": "0.78rem", "marginBottom": "4px"}),
                html.H4(fmt.format(value) + suffix,
                        style={"color": "#F8FAFC", "fontWeight": "600", "margin": 0}),
            ], style=CARD_STYLE),
            width=3,
        )

    kpis = [
        kpi_card("Total de becas",    total_becas),
        kpi_card("Monto total",       monto_total / 1_000_000, fmt="${:.2f}", suffix=" M"),
        kpi_card("Monto promedio",    monto_avg,   fmt="${:,.0f}"),
        kpi_card("Planteles activos", n_planteles),
    ]

    return (
        kpis,
        fig_entidad_bar(d_eg, ent),
        fig_tipo_bar(d_e),
        fig_anio_bar(d_e),
        fig_avg_monto_tipo(d_e),
        fig_yoy_bar(d_eg),
        fig_top_otorg(d_o),
        fig_top_plantel(d_p),
    )


if __name__ == "__main__":
    app.run(debug=True, port=8059)
