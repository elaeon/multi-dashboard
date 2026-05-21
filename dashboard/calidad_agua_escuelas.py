import polars as pl
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc

# ── Data loading & cleaning ───────────────────────────────────────────────────

_RAW = pl.read_csv(
    "data/calidad_agua_escuelas.csv",
    null_values=["-", "N/A", "NA", "_", ""],
    infer_schema_length=10000,
)

# Keep only rows with a state (the rest are empty rows)
df = _RAW.filter(pl.col("Estado").is_not_null())

# Normalize Estado names
df = df.with_columns(
    pl.col("Estado")
    .str.to_titlecase()
    .str.strip_chars()
    .str.replace("Chiápas", "Chiapas")
    .str.replace("Chipas", "Chiapas")
    .alias("Estado")
)

# Valid years
df = df.with_columns(
    pl.when(pl.col("Año").is_between(2010, 2026)).then(pl.col("Año")).otherwise(None).alias("Año")
)

VALID_YEARS = sorted(df["Año"].drop_nulls().unique().to_list())


def _classify_risk(val: str | None) -> str | None:
    if val is None:
        return None
    v = val.lower()
    if "muy alto" in v or "alto riesgo" in v or "no san" in v or "no apto" in v:
        return "Alto riesgo"
    if "intermedio" in v:
        return "Intermedio"
    if "bajo" in v or "apta" in v or "segura" in v or "sin riesgo" in v or "apto" in v:
        return "Bajo riesgo"
    return None


def _classify_supply(val: str | None) -> str:
    if val is None:
        return "Desconocido"
    v = val.upper().strip()
    if "SAE" in v:
        return "SAE"
    if "SCALL" in v:
        return "SCALL"
    if "MANANTIAL" in v:
        return "Manantial"
    if "POZO" in v:
        return "Pozo"
    if "RÍO" in v or "RIO" in v:
        return "Río"
    if "LLUVIA" in v or "LLOV" in v:
        return "Lluvia"
    return "Otro"


df = df.with_columns(
    pl.col("Interpretación de análisis bacteriológico 1")
    .map_elements(_classify_risk, return_dtype=pl.String)
    .alias("riesgo_bact"),
    pl.col("Tipo de abastecimiento")
    .map_elements(_classify_supply, return_dtype=pl.String)
    .alias("abastecimiento"),
    pl.col("pH").cast(pl.Float64, strict=False).alias("pH_num"),
    pl.col("Turbidez").cast(pl.Float64, strict=False).alias("Turbidez_num"),
    pl.col("e__coli_1").cast(pl.Float64, strict=False).alias("ecoli_num"),
    pl.col("Cloro residual").cast(pl.Float64, strict=False).alias("cloro_num"),
)

# Plausible ranges only
df = df.with_columns(
    pl.when(pl.col("pH_num").is_between(0, 14)).then(pl.col("pH_num")).otherwise(None).alias("pH_num"),
    pl.when(pl.col("Turbidez_num") >= 0).then(pl.col("Turbidez_num")).otherwise(None).alias("Turbidez_num"),
    pl.when(pl.col("ecoli_num") >= 0).then(pl.col("ecoli_num")).otherwise(None).alias("ecoli_num"),
    pl.when(pl.col("cloro_num") >= 0).then(pl.col("cloro_num")).otherwise(None).alias("cloro_num"),
)

RISK_COLORS = {
    "Alto riesgo": "#E84855",
    "Intermedio":  "#F4A261",
    "Bajo riesgo": "#3BB273",
}

MUNICIPALITIES = (
    df.filter(pl.col("riesgo_bact").is_not_null())
    .group_by("Municipio")
    .agg(pl.len().alias("n"))
    .filter(pl.col("n") >= 20)
    .sort("Municipio")["Municipio"]
    .to_list()
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

def fig_trend_year(d: pl.DataFrame) -> go.Figure:
    base = d.filter(pl.col("Año").is_not_null() & pl.col("riesgo_bact").is_not_null())
    if len(base) == 0:
        return go.Figure()

    years = sorted(base["Año"].unique().to_list())
    traces = {}
    for risk in ["Alto riesgo", "Intermedio", "Bajo riesgo"]:
        agg = (
            base.filter(pl.col("riesgo_bact") == risk)
            .group_by("Año").agg(pl.len().alias("n"))
        )
        counts = {row["Año"]: row["n"] for row in agg.iter_rows(named=True)}
        traces[risk] = [counts.get(y, 0) for y in years]

    totals = [sum(traces[r][i] for r in traces) for i in range(len(years))]

    fig = go.Figure()
    for risk in ["Bajo riesgo", "Intermedio", "Alto riesgo"]:
        pcts = [
            round(traces[risk][i] / totals[i] * 100, 1) if totals[i] > 0 else 0
            for i in range(len(years))
        ]
        fig.add_trace(go.Bar(
            name=risk,
            x=[str(y) for y in years],
            y=pcts,
            marker_color=RISK_COLORS[risk],
            text=[f"{p:.0f}%" if p > 5 else "" for p in pcts],
            textposition="inside",
            hovertemplate=f"<b>{risk}</b><br>Año: %{{x}}<br>%{{y:.1f}}%<extra></extra>",
        ))

    fig.update_layout(
        barmode="stack",
        title="Riesgo bacteriológico por año",
        height=380,
        xaxis=dict(gridcolor="#334155", title="Año"),
        yaxis=dict(gridcolor="#334155", ticksuffix="%", range=[0, 101], title="% de muestras"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font_color="#94A3B8",
                    orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        **{k: v for k, v in CHART_LAYOUT.items() if k not in ("xaxis", "yaxis", "margin")},
        margin=dict(t=50, b=40, l=10, r=10),
    )
    return fig


def fig_municipio_risk(d: pl.DataFrame) -> go.Figure:
    base = d.filter(pl.col("riesgo_bact").is_not_null() & pl.col("Municipio").is_not_null())
    totals = base.group_by("Municipio").agg(pl.len().alias("total"))
    alto = (
        base.filter(pl.col("riesgo_bact") == "Alto riesgo")
        .group_by("Municipio").agg(pl.len().alias("alto"))
    )
    result = (
        totals.join(alto, on="Municipio", how="left")
        .with_columns(pl.col("alto").fill_null(0))
        .filter(pl.col("total") >= 20)
        .with_columns((pl.col("alto") / pl.col("total") * 100).round(1).alias("pct_alto"))
        .sort("pct_alto", descending=True)
    )

    if len(result) == 0:
        return go.Figure()

    munis = result["Municipio"].to_list()
    pcts  = result["pct_alto"].to_list()
    ns    = result["total"].to_list()
    colors = [RISK_COLORS["Alto riesgo"] if p >= 60 else
              RISK_COLORS["Intermedio"] if p >= 30 else
              RISK_COLORS["Bajo riesgo"] for p in pcts]

    fig = go.Figure(go.Bar(
        x=pcts, y=munis, orientation="h",
        marker_color=colors,
        text=[f"{p:.0f}%  (n={n})" for p, n in zip(pcts, ns)],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Alto riesgo: %{x:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        title="% Alto riesgo por municipio (mín. 20 muestras)",
        height=max(320, len(munis) * 32 + 80),
        xaxis=dict(range=[0, 115], gridcolor="#334155", ticksuffix="%"),
        yaxis=dict(gridcolor="rgba(0,0,0,0)", autorange="reversed"),
        **{k: v for k, v in CHART_LAYOUT.items() if k not in ("xaxis", "yaxis", "margin")},
        margin=dict(t=40, b=40, l=10, r=120),
    )
    return fig


def fig_supply_risk(d: pl.DataFrame) -> go.Figure:
    base = d.filter(pl.col("riesgo_bact").is_not_null() & (pl.col("abastecimiento") != "Desconocido"))
    totals = base.group_by("abastecimiento").agg(pl.len().alias("total"))
    alto = (
        base.filter(pl.col("riesgo_bact") == "Alto riesgo")
        .group_by("abastecimiento").agg(pl.len().alias("alto"))
    )
    result = (
        totals.join(alto, on="abastecimiento", how="left")
        .with_columns(pl.col("alto").fill_null(0))
        .with_columns((pl.col("alto") / pl.col("total") * 100).round(1).alias("pct_alto"))
        .sort("pct_alto", descending=True)
    )

    supplies = result["abastecimiento"].to_list()
    pcts     = result["pct_alto"].to_list()
    ns       = result["total"].to_list()
    colors   = [RISK_COLORS["Alto riesgo"] if p >= 60 else
                RISK_COLORS["Intermedio"] if p >= 30 else
                RISK_COLORS["Bajo riesgo"] for p in pcts]

    fig = go.Figure(go.Bar(
        x=pcts, y=supplies, orientation="h",
        marker_color=colors,
        text=[f"{p:.0f}%  (n={n})" for p, n in zip(pcts, ns)],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Alto riesgo: %{x:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        title="% Alto riesgo por tipo de abastecimiento",
        height=max(280, len(supplies) * 40 + 80),
        xaxis=dict(range=[0, 115], gridcolor="#334155", ticksuffix="%"),
        yaxis=dict(gridcolor="rgba(0,0,0,0)", autorange="reversed"),
        **{k: v for k, v in CHART_LAYOUT.items() if k not in ("xaxis", "yaxis", "margin")},
        margin=dict(t=40, b=40, l=10, r=120),
    )
    return fig


def fig_supply_breakdown(d: pl.DataFrame) -> go.Figure:
    base = d.filter(pl.col("riesgo_bact").is_not_null() & (pl.col("abastecimiento") != "Desconocido"))
    if len(base) == 0:
        return go.Figure()

    totals_df = base.group_by("abastecimiento").agg(pl.len().alias("total"))
    total_map = {r["abastecimiento"]: r["total"] for r in totals_df.iter_rows(named=True)}

    alto_df = (
        base.filter(pl.col("riesgo_bact") == "Alto riesgo")
        .group_by("abastecimiento").agg(pl.len().alias("alto"))
    )
    alto_map = {r["abastecimiento"]: r["alto"] for r in alto_df.iter_rows(named=True)}

    supply_order = sorted(
        total_map.keys(),
        key=lambda s: alto_map.get(s, 0) / total_map[s],
        reverse=True,
    )

    counts_map = {
        (r["abastecimiento"], r["riesgo_bact"]): r["n"]
        for r in base.group_by(["abastecimiento", "riesgo_bact"])
        .agg(pl.len().alias("n")).iter_rows(named=True)
    }

    fig = go.Figure()
    for risk in ["Alto riesgo", "Intermedio", "Bajo riesgo"]:
        pcts, ns_list = [], []
        for s in supply_order:
            n = counts_map.get((s, risk), 0)
            pcts.append(round(n / total_map[s] * 100, 1))
            ns_list.append(n)
        fig.add_trace(go.Bar(
            name=risk,
            y=supply_order, x=pcts, orientation="h",
            marker_color=RISK_COLORS[risk],
            text=[f"{p:.0f}%" if p > 8 else "" for p in pcts],
            textposition="inside",
            customdata=ns_list,
            hovertemplate=f"<b>{risk}</b><br>%{{y}}: %{{x:.1f}}%  (n=%{{customdata}})<extra></extra>",
        ))

    annotations = [
        dict(x=103, y=s, text=f"n={total_map[s]:,}",
             xref="x", yref="y", showarrow=False,
             font=dict(size=10, color="#94A3B8"), xanchor="left")
        for s in supply_order
    ]
    fig.update_layout(
        barmode="stack",
        title="Distribución de riesgo por tipo de abastecimiento",
        height=max(280, len(supply_order) * 40 + 80),
        xaxis=dict(range=[0, 120], gridcolor="#334155", ticksuffix="%"),
        yaxis=dict(gridcolor="rgba(0,0,0,0)"),
        legend=dict(orientation="h", y=1.08, x=0, bgcolor="rgba(0,0,0,0)"),
        annotations=annotations,
        **{k: v for k, v in CHART_LAYOUT.items() if k not in ("xaxis", "yaxis", "margin")},
        margin=dict(t=60, b=40, l=10, r=50),
    )
    return fig


def fig_symbol_map(d: pl.DataFrame) -> go.Figure:
    geo = d.with_columns(
        pl.col("Latitud").cast(pl.Float64, strict=False).alias("lat"),
        pl.col("Longitud").cast(pl.Float64, strict=False).alias("lon"),
    ).filter(
        pl.col("lat").is_between(14, 33)
        & pl.col("lon").is_between(-118, -86)
        & pl.col("riesgo_bact").is_not_null()
    )
    if len(geo) == 0:
        return go.Figure()

    center_lat = float(geo["lat"].mean())
    center_lon = float(geo["lon"].mean())

    fig = go.Figure()
    for risk, color in RISK_COLORS.items():
        subset = geo.filter(pl.col("riesgo_bact") == risk)
        if len(subset) == 0:
            continue
        escuelas   = subset["Escuela"].fill_null("Sin nombre").to_list()
        municipios = subset["Municipio"].fill_null("—").to_list()
        fig.add_trace(go.Scattermap(
            lat=subset["lat"].to_list(),
            lon=subset["lon"].to_list(),
            mode="markers",
            marker=dict(size=7, color=color, opacity=0.7),
            name=risk,
            customdata=list(zip(escuelas, municipios)),
            hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]}<extra></extra>",
        ))

    fig.update_layout(
        map=dict(style="carto-darkmatter",
                 center=dict(lat=center_lat, lon=center_lon), zoom=6),
        title="Ubicación de escuelas monitoreadas",
        height=500,
        margin=dict(t=40, b=10, l=0, r=0),
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#CBD5E1",
        legend=dict(orientation="h", y=-0.05, x=0, bgcolor="rgba(0,0,0,0)"),
    )
    return fig


def fig_ecoli_dist(d: pl.DataFrame) -> go.Figure:
    vals = d["ecoli_num"].drop_nulls().filter(pl.Series(d["ecoli_num"].drop_nulls()) <= 200)
    if len(vals) == 0:
        return go.Figure()
    fig = go.Figure(go.Histogram(
        x=vals.to_list(),
        nbinsx=40,
        marker_color="#E84855",
        opacity=0.8,
    ))
    fig.add_vline(x=0, line=dict(color="#3BB273", width=2, dash="dash"))
    fig.update_layout(
        title="Distribución de E. coli (UFC/100mL)",
        height=320,
        xaxis=dict(gridcolor="#334155", title="E. coli (UFC/100mL)"),
        yaxis=dict(gridcolor="#334155", title="Muestras"),
        **{k: v for k, v in CHART_LAYOUT.items() if k not in ("xaxis", "yaxis")},
    )
    return fig


def compute_kpis(d: pl.DataFrame):
    n_total = len(d)
    with_risk = d.filter(pl.col("riesgo_bact").is_not_null())
    n_risk = len(with_risk)
    pct_alto = float((with_risk["riesgo_bact"] == "Alto riesgo").sum() / n_risk * 100) if n_risk > 0 else 0
    pct_bajo = float((with_risk["riesgo_bact"] == "Bajo riesgo").sum() / n_risk * 100) if n_risk > 0 else 0
    n_munis  = d["Municipio"].drop_nulls().n_unique()
    return n_total, pct_alto, pct_bajo, n_munis


# ── Layout ────────────────────────────────────────────────────────────────────

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP],
           title="Calidad del Agua en Escuelas")

year_options = [{"label": "Todos los años", "value": "all"}] + [
    {"label": str(y), "value": y} for y in VALID_YEARS
]
muni_options = [{"label": "Todos los municipios", "value": "all"}] + [
    {"label": m, "value": m} for m in sorted(MUNICIPALITIES)
]

app.layout = html.Div(
    style={"backgroundColor": "#0F172A", "minHeight": "100vh", "padding": "24px"},
    children=[
        html.H2("Calidad del Agua en Escuelas — Chiapas y México",
                style={"color": "#F8FAFC", "fontWeight": "700", "marginBottom": "4px"}),
        html.P("Análisis bacteriológico y fisicoquímico del agua en planteles educativos (2014–2026)",
               style={"color": "#94A3B8", "marginBottom": "24px"}),

        # Filters
        dbc.Row(style={"marginBottom": "20px"}, children=[
            dbc.Col([
                html.Label("Año", style={"color": "#94A3B8", "fontSize": "12px"}),
                dcc.Dropdown(
                    id="year-filter", options=year_options, value="all", clearable=False,
                    style={"backgroundColor": "#1E293B", "color": "#CBD5E1", "border": "1px solid #334155"},
                ),
            ], md=4),
            dbc.Col([
                html.Label("Municipio", style={"color": "#94A3B8", "fontSize": "12px"}),
                dcc.Dropdown(
                    id="muni-filter", options=muni_options, value="all", clearable=False,
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
                        dbc.Col(dcc.Graph(id="graph-trend"), md=7),
                        dbc.Col(dcc.Graph(id="graph-muni"), md=5),
                    ]),
                    dbc.Row(className="g-3", children=[
                        dbc.Col(dcc.Graph(id="graph-map"), md=12),
                    ]),
                ]),
            ]),

            dcc.Tab(label="Tipo de Abastecimiento", style=TAB_STYLE, selected_style=TAB_SEL, children=[
                html.Div(style={"paddingTop": "20px"}, children=[
                    dbc.Row(className="g-3", children=[
                        dbc.Col(dcc.Graph(id="graph-supply-risk"), md=7),
                        dbc.Col(dcc.Graph(id="graph-supply-donut"), md=5),
                    ]),
                    dbc.Row(className="g-3", style={"marginTop": "20px"}, children=[
                        dbc.Col(dcc.Graph(id="graph-ecoli"), md=6),
                        dbc.Col(
                            html.Div([
                                html.P("Nota metodológica",
                                       style={"color": "#F8FAFC", "fontWeight": "600", "marginBottom": "8px"}),
                                html.P(
                                    "SAE = Sistema de Agua de Escuela (red de distribución intraescolar). "
                                    "SCALL = Sistema Comunitario de Agua a Lluvia. "
                                    "El umbral de riesgo bacteriológico sigue la clasificación del IMTA: "
                                    "Bajo (<1 UFC E. coli/100mL), Intermedio (1–10) y Alto (>10).",
                                    style={"color": "#94A3B8", "fontSize": "13px", "lineHeight": "1.6"},
                                ),
                            ], style={**CARD_STYLE, "textAlign": "left"}),
                            md=6,
                        ),
                    ]),
                ]),
            ]),
        ]),
    ],
)


# ── Callbacks ─────────────────────────────────────────────────────────────────

_CARD_STYLE = CARD_STYLE


@app.callback(
    Output("kpi-row", "children"),
    Output("graph-trend", "figure"),
    Output("graph-muni", "figure"),
    Output("graph-map", "figure"),
    Input("year-filter", "value"),
    Input("muni-filter", "value"),
)
def update_panorama(year, muni):
    d = df
    if year != "all":
        d = d.filter(pl.col("Año") == int(year))
    if muni != "all":
        d = d.filter(pl.col("Municipio") == muni)

    n_total, pct_alto, pct_bajo, n_munis = compute_kpis(d)
    kpis = [
        kpi_card("Total de muestras", f"{n_total:,}"),
        kpi_card("Alto riesgo bacteriológico", f"{pct_alto:.1f}%", "#E84855"),
        kpi_card("Agua segura (Bajo riesgo)", f"{pct_bajo:.1f}%", "#3BB273"),
        kpi_card("Municipios monitoreados", f"{n_munis:,}", "#2E86AB"),
    ]
    return kpis, fig_trend_year(d), fig_municipio_risk(d), fig_symbol_map(d)


@app.callback(
    Output("graph-supply-risk", "figure"),
    Output("graph-supply-donut", "figure"),
    Output("graph-ecoli", "figure"),
    Input("year-filter", "value"),
    Input("muni-filter", "value"),
)
def update_supply(year, muni):
    d = df
    if year != "all":
        d = d.filter(pl.col("Año") == int(year))
    if muni != "all":
        d = d.filter(pl.col("Municipio") == muni)
    return fig_supply_risk(d), fig_supply_breakdown(d), fig_ecoli_dist(d)


if __name__ == "__main__":
    app.run(debug=True, port=8055)
