import polars as pl
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc

# ── Theme ─────────────────────────────────────────────────────────────────────
CARD_STYLE = {
    "background": "#1E293B", "border": "1px solid #334155",
    "borderRadius": "8px", "padding": "16px", "textAlign": "center",
}
TAB_STYLE = {"backgroundColor": "#0F172A", "color": "#94A3B8", "borderTop": "none"}
TAB_SEL = {
    "backgroundColor": "#1E293B", "color": "#F8FAFC",
    "borderTop": "2px solid #2E86AB", "fontWeight": "600",
}
FOCUS, CONTEXT = "#2E86AB", "#475569"


def _theme(fig: go.Figure, height: int = 420, **kwargs) -> go.Figure:
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#CBD5E1", height=height, **kwargs,
    )
    fig.update_xaxes(gridcolor="#334155")
    fig.update_yaxes(gridcolor="#334155")
    return fig


# ── Data loading (lazy for 138 MB bilateral file) ─────────────────────────────
_BI = "data/migration_flows/mig_bilateral.csv"
_UNI = "data/migration_flows/mig_unilateral.csv"

lf_bi = (
    pl.scan_csv(_BI)
    .filter(
        ((pl.col("Origin ISO") == "MEX") | (pl.col("Destination ISO") == "MEX"))
        & ~((pl.col("Origin ISO") == "MEX") & (pl.col("Destination ISO") == "MEX"))
    )
    .select(["Origin ISO", "Destination ISO", "Year", "stock_mean"])
)

# Two aggregation queries — collect_all scans the file once
q_emi = (
    lf_bi.filter(pl.col("Origin ISO") == "MEX")
    .group_by(["Destination ISO", "Year"])
    .agg(pl.col("stock_mean").sum())
)
q_imm = (
    lf_bi.filter(pl.col("Destination ISO") == "MEX")
    .group_by(["Origin ISO", "Year"])
    .agg(pl.col("stock_mean").sum())
)
emi_dest_yr, imm_orig_yr = pl.collect_all([q_emi, q_imm])

uni = pl.read_csv(_UNI).filter(pl.col("Country ISO") == "MEX").sort("Year")

# Pre-compute US share trend
_total_by_yr = emi_dest_yr.group_by("Year").agg(pl.col("stock_mean").sum().alias("total"))
_us_by_yr = emi_dest_yr.filter(pl.col("Destination ISO") == "USA").rename({"stock_mean": "us"}).select(["Year", "us"])
us_share_trend = (
    _total_by_yr.join(_us_by_yr, on="Year")
    .with_columns((pl.col("us") / pl.col("total") * 100).alias("share"))
    .sort("Year")
)

MIN_YEAR = int(uni["Year"].min())
MAX_YEAR = int(uni["Year"].max())

# ── KPI values ────────────────────────────────────────────────────────────────
def _val(col: str, year: int) -> float:
    rows = uni.filter(pl.col("Year") == year)[col]
    return float(rows[0]) if len(rows) > 0 and rows[0] is not None else float("nan")


emi_pop_2024 = _val("emi_pop", 2024)
imm_pop_2024 = _val("imm_pop", 2024)
imm_pop_1990 = _val("imm_pop", 1990)
net_2023 = _val("net", 2023)
us_share_2024 = float(us_share_trend.filter(pl.col("Year") == 2024)["share"][0])
us_share_1990 = float(us_share_trend.filter(pl.col("Year") == 1990)["share"][0])


# ── Figure factories ──────────────────────────────────────────────────────────
def fig_flows() -> go.Figure:
    d = uni.filter(pl.col("emi").is_not_null()).sort("Year")
    emi_2022 = float(d.filter(pl.col("Year") == 2022)["emi"][0])
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=d["Year"].to_list(), y=(d["emi"] / 1e3).to_list(),
        name="Emigración", mode="lines", line=dict(color="#E84855", width=2.5),
        hovertemplate="<b>%{x}</b>: %{y:.0f}k emi<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=d["Year"].to_list(), y=(d["imm"] / 1e3).to_list(),
        name="Inmigración", mode="lines", line=dict(color="#3BB273", width=2.5),
        hovertemplate="<b>%{x}</b>: %{y:.0f}k imm<extra></extra>",
    ))
    fig.add_vline(x=2020, line_dash="dot", line_color="#94A3B8",
                  annotation_text="2020: mínimo histórico\nemi (280k)",
                  annotation_font_color="#94A3B8", annotation_position="top left")
    fig.add_annotation(
        x=2022, y=emi_2022 / 1e3 + 40,
        text="<b>2022: récord histórico (765k)</b>",
        font=dict(color="#E84855", size=11), showarrow=True,
        arrowcolor="#94A3B8", ax=-60, ay=-30,
    )
    return _theme(fig,
        title=dict(text=(
            "<b>Emigración alcanzó récord histórico en 2022, 2.7× el mínimo de 2020</b>"
            "<br><sup style='color:#94A3B8'>Flujos anuales, miles de personas, 1990–2023</sup>"
        )),
        yaxis_title="Miles",
        legend=dict(orientation="h", y=-0.15, x=0),
        margin=dict(t=80, b=70, l=60, r=20),
    )


def fig_net() -> go.Figure:
    d = uni.filter(pl.col("net").is_not_null()).sort("Year")
    nets_k = (d["net"] / 1e3).to_list()
    colors = ["#3BB273" if v > 0 else "#E84855" for v in nets_k]
    fig = go.Figure()
    fig.add_vrect(x0=2015, x1=2020, fillcolor="rgba(59,178,115,0.08)", line_width=0)
    fig.add_hline(y=0, line_dash="dash", line_color="#64748B",
                  annotation_text="Equilibrio", annotation_font_color="#94A3B8",
                  annotation_position="right")
    fig.add_trace(go.Scatter(
        x=d["Year"].to_list(), y=nets_k,
        mode="lines+markers",
        line=dict(color="#2E86AB", width=2.5),
        marker=dict(color=colors, size=7),
        hovertemplate="<b>%{x}</b>: %{y:.0f}k<extra></extra>",
        showlegend=False,
    ))
    fig.add_annotation(x=2017, y=2, text="Era receptora\n2015–2020",
                       font=dict(color="#3BB273", size=10), showarrow=False)
    return _theme(fig,
        title=dict(text=(
            "<b>México fue receptor neto 2015–2020; revirtió a emigración neta en 2021</b>"
            "<br><sup style='color:#94A3B8'>Migración neta anual, miles de personas</sup>"
        )),
        yaxis_title="Miles",
        margin=dict(t=80, b=40, l=60, r=60),
    )


def fig_top_destinations(year: int, n: int = 15) -> go.Figure:
    year_data = emi_dest_yr.filter(pl.col("Year") == year)
    total_all = float(year_data["stock_mean"].sum())
    d = year_data.sort("stock_mean", descending=True).head(n)
    entities = d["Destination ISO"].to_list()
    values_m = (d["stock_mean"] / 1e6).to_list()
    colors = [FOCUS if e == "USA" else CONTEXT for e in entities]
    labels = [f"{v:.2f}M" if v >= 0.1 else f"{v*1000:.0f}k" for v in values_m]
    top3 = float(d.head(3)["stock_mean"].sum()) / total_all * 100
    fig = go.Figure(go.Bar(
        x=values_m, y=entities, orientation="h",
        marker_color=colors,
        text=labels, textposition="inside", insidetextanchor="middle",
        hovertemplate="<b>%{y}</b>: %{x:.3f}M<extra></extra>",
    ))
    return _theme(fig, height=max(300, n * 28 + 100),
        title=dict(text=(
            f"<b>{top3:.0f}% de los mexicanos en el exterior viven en solo 3 países ({year})</b>"
            f"<br><sup style='color:#94A3B8'>Stock migratorio, millones de personas</sup>"
        )),
        xaxis_title="Millones",
        yaxis=dict(autorange="reversed", gridcolor="rgba(0,0,0,0)"),
        margin=dict(t=80, b=40, l=60, r=20),
    )


def fig_us_share() -> go.Figure:
    d = us_share_trend.sort("Year")
    fig = go.Figure(go.Scatter(
        x=d["Year"].to_list(), y=d["share"].to_list(),
        mode="lines+markers", line=dict(color="#2E86AB", width=2.5),
        marker=dict(size=5),
        hovertemplate="<b>%{x}</b>: %{y:.1f}%<extra></extra>",
        showlegend=False,
    ))
    fig.update_yaxes(range=[90, 100], title_text="%")
    return _theme(fig, height=340,
        title=dict(text=(
            "<b>La dependencia de EUA bajó solo 3 puntos en 34 años (98.8% → 95.6%)</b>"
            "<br><sup style='color:#94A3B8'>% del stock emigrante total residente en EUA</sup>"
        )),
        margin=dict(t=80, b=40, l=60, r=20),
    )


_NON_US = ["CAN", "ESP", "DEU", "GTM", "FRA", "GBR"]
_NON_US_COLORS = {
    "CAN": "#2E86AB", "ESP": "#F4A261", "DEU": "#3BB273",
    "GTM": "#E84855", "FRA": "#94A3B8", "GBR": "#64748B",
}


def fig_non_us_growth() -> go.Figure:
    fig = go.Figure()
    for iso in _NON_US:
        sub = emi_dest_yr.filter(pl.col("Destination ISO") == iso).sort("Year")
        fig.add_trace(go.Scatter(
            x=sub["Year"].to_list(), y=(sub["stock_mean"] / 1e3).to_list(),
            name=iso, mode="lines+markers",
            line=dict(color=_NON_US_COLORS.get(iso, CONTEXT), width=1.8),
            marker=dict(size=4),
            hovertemplate=f"<b>{iso}</b> %{{x}}: %{{y:.1f}}k<extra></extra>",
        ))
    return _theme(fig, height=320,
        title=dict(text=(
            "<b>Canadá y España lideran destinos alternativos — pero suman < 2% del total</b>"
            "<br><sup style='color:#94A3B8'>Stock migratorio, miles de personas, excluyendo EUA</sup>"
        )),
        yaxis_title="Miles",
        legend=dict(orientation="h", y=-0.2, x=0),
        margin=dict(t=80, b=70, l=60, r=20),
    )


def fig_top_sources(year: int, n: int = 15) -> go.Figure:
    d = (
        imm_orig_yr.filter(pl.col("Year") == year)
        .sort("stock_mean", descending=True)
        .head(n)
    )
    entities = d["Origin ISO"].to_list()
    values_k = (d["stock_mean"] / 1e3).to_list()
    highlight = {"GTM", "HND", "VEN", "HTI"}
    colors = [FOCUS if e in highlight else CONTEXT for e in entities]
    fig = go.Figure(go.Bar(
        x=values_k, y=entities, orientation="h",
        marker_color=colors,
        text=[f"{v:.0f}k" for v in values_k],
        textposition="inside", insidetextanchor="middle",
        hovertemplate="<b>%{y}</b>: %{x:.1f}k<extra></extra>",
    ))
    return _theme(fig, height=max(300, n * 28 + 100),
        title=dict(text=(
            f"<b>La población inmigrante en México se cuadruplicó desde 1990 ({year})</b>"
            f"<br><sup style='color:#94A3B8'>Stock de inmigrantes, miles de personas</sup>"
        )),
        xaxis_title="Miles",
        yaxis=dict(autorange="reversed", gridcolor="rgba(0,0,0,0)"),
        margin=dict(t=80, b=40, l=60, r=20),
    )


_SURGE = ["HTI", "VEN", "HND", "GTM", "CUB"]
_SURGE_COLORS = {
    "HTI": "#E84855", "VEN": "#F4A261", "HND": "#2E86AB",
    "GTM": "#3BB273", "CUB": "#94A3B8",
}


def fig_surge_sources() -> go.Figure:
    fig = go.Figure()
    for iso in _SURGE:
        sub = imm_orig_yr.filter(pl.col("Origin ISO") == iso).sort("Year")
        fig.add_trace(go.Scatter(
            x=sub["Year"].to_list(), y=(sub["stock_mean"] / 1e3).to_list(),
            name=iso, mode="lines+markers",
            line=dict(color=_SURGE_COLORS.get(iso, CONTEXT), width=1.8),
            marker=dict(size=5),
            hovertemplate=f"<b>{iso}</b> %{{x}}: %{{y:.1f}}k<extra></extra>",
        ))
    fig.add_vline(x=2015, line_dash="dot", line_color="#94A3B8",
                  annotation_text="2015: aceleración",
                  annotation_font_color="#94A3B8", annotation_position="top left")
    return _theme(fig, height=420,
        title=dict(text=(
            "<b>Haití creció ×15 en 9 años (2015–2024); Venezuela ×8, Honduras ×7</b>"
            "<br><sup style='color:#94A3B8'>Stock de inmigrantes en México, miles de personas</sup>"
        )),
        yaxis_title="Miles",
        legend=dict(orientation="h", y=-0.15, x=0),
        margin=dict(t=80, b=70, l=60, r=20),
    )


# ── KPI card ──────────────────────────────────────────────────────────────────
def kpi_card(label: str, value: str, delta: str, color: str) -> dbc.Card:
    return dbc.Card(
        dbc.CardBody([
            html.P(label, style={"color": "#94A3B8", "fontSize": "0.8rem", "marginBottom": "4px"}),
            html.H3(value, style={"color": "#F8FAFC", "margin": "0", "fontSize": "1.5rem"}),
            html.Span(delta, style={"color": color, "fontSize": "0.82rem"}),
        ]),
        style=CARD_STYLE,
    )


# ── Layout ────────────────────────────────────────────────────────────────────
app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
app.title = "Migración México"

_slider_marks = {y: str(y) for y in range(MIN_YEAR, MAX_YEAR + 1, 5)}

kpis = dbc.Row([
    dbc.Col(kpi_card(
        "Mexicanos en el exterior (2024)",
        f"{emi_pop_2024/1e6:.1f}M",
        "vs 5.4M en 1990 (+110%)", "#3BB273",
    ), md=3),
    dbc.Col(kpi_card(
        "Inmigrantes en México (2024)",
        f"{imm_pop_2024/1e6:.2f}M",
        f"vs 454k en 1990 (×{imm_pop_2024/imm_pop_1990:.1f})", "#3BB273",
    ), md=3),
    dbc.Col(kpi_card(
        "Migración neta (2023)",
        f"{net_2023/1e3:+.0f}k",
        "▼ Revirtió a emigración neta en 2021", "#E84855",
    ), md=3),
    dbc.Col(kpi_card(
        "% emigrantes en EUA (2024)",
        f"{us_share_2024:.1f}%",
        f"vs {us_share_1990:.1f}% en 1990 (−{us_share_1990-us_share_2024:.1f} pp)", "#F4A261",
    ), md=3),
], className="g-3 mb-4")

app.layout = html.Div(
    style={"backgroundColor": "#0F172A", "minHeight": "100vh", "padding": "24px"},
    children=[
        html.H2("Flujos Migratorios de México 1990–2024",
                style={"color": "#F8FAFC", "marginBottom": "4px"}),
        html.P(
            "Estimaciones de stock y flujos migratorios · Fuente: ThGaskin/Migration_flows (HuggingFace)",
            style={"color": "#64748B", "marginBottom": "24px", "fontSize": "0.85rem"},
        ),
        kpis,
        dcc.Tabs([
            dcc.Tab(label="Balance Migratorio", style=TAB_STYLE, selected_style=TAB_SEL, children=[
                dbc.Row([
                    dbc.Col(dcc.Graph(id="fig-flows", figure=fig_flows()), md=6),
                    dbc.Col(dcc.Graph(id="fig-net", figure=fig_net()), md=6),
                ], className="g-3 mt-3"),
            ]),
            dcc.Tab(label="Mexicanos en el Mundo", style=TAB_STYLE, selected_style=TAB_SEL, children=[
                dbc.Row([
                    dbc.Col([
                        html.Label("Año:", style={"color": "#94A3B8", "fontSize": "0.85rem"}),
                        dcc.Slider(
                            id="slider-mundo", min=MIN_YEAR, max=MAX_YEAR, step=1,
                            value=MAX_YEAR, marks=_slider_marks,
                            tooltip={"placement": "bottom"},
                        ),
                    ], md=8, className="mt-3 mb-1"),
                ]),
                dbc.Row([
                    dbc.Col(dcc.Graph(id="fig-top-dests"), md=5),
                    dbc.Col([
                        dcc.Graph(id="fig-us-share", figure=fig_us_share()),
                        dcc.Graph(id="fig-non-us", figure=fig_non_us_growth()),
                    ], md=7),
                ], className="g-3"),
            ]),
            dcc.Tab(label="Migrantes en México", style=TAB_STYLE, selected_style=TAB_SEL, children=[
                dbc.Row([
                    dbc.Col([
                        html.Label("Año:", style={"color": "#94A3B8", "fontSize": "0.85rem"}),
                        dcc.Slider(
                            id="slider-mex", min=MIN_YEAR, max=MAX_YEAR, step=1,
                            value=MAX_YEAR, marks=_slider_marks,
                            tooltip={"placement": "bottom"},
                        ),
                    ], md=8, className="mt-3 mb-1"),
                ]),
                dbc.Row([
                    dbc.Col(dcc.Graph(id="fig-top-sources"), md=5),
                    dbc.Col(dcc.Graph(id="fig-surge", figure=fig_surge_sources()), md=7),
                ], className="g-3"),
            ]),
        ]),
    ],
)


# ── Callbacks ─────────────────────────────────────────────────────────────────
@app.callback(Output("fig-top-dests", "figure"), Input("slider-mundo", "value"))
def update_top_dests(year: int) -> go.Figure:
    return fig_top_destinations(year)


@app.callback(Output("fig-top-sources", "figure"), Input("slider-mex", "value"))
def update_top_sources(year: int) -> go.Figure:
    return fig_top_sources(year)


if __name__ == "__main__":
    app.run(debug=True, port=8060)
