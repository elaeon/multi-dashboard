import json
from pathlib import Path

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import polars as pl
from dash import Dash, html, dcc, Input, Output
import dash_bootstrap_components as dbc

# ── Constants ────────────────────────────────────────────────────────────────

YEARS = list(range(2017, 2025))
BASE_DIR = Path("data/inegi/nacimientos_descesos")
VALID_STATES = set(range(1, 33))

STATES = {
    1: "Aguascalientes", 2: "Baja California", 3: "Baja California Sur",
    4: "Campeche", 5: "Coahuila", 6: "Colima", 7: "Chiapas", 8: "Chihuahua",
    9: "CDMX", 10: "Durango", 11: "Guanajuato", 12: "Guerrero", 13: "Hidalgo",
    14: "Jalisco", 15: "México", 16: "Michoacán", 17: "Morelos", 18: "Nayarit",
    19: "Nuevo León", 20: "Oaxaca", 21: "Puebla", 22: "Querétaro",
    23: "Quintana Roo", 24: "San Luis Potosí", 25: "Sinaloa", 26: "Sonora",
    27: "Tabasco", 28: "Tamaulipas", 29: "Tlaxcala", 30: "Veracruz",
    31: "Yucatán", 32: "Zacatecas",
}
INEGI_TO_ISO = {
    1: "MX-AGU", 2: "MX-BCN", 3: "MX-BCS", 4: "MX-CAM", 5: "MX-COA",
    6: "MX-COL", 7: "MX-CHP", 8: "MX-CHH", 9: "MX-CMX", 10: "MX-DUR",
    11: "MX-GUA", 12: "MX-GRO", 13: "MX-HID", 14: "MX-JAL", 15: "MX-MEX",
    16: "MX-MIC", 17: "MX-MOR", 18: "MX-NAY", 19: "MX-NLE", 20: "MX-OAX",
    21: "MX-PUE", 22: "MX-QUE", 23: "MX-ROO", 24: "MX-SLP", 25: "MX-SIN",
    26: "MX-SON", 27: "MX-TAB", 28: "MX-TAM", 29: "MX-TLA", 30: "MX-VER",
    31: "MX-YUC", 32: "MX-ZAC",
}
_S = {str(k): v for k, v in STATES.items()}
_I = {str(k): v for k, v in INEGI_TO_ISO.items()}

CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font_color="#CBD5E1",
)
_GRID = dict(xaxis=dict(gridcolor="#334155"), yaxis=dict(gridcolor="#334155"))
CARD_STYLE = {
    "background": "#1E293B", "border": "1px solid #334155",
    "borderRadius": "8px", "padding": "16px", "textAlign": "center",
}
TAB_STYLE = {"backgroundColor": "#0F172A", "color": "#94A3B8", "borderTop": "none"}
TAB_SEL = {
    "backgroundColor": "#1E293B", "color": "#F8FAFC",
    "borderTop": "2px solid #2E86AB", "fontWeight": "600",
}

# ── Data loading ──────────────────────────────────────────────────────────────

nat_inc = pl.concat([pl.read_csv(BASE_DIR / f"{y}.csv") for y in YEARS])
od_all = pl.concat([pl.read_csv(BASE_DIR / f"od_{y}.csv") for y in YEARS])

state_ni = (
    nat_inc
    .filter(pl.col("ent_resid").is_in(VALID_STATES))
    .group_by(["ent_resid", "anio"])
    .agg(pl.col("total_nac").sum(), pl.col("total_des").sum())
    .with_columns(
        (pl.col("total_nac") - pl.col("total_des")).alias("crecimiento_natural"),
        pl.col("ent_resid").cast(pl.String).replace(_I).alias("iso_id"),
        pl.col("ent_resid").cast(pl.String).replace(_S).alias("estado"),
    )
)

od_cross = (
    od_all
    .filter(
        pl.col("ent_resid").is_in(VALID_STATES),
        pl.col("ent_ocurr").is_in(VALID_STATES),
        pl.col("ent_resid") != pl.col("ent_ocurr"),
    )
    .group_by(["ent_resid", "ent_ocurr", "anio"])
    .agg(
        pl.col("total_nac").fill_null(0).sum().alias("nac"),
        pl.col("total_des").fill_null(0).sum().alias("des"),
    )
    .with_columns(
        (pl.col("nac") + pl.col("des")).alias("total"),
        pl.col("ent_resid").cast(pl.String).replace(_S).alias("estado_resid"),
        pl.col("ent_ocurr").cast(pl.String).replace(_S).alias("estado_ocurr"),
    )
)

national = (
    nat_inc
    .filter(pl.col("ent_resid").is_in(VALID_STATES))
    .group_by("anio")
    .agg(pl.col("total_nac").sum(), pl.col("total_des").sum())
    .with_columns((pl.col("total_nac") - pl.col("total_des")).alias("ni"))
    .sort("anio")
)


def _classify_tloc(col: str) -> pl.Expr:
    return (
        pl.when(pl.col(col) <= 3).then(pl.lit("Rural"))
        .when(pl.col(col) <= 6).then(pl.lit("Semi-urbano"))
        .otherwise(pl.lit("Urbano"))
        .alias(col)
    )


tloc_flows = (
    od_all
    .filter(
        pl.col("ent_resid").is_in(VALID_STATES),
        pl.col("ent_ocurr").is_in(VALID_STATES),
        pl.col("tloc_resid").is_between(1, 17),
        pl.col("tloc_ocurr").is_between(1, 17),
    )
    .with_columns(_classify_tloc("tloc_resid"), _classify_tloc("tloc_ocurr"))
    .group_by(["tloc_resid", "tloc_ocurr", "anio"])
    .agg(
        pl.col("total_nac").fill_null(0).sum().alias("nac"),
        pl.col("total_des").fill_null(0).sum().alias("des"),
    )
)

# KPI precomputation
kpi_by_year: dict = {}
for _y in YEARS:
    _sni = state_ni.filter(pl.col("anio") == _y)
    _total_nac = int(_sni["total_nac"].sum())
    _total_des = int(_sni["total_des"].sum())
    _od_y = od_all.filter(
        pl.col("anio") == _y,
        pl.col("ent_resid").is_in(VALID_STATES),
        pl.col("ent_ocurr").is_in(VALID_STATES),
    )
    _tot_ev = int(_od_y["total_nac"].fill_null(0).sum()) + int(_od_y["total_des"].fill_null(0).sum())
    _cross = _od_y.filter(pl.col("ent_resid") != pl.col("ent_ocurr"))
    _cross_ev = int(_cross["total_nac"].fill_null(0).sum()) + int(_cross["total_des"].fill_null(0).sum())
    kpi_by_year[_y] = {
        "nac": _total_nac,
        "des": _total_des,
        "ni": _total_nac - _total_des,
        "pct_cross": _cross_ev / _tot_ev * 100 if _tot_ev else 0.0,
    }

with open("data/mexico_states.geojson") as _f:
    _geojson = json.load(_f)

# ── Figure factories ──────────────────────────────────────────────────────────

def fig_choropleth(year: int) -> go.Figure:
    d = state_ni.filter(pl.col("anio") == year)
    fig = px.choropleth_map(
        d,
        geojson=_geojson,
        locations="iso_id",
        featureidkey="id",
        color="crecimiento_natural",
        hover_name="estado",
        hover_data={"total_nac": True, "total_des": True,
                    "crecimiento_natural": True, "iso_id": False},
        color_continuous_scale="RdYlGn",
        center={"lat": 24, "lon": -102},
        zoom=4,
        labels={
            "crecimiento_natural": "Crec. natural",
            "total_nac": "Nacimientos",
            "total_des": "Defunciones",
        },
    )
    fig.update_layout(
        height=520,
        map_style="carto-darkmatter",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#CBD5E1",
        margin=dict(t=10, b=0, l=0, r=0),
        coloraxis_colorbar=dict(
            title="Crec.<br>natural",
            tickfont=dict(color="#CBD5E1"),
            title_font=dict(color="#CBD5E1"),
        ),
    )
    return fig


def fig_national_trend() -> go.Figure:
    yrs = national["anio"].to_list()
    fig = go.Figure()
    for col, name, color in [
        ("total_nac", "Nacimientos", "#3BB273"),
        ("total_des", "Defunciones", "#E84855"),
        ("ni", "Crec. natural", "#F4A261"),
    ]:
        fig.add_trace(go.Scatter(
            x=yrs, y=national[col].to_list(), name=name,
            mode="lines+markers",
            line=dict(color=color, width=2, dash="dot" if col == "ni" else "solid"),
            marker=dict(size=6),
            hovertemplate=f"<b>%{{x}}</b><br>{name}: %{{y:,}}<extra></extra>",
        ))
    fig.add_vrect(
        x0=2019.5, x1=2021.5, fillcolor="rgba(244,162,97,0.1)", line_width=0,
        annotation_text="COVID-19", annotation_font_color="#94A3B8",
        annotation_position="top left",
    )
    fig.update_layout(
        title=dict(
            text="<b>COVID-19 redujo el crecimiento natural a un tercio en 2020</b>"
                 "<br><sup style='color:#94A3B8'>Totales nacionales, 2017–2024</sup>",
        ),
        height=380, legend=dict(orientation="h", y=-0.2, x=0),
        margin=dict(t=65, b=80, l=40, r=10),
        **CHART_LAYOUT, **_GRID,
    )
    return fig


def fig_sankey(year: int) -> go.Figure:
    d = od_cross.filter(pl.col("anio") == year).sort("total", descending=True).head(40)
    if len(d) == 0:
        return go.Figure()
    nodes = sorted(set(d["estado_resid"].to_list() + d["estado_ocurr"].to_list()))
    idx = {s: i for i, s in enumerate(nodes)}
    fig = go.Figure(go.Sankey(
        node=dict(
            pad=10, thickness=14, label=nodes, color="#2E86AB",
            hovertemplate="<b>%{label}</b><extra></extra>",
        ),
        link=dict(
            source=[idx[s] for s in d["estado_resid"].to_list()],
            target=[idx[s] for s in d["estado_ocurr"].to_list()],
            value=d["total"].to_list(),
            color="rgba(46,134,171,0.22)",
            customdata=list(zip(d["nac"].to_list(), d["des"].to_list())),
            hovertemplate=(
                "<b>%{source.label} → %{target.label}</b><br>"
                "Nacimientos: %{customdata[0]:,}<br>"
                "Defunciones: %{customdata[1]:,}<br>"
                "Total: %{value:,}<extra></extra>"
            ),
        ),
    ))
    fig.update_layout(
        title=dict(
            text="<b>El par México↔CDMX concentra el 44% de los eventos vitales fuera del estado</b>"
                 "<br><sup style='color:#94A3B8'>Top 40 flujos residencia → lugar de ocurrencia</sup>",
        ),
        height=580, paper_bgcolor="rgba(0,0,0,0)", font_color="#CBD5E1",
        margin=dict(t=65, b=10, l=10, r=10),
    )
    return fig


def fig_top_flows(year: int) -> go.Figure:
    d = od_cross.filter(pl.col("anio") == year).sort("total", descending=True).head(15)
    labels = [
        f"{r} → {o}"
        for r, o in zip(d["estado_resid"].to_list()[::-1], d["estado_ocurr"].to_list()[::-1])
    ]
    fig = go.Figure()
    for col, name, color in [("nac", "Nacimientos", "#3BB273"), ("des", "Defunciones", "#E84855")]:
        fig.add_trace(go.Bar(
            x=d[col].to_list()[::-1], y=labels, orientation="h",
            name=name, marker_color=color,
            hovertemplate=f"<b>%{{y}}</b><br>{name}: %{{x:,}}<extra></extra>",
        ))
    fig.update_layout(
        barmode="stack",
        title=dict(text="<b>Top 15 pares por volumen</b>"),
        height=max(320, 15 * 30 + 100),
        xaxis=dict(gridcolor="#334155", title="Eventos vitales"),
        yaxis=dict(gridcolor="rgba(0,0,0,0)"),
        legend=dict(orientation="h", y=-0.15, x=0),
        margin=dict(t=50, b=70, l=10, r=10),
        **CHART_LAYOUT,
    )
    return fig


def fig_tloc(year: int) -> go.Figure:
    d = tloc_flows.filter(pl.col("anio") == year)
    order = ["Rural", "Semi-urbano", "Urbano"]
    colors = {"Rural": "#E84855", "Semi-urbano": "#F4A261", "Urbano": "#3BB273"}

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Nacimientos", "Defunciones"],
        shared_yaxes=True,
    )
    legend_added: set = set()
    for col_idx, event_col in enumerate(["nac", "des"], start=1):
        for tloc_r in order:
            sub = d.filter(pl.col("tloc_resid") == tloc_r)
            total = int(sub[event_col].sum())
            if total == 0:
                continue
            for tloc_o in order:
                row = sub.filter(pl.col("tloc_ocurr") == tloc_o)
                val = int(row[event_col].sum()) if len(row) else 0
                pct = val / total * 100 if total else 0.0
                name = f"Ocurre en: {tloc_o}"
                show = name not in legend_added
                if show:
                    legend_added.add(name)
                fig.add_trace(go.Bar(
                    x=[pct], y=[tloc_r], orientation="h",
                    name=name, legendgroup=name, showlegend=show,
                    marker_color=colors[tloc_o],
                    text=f"{pct:.0f}%" if pct > 7 else "",
                    textposition="inside", insidetextanchor="middle",
                    customdata=[[val]],
                    hovertemplate=(
                        f"Viven en <b>{tloc_r}</b> → ocurre en <b>{tloc_o}</b><br>"
                        "%{x:.1f}%  (%{customdata[0][0]:,} eventos)<extra></extra>"
                    ),
                ), row=1, col=col_idx)

    fig.update_layout(
        barmode="stack",
        title=dict(
            text="<b>Residentes rurales viajan a zonas urbanas para dar a luz, pero mueren en casa</b>"
                 "<br><sup style='color:#94A3B8'>Distribución de localidad de ocurrencia según tipo de residencia</sup>",
        ),
        height=340, paper_bgcolor="rgba(0,0,0,0)", font_color="#CBD5E1",
        legend=dict(orientation="h", y=-0.28, x=0),
        margin=dict(t=70, b=95, l=10, r=10),
    )
    fig.update_xaxes(range=[0, 100], showticklabels=False)
    fig.update_yaxes(gridcolor="rgba(0,0,0,0)")
    for ann in fig.layout.annotations:
        ann.font.color = "#CBD5E1"
    return fig


# ── App ───────────────────────────────────────────────────────────────────────

def _fmt(n: int) -> str:
    return f"{n:,}"


def _delta(curr: float, prev: float | None, harm_when_up: bool = False) -> tuple[str, str]:
    if prev is None or prev == 0:
        return "—", "#94A3B8"
    pct = (curr - prev) / abs(prev) * 100
    worsened = (pct > 0) == harm_when_up
    color = "#E84855" if worsened else "#3BB273"
    arrow = "▲" if pct > 0 else "▼"
    return f"{arrow} {abs(pct):.1f}% vs año anterior", color


def _kpi_card(label: str, value: str, delta_text: str, delta_color: str) -> dbc.Col:
    return dbc.Col(
        html.Div([
            html.P(label, style={"color": "#94A3B8", "fontSize": "0.8rem", "marginBottom": "4px"}),
            html.H4(value, style={"color": "#F8FAFC", "marginBottom": "2px"}),
            html.Span(delta_text, style={"color": delta_color, "fontSize": "0.8rem"}),
        ], style=CARD_STYLE),
        width=3,
    )


_trend_fig = fig_national_trend()

app = Dash(__name__, external_stylesheets=[dbc.themes.SLATE])
app.layout = dbc.Container(
    [
        html.Div(style={"height": "16px"}),
        html.H2("Nacimientos y Defunciones en México",
                style={"color": "#F8FAFC", "marginBottom": "4px"}),
        html.P(
            "Crecimiento natural y movilidad por lugar de ocurrencia vs residencia · INEGI 2017–2024",
            style={"color": "#94A3B8", "fontSize": "0.9rem"},
        ),
        html.Hr(style={"borderColor": "#334155"}),

        dbc.Row(
            dbc.Col([
                html.Label("Año analizado", style={"color": "#94A3B8", "fontSize": "0.8rem"}),
                dcc.Slider(
                    min=2017, max=2024, step=1, value=2024,
                    marks={y: {"label": str(y), "style": {"color": "#94A3B8"}} for y in YEARS},
                    id="year-slider",
                    tooltip={"placement": "bottom", "always_visible": False},
                ),
            ], width=10),
            className="mb-4",
        ),

        dbc.Row(id="kpi-row", className="mb-4 g-2"),

        dbc.Tabs([
            dbc.Tab(
                dbc.Row([
                    dbc.Col(dcc.Graph(id="choropleth", config={"displayModeBar": False}), width=7),
                    dbc.Col(dcc.Graph(figure=_trend_fig, config={"displayModeBar": False}), width=5),
                ], className="mt-3"),
                label="Crecimiento Natural",
                tab_style=TAB_STYLE, active_tab_style=TAB_SEL,
            ),
            dbc.Tab(
                dbc.Row([
                    dbc.Col(dcc.Graph(id="sankey", config={"displayModeBar": False}), width=7),
                    dbc.Col(dcc.Graph(id="top-flows", config={"displayModeBar": False}), width=5),
                ], className="mt-3"),
                label="Flujos entre Estados",
                tab_style=TAB_STYLE, active_tab_style=TAB_SEL,
            ),
            dbc.Tab(
                dbc.Row(
                    dbc.Col(dcc.Graph(id="tloc-chart", config={"displayModeBar": False})),
                    className="mt-3",
                ),
                label="Rural vs Urbano",
                tab_style=TAB_STYLE, active_tab_style=TAB_SEL,
            ),
        ]),
    ],
    fluid=True,
    style={"backgroundColor": "#0F172A", "minHeight": "100vh"},
)


@app.callback(
    Output("kpi-row", "children"),
    Output("choropleth", "figure"),
    Output("sankey", "figure"),
    Output("top-flows", "figure"),
    Output("tloc-chart", "figure"),
    Input("year-slider", "value"),
)
def update(year: int):
    kpi = kpi_by_year[year]
    prev = kpi_by_year.get(year - 1)

    d_nac, c_nac = _delta(kpi["nac"], prev["nac"] if prev else None)
    d_des, c_des = _delta(kpi["des"], prev["des"] if prev else None, harm_when_up=True)
    d_ni, c_ni = _delta(kpi["ni"], prev["ni"] if prev else None)
    d_cross, c_cross = _delta(kpi["pct_cross"], prev["pct_cross"] if prev else None)

    return (
        [
            _kpi_card("Nacimientos", _fmt(kpi["nac"]), d_nac, c_nac),
            _kpi_card("Defunciones", _fmt(kpi["des"]), d_des, c_des),
            _kpi_card("Crecimiento natural", _fmt(kpi["ni"]), d_ni, c_ni),
            _kpi_card("Eventos fuera del estado", f"{kpi['pct_cross']:.1f}%", d_cross, c_cross),
        ],
        fig_choropleth(year),
        fig_sankey(year),
        fig_top_flows(year),
        fig_tloc(year),
    )


if __name__ == "__main__":
    app.run(debug=True, port=8065)
