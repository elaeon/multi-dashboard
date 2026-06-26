import json
import re
import polars as pl
import plotly.graph_objects as go
import plotly.express as px
from dash import Dash, dcc, html
import dash_bootstrap_components as dbc

# ── Constants ─────────────────────────────────────────────────────────────────

ESTADO_MAP = {
    "Estado De Mexico": "Mexico",
    "Veracruz De Ignacio De La Llave": "Veracruz",
    "Michoacan De Ocampo": "Michoacan",
    "Coiudad De Mexico": "Ciudad De Mexico",
    "Merida Yucatan": "Yucatan",
}
VALID_ESTADOS = {
    "Aguascalientes", "Baja California", "Baja California Sur", "Campeche",
    "Chiapas", "Chihuahua", "Ciudad De Mexico", "Coahuila", "Colima",
    "Durango", "Guanajuato", "Guerrero", "Hidalgo", "Jalisco", "Mexico",
    "Michoacan", "Morelos", "Nayarit", "Nuevo Leon", "Oaxaca", "Puebla",
    "Queretaro", "Quintana Roo", "San Luis Potosi", "Sinaloa", "Sonora",
    "Tabasco", "Tamaulipas", "Tlaxcala", "Veracruz", "Yucatan", "Zacatecas",
}
STATE_ISO = {
    "Aguascalientes": "MX-AGU", "Baja California": "MX-BCN",
    "Baja California Sur": "MX-BCS", "Campeche": "MX-CAM",
    "Chiapas": "MX-CHP", "Chihuahua": "MX-CHH",
    "Ciudad De Mexico": "MX-CMX", "Coahuila": "MX-COA",
    "Colima": "MX-COL", "Durango": "MX-DUR",
    "Guanajuato": "MX-GUA", "Guerrero": "MX-GRO",
    "Hidalgo": "MX-HID", "Jalisco": "MX-JAL",
    "Mexico": "MX-MEX", "Michoacan": "MX-MIC",
    "Morelos": "MX-MOR", "Nayarit": "MX-NAY",
    "Nuevo Leon": "MX-NLE", "Oaxaca": "MX-OAX",
    "Puebla": "MX-PUE", "Queretaro": "MX-QUE",
    "Quintana Roo": "MX-ROO", "San Luis Potosi": "MX-SLP",
    "Sinaloa": "MX-SIN", "Sonora": "MX-SON",
    "Tabasco": "MX-TAB", "Tamaulipas": "MX-TAM",
    "Tlaxcala": "MX-TLA", "Veracruz": "MX-VER",
    "Yucatan": "MX-YUC", "Zacatecas": "MX-ZAC",
}
ESTADO_LABEL = {"Mexico": "Estado de México", "Ciudad De Mexico": "Ciudad de México"}

_LEGAL_SUFFIX = re.compile(
    r",?\s*(S\.?A\.?\s*De\s*C\.?V\.?|I\.A\.P\.|S\.\s*De\s*R\.L\.?\s*De\s*C\.?V\.?|"
    r"Sa\s*De\s*Cv|A\.C\.|S\.C\.)\s*$",
    re.IGNORECASE,
)

FOCUS, CONTEXT = "#2E86AB", "#475569"

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
TAB_STYLE = {"backgroundColor": "#0F172A", "color": "#94A3B8", "borderTop": "none",
             "borderBottom": "1px solid #334155", "padding": "10px 18px"}
TAB_SEL = {"backgroundColor": "#1E293B", "color": "#F8FAFC",
           "borderTop": "2px solid #2E86AB", "borderBottom": "none",
           "fontWeight": "600", "padding": "10px 18px"}

# ── Data loading & cleaning ───────────────────────────────────────────────────

df = (
    pl.read_csv("data/profeco/casas_empeño/registro_casas_empeno_2014_2025.csv")
    .unique(subset=["nombre_comercial", "razon_social", "domicilio", "fecha_registro"])
    .with_columns(
        pl.col("estado").replace(ESTADO_MAP).alias("estado"),
        pl.when(pl.col("razon_social").str.to_lowercase().str.contains("first cash"))
          .then(pl.lit("First Cash"))
          .otherwise(pl.col("razon_social"))
          .alias("cadena"),
        pl.col("fecha_registro").str.to_date("%Y-%m-%d").alias("fecha_registro"),
    )
    .with_columns(pl.col("fecha_registro").dt.year().cast(pl.Int32).alias("año"))
    .filter(pl.col("estado").is_in(VALID_ESTADOS))
)

TOTAL = len(df)

# ── Pre-aggregations ──────────────────────────────────────────────────────────

agg_por_año = df.group_by("año").agg(pl.len().alias("n")).sort("año")

agg_por_estado = (
    df.group_by("estado").agg(pl.len().alias("n"))
    .with_columns(
        pl.col("estado").replace(STATE_ISO).alias("iso"),
        pl.col("estado").replace(ESTADO_LABEL).alias("label"),
    )
    .sort("n", descending=True)
)

_cadena_counts = df.group_by("cadena").agg(pl.len().alias("n")).sort("n", descending=True)
agg_top_cadenas = _cadena_counts.head(15)

N_FIRST_CASH = int(_cadena_counts.filter(pl.col("cadena") == "First Cash")["n"][0])
TOP5_N = int(_cadena_counts.head(5)["n"].sum())
N_SOLES = int(_cadena_counts.filter(pl.col("n") == 1).height)

# First Cash share per year (organic window: 2016–2024)
_fc_año = (
    df.filter(pl.col("año").is_between(2016, 2024) & (pl.col("cadena") == "First Cash"))
    .group_by("año").agg(pl.len().alias("n_fc")).sort("año")
)
agg_fc_share = (
    agg_por_año.filter(pl.col("año").is_between(2016, 2024))
    .join(_fc_año, on="año", how="left")
    .with_columns(
        pl.col("n_fc").fill_null(0),
        (pl.col("n_fc").fill_null(0) / pl.col("n") * 100).alias("share_pct"),
    )
)

# Slope data: state % of new registrations — 2016–2019 vs 2020–2024
_period_df = (
    df.filter(pl.col("año").is_between(2016, 2024))
    .with_columns(
        pl.when(pl.col("año") <= 2019)
          .then(pl.lit("2016–2019"))
          .otherwise(pl.lit("2020–2024"))
          .alias("periodo")
    )
)
_period_totals = _period_df.group_by("periodo").agg(pl.len().alias("total"))
_period_estado = _period_df.group_by(["periodo", "estado"]).agg(pl.len().alias("n"))
agg_slope_data = (
    _period_estado.join(_period_totals, on="periodo")
    .with_columns((pl.col("n") / pl.col("total") * 100).alias("pct"))
)

with open("data/mexico_states.geojson") as _f:
    GEO = json.load(_f)

# ── Helpers ───────────────────────────────────────────────────────────────────

def kpi_card(title: str, value: str, color: str = "#CBD5E1",
             delta: str = None, delta_color: str = None) -> dbc.Col:
    children = [
        html.P(title, style={"color": "#94A3B8", "fontSize": "12px", "margin": 0}),
        html.H3(value, style={"color": color, "margin": "4px 0 0", "fontSize": "22px"}),
    ]
    if delta:
        children.append(html.Span(delta, style={"color": delta_color or "#94A3B8", "fontSize": "11px"}))
    return dbc.Col(html.Div(children, style=CARD_STYLE), xs=12, sm=6, md=3)


def _shorten(s: str) -> str:
    s = _LEGAL_SUFFIX.sub("", s).strip(" ,")
    return s if len(s) <= 38 else s[:36] + "…"


# ── Figure factories ──────────────────────────────────────────────────────────

def fig_ranking_cadenas() -> go.Figure:
    d = agg_top_cadenas.sort("n")
    labels = [_shorten(c) if c != "First Cash" else "First Cash" for c in d["cadena"].to_list()]
    colors = [FOCUS if c == "First Cash" else CONTEXT for c in d["cadena"].to_list()]
    pcts = [n_ / TOTAL * 100 for n_ in d["n"].to_list()]
    n = len(d)

    fig = go.Figure(go.Bar(
        x=d["n"].to_list(), y=labels,
        orientation="h", marker_color=colors,
        customdata=list(zip(d["cadena"].to_list(), d["n"].to_list(), pcts)),
        hovertemplate="<b>%{customdata[0]}</b><br>Sucursales: %{customdata[1]:,}<br>%{customdata[2]:.1f}% del total<extra></extra>",
    ))
    fig.update_layout(**CHART_LAYOUT)
    fig.update_layout(
        title=dict(
            text=f"<b>First Cash controla 1 de cada 5 sucursales de empeño en México ({N_FIRST_CASH/TOTAL*100:.0f}%)</b>"
                 f"<br><sup style='color:#94A3B8'>Top 15 cadenas por sucursales registradas · {TOTAL:,} total (deduplicado)</sup>",
            font=dict(size=13),
        ),
        height=max(300, n * 32 + 100),
        yaxis=dict(autorange="reversed", gridcolor="#334155"),
        xaxis=dict(gridcolor="#334155", title="Sucursales"),
        margin=dict(t=65, b=40, l=10, r=10),
    )
    return fig


def fig_por_año() -> go.Figure:
    d = agg_por_año
    años = d["año"].to_list()
    counts = d["n"].to_list()
    colors = ["#F4A261" if y == 2015 else ("#64748B" if y in (2014, 2025) else FOCUS) for y in años]

    fig = go.Figure(go.Bar(
        x=años, y=counts, marker_color=colors,
        hovertemplate="<b>%{x}</b><br>Registros: %{y:,}<extra></extra>",
    ))
    val_2015 = int(d.filter(pl.col("año") == 2015)["n"][0])
    val_2025 = int(d.filter(pl.col("año") == 2025)["n"][0])
    fig.add_annotation(
        x=2015, y=val_2015,
        text=f"<b>Ola de cumplimiento regulatorio</b><br>{val_2015:,} registros · 40.6% del total",
        font=dict(color="#F4A261", size=10),
        arrowcolor="#F4A261", ax=0, ay=-60,
        bgcolor="#0F172A", borderpad=4, bordercolor="#F4A261", borderwidth=1,
    )
    fig.add_annotation(
        x=2025, y=val_2025,
        text="Año parcial<br>(hasta sep-2025)",
        font=dict(color="#64748B", size=9),
        arrowcolor="#64748B", ax=0, ay=-45,
        bgcolor="#0F172A", borderpad=3,
    )
    fig.update_layout(**CHART_LAYOUT)
    fig.update_layout(
        title=dict(
            text="<b>El 40.6% del padrón se registró en 2015 — una ola de cumplimiento, no crecimiento orgánico</b>"
                 "<br><sup style='color:#94A3B8'>Nuevas casas de empeño registradas por año · PROFECO</sup>",
            font=dict(size=13),
        ),
        height=400,
        xaxis=dict(gridcolor="#334155", title="Año", dtick=1, tickangle=-45),
        yaxis=dict(gridcolor="#334155", title="Nuevos registros"),
        margin=dict(t=65, b=65, l=10, r=10),
    )
    return fig


def fig_share_first_cash() -> go.Figure:
    d = agg_fc_share.sort("año")

    fig = go.Figure(go.Scatter(
        x=d["año"].to_list(), y=d["share_pct"].to_list(),
        mode="lines+markers",
        line=dict(color=FOCUS, width=2.5),
        marker=dict(color=FOCUS, size=9),
        fill="tozeroy", fillcolor="rgba(46,134,171,0.12)",
        hovertemplate="<b>%{x}</b><br>First Cash: %{y:.1f}% de nuevos registros<extra></extra>",
    ))
    peak = d.sort("share_pct", descending=True).row(0, named=True)
    fig.add_annotation(
        x=peak["año"], y=peak["share_pct"],
        text=f"<b>{peak['share_pct']:.1f}%</b>",
        font=dict(color=FOCUS, size=11),
        arrowcolor=FOCUS, ax=30, ay=-30,
        bgcolor="#0F172A", borderpad=3,
    )
    fig.update_layout(**CHART_LAYOUT)
    fig.update_layout(
        title=dict(
            text="<b>First Cash pasó del 4% al 25% de los registros nuevos en 8 años</b>"
                 "<br><sup style='color:#94A3B8'>Participación de First Cash en registros orgánicos · 2016–2024 (excluye ola 2015)</sup>",
            font=dict(size=13),
        ),
        height=340,
        xaxis=dict(gridcolor="#334155", title="Año", dtick=1),
        yaxis=dict(gridcolor="#334155", title="% de nuevos registros", ticksuffix="%"),
        margin=dict(t=65, b=40, l=10, r=10),
    )
    return fig


def fig_mapa() -> go.Figure:
    d = agg_por_estado
    fig = px.choropleth_map(
        d, geojson=GEO, locations="iso", featureidkey="properties.id",
        color="n", color_continuous_scale="Blues",
        hover_name="label", map_style="carto-darkmatter",
        center={"lat": 23.5, "lon": -102.5}, zoom=3.8,
    )
    fig.update_traces(hovertemplate="<b>%{hovertext}</b><br>Sucursales: %{z:,}<extra></extra>")
    fig.update_layout(
        height=520,
        paper_bgcolor="rgba(0,0,0,0)", font_color="#CBD5E1",
        margin=dict(t=0, b=0, l=0, r=0),
        coloraxis_colorbar=dict(
            title=dict(text="Sucursales", font=dict(color="#CBD5E1")),
            tickfont=dict(color="#CBD5E1"),
        ),
    )
    return fig


def fig_ranking_estados() -> go.Figure:
    d = agg_por_estado.sort("n", descending=False)
    n = len(d)
    labels = [ESTADO_LABEL.get(e, e) for e in d["estado"].to_list()]
    colors = [FOCUS if i >= n - 5 else CONTEXT for i in range(n)]

    fig = go.Figure(go.Bar(
        x=d["n"].to_list(), y=labels,
        orientation="h", marker_color=colors,
        hovertemplate="<b>%{y}</b><br>Sucursales: %{x:,}<extra></extra>",
    ))
    top_label = ESTADO_LABEL.get(d["estado"][-1], d["estado"][-1])
    fig.update_layout(**CHART_LAYOUT)
    fig.update_layout(
        title=dict(
            text=f"<b>{top_label} concentra más sucursales, pero se necesitan 18 estados para llegar al 80%</b>"
                 f"<br><sup style='color:#94A3B8'>Sucursales por estado (total acumulado 2014–2025)</sup>",
            font=dict(size=13),
        ),
        height=max(300, n * 22 + 80),
        yaxis=dict(gridcolor="#334155"),
        xaxis=dict(gridcolor="#334155", title="Sucursales"),
        margin=dict(t=65, b=40, l=10, r=10),
    )
    return fig


def fig_slope_geografico() -> go.Figure:
    P_A, P_B = "2016–2019", "2020–2024"
    a = agg_slope_data.filter(pl.col("periodo") == P_A)[["estado", "pct"]].rename({"pct": "pct_a"})
    b = agg_slope_data.filter(pl.col("periodo") == P_B)[["estado", "pct"]].rename({"pct": "pct_b"})
    slope_d = (
        a.join(b, on="estado", how="inner")
        .with_columns((pl.col("pct_b") - pl.col("pct_a")).alias("delta"))
        .sort("pct_b", descending=True)
    )

    GAINED = {"Jalisco", "Oaxaca", "San Luis Potosi", "Quintana Roo"}
    LOST = {"Nuevo Leon", "Veracruz"}
    n_total = len(slope_d)
    n_up = slope_d.filter(pl.col("delta") > 0).height
    n_dn = n_total - n_up

    fig = go.Figure()
    for i, row in enumerate(slope_d.iter_rows(named=True)):
        if row["estado"] in GAINED:
            color, width = "#3BB273", 2
        elif row["estado"] in LOST:
            color, width = "#E84855", 2
        else:
            color, width = "#475569", 1

        show_label = (i < 2) or (i >= n_total - 2) or (row["estado"] in GAINED | LOST)
        label = ESTADO_LABEL.get(row["estado"], row["estado"])

        fig.add_trace(go.Scatter(
            x=[P_A, P_B],
            y=[row["pct_a"], row["pct_b"]],
            mode="lines+markers" + ("+text" if show_label else ""),
            line=dict(color=color, width=width),
            marker=dict(color=color, size=6 if width == 2 else 4),
            text=[None, label] if show_label else None,
            textposition="middle right",
            textfont=dict(size=8, color=color),
            showlegend=False,
            hovertemplate=f"<b>{label}</b><br>%{{x}}: %{{y:.2f}}%  Δ {row['delta']:+.2f} pp<extra></extra>",
        ))

    fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines+markers",
                              line=dict(color="#3BB273"), marker=dict(color="#3BB273", size=7),
                              name=f"▲ Ganó participación ({n_up} estados)"))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines+markers",
                              line=dict(color="#E84855"), marker=dict(color="#E84855", size=7),
                              name=f"▼ Perdió participación ({n_dn} estados)"))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines",
                              line=dict(color="#475569"), name="Cambio menor"))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#CBD5E1",
        title=dict(
            text="<b>Los nuevos registros se desplazan del norte hacia occidente y sur</b>"
                 "<br><sup style='color:#94A3B8'>Participación estatal en registros orgánicos · 2016–2019 vs 2020–2024</sup>",
            font=dict(size=13),
        ),
        height=max(400, n_total * 20 + 100),
        xaxis=dict(type="category", gridcolor="rgba(0,0,0,0)"),
        yaxis=dict(gridcolor="#334155", title="Participación (%)"),
        legend=dict(orientation="h", y=-0.06, x=0, font=dict(size=10)),
        margin=dict(t=65, b=80, l=10, r=200),
    )
    return fig


# ── Pre-compute all figures ───────────────────────────────────────────────────

_fig_cadenas = fig_ranking_cadenas()
_fig_año = fig_por_año()
_fig_fc_share = fig_share_first_cash()
_fig_mapa = fig_mapa()
_fig_estados = fig_ranking_estados()
_fig_slope = fig_slope_geografico()

# ── Layout ─────────────────────────────────────────────────────────────────────

app = Dash(__name__, external_stylesheets=[dbc.themes.SLATE])
app.title = "Casas de Empeño · PROFECO"

_kpi_row = dbc.Row([
    kpi_card("Total sucursales registradas", f"{TOTAL:,}", "#CBD5E1",
             "sin duplicados · 5 domicilios inválidos excluidos"),
    kpi_card("First Cash", f"{N_FIRST_CASH:,} sucursales", FOCUS,
             f"{N_FIRST_CASH/TOTAL*100:.1f}% del mercado"),
    kpi_card("Top 5 cadenas", f"{TOP5_N/TOTAL*100:.1f}%", "#F4A261",
             f"{TOP5_N:,} de {TOTAL:,} sucursales"),
    kpi_card("Operadores con 1 sucursal", f"{N_SOLES:,}", "#94A3B8",
             f"{N_SOLES/TOTAL*100:.1f}% de entidades · 14.1% de sucursales"),
], className="mb-3 g-2")

app.layout = html.Div(
    style={"backgroundColor": "#0F172A", "minHeight": "100vh",
           "padding": "24px", "fontFamily": "sans-serif"},
    children=[
        html.H2("Casas de Empeño en México",
                style={"color": "#F8FAFC", "marginBottom": "4px"}),
        html.P("Registro Público de Casas de Empeño · PROFECO · 2014–2025",
               style={"color": "#64748B", "marginBottom": "20px"}),

        dcc.Tabs(id="tabs", value="tab-mercado", children=[

            dcc.Tab(label="Mercado", value="tab-mercado",
                    style=TAB_STYLE, selected_style=TAB_SEL, children=[
                html.Div(className="mt-3", children=[
                    _kpi_row,
                    dbc.Row([
                        dbc.Col(
                            html.Div(dcc.Graph(figure=_fig_cadenas),
                                     style={**CARD_STYLE, "padding": "12px"}),
                            md=12,
                        ),
                    ], className="mb-3"),
                ]),
            ]),

            dcc.Tab(label="Registro histórico", value="tab-temporal",
                    style=TAB_STYLE, selected_style=TAB_SEL, children=[
                html.Div(className="mt-3", children=[
                    dbc.Row([
                        dbc.Col(
                            html.Div(dcc.Graph(figure=_fig_año),
                                     style={**CARD_STYLE, "padding": "12px"}),
                            md=12,
                        ),
                    ], className="mb-3"),
                    dbc.Row([
                        dbc.Col(
                            html.Div(dcc.Graph(figure=_fig_fc_share),
                                     style={**CARD_STYLE, "padding": "12px"}),
                            md=12,
                        ),
                    ], className="mb-3"),
                ]),
            ]),

            dcc.Tab(label="Geografía", value="tab-geo",
                    style=TAB_STYLE, selected_style=TAB_SEL, children=[
                html.Div(className="mt-3", children=[
                    dbc.Row([
                        dbc.Col(
                            html.Div(dcc.Graph(figure=_fig_mapa),
                                     style={**CARD_STYLE, "padding": "8px"}),
                            md=7,
                        ),
                        dbc.Col(
                            html.Div(dcc.Graph(figure=_fig_estados),
                                     style={**CARD_STYLE, "padding": "8px"}),
                            md=5,
                        ),
                    ], className="mb-3 g-2"),
                    dbc.Row([
                        dbc.Col(
                            html.Div(dcc.Graph(figure=_fig_slope),
                                     style={**CARD_STYLE, "padding": "12px"}),
                            md=12,
                        ),
                    ], className="mb-3"),
                ]),
            ]),
        ]),

        html.P(
            "Fuente: PROFECO – Registro Público de Casas de Empeño | datos.gob.mx",
            style={"color": "#475569", "fontSize": "11px", "textAlign": "center",
                   "marginTop": "16px"},
        ),
    ],
)

if __name__ == "__main__":
    app.run(debug=True, port=8062)
