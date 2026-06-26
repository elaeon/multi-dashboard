import json
import re
import polars as pl
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
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

# ── CONAPO population data ────────────────────────────────────────────────────

_pob = pl.read_parquet("data/conapo/proyecciones_poblacion/pob_estado_año.parquet")

_annual_new = (
    df.filter(pl.col("año").is_between(2014, 2024))
    .group_by(["estado", "año"]).agg(pl.len().alias("new_branches"))
)
_estados_list = sorted(df.filter(pl.col("año").is_between(2014, 2024))["estado"].unique().to_list())
_grid = pl.DataFrame({
    "estado": [e for e in _estados_list for _ in range(11)],
    "año":    [y for _ in _estados_list for y in range(2014, 2025)],
})
agg_cumstock = (
    _grid.join(_annual_new, on=["estado", "año"], how="left")
    .with_columns(pl.col("new_branches").fill_null(0).cast(pl.Int64))
    .sort(["estado", "año"])
    .with_columns(pl.col("new_branches").cum_sum().over("estado").alias("cum_branches"))
)

agg_densidad = (
    agg_cumstock
    .join(
        _pob.filter(pl.col("año").is_between(2014, 2024)).select(["estado", "año", "pob_total"]),
        on=["estado", "año"], how="left",
    )
    .with_columns((pl.col("cum_branches") / pl.col("pob_total") * 100_000).round(2).alias("den_100k"))
)
agg_densidad_2024 = agg_densidad.filter(pl.col("año") == 2024).sort("den_100k", descending=True)

_nat_pop = (
    _pob.filter(pl.col("año").is_between(2014, 2024))
    .group_by("año").agg(pl.col("pob_total").sum().alias("pob_nacional"))
)
_nat_cum = (
    df.filter(pl.col("año").is_between(2014, 2024))
    .group_by("año").agg(pl.len().alias("new_nat")).sort("año")
    .with_columns(pl.col("new_nat").cum_sum().alias("cum_nat"))
)
agg_densidad_nacional = (
    _nat_cum.join(_nat_pop, on="año")
    .with_columns((pl.col("cum_nat") / pl.col("pob_nacional") * 100_000).round(3).alias("den_100k"))
    .sort("año")
)

agg_scatter = (
    agg_densidad_2024.select(["estado", "den_100k"])
    .join(_pob.filter(pl.col("año") == 2020).select(["estado", "raz_dep"]), on="estado", how="left")
)

_coneval = pl.read_parquet("data/coneval/deprivacion_estado_2022.parquet")

agg_combined = (
    agg_scatter
    .join(
        _coneval.select(["estado", "pct_carencias3", "promedio_carencias"]),
        on="estado", how="left",
    )
)
agg_coneval_scatter = (
    agg_densidad_2024.select(["estado", "den_100k"])
    .join(_coneval, on="estado", how="left")
)

_panel = pl.read_parquet("data/coneval/deprivacion_panel.parquet")

_IC_PANEL_COLS = ["pct_carencias3", "pct_ic_segsoc", "pct_ic_sbv",
                  "pct_ic_rezedu", "pct_ic_ali", "pct_ic_cv"]
_den_delta = (
    agg_densidad.filter(pl.col("año").is_in([2016, 2022]))
    .pivot(on="año", index="estado", values="den_100k")
    .rename({"2016": "den_2016", "2022": "den_2022"})
    .with_columns((pl.col("den_2022") - pl.col("den_2016")).alias("delta_den"))
)
_pan_2016 = _panel.filter(pl.col("año") == 2016).select(["estado"] + _IC_PANEL_COLS)
_pan_2022 = _panel.filter(pl.col("año") == 2022).select(["estado"] + _IC_PANEL_COLS)
agg_panel_delta = (
    _pan_2016
    .join(_pan_2022, on="estado", suffix="_22")
    .with_columns([
        (pl.col(f"{c}_22") - pl.col(c)).alias(f"delta_{c}")
        for c in _IC_PANEL_COLS
    ])
    .join(_den_delta, on="estado")
)

DEN_NAC_2024 = float(agg_densidad_nacional.filter(pl.col("año") == 2024)["den_100k"][0])
DEN_NAC_2016 = float(agg_densidad_nacional.filter(pl.col("año") == 2016)["den_100k"][0])
TOP_ESTADO_DEN     = str(agg_densidad_2024["estado"][0])
TOP_ESTADO_DEN_VAL = float(agg_densidad_2024["den_100k"][0])
BOT_ESTADO_DEN     = str(agg_densidad_2024["estado"][-1])
BOT_ESTADO_DEN_VAL = float(agg_densidad_2024["den_100k"][-1])

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


def fig_densidad_nacional() -> go.Figure:
    d = agg_densidad_nacional
    years = d["año"].to_list()
    vals  = d["den_100k"].to_list()

    fig = go.Figure(go.Scatter(
        x=years, y=vals,
        mode="lines+markers",
        line=dict(color=FOCUS, width=2.5),
        marker=dict(color=FOCUS, size=7),
        fill="tozeroy", fillcolor="rgba(46,134,171,0.1)",
        hovertemplate="<b>%{x}</b><br>Densidad: %{y:.2f} por 100k hab.<extra></extra>",
    ))
    fig.add_vline(x=2015, line_dash="dot", line_color="#F4A261",
                  annotation_text="Ola regulatoria 2015",
                  annotation_font_color="#F4A261", annotation_position="top right",
                  annotation_font_size=9)
    fig.add_vline(x=2020, line_dash="dot", line_color="#64748B",
                  annotation_text="COVID-19",
                  annotation_font_color="#64748B", annotation_position="top left",
                  annotation_font_size=9)
    fig.update_layout(**CHART_LAYOUT)
    fig.update_layout(
        title=dict(
            text=f"<b>La densidad nacional pasó de {DEN_NAC_2016:.1f} a {DEN_NAC_2024:.1f} sucursales por 100k hab. entre 2016 y 2024</b>"
                 "<br><sup style='color:#94A3B8'>Stock acumulado de casas de empeño registradas por 100,000 habitantes · PROFECO + CONAPO</sup>",
            font=dict(size=13),
        ),
        height=360,
        xaxis=dict(gridcolor="#334155", title="Año", dtick=1, tickangle=-45),
        yaxis=dict(gridcolor="#334155", title="Sucursales por 100k hab."),
        margin=dict(t=65, b=65, l=10, r=10),
    )
    return fig


def fig_ranking_densidad() -> go.Figure:
    d = agg_densidad_2024.sort("den_100k")
    labels = [ESTADO_LABEL.get(e, e) for e in d["estado"].to_list()]
    colors = [FOCUS if e == TOP_ESTADO_DEN else CONTEXT for e in d["estado"].to_list()]

    fig = go.Figure(go.Bar(
        x=d["den_100k"].to_list(), y=labels,
        orientation="h", marker_color=colors,
        hovertemplate="<b>%{y}</b><br>%{x:.1f} sucursales por 100k hab.<extra></extra>",
    ))
    fig.add_vline(x=DEN_NAC_2024, line_dash="dash", line_color="#64748B",
                  annotation_text=f"media nacional {DEN_NAC_2024:.1f}",
                  annotation_font_color="#94A3B8", annotation_font_size=9,
                  annotation_position="top right")
    top_label = ESTADO_LABEL.get(TOP_ESTADO_DEN, TOP_ESTADO_DEN)
    bot_label = ESTADO_LABEL.get(BOT_ESTADO_DEN, BOT_ESTADO_DEN)
    fig.update_layout(**CHART_LAYOUT)
    fig.update_layout(
        title=dict(
            text=f"<b>{top_label} lidera con {TOP_ESTADO_DEN_VAL:.1f}/100k — más del doble de la media; {bot_label} registra la menor penetración con {BOT_ESTADO_DEN_VAL:.1f}</b>"
                 "<br><sup style='color:#94A3B8'>Stock acumulado 2014–2024 por 100,000 habitantes · CONAPO 2024</sup>",
            font=dict(size=13),
        ),
        height=max(300, 32 * 22 + 80),
        yaxis=dict(gridcolor="#334155"),
        xaxis=dict(gridcolor="#334155", title="Sucursales por 100k hab."),
        margin=dict(t=80, b=40, l=10, r=10),
    )
    return fig


def fig_scatter_dep_densidad() -> go.Figure:
    d = agg_scatter
    estados = d["estado"].to_list()
    labels  = [ESTADO_LABEL.get(e, e) for e in estados]
    raz_dep = d["raz_dep"].to_list()
    den     = d["den_100k"].to_list()

    mean_dep = d["raz_dep"].mean()
    mean_den = d["den_100k"].mean()

    fig = go.Figure(go.Scatter(
        x=raz_dep, y=den,
        mode="markers",
        marker=dict(color=FOCUS, size=9, opacity=0.8),
        customdata=labels,
        hovertemplate="<b>%{customdata}</b><br>Razón de dependencia: %{x:.1f}<br>Densidad: %{y:.1f}/100k<extra></extra>",
    ))

    ANNOTATE = {"Yucatan", "Baja California Sur", "Chiapas", "Guerrero", "Michoacan"}
    for i, estado in enumerate(estados):
        if estado in ANNOTATE:
            label = ESTADO_LABEL.get(estado, estado)
            ax_off = 35 if raz_dep[i] > mean_dep else -35
            fig.add_annotation(
                x=raz_dep[i], y=den[i], text=f"<b>{label}</b>",
                font=dict(color="#F4A261", size=9),
                arrowcolor="#64748B", ax=ax_off, ay=-20,
                bgcolor="rgba(15,23,42,0.7)", borderpad=2,
            )

    fig.add_hline(y=mean_den, line_dash="dash", line_color="#64748B",
                  annotation_text=f"media densidad {mean_den:.1f}", annotation_font_color="#94A3B8",
                  annotation_font_size=9, annotation_position="bottom right")
    fig.add_vline(x=mean_dep, line_dash="dash", line_color="#64748B",
                  annotation_text=f"media dep. {mean_dep:.0f}", annotation_font_color="#94A3B8",
                  annotation_font_size=9, annotation_position="top left")

    fig.update_layout(**CHART_LAYOUT)
    fig.update_layout(
        title=dict(
            text="<b>A mayor razón de dependencia, menos empeños per cápita — el mercado formal se concentra donde la PEA es mayor</b>"
                 "<br><sup style='color:#94A3B8'>Razón de dependencia económica 2020 (CONAPO) vs densidad de sucursales 2024 · 32 estados</sup>",
            font=dict(size=13),
        ),
        height=420,
        xaxis=dict(gridcolor="#334155", title="Razón de dependencia (%)"),
        yaxis=dict(gridcolor="#334155", title="Sucursales por 100k hab. (2024)"),
        margin=dict(t=75, b=50, l=10, r=10),
    )
    return fig


def fig_scatter_combined() -> go.Figure:
    d = agg_combined.drop_nulls()
    estados  = d["estado"].to_list()
    labels   = [ESTADO_LABEL.get(e, e) for e in estados]
    x_vals   = d["raz_dep"].to_list()
    y_vals   = d["den_100k"].to_list()
    c_vals   = d["pct_carencias3"].to_list()
    prom_vals = d["promedio_carencias"].to_list()

    mean_x = float(d["raz_dep"].mean())
    mean_y = float(d["den_100k"].mean())

    fig = go.Figure(go.Scatter(
        x=x_vals, y=y_vals,
        mode="markers",
        marker=dict(
            color=c_vals,
            colorscale=[[0, "#2E86AB"], [0.5, "#F4A261"], [1, "#E84855"]],
            size=10,
            opacity=0.85,
            colorbar=dict(
                title=dict(text="% con 3+<br>carencias", font=dict(color="#CBD5E1", size=10)),
                tickfont=dict(color="#CBD5E1"),
                thickness=12,
                len=0.75,
            ),
            showscale=True,
            line=dict(color="#0F172A", width=0.5),
        ),
        customdata=list(zip(labels, c_vals, prom_vals)),
        hovertemplate=(
            "<b>%{customdata[0]}</b>"
            "<br>Razón de dependencia: %{x:.1f}"
            "<br>Densidad: %{y:.1f}/100k"
            "<br>3+ carencias: %{customdata[1]:.1f}%"
            "<br>Promedio carencias: %{customdata[2]:.2f}"
            "<extra></extra>"
        ),
    ))

    ANNOTATE = {"Yucatan", "Guerrero", "Chiapas", "Nuevo Leon"}
    for i, estado in enumerate(estados):
        if estado in ANNOTATE:
            label = ESTADO_LABEL.get(estado, estado)
            ax_off = 45 if x_vals[i] < mean_x else -45
            fig.add_annotation(
                x=x_vals[i], y=y_vals[i], text=f"<b>{label}</b>",
                font=dict(color="#F4A261", size=9),
                arrowcolor="#64748B", ax=ax_off, ay=-20,
                bgcolor="rgba(15,23,42,0.75)", borderpad=2,
            )

    fig.add_hline(y=mean_y, line_dash="dash", line_color="#64748B",
                  annotation_text=f"media densidad {mean_y:.1f}",
                  annotation_font_color="#94A3B8", annotation_font_size=9,
                  annotation_position="bottom right")
    fig.add_vline(x=mean_x, line_dash="dash", line_color="#64748B",
                  annotation_text=f"media dep. {mean_x:.0f}",
                  annotation_font_color="#94A3B8", annotation_font_size=9,
                  annotation_position="top left")

    for x_pos, y_pos, text in [
        (0.02, 0.97, "Baja dep.<br>Muchos empeños"),
        (0.98, 0.97, "Alta dep.<br>Muchos empeños"),
        (0.02, 0.03, "Baja dep.<br>Pocos empeños"),
        (0.98, 0.03, "Alta dep.<br>Pocos empeños"),
    ]:
        fig.add_annotation(
            x=x_pos, y=y_pos, xref="paper", yref="paper",
            text=text, font=dict(color="#475569", size=8),
            showarrow=False, align="center",
        )

    fig.update_layout(**CHART_LAYOUT)
    fig.update_layout(
        title=dict(
            text=(
                "<b>Los estados con mayor rezago social y mayor dependencia tienen"
                " sistemáticamente menos empeños per cápita</b>"
                "<br><sup style='color:#94A3B8'>Razón de dependencia (CONAPO 2020) · Densidad 2024 · "
                "color = % con 3+ carencias sociales (CONEVAL 2022) · 32 estados</sup>"
            ),
            font=dict(size=13),
        ),
        height=480,
        xaxis=dict(gridcolor="#334155", title="Razón de dependencia (%)"),
        yaxis=dict(gridcolor="#334155", title="Sucursales por 100k hab. (2024)"),
        margin=dict(t=80, b=50, l=10, r=80),
    )
    return fig


def fig_slope_densidad() -> go.Figure:
    Y0, Y1 = 2016, 2024
    a = agg_densidad.filter(pl.col("año") == Y0)[["estado", "den_100k"]].rename({"den_100k": "den_0"})
    b = agg_densidad.filter(pl.col("año") == Y1)[["estado", "den_100k"]].rename({"den_100k": "den_1"})
    slope_d = (
        a.join(b, on="estado", how="inner")
        .with_columns((pl.col("den_1") - pl.col("den_0")).alias("delta"))
        .sort("den_1", descending=True)
    )
    delta_median = float(slope_d["delta"].median())
    n_total = len(slope_d)

    fig = go.Figure()
    for i, row in enumerate(slope_d.iter_rows(named=True)):
        color = FOCUS if row["delta"] >= delta_median else CONTEXT
        width = 2 if row["delta"] >= delta_median else 1
        show_label = i < 4
        label = ESTADO_LABEL.get(row["estado"], row["estado"])

        fig.add_trace(go.Scatter(
            x=[str(Y0), str(Y1)],
            y=[row["den_0"], row["den_1"]],
            mode="lines+markers" + ("+text" if show_label else ""),
            line=dict(color=color, width=width),
            marker=dict(color=color, size=6 if width == 2 else 4),
            text=[None, label] if show_label else None,
            textposition="middle right",
            textfont=dict(size=8, color=FOCUS),
            showlegend=False,
            hovertemplate=f"<b>{label}</b><br>%{{x}}: %{{y:.2f}}/100k  Δ {row['delta']:+.2f}<extra></extra>",
        ))

    fig.add_hline(y=DEN_NAC_2024, line_dash="dash", line_color="#64748B",
                  annotation_text=f"media nacional 2024 · {DEN_NAC_2024:.1f}",
                  annotation_font_color="#94A3B8", annotation_font_size=9,
                  annotation_position="bottom right")

    n_above = slope_d.filter(pl.col("delta") >= delta_median).height
    n_below = n_total - n_above
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines+markers",
                              line=dict(color=FOCUS), marker=dict(color=FOCUS, size=7),
                              name=f"Mayor crecimiento per cápita ({n_above} estados)"))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines",
                              line=dict(color=CONTEXT), name=f"Menor crecimiento ({n_below} estados)"))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#CBD5E1",
        title=dict(
            text="<b>Todos los estados aumentaron su densidad per cápita 2016–2024; Yucatán, BCS y Nayarit con el mayor salto</b>"
                 "<br><sup style='color:#94A3B8'>Sucursales acumuladas por 100,000 habitantes · PROFECO + CONAPO</sup>",
            font=dict(size=13),
        ),
        height=max(400, n_total * 20 + 100),
        xaxis=dict(type="category", gridcolor="rgba(0,0,0,0)"),
        yaxis=dict(gridcolor="#334155", title="Sucursales por 100k hab."),
        legend=dict(orientation="h", y=-0.04, x=0, font=dict(size=10)),
        margin=dict(t=65, b=70, l=10, r=180),
    )
    return fig


_IC_PANEL_META = [
    ("delta_pct_carencias3", "3+ carencias"),
    ("delta_pct_ic_segsoc",  "Sin seguridad social"),
    ("delta_pct_ic_sbv",     "Sin servicios básicos"),
    ("delta_pct_ic_rezedu",  "Rezago educativo"),
    ("delta_pct_ic_ali",     "Carencia alimentaria"),
    ("delta_pct_ic_cv",      "Calidad de vivienda"),
]


def fig_panel_delta() -> go.Figure:
    d       = agg_panel_delta
    estados = d["estado"].to_list()
    labels  = [ESTADO_LABEL.get(e, e) for e in estados]
    x_vals  = d["delta_den"].to_list()
    mean_x  = float(d["delta_den"].mean())

    subplot_titles = [m[1] for m in _IC_PANEL_META]
    fig = make_subplots(
        rows=2, cols=3,
        shared_xaxes=True,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.08,
        vertical_spacing=0.18,
    )

    for idx, (col, _) in enumerate(_IC_PANEL_META):
        row, col_pos = divmod(idx, 3)
        y_vals = d[col].to_list()

        colors = [FOCUS if y < 0 else "#E84855" for y in y_vals]

        fig.add_trace(
            go.Scatter(
                x=x_vals, y=y_vals,
                mode="markers",
                marker=dict(color=colors, size=8, opacity=0.85),
                customdata=list(zip(labels, y_vals)),
                hovertemplate=(
                    "<b>%{customdata[0]}</b>"
                    "<br>Δ densidad: %{x:+.2f}/100k"
                    "<br>Δ carencia: %{customdata[1]:+.1f} pp"
                    "<extra></extra>"
                ),
                showlegend=False,
            ),
            row=row + 1, col=col_pos + 1,
        )

        fig.add_hline(y=0, row=row + 1, col=col_pos + 1,
                      line_dash="dash", line_color="#64748B", line_width=1)
        fig.add_vline(x=mean_x, row=row + 1, col=col_pos + 1,
                      line_dash="dash", line_color="#64748B", line_width=1)

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#CBD5E1",
        title=dict(
            text="<b>La expansión de empeños 2016–2022 no predice mejoras en carencias — correlación prácticamente nula (r ≈ −0.05)</b>"
                 "<br><sup style='color:#94A3B8'>Δ densidad sucursales/100k hab. vs Δ pp de cada carencia · azul = mejoró · rojo = empeoró · eje vertical = sin cambio · CONEVAL + PROFECO</sup>",
            font=dict(size=13),
        ),
        height=680,
        margin=dict(t=80, b=50, l=10, r=10),
    )
    for ann in fig.layout.annotations:
        ann.font.color = "#94A3B8"
        ann.font.size  = 11

    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
                              marker=dict(color=FOCUS, size=9),
                              name="Carencia mejoró (↓ pp)"))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
                              marker=dict(color="#E84855", size=9),
                              name="Carencia empeoró (↑ pp)"))
    fig.update_layout(
        legend=dict(orientation="h", y=-0.04, x=0, font=dict(size=10)),
    )
    return fig


_IC_META = [
    ("pct_ic_segsoc", "Sin seguridad social"),
    ("pct_ic_sbv",    "Sin servicios básicos en vivienda"),
    ("pct_ic_rezedu", "Rezago educativo"),
    ("pct_ic_ali",    "Carencia alimentaria"),
    ("pct_ic_asalud", "Sin acceso a salud"),
    ("pct_ic_cv",     "Calidad de vivienda"),
]


def fig_scatter_carencias() -> go.Figure:
    d = agg_coneval_scatter
    estados = d["estado"].to_list()
    labels  = [ESTADO_LABEL.get(e, e) for e in estados]
    y_vals  = d["den_100k"].to_list()
    y_max   = float(d["den_100k"].max()) * 1.12
    mean_y  = float(d["den_100k"].mean())

    subplot_titles = [meta[1] for meta in _IC_META]
    fig = make_subplots(
        rows=2, cols=3,
        shared_yaxes=True,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.06,
        vertical_spacing=0.16,
    )

    for idx, (col, _) in enumerate(_IC_META):
        row, col_pos = divmod(idx, 3)
        x_vals = d[col].to_list()
        x_max  = max(x_vals) * 1.08

        fig.add_trace(
            go.Scatter(
                x=x_vals, y=y_vals,
                mode="markers",
                marker=dict(color=FOCUS, size=7, opacity=0.75),
                customdata=list(zip(labels, x_vals)),
                hovertemplate=(
                    "<b>%{customdata[0]}</b>"
                    "<br>Carencia: %{customdata[1]:.1f}%"
                    "<br>Densidad: %{y:.1f}/100k"
                    "<extra></extra>"
                ),
                showlegend=False,
            ),
            row=row + 1, col=col_pos + 1,
        )

        mean_x = float(d[col].mean())
        fig.add_vline(x=mean_x, row=row + 1, col=col_pos + 1,
                      line_dash="dash", line_color="#64748B", line_width=1)
        fig.add_hline(y=mean_y, row=row + 1, col=col_pos + 1,
                      line_dash="dash", line_color="#64748B", line_width=1)

        axis_n = "" if idx == 0 else str(idx + 1)
        fig.update_layout(**{
            f"xaxis{axis_n}": dict(gridcolor="#334155", range=[0, x_max], ticksuffix="%"),
            f"yaxis{axis_n}": dict(gridcolor="#334155", range=[0, y_max]),
        })

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#CBD5E1",
        title=dict(
            text="<b>La carencia de seguridad social y la falta de servicios básicos son las que más se asocian con menor densidad de empeños</b>"
                 "<br><sup style='color:#94A3B8'>% de población con cada carencia social por estado (CONEVAL 2022) vs sucursales registradas por 100k hab. (2024)</sup>",
            font=dict(size=13),
        ),
        height=680,
        margin=dict(t=80, b=40, l=10, r=10),
    )
    for ann in fig.layout.annotations:
        ann.font.color = "#94A3B8"
        ann.font.size  = 11

    return fig


def fig_scatter_coneval() -> go.Figure:
    d = agg_coneval_scatter
    estados = d["estado"].to_list()
    labels  = [ESTADO_LABEL.get(e, e) for e in estados]
    x_vals  = d["pct_carencias3"].to_list()
    y_vals  = d["den_100k"].to_list()

    mean_x = float(d["pct_carencias3"].mean())
    mean_y = float(d["den_100k"].mean())

    fig = go.Figure(go.Scatter(
        x=x_vals, y=y_vals,
        mode="markers",
        marker=dict(color=FOCUS, size=9, opacity=0.8),
        customdata=list(zip(labels, d["pct_pobreza"].to_list())),
        hovertemplate=(
            "<b>%{customdata[0]}</b>"
            "<br>3+ carencias: %{x:.1f}%"
            "<br>Pobreza: %{customdata[1]:.1f}%"
            "<br>Densidad empeños: %{y:.1f}/100k"
            "<extra></extra>"
        ),
    ))

    ANNOTATE_HIGH = {"Chiapas", "Guerrero", "Oaxaca"}
    ANNOTATE_LOW  = {"Yucatan", "Baja California Sur", "Nuevo Leon"}
    for i, estado in enumerate(estados):
        if estado in ANNOTATE_HIGH | ANNOTATE_LOW:
            label = ESTADO_LABEL.get(estado, estado)
            ax_off = -45 if x_vals[i] > mean_x else 45
            fig.add_annotation(
                x=x_vals[i], y=y_vals[i], text=f"<b>{label}</b>",
                font=dict(color="#F4A261", size=9),
                arrowcolor="#64748B", ax=ax_off, ay=-18,
                bgcolor="rgba(15,23,42,0.75)", borderpad=2,
            )

    fig.add_hline(y=mean_y, line_dash="dash", line_color="#64748B",
                  annotation_text=f"media densidad {mean_y:.1f}", annotation_font_color="#94A3B8",
                  annotation_font_size=9, annotation_position="bottom right")
    fig.add_vline(x=mean_x, line_dash="dash", line_color="#64748B",
                  annotation_text=f"media carencias {mean_x:.0f}%", annotation_font_color="#94A3B8",
                  annotation_font_size=9, annotation_position="top left")

    fig.add_annotation(
        x=0.97, y=0.97, xref="paper", yref="paper",
        text="Alto rezago<br>Pocos empeños",
        font=dict(color="#64748B", size=9), showarrow=False, align="right",
    )
    fig.add_annotation(
        x=0.03, y=0.03, xref="paper", yref="paper",
        text="Bajo rezago<br>Muchos empeños",
        font=dict(color="#64748B", size=9), showarrow=False, align="left",
    )

    fig.update_layout(**CHART_LAYOUT)
    fig.update_layout(
        title=dict(
            text="<b>Las casas de empeño NO son indicador de rezago social — los estados con más carencias tienen MENOS empeños per cápita</b>"
                 "<br><sup style='color:#94A3B8'>% población con 3+ carencias sociales (CONEVAL 2022) vs densidad de sucursales registradas (PROFECO 2024)</sup>",
            font=dict(size=13),
        ),
        height=440,
        xaxis=dict(gridcolor="#334155", title="Población con 3 o más carencias sociales (%)"),
        yaxis=dict(gridcolor="#334155", title="Sucursales por 100k hab. (2024)"),
        margin=dict(t=80, b=50, l=10, r=10),
    )
    return fig


# ── Pre-compute all figures ───────────────────────────────────────────────────

_fig_cadenas = fig_ranking_cadenas()
_fig_año = fig_por_año()
_fig_fc_share = fig_share_first_cash()
_fig_mapa = fig_mapa()
_fig_estados = fig_ranking_estados()
_fig_slope = fig_slope_geografico()
_fig_densidad_nac = fig_densidad_nacional()
_fig_ranking_den  = fig_ranking_densidad()
_fig_scatter_dep  = fig_scatter_dep_densidad()
_fig_combined     = fig_scatter_combined()
_fig_slope_den    = fig_slope_densidad()
_fig_coneval      = fig_scatter_coneval()
_fig_carencias    = fig_scatter_carencias()
_fig_panel_delta  = fig_panel_delta()

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
                    dbc.Row([
                        dbc.Col(
                            html.Div(dcc.Graph(figure=_fig_densidad_nac),
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

            dcc.Tab(label="Densidad", value="tab-densidad",
                    style=TAB_STYLE, selected_style=TAB_SEL, children=[
                html.Div(className="mt-3", children=[
                    dbc.Row([
                        kpi_card(
                            "Densidad nacional 2024",
                            f"{DEN_NAC_2024:.1f} por 100k",
                            FOCUS,
                            f"+{(DEN_NAC_2024/DEN_NAC_2016 - 1)*100:.0f}% vs 2016",
                            "#3BB273",
                        ),
                        kpi_card(
                            "Mayor penetración",
                            f"{ESTADO_LABEL.get(TOP_ESTADO_DEN, TOP_ESTADO_DEN)} · {TOP_ESTADO_DEN_VAL:.1f}/100k",
                            "#F4A261",
                        ),
                        kpi_card(
                            "Menor penetración",
                            f"{ESTADO_LABEL.get(BOT_ESTADO_DEN, BOT_ESTADO_DEN)} · {BOT_ESTADO_DEN_VAL:.1f}/100k",
                            "#94A3B8",
                        ),
                        kpi_card(
                            "Correlación rezago–empeños",
                            "Inversa",
                            "#94A3B8",
                            "más carencias → menos empeños/100k · CONEVAL 2022",
                        ),
                    ], className="mb-3 g-2"),
                    dbc.Row([
                        dbc.Col(
                            html.Div(dcc.Graph(figure=_fig_slope_den),
                                     style={**CARD_STYLE, "padding": "12px"}),
                            md=12,
                        ),
                    ], className="mb-3"),
                    dbc.Row([
                        dbc.Col(
                            html.Div(dcc.Graph(figure=_fig_ranking_den),
                                     style={**CARD_STYLE, "padding": "8px"}),
                            md=5,
                        ),
                        dbc.Col(
                            html.Div(dcc.Graph(figure=_fig_scatter_dep),
                                     style={**CARD_STYLE, "padding": "8px"}),
                            md=7,
                        ),
                    ], className="mb-3 g-2"),
                    dbc.Row([
                        dbc.Col(
                            html.Div(dcc.Graph(figure=_fig_combined),
                                     style={**CARD_STYLE, "padding": "8px"}),
                            md=12,
                        ),
                    ], className="mb-3"),
                    dbc.Row([
                        dbc.Col(
                            html.Div(dcc.Graph(figure=_fig_coneval),
                                     style={**CARD_STYLE, "padding": "8px"}),
                            md=12,
                        ),
                    ], className="mb-3"),
                    dbc.Row([
                        dbc.Col(
                            html.Div(dcc.Graph(figure=_fig_carencias),
                                     style={**CARD_STYLE, "padding": "8px"}),
                            md=12,
                        ),
                    ], className="mb-3"),
                    dbc.Row([
                        dbc.Col(
                            html.Div(dcc.Graph(figure=_fig_panel_delta),
                                     style={**CARD_STYLE, "padding": "8px"}),
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
