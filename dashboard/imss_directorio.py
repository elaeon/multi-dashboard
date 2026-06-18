import json
import polars as pl
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc

# ── Data loading & cleaning ───────────────────────────────────────────────────

_RAW = pl.read_csv("data/imss/directorio_imss.csv", null_values=[""])

# Drop TCPDF artifact (last row)
_raw = _RAW.filter(~pl.col("Nombre de la Unidad").str.contains("TCPDF", literal=True))

# Canonical tipo tokens (ordered longest-first for prefix matching)
_CANONICAL_TIPOS = [
    "PUESTOS DE VACUNACIÓN HOSPITAL",
    "ADMINISTRATI VAS SUBDELEGACIÓN",
    "PUESTOS DE VACUNACIÓN CLÍNICA",
    "UNIDADES MEDICAS IMSS BIENESTAR",
    "ADMINISTRATI VAS DELEGACIÓN",
    "UNIDADES MEDICAS HOSPITAL",
    "OTROS SERVICIOS GUARDERÍA",
    "OTROS SERVICIOS CENTRO DE",
    "OTROS SERVICIOS VELATORIO",
    "UNIDADES MEDICAS CLÍNICA",
    "OTROS SERVICIOS TEATRO",
    "OTROS SERVICIOS TIENDA",
    "OTROS SERVICIOS UNIDAD",
    "PUESTOS DE VACUNACIÓN",
    "MÓDULOS DE ATENCIÓN",
    "UNIDADES MEDICAS",
    "ADMINISTRATI VAS",
    "OTROS SERVICIOS",
]
_tipo_expr = pl.when(pl.col("Tipo de Unidad").is_null()).then(pl.lit(None))
for _c in _CANONICAL_TIPOS:
    _tipo_expr = _tipo_expr.when(pl.col("Tipo de Unidad").str.starts_with(_c)).then(pl.lit(_c))
_tipo_expr = _tipo_expr.otherwise(pl.lit(None))

# Known state tokens (longest-first to avoid prefix conflicts)
_ESTADO_TOKENS = sorted([
    "AGUASCALIENTES", "BAJA CALIFORNIA SUR", "BAJA CALIFORNIA",
    "CAMPECHE", "CHIAPAS", "CHIHUAHUA", "CIUDAD DE MÉXICO",
    "COAHUILA", "COLIMA", "DURANGO", "GUANAJUATO", "GUERRERO",
    "HIDALGO", "JALISCO", "MÉXICO", "MICHOACÁN", "MORELOS",
    "NAYARIT", "NUEVO LEÓN", "OAXACA", "PUEBLA", "QUERÉTARO",
    "QUINTANA ROO", "SAN LUIS POTOSÍ", "SINALOA", "SONORA",
    "TABASCO", "TAMAULIPAS", "TLAXCALA", "VERACRUZ", "YUCATÁN",
    "ZACATECAS", "MONTERREY",
], key=len, reverse=True)
_estado_expr = pl.when(pl.col("Estado").is_null()).then(pl.lit(None))
for _t in _ESTADO_TOKENS:
    _estado_expr = _estado_expr.when(pl.col("Estado").str.starts_with(_t)).then(pl.lit(_t))
_estado_expr = _estado_expr.otherwise(pl.lit(None))

# Uppercase state → GeoJSON title-case name
_ESTADO_TO_GEO = {
    "AGUASCALIENTES": "Aguascalientes",
    "BAJA CALIFORNIA SUR": "Baja California Sur",
    "BAJA CALIFORNIA": "Baja California",
    "CAMPECHE": "Campeche",
    "CHIAPAS": "Chiapas",
    "CHIHUAHUA": "Chihuahua",
    "CIUDAD DE MÉXICO": "Ciudad de México",
    "COAHUILA": "Coahuila",
    "COLIMA": "Colima",
    "DURANGO": "Durango",
    "GUANAJUATO": "Guanajuato",
    "GUERRERO": "Guerrero",
    "HIDALGO": "Hidalgo",
    "JALISCO": "Jalisco",
    "MÉXICO": "México",
    "MICHOACÁN": "Michoacán",
    "MORELOS": "Morelos",
    "NAYARIT": "Nayarit",
    "NUEVO LEÓN": "Nuevo León",
    "MONTERREY": "Nuevo León",
    "OAXACA": "Oaxaca",
    "PUEBLA": "Puebla",
    "QUERÉTARO": "Querétaro",
    "QUINTANA ROO": "Quintana Roo",
    "SAN LUIS POTOSÍ": "San Luis Potosí",
    "SINALOA": "Sinaloa",
    "SONORA": "Sonora",
    "TABASCO": "Tabasco",
    "TAMAULIPAS": "Tamaulipas",
    "TLAXCALA": "Tlaxcala",
    "VERACRUZ": "Veracruz",
    "YUCATÁN": "Yucatán",
    "ZACATECAS": "Zacatecas",
}

# Canonical tipo → display group
_TIPO_GROUPS = {
    "UNIDADES MEDICAS IMSS BIENESTAR": "IMSS Bienestar",
    "UNIDADES MEDICAS CLÍNICA": "Clínica",
    "UNIDADES MEDICAS HOSPITAL": "Hospital",
    "UNIDADES MEDICAS": "Unidades médicas",
    "PUESTOS DE VACUNACIÓN CLÍNICA": "Vacunación",
    "PUESTOS DE VACUNACIÓN HOSPITAL": "Vacunación",
    "PUESTOS DE VACUNACIÓN": "Vacunación",
    "OTROS SERVICIOS GUARDERÍA": "Guardería",
    "MÓDULOS DE ATENCIÓN": "Módulos",
    "OTROS SERVICIOS CENTRO DE": "Otros servicios",
    "OTROS SERVICIOS TEATRO": "Otros servicios",
    "OTROS SERVICIOS TIENDA": "Otros servicios",
    "OTROS SERVICIOS VELATORIO": "Otros servicios",
    "OTROS SERVICIOS UNIDAD": "Otros servicios",
    "OTROS SERVICIOS": "Otros servicios",
    "ADMINISTRATI VAS SUBDELEGACIÓN": "Administrativo",
    "ADMINISTRATI VAS DELEGACIÓN": "Administrativo",
    "ADMINISTRATI VAS": "Administrativo",
}

df = (
    _raw
    .with_columns([
        _tipo_expr.alias("tipo_clean"),
        _estado_expr.alias("_er"),
    ])
    .with_columns([
        pl.col("_er").replace(_ESTADO_TO_GEO).alias("estado_geo"),
        pl.col("tipo_clean").replace(_TIPO_GROUPS).alias("tipo_grupo"),
    ])
    .drop("_er")
)

# CONAPO 2024 state population
_CONAPO_TO_GEO = {
    "Coahuila de Zaragoza": "Coahuila",
    "Michoacán de Ocampo": "Michoacán",
    "Veracruz de Ignacio de la Llave": "Veracruz",
}
pop_by_state = (
    pl.scan_csv("data/conapo/estados_municipios.csv")
    .filter(pl.col("AÑO") == 2024)
    .group_by("NOM_ENT")
    .agg(pl.col("POB_TOTAL").sum().alias("pop"))
    .collect()
    .with_columns(
        pl.col("NOM_ENT").replace(_CONAPO_TO_GEO).alias("estado_geo")
    )
    .drop("NOM_ENT")
)

pop_by_state_2025 = (
    pl.scan_csv("data/conapo/estados_municipios.csv")
    .filter(pl.col("AÑO") == 2025)
    .group_by("NOM_ENT")
    .agg(pl.col("POB_TOTAL").sum().alias("pop"))
    .collect()
    .with_columns(pl.col("NOM_ENT").replace(_CONAPO_TO_GEO).alias("estado_geo"))
    .drop("NOM_ENT")
)

with open("data/mexico_states.geojson") as _f:
    MEXICO_GEO = json.load(_f)

TIPO_ORDER = [
    "IMSS Bienestar", "Clínica", "Vacunación", "Guardería",
    "Módulos", "Hospital", "Otros servicios", "Unidades médicas", "Administrativo",
]
TIPO_COLORS = {
    "IMSS Bienestar":  "#2E86AB",
    "Clínica":         "#3BB273",
    "Vacunación":      "#F4A261",
    "Guardería":       "#A78BFA",
    "Módulos":         "#64748B",
    "Hospital":        "#E84855",
    "Otros servicios": "#475569",
    "Unidades médicas":"#94A3B8",
    "Administrativo":  "#334155",
}

_tipo_coverage_opts = [{"label": "Todos los tipos", "value": "all"}] + [
    {"label": t, "value": t} for t in TIPO_ORDER if t != "Administrativo"
]

# ── Theme ─────────────────────────────────────────────────────────────────────

CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font_color="#CBD5E1",
    xaxis=dict(gridcolor="#334155"),
    yaxis=dict(gridcolor="#334155"),
    margin=dict(t=50, b=40, l=10, r=10),
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


def kpi_card(title: str, value: str, sub: str = "", color: str = "#CBD5E1") -> dbc.Col:
    return dbc.Col(
        html.Div([
            html.P(title, style={"color": "#94A3B8", "fontSize": "12px", "margin": 0}),
            html.H3(value, style={"color": color, "margin": "4px 0 0", "fontSize": "1.6rem"}),
            html.P(sub, style={"color": "#64748B", "fontSize": "11px", "margin": "2px 0 0"}),
        ], style=CARD_STYLE),
        xs=12, sm=6, md=3,
    )


def compute_kpis(d: pl.DataFrame, estado: str) -> list:
    total = len(d)
    n_bienestar = int((d["tipo_grupo"] == "IMSS Bienestar").sum())
    pct_bienestar = n_bienestar / total * 100 if total > 0 else 0
    n_hospitales = int((d["tipo_grupo"] == "Hospital").sum())
    n_estados = int(d.filter(pl.col("estado_geo").is_not_null())["estado_geo"].n_unique())

    if estado != "all":
        pop_row = pop_by_state.filter(pl.col("estado_geo") == estado)
        pop = int(pop_row["pop"][0]) if len(pop_row) > 0 else 1
        label_h = f"{n_hospitales / pop * 1_000_000:.1f} por millón de hab."
    else:
        pop = int(pop_by_state["pop"].sum())
        label_h = f"{n_hospitales / pop * 1_000_000:.1f} por millón · nacional"

    return [
        kpi_card("Unidades en el directorio", f"{total:,}"),
        kpi_card("IMSS Bienestar", f"{pct_bienestar:.1f}%",
                 sub=f"{n_bienestar:,} unidades", color="#2E86AB"),
        kpi_card("Hospitales IMSS", f"{n_hospitales:,}",
                 sub=label_h, color="#E84855"),
        kpi_card("Estados con presencia", f"{n_estados}/32"),
    ]


# ── Figure factories ──────────────────────────────────────────────────────────

def fig_choropleth(d: pl.DataFrame) -> go.Figure:
    state_counts = (
        d.filter(pl.col("estado_geo").is_not_null())
        .group_by("estado_geo")
        .agg([
            pl.len().alias("total"),
            (pl.col("tipo_grupo") == "IMSS Bienestar").sum().alias("bienestar"),
        ])
        .with_columns(
            (pl.col("bienestar") / pl.col("total") * 100).round(1).alias("pct_bienestar")
        )
    )
    fig = px.choropleth_map(
        state_counts,
        geojson=MEXICO_GEO,
        locations="estado_geo",
        color="pct_bienestar",
        featureidkey="properties.name",
        color_continuous_scale="Blues",
        range_color=[0, 100],
        zoom=3.8,
        center={"lat": 23.6, "lon": -102.5},
        opacity=0.85,
        hover_name="estado_geo",
        custom_data=["total", "bienestar", "pct_bienestar"],
        map_style="carto-darkmatter",
    )
    fig.update_traces(
        hovertemplate=(
            "<b>%{hovertext}</b><br>"
            "Bienestar: %{customdata[2]:.1f}%<br>"
            "%{customdata[1]:,} de %{customdata[0]:,} unidades<extra></extra>"
        )
    )
    fig.update_coloraxes(
        colorbar=dict(
            title=dict(text="% Bienestar", font=dict(color="#CBD5E1")),
            tickfont=dict(color="#CBD5E1"),
            ticksuffix="%",
        )
    )
    fig.update_layout(
        title=dict(
            text=(
                "<b>En Chiapas, el 82% de las unidades IMSS son del programa Bienestar</b>"
                "<br><sup style='color:#94A3B8'>% unidades Bienestar por estado · 13 estados registran 0% en el directorio</sup>"
            ),
            font=dict(color="#F8FAFC", size=13),
        ),
        height=520,
        margin=dict(l=0, r=0, t=60, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#CBD5E1",
    )
    return fig


def fig_state_ranking(d: pl.DataFrame, focus_estado: str = None) -> go.Figure:
    agg = (
        d.filter(pl.col("estado_geo").is_not_null())
        .group_by("estado_geo")
        .agg(pl.len().alias("n"))
        .sort("n", descending=True)
        .head(15)
    )
    states = agg["estado_geo"].to_list()
    FOCUS, CONTEXT = "#2E86AB", "#475569"
    colors = [FOCUS if (focus_estado and s == focus_estado) or (not focus_estado and i == 0) else CONTEXT
              for i, s in enumerate(states)]
    fig = go.Figure(go.Bar(
        x=agg["n"].to_list(),
        y=states,
        orientation="h",
        marker_color=colors,
        hovertemplate="<b>%{y}</b><br>Unidades: %{x:,}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(
            text=(
                "<b>Veracruz, Chiapas y Oaxaca: 1 de cada 3 unidades IMSS</b>"
                "<br><sup style='color:#94A3B8'>Top 15 estados por número de unidades</sup>"
            ),
            font=dict(color="#F8FAFC", size=13),
        ),
        yaxis=dict(autorange="reversed", gridcolor="rgba(0,0,0,0)"),
        height=max(300, len(states) * 26 + 100),
        **{k: v for k, v in CHART_LAYOUT.items() if k not in ("xaxis", "yaxis", "margin")},
        margin=dict(t=60, b=40, l=10, r=10),
    )
    return fig


def fig_type_dist(d: pl.DataFrame) -> go.Figure:
    agg = (
        d.filter(pl.col("tipo_grupo").is_not_null())
        .group_by("tipo_grupo")
        .agg(pl.len().alias("n"))
        .sort("n", descending=True)
    )
    if len(agg) == 0:
        return go.Figure()
    tipos = agg["tipo_grupo"].to_list()
    counts = agg["n"].to_list()
    colors = [TIPO_COLORS.get(t, "#475569") for t in tipos]
    fig = go.Figure(go.Bar(
        x=counts,
        y=tipos,
        orientation="h",
        marker_color=colors,
        hovertemplate="<b>%{y}</b><br>Unidades: %{x:,}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(
            text=(
                "<b>IMSS Bienestar: el tipo más numeroso del directorio</b>"
                "<br><sup style='color:#94A3B8'>Unidades por categoría (selección activa)</sup>"
            ),
            font=dict(color="#F8FAFC", size=13),
        ),
        yaxis=dict(autorange="reversed", gridcolor="rgba(0,0,0,0)"),
        height=max(300, len(tipos) * 38 + 100),
        **{k: v for k, v in CHART_LAYOUT.items() if k not in ("xaxis", "yaxis", "margin")},
        margin=dict(t=60, b=40, l=10, r=10),
    )
    return fig


def fig_type_by_state(d: pl.DataFrame) -> go.Figure:
    typed = d.filter(pl.col("estado_geo").is_not_null() & pl.col("tipo_grupo").is_not_null())
    top_states = (
        typed.group_by("estado_geo").agg(pl.len().alias("total"))
        .sort("total", descending=True).head(10)["estado_geo"].to_list()
    )
    agg = (
        typed.filter(pl.col("estado_geo").is_in(top_states))
        .group_by(["estado_geo", "tipo_grupo"]).agg(pl.len().alias("n"))
    )
    totals = agg.group_by("estado_geo").agg(pl.col("n").sum().alias("total"))
    agg_pct = (
        agg.join(totals, on="estado_geo")
        .with_columns((pl.col("n") / pl.col("total") * 100).alias("pct"))
    )

    fig = go.Figure()
    for tipo in TIPO_ORDER:
        sub = agg_pct.filter(pl.col("tipo_grupo") == tipo)
        if len(sub) == 0:
            continue
        pct_map = dict(zip(sub["estado_geo"].to_list(), sub["pct"].to_list()))
        n_map   = dict(zip(sub["estado_geo"].to_list(), sub["n"].to_list()))
        pcts = [pct_map.get(s, 0) for s in top_states]
        ns   = [n_map.get(s, 0) for s in top_states]
        fig.add_trace(go.Bar(
            x=pcts, y=top_states, orientation="h", name=tipo,
            marker_color=TIPO_COLORS.get(tipo, "#475569"),
            customdata=ns,
            text=[f"{p:.0f}%" if p >= 6 else "" for p in pcts],
            textposition="inside", insidetextanchor="middle",
            hovertemplate=f"<b>%{{y}}</b> · {tipo}<br>%{{x:.1f}}%  (%{{customdata:,}})<extra></extra>",
        ))
    fig.update_layout(
        barmode="stack",
        title=dict(
            text=(
                "<b>Chiapas y Oaxaca: 8 de cada 10 unidades son IMSS Bienestar</b>"
                "<br><sup style='color:#94A3B8'>Composición por tipo, top 10 estados</sup>"
            ),
            font=dict(color="#F8FAFC", size=13),
        ),
        xaxis=dict(range=[0, 100], visible=False, gridcolor="rgba(0,0,0,0)"),
        yaxis=dict(autorange="reversed", gridcolor="rgba(0,0,0,0)"),
        legend=dict(orientation="h", y=-0.22, x=0),
        height=max(300, len(top_states) * 32 + 130),
        margin=dict(t=60, b=110, l=10, r=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#CBD5E1",
    )
    return fig


def _coverage_data(tipo: str) -> pl.DataFrame:
    """Left-join all 32 states with facility counts for the given tipo (0 if absent)."""
    if tipo == "all":
        counts = (
            df.filter(pl.col("estado_geo").is_not_null())
            .group_by("estado_geo").agg(pl.len().alias("n"))
        )
    else:
        counts = (
            df.filter(pl.col("estado_geo").is_not_null() & (pl.col("tipo_grupo") == tipo))
            .group_by("estado_geo").agg(pl.len().alias("n"))
        )
    return (
        pop_by_state_2025
        .join(counts, on="estado_geo", how="left")
        .with_columns(pl.col("n").fill_null(0))
        .sort("n", descending=True)
    )


def fig_coverage_bar(tipo: str) -> go.Figure:
    cov = _coverage_data(tipo)
    label = "Todos los tipos" if tipo == "all" else tipo
    color = TIPO_COLORS.get(tipo, "#2E86AB")
    fig = go.Figure(go.Bar(
        x=cov["n"].to_list(),
        y=cov["estado_geo"].to_list(),
        orientation="h",
        marker_color=color,
        customdata=list(zip(cov["pop"].to_list(), (cov["n"] / cov["pop"] * 100_000).to_list())),
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Unidades: %{x:,}<br>"
            "Población 2025: %{customdata[0]:,.0f}<br>"
            "Tasa: %{customdata[1]:.2f} por 100k<extra></extra>"
        ),
    ))
    fig.update_layout(
        title=dict(
            text=(
                f"<b>Unidades '{label}' por estado</b>"
                f"<br><sup style='color:#94A3B8'>Conteo total · población 2025 (CONAPO)</sup>"
            ),
            font=dict(color="#F8FAFC", size=13),
        ),
        yaxis=dict(autorange="reversed", gridcolor="rgba(0,0,0,0)"),
        height=max(400, 32 * 22 + 80),
        **{k: v for k, v in CHART_LAYOUT.items() if k not in ("xaxis", "yaxis", "margin")},
        margin=dict(t=60, b=40, l=10, r=10),
    )
    return fig


def fig_coverage_scatter(tipo: str) -> go.Figure:
    cov = _coverage_data(tipo)
    label = "Todos los tipos" if tipo == "all" else tipo

    ns   = cov["n"].to_list()
    pops = cov["pop"].to_list()
    states = cov["estado_geo"].to_list()

    total_n   = sum(ns)
    total_pop = sum(pops)
    rate = total_n / total_pop if total_pop > 0 else 0
    rate_100k = rate * 100_000

    # Reference line across the full population range
    max_pop = max(pops) * 1.08
    ref_x = [0, max_pop]
    ref_y = [0, max_pop * rate]

    # Color each state above/below the national rate
    point_colors = ["#2E86AB" if (p > 0 and n / p > rate) else "#E84855"
                    for n, p in zip(ns, pops)]

    rates_100k = [n / p * 100_000 if p > 0 else 0 for n, p in zip(ns, pops)]

    fig = go.Figure()

    # Reference line
    fig.add_trace(go.Scatter(
        x=ref_x, y=ref_y,
        mode="lines",
        line=dict(color="#64748B", dash="dot", width=1.5),
        showlegend=False,
        hoverinfo="skip",
    ))

    # State points
    fig.add_trace(go.Scatter(
        x=pops, y=ns,
        mode="markers+text",
        text=states,
        textposition="top center",
        textfont=dict(size=8, color="#94A3B8"),
        marker=dict(color=point_colors, size=9, opacity=0.9),
        customdata=list(zip(ns, pops, rates_100k)),
        hovertemplate=(
            "<b>%{text}</b><br>"
            "Unidades: %{customdata[0]:,}<br>"
            "Población: %{customdata[1]:,.0f}<br>"
            "Tasa: %{customdata[2]:.2f} por 100k<extra></extra>"
        ),
        showlegend=False,
    ))

    # Legend proxies
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(color="#2E86AB", size=9),
        name="Sobre el promedio",
    ))
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(color="#E84855", size=9),
        name="Bajo el promedio",
    ))

    # Annotate the reference line
    fig.add_annotation(
        x=max_pop * 0.97, y=max_pop * rate,
        text=f"Promedio: {rate_100k:.1f} por 100k",
        font=dict(color="#94A3B8", size=10),
        showarrow=False, xanchor="right",
    )

    fig.update_layout(
        title=dict(
            text=(
                f"<b>Cobertura de '{label}': población vs unidades por estado</b>"
                f"<br><sup style='color:#94A3B8'>Sobre la línea = cobertura superior al promedio nacional</sup>"
            ),
            font=dict(color="#F8FAFC", size=13),
        ),
        xaxis=dict(title="Población 2025", gridcolor="#334155",
                   tickformat=",", showgrid=True),
        yaxis=dict(title="Unidades", gridcolor="#334155", showgrid=True),
        legend=dict(orientation="h", y=-0.14, x=0),
        height=520,
        margin=dict(t=70, b=80, l=60, r=20),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#CBD5E1",
    )
    return fig


# ── App layout ────────────────────────────────────────────────────────────────

app = Dash(__name__, external_stylesheets=[dbc.themes.SLATE])
app.title = "Directorio IMSS"

_estados = sorted(df.filter(pl.col("estado_geo").is_not_null())["estado_geo"].unique().to_list())
_estado_opts = [{"label": "Nacional (todos los estados)", "value": "all"}] + [
    {"label": e, "value": e} for e in _estados
]

app.layout = html.Div(
    style={"backgroundColor": "#0F172A", "minHeight": "100vh", "padding": "24px"},
    children=[
        html.H2("Directorio de Unidades IMSS",
                style={"color": "#F8FAFC", "fontWeight": "700", "margin": "0 0 4px"}),
        html.P(
            "Red de atención médica del Instituto Mexicano del Seguro Social · "
            "Fuente: IMSS, 2024",
            style={"color": "#94A3B8", "fontSize": "13px", "marginBottom": "20px"},
        ),
        dbc.Row(style={"marginBottom": "20px"}, children=[
            dbc.Col([
                html.Label("Estado", style={"color": "#94A3B8", "fontSize": "12px"}),
                dcc.Dropdown(
                    id="estado-filter",
                    options=_estado_opts,
                    value="all",
                    clearable=False,
                    style={"backgroundColor": "#1E293B", "color": "#F8FAFC",
                           "border": "1px solid #334155", "borderRadius": "6px"},
                ),
            ], md=4),
        ]),
        dbc.Row(id="kpi-row", className="g-3", style={"marginBottom": "24px"}),
        dcc.Tabs(style={"marginBottom": "0"}, children=[
            dcc.Tab(label="Distribución geográfica", style=TAB_STYLE, selected_style=TAB_SEL,
                    children=[
                        dbc.Row(className="g-3 mt-1", children=[
                            dbc.Col(dcc.Graph(id="fig-choropleth"), md=7),
                            dbc.Col(dcc.Graph(id="fig-state-ranking"), md=5),
                        ]),
                    ]),
            dcc.Tab(label="Tipos de unidad", style=TAB_STYLE, selected_style=TAB_SEL,
                    children=[
                        dbc.Row(className="g-3 mt-1", children=[
                            dbc.Col(dcc.Graph(id="fig-type-dist"), md=5),
                            dbc.Col(dcc.Graph(id="fig-type-by-state"), md=7),
                        ]),
                    ]),
            dcc.Tab(label="Cobertura por tipo", style=TAB_STYLE, selected_style=TAB_SEL,
                    children=[
                        dbc.Row(style={"marginTop": "12px", "marginBottom": "12px"}, children=[
                            dbc.Col([
                                html.Label("Tipo de unidad", style={"color": "#94A3B8", "fontSize": "12px"}),
                                dcc.Dropdown(
                                    id="tipo-coverage-filter",
                                    options=_tipo_coverage_opts,
                                    value="Hospital",
                                    clearable=False,
                                    style={"backgroundColor": "#1E293B", "color": "#F8FAFC",
                                           "border": "1px solid #334155", "borderRadius": "6px"},
                                ),
                            ], md=4),
                        ]),
                        dbc.Row(className="g-3", children=[
                            dbc.Col(dcc.Graph(id="fig-coverage-bar"), md=5),
                            dbc.Col(dcc.Graph(id="fig-coverage-scatter"), md=7),
                        ]),
                    ]),
        ]),
        html.P(
            "Nota: el directorio fue extraído del PDF oficial de IMSS. "
            "~10% de registros tienen estado no recuperable. "
            "El mapa de % Bienestar muestra la proporción de unidades del programa Bienestar sobre el total del estado.",
            style={"color": "#475569", "fontSize": "11px", "marginTop": "16px"},
        ),
    ],
)

# ── Callback ──────────────────────────────────────────────────────────────────

@app.callback(
    Output("kpi-row", "children"),
    Output("fig-choropleth", "figure"),
    Output("fig-state-ranking", "figure"),
    Output("fig-type-dist", "figure"),
    Output("fig-type-by-state", "figure"),
    Input("estado-filter", "value"),
)
def update_all(estado):
    estado = estado or "all"
    d = df if estado == "all" else df.filter(pl.col("estado_geo") == estado)
    focus = None if estado == "all" else estado
    return (
        compute_kpis(d, estado),
        fig_choropleth(df),
        fig_state_ranking(df, focus_estado=focus),
        fig_type_dist(d),
        fig_type_by_state(df),
    )


@app.callback(
    Output("fig-coverage-bar", "figure"),
    Output("fig-coverage-scatter", "figure"),
    Input("tipo-coverage-filter", "value"),
)
def update_coverage(tipo):
    tipo = tipo or "Hospital"
    return fig_coverage_bar(tipo), fig_coverage_scatter(tipo)


if __name__ == "__main__":
    app.run(debug=True, port=8061)
