import plotly.graph_objects as go
from dash import Dash, dcc, html, callback, Input, Output
import dash_bootstrap_components as dbc

# ── Datos ─────────────────────────────────────────────────────────────────────
YEARS = [200, 600, 900, 1200, 1450, 1519]

GRUPOS = {
    "Mayas":                       [4.0,  8.0,  5.5, 4.0,  5.0,  5.5],
    "Altiplano Central / Mexicas": [3.0,  5.0,  4.0, 6.0, 10.0, 12.5],
    "Zapotecos":                   [1.0,  1.25, 1.0, 0.85, 1.0,  1.25],
    "Mixtecos":                    [0.65, 1.0,  1.25,1.2,  1.25, 1.5],
    "Totonacos":                   [0.5,  0.65, 0.6, 0.75, 1.0,  1.25],
    "Purépechas":                  [0.4,  0.65, 0.85,1.25, 2.0,  2.5],
    "Huastecos":                   [0.4,  0.55, 0.5, 0.65, 0.8,  1.0],
    "Occidente y otros":           [1.5,  2.0,  2.5, 2.5,  3.0,  3.0],
    "Total":                       [11.0, 19.0, 16.0,18.0, 24.0, 28.5],
}

CIUDADES = [
    {"ciudad": "Tenochtitlan",  "civ": "Mexica",           "año": 1519, "pop_min": 200_000, "pop_max": 400_000},
    {"ciudad": "Teotihuacan",   "civ": "Teotihuacana",     "año": 500,  "pop_min": 125_000, "pop_max": 250_000},
    {"ciudad": "Cantona",       "civ": "Epiclásica",       "año": 700,  "pop_min":  60_000, "pop_max":  90_000},
    {"ciudad": "Tikal",         "civ": "Maya",             "año": 700,  "pop_min":  60_000, "pop_max": 120_000},
    {"ciudad": "Calakmul",      "civ": "Maya",             "año": 700,  "pop_min":  50_000, "pop_max": 100_000},
    {"ciudad": "Cholula",       "civ": "Nahua/Tolteca",    "año": 1500, "pop_min":  80_000, "pop_max": 120_000},
    {"ciudad": "Tzintzuntzan",  "civ": "Purépecha",        "año": 1500, "pop_min":  25_000, "pop_max":  40_000},
    {"ciudad": "Monte Albán",   "civ": "Zapoteca",         "año": 500,  "pop_min":  30_000, "pop_max":  60_000},
    {"ciudad": "El Tajín",      "civ": "Totonaca",         "año": 900,  "pop_min":  15_000, "pop_max":  30_000},
    {"ciudad": "Palenque",      "civ": "Maya",             "año": 700,  "pop_min":  20_000, "pop_max":  50_000},
    {"ciudad": "Copán",         "civ": "Maya",             "año": 750,  "pop_min":  20_000, "pop_max":  40_000},
    {"ciudad": "Yaxchilán",     "civ": "Maya",             "año": 750,  "pop_min":  10_000, "pop_max":  25_000},
    {"ciudad": "Uxmal",         "civ": "Maya",             "año": 900,  "pop_min":  20_000, "pop_max":  40_000},
    {"ciudad": "Mayapán",       "civ": "Maya",             "año": 1400, "pop_min":  15_000, "pop_max":  25_000},
    {"ciudad": "Mitla",         "civ": "Zapoteca/Mixteca", "año": 1400, "pop_min":  10_000, "pop_max":  25_000},
    {"ciudad": "Cacaxtla",      "civ": "Epiclásica",       "año": 800,  "pop_min":  10_000, "pop_max":  20_000},
]

GRUPO_COLORS = {
    "Mayas":                       "#2E86AB",
    "Altiplano Central / Mexicas": "#E84855",
    "Zapotecos":                   "#3BB273",
    "Mixtecos":                    "#F4A261",
    "Totonacos":                   "#94A3B8",
    "Purépechas":                  "#A78BFA",
    "Huastecos":                   "#64748B",
    "Occidente y otros":           "#F59E0B",
    "Total":                       "#F8FAFC",
}

CIV_COLORS = {
    "Maya":            "#2E86AB",
    "Mexica":          "#E84855",
    "Teotihuacana":    "#F4A261",
    "Nahua/Tolteca":   "#F59E0B",
    "Zapoteca":        "#3BB273",
    "Zapoteca/Mixteca":"#10B981",
    "Purépecha":       "#A78BFA",
    "Totonaca":        "#94A3B8",
    "Epiclásica":      "#64748B",
}

CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font_color="#CBD5E1",
)
AXIS_STYLE = dict(gridcolor="#334155")

CARD_STYLE = {
    "background": "#1E293B", "border": "1px solid #334155",
    "borderRadius": "8px", "padding": "16px", "textAlign": "center",
}

TAB_STYLE = {"backgroundColor": "#0F172A", "color": "#94A3B8", "borderTop": "none"}
TAB_SEL   = {"backgroundColor": "#1E293B", "color": "#F8FAFC",
             "borderTop": "2px solid #2E86AB", "fontWeight": "600"}


# ── Figuras ───────────────────────────────────────────────────────────────────

def fig_poblaciones(ciudades_sel: list) -> go.Figure:
    fig = go.Figure()

    for grupo, vals in GRUPOS.items():
        is_total = grupo == "Total"
        fig.add_trace(go.Scatter(
            x=YEARS, y=vals,
            mode="lines+markers",
            name=grupo,
            line=dict(
                color=GRUPO_COLORS[grupo],
                width=3 if is_total else 2,
                dash="dot" if is_total else "solid",
            ),
            marker=dict(size=8 if is_total else 6),
            hovertemplate=f"<b>{grupo}</b><br>Año: %{{x}} d.C.<br>Población: %{{y:.2f}} M<extra></extra>",
        ))

    # Líneas verticales agrupadas por año de pico
    year_to_cities: dict[int, list[str]] = {}
    for c in CIUDADES:
        if c["ciudad"] in (ciudades_sel or []):
            yr = c["año"]
            year_to_cities.setdefault(yr, []).append(c["ciudad"])

    for yr, names in sorted(year_to_cities.items()):
        label = "<br>".join(names)
        fig.add_vline(
            x=yr,
            line_dash="dash",
            line_color="rgba(248,250,252,0.55)",
            line_width=1.8,
            annotation_text=label,
            annotation_position="top left",
            annotation=dict(
                font=dict(size=11, color="#F8FAFC"),
                bgcolor="rgba(15,23,42,0.75)",
                borderpad=4,
                textangle=-45,
            ),
        )

    fig.update_layout(
        **CHART_LAYOUT,
        yaxis=dict(**AXIS_STYLE, type="log", title="Millones de personas (escala log)", tickformat=".2f"),
        xaxis=dict(**AXIS_STYLE, title="Año d.C.", tickvals=YEARS, ticktext=[str(y) for y in YEARS]),
        height=540,
        legend=dict(orientation="h", y=-0.22, x=0, font=dict(size=11)),
        margin=dict(t=50, b=110, l=70, r=20),
        title=dict(text="Población por grupo cultural · escala logarítmica", font=dict(size=15)),
    )
    return fig


def fig_ciudades_bar() -> go.Figure:
    cities_sorted = sorted(CIUDADES, key=lambda c: (c["pop_min"] + c["pop_max"]) / 2)
    names    = [c["ciudad"] for c in cities_sorted]
    pop_mids = [(c["pop_min"] + c["pop_max"]) / 2 for c in cities_sorted]
    pop_errs = [(c["pop_max"] - c["pop_min"]) / 2 for c in cities_sorted]
    bar_cols = [CIV_COLORS[c["civ"]] for c in cities_sorted]
    civs     = [c["civ"]   for c in cities_sorted]
    years    = [c["año"]   for c in cities_sorted]
    p_mins   = [c["pop_min"] for c in cities_sorted]
    p_maxs   = [c["pop_max"] for c in cities_sorted]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=pop_mids, y=names, orientation="h",
        marker_color=bar_cols,
        error_x=dict(
            type="data", symmetric=True, array=pop_errs,
            color="#475569", thickness=1.5, width=6,
        ),
        customdata=list(zip(civs, years, p_mins, p_maxs)),
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Civilización: %{customdata[0]}<br>"
            "Pico: %{customdata[1]} d.C.<br>"
            "Población: %{customdata[2]:,} – %{customdata[3]:,}"
            "<extra></extra>"
        ),
        showlegend=False,
    ))

    # Proxy traces para leyenda por civilización
    seen: set = set()
    for c in cities_sorted:
        civ = c["civ"]
        if civ not in seen:
            fig.add_trace(go.Bar(
                x=[None], y=[None], orientation="h",
                marker_color=CIV_COLORS[civ], name=civ,
            ))
            seen.add(civ)

    fig.update_layout(
        **CHART_LAYOUT,
        xaxis=dict(**AXIS_STYLE, title="Población estimada en su apogeo"),
        yaxis=dict(**AXIS_STYLE),
        height=max(300, len(CIUDADES) * 28 + 90),
        margin=dict(t=50, b=80, l=10, r=20),
        legend=dict(orientation="h", y=-0.18, x=0, font=dict(size=11)),
        title=dict(text="Ciudades mesoamericanas · población en su apogeo", font=dict(size=15)),
        barmode="overlay",
    )
    return fig


# ── Layout ────────────────────────────────────────────────────────────────────

def kpi(value: str, label: str):
    return dbc.Col(html.Div([
        html.Div(value, style={"fontSize": "2rem", "fontWeight": "700", "color": "#F8FAFC"}),
        html.Div(label, style={"fontSize": "0.85rem", "color": "#64748B", "marginTop": "4px"}),
    ], style=CARD_STYLE), md=4, className="mb-3")


CIUDAD_OPTIONS = [{"label": c["ciudad"], "value": c["ciudad"]} for c in CIUDADES]

app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    title="Mesoamérica",
)

app.layout = dbc.Container([
    html.H1(
        "Civilizaciones Mesoamericanas",
        style={"color": "#F8FAFC", "fontWeight": "700", "marginBottom": "4px", "fontSize": "1.8rem"},
    ),
    html.P(
        "Demografía y ciudades prehispánicas · 200–1519 d.C.",
        style={"color": "#64748B", "marginBottom": "24px"},
    ),

    # KPIs
    dbc.Row([
        kpi("16", "ciudades principales"),
        kpi("8",  "grupos culturales"),
        kpi("1,319", "años de historia"),
    ], className="mb-4"),

    # Tabs
    dbc.Tabs([
        dbc.Tab(label="Tendencias demográficas", tab_style=TAB_STYLE, active_tab_style=TAB_SEL, children=[
            html.Div([
                html.Label(
                    "Ciudades · líneas verticales de pico poblacional:",
                    style={"color": "#94A3B8", "fontSize": "0.85rem", "marginBottom": "6px"},
                ),
                dcc.Dropdown(
                    id="cities-dd",
                    options=CIUDAD_OPTIONS,
                    value=[c["ciudad"] for c in CIUDADES],
                    multi=True,
                    style={"backgroundColor": "#1E293B", "color": "#F8FAFC"},
                ),
            ], style={"padding": "16px 0 8px"}),
            dcc.Graph(id="graph-pop"),
        ]),
        dbc.Tab(label="Ciudades", tab_style=TAB_STYLE, active_tab_style=TAB_SEL, children=[
            dcc.Graph(id="graph-cities", figure=fig_ciudades_bar()),
        ]),
    ], style={"marginTop": "8px"}),

], fluid=True, style={"backgroundColor": "#0F172A", "minHeight": "100vh", "padding": "24px 32px"})


# ── Callbacks ─────────────────────────────────────────────────────────────────

@callback(Output("graph-pop", "figure"), Input("cities-dd", "value"))
def update_pop(cities_sel):
    return fig_poblaciones(cities_sel or [])


if __name__ == "__main__":
    app.run(debug=True, port=8060)
