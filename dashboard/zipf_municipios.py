import numpy as np
import plotly.graph_objects as go
from dash import Dash, dcc, html, callback, Input, Output, dash_table
import dash_bootstrap_components as dbc
import polars as pl

# ── Datos ─────────────────────────────────────────────────────────────────────
d_all = pl.read_parquet("dashboard_data/zipf_municipios.parquet")
d_estados_all = pl.read_parquet("dashboard_data/zipf_estados.parquet")

YEAR_MIN, YEAR_MAX, YEAR_DEFAULT = 2017, 2025, 2025
TOP_N_MUNICIPIOS = 10
TOP_N_ESTADOS = 5
SLOPE_RANK_MIN_MUNICIPIOS = 20  # excluir outliers de cabecera al ajustar la pendiente empírica
SLOPE_RANK_MIN_ESTADOS = 3
NACIONAL = "Nacional"

n_municipios = d_all.filter(pl.col("AÑO") == YEAR_DEFAULT).height
n_estados = d_estados_all.filter(pl.col("AÑO") == YEAR_DEFAULT).height
ESTADO_OPTIONS = [NACIONAL] + sorted(d_all["NOM_ENT"].unique().to_list())

FOCUS = "#2E86AB"
CONTEXT = "#475569"

CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font_color="#CBD5E1",
)
AXIS_STYLE = dict(gridcolor="#334155")

CARD_STYLE = {
    "background": "#1E293B", "border": "1px solid #334155",
    "borderRadius": "8px", "padding": "16px", "textAlign": "center",
}


def year_slice(year: int, estado: str = NACIONAL) -> pl.DataFrame:
    d = d_all.filter(pl.col("AÑO") == year)
    if estado == NACIONAL:
        return d.sort("rank")

    d = d.filter(pl.col("NOM_ENT") == estado).drop(["rank", "pob_ideal_zipf"])
    d = d.sort("POB_TOTAL", descending=True).with_row_index("rank", offset=1)
    pob_rank1 = d.item(0, "POB_TOTAL")
    return d.with_columns((pob_rank1 / pl.col("rank")).alias("pob_ideal_zipf"))


def year_slice_estados(year: int) -> pl.DataFrame:
    return d_estados_all.filter(pl.col("AÑO") == year).sort("rank")


def estado_slopes(year: int) -> pl.DataFrame:
    rows = []
    for estado in ESTADO_OPTIONS[1:]:  # excluye "Nacional"
        d = year_slice(year, estado)
        slope = zipf_slope(d, SLOPE_RANK_MIN_MUNICIPIOS)
        rows.append({"Estado": estado, "Municipios": d.height, "Pendiente": round(slope, 2)})
    return pl.DataFrame(rows).sort("Pendiente", descending=True)


def zipf_slope(d: pl.DataFrame, rank_min: int = 0) -> float:
    rank_min = min(rank_min, max(0, d.height - 5))
    tail = d.filter(pl.col("rank") > rank_min)
    logr = np.log(tail["rank"].to_numpy())
    logp = np.log(tail["POB_TOTAL"].to_numpy())
    slope, _ = np.polyfit(logr, logp, 1)
    return slope


# ── Figura (compartida por ambas pestañas) ───────────────────────────────────

def _fig_zipf(
    d: pl.DataFrame, top_n: int, customdata_cols: list[str],
    trace_name: str, annotation_text: str, title_text: str,
) -> go.Figure:
    top = d.head(top_n)
    rest = d.tail(-top_n)

    header = ", ".join(
        f"<b>%{{customdata[0]}}</b>" if i == 0 else f"%{{customdata[{i}]}}"
        for i in range(len(customdata_cols))
    )
    hovertemplate = f"{header}<br>Rango: %{{x}}<br>Población: %{{y:,}}<extra></extra>"

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=rest["rank"], y=rest["POB_TOTAL"],
        mode="markers",
        marker=dict(size=5, color=CONTEXT, opacity=0.6),
        name=trace_name,
        customdata=rest.select(customdata_cols).to_numpy(),
        hovertemplate=hovertemplate,
    ))

    fig.add_trace(go.Scatter(
        x=top["rank"], y=top["POB_TOTAL"],
        mode="markers",
        marker=dict(size=9, color=FOCUS, line=dict(width=1, color="#F8FAFC")),
        name=f"Top {top_n}",
        customdata=top.select(customdata_cols).to_numpy(),
        hovertemplate=hovertemplate,
    ))

    fig.add_trace(go.Scatter(
        x=d["rank"], y=d["pob_ideal_zipf"],
        mode="lines",
        line=dict(color="#F8FAFC", width=1.5, dash="dash"),
        name="Zipf ideal (pendiente -1)",
        hoverinfo="skip",
    ))

    if annotation_text:
        fig.add_annotation(
            x=0.02, y=0.06, xref="paper", yref="paper",
            xanchor="left", yanchor="bottom",
            align="left",
            text=annotation_text,
            showarrow=False,
            font=dict(size=11, color="#94A3B8"),
            bgcolor="rgba(15,23,42,0.75)",
            borderpad=6,
        )

    fig.update_layout(
        **CHART_LAYOUT,
        xaxis=dict(**AXIS_STYLE, type="log", title="Rango (escala log)"),
        yaxis=dict(**AXIS_STYLE, type="log", title="Población (escala log)"),
        height=560,
        legend=dict(orientation="h", y=-0.15, x=0, font=dict(size=11)),
        margin=dict(t=70, b=90, l=70, r=20),
        title=dict(text=title_text, font=dict(size=15)),
    )
    return fig


def fig_zipf_municipios(d: pl.DataFrame, year: int, slope: float, estado: str = NACIONAL) -> go.Figure:
    if estado == NACIONAL:
        title_text = (
            "Los municipios de México no siguen la Ley de Zipf: "
            f"la población cae más rápido de lo ideal (pendiente ≈ {slope:.2f}, no -1)"
            f"<br><sup>Rango vs. población, escala log-log · {d.height:,} municipios, {year} · CONAPO</sup>"
        )
        annotation_text = (
            "Nota: la Ciudad de México se reporta como 16 alcaldías separadas,<br>"
            "no como una sola unidad — por eso no domina el rango 1."
        )
    else:
        title_text = (
            f"Municipios de {estado}: pendiente empírica ≈ {slope:.2f} (ideal: -1.00)"
            f"<br><sup>Rango vs. población, escala log-log · {d.height:,} municipios, {year} · CONAPO</sup>"
        )
        annotation_text = ""

    top_n = min(TOP_N_MUNICIPIOS, d.height)
    return _fig_zipf(
        d, top_n, ["NOM_MUN", "NOM_ENT"],
        "Municipios", annotation_text, title_text,
    )


def fig_zipf_estados(d: pl.DataFrame, year: int, slope: float) -> go.Figure:
    title_text = (
        f"A escala estatal, México sí sigue la Ley de Zipf (pendiente ≈ {slope:.2f}, cerca de -1)"
        f"<br><sup>Rango vs. población, escala log-log · {d.height} estados, {year} · CONAPO</sup>"
    )
    annotation_text = (
        "A diferencia de los municipios, aquí la Ciudad de México sí aparece<br>"
        "como una sola entidad — por eso la curva se acerca más al ideal."
    )
    return _fig_zipf(
        d, TOP_N_ESTADOS, ["NOM_ENT"],
        "Estados", annotation_text, title_text,
    )


# ── Layout ────────────────────────────────────────────────────────────────────

def kpi(value_id: str, value: str, label: str):
    return dbc.Col(html.Div([
        html.Div(value, id=value_id, style={"fontSize": "2rem", "fontWeight": "700", "color": "#F8FAFC"}),
        html.Div(label, style={"fontSize": "0.85rem", "color": "#64748B", "marginTop": "4px"}),
    ], style=CARD_STYLE), md=4, className="mb-3")


d_default = year_slice(YEAR_DEFAULT)
slope_default = zipf_slope(d_default, SLOPE_RANK_MIN_MUNICIPIOS)
top10_share_default = d_default.head(TOP_N_MUNICIPIOS)["POB_TOTAL"].sum() / d_default["POB_TOTAL"].sum() * 100

d_estados_default = year_slice_estados(YEAR_DEFAULT)
slope_estados_default = zipf_slope(d_estados_default, SLOPE_RANK_MIN_ESTADOS)
top5_estados_share_default = (
    d_estados_default.head(TOP_N_ESTADOS)["POB_TOTAL"].sum() / d_estados_default["POB_TOTAL"].sum() * 100
)

pendientes_default = estado_slopes(YEAR_DEFAULT)

TAB_STYLE = {"backgroundColor": "#0F172A", "color": "#94A3B8", "borderTop": "none"}
TAB_SEL   = {"backgroundColor": "#1E293B", "color": "#F8FAFC",
             "borderTop": "2px solid #2E86AB", "fontWeight": "600"}

app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    title="Zipf municipios",
)

app.layout = dbc.Container([
    html.H1(
        "Ley de Zipf: población de México",
        style={"color": "#F8FAFC", "fontWeight": "700", "marginBottom": "4px", "fontSize": "1.8rem"},
    ),
    html.P(
        "Rango vs. población · municipios y estados · CONAPO",
        style={"color": "#64748B", "marginBottom": "24px"},
    ),

    html.Div([
        html.Label("Año:", style={"color": "#94A3B8", "fontSize": "0.85rem", "marginBottom": "6px"}),
        dcc.Slider(
            id="year-slider",
            min=YEAR_MIN, max=YEAR_MAX, step=1, value=YEAR_DEFAULT,
            marks={y: str(y) for y in range(YEAR_MIN, YEAR_MAX + 1)},
        ),
    ], style={"padding": "0 8px 16px"}),

    dbc.Tabs([
        dbc.Tab(label="Municipios", tab_style=TAB_STYLE, active_tab_style=TAB_SEL, children=[
            dbc.Row([
                kpi("kpi-municipios", f"{n_municipios:,}", "municipios"),
                kpi("kpi-slope", f"{slope_default:.2f}", "pendiente real (ideal: -1.00)"),
                kpi("kpi-top10", f"{top10_share_default:.1f}%", f"población en el top {TOP_N_MUNICIPIOS} de municipios"),
            ], className="mb-4 mt-3"),
            html.Div([
                html.Label("Estado:", style={"color": "#94A3B8", "fontSize": "0.85rem", "marginBottom": "6px"}),
                dcc.Dropdown(
                    id="estado-dropdown",
                    options=ESTADO_OPTIONS,
                    value=NACIONAL,
                    clearable=False,
                    style={"backgroundColor": "#1E293B", "color": "#0F172A"},
                ),
            ], style={"padding": "0 8px 16px", "maxWidth": "360px"}),
            dcc.Graph(
                id="graph-zipf",
                figure=fig_zipf_municipios(d_default, YEAR_DEFAULT, slope_default),
                config={"displayModeBar": False},
            ),
        ]),
        dbc.Tab(label="Estados", tab_style=TAB_STYLE, active_tab_style=TAB_SEL, children=[
            dbc.Row([
                kpi("kpi-estados", f"{n_estados}", "estados"),
                kpi("kpi-slope-estados", f"{slope_estados_default:.2f}", "pendiente real (ideal: -1.00)"),
                kpi("kpi-top5-estados", f"{top5_estados_share_default:.1f}%", f"población en el top {TOP_N_ESTADOS} de estados"),
            ], className="mb-4 mt-3"),
            dcc.Graph(
                id="graph-zipf-estados",
                figure=fig_zipf_estados(d_estados_default, YEAR_DEFAULT, slope_estados_default),
                config={"displayModeBar": False},
            ),
        ]),
        dbc.Tab(label="Pendientes por estado", tab_style=TAB_STYLE, active_tab_style=TAB_SEL, children=[
            html.P(
                "Pendiente empírica de la Ley de Zipf ajustada a los municipios de cada "
                "estado, de mayor a menor (más negativo = se aleja más del ideal -1.00).",
                style={"color": "#64748B", "margin": "16px 0"},
            ),
            dash_table.DataTable(
                id="table-pendientes",
                data=pendientes_default.to_dicts(),
                columns=[{"name": c, "id": c} for c in pendientes_default.columns],
                sort_action="native",
                sort_by=[{"column_id": "Pendiente", "direction": "desc"}],
                style_table={"overflowX": "auto"},
                style_header={
                    "backgroundColor": "#1E293B", "color": "#94A3B8",
                    "fontWeight": "600", "border": "1px solid #334155",
                },
                style_cell={
                    "backgroundColor": "#0F172A", "color": "#CBD5E1",
                    "border": "1px solid #1E293B",
                    "textAlign": "center", "padding": "8px 14px",
                    "fontFamily": "monospace",
                },
                style_cell_conditional=[
                    {"if": {"column_id": "Estado"}, "textAlign": "left",
                     "fontFamily": "inherit", "fontWeight": "500"},
                ],
            ),
        ]),
    ], style={"marginTop": "8px"}),

], fluid=True, style={"backgroundColor": "#0F172A", "minHeight": "100vh", "padding": "24px 32px"})


# ── Callbacks ─────────────────────────────────────────────────────────────────

@callback(
    Output("graph-zipf", "figure"),
    Output("kpi-slope", "children"),
    Output("kpi-top10", "children"),
    Input("year-slider", "value"),
    Input("estado-dropdown", "value"),
)
def update_year(year, estado):
    d = year_slice(year, estado)
    slope = zipf_slope(d, SLOPE_RANK_MIN_MUNICIPIOS)
    top_n = min(TOP_N_MUNICIPIOS, d.height)
    top_share = d.head(top_n)["POB_TOTAL"].sum() / d["POB_TOTAL"].sum() * 100
    return fig_zipf_municipios(d, year, slope, estado), f"{slope:.2f}", f"{top_share:.1f}%"


@callback(
    Output("graph-zipf-estados", "figure"),
    Output("kpi-slope-estados", "children"),
    Output("kpi-top5-estados", "children"),
    Input("year-slider", "value"),
)
def update_year_estados(year):
    d = year_slice_estados(year)
    slope = zipf_slope(d, SLOPE_RANK_MIN_ESTADOS)
    top5_share = d.head(TOP_N_ESTADOS)["POB_TOTAL"].sum() / d["POB_TOTAL"].sum() * 100
    return fig_zipf_estados(d, year, slope), f"{slope:.2f}", f"{top5_share:.1f}%"


@callback(
    Output("table-pendientes", "data"),
    Input("year-slider", "value"),
)
def update_pendientes(year):
    return estado_slopes(year).to_dicts()


if __name__ == "__main__":
    app.run(debug=True, port=8068)
