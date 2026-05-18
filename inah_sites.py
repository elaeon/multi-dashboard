import json
import polars as pl
import plotly.express as px
import plotly.graph_objects as go
import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc

# ── Data ──────────────────────────────────────────────────────────────────────
df = (
    pl.read_csv("data/visita_museos.csv")
    .with_columns(
        pl.col("PERIODO").str.to_date("%Y-%m-%d"),
        pl.col("PERIODO").str.to_date("%Y-%m-%d").dt.year().alias("year"),
        pl.col("PERIODO").str.to_date("%Y-%m-%d").dt.month().alias("month"),
    )
    .filter(pl.col("year") < 2026)  # exclude partial year
)

YEAR_MIN = df["year"].min()
YEAR_MAX = df["year"].max()
TOTAL_SITES = df["CENTRO DE TRABAJO"].n_unique()
ALL_CENTROS = sorted(df["CENTRO DE TRABAJO"].unique().to_list())

with open("data/mexico_states.geojson") as _f:
    MEXICO_GEO = json.load(_f)
for _feat in MEXICO_GEO["features"]:
    if _feat["properties"]["name"] == "México":
        _feat["properties"]["name"] = "Estado de México"

SITE_COLORS = {"Zona Arqueológica": "#2E86AB", "Museo": "#E84855"}
NAT_COLORS = {"Nacional": "#3BB273", "Extranjeros": "#F4A261"}
MONTH_NAMES = ["Ene", "Feb", "Mar", "Abr", "May", "Jun",
               "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]
VISITOR_SHORT = {
    "Boleto pagado": "Boleto pagado",
    "Entrada dominical": "Dominical (gratis)",
    "Entrada sin costo": "Sin costo",
    "Estudiantes nivel básico": "Est. básico",
    "Estudiantes nivel superior": "Est. superior",
    "Personas de la tercera edad, pensionadas y jubiladas": "3ª edad",
    "Personas docentes": "Docentes",
    "Personas con discapacidad": "Discapacidad",
    "Personas trabajadoras del INAH": "INAH",
    "Exposiciones temporales con costo adicional": "Exp. temporales",
}

CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font_color="#CBD5E1",
    xaxis=dict(gridcolor="#334155"),
    yaxis=dict(gridcolor="#334155"),
)


# ── Figure factories (all accept a filtered Polars DataFrame) ─────────────────

def fig_annual_trend(d: pl.DataFrame) -> go.Figure:
    yearly = (
        d.group_by("year", "NACIONALIDAD")
        .agg(pl.col("NÚMERO DE VISITAS").sum())
        .sort("year", "NACIONALIDAD")
    )
    yr_min, yr_max = d["year"].min(), d["year"].max()
    fig = px.area(
        yearly, x="year", y="NÚMERO DE VISITAS", color="NACIONALIDAD",
        color_discrete_map=NAT_COLORS,
        labels={"NÚMERO DE VISITAS": "Visitas", "year": "Año", "NACIONALIDAD": "Nacionalidad"},
        title=f"Tendencia anual de visitas {yr_min}–{yr_max}",
    )
    if yr_min <= 2020 <= yr_max:
        fig.add_vline(x=2020, line_dash="dash", line_color="#FF6B6B", opacity=0.8, line_width=2)
        fig.add_annotation(
            x=2021, y=yearly["NÚMERO DE VISITAS"].max() * 0.85,
            text="COVID-19<br>−73% en 2020",
            showarrow=False, font=dict(color="#FF6B6B", size=11),
            bgcolor="rgba(0,0,0,0.5)", bordercolor="#FF6B6B", borderwidth=1,
        )
    fig.update_layout(height=420, legend=dict(orientation="h", y=-0.15), **CHART_LAYOUT)
    return fig


def fig_yoy_change(d: pl.DataFrame) -> go.Figure:
    yearly = (
        d.group_by("year")
        .agg(pl.col("NÚMERO DE VISITAS").sum().alias("total"))
        .sort("year")
        .with_columns(
            (pl.col("total") / pl.col("total").shift(1) * 100 - 100).round(1).alias("yoy")
        )
        .drop_nulls("yoy")
    )
    yr_min, yr_max = d["year"].min(), d["year"].max()
    yoy_vals = yearly["yoy"].to_list()
    colors = ["#E84855" if v < 0 else "#3BB273" for v in yoy_vals]
    fig = go.Figure(go.Bar(
        x=yearly["year"].to_list(), y=yoy_vals,
        marker_color=colors,
        hovertemplate="Año: %{x}<br>Cambio: %{y:.1f}%<extra></extra>",
    ))
    fig.add_hline(y=0, line_color="#94A3B8", line_width=1)
    fig.update_layout(
        title=f"Variación anual de visitas (%) — {yr_min}–{yr_max}",
        height=320,
        xaxis=dict(gridcolor="#334155", title="Año"),
        yaxis=dict(gridcolor="#334155", title="Cambio %"),
        **{k: v for k, v in CHART_LAYOUT.items() if k not in ("xaxis", "yaxis")},
    )
    return fig


def fig_site_donut(d: pl.DataFrame) -> go.Figure:
    totals = d.group_by("TIPO DE SITIO").agg(pl.col("NÚMERO DE VISITAS").sum())
    fig = px.pie(
        totals, names="TIPO DE SITIO", values="NÚMERO DE VISITAS",
        color="TIPO DE SITIO", color_discrete_map=SITE_COLORS,
        hole=0.5, title="Visitas por tipo de sitio",
    )
    fig.update_traces(textinfo="percent+label", textfont_size=13)
    fig.update_layout(height=370, showlegend=False, paper_bgcolor="rgba(0,0,0,0)", font_color="#CBD5E1")
    return fig


def fig_visitor_type_donut(d: pl.DataFrame) -> go.Figure:
    vtype = (
        d.group_by("TIPO DE VISITANTES")
        .agg(pl.col("NÚMERO DE VISITAS").sum())
        .sort("NÚMERO DE VISITAS", descending=True)
        .with_columns(pl.col("TIPO DE VISITANTES").replace(VISITOR_SHORT).alias("label"))
    )
    fig = px.pie(
        vtype, names="label", values="NÚMERO DE VISITAS",
        hole=0.5, title="Tipo de visitante",
    )
    fig.update_traces(textinfo="percent+label", textfont_size=11)
    fig.update_layout(height=370, showlegend=False, paper_bgcolor="rgba(0,0,0,0)", font_color="#CBD5E1")
    return fig


def fig_top_states(d: pl.DataFrame) -> go.Figure:
    totals = (
        d.group_by("ESTADO")
        .agg(pl.col("NÚMERO DE VISITAS").sum())
        .sort("NÚMERO DE VISITAS", descending=True)
        .head(15)
        .sort("NÚMERO DE VISITAS")
        .with_columns(
            (pl.col("NÚMERO DE VISITAS") / 1e6).round(1).cast(pl.String).alias("label")
        )
    )
    fig = px.bar(
        totals, x="NÚMERO DE VISITAS", y="ESTADO", orientation="h",
        title="Top 15 estados por total de visitas",
        labels={"NÚMERO DE VISITAS": "Total de visitas", "ESTADO": ""},
        color="NÚMERO DE VISITAS",
        color_continuous_scale=[[0, "#1E3A5F"], [1, "#2E86AB"]],
        text="label",
    )
    fig.update_traces(textposition="outside", textfont_color="#CBD5E1")
    fig.update_layout(
        height=500, coloraxis_showscale=False,
        xaxis=dict(gridcolor="#334155"),
        yaxis=dict(gridcolor="rgba(0,0,0,0)"),
        **{k: v for k, v in CHART_LAYOUT.items() if k not in ("xaxis", "yaxis")},
    )
    return fig


def fig_foreign_ratio(d: pl.DataFrame) -> go.Figure:
    state_nat = d.group_by("ESTADO", "NACIONALIDAD").agg(pl.col("NÚMERO DE VISITAS").sum())
    state_total = d.group_by("ESTADO").agg(pl.col("NÚMERO DE VISITAS").sum().alias("total"))
    state_foreign = (
        state_nat.filter(pl.col("NACIONALIDAD") == "Extranjeros")
        .rename({"NÚMERO DE VISITAS": "foreign"})
        .select("ESTADO", "foreign")
    )
    result = (
        state_total.join(state_foreign, on="ESTADO", how="left")
        .with_columns(pl.col("foreign").fill_null(0))
        .with_columns((pl.col("foreign") / pl.col("total") * 100).round(1).alias("pct_foreign"))
        .sort("pct_foreign")
    )
    pct_vals = result["pct_foreign"].to_list()
    colors = ["#F4A261" if v > 20 else "#2E86AB" for v in pct_vals]
    fig = go.Figure(go.Bar(
        x=pct_vals, y=result["ESTADO"].to_list(), orientation="h",
        marker_color=colors,
        text=[f"{v:.1f}%" for v in pct_vals],
        textposition="outside",
        hovertemplate="%{y}<br>Extranjeros: %{x:.1f}%<extra></extra>",
    ))
    fig.add_vline(x=20, line_dash="dot", line_color="#94A3B8", line_width=1)
    fig.update_layout(
        title="% de visitantes extranjeros por estado",
        height=700,
        xaxis=dict(gridcolor="#334155", title="% Extranjeros"),
        yaxis=dict(gridcolor="rgba(0,0,0,0)", title=""),
        **{k: v for k, v in CHART_LAYOUT.items() if k not in ("xaxis", "yaxis")},
    )
    return fig


def fig_top_sites(d: pl.DataFrame) -> go.Figure:
    # Get top 20 sites by total, keeping site type and a truncated label
    top20 = (
        d.group_by("CENTRO DE TRABAJO", "TIPO DE SITIO")
        .agg(pl.col("NÚMERO DE VISITAS").sum().alias("total"))
        .sort("total", descending=True)
        .head(20)
        .with_columns(pl.col("CENTRO DE TRABAJO").str.slice(0, 42).alias("nombre"))
    )

    # Nationality breakdown for those 20 sites
    breakdown = (
        d.join(top20.select("CENTRO DE TRABAJO"), on="CENTRO DE TRABAJO")
        .group_by("CENTRO DE TRABAJO", "NACIONALIDAD")
        .agg(pl.col("NÚMERO DE VISITAS").sum())
        .join(top20.select("CENTRO DE TRABAJO", "nombre", "total", "TIPO DE SITIO"), on="CENTRO DE TRABAJO")
        .with_columns(
            (pl.col("NÚMERO DE VISITAS") / pl.col("total") * 100).round(1).alias("pct")
        )
        .sort("total")  # ascending so lowest-total site is at bottom of horizontal bar
    )

    yr_min, yr_max = d["year"].min(), d["year"].max()
    fig = px.bar(
        breakdown, x="NÚMERO DE VISITAS", y="nombre",
        color="NACIONALIDAD", color_discrete_map=NAT_COLORS,
        orientation="h", barmode="stack",
        title=f"Top 20 sitios por total de visitas ({yr_min}–{yr_max})",
        labels={"NÚMERO DE VISITAS": "Visitas", "nombre": "", "NACIONALIDAD": ""},
        custom_data=["pct", "TIPO DE SITIO"],
    )
    fig.update_traces(
        hovertemplate="<b>%{y}</b><br>%{data.name}: %{x:,.0f} (%{customdata[0]:.1f}%)"
                      "<br>Tipo: %{customdata[1]}<extra></extra>"
    )
    fig.update_layout(
        height=620, legend=dict(orientation="h", y=-0.08, title=""),
        xaxis=dict(gridcolor="#334155"),
        yaxis=dict(gridcolor="rgba(0,0,0,0)"),
        **{k: v for k, v in CHART_LAYOUT.items() if k not in ("xaxis", "yaxis")},
    )
    return fig


def fig_foreign_by_site_type(d: pl.DataFrame) -> go.Figure:
    cross = d.group_by("TIPO DE SITIO", "NACIONALIDAD").agg(pl.col("NÚMERO DE VISITAS").sum())
    total_by_type = d.group_by("TIPO DE SITIO").agg(pl.col("NÚMERO DE VISITAS").sum().alias("total"))
    cross = cross.join(total_by_type, on="TIPO DE SITIO").with_columns(
        (pl.col("NÚMERO DE VISITAS") / pl.col("total") * 100).round(1).alias("pct")
    )
    fig = px.bar(
        cross, x="TIPO DE SITIO", y="pct",
        color="NACIONALIDAD", color_discrete_map=NAT_COLORS,
        barmode="stack",
        title="Composición Nacional vs Extranjero por tipo de sitio",
        labels={"pct": "% de visitas", "TIPO DE SITIO": ""},
        text="pct",
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="inside")
    fig.update_layout(
        height=380, legend=dict(orientation="h", y=-0.2, title=""),
        yaxis=dict(gridcolor="#334155", title="% de visitas"),
        **{k: v for k, v in CHART_LAYOUT.items() if k != "yaxis"},
    )
    return fig


def fig_visitor_trend(d: pl.DataFrame) -> go.Figure:
    main_types = [
        "Boleto pagado", "Entrada dominical", "Entrada sin costo",
        "Estudiantes nivel básico", "Estudiantes nivel superior",
    ]
    trend = (
        d.filter(pl.col("TIPO DE VISITANTES").is_in(main_types))
        .group_by("year", "TIPO DE VISITANTES")
        .agg(pl.col("NÚMERO DE VISITAS").sum())
        .sort("year")
        .with_columns(pl.col("TIPO DE VISITANTES").replace(VISITOR_SHORT).alias("Tipo"))
    )
    yr_min, yr_max = d["year"].min(), d["year"].max()
    fig = px.area(
        trend, x="year", y="NÚMERO DE VISITAS", color="Tipo",
        title="Evolución por tipo de visitante (principales categorías)",
        labels={"NÚMERO DE VISITAS": "Visitas", "year": "Año"},
    )
    if yr_min <= 2020 <= yr_max:
        fig.add_vline(x=2020, line_dash="dash", line_color="#FF6B6B", opacity=0.6)
    fig.update_layout(
        height=420, legend=dict(orientation="h", y=-0.2, title=""), **CHART_LAYOUT
    )
    return fig


def fig_paid_free_ratio(d: pl.DataFrame) -> go.Figure:
    paid_cats = ["Boleto pagado", "Exposiciones temporales con costo adicional"]
    yearly_total = d.group_by("year").agg(pl.col("NÚMERO DE VISITAS").sum().alias("total"))
    paid = (
        d.filter(pl.col("TIPO DE VISITANTES").is_in(paid_cats))
        .group_by("year").agg(pl.col("NÚMERO DE VISITAS").sum().alias("paid"))
        .join(yearly_total, on="year")
        .with_columns((pl.col("paid") / pl.col("total") * 100).round(1).alias("pct_paid"))
        .sort("year")
    )
    fig = go.Figure(go.Scatter(
        x=paid["year"].to_list(), y=paid["pct_paid"].to_list(),
        mode="lines+markers",
        line=dict(color="#F4A261", width=2),
        fill="tozeroy", fillcolor="rgba(244,162,97,0.15)",
        hovertemplate="Año: %{x}<br>Pagaron: %{y:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        title="% de visitantes que pagaron boleto",
        height=360,
        xaxis=dict(gridcolor="#334155", title="Año"),
        yaxis=dict(gridcolor="#334155", title="% pagó boleto", range=[0, 80]),
        **{k: v for k, v in CHART_LAYOUT.items() if k not in ("xaxis", "yaxis")},
    )
    return fig


def fig_foreign_trend(d: pl.DataFrame) -> go.Figure:
    yearly_total = d.group_by("year").agg(pl.col("NÚMERO DE VISITAS").sum().alias("total"))
    foreign_pct = (
        d.filter(pl.col("NACIONALIDAD") == "Extranjeros")
        .group_by("year").agg(pl.col("NÚMERO DE VISITAS").sum().alias("foreign"))
        .join(yearly_total, on="year")
        .with_columns((pl.col("foreign") / pl.col("total") * 100).round(1).alias("pct"))
        .sort("year")
    )
    yr_min = d["year"].min()
    fig = go.Figure(go.Scatter(
        x=foreign_pct["year"].to_list(), y=foreign_pct["pct"].to_list(),
        mode="lines+markers",
        line=dict(color="#F4A261", width=2),
        fill="tozeroy", fillcolor="rgba(244,162,97,0.15)",
        hovertemplate="Año: %{x}<br>Extranjeros: %{y:.1f}%<extra></extra>",
    ))
    if yr_min <= 2020 <= d["year"].max():
        fig.add_vline(x=2020, line_dash="dash", line_color="#FF6B6B", opacity=0.6)
    fig.update_layout(
        title="% de visitantes extranjeros por año",
        height=360,
        xaxis=dict(gridcolor="#334155", title="Año"),
        yaxis=dict(gridcolor="#334155", title="% extranjeros"),
        **{k: v for k, v in CHART_LAYOUT.items() if k not in ("xaxis", "yaxis")},
    )
    return fig


def fig_seasonality_heatmap(d: pl.DataFrame) -> go.Figure:
    heat = d.group_by("year", "month").agg(pl.col("NÚMERO DE VISITAS").sum())
    wide = heat.pivot(values="NÚMERO DE VISITAS", index="year", on="month").sort("year")
    col_order = [str(m) for m in range(1, 13) if str(m) in wide.columns]
    month_labels = [MONTH_NAMES[int(c) - 1] for c in col_order]
    z = wide.select(col_order).to_numpy()
    yr_min, yr_max = d["year"].min(), d["year"].max()
    fig = go.Figure(go.Heatmap(
        z=z, x=month_labels, y=wide["year"].to_list(),
        colorscale="YlOrRd",
        hovertemplate="Año: %{y}<br>Mes: %{x}<br>Visitas: %{z:,.0f}<extra></extra>",
        colorbar=dict(
            title=dict(text="Visitas", font=dict(color="#CBD5E1")),
            tickfont=dict(color="#CBD5E1"),
        ),
    ))
    fig.update_layout(
        title=f"Mapa de calor: visitas mensuales {yr_min}–{yr_max}",
        height=max(350, len(wide) * 22),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#CBD5E1",
        xaxis_title="Mes", yaxis_title="Año",
    )
    return fig


def fig_monthly_pattern(d: pl.DataFrame) -> go.Figure:
    monthly = (
        d.group_by("month")
        .agg(pl.col("NÚMERO DE VISITAS").mean().alias("avg"))
        .sort("month")
        .with_columns(
            pl.col("month").map_elements(lambda m: MONTH_NAMES[m - 1], return_dtype=pl.String).alias("mes")
        )
    )
    yr_min, yr_max = d["year"].min(), d["year"].max()
    months_list = monthly["month"].to_list()
    avg_list = monthly["avg"].to_list()
    colors = ["#E84855" if m in [3, 12, 1] else "#2E86AB" for m in months_list]
    fig = go.Figure(go.Bar(
        x=monthly["mes"].to_list(), y=avg_list,
        marker_color=colors,
        hovertemplate="%{x}: %{y:,.0f} visitas promedio<extra></extra>",
    ))
    if 3 in months_list:
        mar_avg = monthly.filter(pl.col("month") == 3)["avg"][0]
        fig.add_annotation(
            x="Mar", y=mar_avg * 1.05, text="Semana Santa",
            showarrow=True, ay=-40, font=dict(color="#E84855", size=11),
        )
    fig.update_layout(
        title=f"Patrón estacional promedio ({yr_min}–{yr_max})",
        height=360,
        xaxis=dict(gridcolor="rgba(0,0,0,0)"),
        yaxis=dict(gridcolor="#334155", title="Visitas promedio"),
        **{k: v for k, v in CHART_LAYOUT.items() if k not in ("xaxis", "yaxis")},
    )
    return fig


def fig_sites_by_state(d: pl.DataFrame) -> go.Figure:
    counts = (
        d.group_by("ESTADO", "TIPO DE SITIO")
        .agg(pl.col("CENTRO DE TRABAJO").n_unique().alias("n_centros"))
        .sort("ESTADO")
    )
    # Order states by total number of centros descending
    state_order = (
        counts.group_by("ESTADO")
        .agg(pl.col("n_centros").sum().alias("total"))
        .sort("total")
        ["ESTADO"].to_list()
    )
    fig = px.bar(
        counts,
        x="n_centros", y="ESTADO",
        color="TIPO DE SITIO", color_discrete_map=SITE_COLORS,
        orientation="h", barmode="stack",
        title="Centros de trabajo por entidad y tipo de sitio",
        labels={"n_centros": "Número de centros", "ESTADO": "", "TIPO DE SITIO": ""},
        category_orders={"ESTADO": state_order},
        text="n_centros",
    )
    fig.update_traces(textposition="inside", textfont_size=11)
    fig.update_layout(
        height=1360,
        legend=dict(orientation="h", y=-0.06, title=""),
        xaxis=dict(gridcolor="#334155", title="Número de centros de trabajo"),
        yaxis=dict(gridcolor="rgba(0,0,0,0)"),
        **{k: v for k, v in CHART_LAYOUT.items() if k not in ("xaxis", "yaxis")},
    )
    return fig


def fig_states_map(d: pl.DataFrame) -> go.Figure:
    state_totals = d.group_by("ESTADO").agg(pl.col("NÚMERO DE VISITAS").sum().alias("total"))
    state_sites = d.group_by("ESTADO").agg(pl.col("CENTRO DE TRABAJO").n_unique().alias("n_sitios"))
    state_foreign = (
        d.filter(pl.col("NACIONALIDAD") == "Extranjeros")
        .group_by("ESTADO").agg(pl.col("NÚMERO DE VISITAS").sum().alias("extranjeros"))
    )
    top_site = (
        d.group_by("ESTADO", "CENTRO DE TRABAJO")
        .agg(pl.col("NÚMERO DE VISITAS").sum().alias("v"))
        .sort("v", descending=True)
        .unique("ESTADO", keep="first")
        .select("ESTADO", pl.col("CENTRO DE TRABAJO").str.slice(0, 38).alias("top_sitio"))
    )

    summary = (
        state_totals
        .join(state_sites, on="ESTADO", how="left")
        .join(state_foreign, on="ESTADO", how="left")
        .join(top_site, on="ESTADO", how="left")
        .with_columns(
            pl.col("extranjeros").fill_null(0),
            (pl.col("extranjeros").fill_null(0) / pl.col("total") * 100).round(1).alias("pct_ext"),
            (pl.col("total") / 1e6).round(2).alias("total_M"),
        )
    )

    yr_min, yr_max = d["year"].min(), d["year"].max()
    fig = px.choropleth_map(
        summary,
        geojson=MEXICO_GEO,
        locations="ESTADO",
        color="total",
        featureidkey="properties.name",
        color_continuous_scale="YlOrRd",
        zoom=4.0,
        center={"lat": 23.6, "lon": -102.5},
        opacity=0.85,
        hover_name="ESTADO",
        custom_data=["total_M", "n_sitios", "pct_ext", "top_sitio"],
        title=f"Visitas por entidad federativa ({yr_min}–{yr_max})",
        map_style="carto-darkmatter",
    )
    fig.update_traces(
        hovertemplate=(
            "<b>%{hovertext}</b><br>"
            "Visitas: %{customdata[0]:.2f}M<br>"
            "Sitios: %{customdata[1]}<br>"
            "Extranjeros: %{customdata[2]:.1f}%<br>"
            "Top sitio: %{customdata[3]}<extra></extra>"
        )
    )
    fig.update_coloraxes(
        colorbar=dict(
            title=dict(text="Visitas", font=dict(color="#CBD5E1")),
            tickfont=dict(color="#CBD5E1"),
            tickformat=".2s",
        )
    )
    fig.update_layout(
        height=620,
        margin=dict(l=0, r=0, t=40, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#CBD5E1",
    )
    return fig


# ── KPI computation ───────────────────────────────────────────────────────────

def compute_kpis(d: pl.DataFrame) -> tuple:
    total = d["NÚMERO DE VISITAS"].sum()
    pct_foreign = (
        d.filter(pl.col("NACIONALIDAD") == "Extranjeros")["NÚMERO DE VISITAS"].sum()
        / total * 100
    )
    peak_yr = (
        d.group_by("year").agg(pl.col("NÚMERO DE VISITAS").sum())
        .sort("NÚMERO DE VISITAS", descending=True)
        .head(1)["year"][0]
    )
    return total, pct_foreign, peak_yr


# ── Layout ────────────────────────────────────────────────────────────────────
CARD_STYLE = {
    "background": "#1E293B", "border": "1px solid #334155",
    "borderRadius": "8px", "padding": "16px", "textAlign": "center",
}
TAB_STYLE = {"backgroundColor": "#0F172A", "color": "#94A3B8", "borderTop": "none"}
TAB_SEL = {"backgroundColor": "#1E293B", "color": "#F8FAFC",
           "borderTop": "2px solid #2E86AB", "fontWeight": "600"}

SLIDER_MARKS = {y: str(y) for y in range(YEAR_MIN, YEAR_MAX + 1, 5)}
SLIDER_MARKS[YEAR_MIN] = str(YEAR_MIN)
SLIDER_MARKS[YEAR_MAX] = str(YEAR_MAX)


def kpi_card(title, value_id, sub_id):
    return html.Div([
        html.P(title, style={"color": "#94A3B8", "fontSize": "0.8rem", "marginBottom": "4px"}),
        html.H3(id=value_id, style={"color": "#F8FAFC", "fontWeight": "700", "margin": "0"}),
        html.Small(id=sub_id, style={"color": "#64748B"}),
    ], style=CARD_STYLE)


def graph_row(*id_width_pairs: tuple[str, int]):
    return dbc.Row([
        dbc.Col(dcc.Graph(id=gid, config={"displayModeBar": False}), width=w)
        for gid, w in id_width_pairs
    ], className="mb-3 g-3")


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SLATE])
app.title = "INAH · Museos y Zonas Arqueológicas"

app.layout = html.Div(
    style={"backgroundColor": "#0F172A", "minHeight": "100vh", "padding": "24px"},
    children=[
        # Header
        html.Div([
            html.H2("Visitas a Museos y Zonas Arqueológicas · INAH",
                    style={"color": "#F8FAFC", "fontWeight": "700", "marginBottom": "4px"}),
            html.P("México · 1996–2025 · 418 mil registros · SIINAH",
                   style={"color": "#64748B", "fontSize": "0.9rem"}),
        ], className="mb-4"),

        # KPI row
        dbc.Row([
            dbc.Col(kpi_card("Total de visitas", "kpi-total-val", "kpi-total-sub"), width=3),
            dbc.Col(kpi_card("Sitios únicos", "kpi-sites-val", "kpi-sites-sub"), width=3),
            dbc.Col(kpi_card("Visitantes extranjeros", "kpi-foreign-val", "kpi-foreign-sub"), width=3),
            dbc.Col(kpi_card("Año pico", "kpi-peak-val", "kpi-peak-sub"), width=3),
        ], className="mb-3 g-3"),

        # Filters row: year slider + centro dropdown
        dbc.Row([
            dbc.Col(html.Div([
                html.Label("Rango de años",
                           style={"color": "#94A3B8", "fontSize": "0.85rem", "marginBottom": "8px"}),
                dcc.RangeSlider(
                    id="year-range",
                    min=YEAR_MIN, max=YEAR_MAX,
                    value=[YEAR_MIN, YEAR_MAX],
                    marks=SLIDER_MARKS,
                    step=1,
                    tooltip={"placement": "bottom", "always_visible": True},
                    allowCross=False,
                ),
            ], style={"background": "#1E293B", "border": "1px solid #334155",
                      "borderRadius": "8px", "padding": "16px 24px 8px"}),
            width=8),
            dbc.Col(html.Div([
                html.Div([
                    html.Label("Centro de trabajo",
                               style={"color": "#94A3B8", "fontSize": "0.85rem"}),
                    html.Span(id="centro-badge", style={"display": "none"}),
                ], style={"marginBottom": "8px"}),
                dcc.Dropdown(
                    id="centro-dropdown",
                    options=[{"label": c, "value": c} for c in ALL_CENTROS],
                    value=None,
                    placeholder="Todos los centros de trabajo…",
                    clearable=True,
                    searchable=True,
                    className="dark-dropdown",
                ),
            ], style={"background": "#1E293B", "border": "1px solid #334155",
                      "borderRadius": "8px", "padding": "16px"}),
            width=4),
        ], className="mb-4 g-3"),

        # Tabs
        dcc.Tabs(style={"marginBottom": "16px"}, children=[
            dcc.Tab(label="📈 Tendencias", style=TAB_STYLE, selected_style=TAB_SEL, children=[
                graph_row(("graph-annual-trend", 12)),
                graph_row(("graph-yoy", 12)),
                graph_row(("graph-site-donut", 6), ("graph-visitor-donut", 6)),
            ]),
            dcc.Tab(label="🗾 Entidades", style=TAB_STYLE, selected_style=TAB_SEL, children=[
                graph_row(("graph-states-map", 12)),
            ]),
            dcc.Tab(label="🗺 Geografía", style=TAB_STYLE, selected_style=TAB_SEL, children=[
                graph_row(("graph-top-states", 12)),
                graph_row(("graph-foreign-ratio", 12)),
            ]),
            dcc.Tab(label="🏛 Sitios", style=TAB_STYLE, selected_style=TAB_SEL, children=[
                graph_row(("graph-top-sites", 12)),
                graph_row(("graph-foreign-site-type", 12)),
                graph_row(("graph-sites-by-state", 12)),
            ]),
            dcc.Tab(label="👥 Visitantes", style=TAB_STYLE, selected_style=TAB_SEL, children=[
                graph_row(("graph-visitor-trend", 12)),
                graph_row(("graph-paid-free", 6), ("graph-foreign-trend", 6)),
            ]),
            dcc.Tab(label="📅 Estacionalidad", style=TAB_STYLE, selected_style=TAB_SEL, children=[
                graph_row(("graph-heatmap", 12)),
                graph_row(("graph-monthly", 12)),
            ]),
        ]),

        # Insights footer
        html.Div([
            html.H6("Hallazgos clave", style={"color": "#94A3B8", "fontWeight": "600", "marginBottom": "12px"}),
            dbc.Row([
                dbc.Col(html.Div([
                    html.Strong("Teotihuacán lidera con 72M visitas", style={"color": "#F8FAFC"}),
                    html.P("Más del doble que el Museo Nacional de Antropología (58M). "
                           "Las zonas arqueológicas concentran el 57% del total.",
                           style={"color": "#94A3B8", "fontSize": "0.83rem", "marginTop": "4px"}),
                ], style={**CARD_STYLE, "textAlign": "left"}), width=4),
                dbc.Col(html.Div([
                    html.Strong("Quintana Roo: 65% visitantes extranjeros", style={"color": "#F8FAFC"}),
                    html.P("Tulum y Chichén Itzá impulsan el turismo extranjero en el sureste. "
                           "Las zonas arq. atraen 3× más extranjeros que los museos.",
                           style={"color": "#94A3B8", "fontSize": "0.83rem", "marginTop": "4px"}),
                ], style={**CARD_STYLE, "textAlign": "left"}), width=4),
                dbc.Col(html.Div([
                    html.Strong("COVID: caída del 73% en 2020", style={"color": "#F8FAFC"}),
                    html.P("Tras el mínimo de 7M en 2021, se recuperó a 21.4M en 2025, "
                           "aún ~22% por debajo del pico de 2019 (27.4M).",
                           style={"color": "#94A3B8", "fontSize": "0.83rem", "marginTop": "4px"}),
                ], style={**CARD_STYLE, "textAlign": "left"}), width=4),
            ], className="g-3"),
            dbc.Row([
                dbc.Col(html.Div([
                    html.Strong("Entrada dominical: 18.9% del total", style={"color": "#F8FAFC"}),
                    html.P("El acceso gratuito los domingos es la 2ª categoría más grande. "
                           "Solo el 52% de las visitas son boleto pagado.",
                           style={"color": "#94A3B8", "fontSize": "0.83rem", "marginTop": "4px"}),
                ], style={**CARD_STYLE, "textAlign": "left"}), width=4),
                dbc.Col(html.Div([
                    html.Strong("Marzo es el mes pico (Semana Santa)", style={"color": "#F8FAFC"}),
                    html.P("Mayo y septiembre son los meses más bajos. "
                           "El patrón estacional es estable y predecible.",
                           style={"color": "#94A3B8", "fontSize": "0.83rem", "marginTop": "4px"}),
                ], style={**CARD_STYLE, "textAlign": "left"}), width=4),
                dbc.Col(html.Div([
                    html.Strong("CDMX domina en volumen, Quintana Roo en turismo", style={"color": "#F8FAFC"}),
                    html.P("Ciudad de México concentra 26% de todas las visitas (150M), "
                           "pero Quintana Roo tiene la mayor proporción de extranjeros (65.5%).",
                           style={"color": "#94A3B8", "fontSize": "0.83rem", "marginTop": "4px"}),
                ], style={**CARD_STYLE, "textAlign": "left"}), width=4),
            ], className="g-3 mt-0"),
        ], className="mt-4"),
    ]
)


# ── Single callback — driven by year range slider + centro dropdown ────────────
@app.callback(
    Output("kpi-total-val", "children"),
    Output("kpi-total-sub", "children"),
    Output("kpi-sites-val", "children"),
    Output("kpi-sites-sub", "children"),
    Output("kpi-foreign-val", "children"),
    Output("kpi-foreign-sub", "children"),
    Output("kpi-peak-val", "children"),
    Output("kpi-peak-sub", "children"),
    Output("centro-badge", "children"),
    Output("centro-badge", "style"),
    Output("graph-states-map", "figure"),
    Output("graph-annual-trend", "figure"),
    Output("graph-yoy", "figure"),
    Output("graph-site-donut", "figure"),
    Output("graph-visitor-donut", "figure"),
    Output("graph-top-states", "figure"),
    Output("graph-foreign-ratio", "figure"),
    Output("graph-top-sites", "figure"),
    Output("graph-foreign-site-type", "figure"),
    Output("graph-sites-by-state", "figure"),
    Output("graph-visitor-trend", "figure"),
    Output("graph-paid-free", "figure"),
    Output("graph-foreign-trend", "figure"),
    Output("graph-heatmap", "figure"),
    Output("graph-monthly", "figure"),
    Input("year-range", "value"),
    Input("centro-dropdown", "value"),
)
def update_all(year_range, centro):
    yr_min, yr_max = year_range
    d = df.filter(pl.col("year").is_between(yr_min, yr_max))
    if centro:
        d = d.filter(pl.col("CENTRO DE TRABAJO") == centro)

    total, pct_foreign, peak_yr = compute_kpis(d)
    n_sites = d["CENTRO DE TRABAJO"].n_unique()
    sites_sub = centro if centro else "museos y zonas arqueológicas"

    badge_text = "filtro activo" if centro else ""
    badge_style = {**{"display": "inline-block", "background": "#2E86AB", "color": "#fff",
                      "borderRadius": "4px", "fontSize": "0.72rem", "padding": "1px 7px",
                      "marginLeft": "8px", "verticalAlign": "middle"},
                   **({"display": "none"} if not centro else {})}

    return (
        f"{total / 1e6:.1f}M",
        f"{yr_min}–{yr_max}",
        f"{n_sites:,}",
        sites_sub,
        f"{pct_foreign:.1f}%",
        "del total en el período",
        str(peak_yr),
        "mayor número de visitas",
        badge_text,
        badge_style,
        fig_states_map(d),
        fig_annual_trend(d),
        fig_yoy_change(d),
        fig_site_donut(d),
        fig_visitor_type_donut(d),
        fig_top_states(d),
        fig_foreign_ratio(d),
        fig_top_sites(d),
        fig_foreign_by_site_type(d),
        fig_sites_by_state(d),
        fig_visitor_trend(d),
        fig_paid_free_ratio(d),
        fig_foreign_trend(d),
        fig_seasonality_heatmap(d),
        fig_monthly_pattern(d),
    )


if __name__ == "__main__":
    print("Dashboard disponible en http://localhost:8050")
    app.run(debug=False, port=8050)
