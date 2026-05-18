import json
import polars as pl
import plotly.express as px
import plotly.graph_objects as go
import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc

# ── Data ──────────────────────────────────────────────────────────────────────
df = (
    pl.read_csv("data/productos_agricolas.csv")
    .with_columns(pl.col("AÑO").cast(pl.Int32))
    .filter(pl.col("AÑO") <= 2024)
)

YEAR_MIN  = df["AÑO"].min()
YEAR_MAX  = df["AÑO"].max()
ALL_STATES = sorted(df["ENTIDAD"].unique().to_list())
ALL_CROPS  = sorted(df["CULTIVO"].unique().to_list())

with open("data/mexico_states.geojson") as _f:
    MEXICO_GEO = json.load(_f)
for _feat in MEXICO_GEO["features"]:
    if _feat["properties"]["name"] == "México":
        _feat["properties"]["name"] = "Estado de México"

TECH_COLORS = {
    "Cielo abierto": "#2E86AB",
    "Invernadero":   "#3BB273",
    "Macro túnel":   "#F4A261",
    "Malla sombra":  "#9B59B6",
}
MARKET_COLORS = {"Nacional": "#3BB273", "Exportación": "#F4A261"}
PROD_COLORS   = {"Convencional": "#2E86AB", "Orgánico": "#3BB273"}

CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font_color="#CBD5E1",
    xaxis=dict(gridcolor="#334155"),
    yaxis=dict(gridcolor="#334155"),
)


def _fmt_pesos(v: float) -> str:
    if v >= 1e12:
        return f"${v / 1e12:.1f}T"
    if v >= 1e9:
        return f"${v / 1e9:.1f}B"
    return f"${v / 1e6:.1f}M"


# ── Figure factories ──────────────────────────────────────────────────────────

def fig_produccion_anual(d: pl.DataFrame) -> go.Figure:
    yearly = (
        d.group_by("AÑO")
        .agg((pl.col("VALOR_PRODUCCION").sum() / 1e9).round(2).alias("valor_B"))
        .sort("AÑO")
    )
    yr_min, yr_max = d["AÑO"].min(), d["AÑO"].max()
    fig = go.Figure(go.Scatter(
        x=yearly["AÑO"].to_list(), y=yearly["valor_B"].to_list(),
        mode="lines+markers",
        line=dict(color="#2E86AB", width=2),
        fill="tozeroy", fillcolor="rgba(46,134,171,0.15)",
        hovertemplate="Año: %{x}<br>Valor: $%{y:.1f}B<extra></extra>",
    ))
    fig.update_layout(
        title=f"Valor total de producción agrícola {yr_min}–{yr_max}",
        height=380,
        xaxis=dict(gridcolor="#334155", title="Año"),
        yaxis=dict(gridcolor="#334155", title="Miles de millones de pesos"),
        **{k: v for k, v in CHART_LAYOUT.items() if k not in ("xaxis", "yaxis")},
    )
    return fig


def fig_top_cultivos_valor(d: pl.DataFrame) -> go.Figure:
    top10 = (
        d.group_by("CULTIVO")
        .agg((pl.col("VALOR_PRODUCCION").sum() / 1e9).round(1).alias("valor_B"))
        .sort("valor_B", descending=True)
        .head(10)
        .sort("valor_B")
    )
    fig = px.bar(
        top10, x="valor_B", y="CULTIVO", orientation="h",
        title="Top 10 cultivos por valor de producción acumulado",
        labels={"valor_B": "Miles de millones de pesos", "CULTIVO": ""},
        color="valor_B",
        color_continuous_scale=[[0, "#1E3A5F"], [1, "#2E86AB"]],
        text="valor_B",
    )
    fig.update_traces(
        textposition="outside", textfont_color="#CBD5E1",
        texttemplate="$%{text:.1f}B",
    )
    fig.update_layout(
        height=420, coloraxis_showscale=False,
        xaxis=dict(gridcolor="#334155"),
        yaxis=dict(gridcolor="rgba(0,0,0,0)"),
        **{k: v for k, v in CHART_LAYOUT.items() if k not in ("xaxis", "yaxis")},
    )
    return fig


def fig_mapa_estados(d: pl.DataFrame) -> go.Figure:
    state_data = (
        d.group_by("ENTIDAD")
        .agg(
            pl.col("VALOR_PRODUCCION").sum().alias("valor"),
            pl.col("VOLUMEN_PRODUCCION").sum().alias("volumen"),
            pl.col("CULTIVO").n_unique().alias("n_cultivos"),
        )
        .with_columns(
            (pl.col("valor") / 1e9).round(2).alias("valor_B"),
            (pl.col("volumen") / 1e6).round(2).alias("volumen_M"),
        )
    )
    yr_min, yr_max = d["AÑO"].min(), d["AÑO"].max()
    fig = px.choropleth_map(
        state_data,
        geojson=MEXICO_GEO,
        locations="ENTIDAD",
        color="valor",
        featureidkey="properties.name",
        color_continuous_scale="YlOrRd",
        zoom=4.0,
        center={"lat": 23.6, "lon": -102.5},
        opacity=0.85,
        hover_name="ENTIDAD",
        custom_data=["valor_B", "volumen_M", "n_cultivos"],
        title=f"Valor de producción por entidad ({yr_min}–{yr_max})",
        map_style="carto-darkmatter",
    )
    fig.update_traces(
        hovertemplate=(
            "<b>%{hovertext}</b><br>"
            "Valor: $%{customdata[0]:.2f}B<br>"
            "Volumen: %{customdata[1]:.2f}M ton<br>"
            "Cultivos únicos: %{customdata[2]}<extra></extra>"
        )
    )
    fig.update_coloraxes(
        colorbar=dict(
            title=dict(text="Valor (pesos)", font=dict(color="#CBD5E1")),
            tickfont=dict(color="#CBD5E1"),
            tickformat=".2s",
        )
    )
    fig.update_layout(
        height=580,
        margin=dict(l=0, r=0, t=40, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#CBD5E1",
    )
    return fig


def fig_top_estados(d: pl.DataFrame) -> go.Figure:
    top15 = (
        d.group_by("ENTIDAD")
        .agg((pl.col("VALOR_PRODUCCION").sum() / 1e9).round(1).alias("valor_B"))
        .sort("valor_B", descending=True)
        .head(15)
        .sort("valor_B")
    )
    fig = px.bar(
        top15, x="valor_B", y="ENTIDAD", orientation="h",
        title="Top 15 estados por valor de producción",
        labels={"valor_B": "Miles de millones de pesos", "ENTIDAD": ""},
        color="valor_B",
        color_continuous_scale=[[0, "#1E3A5F"], [1, "#2E86AB"]],
        text="valor_B",
    )
    fig.update_traces(
        textposition="outside", textfont_color="#CBD5E1",
        texttemplate="$%{text:.1f}B",
    )
    fig.update_layout(
        height=500, coloraxis_showscale=False,
        xaxis=dict(gridcolor="#334155"),
        yaxis=dict(gridcolor="rgba(0,0,0,0)"),
        **{k: v for k, v in CHART_LAYOUT.items() if k not in ("xaxis", "yaxis")},
    )
    return fig


def fig_rendimiento_estados(d: pl.DataFrame) -> go.Figure:
    top15 = (
        d.filter((pl.col("RENDIMIENTO") > 0) & (pl.col("UNIDAD_MEDIDA") == "Tonelada"))
        .group_by("ENTIDAD")
        .agg(pl.col("RENDIMIENTO").mean().round(2).alias("rend"))
        .sort("rend", descending=True)
        .head(15)
        .sort("rend")
    )
    fig = px.bar(
        top15, x="rend", y="ENTIDAD", orientation="h",
        title="Rendimiento promedio por estado (top 15)",
        labels={"rend": "Rendimiento promedio (ton/ha)", "ENTIDAD": ""},
        color="rend",
        color_continuous_scale=[[0, "#1B4332"], [1, "#3BB273"]],
        text="rend",
    )
    fig.update_traces(
        textposition="outside", textfont_color="#CBD5E1",
        texttemplate="%{text:.1f}",
    )
    fig.update_layout(
        height=500, coloraxis_showscale=False,
        xaxis=dict(gridcolor="#334155"),
        yaxis=dict(gridcolor="rgba(0,0,0,0)"),
        **{k: v for k, v in CHART_LAYOUT.items() if k not in ("xaxis", "yaxis")},
    )
    return fig


def fig_treemap_cultivos(d: pl.DataFrame) -> go.Figure:
    top20 = (
        d.group_by("CULTIVO")
        .agg(pl.col("VALOR_PRODUCCION").sum().alias("valor"))
        .sort("valor", descending=True)
        .head(20)
    )
    fig = px.treemap(
        top20, path=["CULTIVO"], values="valor",
        title="Top 20 cultivos — distribución del valor de producción",
        color="valor",
        color_continuous_scale="Blues",
    )
    fig.update_traces(textinfo="label+percent root")
    fig.update_layout(
        height=520,
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#CBD5E1",
        margin=dict(l=0, r=0, t=40, b=0),
        coloraxis_showscale=False,
    )
    return fig


def fig_scatter_rendimiento_precio(d: pl.DataFrame) -> go.Figure:
    top30 = (
        d.filter(
            (pl.col("RENDIMIENTO") > 0)
            & (pl.col("PRECIO_MEDIO_RURAL") > 0)
            & (pl.col("UNIDAD_MEDIDA") == "Tonelada")
        )
        .group_by("CULTIVO")
        .agg(
            pl.col("RENDIMIENTO").mean().round(2).alias("rend"),
            pl.col("PRECIO_MEDIO_RURAL").mean().round(2).alias("precio"),
            pl.col("VOLUMEN_PRODUCCION").sum().alias("volumen"),
        )
        .sort("volumen", descending=True)
        .head(30)
    )
    fig = px.scatter(
        top30,
        x="rend", y="precio",
        size="volumen", color="CULTIVO",
        hover_name="CULTIVO",
        title="Rendimiento vs. Precio por cultivo (top 30 por volumen)",
        labels={
            "rend":    "Rendimiento promedio (ton/ha)",
            "precio":  "Precio medio rural ($/ton)",
            "volumen": "Volumen producido",
        },
        size_max=50,
    )
    fig.update_layout(height=480, showlegend=False, **CHART_LAYOUT)
    return fig


def fig_evolucion_top_cultivos(d: pl.DataFrame) -> go.Figure:
    top5 = (
        d.group_by("CULTIVO")
        .agg(pl.col("VALOR_PRODUCCION").sum().alias("total"))
        .sort("total", descending=True)
        .head(5)
        ["CULTIVO"].to_list()
    )
    trend = (
        d.filter(pl.col("CULTIVO").is_in(top5))
        .group_by("AÑO", "CULTIVO")
        .agg((pl.col("VALOR_PRODUCCION").sum() / 1e9).round(2).alias("valor_B"))
        .sort("AÑO")
    )
    fig = px.line(
        trend, x="AÑO", y="valor_B", color="CULTIVO",
        title="Evolución histórica — top 5 cultivos por valor",
        labels={"valor_B": "Miles de millones de pesos", "AÑO": "Año", "CULTIVO": ""},
        markers=False,
    )
    fig.update_layout(
        height=480,
        legend=dict(orientation="h", y=-0.2, title=""),
        **CHART_LAYOUT,
    )
    return fig


def fig_tecnologia_trend(d: pl.DataFrame) -> go.Figure:
    trend = (
        d.group_by("AÑO", "TIPO_TECNOLOGIA")
        .agg(pl.col("SUPERFICIE_SEMBRADA").sum().alias("superficie"))
        .sort("AÑO")
    )
    fig = px.bar(
        trend, x="AÑO", y="superficie", color="TIPO_TECNOLOGIA",
        barmode="stack",
        color_discrete_map=TECH_COLORS,
        title="Superficie sembrada por tipo de tecnología agrícola",
        labels={"superficie": "Superficie (ha)", "AÑO": "Año", "TIPO_TECNOLOGIA": ""},
    )
    fig.update_layout(
        height=420,
        legend=dict(orientation="h", y=-0.15, title=""),
        xaxis=dict(gridcolor="#334155"),
        yaxis=dict(gridcolor="#334155"),
        **{k: v for k, v in CHART_LAYOUT.items() if k not in ("xaxis", "yaxis")},
    )
    return fig


def fig_rendimiento_por_tecnologia(d: pl.DataFrame) -> go.Figure:
    rend = (
        d.filter((pl.col("RENDIMIENTO") > 0) & (pl.col("UNIDAD_MEDIDA") == "Tonelada"))
        .group_by("TIPO_TECNOLOGIA")
        .agg(pl.col("RENDIMIENTO").mean().round(2).alias("rend"))
        .sort("rend", descending=True)
    )
    colors = [TECH_COLORS.get(t, "#2E86AB") for t in rend["TIPO_TECNOLOGIA"].to_list()]
    fig = go.Figure(go.Bar(
        x=rend["TIPO_TECNOLOGIA"].to_list(),
        y=rend["rend"].to_list(),
        marker_color=colors,
        text=[f"{v:.1f}" for v in rend["rend"].to_list()],
        textposition="outside",
        hovertemplate="%{x}<br>Rendimiento: %{y:.1f} ton/ha<extra></extra>",
    ))
    fig.update_layout(
        title="Rendimiento promedio por tipo de tecnología",
        height=360,
        xaxis=dict(gridcolor="rgba(0,0,0,0)"),
        yaxis=dict(gridcolor="#334155", title="Rendimiento promedio (ton/ha)"),
        **{k: v for k, v in CHART_LAYOUT.items() if k not in ("xaxis", "yaxis")},
    )
    return fig


def fig_produccion_tipo(d: pl.DataFrame) -> go.Figure:
    tipo = (
        d.group_by("TIPO_PRODUCCION")
        .agg(pl.col("VALOR_PRODUCCION").sum().alias("valor"))
    )
    fig = px.pie(
        tipo, names="TIPO_PRODUCCION", values="valor",
        color="TIPO_PRODUCCION", color_discrete_map=PROD_COLORS,
        hole=0.5,
        title="Valor de producción: Convencional vs. Orgánico",
    )
    fig.update_traces(textinfo="percent+label", textfont_size=13)
    fig.update_layout(
        height=370, showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)", font_color="#CBD5E1",
    )
    return fig


def fig_mercado_trend(d: pl.DataFrame) -> go.Figure:
    trend = (
        d.group_by("AÑO", "TIPO_MERCADO")
        .agg((pl.col("VALOR_PRODUCCION").sum() / 1e9).round(2).alias("valor_B"))
        .sort("AÑO")
    )
    fig = px.area(
        trend, x="AÑO", y="valor_B", color="TIPO_MERCADO",
        color_discrete_map=MARKET_COLORS,
        title="Valor de producción: Nacional vs. Exportación",
        labels={"valor_B": "Miles de millones de pesos", "AÑO": "Año", "TIPO_MERCADO": ""},
    )
    fig.update_layout(
        height=420,
        legend=dict(orientation="h", y=-0.15, title=""),
        **CHART_LAYOUT,
    )
    return fig


def fig_top_cultivos_exportacion(d: pl.DataFrame) -> go.Figure:
    top10 = (
        d.filter(pl.col("TIPO_MERCADO") == "Exportación")
        .group_by("CULTIVO")
        .agg((pl.col("VALOR_PRODUCCION").sum() / 1e9).round(2).alias("valor_B"))
        .sort("valor_B", descending=True)
        .head(10)
        .sort("valor_B")
    )
    fig = px.bar(
        top10, x="valor_B", y="CULTIVO", orientation="h",
        title="Top 10 cultivos de exportación (por valor acumulado)",
        labels={"valor_B": "Miles de millones de pesos", "CULTIVO": ""},
        color="valor_B",
        color_continuous_scale=[[0, "#5C2D00"], [1, "#F4A261"]],
        text="valor_B",
    )
    fig.update_traces(
        textposition="outside", textfont_color="#CBD5E1",
        texttemplate="$%{text:.1f}B",
    )
    fig.update_layout(
        height=420, coloraxis_showscale=False,
        xaxis=dict(gridcolor="#334155"),
        yaxis=dict(gridcolor="rgba(0,0,0,0)"),
        **{k: v for k, v in CHART_LAYOUT.items() if k not in ("xaxis", "yaxis")},
    )
    return fig


def fig_exportacion_por_estado(d: pl.DataFrame) -> go.Figure:
    top10_states = (
        d.group_by("ENTIDAD")
        .agg(pl.col("VALOR_PRODUCCION").sum().alias("total"))
        .sort("total", descending=True)
        .head(10)
        ["ENTIDAD"].to_list()
    )
    breakdown = (
        d.filter(pl.col("ENTIDAD").is_in(top10_states))
        .group_by("ENTIDAD", "TIPO_MERCADO")
        .agg((pl.col("VALOR_PRODUCCION").sum() / 1e9).round(2).alias("valor_B"))
    )
    state_order = (
        breakdown.group_by("ENTIDAD")
        .agg(pl.col("valor_B").sum().alias("total"))
        .sort("total", descending=True)
        ["ENTIDAD"].to_list()
    )
    fig = px.bar(
        breakdown, x="ENTIDAD", y="valor_B", color="TIPO_MERCADO",
        barmode="stack",
        color_discrete_map=MARKET_COLORS,
        title="Mercado Nacional vs. Exportación por estado (top 10)",
        labels={"valor_B": "Miles de millones de pesos", "ENTIDAD": "", "TIPO_MERCADO": ""},
        category_orders={"ENTIDAD": state_order},
    )
    fig.update_layout(
        height=420,
        legend=dict(orientation="h", y=-0.2, title=""),
        xaxis=dict(gridcolor="rgba(0,0,0,0)", tickangle=-30),
        yaxis=dict(gridcolor="#334155"),
        **{k: v for k, v in CHART_LAYOUT.items() if k not in ("xaxis", "yaxis")},
    )
    return fig


def fig_siniestro_anual(d: pl.DataFrame) -> go.Figure:
    yearly = (
        d.group_by("AÑO")
        .agg(
            pl.col("SUPERFICIE_SINIESTRADA").sum().alias("siniestrada"),
            pl.col("SUPERFICIE_SEMBRADA").sum().alias("sembrada"),
        )
        .sort("AÑO")
        .with_columns(
            (pl.col("siniestrada") / pl.col("sembrada") * 100).round(2).alias("tasa")
        )
    )
    fig = go.Figure(go.Scatter(
        x=yearly["AÑO"].to_list(), y=yearly["tasa"].to_list(),
        mode="lines+markers",
        line=dict(color="#E84855", width=2),
        fill="tozeroy", fillcolor="rgba(232,72,85,0.12)",
        hovertemplate="Año: %{x}<br>Tasa de siniestro: %{y:.2f}%<extra></extra>",
    ))
    fig.update_layout(
        title="Tasa de siniestro por año (sup. siniestrada / sup. sembrada × 100)",
        height=360,
        xaxis=dict(gridcolor="#334155", title="Año"),
        yaxis=dict(gridcolor="#334155", title="Tasa de siniestro (%)"),
        **{k: v for k, v in CHART_LAYOUT.items() if k not in ("xaxis", "yaxis")},
    )
    return fig


def fig_siniestro_estados(d: pl.DataFrame) -> go.Figure:
    state_risk = (
        d.group_by("ENTIDAD")
        .agg(
            pl.col("SUPERFICIE_SINIESTRADA").sum().alias("siniestrada"),
            pl.col("SUPERFICIE_SEMBRADA").sum().alias("sembrada"),
        )
        .with_columns(
            (pl.col("siniestrada") / pl.col("sembrada") * 100).round(2).alias("tasa")
        )
        .sort("tasa", descending=True)
        .head(15)
        .sort("tasa")
    )
    pct_vals = state_risk["tasa"].to_list()
    avg_tasa = sum(pct_vals) / len(pct_vals) if pct_vals else 0
    colors = ["#E84855" if v > avg_tasa else "#F4A261" for v in pct_vals]
    fig = go.Figure(go.Bar(
        x=pct_vals, y=state_risk["ENTIDAD"].to_list(), orientation="h",
        marker_color=colors,
        text=[f"{v:.1f}%" for v in pct_vals],
        textposition="outside",
        hovertemplate="%{y}<br>Tasa de siniestro: %{x:.2f}%<extra></extra>",
    ))
    fig.update_layout(
        title="Estados con mayor tasa de siniestro agrícola (top 15)",
        height=500,
        xaxis=dict(gridcolor="#334155", title="Tasa de siniestro (%)"),
        yaxis=dict(gridcolor="rgba(0,0,0,0)"),
        **{k: v for k, v in CHART_LAYOUT.items() if k not in ("xaxis", "yaxis")},
    )
    return fig


def fig_siniestro_scatter(d: pl.DataFrame) -> go.Figure:
    state_data = (
        d.group_by("ENTIDAD")
        .agg(
            pl.col("SUPERFICIE_SEMBRADA").sum().alias("sembrada"),
            pl.col("SUPERFICIE_SINIESTRADA").sum().alias("siniestrada"),
            pl.col("VALOR_PRODUCCION").sum().alias("valor"),
        )
        .with_columns(
            (pl.col("siniestrada") / pl.col("sembrada") * 100).round(2).alias("tasa"),
            (pl.col("sembrada") / 1e6).alias("sembrada_M"),
        )
    )
    fig = px.scatter(
        state_data,
        x="sembrada_M", y="tasa",
        size="valor", color="ENTIDAD",
        hover_name="ENTIDAD",
        title="Superficie sembrada vs. Tasa de siniestro (tamaño = valor producción)",
        labels={
            "sembrada_M": "Superficie sembrada (millones de ha)",
            "tasa":       "Tasa de siniestro (%)",
            "valor":      "Valor producción",
        },
        size_max=50,
    )
    fig.update_layout(height=480, showlegend=False, **CHART_LAYOUT)
    return fig


# ── Crop explorer figures ────────────────────────────────────────────────────

def _empty_fig(msg: str) -> go.Figure:
    return go.Figure().update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#64748B", height=380,
        annotations=[dict(text=msg, showarrow=False,
                          font=dict(size=14, color="#64748B"),
                          xref="paper", yref="paper", x=0.5, y=0.5)],
    )


def fig_cultivo_trend(d: pl.DataFrame, cultivo: str) -> go.Figure:
    crop = d.filter(pl.col("CULTIVO") == cultivo)
    if crop.is_empty():
        return _empty_fig(f"Sin datos para {cultivo} en el período seleccionado")
    top_states = (
        crop.group_by("ENTIDAD")
        .agg(pl.col("VALOR_PRODUCCION").sum().alias("total"))
        .sort("total", descending=True)
        .head(8)
        ["ENTIDAD"].to_list()
    )
    trend = (
        crop.filter(pl.col("ENTIDAD").is_in(top_states))
        .group_by("AÑO", "ENTIDAD")
        .agg((pl.col("VALOR_PRODUCCION").sum() / 1e9).round(3).alias("valor_B"))
        .sort("AÑO")
    )
    fig = px.area(
        trend, x="AÑO", y="valor_B", color="ENTIDAD",
        title=f"Evolución de {cultivo} — valor por estado (top 8)",
        labels={"valor_B": "Miles de millones de pesos", "AÑO": "Año", "ENTIDAD": ""},
    )
    fig.update_layout(
        height=420,
        legend=dict(orientation="h", y=-0.2, title=""),
        **CHART_LAYOUT,
    )
    return fig


def fig_cultivo_estados(d: pl.DataFrame, cultivo: str) -> go.Figure:
    crop = (
        d.filter(pl.col("CULTIVO") == cultivo)
        .group_by("ENTIDAD")
        .agg(
            (pl.col("VALOR_PRODUCCION").sum() / 1e9).round(3).alias("valor_B"),
            pl.col("SUPERFICIE_SEMBRADA").sum().alias("superficie"),
            pl.col("AÑO").min().alias("desde"),
            pl.col("AÑO").max().alias("hasta"),
        )
        .sort("valor_B")
    )
    if crop.is_empty():
        return _empty_fig(f"Sin datos para {cultivo} en el período seleccionado")
    n = len(crop)
    fig = px.bar(
        crop, x="valor_B", y="ENTIDAD", orientation="h",
        title=f"Estados productores de {cultivo}",
        labels={"valor_B": "Miles de millones de pesos", "ENTIDAD": ""},
        color="valor_B",
        color_continuous_scale=[[0, "#1E3A5F"], [1, "#2E86AB"]],
        text="valor_B",
        custom_data=["superficie", "desde", "hasta"],
    )
    fig.update_traces(
        texttemplate="$%{text:.2f}B", textposition="outside", textfont_color="#CBD5E1",
        hovertemplate=(
            "<b>%{y}</b><br>Valor: $%{x:.3f}B<br>"
            "Superficie: %{customdata[0]:,.0f} ha<br>"
            "Cultivado: %{customdata[1]}–%{customdata[2]}<extra></extra>"
        ),
    )
    fig.update_layout(
        height=max(300, n * 28 + 80), coloraxis_showscale=False,
        xaxis=dict(gridcolor="#334155"),
        yaxis=dict(gridcolor="rgba(0,0,0,0)"),
        **{k: v for k, v in CHART_LAYOUT.items() if k not in ("xaxis", "yaxis")},
    )
    return fig


def fig_cultivo_precio(d: pl.DataFrame, cultivo: str) -> go.Figure:
    precio = (
        d.filter((pl.col("CULTIVO") == cultivo) & (pl.col("PRECIO_MEDIO_RURAL") > 0))
        .group_by("AÑO")
        .agg(pl.col("PRECIO_MEDIO_RURAL").mean().round(2).alias("precio"))
        .sort("AÑO")
    )
    if precio.is_empty():
        return _empty_fig(f"Sin datos de precio para {cultivo}")
    fig = go.Figure(go.Scatter(
        x=precio["AÑO"].to_list(), y=precio["precio"].to_list(),
        mode="lines+markers",
        line=dict(color="#F4A261", width=2),
        fill="tozeroy", fillcolor="rgba(244,162,97,0.12)",
        hovertemplate="Año: %{x}<br>Precio: $%{y:,.0f}/ton<extra></extra>",
    ))
    fig.update_layout(
        title=f"Precio medio rural de {cultivo} ($/ton)",
        height=420,
        xaxis=dict(gridcolor="#334155", title="Año"),
        yaxis=dict(gridcolor="#334155", title="Precio medio rural ($/ton)"),
        **{k: v for k, v in CHART_LAYOUT.items() if k not in ("xaxis", "yaxis")},
    )
    return fig


# ── KPI computation ──────────────────────────────────────────────────────────

def compute_kpis(d: pl.DataFrame) -> tuple:
    valor   = d["VALOR_PRODUCCION"].sum()
    volumen = d["VOLUMEN_PRODUCCION"].sum()
    n_cult  = d["CULTIVO"].n_unique()
    rend    = (
        d.filter((pl.col("RENDIMIENTO") > 0) & (pl.col("UNIDAD_MEDIDA") == "Tonelada"))
        ["RENDIMIENTO"].mean()
    )
    return valor, volumen, n_cult, rend or 0.0


# ── Layout ────────────────────────────────────────────────────────────────────

CARD_STYLE = {
    "background": "#1E293B", "border": "1px solid #334155",
    "borderRadius": "8px", "padding": "16px", "textAlign": "center",
}
TAB_STYLE = {"backgroundColor": "#0F172A", "color": "#94A3B8", "borderTop": "none"}
TAB_SEL   = {
    "backgroundColor": "#1E293B", "color": "#F8FAFC",
    "borderTop": "2px solid #2E86AB", "fontWeight": "600",
}

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
app.title = "Productos Agrícolas · México"

app.layout = html.Div(
    style={"backgroundColor": "#0F172A", "minHeight": "100vh", "padding": "24px"},
    children=[
        html.Div([
            html.H2("Producción Agrícola de México",
                    style={"color": "#F8FAFC", "fontWeight": "700", "marginBottom": "4px"}),
            html.P("México · 1980–2024 · 114,841 registros · 32 estados · 366 cultivos",
                   style={"color": "#64748B", "fontSize": "0.9rem"}),
        ], className="mb-4"),

        dbc.Row([
            dbc.Col(kpi_card("Valor total de producción", "kpi-valor-val", "kpi-valor-sub"), width=3),
            dbc.Col(kpi_card("Volumen total producido",   "kpi-vol-val",   "kpi-vol-sub"),   width=3),
            dbc.Col(kpi_card("Cultivos activos",          "kpi-cult-val",  "kpi-cult-sub"),  width=3),
            dbc.Col(kpi_card("Rendimiento promedio",      "kpi-rend-val",  "kpi-rend-sub"),  width=3),
        ], className="mb-3 g-3"),

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
                    html.Label("Estado",
                               style={"color": "#94A3B8", "fontSize": "0.85rem"}),
                    html.Span(id="estado-badge", style={"display": "none"}),
                ], style={"marginBottom": "8px"}),
                dcc.Dropdown(
                    id="estado-dropdown",
                    options=[{"label": s, "value": s} for s in ALL_STATES],
                    value=None,
                    placeholder="Todos los estados…",
                    clearable=True,
                    searchable=True,
                    className="dark-dropdown",
                ),
            ], style={"background": "#1E293B", "border": "1px solid #334155",
                      "borderRadius": "8px", "padding": "16px"}),
            width=4),
        ], className="mb-4 g-3"),

        dcc.Tabs(style={"marginBottom": "16px"}, children=[
            dcc.Tab(label="📈 Panorama", style=TAB_STYLE, selected_style=TAB_SEL, children=[
                graph_row(("graph-prod-anual", 12)),
                graph_row(("graph-top-cultivos", 12)),
            ]),
            dcc.Tab(label="🗺 Geografía", style=TAB_STYLE, selected_style=TAB_SEL, children=[
                graph_row(("graph-mapa", 12)),
                graph_row(("graph-top-estados", 6), ("graph-rend-estados", 6)),
            ]),
            dcc.Tab(label="🌱 Cultivos", style=TAB_STYLE, selected_style=TAB_SEL, children=[
                graph_row(("graph-treemap", 12)),
                graph_row(("graph-scatter-rend", 6), ("graph-evolucion", 6)),
            ]),
            dcc.Tab(label="⚙ Tecnología", style=TAB_STYLE, selected_style=TAB_SEL, children=[
                graph_row(("graph-tech-trend", 12)),
                graph_row(("graph-rend-tech", 6), ("graph-tipo-donut", 6)),
            ]),
            dcc.Tab(label="🛒 Mercado", style=TAB_STYLE, selected_style=TAB_SEL, children=[
                graph_row(("graph-mercado-trend", 12)),
                graph_row(("graph-top-export", 6), ("graph-export-estado", 6)),
            ]),
            dcc.Tab(label="⚠ Riesgo Agrícola", style=TAB_STYLE, selected_style=TAB_SEL, children=[
                graph_row(("graph-siniestro-anual", 12)),
                graph_row(("graph-siniestro-estados", 6), ("graph-siniestro-scatter", 6)),
            ]),
            dcc.Tab(label="🌾 Explorador", style=TAB_STYLE, selected_style=TAB_SEL, children=[
                dbc.Row([
                    dbc.Col(html.Div([
                        html.Label("Cultivo",
                                   style={"color": "#94A3B8", "fontSize": "0.85rem", "marginBottom": "8px"}),
                        dcc.Dropdown(
                            id="cultivo-dropdown",
                            options=[{"label": c, "value": c} for c in ALL_CROPS],
                            value=None,
                            placeholder="Selecciona un cultivo para explorar…",
                            clearable=True,
                            searchable=True,
                            className="dark-dropdown",
                        ),
                    ], style={"background": "#1E293B", "border": "1px solid #334155",
                              "borderRadius": "8px", "padding": "16px"}),
                    width=12),
                ], className="mb-3 g-3"),
                graph_row(("graph-cult-trend", 12)),
                graph_row(("graph-cult-estados", 6), ("graph-cult-precio", 6)),
            ]),
        ]),

        html.Div([
            html.H6("Hallazgos clave", style={"color": "#94A3B8", "fontWeight": "600", "marginBottom": "12px"}),
            dbc.Row([
                dbc.Col(html.Div([
                    html.Strong("Sinaloa, Jalisco y Sonora: el triángulo productivo", style={"color": "#F8FAFC"}),
                    html.P("Estos tres estados concentran más del 30% del valor agrícola nacional. "
                           "Sinaloa domina en volumen de granos y hortalizas de exportación.",
                           style={"color": "#94A3B8", "fontSize": "0.83rem", "marginTop": "4px"}),
                ], style={**CARD_STYLE, "textAlign": "left"}), width=4),
                dbc.Col(html.Div([
                    html.Strong("Invernadero: hasta 5× más rendimiento", style={"color": "#F8FAFC"}),
                    html.P("Los cultivos bajo invernadero superan al campo abierto en rendimiento por hectárea. "
                           "Su adopción ha crecido sostenidamente desde los 2000.",
                           style={"color": "#94A3B8", "fontSize": "0.83rem", "marginTop": "4px"}),
                ], style={**CARD_STYLE, "textAlign": "left"}), width=4),
                dbc.Col(html.Div([
                    html.Strong("Exportación concentrada en pocos cultivos", style={"color": "#F8FAFC"}),
                    html.P("Aguacate, tomate y berries dominan las exportaciones. "
                           "El valor de exportación agrícola se multiplicó 10× desde los 90.",
                           style={"color": "#94A3B8", "fontSize": "0.83rem", "marginTop": "4px"}),
                ], style={**CARD_STYLE, "textAlign": "left"}), width=4),
            ], className="g-3"),
        ], className="mt-4"),
    ]
)


# ── Single callback ──────────────────────────────────────────────────────────
@app.callback(
    Output("kpi-valor-val", "children"),
    Output("kpi-valor-sub", "children"),
    Output("kpi-vol-val",   "children"),
    Output("kpi-vol-sub",   "children"),
    Output("kpi-cult-val",  "children"),
    Output("kpi-cult-sub",  "children"),
    Output("kpi-rend-val",  "children"),
    Output("kpi-rend-sub",  "children"),
    Output("estado-badge",  "children"),
    Output("estado-badge",  "style"),
    Output("graph-prod-anual",        "figure"),
    Output("graph-top-cultivos",      "figure"),
    Output("graph-mapa",              "figure"),
    Output("graph-top-estados",       "figure"),
    Output("graph-rend-estados",      "figure"),
    Output("graph-treemap",           "figure"),
    Output("graph-scatter-rend",      "figure"),
    Output("graph-evolucion",         "figure"),
    Output("graph-tech-trend",        "figure"),
    Output("graph-rend-tech",         "figure"),
    Output("graph-tipo-donut",        "figure"),
    Output("graph-mercado-trend",     "figure"),
    Output("graph-top-export",        "figure"),
    Output("graph-export-estado",     "figure"),
    Output("graph-siniestro-anual",   "figure"),
    Output("graph-siniestro-estados", "figure"),
    Output("graph-siniestro-scatter", "figure"),
    Input("year-range",       "value"),
    Input("estado-dropdown",  "value"),
)
def update_all(year_range, estado):
    yr_min, yr_max = year_range
    d = df.filter(pl.col("AÑO").is_between(yr_min, yr_max))
    if estado:
        d = d.filter(pl.col("ENTIDAD") == estado)

    valor, volumen, n_cult, rend = compute_kpis(d)

    badge_style = {
        "display": "inline-block", "background": "#2E86AB", "color": "#fff",
        "borderRadius": "4px", "fontSize": "0.72rem", "padding": "1px 7px",
        "marginLeft": "8px", "verticalAlign": "middle",
    }
    if not estado:
        badge_style = {"display": "none"}

    return (
        _fmt_pesos(valor),
        f"{yr_min}–{yr_max}",
        f"{volumen / 1e6:.1f}M ton",
        "volumen producido total",
        f"{n_cult:,}",
        "cultivos únicos en el período",
        f"{rend:.1f} ton/ha",
        "rendimiento promedio",
        "filtro activo" if estado else "",
        badge_style,
        fig_produccion_anual(d),
        fig_top_cultivos_valor(d),
        fig_mapa_estados(d),
        fig_top_estados(d),
        fig_rendimiento_estados(d),
        fig_treemap_cultivos(d),
        fig_scatter_rendimiento_precio(d),
        fig_evolucion_top_cultivos(d),
        fig_tecnologia_trend(d),
        fig_rendimiento_por_tecnologia(d),
        fig_produccion_tipo(d),
        fig_mercado_trend(d),
        fig_top_cultivos_exportacion(d),
        fig_exportacion_por_estado(d),
        fig_siniestro_anual(d),
        fig_siniestro_estados(d),
        fig_siniestro_scatter(d),
    )


@app.callback(
    Output("graph-cult-trend",   "figure"),
    Output("graph-cult-estados", "figure"),
    Output("graph-cult-precio",  "figure"),
    Input("year-range",       "value"),
    Input("estado-dropdown",  "value"),
    Input("cultivo-dropdown", "value"),
)
def update_cultivo(year_range, estado, cultivo):
    yr_min, yr_max = year_range
    d = df.filter(pl.col("AÑO").is_between(yr_min, yr_max))
    if estado:
        d = d.filter(pl.col("ENTIDAD") == estado)
    if not cultivo:
        placeholder = _empty_fig("← Selecciona un cultivo en el menú de arriba")
        return placeholder, placeholder, placeholder
    return fig_cultivo_trend(d, cultivo), fig_cultivo_estados(d, cultivo), fig_cultivo_precio(d, cultivo)


if __name__ == "__main__":
    print("Dashboard disponible en http://localhost:8050")
    app.run(debug=False, port=8050)
