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

YEAR_MIN   = df["AÑO"].min()
YEAR_MAX   = df["AÑO"].max()
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
TAB_STYLE = {"backgroundColor": "#0F172A", "color": "#94A3B8", "borderTop": "none"}
TAB_SEL   = {
    "backgroundColor": "#1E293B", "color": "#F8FAFC",
    "borderTop": "2px solid #2E86AB", "fontWeight": "600",
}


def _fmt_pesos(v: float) -> str:
    if v >= 1e12:
        return f"${v / 1e12:.1f}T"
    if v >= 1e9:
        return f"${v / 1e9:.1f}B"
    return f"${v / 1e6:.1f}M"


def _delta_span(val_curr, val_prev, harm_when_up=False):
    if not val_prev or val_prev == 0:
        return ""
    pct = (val_curr - val_prev) / val_prev * 100
    worsened = (pct > 0) == harm_when_up
    color = "#E84855" if worsened else "#3BB273"
    arrow = "▲" if pct > 0 else "▼"
    return html.Span(
        f"{arrow} {abs(pct):.1f}% vs año anterior",
        style={"color": color, "fontSize": "0.78rem", "display": "block"},
    )


# ── Figure factories ──────────────────────────────────────────────────────────

def fig_produccion_anual(d: pl.DataFrame) -> go.Figure:
    yearly = (
        d.group_by("AÑO")
        .agg((pl.col("VALOR_PRODUCCION").sum() / 1e9).round(2).alias("valor_B"))
        .sort("AÑO")
    )
    yr_min_d = int(d["AÑO"].min())
    yr_max_d = int(d["AÑO"].max())
    val_min_row = yearly.filter(pl.col("AÑO") == yr_min_d)
    val_max_row = yearly.filter(pl.col("AÑO") == yr_max_d)
    val_min = float(val_min_row["valor_B"][0]) if not val_min_row.is_empty() else 1
    val_max = float(val_max_row["valor_B"][0]) if not val_max_row.is_empty() else 0
    ratio = val_max / val_min if val_min > 0 else 0

    fig = go.Figure(go.Scatter(
        x=yearly["AÑO"].to_list(), y=yearly["valor_B"].to_list(),
        mode="lines+markers",
        line=dict(color="#2E86AB", width=2),
        fill="tozeroy", fillcolor="rgba(46,134,171,0.15)",
        hovertemplate="Año: %{x}<br>Valor: $%{y:.1f}B<extra></extra>",
    ))
    if yr_min_d <= 2022 <= yr_max_d:
        fig.add_vline(
            x=2022, line_dash="dot", line_color="#94A3B8",
            annotation_text="2022: mayor salto anual (+$192B)",
            annotation_font_color="#94A3B8", annotation_position="top left",
            annotation_font_size=11,
        )
    fig.update_layout(
        title=dict(text=(
            f"<b>El valor agrícola creció {ratio:.0f}× entre {yr_min_d} y {yr_max_d}</b>"
            f"<br><sup style='color:#94A3B8'>Valor total de producción (miles de millones de pesos)</sup>"
        )),
        height=380,
        xaxis=dict(gridcolor="#334155", title="Año"),
        yaxis=dict(gridcolor="#334155", title="Miles de millones de pesos"),
        **{k: v for k, v in CHART_LAYOUT.items() if k not in ("xaxis", "yaxis")},
    )
    return fig


def fig_top_cultivos_valor(d: pl.DataFrame) -> go.Figure:
    ranked = (
        d.group_by("CULTIVO")
        .agg((pl.col("VALOR_PRODUCCION").sum() / 1e9).round(1).alias("valor_B"))
        .sort("valor_B", descending=True)
    )
    top10 = ranked.head(10).sort("valor_B")
    total = d["VALOR_PRODUCCION"].sum()
    leader = ranked["CULTIVO"][0]
    leader_share = float(ranked["valor_B"][0]) * 1e9 / total * 100

    cultivos = top10["CULTIVO"].to_list()
    valores  = top10["valor_B"].to_list()
    # Highlight aguacate as the growth story; fall back to leader if not in top 10
    focus_crop = "Aguacate" if "Aguacate" in cultivos else leader
    colors = ["#F4A261" if c == focus_crop else CONTEXT for c in cultivos]

    fig = go.Figure(go.Bar(
        x=valores, y=cultivos, orientation="h",
        marker_color=colors,
        text=[f"${v:.1f}B" for v in valores],
        textposition="outside",
        textfont=dict(color="#CBD5E1"),
        hovertemplate="%{y}<br>Valor: $%{x:.1f}B<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=(
            f"<b>{leader} concentra el {leader_share:.0f}% del valor — aguacate escaló del puesto 12 al 2 desde 2000</b>"
            f"<br><sup style='color:#94A3B8'>Top 10 cultivos por valor acumulado (miles de millones de pesos)</sup>"
        )),
        height=420,
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
    yr_min_d = int(d["AÑO"].min())
    yr_max_d = int(d["AÑO"].max())
    top3 = state_data.sort("valor", descending=True).head(3)
    top3_share = top3["valor"].sum() / state_data["valor"].sum() * 100
    t3 = top3["ENTIDAD"].to_list()
    while len(t3) < 3:
        t3.append(t3[-1] if t3 else "N/A")

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
        title=dict(text=(
            f"<b>{t3[0]}, {t3[1]} y {t3[2]} generan el {top3_share:.0f}% del valor — "
            f"distinto mapa en exportaciones</b>"
            f"<br><sup style='color:#94A3B8'>Valor total de producción por estado ({yr_min_d}–{yr_max_d})</sup>"
        )),
        height=580,
        margin=dict(l=0, r=0, t=50, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#CBD5E1",
    )
    return fig


def fig_top_estados(d: pl.DataFrame) -> go.Figure:
    ranked = (
        d.group_by("ENTIDAD")
        .agg((pl.col("VALOR_PRODUCCION").sum() / 1e9).round(1).alias("valor_B"))
        .sort("valor_B", descending=True)
    )
    top15 = ranked.head(15).sort("valor_B")
    total = d["VALOR_PRODUCCION"].sum()
    top3_share = ranked.head(3)["valor_B"].sum() * 1e9 / total * 100
    leader = ranked["ENTIDAD"][0]

    estados = top15["ENTIDAD"].to_list()
    valores  = top15["valor_B"].to_list()
    colors   = [FOCUS if e == leader else CONTEXT for e in estados]

    fig = go.Figure(go.Bar(
        x=valores, y=estados, orientation="h",
        marker_color=colors,
        text=[f"${v:.1f}B" for v in valores],
        textposition="outside",
        textfont=dict(color="#CBD5E1"),
        hovertemplate="%{y}<br>Valor: $%{x:.1f}B<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=(
            f"<b>Top 3 estados generan el {top3_share:.0f}% del valor — {leader} lidera</b>"
            f"<br><sup style='color:#94A3B8'>Top 15 estados por valor acumulado (miles de millones de pesos)</sup>"
        )),
        height=500,
        xaxis=dict(gridcolor="#334155"),
        yaxis=dict(gridcolor="rgba(0,0,0,0)"),
        **{k: v for k, v in CHART_LAYOUT.items() if k not in ("xaxis", "yaxis")},
    )
    return fig


def fig_rendimiento_estados(d: pl.DataFrame) -> go.Figure:
    ranked = (
        d.filter((pl.col("RENDIMIENTO") > 0) & (pl.col("UNIDAD_MEDIDA") == "Tonelada"))
        .group_by("ENTIDAD")
        .agg(pl.col("RENDIMIENTO").mean().round(2).alias("rend"))
        .sort("rend", descending=True)
    )
    if ranked.is_empty():
        return go.Figure()
    top15  = ranked.head(15).sort("rend")
    leader = ranked["ENTIDAD"][0]
    top_v  = float(ranked["rend"][0])

    estados = top15["ENTIDAD"].to_list()
    vals    = top15["rend"].to_list()
    colors  = [FOCUS if e == leader else CONTEXT for e in estados]

    fig = go.Figure(go.Bar(
        x=vals, y=estados, orientation="h",
        marker_color=colors,
        text=[f"{v:.1f}" for v in vals],
        textposition="outside",
        textfont=dict(color="#CBD5E1"),
        hovertemplate="%{y}<br>Rendimiento: %{x:.1f} ton/ha<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=(
            f"<b>{leader} lidera con {top_v:.1f} ton/ha de rendimiento promedio</b>"
            f"<br><sup style='color:#94A3B8'>Top 15 estados, solo cultivos medidos en toneladas</sup>"
        )),
        height=500,
        xaxis=dict(gridcolor="#334155"),
        yaxis=dict(gridcolor="rgba(0,0,0,0)"),
        **{k: v for k, v in CHART_LAYOUT.items() if k not in ("xaxis", "yaxis")},
    )
    return fig


def fig_treemap_cultivos(d: pl.DataFrame) -> go.Figure:
    all_c  = d.group_by("CULTIVO").agg(pl.col("VALOR_PRODUCCION").sum().alias("valor"))
    top20  = all_c.sort("valor", descending=True).head(20)
    share  = top20["valor"].sum() / all_c["valor"].sum() * 100
    n_rest = all_c.height - 20

    fig = px.treemap(
        top20, path=["CULTIVO"], values="valor",
        color="valor", color_continuous_scale="Blues",
    )
    fig.update_traces(textinfo="label+percent root")
    fig.update_layout(
        title=dict(text=(
            f"<b>20 cultivos explican el {share:.0f}% del valor — {n_rest} cultivos se reparten el resto</b>"
            f"<br><sup style='color:#94A3B8'>Distribución del valor de producción, top 20</sup>"
        )),
        height=520,
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#CBD5E1",
        margin=dict(l=0, r=0, t=60, b=0),
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
            (pl.col("TIPO_MERCADO") == "Exportación").mean().alias("pct_exp"),
        )
        .sort("volumen", descending=True)
        .head(30)
        .with_columns(
            pl.when(pl.col("pct_exp") > 0.3)
            .then(pl.lit("Export (>30% del valor)"))
            .otherwise(pl.lit("Mercado nacional"))
            .alias("orientacion")
        )
    )
    SCATTER_COLORS = {"Export (>30% del valor)": "#F4A261", "Mercado nacional": "#2E86AB"}
    fig = px.scatter(
        top30,
        x="rend", y="precio",
        size="volumen", color="orientacion",
        color_discrete_map=SCATTER_COLORS,
        hover_name="CULTIVO",
        labels={
            "rend":        "Rendimiento promedio (ton/ha)",
            "precio":      "Precio medio rural ($/ton)",
            "orientacion": "",
        },
        size_max=50,
    )
    med_rend  = float(top30["rend"].median())
    med_precio = float(top30["precio"].median())
    fig.add_vline(x=med_rend,   line_dash="dot", line_color="#334155")
    fig.add_hline(y=med_precio, line_dash="dot", line_color="#334155")
    fig.update_layout(
        title=dict(text=(
            "<b>Berries y hortalizas de exportación: alto precio, menor rendimiento por hectárea</b>"
            "<br><sup style='color:#94A3B8'>Rendimiento vs. precio por cultivo (burbuja = volumen producido), top 30</sup>"
        )),
        height=480,
        showlegend=True,
        legend=dict(orientation="h", y=-0.18, x=0),
        margin=dict(t=60, b=70, l=10, r=10),
        **CHART_LAYOUT,
    )
    return fig


def fig_evolucion_top_cultivos(d: pl.DataFrame) -> go.Figure:
    top5 = (
        d.group_by("CULTIVO")
        .agg(pl.col("VALOR_PRODUCCION").sum().alias("total"))
        .sort("total", descending=True)
        .head(5)["CULTIVO"].to_list()
    )
    trend = (
        d.filter(pl.col("CULTIVO").is_in(top5))
        .group_by("AÑO", "CULTIVO")
        .agg((pl.col("VALOR_PRODUCCION").sum() / 1e9).round(2).alias("valor_B"))
        .sort("AÑO")
    )
    # Focus aguacate in orange; other crops in distinguishable grays
    GRAY_PALETTE = ["#64748B", "#94A3B8", "#475569", "#334155"]
    non_av = [c for c in top5 if c != "Aguacate"]
    color_map = {c: GRAY_PALETTE[i % len(GRAY_PALETTE)] for i, c in enumerate(non_av)}
    color_map["Aguacate"] = "#F4A261"

    fig = px.line(
        trend, x="AÑO", y="valor_B", color="CULTIVO",
        color_discrete_map=color_map,
        labels={"valor_B": "Miles de millones de pesos", "AÑO": "Año", "CULTIVO": ""},
        markers=False,
    )
    # Annotate aguacate's last value
    if "Aguacate" in top5:
        av_last = trend.filter(pl.col("CULTIVO") == "Aguacate").sort("AÑO").tail(1)
        if not av_last.is_empty():
            fig.add_annotation(
                x=int(av_last["AÑO"][0]), y=float(av_last["valor_B"][0]),
                text=f"<b>Aguacate</b>  ${float(av_last['valor_B'][0]):.0f}B",
                font=dict(color="#F4A261", size=11),
                showarrow=True, arrowcolor="#94A3B8", ax=45, ay=-25,
                xanchor="left",
            )
    fig.update_layout(
        title=dict(text=(
            "<b>Aguacate escaló del puesto 12 al 2 desde 2000 — la mayor transformación del campo mexicano</b>"
            "<br><sup style='color:#94A3B8'>Evolución de los top 5 cultivos por valor total (miles de millones de pesos)</sup>"
        )),
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
    yr_last = int(d["AÑO"].max())
    last = (
        d.filter(pl.col("AÑO") == yr_last)
        .group_by("TIPO_TECNOLOGIA")
        .agg(pl.col("SUPERFICIE_SEMBRADA").sum().alias("sup"))
    )
    total_last = last["sup"].sum()
    protected  = last.filter(pl.col("TIPO_TECNOLOGIA") != "Cielo abierto")["sup"].sum()
    pct_prot   = protected / total_last * 100 if total_last > 0 else 0

    fig = px.bar(
        trend, x="AÑO", y="superficie", color="TIPO_TECNOLOGIA",
        barmode="stack",
        color_discrete_map=TECH_COLORS,
        labels={"superficie": "Superficie (ha)", "AÑO": "Año", "TIPO_TECNOLOGIA": ""},
    )
    fig.update_layout(
        title=dict(text=(
            f"<b>Superficie protegida ({pct_prot:.1f}% del total en {yr_last}) crece sostenidamente</b>"
            f"<br><sup style='color:#94A3B8'>Superficie sembrada por tipo de tecnología (hectáreas)</sup>"
        )),
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
    if rend.is_empty():
        return go.Figure()

    colors = [TECH_COLORS.get(t, "#2E86AB") for t in rend["TIPO_TECNOLOGIA"].to_list()]
    inv_row = rend.filter(pl.col("TIPO_TECNOLOGIA") == "Invernadero")
    ca_row  = rend.filter(pl.col("TIPO_TECNOLOGIA") == "Cielo abierto")
    inv_val = float(inv_row["rend"][0]) if not inv_row.is_empty() else None
    ca_val  = float(ca_row["rend"][0])  if not ca_row.is_empty()  else None
    ratio   = inv_val / ca_val if inv_val and ca_val and ca_val > 0 else None

    fig = go.Figure(go.Bar(
        x=rend["TIPO_TECNOLOGIA"].to_list(),
        y=rend["rend"].to_list(),
        marker_color=colors,
        text=[f"{v:.1f}" for v in rend["rend"].to_list()],
        textposition="outside",
        hovertemplate="%{x}<br>Rendimiento: %{y:.1f} ton/ha<extra></extra>",
    ))
    if ca_val:
        fig.add_hline(
            y=ca_val, line_dash="dash", line_color="#64748B",
            annotation_text=f"campo abierto ({ca_val:.1f} t/ha)",
            annotation_font_color="#94A3B8",
        )
    claim = (
        f"Invernadero: {ratio:.1f}× más rendimiento que campo abierto ({inv_val:.0f} vs {ca_val:.0f} ton/ha)"
        if ratio else "Rendimiento promedio por tipo de tecnología"
    )
    fig.update_layout(
        title=dict(text=(
            f"<b>{claim}</b>"
            f"<br><sup style='color:#94A3B8'>Rendimiento promedio (ton/ha), solo cultivos medidos en toneladas</sup>"
        )),
        height=360,
        xaxis=dict(gridcolor="rgba(0,0,0,0)"),
        yaxis=dict(gridcolor="#334155", title="Rendimiento promedio (ton/ha)"),
        **{k: v for k, v in CHART_LAYOUT.items() if k not in ("xaxis", "yaxis")},
    )
    return fig


def fig_produccion_tipo(d: pl.DataFrame) -> go.Figure:
    tipo  = d.group_by("TIPO_PRODUCCION").agg(pl.col("VALOR_PRODUCCION").sum().alias("valor"))
    total = tipo["valor"].sum()
    count_map = {r["TIPO_PRODUCCION"]: r["valor"] for r in tipo.iter_rows(named=True)}
    conv_pct  = count_map.get("Convencional", 0) / total * 100 if total > 0 else 0

    fig = go.Figure()
    for cat in ["Convencional", "Orgánico"]:
        if cat in count_map:
            pct = count_map[cat] / total * 100
            fig.add_trace(go.Bar(
                x=[pct], y=["Producción"], orientation="h", name=cat,
                marker_color=PROD_COLORS[cat],
                text=f"{pct:.1f}%",
                textposition="inside", insidetextanchor="middle",
                customdata=[count_map[cat]],
                hovertemplate=(
                    f"<b>{cat}</b>: %{{x:.1f}}%  (${count_map[cat]/1e9:.1f}B)<extra></extra>"
                ),
            ))
    fig.update_layout(
        barmode="stack",
        title=dict(text=(
            f"<b>El {conv_pct:.0f}% del valor agrícola es producción convencional — orgánico apenas despunta</b>"
            f"<br><sup style='color:#94A3B8'>Valor de producción: convencional vs. orgánico</sup>"
        )),
        height=200,
        xaxis=dict(range=[0, 100], visible=False, gridcolor="#334155"),
        yaxis=dict(gridcolor="rgba(0,0,0,0)"),
        legend=dict(orientation="h", y=-0.5, x=0),
        margin=dict(t=70, b=60, l=10, r=10),
        **{k: v for k, v in CHART_LAYOUT.items() if k not in ("xaxis", "yaxis")},
    )
    return fig


def fig_mercado_trend(d: pl.DataFrame) -> go.Figure:
    trend = (
        d.group_by("AÑO", "TIPO_MERCADO")
        .agg((pl.col("VALOR_PRODUCCION").sum() / 1e9).round(2).alias("valor_B"))
        .sort("AÑO")
    )
    yr_total = d.group_by("AÑO").agg(pl.col("VALOR_PRODUCCION").sum().alias("total"))
    mkt_pct = (
        d.group_by("AÑO", "TIPO_MERCADO")
        .agg(pl.col("VALOR_PRODUCCION").sum().alias("val"))
        .join(yr_total, on="AÑO")
        .with_columns((pl.col("val") / pl.col("total") * 100).round(1).alias("pct"))
        .filter(pl.col("TIPO_MERCADO") == "Exportación")
        .sort("AÑO")
    )
    yr_first, yr_last = int(d["AÑO"].min()), int(d["AÑO"].max())
    r_first = mkt_pct.filter(pl.col("AÑO") == yr_first)
    r_last  = mkt_pct.filter(pl.col("AÑO") == yr_last)
    pct_f   = float(r_first["pct"][0]) if not r_first.is_empty() else None
    pct_l   = float(r_last["pct"][0])  if not r_last.is_empty()  else None
    claim   = (
        f"Exportaciones: del {pct_f:.1f}% al {pct_l:.1f}% del valor total entre {yr_first} y {yr_last}"
        if pct_f is not None and pct_l is not None
        else "Valor de producción: Nacional vs. Exportación"
    )
    fig = px.area(
        trend, x="AÑO", y="valor_B", color="TIPO_MERCADO",
        color_discrete_map=MARKET_COLORS,
        labels={"valor_B": "Miles de millones de pesos", "AÑO": "Año", "TIPO_MERCADO": ""},
    )
    fig.update_layout(
        title=dict(text=(
            f"<b>{claim}</b>"
            f"<br><sup style='color:#94A3B8'>Valor de producción por destino de mercado (miles de millones de pesos)</sup>"
        )),
        height=420,
        legend=dict(orientation="h", y=-0.15, title=""),
        **CHART_LAYOUT,
    )
    return fig


def fig_top_cultivos_exportacion(d: pl.DataFrame) -> go.Figure:
    ranked = (
        d.filter(pl.col("TIPO_MERCADO") == "Exportación")
        .group_by("CULTIVO")
        .agg((pl.col("VALOR_PRODUCCION").sum() / 1e9).round(2).alias("valor_B"))
        .sort("valor_B", descending=True)
    )
    top10   = ranked.head(10).sort("valor_B")
    leader  = ranked["CULTIVO"][0] if not ranked.is_empty() else ""
    cultivos = top10["CULTIVO"].to_list()
    valores  = top10["valor_B"].to_list()
    colors   = ["#F4A261" if c == leader else CONTEXT for c in cultivos]

    fig = go.Figure(go.Bar(
        x=valores, y=cultivos, orientation="h",
        marker_color=colors,
        text=[f"${v:.1f}B" for v in valores],
        textposition="outside",
        textfont=dict(color="#CBD5E1"),
        hovertemplate="%{y}<br>Valor exportación: $%{x:.1f}B<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=(
            f"<b>{leader} encabeza las exportaciones — berries y hortalizas concentran el mercado exterior</b>"
            f"<br><sup style='color:#94A3B8'>Top 10 cultivos de exportación por valor acumulado</sup>"
        )),
        height=420,
        xaxis=dict(gridcolor="#334155"),
        yaxis=dict(gridcolor="rgba(0,0,0,0)"),
        **{k: v for k, v in CHART_LAYOUT.items() if k not in ("xaxis", "yaxis")},
    )
    return fig


def fig_exportacion_por_estado(d: pl.DataFrame) -> go.Figure:
    # Rank by export value, not total — shows the real export leaders
    top_exp_states = (
        d.filter(pl.col("TIPO_MERCADO") == "Exportación")
        .group_by("ENTIDAD")
        .agg(pl.col("VALOR_PRODUCCION").sum().alias("exp_total"))
        .sort("exp_total", descending=True)
        .head(10)["ENTIDAD"].to_list()
    )
    if not top_exp_states:
        return go.Figure()
    breakdown = (
        d.filter(pl.col("ENTIDAD").is_in(top_exp_states))
        .group_by("ENTIDAD", "TIPO_MERCADO")
        .agg((pl.col("VALOR_PRODUCCION").sum() / 1e9).round(2).alias("valor_B"))
    )
    exp_order = (
        breakdown.filter(pl.col("TIPO_MERCADO") == "Exportación")
        .sort("valor_B", descending=True)["ENTIDAD"].to_list()
    )
    total_exp = d.filter(pl.col("TIPO_MERCADO") == "Exportación")["VALOR_PRODUCCION"].sum()
    if total_exp > 0 and len(exp_order) >= 2:
        top2_exp = d.filter(
            (pl.col("TIPO_MERCADO") == "Exportación") & (pl.col("ENTIDAD").is_in(exp_order[:2]))
        )["VALOR_PRODUCCION"].sum()
        top2_share = top2_exp / total_exp * 100
        claim = f"{exp_order[0]} y {exp_order[1]} concentran el {top2_share:.0f}% de las exportaciones — los grandes productores del centro exportan poco"
    else:
        claim = "Exportaciones agrícolas por estado"

    fig = px.bar(
        breakdown, x="ENTIDAD", y="valor_B", color="TIPO_MERCADO",
        barmode="stack",
        color_discrete_map=MARKET_COLORS,
        labels={"valor_B": "Miles de millones de pesos", "ENTIDAD": "", "TIPO_MERCADO": ""},
        category_orders={"ENTIDAD": exp_order},
    )
    fig.update_layout(
        title=dict(text=(
            f"<b>{claim}</b>"
            f"<br><sup style='color:#94A3B8'>Top 10 estados exportadores (ordenados por valor de exportación)</sup>"
        )),
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
    peak_row  = yearly.sort("tasa", descending=True).head(1)
    peak_yr   = int(peak_row["AÑO"][0])   if not peak_row.is_empty() else None
    peak_val  = float(peak_row["tasa"][0]) if not peak_row.is_empty() else None

    fig = go.Figure(go.Scatter(
        x=yearly["AÑO"].to_list(), y=yearly["tasa"].to_list(),
        mode="lines+markers",
        line=dict(color="#E84855", width=2),
        fill="tozeroy", fillcolor="rgba(232,72,85,0.12)",
        hovertemplate="Año: %{x}<br>Tasa de siniestro: %{y:.2f}%<extra></extra>",
    ))
    if peak_yr and peak_val:
        fig.add_annotation(
            x=peak_yr, y=peak_val,
            text=f"<b>{peak_yr}</b>: {peak_val:.1f}%",
            font=dict(color="#E84855", size=11),
            showarrow=True, arrowcolor="#94A3B8", ax=30, ay=-30,
        )
    claim = (
        f"Tasa de siniestro: pico de {peak_val:.1f}% en {peak_yr} — el nivel más alto del período"
        if peak_yr else "Tasa de siniestro por año"
    )
    fig.update_layout(
        title=dict(text=(
            f"<b>{claim}</b>"
            f"<br><sup style='color:#94A3B8'>Superficie siniestrada / sembrada × 100 (%)</sup>"
        )),
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
    if state_risk.is_empty():
        return go.Figure()
    sem_total = d["SUPERFICIE_SEMBRADA"].sum()
    sin_total = d["SUPERFICIE_SINIESTRADA"].sum()
    nat_avg   = sin_total / sem_total * 100 if sem_total > 0 else 0
    worst     = state_risk.sort("tasa", descending=True).head(1)
    w_state   = worst["ENTIDAD"][0]
    w_tasa    = float(worst["tasa"][0])

    pct_vals = state_risk["tasa"].to_list()
    colors   = ["#E84855" if v > nat_avg else "#F4A261" for v in pct_vals]

    fig = go.Figure(go.Bar(
        x=pct_vals, y=state_risk["ENTIDAD"].to_list(), orientation="h",
        marker_color=colors,
        text=[f"{v:.1f}%" for v in pct_vals],
        textposition="outside",
        hovertemplate="%{y}<br>Tasa de siniestro: %{x:.2f}%<extra></extra>",
    ))
    fig.add_vline(
        x=nat_avg, line_dash="dash", line_color="#64748B",
        annotation_text=f"media nacional ({nat_avg:.1f}%)",
        annotation_font_color="#94A3B8",
    )
    fig.update_layout(
        title=dict(text=(
            f"<b>{w_state} encabeza el riesgo con {w_tasa:.1f}% de superficie siniestrada</b>"
            f"<br><sup style='color:#94A3B8'>Top 15 estados — rojo = sobre la media nacional</sup>"
        )),
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
        labels={
            "sembrada_M": "Superficie sembrada (millones de ha)",
            "tasa":       "Tasa de siniestro (%)",
            "valor":      "Valor producción",
        },
        size_max=50,
    )
    fig.update_layout(
        title=dict(text=(
            "<b>Grandes productores no siempre tienen mayor riesgo — el siniestro no sigue el valor</b>"
            "<br><sup style='color:#94A3B8'>Superficie sembrada vs. tasa de siniestro (burbuja = valor producción)</sup>"
        )),
        height=480, showlegend=False, **CHART_LAYOUT,
    )
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
        .head(8)["ENTIDAD"].to_list()
    )
    trend = (
        crop.filter(pl.col("ENTIDAD").is_in(top_states))
        .group_by("AÑO", "ENTIDAD")
        .agg((pl.col("VALOR_PRODUCCION").sum() / 1e9).round(3).alias("valor_B"))
        .sort("AÑO")
    )
    fig = px.area(
        trend, x="AÑO", y="valor_B", color="ENTIDAD",
        labels={"valor_B": "Miles de millones de pesos", "AÑO": "Año", "ENTIDAD": ""},
    )
    fig.update_layout(
        title=dict(text=(
            f"<b>Evolución de {cultivo} — top 8 estados productores</b>"
            f"<br><sup style='color:#94A3B8'>Valor de producción (miles de millones de pesos)</sup>"
        )),
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
    leader = crop.sort("valor_B", descending=True)["ENTIDAD"][0]
    estados = crop["ENTIDAD"].to_list()
    colors  = [FOCUS if e == leader else CONTEXT for e in estados]
    n = len(crop)

    fig = go.Figure(go.Bar(
        x=crop["valor_B"].to_list(), y=estados, orientation="h",
        marker_color=colors,
        text=[f"${v:.2f}B" for v in crop["valor_B"].to_list()],
        textposition="outside",
        textfont=dict(color="#CBD5E1"),
        customdata=crop[["superficie", "desde", "hasta"]].to_numpy(),
        hovertemplate=(
            "<b>%{y}</b><br>Valor: $%{x:.3f}B<br>"
            "Superficie: %{customdata[0]:,.0f} ha<br>"
            "Cultivado: %{customdata[1]}–%{customdata[2]}<extra></extra>"
        ),
    ))
    fig.update_layout(
        title=dict(text=(
            f"<b>{leader} domina la producción de {cultivo}</b>"
            f"<br><sup style='color:#94A3B8'>Estados productores, valor acumulado</sup>"
        )),
        height=max(300, n * 28 + 80),
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
    yr_first = int(precio["AÑO"][0])
    yr_last  = int(precio["AÑO"][-1])
    p_first  = float(precio["precio"][0])
    p_last   = float(precio["precio"][-1])
    ratio    = p_last / p_first if p_first > 0 else 0

    fig = go.Figure(go.Scatter(
        x=precio["AÑO"].to_list(), y=precio["precio"].to_list(),
        mode="lines+markers",
        line=dict(color="#F4A261", width=2),
        fill="tozeroy", fillcolor="rgba(244,162,97,0.12)",
        hovertemplate="Año: %{x}<br>Precio: $%{y:,.0f}/ton<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=(
            f"<b>El precio de {cultivo} creció {ratio:.1f}× entre {yr_first} y {yr_last}</b>"
            f"<br><sup style='color:#94A3B8'>Precio medio rural ($/ton)</sup>"
        )),
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

SLIDER_MARKS = {y: str(y) for y in range(YEAR_MIN, YEAR_MAX + 1, 5)}
SLIDER_MARKS[YEAR_MIN] = str(YEAR_MIN)
SLIDER_MARKS[YEAR_MAX] = str(YEAR_MAX)


def kpi_card(title, value_id, sub_id, delta_id):
    return html.Div([
        html.P(title, style={"color": "#94A3B8", "fontSize": "0.8rem", "marginBottom": "4px"}),
        html.H3(id=value_id, style={"color": "#F8FAFC", "fontWeight": "700", "margin": "0"}),
        html.Div(id=delta_id, style={"minHeight": "18px", "marginTop": "2px"}),
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
        ], className="mb-3"),

        dbc.Alert(
            [
                html.Strong("⚠ Nota metodológica: "),
                "Los años 2020–2021 contienen solo 78 cultivos registrados (vs. 300+ en años normales), "
                "probablemente por reducción en cobertura de datos durante la pandemia. "
                "Las cifras de esos años son parciales — interpretar con precaución.",
            ],
            id="artifact-alert",
            color="warning",
            style={"fontSize": "0.83rem", "padding": "8px 14px", "display": "none"},
            className="mb-3",
        ),

        dbc.Row([
            dbc.Col(kpi_card("Valor total de producción", "kpi-valor-val", "kpi-valor-sub", "kpi-valor-delta"), width=3),
            dbc.Col(kpi_card("Volumen total producido",   "kpi-vol-val",   "kpi-vol-sub",   "kpi-vol-delta"),   width=3),
            dbc.Col(kpi_card("Cultivos activos",          "kpi-cult-val",  "kpi-cult-sub",  "kpi-cult-delta"),  width=3),
            dbc.Col(kpi_card("Rendimiento promedio",      "kpi-rend-val",  "kpi-rend-sub",  "kpi-rend-delta"),  width=3),
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
                    clearable=True, searchable=True,
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
                            clearable=True, searchable=True,
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
                    html.Strong("Michoacán lidera la producción — Sonora lidera las exportaciones", style={"color": "#F8FAFC"}),
                    html.P(
                        "Michoacán, Jalisco y Sinaloa generan el 30% del valor agrícola nacional. "
                        "Pero el mapa de exportaciones es otro: Sonora ($137B) y Baja California ($123B) "
                        "concentran el 60% de lo que México vende al exterior.",
                        style={"color": "#94A3B8", "fontSize": "0.83rem", "marginTop": "4px"},
                    ),
                ], style={**CARD_STYLE, "textAlign": "left"}), width=4),
                dbc.Col(html.Div([
                    html.Strong("Aguacate: de la posición 12 a la 2 en valor desde 2000", style={"color": "#F8FAFC"}),
                    html.P(
                        "En 2000, el aguacate era el 12.° cultivo por valor ($4.2B). "
                        "En 2024 alcanzó $57B — un crecimiento de 13× — y hoy es el segundo cultivo "
                        "más valioso de México, solo detrás del maíz.",
                        style={"color": "#94A3B8", "fontSize": "0.83rem", "marginTop": "4px"},
                    ),
                ], style={**CARD_STYLE, "textAlign": "left"}), width=4),
                dbc.Col(html.Div([
                    html.Strong("Invernadero: 2.5× más rendimiento en <0.1% del territorio", style={"color": "#F8FAFC"}),
                    html.P(
                        "Los cultivos bajo invernadero promedian 126 ton/ha vs. 49 ton/ha a cielo abierto. "
                        "Representan menos del 0.1% de la superficie sembrada pero concentran "
                        "una fracción creciente del valor de exportación.",
                        style={"color": "#94A3B8", "fontSize": "0.83rem", "marginTop": "4px"},
                    ),
                ], style={**CARD_STYLE, "textAlign": "left"}), width=4),
            ], className="g-3"),
        ], className="mt-4"),
    ]
)


# ── Single callback ──────────────────────────────────────────────────────────
@app.callback(
    Output("kpi-valor-val",   "children"),
    Output("kpi-valor-sub",   "children"),
    Output("kpi-valor-delta", "children"),
    Output("kpi-vol-val",     "children"),
    Output("kpi-vol-sub",     "children"),
    Output("kpi-vol-delta",   "children"),
    Output("kpi-cult-val",    "children"),
    Output("kpi-cult-sub",    "children"),
    Output("kpi-cult-delta",  "children"),
    Output("kpi-rend-val",    "children"),
    Output("kpi-rend-sub",    "children"),
    Output("kpi-rend-delta",  "children"),
    Output("estado-badge",    "children"),
    Output("estado-badge",    "style"),
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
    Input("year-range",      "value"),
    Input("estado-dropdown", "value"),
)
def update_all(year_range, estado):
    yr_min, yr_max = year_range
    d = df.filter(pl.col("AÑO").is_between(yr_min, yr_max))
    if estado:
        d = d.filter(pl.col("ENTIDAD") == estado)

    valor, volumen, n_cult, rend = compute_kpis(d)

    # Year-over-year deltas (last year in range vs prior year)
    yr_last = int(d["AÑO"].max()) if not d.is_empty() else yr_max
    d_last  = d.filter(pl.col("AÑO") == yr_last)
    d_prev  = d.filter(pl.col("AÑO") == yr_last - 1)

    val_last  = d_last["VALOR_PRODUCCION"].sum()
    val_prev  = d_prev["VALOR_PRODUCCION"].sum() if not d_prev.is_empty() else None
    vol_last  = d_last["VOLUMEN_PRODUCCION"].sum()
    vol_prev  = d_prev["VOLUMEN_PRODUCCION"].sum() if not d_prev.is_empty() else None
    cult_last = d_last["CULTIVO"].n_unique()
    cult_prev = d_prev["CULTIVO"].n_unique() if not d_prev.is_empty() else None
    rend_last = (
        d_last.filter((pl.col("RENDIMIENTO") > 0) & (pl.col("UNIDAD_MEDIDA") == "Tonelada"))
        ["RENDIMIENTO"].mean()
    )
    rend_prev_val = (
        d_prev.filter((pl.col("RENDIMIENTO") > 0) & (pl.col("UNIDAD_MEDIDA") == "Tonelada"))
        ["RENDIMIENTO"].mean()
        if not d_prev.is_empty() else None
    )

    badge_style = {
        "display": "inline-block", "background": "#2E86AB", "color": "#fff",
        "borderRadius": "4px", "fontSize": "0.72rem", "padding": "1px 7px",
        "marginLeft": "8px", "verticalAlign": "middle",
    } if estado else {"display": "none"}

    return (
        _fmt_pesos(valor),
        f"{yr_min}–{yr_max}",
        _delta_span(val_last, val_prev, harm_when_up=False),
        f"{volumen / 1e6:.1f}M ton",
        "volumen producido total",
        _delta_span(vol_last, vol_prev, harm_when_up=False),
        f"{n_cult:,}",
        "cultivos únicos en el período",
        _delta_span(cult_last, cult_prev, harm_when_up=False),
        f"{rend:.1f} ton/ha",
        "rendimiento promedio",
        _delta_span(rend_last or 0, rend_prev_val, harm_when_up=False),
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
    Output("artifact-alert", "style"),
    Input("year-range", "value"),
)
def toggle_artifact_alert(year_range):
    yr_min, yr_max = year_range
    base = {"fontSize": "0.83rem", "padding": "8px 14px"}
    if yr_min <= 2021 and yr_max >= 2020:
        return {**base, "display": "block"}
    return {**base, "display": "none"}


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
