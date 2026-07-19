import json

import dash
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
from dash import Input, Output, dcc, html

# ── Data ──────────────────────────────────────────────────────────────────────
df = pl.read_parquet("dashboard_data/produccion_minera_clean.parquet")

# INEGI EIMM — volumen físico municipal (complementa el valor monetario SGM arriba).
eimm_nacional = pl.read_parquet("dashboard_data/eimm_nacional_anual.parquet")
eimm_estado = pl.read_parquet("dashboard_data/eimm_estado_producto.parquet")
eimm_muni_activos = pl.read_parquet("dashboard_data/eimm_municipios_activos.parquet")
eimm_top_muni = pl.read_parquet("dashboard_data/eimm_top_municipio_2025.parquet")

# SGM Anuario — comercio exterior minero (exportaciones/importaciones, 2023-2024).
comercio_producto = pl.read_parquet("dashboard_data/comercio_producto.parquet")
comercio_pais = pl.read_parquet("dashboard_data/comercio_pais.parquet")

YEAR_MIN = min(int(df["año"].min()), int(eimm_nacional["año"].min()))
YEAR_MAX = max(int(df["año"].max()), int(eimm_nacional["año"].max()))
ALL_STATES = sorted(df["estado"].unique().to_list())

with open("data/mexico_states.geojson") as _f:
    MEXICO_GEO = json.load(_f)

# Comercio exterior solo cubre 2023-2024 (sin serie larga) -> KPIs estáticos, sin callback.
COMERCIO_EXP_2023 = float(comercio_producto.filter(pl.col("flujo") == "Exportación")["valor_2023"].sum())
COMERCIO_EXP_2024 = float(comercio_producto.filter(pl.col("flujo") == "Exportación")["valor_2024"].sum())
COMERCIO_IMP_2023 = float(comercio_producto.filter(pl.col("flujo") == "Importación")["valor_2023"].sum())
COMERCIO_IMP_2024 = float(comercio_producto.filter(pl.col("flujo") == "Importación")["valor_2024"].sum())
COMERCIO_BALANCE_2023 = COMERCIO_EXP_2023 - COMERCIO_IMP_2023
COMERCIO_BALANCE_2024 = COMERCIO_EXP_2024 - COMERCIO_IMP_2024
_comercio_leader_2023 = (
    comercio_pais.filter(pl.col("flujo") == "Exportación").sort("valor_2023", descending=True).row(0, named=True)
)
_comercio_leader_2024 = (
    comercio_pais.filter(pl.col("flujo") == "Exportación").sort("valor_2024", descending=True).row(0, named=True)
)
COMERCIO_LEADER_PAIS = _comercio_leader_2024["pais"]
COMERCIO_LEADER_PCT_2023 = _comercio_leader_2023["valor_2023"] / COMERCIO_EXP_2023 * 100
COMERCIO_LEADER_PCT_2024 = _comercio_leader_2024["valor_2024"] / COMERCIO_EXP_2024 * 100

PERIODS = [(2000, 2009, "2000–09"), (2010, 2019, "2010–19"), (2020, 2024, "2020–24")]
LEADERSHIP_STATES = ["Sonora", "Michoacán", "Zacatecas"]
CAT_COLORS = {"Metálicos": "#2E86AB", "No metálicos": "#F4A261"}
LEADERSHIP_COLORS = {"Sonora": "#2E86AB", "Michoacán": "#E84855", "Zacatecas": "#94A3B8"}

VOLUMEN_STYLE = {
    "Oro": ("#F4A261", "rgba(244,162,97,0.15)", "Oro", "Kilogramos"),
    "Plata": ("#9B59B6", "rgba(155,89,182,0.15)", "Plata", "Kilogramos"),
    "Fierro en Extraccion": ("#2E86AB", "rgba(46,134,171,0.15)", "Fierro (extracción)", "Toneladas"),
}
CONCENTRACION_PRODUCTOS = ["Cobre", "Fluorita", "Coque", "Barita", "Plata", "Plomo", "Zinc"]
STATE_COLOR_PALETTE = ["#2E86AB", "#E84855", "#3BB273", "#F4A261", "#9B59B6", "#94A3B8"]
FLUJO_PLURAL = {"Exportación": "exportaciones", "Importación": "importaciones"}

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
TAB_SEL = {
    "backgroundColor": "#1E293B", "color": "#F8FAFC",
    "borderTop": "2px solid #2E86AB", "fontWeight": "600",
}


def _fmt_pesos(v: float) -> str:
    if v >= 1e12:
        return f"${v / 1e12:.1f}T"
    if v >= 1e9:
        return f"${v / 1e9:.1f}B"
    return f"${v / 1e6:.1f}M"


def _fmt_usd(v: float) -> str:
    sign = "-" if v < 0 else ""
    v = abs(v)
    if v >= 1e9:
        return f"{sign}${v / 1e9:.2f}B"
    return f"{sign}${v / 1e6:.1f}M"


def _delta_span(val_curr, val_prev, harm_when_up=False, label="vs año anterior"):
    if not val_prev:
        return ""
    pct = (val_curr - val_prev) / val_prev * 100
    worsened = (pct > 0) == harm_when_up
    color = "#E84855" if worsened else "#3BB273"
    arrow = "▲" if pct > 0 else "▼"
    return html.Span(
        f"{arrow} {abs(pct):.1f}% {label}",
        style={"color": color, "fontSize": "0.78rem", "display": "block"},
    )


def _empty_fig(msg: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=msg, showarrow=False, font=dict(color="#64748B", size=14))
    fig.update_layout(
        height=380, xaxis=dict(visible=False), yaxis=dict(visible=False),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def _strip_axes(**overrides) -> dict:
    return {**{k: v for k, v in CHART_LAYOUT.items() if k not in ("xaxis", "yaxis")}, **overrides}


# ── Figure factories ────────────────────────────────────────────────────────


def fig_trend_nacional(d: pl.DataFrame) -> go.Figure:
    clean = d.filter(~pl.col("is_agregados"))
    yearly = clean.group_by("año").agg(pl.col("valor_pesos").sum()).sort("año")
    if yearly.is_empty():
        return _empty_fig("Sin datos en el período seleccionado")

    yr0, yr1 = int(yearly["año"][0]), int(yearly["año"][-1])
    v0 = float(yearly["valor_pesos"][0])

    # 2023–2024 are provisional (thin record counts) and understate the true total —
    # anchor the growth claim on the last non-provisional year in range, not the raw
    # last year, so a partial final year doesn't fake a decline.
    stable = yearly.filter(pl.col("año") < 2023)
    yr_stable = int(stable["año"][-1]) if not stable.is_empty() else yr1
    v_stable = float(stable["valor_pesos"][-1]) if not stable.is_empty() else float(yearly["valor_pesos"][-1])
    ratio = v_stable / v0 if v0 > 0 else 0

    fig = go.Figure(go.Scatter(
        x=yearly["año"].to_list(), y=(yearly["valor_pesos"] / 1e9).to_list(),
        mode="lines+markers",
        line=dict(color="#2E86AB", width=2),
        fill="tozeroy", fillcolor="rgba(46,134,171,0.15)",
        hovertemplate="Año: %{x}<br>Valor: $%{y:,.0f} mil millones<extra></extra>",
    ))

    provisional_years = [y for y in (2023, 2024) if yr0 <= y <= yr1]
    if provisional_years:
        fig.add_vrect(
            x0=min(provisional_years) - 0.5, x1=yr1 + 0.5,
            fillcolor="rgba(244,162,97,0.12)", line_width=0,
            annotation_text="datos provisionales", annotation_font_color="#94A3B8",
            annotation_position="top left",
        )

    fig.update_layout(
        title=dict(text=(
            f"<b>El valor minero limpio creció {ratio:.0f}× entre {yr0} y {yr_stable}</b>"
            "<br><sup style='color:#94A3B8'>Valor total anual, pesos nominales (sin Agregados pétreos)"
            + (f"; {'/'.join(str(y) for y in provisional_years)} preliminar" if provisional_years else "")
            + "</sup>"
        )),
        height=420,
        xaxis=dict(gridcolor="#334155", title="Año"),
        yaxis=dict(gridcolor="#334155", title="Miles de millones de pesos"),
        **_strip_axes(),
    )
    return fig


def fig_choropleth_estado(d: pl.DataFrame) -> go.Figure:
    clean = d.filter(~pl.col("is_agregados"))
    state_data = clean.group_by("estado").agg(pl.col("valor_pesos").sum().alias("valor"))
    if state_data.is_empty():
        return _empty_fig("Sin datos en el período seleccionado")
    state_data = state_data.with_columns((pl.col("valor") / 1e9).round(1).alias("valor_B"))

    leader = state_data.sort("valor", descending=True).row(0, named=True)

    fig = px.choropleth_map(
        state_data, geojson=MEXICO_GEO, locations="estado", color="valor",
        featureidkey="properties.name", color_continuous_scale="YlOrRd",
        zoom=4.0, center={"lat": 23.6, "lon": -102.5}, opacity=0.85,
        hover_name="estado", custom_data=["valor_B"], map_style="carto-darkmatter",
    )
    fig.update_traces(
        hovertemplate="<b>%{hovertext}</b><br>Valor: $%{customdata[0]:,.0f} mil millones<extra></extra>"
    )
    fig.update_layout(
        title=dict(text=(
            f"<b>{leader['estado']} concentra el mayor valor minero del país</b>"
            "<br><sup style='color:#94A3B8'>Valor acumulado por estado, pesos nominales, sin Agregados pétreos</sup>"
        )),
        height=580, paper_bgcolor="rgba(0,0,0,0)", font_color="#CBD5E1",
        margin=dict(t=60, b=10, l=10, r=10),
        coloraxis_colorbar=dict(title="Miles de<br>millones $"),
    )
    return fig


def fig_ranking_bar(d: pl.DataFrame) -> go.Figure:
    clean = d.filter(~pl.col("is_agregados"))
    rank = (
        clean.group_by("estado").agg(pl.col("valor_pesos").sum().alias("valor"))
        .sort("valor", descending=True)
    )
    if rank.is_empty():
        return _empty_fig("Sin datos en el período seleccionado")

    total = rank["valor"].sum()
    top5_pct = float(rank.head(5)["valor"].sum() / total * 100) if total else 0
    top10 = rank.head(10).sort("valor")
    leader = top10["estado"][-1]
    colors = [FOCUS if e == leader else CONTEXT for e in top10["estado"].to_list()]

    max_val = float(top10["valor"].max()) / 1e9

    fig = go.Figure(go.Bar(
        x=(top10["valor"] / 1e9).to_list(), y=top10["estado"].to_list(), orientation="h",
        marker_color=colors,
        text=[f"${v / 1e9:,.0f}B" for v in top10["valor"].to_list()],
        textposition="outside", textfont=dict(color="#CBD5E1"), cliponaxis=False,
        hovertemplate="<b>%{y}</b><br>$%{x:,.0f} mil millones<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=(
            f"<b>Los 5 estados líderes concentran {top5_pct:.0f}% del valor minero</b>"
            f"<br><sup style='color:#94A3B8'>Top 10 estados, {leader} a la cabeza</sup>"
        )),
        height=max(320, 10 * 28 + 80),
        xaxis=dict(gridcolor="#334155", title="Miles de millones de pesos", range=[0, max_val * 1.2]),
        yaxis=dict(gridcolor="rgba(0,0,0,0)"),
        **_strip_axes(),
    )
    return fig


def fig_detalle_estado(d: pl.DataFrame, estado: str) -> go.Figure:
    sub = d.filter((pl.col("estado") == estado) & (~pl.col("is_agregados")))
    if sub.is_empty():
        return _empty_fig(f"Sin datos para {estado} en el período seleccionado")

    yearly = sub.group_by("año").agg(pl.col("valor_pesos").sum()).sort("año")
    comp = sub.group_by("categoria").agg(pl.col("valor_pesos").sum())
    tot = comp["valor_pesos"].sum()
    met = comp.filter(pl.col("categoria") == "Metálicos")["valor_pesos"].sum()
    met_pct = float(met / tot * 100) if tot else 0

    fig = go.Figure(go.Scatter(
        x=yearly["año"].to_list(), y=(yearly["valor_pesos"] / 1e9).to_list(),
        mode="lines+markers",
        line=dict(color=FOCUS, width=2),
        fill="tozeroy", fillcolor="rgba(46,134,171,0.15)",
        hovertemplate="Año: %{x}<br>$%{y:,.0f} mil millones<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=(
            f"<b>{estado}: valor minero por año</b>"
            f"<br><sup style='color:#94A3B8'>{met_pct:.0f}% metálico · {100 - met_pct:.0f}% no metálico"
            f", {int(yearly['año'][0])}–{int(yearly['año'][-1])}</sup>"
        )),
        height=380,
        xaxis=dict(gridcolor="#334155", title="Año"),
        yaxis=dict(gridcolor="#334155", title="Miles de millones de pesos"),
        **_strip_axes(),
    )
    return fig


def fig_composicion_stacked(d: pl.DataFrame) -> go.Figure:
    clean = d.filter(~pl.col("is_agregados"))
    periods_present = [
        (lo, hi, label) for lo, hi, label in PERIODS
        if not clean.filter(pl.col("año").is_between(lo, hi)).is_empty()
    ]
    if not periods_present:
        return _empty_fig("Sin datos en el período seleccionado")

    fig = go.Figure()
    met_pcts = []
    for cat, color in CAT_COLORS.items():
        pcts, valores = [], []
        for lo, hi, _label in periods_present:
            sub = clean.filter(pl.col("año").is_between(lo, hi))
            tot = sub["valor_pesos"].sum()
            v = sub.filter(pl.col("categoria") == cat)["valor_pesos"].sum()
            pcts.append(v / tot * 100 if tot else 0)
            valores.append(v)
        if cat == "Metálicos":
            met_pcts = pcts
        fig.add_trace(go.Bar(
            x=pcts, y=[label for _, _, label in periods_present], orientation="h",
            name=cat, marker_color=color,
            text=[f"{p:.0f}%" for p in pcts], textposition="inside", insidetextanchor="middle",
            customdata=valores,
            hovertemplate=f"<b>{cat}</b>: %{{x:.1f}}%  ($%{{customdata:,.0f}})<extra></extra>",
        ))

    first_label, last_label = periods_present[0][2], periods_present[-1][2]
    first_pct, last_pct = met_pcts[0], met_pcts[-1]
    if len(periods_present) == 1:
        leader, pct = ("metálicos", last_pct) if last_pct >= 50 else ("no metálicos", 100 - last_pct)
        title_text = f"En {last_label}, los {leader} lideran el valor minero ({pct:.0f}%)"
    elif first_pct >= 50 > last_pct:
        title_text = (
            f"Los no metálicos superaron a los metálicos en {last_label} "
            f"({100 - last_pct:.0f}% vs {last_pct:.0f}%), tras dominar en {first_label} ({first_pct:.0f}%)"
        )
    elif first_pct < 50 <= last_pct:
        title_text = f"Los metálicos pasaron a liderar el valor minero en {last_label} ({last_pct:.0f}%)"
    else:
        leader, pct = ("metálicos", last_pct) if last_pct >= 50 else ("no metálicos", 100 - last_pct)
        title_text = f"Los {leader} lideran el valor minero en {last_label} ({pct:.0f}%)"

    fig.update_layout(
        barmode="stack",
        title=dict(text=(
            f"<b>{title_text}</b>"
            "<br><sup style='color:#94A3B8'>Participación por categoría y periodo, sin Agregados pétreos</sup>"
        )),
        height=280,
        xaxis=dict(range=[0, 100], visible=False),
        legend=dict(orientation="h", y=-0.25, x=0),
        margin=dict(t=60, b=50, l=10, r=10),
        **_strip_axes(),
    )
    return fig


def fig_liderazgo_slope(d: pl.DataFrame) -> go.Figure:
    metalicos = d.filter((~pl.col("is_agregados")) & (pl.col("categoria") == "Metálicos"))
    periods_present = [
        (lo, hi, label) for lo, hi, label in PERIODS
        if not metalicos.filter(pl.col("año").is_between(lo, hi)).is_empty()
    ]
    if len(periods_present) < 2:
        return _empty_fig("Rango de años insuficiente para comparar periodos")

    fig = go.Figure()
    for estado in LEADERSHIP_STATES:
        xs, ys = [], []
        for lo, hi, label in periods_present:
            sub = metalicos.filter(pl.col("año").is_between(lo, hi))
            tot = sub["valor_pesos"].sum()
            v = sub.filter(pl.col("estado") == estado)["valor_pesos"].sum()
            xs.append(label)
            ys.append(v / tot * 100 if tot else 0)
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines+markers", name=estado,
            line=dict(color=LEADERSHIP_COLORS[estado], width=2),
            marker=dict(color=LEADERSHIP_COLORS[estado], size=8),
            hovertemplate=f"<b>{estado}</b><br>%{{x}}: %{{y:.1f}}%<extra></extra>",
        ))
    fig.update_xaxes(type="category", gridcolor="rgba(0,0,0,0)")
    fig.update_layout(
        title=dict(text=(
            "<b>Michoacán desplazó a Sonora como líder del valor metálico en 2020–24</b>"
            "<br><sup style='color:#94A3B8'>Participación en el valor metálico nacional por periodo</sup>"
        )),
        height=380,
        yaxis=dict(gridcolor="#334155", title="% del valor metálico"),
        **_strip_axes(),
    )
    return fig


def fig_top_minerales(d: pl.DataFrame) -> go.Figure:
    clean = d.filter(~pl.col("is_agregados"))
    ranked = (
        clean.group_by("producto").agg(pl.col("valor_pesos").sum().alias("valor"))
        .sort("valor", descending=True)
    )
    if ranked.is_empty():
        return _empty_fig("Sin datos en el período seleccionado")

    leader = ranked["producto"][0]
    top10 = ranked.head(10).sort("valor")
    colors = [FOCUS if p == leader else CONTEXT for p in top10["producto"].to_list()]

    max_val = float(top10["valor"].max()) / 1e9

    fig = go.Figure(go.Bar(
        x=(top10["valor"] / 1e9).to_list(), y=top10["producto"].to_list(), orientation="h",
        marker_color=colors,
        text=[f"${v / 1e9:,.0f}B" for v in top10["valor"].to_list()],
        textposition="outside", textfont=dict(color="#CBD5E1"), cliponaxis=False,
        hovertemplate="<b>%{y}</b><br>$%{x:,.0f} mil millones<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=(
            f"<b>{leader} es el mineral de mayor valor acumulado</b>"
            "<br><sup style='color:#94A3B8'>Top 10 minerales, sin Agregados pétreos</sup>"
        )),
        height=max(320, 10 * 28 + 80),
        xaxis=dict(gridcolor="#334155", title="Miles de millones de pesos", range=[0, max_val * 1.2]),
        yaxis=dict(gridcolor="rgba(0,0,0,0)"),
        **_strip_axes(),
    )
    return fig


def fig_artefacto(d: pl.DataFrame) -> go.Figure:
    agregados = d.filter(pl.col("is_agregados"))
    if agregados.is_empty():
        return _empty_fig("Sin datos de Agregados pétreos en el período seleccionado")

    by_state_year = agregados.group_by(["estado", "año"]).agg(pl.col("valor_pesos").sum().alias("valor"))
    hidalgo = by_state_year.filter(pl.col("estado") == "Hidalgo").sort("año")
    resto = (
        by_state_year.filter(pl.col("estado") != "Hidalgo")
        .group_by("año").agg(pl.col("valor").sum().alias("valor")).sort("año")
    )

    yr_max = int(agregados["año"].max())
    nat_top_year = by_state_year.filter(pl.col("año") == yr_max)["valor"].sum()
    hid_top_year = hidalgo.filter(pl.col("año") == yr_max)["valor"].sum() if not hidalgo.is_empty() else 0
    hid_share = hid_top_year / nat_top_year * 100 if nat_top_year else 0

    prev_years = hidalgo.filter(pl.col("año") < yr_max).sort("año")
    ratio_txt = ""
    if not prev_years.is_empty():
        prev_val = float(prev_years["valor"][-1])
        if prev_val > 0:
            ratio = hid_top_year / prev_val
            ratio_txt = f", {ratio:.0f}× su valor de {int(prev_years['año'][-1])}"

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=resto["año"].to_list(), y=(resto["valor"] / 1e9).to_list(),
        name="Resto de estados", marker_color=CONTEXT,
        hovertemplate="Resto de estados %{x}<br>$%{y:,.0f} mil millones<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        x=hidalgo["año"].to_list(), y=(hidalgo["valor"] / 1e9).to_list(),
        name="Hidalgo", marker_color="#E84855",
        hovertemplate="Hidalgo %{x}<br>$%{y:,.0f} mil millones<extra></extra>",
    ))
    fig.update_layout(
        barmode="group",
        title=dict(text=(
            f"<b>Hidalgo {yr_max}: {hid_share:.0f}% del valor nacional de Agregados pétreos{ratio_txt}</b>"
            "<br><sup style='color:#94A3B8'>Valor de \"Agregados pétreos\" por estado y año (escala logarítmica)</sup>"
        )),
        height=420,
        yaxis=dict(gridcolor="#334155", title="Miles de millones de pesos (log)", type="log"),
        xaxis=dict(gridcolor="#334155", title="Año"),
        legend=dict(orientation="h", y=-0.2, x=0),
        **_strip_axes(),
    )
    return fig


def fig_agregados_por_estado(d: pl.DataFrame) -> go.Figure:
    agregados = d.filter(pl.col("is_agregados"))
    if agregados.is_empty():
        return _empty_fig("Sin datos de Agregados pétreos en el período seleccionado")

    rank = (
        agregados.group_by("estado").agg(pl.col("valor_pesos").sum().alias("valor"))
        .sort("valor", descending=True)
    )
    total = float(rank["valor"].sum())
    hidalgo_row = rank.filter(pl.col("estado") == "Hidalgo")
    hidalgo_pct = float(hidalgo_row["valor"][0]) / total * 100 if not hidalgo_row.is_empty() and total else 0

    top = rank.sort("valor")  # ascending: largest at top of a horizontal bar
    colors = [FOCUS if e == "Hidalgo" else CONTEXT for e in top["estado"].to_list()]

    fig = go.Figure(go.Bar(
        x=(top["valor"] / 1e9).to_list(), y=top["estado"].to_list(), orientation="h",
        marker_color=colors,
        text=[f"${v / 1e9:,.1f}B" for v in top["valor"].to_list()],
        textposition="outside", textfont=dict(color="#CBD5E1"), cliponaxis=False,
        hovertemplate="<b>%{y}</b><br>$%{x:,.2f} mil millones<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=(
            f"<b>{rank.height} estados registran Agregados pétreos"
            + (f"; Hidalgo por sí solo es {hidalgo_pct:.0f}% del total" if hidalgo_pct else "")
            + "</b>"
            "<br><sup style='color:#94A3B8'>Valor por estado, pesos nominales (escala logarítmica)</sup>"
        )),
        height=max(320, top.height * 28 + 80),
        xaxis=dict(gridcolor="#334155", title="Miles de millones de pesos (log)", type="log"),
        yaxis=dict(gridcolor="rgba(0,0,0,0)"),
        **_strip_axes(),
    )
    return fig


# ── Figure factories: EIMM (volumen físico) ────────────────────────────────────


def fig_volumen_trend(d: pl.DataFrame, producto: str) -> go.Figure:
    color, fillcolor, display_name, unit = VOLUMEN_STYLE[producto]
    serie = d.filter(pl.col("producto") == producto).sort("año")
    if serie.is_empty():
        return _empty_fig(f"Sin datos de {display_name} en el período seleccionado")

    yr0, yr1 = int(serie["año"][0]), int(serie["año"][-1])
    peak_row = serie.sort("volumen", descending=True).row(0, named=True)
    peak_year, peak_val = int(peak_row["año"]), float(peak_row["volumen"])
    last_val = float(serie["volumen"][-1])

    if peak_year == yr1:
        title_text = f"{display_name}: máximo histórico en {yr1} ({last_val:,.0f} {unit.lower()})"
    else:
        drop_pct = (peak_val - last_val) / peak_val * 100 if peak_val else 0
        title_text = f"{display_name} cayó {drop_pct:.0f}% desde su pico de {peak_year}"

    fig = go.Figure(go.Scatter(
        x=serie["año"].to_list(), y=serie["volumen"].to_list(),
        mode="lines+markers", line=dict(color=color, width=2),
        fill="tozeroy", fillcolor=fillcolor,
        hovertemplate=f"Año: %{{x}}<br>%{{y:,.0f}} {unit.lower()}<extra></extra>",
    ))
    if peak_year != yr1:
        fig.add_vline(
            x=peak_year, line_dash="dot", line_color="#94A3B8",
            annotation_text=f"{peak_year}: pico", annotation_font_color="#94A3B8",
            annotation_position="top left", annotation_font_size=11,
        )
    fig.update_layout(
        title=dict(text=(
            f"<b>{title_text}</b>"
            f"<br><sup style='color:#94A3B8'>Producción nacional de {display_name.lower()}, {unit}, {yr0}–{yr1}</sup>"
        )),
        height=340,
        xaxis=dict(gridcolor="#334155", title="Año"),
        yaxis=dict(gridcolor="#334155", title=unit),
        **_strip_axes(),
    )
    return fig


def fig_municipios_activos_trend(d: pl.DataFrame) -> go.Figure:
    serie = d.sort("año")
    if serie.is_empty():
        return _empty_fig("Sin datos en el período seleccionado")

    yr0, yr1 = int(serie["año"][0]), int(serie["año"][-1])
    peak_row = serie.sort("n_municipios", descending=True).row(0, named=True)
    peak_year, peak_val = int(peak_row["año"]), int(peak_row["n_municipios"])
    last_val = int(serie["n_municipios"][-1])

    if peak_year == yr1:
        title_text = f"Máximo histórico de cobertura municipal en {yr1} ({last_val} municipios)"
    else:
        drop_pct = (peak_val - last_val) / peak_val * 100 if peak_val else 0
        title_text = f"Los municipios mineros activos cayeron {drop_pct:.0f}% desde su pico de {peak_year}"

    fig = go.Figure(go.Scatter(
        x=serie["año"].to_list(), y=serie["n_municipios"].to_list(),
        mode="lines+markers", line=dict(color="#E84855", width=2),
        fill="tozeroy", fillcolor="rgba(232,72,85,0.12)",
        hovertemplate="Año: %{x}<br>%{y} municipios activos<extra></extra>",
    ))
    if peak_year != yr1:
        fig.add_vline(
            x=peak_year, line_dash="dot", line_color="#94A3B8",
            annotation_text=f"{peak_year}: máximo ({peak_val})", annotation_font_color="#94A3B8",
            annotation_position="top left", annotation_font_size=11,
        )
    fig.update_layout(
        title=dict(text=(
            f"<b>{title_text}</b>"
            f"<br><sup style='color:#94A3B8'>Municipios con producción reportada por año, {yr0}–{yr1}</sup>"
        )),
        height=320,
        xaxis=dict(gridcolor="#334155", title="Año"),
        yaxis=dict(gridcolor="#334155", title="Municipios activos"),
        **_strip_axes(),
    )
    return fig


def fig_concentracion_producto(d: pl.DataFrame) -> go.Figure:
    sub = d.filter(pl.col("producto").is_in(CONCENTRACION_PRODUCTOS))
    if sub.is_empty():
        return _empty_fig("Sin datos en el período seleccionado")

    rows = []
    for prod in CONCENTRACION_PRODUCTOS:
        psub = sub.filter(pl.col("producto") == prod)
        if psub.is_empty():
            continue
        ranked = (
            psub.group_by("estado").agg(pl.col("volumen").sum().alias("volumen"))
            .sort("volumen", descending=True)
        )
        total = ranked["volumen"].sum()
        top = ranked.row(0, named=True)
        rows.append({"producto": prod, "estado": top["estado"], "pct": top["volumen"] / total * 100 if total else 0})
    if not rows:
        return _empty_fig("Sin datos en el período seleccionado")

    tbl = pl.DataFrame(rows).sort("pct")

    leading_states = []
    for e in tbl["estado"].to_list():
        if e not in leading_states:
            leading_states.append(e)
    color_map = {e: STATE_COLOR_PALETTE[i % len(STATE_COLOR_PALETTE)] for i, e in enumerate(leading_states)}
    colors = [color_map[e] for e in tbl["estado"].to_list()]

    top_row = tbl.sort("pct", descending=True).row(0, named=True)
    zac_products = [p for p, e in zip(tbl["producto"].to_list(), tbl["estado"].to_list()) if e == "Zacatecas"]
    zac_txt = f"; Zacatecas lidera {len(zac_products)} de {len(tbl)}" if zac_products else ""

    max_pct = float(tbl["pct"].max())

    fig = go.Figure(go.Bar(
        x=tbl["pct"].to_list(), y=tbl["producto"].to_list(), orientation="h",
        marker_color=colors,
        text=[f"{p:.0f}% {e}" for p, e in zip(tbl["pct"].to_list(), tbl["estado"].to_list())],
        textposition="outside", textfont=dict(color="#CBD5E1"), cliponaxis=False,
        hovertemplate="<b>%{y}</b><br>%{text}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=(
            f"<b>{top_row['producto']} depende en {top_row['pct']:.0f}% de un solo estado "
            f"({top_row['estado']}){zac_txt}</b>"
            "<br><sup style='color:#94A3B8'>Participación del estado líder en el volumen nacional acumulado</sup>"
        )),
        height=max(300, len(tbl) * 40 + 80),
        xaxis=dict(gridcolor="#334155", title="% del volumen nacional", range=[0, max_pct * 1.3]),
        yaxis=dict(gridcolor="rgba(0,0,0,0)"),
        **_strip_axes(),
    )
    return fig


# ── Figure factories: Comercio Exterior ────────────────────────────────────────


def fig_comercio_mapa(d: pl.DataFrame, flujo: str) -> go.Figure:
    mapped = d.filter(pl.col("iso3").is_not_null())
    if mapped.is_empty():
        return _empty_fig("Sin datos en el período seleccionado")

    leader = mapped.sort("valor_2024", descending=True).row(0, named=True)
    total = float(mapped["valor_2024"].sum())
    leader_pct = leader["valor_2024"] / total * 100 if total else 0
    plural = FLUJO_PLURAL[flujo]

    fig = px.choropleth(
        mapped, locations="iso3", color="valor_2024", locationmode="ISO-3",
        color_continuous_scale="YlOrRd", hover_name="pais", custom_data=["pais"],
    )
    fig.update_traces(hovertemplate="<b>%{customdata[0]}</b><br>$%{z:,.0f}<extra></extra>")
    fig.update_geos(
        bgcolor="rgba(0,0,0,0)", showframe=False, showcoastlines=False,
        landcolor="#1E293B", subunitcolor="#334155", countrycolor="#334155",
        projection_type="natural earth",
    )
    fig.update_layout(
        title=dict(text=(
            f"<b>{leader['pais']} concentra el {leader_pct:.0f}% de las {plural} mineras de 2024</b>"
            f"<br><sup style='color:#94A3B8'>{plural.capitalize()} por país, dólares corrientes 2024</sup>"
        )),
        height=460, paper_bgcolor="rgba(0,0,0,0)", font_color="#CBD5E1",
        margin=dict(t=60, b=10, l=10, r=10),
        coloraxis_colorbar=dict(title="USD"),
    )
    return fig


def fig_comercio_top_paises(d: pl.DataFrame, flujo: str) -> go.Figure:
    if d.is_empty():
        return _empty_fig("Sin datos en el período seleccionado")
    top10 = d.sort("valor_2024", descending=True).head(10).sort("valor_2024")
    leader = top10["pais"][-1]
    colors = [FOCUS if p == leader else CONTEXT for p in top10["pais"].to_list()]
    max_val = float(top10["valor_2024"].max()) / 1e9
    plural = FLUJO_PLURAL[flujo]

    fig = go.Figure(go.Bar(
        x=(top10["valor_2024"] / 1e9).to_list(), y=top10["pais"].to_list(), orientation="h",
        marker_color=colors,
        text=[f"${v / 1e9:,.1f}B" for v in top10["valor_2024"].to_list()],
        textposition="outside", textfont=dict(color="#CBD5E1"), cliponaxis=False,
        hovertemplate="<b>%{y}</b><br>$%{x:,.2f}B<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=(
            f"<b>Top 10 países — {plural} mineras 2024</b>"
            f"<br><sup style='color:#94A3B8'>{leader} a la cabeza</sup>"
        )),
        height=max(320, 10 * 28 + 80),
        xaxis=dict(gridcolor="#334155", title="Miles de millones de USD", range=[0, max_val * 1.2]),
        yaxis=dict(gridcolor="rgba(0,0,0,0)"),
        **_strip_axes(),
    )
    return fig


def fig_comercio_top_productos(d: pl.DataFrame, flujo: str) -> go.Figure:
    if d.is_empty():
        return _empty_fig("Sin datos en el período seleccionado")
    top10 = d.sort("valor_2024", descending=True).head(10).sort("valor_2024")
    leader = top10["producto"][-1]
    colors = [FOCUS if p == leader else CONTEXT for p in top10["producto"].to_list()]
    max_val = float(top10["valor_2024"].max()) / 1e9
    plural = FLUJO_PLURAL[flujo]

    fig = go.Figure(go.Bar(
        x=(top10["valor_2024"] / 1e9).to_list(), y=top10["producto"].to_list(), orientation="h",
        marker_color=colors,
        text=[f"${v / 1e9:,.1f}B" for v in top10["valor_2024"].to_list()],
        textposition="outside", textfont=dict(color="#CBD5E1"), cliponaxis=False,
        hovertemplate="<b>%{y}</b><br>$%{x:,.2f}B<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=(
            f"<b>Top 10 productos — {plural} mineras 2024</b>"
            f"<br><sup style='color:#94A3B8'>{leader} a la cabeza</sup>"
        )),
        height=max(320, 10 * 28 + 80),
        xaxis=dict(gridcolor="#334155", title="Miles de millones de USD", range=[0, max_val * 1.2]),
        yaxis=dict(gridcolor="rgba(0,0,0,0)"),
        **_strip_axes(),
    )
    return fig


# ── KPI computation ──────────────────────────────────────────────────────────


def compute_kpis(d: pl.DataFrame) -> dict:
    clean = d.filter(~pl.col("is_agregados"))
    total = float(clean["valor_pesos"].sum())
    rank = (
        clean.group_by("estado").agg(pl.col("valor_pesos").sum().alias("valor"))
        .sort("valor", descending=True)
    )
    if rank.is_empty():
        leader, leader_pct, top5_pct = "N/A", 0.0, 0.0
    else:
        leader_row = rank.row(0, named=True)
        leader = leader_row["estado"]
        leader_pct = float(leader_row["valor"] / total * 100) if total else 0.0
        top5_pct = float(rank.head(5)["valor"].sum() / total * 100) if total else 0.0
    agregados_total = float(d.filter(pl.col("is_agregados"))["valor_pesos"].sum())
    return dict(total=total, leader=leader, leader_pct=leader_pct, top5_pct=top5_pct,
                agregados_total=agregados_total)


def compute_kpis_eimm(d_nacional: pl.DataFrame, d_muni: pl.DataFrame) -> dict:
    out = {}
    for producto, key in [("Oro", "oro"), ("Plata", "plata"), ("Fierro en Extraccion", "fierro")]:
        serie = d_nacional.filter(pl.col("producto") == producto).sort("año")
        if serie.is_empty():
            out[key] = {"last": 0.0, "last_year": None, "peak": 0.0, "peak_year": None}
            continue
        peak_row = serie.sort("volumen", descending=True).row(0, named=True)
        out[key] = {
            "last": float(serie["volumen"][-1]), "last_year": int(serie["año"][-1]),
            "peak": float(peak_row["volumen"]), "peak_year": int(peak_row["año"]),
        }
    serie_m = d_muni.sort("año")
    if serie_m.is_empty():
        out["muni"] = {"last": 0, "last_year": None, "peak": 0, "peak_year": None}
    else:
        peak_row = serie_m.sort("n_municipios", descending=True).row(0, named=True)
        out["muni"] = {
            "last": int(serie_m["n_municipios"][-1]), "last_year": int(serie_m["año"][-1]),
            "peak": int(peak_row["n_municipios"]), "peak_year": int(peak_row["año"]),
        }
    return out


# ── Layout ────────────────────────────────────────────────────────────────────

SLIDER_MARKS = {y: str(y) for y in range(YEAR_MIN, YEAR_MAX + 1, 5)}
SLIDER_MARKS[YEAR_MIN] = str(YEAR_MIN)
SLIDER_MARKS[YEAR_MAX] = str(YEAR_MAX)


def kpi_card(title, value_id, sub_id, delta_id=None):
    children = [
        html.P(title, style={"color": "#94A3B8", "fontSize": "0.8rem", "marginBottom": "4px"}),
        html.H3(id=value_id, style={"color": "#F8FAFC", "fontWeight": "700", "margin": "0"}),
    ]
    if delta_id:
        children.append(html.Div(id=delta_id, style={"minHeight": "18px", "marginTop": "2px"}))
    children.append(html.Small(id=sub_id, style={"color": "#64748B"}))
    return html.Div(children, style=CARD_STYLE)


def graph_row(*id_width_pairs: tuple[str, int]):
    return dbc.Row([
        dbc.Col(dcc.Graph(id=gid, config={"displayModeBar": False}), width=w)
        for gid, w in id_width_pairs
    ], className="mb-3 g-3")


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SLATE])
app.title = "Producción Minera · México"

app.layout = html.Div(
    style={"backgroundColor": "#0F172A", "minHeight": "100vh", "padding": "24px"},
    children=[
        html.Div([
            html.H2("Producción Minera de México",
                    style={"color": "#F8FAFC", "fontWeight": "700", "marginBottom": "4px"}),
            html.P(f"Servicio Geológico Mexicano (valor) · INEGI EIMM (volumen físico) · {YEAR_MIN}–{YEAR_MAX}",
                   style={"color": "#64748B", "fontSize": "0.9rem"}),
        ], className="mb-3"),

        dbc.Alert(
            [
                html.Strong("⚠ Nota metodológica: "),
                "\"Agregados pétreos\" (grava/arena de construcción) se excluyó de todos los análisis: "
                "sus valores son erráticos en varios estados y años, y el caso de Hidalgo 2024 por sí solo "
                "sería 91% del valor nacional limpio de ese año. Ver la pestaña \"Agregados pétreos\" para el detalle. "
                "Los años 2023–2024 tienen menos registros que el promedio y deben interpretarse como preliminares.",
            ],
            color="warning",
            style={"fontSize": "0.83rem", "padding": "8px 14px"},
            className="mb-3",
        ),

        dbc.Row([
            dbc.Col(kpi_card("Valor minero limpio", "kpi-total-val", "kpi-total-sub"), width=3),
            dbc.Col(kpi_card("Estado líder", "kpi-leader-val", "kpi-leader-sub"), width=3),
            dbc.Col(kpi_card("Concentración top-5", "kpi-top5-val", "kpi-top5-sub"), width=3),
            dbc.Col(kpi_card("Excluido (Agregados pétreos)", "kpi-agregados-val", "kpi-agregados-sub"), width=3),
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
            width=12),
        ], className="mb-4 g-3"),

        dcc.Tabs(style={"marginBottom": "16px"}, children=[
            dcc.Tab(label="📈 Panorama Nacional", style=TAB_STYLE, selected_style=TAB_SEL, children=[
                graph_row(("graph-trend-nacional", 12)),
                graph_row(("graph-composicion", 12)),
            ]),
            dcc.Tab(label="🗺 Ranking Estatal", style=TAB_STYLE, selected_style=TAB_SEL, children=[
                graph_row(("graph-mapa", 12)),
                graph_row(("graph-ranking", 7), ("graph-detalle-estado", 5)),
                dbc.Row([
                    dbc.Col(html.Div([
                        html.Span(id="drill-label", style={"color": "#94A3B8", "fontSize": "0.85rem"}),
                        html.Button("✕ Ver nacional", id="clear-drill", n_clicks=0,
                                    style={"marginLeft": "12px", "fontSize": "0.78rem",
                                           "background": "#1E293B", "color": "#94A3B8",
                                           "border": "1px solid #334155", "borderRadius": "4px",
                                           "padding": "2px 10px", "cursor": "pointer"}),
                    ]), width=12),
                ], className="mb-3"),
                dcc.Store(id="selected-estado", data=None),
            ]),
            dcc.Tab(label="⛏ Liderazgo Metálico", style=TAB_STYLE, selected_style=TAB_SEL, children=[
                graph_row(("graph-liderazgo", 12)),
                graph_row(("graph-top-minerales", 12)),
            ]),
            dcc.Tab(label="⚠ Agregados Pétreos", style=TAB_STYLE, selected_style=TAB_SEL, children=[
                dbc.Row([
                    dbc.Col(html.Div([
                        html.Strong("¿Por qué se excluye \"Agregados pétreos\" del resto del dashboard?",
                                    style={"color": "#F8FAFC"}),
                        html.P(
                            "Es la única categoría con valores lo bastante erráticos como para dominar "
                            "cualquier ranking sin representar necesariamente la realidad económica: un solo "
                            "registro (Hidalgo, 2024) salta ~99× su propio valor del año anterior y por sí solo "
                            "supera el valor combinado de todos los demás estados en ese producto. "
                            "Antes de publicar cualquier cifra basada en este producto, debe reconciliarse "
                            "contra el Anuario Estadístico fuente del SGM. Este patrón se confirma de forma "
                            "independiente en la publicación oficial más reciente del SGM (18 nov 2025): esa "
                            "fuente separada reporta $16.24 billones de pesos para Agregados pétreos a nivel "
                            "nacional en 2024, del mismo orden de magnitud que los $15.57 billones de Hidalgo "
                            "solo en esta base de datos.",
                            style={"color": "#94A3B8", "fontSize": "0.83rem", "marginTop": "4px"},
                        ),
                    ], style={**CARD_STYLE, "textAlign": "left"}), width=12),
                ], className="mb-3 g-3"),
                graph_row(("graph-artefacto", 12)),
                graph_row(("graph-agregados-estados", 12)),
            ]),
            dcc.Tab(label="📦 Volumen Físico (EIMM)", style=TAB_STYLE, selected_style=TAB_SEL, children=[
                dbc.Alert(
                    [
                        html.Strong("⚠ Nota metodológica: "),
                        "Los municipios mineros con producción reportada cayeron de 138 (2013) a 78 (2025), "
                        "una contracción de 43%. Parte de la caída de oro, plata y hierro que se ve abajo "
                        "refleja menos sitios reportando, no necesariamente menor producción por sitio activo.",
                    ],
                    color="warning",
                    style={"fontSize": "0.83rem", "padding": "8px 14px"},
                    className="mb-3",
                ),
                dbc.Row([
                    dbc.Col(kpi_card("Oro", "kpi-eimm-oro-val", "kpi-eimm-oro-sub", "kpi-eimm-oro-delta"), width=3),
                    dbc.Col(kpi_card("Plata", "kpi-eimm-plata-val", "kpi-eimm-plata-sub", "kpi-eimm-plata-delta"), width=3),
                    dbc.Col(kpi_card("Fierro (extracción)", "kpi-eimm-fierro-val", "kpi-eimm-fierro-sub", "kpi-eimm-fierro-delta"), width=3),
                    dbc.Col(kpi_card("Municipios activos", "kpi-eimm-muni-val", "kpi-eimm-muni-sub", "kpi-eimm-muni-delta"), width=3),
                ], className="mb-3 g-3"),
                graph_row(("graph-eimm-oro", 6), ("graph-eimm-plata", 6)),
                graph_row(("graph-eimm-fierro", 12)),
                graph_row(("graph-eimm-municipios", 12)),
            ]),
            dcc.Tab(label="🗺 Concentración por Estado (EIMM)", style=TAB_STYLE, selected_style=TAB_SEL, children=[
                dbc.Row([
                    dbc.Col(html.Div([
                        html.Strong("Un solo municipio lideró 4 minerales a la vez en 2025", style={"color": "#F8FAFC"}),
                        html.P(
                            "Mazapil, Zacatecas fue el municipio #1 del país en Oro, Plata, Plomo y Zinc "
                            "simultáneamente en 2025 (el cobre lo lidera Cananea, Sonora). Varios minerales "
                            "dependen de un solo estado para la mayoría de su producción, como muestra la "
                            "gráfica de abajo.",
                            style={"color": "#94A3B8", "fontSize": "0.83rem", "marginTop": "4px"},
                        ),
                    ], style={**CARD_STYLE, "textAlign": "left"}), width=12),
                ], className="mb-3 g-3"),
                graph_row(("graph-eimm-concentracion", 12)),
            ]),
            dcc.Tab(label="🌎 Comercio Exterior", style=TAB_STYLE, selected_style=TAB_SEL, children=[
                dbc.Row([
                    dbc.Col(html.Div([
                        html.P("Exportaciones 2024", style={"color": "#94A3B8", "fontSize": "0.8rem", "marginBottom": "4px"}),
                        html.H3(_fmt_usd(COMERCIO_EXP_2024), style={"color": "#F8FAFC", "fontWeight": "700", "margin": "0"}),
                        _delta_span(COMERCIO_EXP_2024, COMERCIO_EXP_2023, label="vs 2023"),
                        html.Small("dólares corrientes", style={"color": "#64748B"}),
                    ], style=CARD_STYLE), width=3),
                    dbc.Col(html.Div([
                        html.P("Importaciones 2024", style={"color": "#94A3B8", "fontSize": "0.8rem", "marginBottom": "4px"}),
                        html.H3(_fmt_usd(COMERCIO_IMP_2024), style={"color": "#F8FAFC", "fontWeight": "700", "margin": "0"}),
                        _delta_span(COMERCIO_IMP_2024, COMERCIO_IMP_2023, label="vs 2023", harm_when_up=True),
                        html.Small("dólares corrientes", style={"color": "#64748B"}),
                    ], style=CARD_STYLE), width=3),
                    dbc.Col(html.Div([
                        html.P("Balanza comercial 2024", style={"color": "#94A3B8", "fontSize": "0.8rem", "marginBottom": "4px"}),
                        html.H3(_fmt_usd(COMERCIO_BALANCE_2024),
                                style={"color": "#E84855" if COMERCIO_BALANCE_2024 < 0 else "#3BB273", "fontWeight": "700", "margin": "0"}),
                        _delta_span(COMERCIO_BALANCE_2024, COMERCIO_BALANCE_2023, label="vs 2023", harm_when_up=False),
                        html.Small("déficit" if COMERCIO_BALANCE_2024 < 0 else "superávit", style={"color": "#64748B"}),
                    ], style=CARD_STYLE), width=3),
                    dbc.Col(html.Div([
                        html.P("País líder en exportación", style={"color": "#94A3B8", "fontSize": "0.8rem", "marginBottom": "4px"}),
                        html.H3(COMERCIO_LEADER_PAIS, style={"color": "#F8FAFC", "fontWeight": "700", "margin": "0"}),
                        _delta_span(COMERCIO_LEADER_PCT_2024, COMERCIO_LEADER_PCT_2023, label="pts. vs 2023"),
                        html.Small(f"{COMERCIO_LEADER_PCT_2024:.0f}% de las exportaciones", style={"color": "#64748B"}),
                    ], style=CARD_STYLE), width=3),
                ], className="mb-3 g-3"),
                dbc.Row([
                    dbc.Col(html.Div([
                        html.Label("Flujo", style={"color": "#94A3B8", "fontSize": "0.85rem", "marginBottom": "8px"}),
                        dcc.RadioItems(
                            id="comercio-flujo",
                            options=[{"label": " Exportación", "value": "Exportación"},
                                     {"label": " Importación", "value": "Importación"}],
                            value="Exportación", inline=True,
                            inputStyle={"marginRight": "6px", "marginLeft": "16px"},
                            style={"color": "#CBD5E1"},
                        ),
                    ], style={"background": "#1E293B", "border": "1px solid #334155",
                              "borderRadius": "8px", "padding": "16px"}), width=12),
                ], className="mb-3 g-3"),
                graph_row(("graph-comercio-mapa", 12)),
                graph_row(("graph-comercio-paises", 6), ("graph-comercio-productos", 6)),
            ]),
        ]),
    ]
)


# ── Single callback ──────────────────────────────────────────────────────────
@app.callback(
    Output("kpi-total-val", "children"),
    Output("kpi-total-sub", "children"),
    Output("kpi-leader-val", "children"),
    Output("kpi-leader-sub", "children"),
    Output("kpi-top5-val", "children"),
    Output("kpi-top5-sub", "children"),
    Output("kpi-agregados-val", "children"),
    Output("kpi-agregados-sub", "children"),
    Output("graph-trend-nacional", "figure"),
    Output("graph-composicion", "figure"),
    Output("graph-mapa", "figure"),
    Output("graph-ranking", "figure"),
    Output("graph-liderazgo", "figure"),
    Output("graph-top-minerales", "figure"),
    Output("graph-artefacto", "figure"),
    Output("graph-agregados-estados", "figure"),
    Input("year-range", "value"),
)
def update_all(year_range):
    yr_min, yr_max = year_range
    d = df.filter(pl.col("año").is_between(yr_min, yr_max))

    kpis = compute_kpis(d)

    return (
        _fmt_pesos(kpis["total"]),
        f"{yr_min}–{yr_max}, sin Agregados pétreos",
        kpis["leader"],
        f"{kpis['leader_pct']:.1f}% del valor limpio",
        f"{kpis['top5_pct']:.0f}%",
        "de 32 estados",
        _fmt_pesos(kpis["agregados_total"]),
        "ver pestaña Agregados Pétreos",
        fig_trend_nacional(d),
        fig_composicion_stacked(d),
        fig_choropleth_estado(d),
        fig_ranking_bar(d),
        fig_liderazgo_slope(d),
        fig_top_minerales(d),
        fig_artefacto(d),
        fig_agregados_por_estado(d),
    )


@app.callback(
    Output("selected-estado", "data"),
    Input("graph-mapa", "clickData"),
    Input("graph-ranking", "clickData"),
    Input("clear-drill", "n_clicks"),
    prevent_initial_call=True,
)
def set_selected_estado(map_click, bar_click, _clear):
    from dash import ctx

    if ctx.triggered_id == "clear-drill":
        return None
    if ctx.triggered_id == "graph-mapa" and map_click:
        return map_click["points"][0]["location"]
    if ctx.triggered_id == "graph-ranking" and bar_click:
        return bar_click["points"][0]["y"]
    return dash.no_update


@app.callback(
    Output("graph-detalle-estado", "figure"),
    Output("drill-label", "children"),
    Input("year-range", "value"),
    Input("selected-estado", "data"),
)
def update_detalle(year_range, estado):
    yr_min, yr_max = year_range
    d = df.filter(pl.col("año").is_between(yr_min, yr_max))

    if not estado:
        return (
            _empty_fig("Haz clic en un estado (mapa o ranking) para ver su detalle"),
            "Ningún estado seleccionado",
        )
    return fig_detalle_estado(d, estado), estado


@app.callback(
    Output("kpi-eimm-oro-val", "children"),
    Output("kpi-eimm-oro-sub", "children"),
    Output("kpi-eimm-oro-delta", "children"),
    Output("kpi-eimm-plata-val", "children"),
    Output("kpi-eimm-plata-sub", "children"),
    Output("kpi-eimm-plata-delta", "children"),
    Output("kpi-eimm-fierro-val", "children"),
    Output("kpi-eimm-fierro-sub", "children"),
    Output("kpi-eimm-fierro-delta", "children"),
    Output("kpi-eimm-muni-val", "children"),
    Output("kpi-eimm-muni-sub", "children"),
    Output("kpi-eimm-muni-delta", "children"),
    Output("graph-eimm-oro", "figure"),
    Output("graph-eimm-plata", "figure"),
    Output("graph-eimm-fierro", "figure"),
    Output("graph-eimm-municipios", "figure"),
    Output("graph-eimm-concentracion", "figure"),
    Input("year-range", "value"),
)
def update_eimm(year_range):
    yr_min, yr_max = year_range
    d_nacional = eimm_nacional.filter(pl.col("año").is_between(yr_min, yr_max))
    d_estado = eimm_estado.filter(pl.col("año").is_between(yr_min, yr_max))
    d_muni = eimm_muni_activos.filter(pl.col("año").is_between(yr_min, yr_max))

    kpis = compute_kpis_eimm(d_nacional, d_muni)

    def _kpi(info, unit, decimals=0):
        val = f"{info['last']:,.{decimals}f} {unit}" if info["last_year"] else "N/D"
        sub = f"{info['last_year']}, pico {info['peak_year']}: {info['peak']:,.{decimals}f} {unit}" if info["peak_year"] else ""
        delta = ""
        if info["peak_year"] and info["last_year"] and info["peak_year"] != info["last_year"]:
            delta = _delta_span(info["last"], info["peak"], label=f"vs pico {info['peak_year']}")
        return val, sub, delta

    oro_val, oro_sub, oro_delta = _kpi(kpis["oro"], "kg")
    plata_val, plata_sub, plata_delta = _kpi(kpis["plata"], "kg")
    fierro_val, fierro_sub, fierro_delta = _kpi(kpis["fierro"], "t")
    muni_val, muni_sub, muni_delta = _kpi(kpis["muni"], "municipios")

    return (
        oro_val, oro_sub, oro_delta,
        plata_val, plata_sub, plata_delta,
        fierro_val, fierro_sub, fierro_delta,
        muni_val, muni_sub, muni_delta,
        fig_volumen_trend(d_nacional, "Oro"),
        fig_volumen_trend(d_nacional, "Plata"),
        fig_volumen_trend(d_nacional, "Fierro en Extraccion"),
        fig_municipios_activos_trend(d_muni),
        fig_concentracion_producto(d_estado),
    )


@app.callback(
    Output("graph-comercio-mapa", "figure"),
    Output("graph-comercio-paises", "figure"),
    Output("graph-comercio-productos", "figure"),
    Input("comercio-flujo", "value"),
)
def update_comercio(flujo):
    d_pais = comercio_pais.filter(pl.col("flujo") == flujo)
    d_producto = comercio_producto.filter(pl.col("flujo") == flujo)
    return (
        fig_comercio_mapa(d_pais, flujo),
        fig_comercio_top_paises(d_pais, flujo),
        fig_comercio_top_productos(d_producto, flujo),
    )


if __name__ == "__main__":
    print("Dashboard disponible en http://localhost:8064")
    app.run(debug=True, port=8064)
