"""
Dashboard de Cuenta Pública Federal (SHCP, 2011-2025).

Seis hallazgos validados en dashboard_data/{cp,at,subsidios,pibe}_*.parquet
(scripts/prepare_cuenta_publica.py, scripts/prepare_pibe.py): crecimiento y
tasa de ejercicio, concentración por RAMO y tipo de gasto, el artefacto
administrativo de CDMX en la geografía, seguimiento de los anexos
transversales, el subsidio per cápita real (vía población CONAPO), y
Tabasco/Campeche: transferencias federalizadas (R28+R33) recibidas vs.
PIB petrolero producido (INEGI PIBE).
"""

import json
from pathlib import Path

import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
from dash import Dash, dcc, html

# ── paths ────────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent.parent / "dashboard_data"
GEO_FILE = Path(__file__).parent.parent / "data" / "mexico_states.geojson"

# ── theme ────────────────────────────────────────────────────────────────────
CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font_color="#CBD5E1",
)
CARD_STYLE = {
    "background": "#1E293B", "border": "1px solid #334155",
    "borderRadius": "8px", "padding": "16px", "textAlign": "center",
}
TAB_STYLE = {"backgroundColor": "#0F172A", "color": "#94A3B8", "borderTop": "none"}
TAB_SEL = {"backgroundColor": "#1E293B", "color": "#F8FAFC",
           "borderTop": "2px solid #2E86AB", "fontWeight": "600"}
GRAPH_CONFIG = {"displayModeBar": False}

FOCUS, CONTEXT, GREEN, RED, ORANGE = "#2E86AB", "#475569", "#3BB273", "#E84855", "#F4A261"

with open(GEO_FILE) as f:
    GEOJSON = json.load(f)

# ── data (pre-built by scripts/prepare_cuenta_publica.py) ──────────────────
cp_totals = pl.read_parquet(DATA_DIR / "cp_totals_by_year.parquet").sort("year")
cp_ramo = pl.read_parquet(DATA_DIR / "cp_ramo_year.parquet")
cp_tipogasto = pl.read_parquet(DATA_DIR / "cp_tipogasto_year.parquet")
cp_geo = pl.read_parquet(DATA_DIR / "cp_geo_year.parquet")
cp_transfers = pl.read_parquet(DATA_DIR / "cp_transfers_state_year.parquet")
pibe_oil_share = pl.read_parquet(DATA_DIR / "pibe_oil_share_long.parquet")
at_transversal = pl.read_parquet(DATA_DIR / "at_transversal_year.parquet")
subsidios_state = pl.read_parquet(DATA_DIR / "subsidios_state_year.parquet")
subsidios_percapita = pl.read_parquet(DATA_DIR / "subsidios_percapita_year.parquet")

SNAPSHOT_YEAR = 2024
BILLON = 1e12  # billón (es) = 10^12, igual que la "T" de DATA_OVERVIEW.md

# ── KPI values ───────────────────────────────────────────────────────────────
_tot_2024 = cp_totals.filter(pl.col("year") == SNAPSHOT_YEAR).to_dicts()[0]
_tot_2012 = cp_totals.filter(pl.col("year") == 2012)["ejercido"][0]
_growth_mult = _tot_2024["ejercido"] / _tot_2012
_mean_rate = float(cp_totals["execution_rate"].mean())
_n_over100 = int((cp_totals["execution_rate"] > 100).sum())
_n_years = cp_totals.height

_ramo_2024 = cp_ramo.filter(pl.col("year") == SNAPSHOT_YEAR).sort("ejercido", descending=True)
_total_ramo_2024 = float(_ramo_2024["ejercido"].sum())
_top10_share = float(_ramo_2024.head(10)["ejercido"].sum() / _total_ramo_2024 * 100)

_tipo_2024 = cp_tipogasto.filter(pl.col("year") == SNAPSHOT_YEAR)
_total_tipo_2024 = float(_tipo_2024["ejercido"].sum())
_pensiones_pct = float(_tipo_2024.filter(pl.col("id_tipogasto") == 4)["ejercido"][0] / _total_tipo_2024 * 100)

_geo_2024 = cp_geo.filter(pl.col("year") == SNAPSHOT_YEAR)
_total_geo_2024 = float(_geo_2024["ejercido"].sum())
_cdmx_pct = float(_geo_2024.filter(pl.col("id_entidad") == 9)["ejercido"][0] / _total_geo_2024 * 100)
_nodist_pct = float(_geo_2024.filter(pl.col("id_entidad") == 34)["ejercido"][0] / _total_geo_2024 * 100)

PETRO_STATES = {4: ("Campeche", RED), 27: ("Tabasco", ORANGE)}

_oil_nat = pibe_oil_share.filter(pl.col("entidad") == "Estados Unidos Mexicanos").select(
    "year", pl.col("oil_mdp").alias("oil_nat")
)
_oil_combined = (
    pibe_oil_share.filter(pl.col("entidad").is_in([n for n, _ in PETRO_STATES.values()]))
    .join(_oil_nat, on="year")
    .with_columns((pl.col("oil_mdp") / pl.col("oil_nat") * 100).alias("share_pct"))
    .group_by("year").agg(pl.col("share_pct").sum().alias("combined_pct"))
    .sort("year")
)
_oil_year0, _oil_year1 = int(_oil_combined["year"][0]), int(_oil_combined["year"][-1])
_oil_combined_latest = float(_oil_combined["combined_pct"][-1])
_oil_combined_min = float(_oil_combined["combined_pct"].min())
_oil_combined_max = float(_oil_combined["combined_pct"].max())


# ── figure factories ─────────────────────────────────────────────────────────

def fig_trend_total() -> go.Figure:
    d = cp_totals
    fig = go.Figure()
    fig.add_vrect(x0=2010.5, x1=2013.5, fillcolor="rgba(148,163,184,0.08)", line_width=0,
                  annotation_text="esquema reducido de datos", annotation_font_color="#94A3B8",
                  annotation_position="top left")
    fig.add_trace(go.Scatter(x=d["year"], y=d["aprobado"] / BILLON, mode="lines+markers", name="Aprobado",
                              line=dict(color=CONTEXT, width=1.5, dash="dot"), marker=dict(size=5)))
    fig.add_trace(go.Scatter(x=d["year"], y=d["ejercido"] / BILLON, mode="lines+markers", name="Ejercido",
                              line=dict(color=FOCUS, width=2.5), marker=dict(size=6)))
    fig.add_annotation(x=2024, y=_tot_2024["ejercido"] / BILLON, text=f"×{_growth_mult:.1f} vs. 2012",
                        showarrow=True, ay=-35, font=dict(color=FOCUS))
    fig.update_layout(
        **CHART_LAYOUT,
        title=dict(text="<b>El gasto ejercido se multiplicó ×3.6 entre 2012 y 2024</b>"
                        "<br><sup style='color:#94A3B8'>Presupuesto aprobado vs. ejercido, billones de pesos corrientes, 2011-2025</sup>"),
        xaxis=dict(gridcolor="#334155", tickmode="linear", dtick=1),
        yaxis=dict(gridcolor="#334155", title="Billones de pesos"),
        legend=dict(orientation="h", y=-0.18, x=0),
        height=420, margin=dict(t=80, b=60),
    )
    return fig


def fig_execution_rate() -> go.Figure:
    d = cp_totals
    colors = [RED if v > 100 else CONTEXT for v in d["execution_rate"]]
    fig = go.Figure(go.Bar(x=d["year"], y=d["execution_rate"], marker_color=colors,
                            hovertemplate="<b>%{x}</b>: %{y:.1f}%<extra></extra>"))
    fig.add_hline(y=100, line_color="#64748B", line_width=1)
    fig.add_hline(y=_mean_rate, line_dash="dot", line_color="#94A3B8",
                  annotation_text=f"promedio: {_mean_rate:.1f}%", annotation_font_color="#94A3B8")
    r2016 = d.filter(pl.col("year") == 2016)["execution_rate"][0]
    fig.add_annotation(x=2016, y=r2016, text=f"pico: {r2016:.1f}%", showarrow=True, ay=-30, font=dict(color=RED))
    fig.update_layout(
        **CHART_LAYOUT,
        title=dict(text=f"<b>El gobierno gastó más de lo aprobado en {_n_over100} de {_n_years} años</b>"
                        "<br><sup style='color:#94A3B8'>Tasa de ejercicio (ejercido / aprobado), %, 2011-2025</sup>"),
        xaxis=dict(gridcolor="#334155", tickmode="linear", dtick=1),
        yaxis=dict(gridcolor="#334155", title="%", ticksuffix="%"),
        height=380, margin=dict(t=80),
    )
    return fig


def fig_ramo_ranking() -> go.Figure:
    d = _ramo_2024.head(15).sort("ejercido")
    top10_ids = set(_ramo_2024.head(10)["id_ramo"])
    colors = [FOCUS if i in top10_ids else CONTEXT for i in d["id_ramo"]]
    fig = go.Figure(go.Bar(
        x=d["ejercido"] / BILLON, y=d["desc_ramo"], orientation="h", marker_color=colors,
        hovertemplate="<b>%{y}</b><br>%{x:.2f} billones<extra></extra>",
    ))
    fig.add_annotation(xref="paper", yref="paper", x=0.98, y=0.94, showarrow=False,
                        text=f"Top 10 RAMOs = {_top10_share:.1f}% del gasto ejercido",
                        font=dict(color=FOCUS, size=12), align="right")
    fig.update_layout(
        **CHART_LAYOUT,
        title=dict(text="<b>10 de 48 RAMOs concentran 82% del gasto federal</b>"
                        f"<br><sup style='color:#94A3B8'>Monto ejercido por RAMO, billones de pesos, {SNAPSHOT_YEAR}</sup>"),
        yaxis=dict(autorange="reversed", gridcolor="rgba(0,0,0,0)", tickfont=dict(size=10)),
        xaxis=dict(gridcolor="#334155", title="Billones de pesos"),
        height=max(420, d.height * 28 + 100), margin=dict(l=280),
    )
    return fig


_TIPOGASTO_GROUPS = {
    1: ("Gasto corriente", FOCUS),
    4: ("Pensiones y jubilaciones", RED),
    5: ("Participaciones", ORANGE),
    3: ("Obra pública", GREEN),
    2: ("Capital diferente de obra", "#A855F7"),
}
_TIPOGASTO_OTROS = "Otros"


def fig_tipogasto_stacked() -> go.Figure:
    d = _tipo_2024.with_columns(
        pl.col("id_tipogasto").map_elements(
            lambda i: _TIPOGASTO_GROUPS.get(i, (_TIPOGASTO_OTROS, "#64748B"))[0], return_dtype=pl.Utf8
        ).alias("grupo")
    ).group_by("grupo").agg(pl.col("ejercido").sum())
    order = [g[0] for g in _TIPOGASTO_GROUPS.values()] + [_TIPOGASTO_OTROS]
    color_map = {g[0]: g[1] for g in _TIPOGASTO_GROUPS.values()} | {_TIPOGASTO_OTROS: "#64748B"}

    fig = go.Figure()
    for grupo in order:
        row = d.filter(pl.col("grupo") == grupo)
        if row.height == 0:
            continue
        pct = float(row["ejercido"][0] / _total_tipo_2024 * 100)
        fig.add_trace(go.Bar(
            x=[pct], y=["Composición"], orientation="h", name=grupo, marker_color=color_map[grupo],
            text=f"{pct:.1f}%", textposition="inside", insidetextanchor="middle",
            hovertemplate=f"<b>{grupo}</b>: %{{x:.1f}}%<extra></extra>",
        ))
    fig.update_layout(
        barmode="stack",
        **CHART_LAYOUT,
        title=dict(text=f"<b>Uno de cada cuatro pesos del gasto va a pensiones ({_pensiones_pct:.1f}%)</b>"
                        f"<br><sup style='color:#94A3B8'>Composición del gasto ejercido por tipo, {SNAPSHOT_YEAR}</sup>"),
        xaxis=dict(range=[0, 100], visible=False),
        yaxis=dict(visible=False),
        legend=dict(orientation="h", y=-0.3, x=0),
        height=260, margin=dict(t=60, b=70, l=10, r=10),
    )
    return fig


def fig_geo_ranking() -> go.Figure:
    d = _geo_2024.sort("ejercido", descending=True).head(12).sort("ejercido")
    colors = [RED if i in (9, 34) else CONTEXT for i in d["id_entidad"]]
    fig = go.Figure(go.Bar(
        x=d["ejercido"] / BILLON, y=d["entidad"], orientation="h", marker_color=colors,
        hovertemplate="<b>%{y}</b><br>%{x:.2f} billones<extra></extra>",
    ))
    fig.add_annotation(xref="paper", yref="paper", x=0.98, y=0.94, showarrow=False,
                        text=f"CDMX + No Distribuible = {_cdmx_pct + _nodist_pct:.1f}% (registro administrativo, no entrega real)",
                        font=dict(color=RED, size=11), align="right")
    fig.update_layout(
        **CHART_LAYOUT,
        title=dict(text="<b>La geografía del gasto es un artefacto de registro, no de entrega</b>"
                        f"<br><sup style='color:#94A3B8'>Monto ejercido por entidad, billones de pesos, {SNAPSHOT_YEAR}</sup>"),
        yaxis=dict(autorange="reversed", gridcolor="rgba(0,0,0,0)", tickfont=dict(size=10)),
        xaxis=dict(gridcolor="#334155", title="Billones de pesos"),
        height=max(420, d.height * 28 + 100), margin=dict(l=180),
    )
    return fig


def fig_cdmx_share_trend() -> go.Figure:
    totals_by_year = cp_geo.group_by("year").agg(pl.col("ejercido").sum().alias("total_yr"))
    cdmx = (
        cp_geo.filter(pl.col("id_entidad") == 9)
        .join(totals_by_year, on="year")
        .with_columns((pl.col("ejercido") / pl.col("total_yr") * 100).alias("share_pct"))
        .sort("year")
    )
    fig = go.Figure(go.Scatter(x=cdmx["year"], y=cdmx["share_pct"], mode="lines+markers",
                                line=dict(color=RED, width=2)))
    for y in (int(cdmx["year"].min()), SNAPSHOT_YEAR):
        row = cdmx.filter(pl.col("year") == y)
        fig.add_annotation(x=y, y=row["share_pct"][0], text=f"{row['share_pct'][0]:.1f}%",
                            showarrow=True, ay=-25, font=dict(color="#94A3B8"))
    fig.update_layout(
        **CHART_LAYOUT,
        title=dict(text="<b>El 40% en CDMX es persistente, no un año atípico</b>"
                        "<br><sup style='color:#94A3B8'>% del gasto ejercido registrado en CDMX, 2012-2025</sup>"),
        xaxis=dict(gridcolor="#334155", tickmode="linear", dtick=1),
        yaxis=dict(gridcolor="#334155", title="% del total", range=[0, 50]),
        height=380,
    )
    return fig


# La columna de texto TRANSVERSAL cambia de redacción y de mayúsculas/minúsculas
# entre años para el mismo anexo (hasta 3 variantes por id_transversal) — se usa
# el id, estable, como llave en vez del texto libre.
_TRANSVERSAL_SHORT = {
    1: "Pueblos indígenas",
    2: "Desarrollo rural (PEC)",
    3: "Ciencia y tecnología",
    4: "Igualdad de género",
    5: "Tecnologías limpias",
    6: "Juventud",
    7: "Grupos vulnerables",
    8: "Niñez y adolescencia",
    10: "Cambio climático",
    11: "Prevención del delito",
    13: "Anticorrupción",
}


def fig_transversal_ranking() -> go.Figure:
    d = at_transversal.filter(pl.col("year") == SNAPSHOT_YEAR).sort("pagado")
    total = float(d["pagado"].sum())
    children_pct = float(d.filter(pl.col("id_transversal") == 8)["pagado"][0] / total * 100)
    colors = [FOCUS if i == 8 else CONTEXT for i in d["id_transversal"]]
    labels = [_TRANSVERSAL_SHORT.get(i, t) for i, t in zip(d["id_transversal"], d["transversal"])]
    fig = go.Figure(go.Bar(
        x=d["pagado"] / BILLON, y=labels, orientation="h", marker_color=colors,
        customdata=d["transversal"],
        hovertemplate="<b>%{customdata}</b><br>%{x:.2f} billones<extra></extra>",
    ))
    fig.update_layout(
        **CHART_LAYOUT,
        title=dict(text=f"<b>La niñez y adolescencia recibe la mayor prioridad transversal ({children_pct:.1f}%)</b>"
                        f"<br><sup style='color:#94A3B8'>Monto pagado por anexo transversal, billones de pesos, {SNAPSHOT_YEAR}</sup>"),
        yaxis=dict(gridcolor="rgba(0,0,0,0)", tickfont=dict(size=11)),
        xaxis=dict(gridcolor="#334155", title="Billones de pesos"),
        height=max(380, d.height * 32 + 100), margin=dict(l=190),
    )
    return fig


def fig_transversal_execution_slope() -> go.Figure:
    first_year = at_transversal.group_by("id_transversal").agg(pl.col("year").min().alias("y0"))
    y0_rows = at_transversal.join(first_year, on="id_transversal").filter(pl.col("year") == pl.col("y0"))
    y1_rows = at_transversal.filter(pl.col("year") == SNAPSHOT_YEAR)
    d = y0_rows.select("id_transversal", "transversal", "y0", pl.col("execution_ratio").alias("r0")).join(
        y1_rows.select("id_transversal", pl.col("execution_ratio").alias("r1")), on="id_transversal"
    ).sort("r1", descending=True)

    extremes = {d.head(1)["id_transversal"][0], d.tail(1)["id_transversal"][0]}
    fig = go.Figure()
    for row in d.iter_rows(named=True):
        color = GREEN if row["r1"] >= 100 else RED
        label = _TRANSVERSAL_SHORT.get(row["id_transversal"], row["transversal"][:20])
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[row["r0"], row["r1"]], mode="lines+markers",
            line=dict(color=color, width=1.5), marker=dict(color=color, size=8),
            showlegend=False,
            hovertemplate=f"<b>{row['transversal']}</b><br>{row['y0']}: %{{y:.1f}}%<br>{SNAPSHOT_YEAR}: %{{customdata:.1f}}%<extra></extra>",
            customdata=[row["r0"], row["r1"]],
        ))
        if row["id_transversal"] in extremes:
            fig.add_annotation(x=1.03, y=row["r1"], text=label, showarrow=False, xanchor="left",
                                font=dict(color="#CBD5E1", size=11))
    fig.add_hline(y=100, line_color="#64748B", line_width=1, line_dash="dot")
    n_under = int((d["r1"] < 100).sum())
    n_over = d.height - n_under
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines+markers", line=dict(color=GREEN), marker=dict(color=GREEN),
                              name=f"▲ Sobre-ejercido, {SNAPSHOT_YEAR} ({n_over})"))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines+markers", line=dict(color=RED), marker=dict(color=RED),
                              name=f"▼ Sub-ejercido, {SNAPSHOT_YEAR} ({n_under})"))
    fig.update_layout(
        **CHART_LAYOUT,
        title=dict(text=f"<b>{n_under} de {d.height} anexos transversales terminaron sub-ejercidos en {SNAPSHOT_YEAR}</b>"
                        "<br><sup style='color:#94A3B8'>Tasa de ejercicio (pagado/aprobado) por anexo, primer año disponible → 2024</sup>"),
        xaxis=dict(tickvals=[0, 1], ticktext=["primer año", str(SNAPSHOT_YEAR)],
                   range=[-0.15, 1.6], gridcolor="rgba(0,0,0,0)", showgrid=False, zeroline=False),
        yaxis=dict(gridcolor="#334155", title="% ejercido", ticksuffix="%"),
        legend=dict(orientation="h", y=-0.12, x=0),
        height=460, margin=dict(t=80, r=170),
    )
    return fig


def fig_subsidios_percapita_map() -> go.Figure:
    year = int(subsidios_percapita["year"].max())
    d = subsidios_percapita.filter(pl.col("year") == year)
    fig = px.choropleth_map(
        d, geojson=GEOJSON, locations="nom_geo", featureidkey="properties.name",
        color="monto_percapita",
        color_continuous_scale=[[0, "#1E293B"], [0.5, "#2E86AB"], [1, "#F4A261"]],
        hover_name="entidad", custom_data=["monto_percapita"],
        map_style="carto-darkmatter", center={"lat": 23.6, "lon": -102.5}, zoom=4,
    )
    fig.update_traces(hovertemplate="<b>%{hovertext}</b><br>$%{customdata[0]:,.0f} per cápita<extra></extra>")
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", font_color="#CBD5E1",
        title=dict(text="<b>El subsidio per cápita real favorece a los estados más pobres</b>"
                        f"<br><sup style='color:#94A3B8'>Subsidios ejercidos ÷ población (CONAPO), pesos por habitante, {year}</sup>"),
        margin=dict(t=70, b=0, l=0, r=0), height=580,
        coloraxis_colorbar=dict(title=dict(text="$/hab", font=dict(color="#CBD5E1")), tickfont=dict(color="#CBD5E1")),
    )
    return fig


def fig_subsidios_percapita_slope() -> go.Figure:
    y0, y1 = int(subsidios_percapita["year"].min()), int(subsidios_percapita["year"].max())
    d0 = subsidios_percapita.filter(pl.col("year") == y0).select("nom_geo", pl.col("monto_percapita").alias("p0"))
    d1 = subsidios_percapita.filter(pl.col("year") == y1).select("nom_geo", pl.col("monto_percapita").alias("p1"))
    d = d0.join(d1, on="nom_geo").sort("p1", descending=True)
    n_up = int((d["p1"] > d["p0"]).sum())
    n_dn = int((d["p1"] <= d["p0"]).sum())

    fig = go.Figure()
    for row in d.iter_rows(named=True):
        color = GREEN if row["p1"] > row["p0"] else RED
        fig.add_trace(go.Scatter(
            x=[str(y0), str(y1)], y=[row["p0"], row["p1"]], mode="lines+markers",
            line=dict(color=color, width=1.5), marker=dict(color=color, size=7),
            showlegend=False,
            hovertemplate=f"<b>{row['nom_geo']}</b><br>%{{x}}: $%{{y:,.0f}}/hab<extra></extra>",
        ))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines+markers", line=dict(color=GREEN), marker=dict(color=GREEN),
                              name=f"▲ Aumentó ({n_up})"))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines+markers", line=dict(color=RED), marker=dict(color=RED),
                              name=f"▼ Disminuyó ({n_dn})"))
    fig.update_xaxes(type="category", gridcolor="rgba(0,0,0,0)")
    fig.update_layout(
        **CHART_LAYOUT,
        title=dict(text=f"<b>{n_up} de {d.height} estados aumentaron su subsidio per cápita desde {y0}</b>"
                        f"<br><sup style='color:#94A3B8'>Subsidios ejercidos per cápita, pesos por habitante, {y0} → {y1}</sup>"),
        yaxis=dict(gridcolor="#334155", title="Pesos por habitante"),
        legend=dict(orientation="h", y=-0.12, x=0),
        height=460, margin=dict(t=80),
    )
    return fig


MIL_MILLONES = 1e9  # miles de millones de pesos (MMDP) — más legible que billones para estos montos


def fig_oil_share_trend() -> go.Figure:
    d = (
        pibe_oil_share.filter(pl.col("entidad").is_in([n for n, _ in PETRO_STATES.values()]))
        .join(_oil_nat, on="year")
        .with_columns((pl.col("oil_mdp") / pl.col("oil_nat") * 100).alias("share_pct"))
        .sort("year")
    )

    fig = go.Figure()
    for _, (nombre, color) in PETRO_STATES.items():
        sub = d.filter(pl.col("entidad") == nombre)
        fig.add_trace(go.Scatter(
            x=sub["year"], y=sub["share_pct"], mode="lines", name=nombre,
            stackgroup="one", line=dict(color=color, width=0.5),
            hovertemplate=f"<b>{nombre}</b><br>%{{x}}: %{{y:.1f}}% del PIB petrolero nacional<extra></extra>",
        ))

    c0 = float(_oil_combined["combined_pct"][0])
    fig.add_annotation(x=_oil_year0, y=c0, text=f"{c0:.0f}%", showarrow=True, ay=-25, font=dict(color="#F8FAFC"))
    fig.add_annotation(x=_oil_year1, y=_oil_combined_latest, text=f"{_oil_combined_latest:.0f}%",
                        showarrow=True, ay=-25, font=dict(color="#F8FAFC"))
    fig.update_layout(
        **CHART_LAYOUT,
        title=dict(text="<b>Juntos producen ~9 de cada 10 pesos del PIB petrolero del país</b>"
                        f"<br><sup style='color:#94A3B8'>% del PIB petrolero nacional (INEGI PIBE), acumulado Campeche + Tabasco, {_oil_year0}-{_oil_year1}</sup>"),
        xaxis=dict(gridcolor="#334155", tickmode="linear", dtick=2),
        yaxis=dict(gridcolor="#334155", title="% del PIB petrolero nacional", range=[0, 100]),
        legend=dict(orientation="h", y=-0.18, x=0),
        height=380, margin=dict(t=80, b=70),
    )
    return fig


def fig_transfers_by_state() -> go.Figure:
    fig = go.Figure()
    for id_entidad, (nombre, color) in PETRO_STATES.items():
        d = cp_transfers.filter(pl.col("id_entidad") == id_entidad)
        for id_ramo, dash in [(28, "solid"), (33, "dash")]:
            sub = d.filter(pl.col("id_ramo") == id_ramo).sort("year")
            tipo = "Participaciones" if id_ramo == 28 else "Aportaciones"
            fig.add_trace(go.Scatter(
                x=sub["year"], y=sub["ejercido"] / MIL_MILLONES, mode="lines", name=f"{nombre} · {tipo}",
                line=dict(color=color, width=2, dash=dash),
                hovertemplate=f"<b>{nombre} · {tipo}</b><br>%{{x}}: %{{y:.1f}} mmdp<extra></extra>",
            ))
    d24 = cp_transfers.filter(pl.col("year") == SNAPSHOT_YEAR)
    tab_total = float(d24.filter(pl.col("id_entidad") == 27)["ejercido"].sum())
    camp_total = float(d24.filter(pl.col("id_entidad") == 4)["ejercido"].sum())
    mult = tab_total / camp_total
    fig.update_layout(
        **CHART_LAYOUT,
        title=dict(text=f"<b>Tabasco recibe {mult:.1f}× más transferencias que Campeche</b>"
                        f"<br><sup style='color:#94A3B8'>Participaciones (RAMO 28) y Aportaciones (RAMO 33) ejercidas, miles de millones de pesos (mmdp), 2012-2025</sup>"),
        xaxis=dict(gridcolor="#334155", tickmode="linear", dtick=1),
        yaxis=dict(gridcolor="#334155", title="Miles de millones de pesos"),
        legend=dict(orientation="h", y=-0.2, x=0),
        height=420, margin=dict(t=80, b=80),
    )
    return fig


def fig_produce_vs_receive_ratio() -> go.Figure:
    nat_transfer = cp_transfers.group_by("year").agg(pl.col("ejercido").sum().alias("nat_transfer"))
    oil_nat = pibe_oil_share.filter(pl.col("entidad") == "Estados Unidos Mexicanos").select(
        "year", pl.col("oil_mdp").alias("oil_nat")
    )

    fig = go.Figure()
    latest = {}
    for id_entidad, (nombre, color) in PETRO_STATES.items():
        transfer = (
            cp_transfers.filter(pl.col("id_entidad") == id_entidad)
            .group_by("year").agg(pl.col("ejercido").sum().alias("transfer"))
            .join(nat_transfer, on="year")
            .with_columns((pl.col("transfer") / pl.col("nat_transfer") * 100).alias("share_transfer"))
        )
        oil = pibe_oil_share.filter(pl.col("entidad") == nombre).select("year", "oil_mdp")
        d = (
            transfer.join(oil, on="year").join(oil_nat, on="year")
            .with_columns((pl.col("oil_mdp") / pl.col("oil_nat") * 100).alias("share_oil"))
            .with_columns((pl.col("share_transfer") / pl.col("share_oil")).alias("razon"))
            .sort("year")
        )
        fig.add_trace(go.Scatter(
            x=d["year"], y=d["razon"], mode="lines+markers", name=nombre,
            line=dict(color=color, width=2),
            hovertemplate=f"<b>{nombre}</b><br>%{{x}}: razón %{{y:.3f}}<extra></extra>",
        ))
        latest[nombre] = (int(d["year"][-1]), float(d["razon"][-1]))

    camp_y, camp_r = latest["Campeche"]
    tab_y, tab_r = latest["Tabasco"]
    fig.add_annotation(x=camp_y, y=camp_r, text=f"Campeche: {camp_r*100:.1f}%", showarrow=True, ay=-30,
                        font=dict(color=RED))
    fig.add_annotation(x=tab_y, y=tab_r, text=f"Tabasco: {tab_r*100:.1f}%", showarrow=True, ay=30,
                        font=dict(color=ORANGE))
    fig.update_layout(
        **CHART_LAYOUT,
        title=dict(text="<b>Por cada peso que su petróleo pesa en el PIB nacional, reciben centavos en transferencias</b>"
                        "<br><sup style='color:#94A3B8'>Razón: % nacional de transferencias (R28+R33) recibidas ÷ % nacional del PIB petrolero producido — paridad = 1.0, 2012-2024</sup>"),
        xaxis=dict(gridcolor="#334155", tickmode="linear", dtick=1),
        yaxis=dict(gridcolor="#334155", title="Razón (recibido / producido)", rangemode="tozero"),
        legend=dict(orientation="h", y=-0.18, x=0),
        height=420, margin=dict(t=90, b=70),
    )
    return fig


# ── app ──────────────────────────────────────────────────────────────────────
app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY], title="Cuenta Pública · SHCP")


def kpi(value, title, sub=""):
    return dbc.Col(html.Div([
        html.Div(str(value), style={"fontSize": "2rem", "fontWeight": "700", "color": "#F8FAFC"}),
        html.Div(title, style={"fontSize": "0.85rem", "color": "#94A3B8", "marginTop": "2px"}),
        html.Div(sub, style={"fontSize": "0.75rem", "color": "#64748B"}) if sub else None,
    ], style=CARD_STYLE), md=3)


app.layout = html.Div(style={"backgroundColor": "#0F172A", "minHeight": "100vh", "padding": "24px"}, children=[
    html.H1("Cuenta Pública Federal 2011–2025", style={"color": "#F8FAFC", "fontWeight": "700", "marginBottom": "4px"}),
    html.P("SHCP · Transparencia Presupuestaria · ejercicio del gasto, billones de pesos corrientes",
           style={"color": "#94A3B8", "marginBottom": "24px"}),

    dbc.Row([
        kpi(f"{_tot_2024['ejercido'] / BILLON:.2f} B", "Ejercido 2024", f"×{_growth_mult:.1f} vs. 2012"),
        kpi(f"{_mean_rate:.1f}%", "Tasa de ejercicio promedio", f"{_n_over100} de {_n_years} años > 100%"),
        kpi(f"{_top10_share:.1f}%", "Top 10 RAMOs / gasto total", f"Pensiones: {_pensiones_pct:.1f}% del gasto"),
        kpi(f"{_cdmx_pct:.1f}%", "Gasto registrado en CDMX", "Artefacto administrativo, no entrega real"),
    ], className="g-2 mb-4"),

    dcc.Tabs(style={"marginBottom": "16px"}, children=[
        dcc.Tab(label="Tendencia", style=TAB_STYLE, selected_style=TAB_SEL, children=[
            dbc.Row([
                dbc.Col(dcc.Graph(figure=fig_trend_total(), config=GRAPH_CONFIG), md=12),
            ], className="mt-3"),
            dbc.Row([
                dbc.Col(dcc.Graph(figure=fig_execution_rate(), config=GRAPH_CONFIG), md=12),
            ], className="mt-3"),
        ]),
        dcc.Tab(label="Concentración", style=TAB_STYLE, selected_style=TAB_SEL, children=[
            dbc.Row([
                dbc.Col(dcc.Graph(figure=fig_ramo_ranking(), config=GRAPH_CONFIG), md=12),
            ], className="mt-3"),
            dbc.Row([
                dbc.Col(dcc.Graph(figure=fig_tipogasto_stacked(), config=GRAPH_CONFIG), md=12),
            ], className="mt-3"),
        ]),
        dcc.Tab(label="Geografía", style=TAB_STYLE, selected_style=TAB_SEL, children=[
            dbc.Row([
                dbc.Col(dcc.Graph(figure=fig_geo_ranking(), config=GRAPH_CONFIG), md=6),
                dbc.Col(dcc.Graph(figure=fig_cdmx_share_trend(), config=GRAPH_CONFIG), md=6),
            ], className="mt-3"),
        ]),
        dcc.Tab(label="Transversales", style=TAB_STYLE, selected_style=TAB_SEL, children=[
            dbc.Row([
                dbc.Col(dcc.Graph(figure=fig_transversal_ranking(), config=GRAPH_CONFIG), md=12),
            ], className="mt-3"),
            dbc.Row([
                dbc.Col(dcc.Graph(figure=fig_transversal_execution_slope(), config=GRAPH_CONFIG), md=12),
            ], className="mt-3"),
        ]),
        dcc.Tab(label="Subsidios (per cápita)", style=TAB_STYLE, selected_style=TAB_SEL, children=[
            dbc.Row([
                dbc.Col(dcc.Graph(figure=fig_subsidios_percapita_map(), config=GRAPH_CONFIG), md=7),
                dbc.Col(dcc.Graph(figure=fig_subsidios_percapita_slope(), config=GRAPH_CONFIG), md=5),
            ], className="mt-3"),
        ]),
        dcc.Tab(label="Tabasco y Campeche", style=TAB_STYLE, selected_style=TAB_SEL, children=[
            dbc.Row([
                dbc.Col(html.Div([
                    html.P(
                        f"Campeche y Tabasco producen en conjunto {_oil_combined_latest:.0f}% del PIB petrolero "
                        f"nacional ({_oil_year1}) — una proporción estable, siempre entre {_oil_combined_min:.0f}% "
                        f"y {_oil_combined_max:.0f}%, desde {_oil_year0}. Las siguientes gráficas comparan esa "
                        "producción con lo que ambos estados reciben de vuelta en transferencias federales.",
                        style={"color": "#94A3B8", "fontSize": "0.9rem", "lineHeight": "1.6", "margin": "0"},
                    ),
                ], style={**CARD_STYLE, "textAlign": "left"}), md=12),
            ], className="mt-3"),
            dbc.Row([
                dbc.Col(dcc.Graph(figure=fig_oil_share_trend(), config=GRAPH_CONFIG), md=12),
            ], className="mt-3"),
            dbc.Row([
                dbc.Col(dcc.Graph(figure=fig_transfers_by_state(), config=GRAPH_CONFIG), md=12),
            ], className="mt-3"),
            dbc.Row([
                dbc.Col(dcc.Graph(figure=fig_produce_vs_receive_ratio(), config=GRAPH_CONFIG), md=12),
            ], className="mt-3"),
            dbc.Row([
                dbc.Col(html.Div([
                    html.P("Cómo leer esta razón:", style={"color": "#CBD5E1", "fontWeight": "600", "marginBottom": "6px"}),
                    html.P(
                        "Cada año se calcula el % que Campeche o Tabasco representan del total nacional de "
                        "transferencias (Participaciones + Aportaciones) y se divide entre el % que representan "
                        "del PIB petrolero nacional (INEGI PIBE). Un valor de 1.0 significaría paridad: la porción "
                        "de transferencias que reciben iguala a la porción del petróleo del país que producen. "
                        "Los valores reales — 2% para Campeche y 5-10% para Tabasco — muestran que, pese a producir "
                        "la mayoría del PIB petrolero nacional, ambos estados reciben de vuelta solo una fracción "
                        "mínima en transferencias federales, porque estas se reparten por fórmulas de población e "
                        "ingresos, no por origen del recurso extraído.",
                        style={"color": "#94A3B8", "fontSize": "0.85rem", "lineHeight": "1.6", "margin": "0"},
                    ),
                ], style={**CARD_STYLE, "textAlign": "left"}), md=12),
            ], className="mt-3"),
        ]),
    ]),
])


if __name__ == "__main__":
    app.run(debug=True, port=8066)
