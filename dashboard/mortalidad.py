"""
Dashboard: Mortalidad — México (2010–2026)
Fuente: INEGI / CONAPO

Tab 1 — Esperanza de vida (life expectancy) by state and sex, 2010-2026
Tab 2 — Defunciones registradas (all causes) by state and sex, 2010-2024
Tab 3 — Diabetes mellitus: deaths by state, age group, and sex, 2010-2024

Run: uv run python dashboard/mortalidad.py
"""

import json
import polars as pl
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc

# ── constants ─────────────────────────────────────────────────────────────────
NAME_MAP = {
    "Coahuila de Zaragoza": "Coahuila",
    "Michoacán de Ocampo": "Michoacán",
    "Veracruz de Ignacio de la Llave": "Veracruz",
}
NACIONAL_EV   = "Estados Unidos Mexicanos"
NACIONAL_DEF  = "Total"
SKIP_ENTITIES = {"Extranjero", "No especificado"}

AGE_ORDER = [
    "0 a 4 años", "5 a 9 años", "10 a 14 años", "15 a 19 años",
    "20 a 24 años", "25 a 29 años", "30 a 34 años", "35 a 39 años",
    "40 a 44 años", "45 a 49 años", "50 a 54 años", "55 a 59 años",
    "60 a 64 años", "65 a 69 años", "70 a 74 años", "75 a 79 años",
    "80 a 84 años", "85 años y más",
]


# ── data loading ──────────────────────────────────────────────────────────────
def _parse_wide(path: str, data_row_start: int, n_data_rows: int,
                years: list[str], sexos: list[str]) -> pl.DataFrame:
    raw = pl.read_excel(path, has_header=False)
    row5 = raw.row(5)
    row6 = raw.row(6)
    cols = ["entidad" if i == 0 else f"{row5[i]}_{row6[i]}" for i in range(len(row5))]
    data = raw.slice(data_row_start, n_data_rows)
    data.columns = cols
    rows = []
    for r in data.iter_rows(named=True):
        entidad = r["entidad"]
        if not entidad or entidad in SKIP_ENTITIES:
            continue
        entidad_geo = NAME_MAP.get(entidad, entidad)
        for yr in years:
            for sexo in sexos:
                val = r.get(f"{yr}_{sexo}")
                if val is not None:
                    try:
                        rows.append({"entidad": entidad, "entidad_geo": entidad_geo,
                                     "año": int(yr), "sexo": sexo, "valor": float(val)})
                    except (ValueError, TypeError):
                        pass
    return pl.DataFrame(rows)


def _parse_dm() -> pl.DataFrame:
    raw = pl.read_excel("data/mortalidad/diabetes_2026.xlsx", has_header=False)
    row5 = raw.row(5)
    row6 = raw.row(6)
    cols = ["entidad" if i == 0 else ("grupo_edad" if i == 1 else f"{row5[i]}_{row6[i]}")
            for i in range(len(row5))]
    years = [str(y) for y in range(2010, 2025)]
    rows = []
    for i in range(7, len(raw)):
        r = dict(zip(cols, raw.row(i)))
        ent = r.get("entidad")
        gr  = r.get("grupo_edad")
        if not ent or ent in SKIP_ENTITIES:
            continue
        if not gr or gr == "No especificado":
            continue
        if str(ent).startswith(("Unidad", "Nota", "Fuente", "Fecha", "Defunc")):
            continue
        for yr in years:
            for sexo in ["Total", "Hombres", "Mujeres"]:
                val = r.get(f"{yr}_{sexo}")
                if val is not None:
                    try:
                        rows.append({"entidad": ent, "grupo_edad": gr,
                                     "año": int(yr), "sexo": sexo,
                                     "muertes_dm": int(float(val))})
                    except (ValueError, TypeError):
                        pass
    return pl.DataFrame(rows)


df_ev = _parse_wide(
    "data/mortalidad/esperanza_vida_2026.xlsx",
    data_row_start=7, n_data_rows=33,
    years=[str(y) for y in range(2010, 2027)],
    sexos=["Total", "Hombres", "Mujeres"],
)

df_def = _parse_wide(
    "data/mortalidad/mortalidad_entidad_2026.xlsx",
    data_row_start=7, n_data_rows=35,
    years=[str(y) for y in range(2010, 2025)],
    sexos=["Total", "Hombres", "Mujeres"],
)

df_dm = _parse_dm()

with open("data/mexico_states.geojson") as f:
    GEO = json.load(f)


# ── pre-computed insight values ───────────────────────────────────────────────
def _ev_nac(sexo: str, año: int) -> float:
    return float(df_ev.filter(
        (pl.col("entidad") == NACIONAL_EV) & (pl.col("año") == año) & (pl.col("sexo") == sexo)
    )["valor"][0])

def _def_nac(año: int) -> float:
    return float(df_def.filter(
        (pl.col("entidad") == NACIONAL_DEF) & (pl.col("año") == año) & (pl.col("sexo") == "Total")
    )["valor"][0])

def _dm_nac(año: int, sexo: str = "Total") -> int:
    return int(df_dm.filter(
        (pl.col("entidad") == "Total") & (pl.col("año") == año)
        & (pl.col("sexo") == sexo) & (pl.col("grupo_edad") == "Total")
    )["muertes_dm"][0])

# EV
KPI_2026 = {s: _ev_nac(s, 2026) for s in ["Total", "Hombres", "Mujeres"]}
KPI_2010 = {s: _ev_nac(s, 2010) for s in ["Total", "Hombres", "Mujeres"]}

EV_2019_TOTAL = _ev_nac("Total", 2019)
EV_2021_TOTAL = _ev_nac("Total", 2021)
EV_2021_HOM   = _ev_nac("Hombres", 2021)
EV_2021_MUJ   = _ev_nac("Mujeres", 2021)
EV_COVID_DROP = EV_2021_TOTAL - EV_2019_TOTAL
GENDER_GAP    = KPI_2026["Mujeres"] - KPI_2026["Hombres"]

_states_2026 = df_ev.filter(
    (pl.col("entidad") != NACIONAL_EV) & (pl.col("año") == 2026) & (pl.col("sexo") == "Total")
)
EV_TOP = _states_2026.sort("valor", descending=True).row(0, named=True)
EV_BOT = _states_2026.sort("valor").row(0, named=True)
STATE_GAP = EV_TOP["valor"] - EV_BOT["valor"]

_ev_pre  = df_ev.filter((pl.col("año") == 2019) & (pl.col("entidad") != NACIONAL_EV) & (pl.col("sexo") == "Total")).rename({"valor": "v19"})
_ev_post = df_ev.filter((pl.col("año") == 2021) & (pl.col("entidad") != NACIONAL_EV) & (pl.col("sexo") == "Total")).rename({"valor": "v21"})
_ev_cov  = _ev_pre.join(_ev_post, on=["entidad", "entidad_geo", "sexo"]).with_columns((pl.col("v21") - pl.col("v19")).alias("delta")).sort("delta")
WORST_EV_STATE = _ev_cov["entidad"][0]
WORST_EV_DELTA = float(_ev_cov["delta"][0])

# Defunciones
DEF_2019 = _def_nac(2019)
DEF_2021 = _def_nac(2021)
DEF_2024 = _def_nac(2024)
DEF_PCT  = (DEF_2021 - DEF_2019) / DEF_2019 * 100

_def_pre  = df_def.filter((pl.col("año") == 2019) & (pl.col("entidad") != NACIONAL_DEF) & (pl.col("sexo") == "Total")).rename({"valor": "v19"})
_def_post = df_def.filter((pl.col("año") == 2021) & (pl.col("entidad") != NACIONAL_DEF) & (pl.col("sexo") == "Total")).rename({"valor": "v21"})
_def_cov  = _def_pre.join(_def_post, on=["entidad", "entidad_geo"]).with_columns(((pl.col("v21") - pl.col("v19")) / pl.col("v19") * 100).alias("pct")).sort("pct", descending=True)
WORST_DEF_STATE = _def_cov["entidad"][0]
WORST_DEF_PCT   = float(_def_cov["pct"][0])

# Diabetes
DM_2019 = _dm_nac(2019)
DM_2020 = _dm_nac(2020)
DM_2024 = _dm_nac(2024)
DM_COVID_PCT = (DM_2020 - DM_2019) / DM_2019 * 100

DM_PCT_ALL = DM_2024 / DEF_2024 * 100

_dm_state = df_dm.filter((pl.col("año") == 2024) & (pl.col("sexo") == "Total") & (pl.col("grupo_edad") == "Total") & (pl.col("entidad") != "Total"))
_all_state = df_def.filter((pl.col("año") == 2024) & (pl.col("sexo") == "Total") & (pl.col("entidad") != "Total")).rename({"valor": "total"}).select(["entidad", "total"])
_pct_state = _dm_state.join(_all_state, on="entidad").with_columns((pl.col("muertes_dm") / pl.col("total") * 100).alias("pct")).sort("pct", descending=True)
DM_WORST_STATE = _pct_state["entidad"][0]
DM_WORST_PCT   = float(_pct_state["pct"][0])

_dm_hom_2019 = _dm_nac(2019, "Hombres")
_dm_muj_2019 = _dm_nac(2019, "Mujeres")
_dm_hom_2020 = _dm_nac(2020, "Hombres")
_dm_muj_2020 = _dm_nac(2020, "Mujeres")
DM_RATIO_2019 = _dm_hom_2019 / _dm_muj_2019
DM_RATIO_2020 = _dm_hom_2020 / _dm_muj_2020

DM_85_2010 = int(df_dm.filter((pl.col("entidad") == "Total") & (pl.col("sexo") == "Total") & (pl.col("grupo_edad") == "85 años y más") & (pl.col("año") == 2010))["muertes_dm"][0])
DM_85_2024 = int(df_dm.filter((pl.col("entidad") == "Total") & (pl.col("sexo") == "Total") & (pl.col("grupo_edad") == "85 años y más") & (pl.col("año") == 2024))["muertes_dm"][0])
DM_85_GROWTH = (DM_85_2024 - DM_85_2010) / DM_85_2010 * 100

N_STATES  = 32
CHART_H_S = max(420, N_STATES * 24 + 80)


# ── theme ─────────────────────────────────────────────────────────────────────
CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font_color="#CBD5E1",
)
GRID      = "#334155"
GRID_NONE = "rgba(0,0,0,0)"
CARD_STYLE = {
    "background": "#1E293B", "border": "1px solid #334155",
    "borderRadius": "8px", "padding": "16px", "textAlign": "center",
}
TAB_STYLE = {"backgroundColor": "#0F172A", "color": "#94A3B8", "borderTop": "none"}
TAB_SEL   = {"backgroundColor": "#1E293B", "color": "#F8FAFC",
             "borderTop": "2px solid #2E86AB", "fontWeight": "600"}

GRAPH_CONFIG = {
    "displayModeBar": "hover",
    "modeBarButtonsToRemove": [
        "zoom2d", "pan2d", "select2d", "lasso2d",
        "zoomIn2d", "zoomOut2d", "autoScale2d", "resetScale2d",
        "hoverClosestCartesian", "hoverCompareCartesian", "toggleSpikelines",
    ],
    "toImageButtonOptions": {"format": "png", "scale": 2},
}

COLOR_TOTAL   = "#3BB273"
COLOR_HOMBRES = "#2E86AB"
COLOR_MUJERES = "#F4A261"
COLORS = {"Total": COLOR_TOTAL, "Hombres": COLOR_HOMBRES, "Mujeres": COLOR_MUJERES}


# ── figure factories — esperanza de vida ──────────────────────────────────────
def fig_ev_trend(_: pl.DataFrame) -> go.Figure:
    nac = df_ev.filter(pl.col("entidad") == NACIONAL_EV).sort("año")
    fig = go.Figure()
    for sexo, color in COLORS.items():
        sub = nac.filter(pl.col("sexo") == sexo)
        fig.add_trace(go.Scatter(
            x=sub["año"].to_list(), y=sub["valor"].to_list(),
            name=sexo, mode="lines+markers",
            line=dict(color=color, width=2.5), marker=dict(size=5),
            hovertemplate=f"<b>{sexo}</b>  %{{x}}: %{{y:.1f}} años<extra></extra>",
        ))
    fig.add_vline(x=2021, line_width=1, line_dash="dash", line_color="#E84855", opacity=0.7)
    fig.add_annotation(x=2021, y=0.97, xref="x", yref="paper",
                       text="Mínimo COVID", showarrow=False, textangle=-90,
                       font=dict(size=10, color="#E84855"), yanchor="top", xanchor="right")
    for sexo, val, ax in [("Total", EV_2021_TOTAL, 38), ("Hombres", EV_2021_HOM, -38), ("Mujeres", EV_2021_MUJ, 38)]:
        fig.add_annotation(
            x=2021, y=val, xref="x", yref="y",
            text=f"{val:.1f}", showarrow=True, arrowhead=2, arrowwidth=1,
            arrowcolor=COLORS[sexo], font=dict(size=10, color=COLORS[sexo]),
            ax=ax, ay=30, bgcolor="#0F172A", borderpad=3,
        )
    fig.update_layout(
        **CHART_LAYOUT, height=380,
        title="Esperanza de vida nacional — 2010 a 2026",
        xaxis=dict(gridcolor=GRID, dtick=2),
        yaxis=dict(gridcolor=GRID, title="Años", range=[62, 83]),
        legend=dict(orientation="h", y=1.12, x=0),
        margin=dict(t=60, b=40, l=60, r=20),
    )
    return fig


def fig_ev_ranking(d: pl.DataFrame) -> go.Figure:
    sexo    = d["sexo"][0]
    states  = d.filter((pl.col("entidad") != NACIONAL_EV) & (pl.col("año") == 2026)).sort("valor")
    nac_val = float(df_ev.filter(
        (pl.col("entidad") == NACIONAL_EV) & (pl.col("año") == 2026) & (pl.col("sexo") == sexo)
    )["valor"][0])
    fig = go.Figure(go.Bar(
        x=states["valor"].to_list(), y=states["entidad"].to_list(),
        orientation="h", marker_color=COLORS[sexo],
        text=states["valor"].to_list(), textposition="outside", texttemplate="%{x:.1f}",
        hovertemplate="<b>%{y}</b>: %{x:.1f} años<extra></extra>",
    ))
    fig.add_vline(x=nac_val, line_width=1.5, line_dash="dot", line_color="#94A3B8",
                  annotation_text=f"Nacional {nac_val:.1f}",
                  annotation_position="top right", annotation_font_color="#94A3B8")
    fig.update_layout(
        **CHART_LAYOUT, height=CHART_H_S,
        title="Ranking estatal — Esperanza de vida 2026",
        xaxis=dict(gridcolor=GRID, range=[60, 82], title="Años"),
        yaxis=dict(gridcolor=GRID_NONE),
        margin=dict(t=50, b=40, l=185, r=60),
    )
    return fig


def fig_ev_map(d: pl.DataFrame) -> go.Figure:
    sexo   = d["sexo"][0]
    states = d.filter((pl.col("entidad") != NACIONAL_EV) & (pl.col("año") == 2026))
    fig = px.choropleth_map(
        states, geojson=GEO,
        locations="entidad_geo", color="valor", featureidkey="properties.name",
        color_continuous_scale="RdYlGn",
        range_color=[states["valor"].min() - 0.5, states["valor"].max() + 0.5],
        hover_name="entidad",
        hover_data={"valor": ":.1f", "entidad_geo": False},
        labels={"valor": "Años"},
        map_style="carto-darkmatter",
        center={"lat": 23.6, "lon": -102.5}, zoom=4,
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", font_color="#CBD5E1",
        height=500, title=f"Esperanza de vida por entidad — 2026 ({sexo})",
        coloraxis_colorbar=dict(title="Años"),
        margin=dict(t=50, b=10, l=0, r=0),
    )
    return fig


def fig_ev_progress(d: pl.DataFrame) -> go.Figure:
    sexo = d["sexo"][0]
    y0, y1 = 2010, 2026
    pre  = d.filter((pl.col("año") == y0) & (pl.col("entidad") != NACIONAL_EV)).rename({"valor": "v0"})
    post = d.filter((pl.col("año") == y1) & (pl.col("entidad") != NACIONAL_EV)).rename({"valor": "v1"})
    comp = (
        pre.join(post, on=["entidad", "entidad_geo", "sexo"])
        .with_columns(((pl.col("v1") - pl.col("v0")).round(2)).alias("delta"))
        .sort("delta")
    )
    nac_v0 = float(df_ev.filter(
        (pl.col("entidad") == NACIONAL_EV) & (pl.col("año") == y0) & (pl.col("sexo") == sexo)
    )["valor"][0])
    nac_v1 = float(df_ev.filter(
        (pl.col("entidad") == NACIONAL_EV) & (pl.col("año") == y1) & (pl.col("sexo") == sexo)
    )["valor"][0])
    nac_delta = round(nac_v1 - nac_v0, 2)
    v0_vals = comp["v0"].to_list()
    fig = go.Figure(go.Bar(
        x=comp["delta"].to_list(), y=comp["entidad"].to_list(),
        orientation="h",
        marker=dict(
            color=v0_vals,
            colorscale="RdYlGn",
            cmin=min(v0_vals) - 0.5,
            cmax=max(v0_vals) + 0.5,
            showscale=True,
            colorbar=dict(
                title=dict(text="EV 2010<br>(línea base)", side="right",
                           font=dict(color="#94A3B8", size=11)),
                thickness=14, x=1.01,
                tickfont=dict(color="#CBD5E1"),
            ),
        ),
        text=comp["delta"].to_list(), textposition="outside", texttemplate="%{x:+.2f}",
        customdata=list(zip(v0_vals, comp["v1"].to_list())),
        hovertemplate=(
            "<b>%{y}</b><br>"
            "2010: %{customdata[0]:.2f} años<br>"
            "2026: %{customdata[1]:.2f} años<br>"
            "Ganancia: %{x:+.2f} años<extra></extra>"
        ),
    ))
    fig.add_vline(x=nac_delta, line_width=1.5, line_dash="dot", line_color="#94A3B8",
                  annotation_text=f"Nacional {nac_delta:+.2f}",
                  annotation_position="top right", annotation_font_color="#94A3B8")
    fig.update_layout(
        **CHART_LAYOUT, height=CHART_H_S,
        title=f"Ganancia en esperanza de vida por entidad — 2010 → 2026 ({sexo})<br>"
              f"<sup style='color:#64748B'>Color = EV en 2010: verde oscuro = base alta (más difícil mejorar), rojo = base baja</sup>",
        xaxis=dict(gridcolor=GRID, title="Años ganados"),
        yaxis=dict(gridcolor=GRID_NONE),
        margin=dict(t=65, b=40, l=185, r=80),
    )
    return fig


def fig_ev_scatter(d: pl.DataFrame) -> go.Figure:
    sexo = d["sexo"][0]
    y0, y1 = 2010, 2026
    pre  = d.filter((pl.col("año") == y0) & (pl.col("entidad") != NACIONAL_EV)).rename({"valor": "v0"})
    post = d.filter((pl.col("año") == y1) & (pl.col("entidad") != NACIONAL_EV)).rename({"valor": "v1"})
    comp = (
        pre.join(post, on=["entidad", "entidad_geo", "sexo"])
        .with_columns(((pl.col("v1") - pl.col("v0")).round(2)).alias("delta"))
    )
    mean_ev = float(comp["v1"].mean())
    fig = go.Figure(go.Scatter(
        x=comp["delta"].to_list(), y=comp["v1"].to_list(),
        mode="markers+text",
        text=comp["entidad"].to_list(),
        textposition="top center",
        textfont=dict(size=9, color="#94A3B8"),
        marker=dict(color=COLORS[sexo], size=9, opacity=0.85,
                    line=dict(color="#0F172A", width=1)),
        customdata=comp["v0"].to_list(),
        hovertemplate=(
            "<b>%{text}</b><br>"
            "2010: %{customdata:.2f} años<br>"
            "2026: %{y:.2f} años<br>"
            "Ganancia: %{x:+.2f} años<extra></extra>"
        ),
    ))
    fig.add_hline(y=mean_ev, line_width=1.5, line_dash="dot", line_color="#94A3B8",
                  annotation_text=f"Media {mean_ev:.2f} años",
                  annotation_position="right", annotation_font_color="#94A3B8")
    fig.update_layout(
        **CHART_LAYOUT, height=520,
        title=f"Esperanza de vida 2026 vs. ganancia 2010→2026 por entidad ({sexo})",
        xaxis=dict(gridcolor=GRID, title="Ganancia (años)"),
        yaxis=dict(gridcolor=GRID, title="Esperanza de vida 2026 (años)"),
        margin=dict(t=50, b=50, l=60, r=20),
    )
    return fig


def fig_ev_covid(d: pl.DataFrame) -> go.Figure:
    sexo = d["sexo"][0]
    pre  = d.filter((pl.col("año") == 2019) & (pl.col("entidad") != NACIONAL_EV)).rename({"valor": "v19"})
    post = d.filter((pl.col("año") == 2021) & (pl.col("entidad") != NACIONAL_EV)).rename({"valor": "v21"})
    comp = (
        pre.join(post, on=["entidad", "entidad_geo", "sexo"])
        .with_columns((pl.col("v21") - pl.col("v19")).alias("delta"))
        .sort("delta")
    )
    labels = comp["entidad"].to_list()
    starts = comp["v19"].to_list()
    ends   = comp["v21"].to_list()
    deltas = comp["delta"].to_list()
    x_lines, y_lines = [], []
    for s, e, lbl in zip(starts, ends, labels):
        x_lines += [s, e, None]
        y_lines += [lbl, lbl, None]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_lines, y=y_lines, mode="lines",
                              line=dict(color="#E84855", width=1.5),
                              showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=starts, y=labels, mode="markers", name="2019",
                              marker=dict(color="#94A3B8", size=9, symbol="circle-open",
                                          line=dict(color="#94A3B8", width=2)),
                              hovertemplate="<b>%{y}</b><br>2019: %{x:.1f} años<extra></extra>"))
    fig.add_trace(go.Scatter(x=ends, y=labels, mode="markers", name="2021 (mínimo COVID)",
                              marker=dict(color="#E84855", size=9),
                              customdata=deltas,
                              hovertemplate="<b>%{y}</b><br>2021: %{x:.1f} años  (Δ %{customdata:.1f})<extra></extra>"))
    fig.update_layout(
        **CHART_LAYOUT, height=CHART_H_S,
        title=f"Impacto COVID-19: caída de esperanza de vida 2019→2021 ({sexo})",
        xaxis=dict(gridcolor=GRID, title="Años", range=[58, 82]),
        yaxis=dict(gridcolor=GRID_NONE),
        legend=dict(orientation="h", y=-0.06, x=0),
        margin=dict(t=50, b=60, l=185, r=20),
    )
    return fig


# ── figure factories — defunciones ────────────────────────────────────────────
def fig_def_trend(_: pl.DataFrame) -> go.Figure:
    nac = df_def.filter(pl.col("entidad") == NACIONAL_DEF).sort("año")
    fig = go.Figure()
    for sexo, color in COLORS.items():
        sub = nac.filter(pl.col("sexo") == sexo)
        fig.add_trace(go.Scatter(
            x=sub["año"].to_list(), y=sub["valor"].to_list(),
            name=sexo, mode="lines+markers",
            line=dict(color=color, width=2.5), marker=dict(size=5),
            hovertemplate=f"<b>{sexo}</b>  %{{x}}: %{{y:,.0f}}<extra></extra>",
        ))
    fig.add_vline(x=2021, line_width=1, line_dash="dash", line_color="#E84855", opacity=0.7)
    fig.add_annotation(x=2021, y=0.97, xref="x", yref="paper",
                       text="Pico COVID", showarrow=False, textangle=-90,
                       font=dict(size=10, color="#E84855"), yanchor="top", xanchor="right")
    fig.add_annotation(
        x=2021, y=DEF_2021, xref="x", yref="y",
        text=f"+{DEF_PCT:.0f}%<br>vs 2019", showarrow=True, arrowhead=2, arrowwidth=1,
        arrowcolor="#E84855", font=dict(size=10, color="#E84855"),
        ax=50, ay=-40, bgcolor="#0F172A", borderpad=3,
    )
    fig.update_layout(
        **CHART_LAYOUT, height=380,
        title="Defunciones registradas nacionales — 2010 a 2024",
        xaxis=dict(gridcolor=GRID, dtick=2),
        yaxis=dict(gridcolor=GRID, title="Defunciones"),
        legend=dict(orientation="h", y=1.12, x=0),
        margin=dict(t=60, b=40, l=80, r=20),
    )
    return fig


def fig_def_covid(d: pl.DataFrame) -> go.Figure:
    sexo = d["sexo"][0]
    dfd  = df_def.filter(pl.col("sexo") == sexo)
    pre  = dfd.filter((pl.col("año") == 2019) & (pl.col("entidad") != NACIONAL_DEF)).rename({"valor": "v19"})
    post = dfd.filter((pl.col("año") == 2021) & (pl.col("entidad") != NACIONAL_DEF)).rename({"valor": "v21"})
    comp = (
        pre.join(post, on=["entidad", "entidad_geo"])
        .with_columns(((pl.col("v21") - pl.col("v19")) / pl.col("v19") * 100).alias("pct"))
        .sort("pct", descending=True)
    )
    colors = ["#E84855" if v > 0 else "#3BB273" for v in comp["pct"].to_list()]
    fig = go.Figure(go.Bar(
        x=comp["pct"].to_list(), y=comp["entidad"].to_list(),
        orientation="h", marker_color=colors,
        text=comp["pct"].to_list(), textposition="outside", texttemplate="%{x:+.1f}%",
        hovertemplate="<b>%{y}</b><br>Cambio: %{x:+.1f}%<extra></extra>",
    ))
    fig.add_vline(x=0, line_width=1, line_color="#475569")
    fig.update_layout(
        **CHART_LAYOUT, height=CHART_H_S,
        title=f"Exceso de defunciones 2019→2021 por entidad (%, {sexo})",
        xaxis=dict(gridcolor=GRID, title="Cambio porcentual"),
        yaxis=dict(gridcolor=GRID_NONE),
        margin=dict(t=50, b=40, l=185, r=80),
    )
    return fig


# ── figure factories — diabetes ───────────────────────────────────────────────
def fig_dm_trend(_: pl.DataFrame) -> go.Figure:
    """Diabetes deaths 2010-2024, all 3 sexes."""
    nac = df_dm.filter((pl.col("entidad") == "Total") & (pl.col("grupo_edad") == "Total")).sort("año")
    fig = go.Figure()
    for sexo, color in COLORS.items():
        sub = nac.filter(pl.col("sexo") == sexo)
        fig.add_trace(go.Scatter(
            x=sub["año"].to_list(), y=sub["muertes_dm"].to_list(),
            name=sexo, mode="lines+markers",
            line=dict(color=color, width=2.5), marker=dict(size=5),
            hovertemplate=f"<b>{sexo}</b>  %{{x}}: %{{y:,.0f}}<extra></extra>",
        ))
    fig.add_vline(x=2020, line_width=1, line_dash="dash", line_color="#E84855", opacity=0.7)
    fig.add_annotation(x=2020, y=0.97, xref="x", yref="paper",
                       text="Pico COVID", showarrow=False, textangle=-90,
                       font=dict(size=10, color="#E84855"), yanchor="top", xanchor="right")
    fig.add_annotation(
        x=2020, y=DM_2020, xref="x", yref="y",
        text=f"+{DM_COVID_PCT:.0f}%<br>vs 2019", showarrow=True, arrowhead=2, arrowwidth=1,
        arrowcolor="#E84855", font=dict(size=10, color="#E84855"),
        ax=50, ay=-40, bgcolor="#0F172A", borderpad=3,
    )
    fig.update_layout(
        **CHART_LAYOUT, height=360,
        title="Muertes por diabetes mellitus — México 2010 a 2024",
        xaxis=dict(gridcolor=GRID, dtick=2),
        yaxis=dict(gridcolor=GRID, title="Defunciones"),
        legend=dict(orientation="h", y=1.12, x=0),
        margin=dict(t=60, b=40, l=80, r=20),
    )
    return fig


def fig_dm_heatmap(d: pl.DataFrame) -> go.Figure:
    """Heatmap: diabetes deaths by year × age group (national, sex-filtered)."""
    sexo = d["sexo"][0]
    sub  = df_dm.filter(
        (pl.col("entidad") == "Total") & (pl.col("sexo") == sexo)
        & pl.col("grupo_edad").is_in(AGE_ORDER)
    )
    years = list(range(2010, 2025))
    z = []
    for age in AGE_ORDER:
        row_data = []
        for yr in years:
            v = sub.filter((pl.col("grupo_edad") == age) & (pl.col("año") == yr))
            row_data.append(int(v["muertes_dm"][0]) if len(v) > 0 else 0)
        z.append(row_data)
    fig = go.Figure(go.Heatmap(
        x=[str(yr) for yr in years],
        y=AGE_ORDER,
        z=z,
        colorscale="YlOrRd",
        hovertemplate="<b>%{y}</b><br>%{x}: %{z:,} muertes<extra></extra>",
        colorbar=dict(title="Muertes"),
    ))
    fig.update_layout(
        **CHART_LAYOUT, height=520,
        title=f"Muertes por diabetes por grupo de edad — {sexo} (2010–2024)",
        xaxis=dict(gridcolor=GRID, title="Año"),
        yaxis=dict(gridcolor=GRID_NONE, title="Grupo de edad"),
        margin=dict(t=50, b=50, l=120, r=20),
    )
    return fig


def fig_dm_pct(d: pl.DataFrame) -> go.Figure:
    """Horizontal bar: diabetes as % of all deaths by state (2024)."""
    sexo  = d["sexo"][0]
    dm_s  = df_dm.filter((pl.col("año") == 2024) & (pl.col("sexo") == sexo)
                          & (pl.col("grupo_edad") == "Total") & (pl.col("entidad") != "Total"))
    all_s = (df_def.filter((pl.col("año") == 2024) & (pl.col("sexo") == sexo)
                            & (pl.col("entidad") != "Total"))
             .rename({"valor": "total"}).select(["entidad", "total"]))
    pct = (
        dm_s.join(all_s, on="entidad")
        .with_columns((pl.col("muertes_dm") / pl.col("total") * 100).alias("pct"))
        .sort("pct")
    )
    dm_nac  = float(df_dm.filter((pl.col("año") == 2024) & (pl.col("sexo") == sexo)
                                  & (pl.col("grupo_edad") == "Total") & (pl.col("entidad") == "Total"))["muertes_dm"][0])
    all_nac = float(df_def.filter((pl.col("año") == 2024) & (pl.col("sexo") == sexo)
                                   & (pl.col("entidad") == "Total"))["valor"][0])
    nac_pct = dm_nac / all_nac * 100
    fig = go.Figure(go.Bar(
        x=pct["pct"].to_list(), y=pct["entidad"].to_list(),
        orientation="h", marker_color=COLOR_MUJERES,
        text=pct["pct"].to_list(), textposition="outside", texttemplate="%{x:.1f}%",
        hovertemplate="<b>%{y}</b>: %{x:.1f}%<extra></extra>",
    ))
    fig.add_vline(x=nac_pct, line_width=1.5, line_dash="dot", line_color="#94A3B8",
                  annotation_text=f"Nacional {nac_pct:.1f}%",
                  annotation_position="top right", annotation_font_color="#94A3B8")
    fig.update_layout(
        **CHART_LAYOUT, height=CHART_H_S,
        title=f"Diabetes como % de todas las muertes — 2024 ({sexo})",
        xaxis=dict(gridcolor=GRID, title="% de defunciones totales"),
        yaxis=dict(gridcolor=GRID_NONE),
        margin=dict(t=50, b=40, l=185, r=70),
    )
    return fig


def fig_dm_covid_state(d: pl.DataFrame) -> go.Figure:
    """Horizontal bar: % change in diabetes deaths 2019→2020 by state."""
    sexo = d["sexo"][0]
    dmd  = df_dm.filter((pl.col("sexo") == sexo) & (pl.col("grupo_edad") == "Total"))
    pre  = dmd.filter((pl.col("año") == 2019) & (pl.col("entidad") != "Total")).rename({"muertes_dm": "v19"})
    post = dmd.filter((pl.col("año") == 2020) & (pl.col("entidad") != "Total")).rename({"muertes_dm": "v20"})
    comp = (
        pre.join(post, on="entidad")
        .with_columns(((pl.col("v20") - pl.col("v19")) / pl.col("v19") * 100).alias("pct"))
        .sort("pct")
    )
    colors = ["#E84855" if v > 0 else "#3BB273" for v in comp["pct"].to_list()]
    fig = go.Figure(go.Bar(
        x=comp["pct"].to_list(), y=comp["entidad"].to_list(),
        orientation="h", marker_color=colors,
        text=comp["pct"].to_list(), textposition="outside", texttemplate="%{x:+.0f}%",
        hovertemplate="<b>%{y}</b><br>Cambio: %{x:+.1f}%<extra></extra>",
    ))
    fig.add_vline(x=0, line_width=1, line_color="#475569")
    fig.update_layout(
        **CHART_LAYOUT, height=CHART_H_S,
        title=f"Exceso de muertes por diabetes 2019→2020 por estado ({sexo})",
        xaxis=dict(gridcolor=GRID, title="Cambio porcentual"),
        yaxis=dict(gridcolor=GRID_NONE),
        margin=dict(t=50, b=40, l=185, r=80),
    )
    return fig


# ── figure factories — anomaly analysis ──────────────────────────────────────
def _anomaly_base():
    nac   = df_def.filter((pl.col("entidad") == "Total") & (pl.col("sexo") == "Total")).sort("año")
    vals  = [float(v) for v in nac["valor"].to_list()]
    years = nac["año"].to_list()
    yoy   = [(vals[i] - vals[i-1]) / vals[i-1] * 100 for i in range(1, len(vals))]
    yrs   = years[1:]
    g_pre = [g for y, g in zip(yrs, yoy) if y <= 2019]
    mu    = sum(g_pre) / len(g_pre)
    sd    = (sum((g - mu) ** 2 for g in g_pre) / (len(g_pre) - 1)) ** 0.5
    z     = [(g - mu) / sd for g in yoy]
    base  = vals[0]
    exp   = [base * (1 + mu/100) ** (y - years[0]) for y in years]
    eu1   = [base * (1 + (mu + sd)   / 100) ** (y - years[0]) for y in years]
    el1   = [base * (1 + (mu - sd)   / 100) ** (y - years[0]) for y in years]
    eu2   = [base * (1 + (mu + 2*sd) / 100) ** (y - years[0]) for y in years]
    el2   = [base * (1 + (mu - 2*sd) / 100) ** (y - years[0]) for y in years]
    excess = {y: v - e for y, v, e in zip(years, vals, exp)}
    return nac, vals, years, yoy, yrs, mu, sd, z, exp, eu1, el1, eu2, el2, excess


def fig_anomaly_trend() -> go.Figure:
    nac, vals, years, _, _, mu, sd, _, exp, eu1, el1, eu2, el2, excess = _anomaly_base()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=years + years[::-1], y=eu2 + el2[::-1],
        fill="toself", fillcolor="rgba(46,134,171,0.10)",
        line=dict(width=0), name="±2σ corredor", hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=years + years[::-1], y=eu1 + el1[::-1],
        fill="toself", fillcolor="rgba(46,134,171,0.20)",
        line=dict(width=0), name="±1σ corredor", hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=years, y=exp, mode="lines", name="Tendencia esperada",
        line=dict(color="#2E86AB", dash="dash", width=1.5, shape="spline", smoothing=1.0),
        hovertemplate="%{x}: %{y:,.0f}<extra>Esperado</extra>",
    ))
    fig.add_trace(go.Scatter(
        x=years, y=vals, mode="lines+markers", name="Observado",
        line=dict(color="#CBD5E1", width=3.5, shape="spline", smoothing=0.8),
        marker=dict(size=6),
        hovertemplate="%{x}: %{y:,.0f}  (exceso %{customdata:+,.0f})<extra></extra>",
        customdata=[excess[y] for y in years],
    ))
    fig.add_vrect(x0=2019.5, x1=2021.5, fillcolor="rgba(232,72,85,0.08)", line_width=0)
    for yr, label, ax in [(2020, "+319k exceso", 55), (2021, "+334k exceso", 55)]:
        obs_v = float(nac.filter(pl.col("año") == yr)["valor"][0])
        fig.add_annotation(
            x=yr, y=obs_v, text=label, showarrow=True, arrowhead=2, arrowwidth=1,
            arrowcolor="#E84855", font=dict(size=10, color="#E84855"),
            ax=ax, ay=-35, bgcolor="#0F172A", borderpad=3,
        )
    fig.update_layout(
        **CHART_LAYOUT, height=400,
        title="Defunciones observadas vs. trayectoria esperada (baseline 2010–2019)",
        xaxis=dict(gridcolor=GRID, dtick=2),
        yaxis=dict(gridcolor=GRID, title="Defunciones"),
        legend=dict(orientation="h", y=1.12, x=0),
        margin=dict(t=60, b=50, l=80, r=20),
    )
    return fig


def fig_anomaly_bars() -> go.Figure:
    _, _, _, yoy, yrs, mu, sd, z, _, _, _, _, _, _ = _anomaly_base()
    bar_colors = [
        "#E84855" if abs(zi) >= 2 else ("#F4A261" if abs(zi) >= 1 else "#3BB273")
        for zi in z
    ]
    fig = go.Figure(go.Bar(
        x=yrs, y=yoy,
        marker_color=bar_colors, name="YoY %",
        text=[f"{g:+.1f}%" for g in yoy],
        textposition="outside", textfont=dict(size=8, color="#94A3B8"),
        hovertemplate="%{x}: %{y:+.2f}%<extra></extra>",
    ))
    fig.add_hrect(y0=mu - 2*sd, y1=mu + 2*sd, fillcolor="rgba(46,134,171,0.08)", line_width=0)
    fig.add_hrect(y0=mu - sd,   y1=mu + sd,   fillcolor="rgba(46,134,171,0.16)", line_width=0)
    for y_val, dash, col in [
        (mu, "dash", "#94A3B8"),
        (mu + sd,   "dot", "#2E86AB"), (mu - sd,   "dot", "#2E86AB"),
        (mu + 2*sd, "dot", "#475569"), (mu - 2*sd, "dot", "#475569"),
    ]:
        fig.add_hline(y=y_val, line_dash=dash, line_color=col, line_width=1.2, opacity=0.7)
    for yr, z_lbl, ay in [(2020, f"z=+{z[yrs.index(2020)]:.0f}σ", -18),
                           (2022, f"z={z[yrs.index(2022)]:.0f}σ",  18),
                           (2023, f"z={z[yrs.index(2023)]:.0f}σ",  18)]:
        fig.add_annotation(
            x=yr, y=yoy[yrs.index(yr)], text=z_lbl,
            showarrow=False, yshift=ay, font=dict(size=9, color="#E84855"),
        )
    fig.update_layout(
        **CHART_LAYOUT, height=360,
        title=f"Crecimiento anual (YoY %)  —  media pre-COVID: {mu:.1f}%  σ={sd:.1f}pp",
        xaxis=dict(gridcolor=GRID, dtick=2),
        yaxis=dict(gridcolor=GRID, title="YoY (%)"),
        showlegend=False,
        margin=dict(t=50, b=50, l=80, r=20),
    )
    return fig


# ── layout helpers ────────────────────────────────────────────────────────────
def _insight_card(stat: str, label: str, detail: str, color: str, md: int = 3) -> dbc.Col:
    return dbc.Col(html.Div([
        html.Div(stat,   style={"color": color, "fontSize": "1.7rem", "fontWeight": "700", "lineHeight": "1.1"}),
        html.Div(label,  style={"color": "#F8FAFC", "fontSize": "13px", "fontWeight": "600", "marginTop": "4px"}),
        html.Div(detail, style={"color": "#64748B", "fontSize": "12px", "marginTop": "3px"}),
    ], style={
        "background": "#1E293B", "border": "1px solid #334155",
        "borderLeft": f"4px solid {color}", "borderRadius": "8px", "padding": "14px 16px",
    }), md=md)


def _ev_insights() -> dbc.Row:
    return dbc.Row([
        _insight_card(
            f"{EV_COVID_DROP:+.1f} años", "Caída COVID nacional",
            "Esperanza de vida 2019→2021 (Total)", "#E84855",
        ),
        _insight_card(
            f"{GENDER_GAP:.1f} años", "Brecha de género 2026",
            "Mujeres viven más que hombres", COLOR_MUJERES,
        ),
        _insight_card(
            f"{STATE_GAP:.1f} años", "Brecha entre estados",
            f"{EV_TOP['entidad']} ({EV_TOP['valor']:.1f}) vs {EV_BOT['entidad']} ({EV_BOT['valor']:.1f})",
            COLOR_HOMBRES,
        ),
        _insight_card(
            f"{WORST_EV_DELTA:+.1f} años", f"{WORST_EV_STATE}: peor impacto",
            "Mayor caída de EV por COVID (Total)", "#E84855",
        ),
    ], class_name="g-3", style={"marginBottom": "24px"})


def _def_insights() -> dbc.Row:
    return dbc.Row([
        _insight_card(
            f"+{DEF_PCT:.0f}%", "Exceso de muertes en 2021",
            f"{DEF_2019/1e6:.3f}M (2019) → {DEF_2021/1e6:.3f}M (2021)", "#E84855", md=4,
        ),
        _insight_card(
            f"+{WORST_DEF_PCT:.0f}%", f"{WORST_DEF_STATE}: mayor incremento",
            "Estado más golpeado en defunciones 2021", "#E84855", md=4,
        ),
        _insight_card(
            f"{DEF_2024/1e3:.0f}k", "Defunciones en 2024",
            f"Bajó un {(DEF_2021-DEF_2024)/DEF_2021*100:.0f}% desde el pico de 2021",
            COLOR_TOTAL, md=4,
        ),
    ], class_name="g-3", style={"marginBottom": "24px"})


def _dm_insights() -> html.Div:
    return html.Div([
        dbc.Row([
            _insight_card(
                f"+{DM_COVID_PCT:.0f}%", "Spike diabetes en 2020",
                f"{DM_2019/1e3:.0f}k (2019) → {DM_2020/1e3:.0f}k (2020)",
                "#E84855", md=4,
            ),
            _insight_card(
                f"{DM_PCT_ALL:.1f}%", "De cada 100 muertes, ~14 son diabetes",
                f"2024: {DM_2024:,} muertes por DM de {DEF_2024/1e3:.0f}k totales",
                COLOR_MUJERES, md=4,
            ),
            _insight_card(
                f"{DM_WORST_PCT:.1f}%", f"{DM_WORST_STATE}: peor estado",
                "Mayor proporción de muertes por diabetes en 2024",
                "#E84855", md=4,
            ),
        ], class_name="g-3", style={"marginBottom": "12px"}),
        dbc.Row([
            _insight_card(
                f"+{DM_85_GROWTH:.0f}%", "Grupo 85+ años, 2010→2024",
                f"{DM_85_2010:,} → {DM_85_2024:,} muertes — efecto envejecimiento",
                COLOR_MUJERES, md=6,
            ),
            _insight_card(
                f"{DM_RATIO_2019:.2f} → {DM_RATIO_2020:.2f}",
                "COVID invirtió la brecha de género",
                f"2019: más mujeres morían (H/M={DM_RATIO_2019:.2f})  →  2020: más hombres (H/M={DM_RATIO_2020:.2f})",
                COLOR_HOMBRES, md=6,
            ),
        ], class_name="g-3", style={"marginBottom": "24px"}),
    ])


def _kpi_cards() -> dbc.Row:
    cards = []
    for sexo, color in COLORS.items():
        val  = KPI_2026[sexo]
        chg  = KPI_2026[sexo] - KPI_2010[sexo]
        sign = "+" if chg >= 0 else ""
        cards.append(dbc.Col(html.Div([
            html.P(f"{sexo} — 2026",
                   style={"color": "#94A3B8", "fontSize": "13px", "margin": "0 0 6px"}),
            html.H2(f"{val:.1f}",
                    style={"color": "#F8FAFC", "margin": "0", "fontSize": "2.2rem"}),
            html.P("años", style={"color": "#64748B", "fontSize": "12px", "margin": "0 0 4px"}),
            html.P(f"{sign}{chg:.1f} vs 2010",
                   style={"color": color, "fontSize": "13px", "fontWeight": "600", "margin": "0"}),
        ], style=CARD_STYLE), md=4))
    return dbc.Row(cards, style={"marginBottom": "24px"})


def _panel(child) -> html.Div:
    return html.Div(child, style={
        "background": "#1E293B", "borderRadius": "8px",
        "padding": "16px", "marginBottom": "24px",
    })


# ── app ───────────────────────────────────────────────────────────────────────
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Mortalidad — México"

def _sexo_filter(filter_id: str) -> html.Div:
    return html.Div([
        html.Label("Sexo:", style={"color": "#94A3B8", "marginRight": "12px", "fontWeight": "600"}),
        dcc.RadioItems(
            id=filter_id, value="Total", inline=True,
            options=[{"label": s, "value": s} for s in ["Total", "Hombres", "Mujeres"]],
            inputStyle={"marginRight": "6px"},
            labelStyle={"marginRight": "24px", "color": "#CBD5E1", "cursor": "pointer"},
        ),
    ], style={"marginBottom": "20px", "display": "flex", "alignItems": "center"})

app.layout = html.Div(
    style={"backgroundColor": "#0F172A", "minHeight": "100vh",
           "padding": "24px", "fontFamily": "Inter, sans-serif"},
    children=[
        html.H1("Mortalidad — México",
                style={"color": "#F8FAFC", "marginBottom": "4px", "fontSize": "1.8rem"}),
        html.P("Esperanza de vida · defunciones · diabetes mellitus · 2010–2026 · Fuente: CONAPO / INEGI",
               style={"color": "#64748B", "marginBottom": "24px"}),

        _kpi_cards(),

        dcc.Tabs(id="tabs", value="ev", style={"marginBottom": "16px"}, children=[
            dcc.Tab(label="Esperanza de Vida", value="ev",
                    style=TAB_STYLE, selected_style=TAB_SEL),
            dcc.Tab(label="Defunciones Registradas", value="def",
                    style=TAB_STYLE, selected_style=TAB_SEL),
            dcc.Tab(label="Diabetes Mellitus", value="dm",
                    style=TAB_STYLE, selected_style=TAB_SEL),
        ]),

        html.Div(id="tab-ev", children=[
            _sexo_filter("sexo-filter-ev"),
            _ev_insights(),
            _panel(dcc.Graph(id="ev-trend",   config=GRAPH_CONFIG)),
            dbc.Row([
                dbc.Col(_panel(dcc.Graph(id="ev-ranking", config=GRAPH_CONFIG)), md=5),
                dbc.Col(_panel(dcc.Graph(id="ev-map",     config=GRAPH_CONFIG)), md=7),
            ]),
            _panel(dcc.Graph(id="ev-covid",    config=GRAPH_CONFIG)),
            _panel(dcc.Graph(id="ev-progress", config=GRAPH_CONFIG)),
            _panel(dcc.Graph(id="ev-scatter",  config=GRAPH_CONFIG)),
        ]),

        html.Div(id="tab-def", children=[
            _sexo_filter("sexo-filter-def"),
            _def_insights(),
            _panel(dcc.Graph(id="def-trend",        config=GRAPH_CONFIG)),
            _panel(dcc.Graph(id="def-covid",        config=GRAPH_CONFIG)),
            _panel(dcc.Graph(id="def-anomaly-trend", config=GRAPH_CONFIG)),
            _panel(dcc.Graph(id="def-anomaly-bars",  config=GRAPH_CONFIG)),
        ]),

        html.Div(id="tab-dm", children=[
            _sexo_filter("sexo-filter-dm"),
            _dm_insights(),
            _panel(dcc.Graph(id="dm-trend",       config=GRAPH_CONFIG)),
            _panel(dcc.Graph(id="dm-heatmap",     config=GRAPH_CONFIG)),
            dbc.Row([
                dbc.Col(_panel(dcc.Graph(id="dm-pct",         config=GRAPH_CONFIG)), md=6),
                dbc.Col(_panel(dcc.Graph(id="dm-covid-state", config=GRAPH_CONFIG)), md=6),
            ]),
        ]),
    ],
)


# ── callbacks ─────────────────────────────────────────────────────────────────
@app.callback(
    Output("tab-ev",  "style"),
    Output("tab-def", "style"),
    Output("tab-dm",  "style"),
    Input("tabs", "value"),
)
def switch_tab(tab: str):
    show = {"display": "block"}
    hide = {"display": "none"}
    return (
        show if tab == "ev"  else hide,
        show if tab == "def" else hide,
        show if tab == "dm"  else hide,
    )


@app.callback(
    Output("ev-trend",    "figure"),
    Output("ev-ranking",  "figure"),
    Output("ev-map",      "figure"),
    Output("ev-covid",    "figure"),
    Output("ev-progress", "figure"),
    Output("ev-scatter",  "figure"),
    Input("sexo-filter-ev", "value"),
)
def update_ev(sexo: str):
    d = df_ev.filter(pl.col("sexo") == sexo)
    return fig_ev_trend(d), fig_ev_ranking(d), fig_ev_map(d), fig_ev_covid(d), fig_ev_progress(d), fig_ev_scatter(d)


@app.callback(
    Output("def-trend",         "figure"),
    Output("def-covid",         "figure"),
    Output("def-anomaly-trend", "figure"),
    Output("def-anomaly-bars",  "figure"),
    Input("sexo-filter-def", "value"),
)
def update_def(sexo: str):
    d = df_def.filter(pl.col("sexo") == sexo)
    return fig_def_trend(d), fig_def_covid(d), fig_anomaly_trend(), fig_anomaly_bars()


@app.callback(
    Output("dm-trend",       "figure"),
    Output("dm-heatmap",     "figure"),
    Output("dm-pct",         "figure"),
    Output("dm-covid-state", "figure"),
    Input("sexo-filter-dm", "value"),
)
def update_dm(sexo: str):
    d = df_dm.filter(pl.col("sexo") == sexo)
    return fig_dm_trend(d), fig_dm_heatmap(d), fig_dm_pct(d), fig_dm_covid_state(d)


if __name__ == "__main__":
    app.run(debug=True, port=8060)
