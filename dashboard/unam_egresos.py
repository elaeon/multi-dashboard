"""Dashboard: UNAM — Egreso y Titulación

Source: data/unam_egresos_y_titulacion/{egreso, egreso_carreras,
examenes_de_grado, examenes_profesionales, titulacion_carreras}.csv

`egreso.csv` se usa solo para la vista por nivel (Bachillerato / Licenciatura
/ Técnico). Para los agregados de licenciatura (eficiencia terminal, pipeline,
contribución por método) se usa `egreso_carreras.csv` — escolarizada por
carrera —, consistente con `titulacion_carreras.csv` y la pestaña de flujo.
La diferencia con la cifra de licenciatura en `egreso.csv` (~5–10%) se debe a
que el agregado incluye SUAYED.

Títulos de licenciatura provienen del desglose por carrera/opción
(`titulacion_carreras.csv`). El agregado «titulos_expedidos.csv» queda
deprecated: tiene 2025 desactualizado (réplica de 2024) e incluye SUAYED,
mientras que el desglose por carrera es escolarizada únicamente y se
extiende hasta 2025 con datos correctos.
"""

import numpy as np
import polars as pl
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc

# ── Estilos ─────────────────────────────────────────────────────────────────

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

NIVEL_COLORS = {
    "Licenciatura":         "#2E86AB",
    "Bachillerato":         "#3BB273",
    "Técnico":              "#F4A261",
}
GRADO_COLORS = {
    "Especialización": "#2E86AB",
    "Maestría":        "#3BB273",
    "Doctorado":       "#F4A261",
}

DATA = "data/unam_egresos_y_titulacion"

# Variantes con la misma carrera en distinta capitalización; se homologan
# al cargar para que aparezcan una sola vez en cualquier agregado.
_CARRERA_ALIAS = {
    "Ciencias ambientales":                  "Ciencias Ambientales",
    "Desarrollo y gestión interculturales":  "Desarrollo y Gestión Interculturales",
    "Manejo sustentable de zonas costeras":  "Manejo Sustentable de Zonas Costeras",
}
# Renombres ortográficos (~2020-2021): la cohorte vieja y la nueva son la misma
# carrera. Se mapea el nombre viejo al actual (el que llega hasta 2025) para no
# reportar una carrera "muerta" en 2020 + una "nueva" en 2021. Sin esto, las
# gráficas de demanda muestran caídas/altas espurias de −100%/+∞.
_CARRERA_RENAME = {
    "Química Farmacéutica Biológica":     "Química Farmacéutico Biológica",
    "Ingeniería Eléctrica y Electrónica": "Ingeniería Eléctrica Electrónica",
}
_CARRERA_FIX = {**_CARRERA_ALIAS, **_CARRERA_RENAME}

# ── Carga y limpieza ────────────────────────────────────────────────────────

def _norm(col: str) -> pl.Expr:
    return (
        pl.col(col).str.strip_chars()
        .str.replace_all("Técnico Profesional", "Técnico")
        .str.replace_all("Diplomas de Especialización", "Especialización")
        .str.replace_all("Examen General de conocimientos", "Examen general de conocimientos")
        .str.replace_all("Estudios en posgrado", "Estudios de posgrado")
        .str.replace_all("Otras opciones", "Otra")
        .str.replace_all("^Otras$", "Otra")
    )

# 1) Egreso por nivel  (valor es string por el "-" de 2024)
_egreso_raw = (
    pl.read_csv(f"{DATA}/egreso.csv", schema_overrides={"valor": pl.Utf8})
    .with_columns(
        _norm("categoria").alias("categoria"),
        pl.col("valor").str.replace("-", "0").cast(pl.Int64, strict=False).alias("valor"),
    )
)
df_egreso = _egreso_raw.filter(pl.col("categoria") != "T O T A L")
df_egreso_total = (
    _egreso_raw.filter(pl.col("categoria") == "T O T A L")
    .select(["año", pl.col("valor").alias("total")])
)

# 2) Exámenes de grado — quedarse con los 3 grados consistentes
GRADO_KEEP = ["Especialización", "Maestría", "Doctorado"]
df_grado = (
    pl.read_csv(f"{DATA}/examenes_de_grado.csv")
    .with_columns(_norm("categoria").alias("categoria"))
    .filter(pl.col("categoria").is_in(GRADO_KEEP))
)

# 3) Títulos expedidos (licenciatura) — desde el desglose por carrera
# Esta tabla alimenta tanto el KPI de títulos como las gráficas de métodos
# (trayectorias / share / trajectory), que antes usaban examenes_profesionales
# pero ese archivo solo trae el desglose por método a partir de 2014.
_titulos_raw = (
    pl.read_csv(f"{DATA}/titulacion_carreras.csv")
    .with_columns(pl.col("carrera").replace(_CARRERA_FIX))
)
df_titulos = (
    _titulos_raw
    .group_by(["año", "opcion_titulacion"]).agg(pl.col("total").sum().alias("valor"))
    .rename({"opcion_titulacion": "categoria"})
)
df_lic_titulos = (
    _titulos_raw.group_by("año").agg(pl.col("total").sum().alias("titulos_expedidos"))
)

# Total licenciatura por año (para eficiencia terminal)
def _total_licenciatura(name: str) -> pl.DataFrame:
    return (
        pl.read_csv(f"{DATA}/{name}.csv")
        .with_columns(_norm("categoria").alias("categoria"))
        .filter((pl.col("categoria") == "Licenciatura") & (pl.col("orden") == 1))
        .select(["año", pl.col("valor").alias(name)])
    )

df_lic_examenes = _total_licenciatura("examenes_profesionales")

# Egreso licenciatura desde el desglose por carrera (escolarizada). Las filas
# con carrera tipo "b Se refiere…" o "c El criterio…" son notas al pie volcadas
# por el extractor; se descartan antes de cualquier agregación.
_FOOTNOTE = r"^[a-z] "
_eg_carr_raw = (
    pl.read_csv(f"{DATA}/egreso_carreras.csv")
    .filter(~pl.col("carrera").str.contains(_FOOTNOTE))
    .with_columns(pl.col("carrera").replace(_CARRERA_FIX))
)
df_lic_egreso = (
    _eg_carr_raw.group_by("año").agg(pl.col("total").sum().alias("egreso_lic"))
)

# Población escolar — primer ingreso por nivel (renglón padre)
df_pob = pl.read_csv("data/unam_poblacion_escolar.csv")
df_ingreso_nivel = (
    df_pob.filter(
        pl.col("nivel_padre").is_null()
        & pl.col("categoria").is_in(["Posgrado", "Licenciatura", "Bachillerato"])
    )
    .select(["año", "categoria", "primer_ingreso"])
)
df_lic_ingreso = (
    df_ingreso_nivel.filter(pl.col("categoria") == "Licenciatura")
    .select(["año", pl.col("primer_ingreso").alias("ingreso_lic")])
)

LAG_LIC = 5  # años típicos para completar licenciatura UNAM

YEARS = sorted(df_egreso["año"].unique().to_list())
Y_MIN, Y_MAX = YEARS[0], YEARS[-1]

METHODS_SORTED = (
    df_titulos.group_by("categoria").agg(pl.col("valor").sum())
    .sort("valor", descending=True)["categoria"].to_list()
)

CARRERAS = sorted(_titulos_raw["carrera"].unique().to_list())

_METHOD_PALETTE = px.colors.qualitative.Plotly + px.colors.qualitative.Set2
METHOD_COLORS = {m: _METHOD_PALETTE[i % len(_METHOD_PALETTE)]
                 for i, m in enumerate(METHODS_SORTED)}

# ── Flujo por carrera ───────────────────────────────────────────────────────
_pob_lic_raw = (
    pl.read_csv(f"{DATA}/poblacion_escolar_lic.csv")
    .filter(~pl.col("carrera").str.contains(_FOOTNOTE))
    .with_columns(pl.col("carrera").replace(_CARRERA_FIX))
)

# Algunas carreras/entidades en pob traen sufijo de pie de página ("Arquitecturab",
# "Cinematografíab"). Se mapean al nombre limpio si existe en eg ∪ tit.
_CLEAN_C = set(_eg_carr_raw["carrera"].unique().to_list()) | set(_titulos_raw["carrera"].unique().to_list())
_CLEAN_E = set(_eg_carr_raw["entidad"].unique().to_list()) | set(_titulos_raw["entidad"].unique().to_list())

def _fix(name: str, clean_set: set) -> str:
    if name in clean_set:
        return name
    for n in (1, 2):
        if len(name) > n and name[-n:].isalpha() and name[-n:].islower() and name[:-n] in clean_set:
            return name[:-n]
    return name

_pob_lic = _pob_lic_raw.with_columns([
    pl.col("carrera").replace({c: _fix(c, _CLEAN_C) for c in _pob_lic_raw["carrera"].unique().to_list()}),
    pl.col("entidad").replace({e: _fix(e, _CLEAN_E) for e in _pob_lic_raw["entidad"].unique().to_list()}),
])

# Agregados por (carrera, año) — se suman todas las entidades que ofrecen la carrera
df_carr_pob = (
    _pob_lic.group_by(["carrera", "año"])
    .agg([
        (pl.col("pi_h") + pl.col("pi_m")).sum().alias("pi"),
        (pl.col("rei_h") + pl.col("rei_m")).sum().alias("rei"),
    ])
)
df_carr_eg = (
    _eg_carr_raw.group_by(["carrera", "año"])
    .agg(pl.col("total").sum().alias("egreso"))
)
df_carr_tit = (
    _titulos_raw.group_by(["carrera", "año"])
    .agg(pl.col("total").sum().alias("titulos"))
)

# Carreras con datos en las 3 fuentes — únicas válidas para el flujo
CARRERAS_FLUJO = sorted(
    set(df_carr_pob["carrera"].to_list())
    & set(df_carr_eg["carrera"].to_list())
    & set(df_carr_tit["carrera"].to_list())
)
# El último año disponible para "continúan al siguiente año" es Y_MAX − 1
FLUJO_YEARS = list(range(Y_MIN, Y_MAX))

# ── Abandono por carrera y sexo ─────────────────────────────────────────────
# Misma lógica del Sankey, aplicada por sexo: por (carrera, año) calcular
# abandono_S = inscritos_S − egreso_S − continúan_S
# donde continúan_S = min(rei_S del siguiente año, inscritos_S − egreso_S).

_pob_carr_sex = (
    _pob_lic.group_by(["carrera", "año"])
    .agg([
        pl.col("pi_h").sum(),  pl.col("pi_m").sum(),
        pl.col("rei_h").sum(), pl.col("rei_m").sum(),
    ])
    .with_columns([
        (pl.col("pi_h") + pl.col("rei_h")).alias("insc_h"),
        (pl.col("pi_m") + pl.col("rei_m")).alias("insc_m"),
    ])
)
_eg_carr_sex = (
    _eg_carr_raw.group_by(["carrera", "año"])
    .agg([
        pl.col("hombres").sum().alias("eg_h"),
        pl.col("mujeres").sum().alias("eg_m"),
    ])
)
# rei del año siguiente: shift hacia atrás (año - 1) para joinear con el año base
_rei_next = (
    _pob_carr_sex.select(["carrera", "año", "rei_h", "rei_m"])
    .with_columns(pl.col("año") - 1)
    .rename({"rei_h": "rei_h_next", "rei_m": "rei_m_next"})
)
df_drop_carr_year = (
    _pob_carr_sex
    .join(_eg_carr_sex, on=["carrera", "año"], how="inner")
    .join(_rei_next, on=["carrera", "año"], how="left")
    .with_columns([
        pl.col("rei_h_next").fill_null(0),
        pl.col("rei_m_next").fill_null(0),
        (pl.col("insc_h") - pl.col("eg_h")).clip(0, None).alias("no_eg_h"),
        (pl.col("insc_m") - pl.col("eg_m")).clip(0, None).alias("no_eg_m"),
    ])
    .with_columns([
        pl.min_horizontal("rei_h_next", "no_eg_h").alias("cont_h"),
        pl.min_horizontal("rei_m_next", "no_eg_m").alias("cont_m"),
    ])
    .with_columns([
        (pl.col("no_eg_h") - pl.col("cont_h")).alias("abandono_h"),
        (pl.col("no_eg_m") - pl.col("cont_m")).alias("abandono_m"),
    ])
    .filter(pl.col("año").is_in(FLUJO_YEARS))
    .select(["carrera", "año", "insc_h", "insc_m", "abandono_h", "abandono_m"])
)

# ── KPIs ─────────────────────────────────────────────────────────────────────

def compute_kpis(yr0: int, yr1: int) -> dict:
    y = yr1  # KPI = año más reciente del rango
    total = df_egreso_total.filter(pl.col("año") == y)["total"].sum()
    doc = df_grado.filter((pl.col("año") == y) & (pl.col("categoria") == "Doctorado"))["valor"].sum()
    tit = df_titulos.filter(pl.col("año") == y)["valor"].sum()
    eg_lic = df_lic_egreso.filter(pl.col("año") == y)["egreso_lic"].sum()
    tit_lic = df_lic_titulos.filter(pl.col("año") == y)["titulos_expedidos"].sum()
    tasa = (tit_lic / eg_lic * 100) if eg_lic else 0.0
    return {"año": y, "total": total, "doctorado": doc, "titulos": tit, "tasa": tasa}

# ── Figuras ──────────────────────────────────────────────────────────────────

def fig_egreso(d: pl.DataFrame) -> go.Figure:
    """Bachillerato y licenciatura son las dos fuentes principales de egreso."""
    fig = go.Figure()
    pivot = (
        d.group_by(["año", "categoria"]).agg(pl.col("valor").sum())
        .pivot(values="valor", index="año", on="categoria")
        .sort("año")
        .fill_null(0)
    )
    for nivel in ["Bachillerato", "Licenciatura", "Técnico"]:
        if nivel not in pivot.columns:
            continue
        fig.add_trace(go.Bar(
            x=pivot["año"].to_list(),
            y=pivot[nivel].to_list(),
            name=nivel,
            marker_color=NIVEL_COLORS[nivel],
            hovertemplate=f"<b>{nivel}</b><br>%{{x}}: %{{y:,}}<extra></extra>",
        ))
    fig.update_layout(
        title="Egreso anual por nivel",
        barmode="stack", height=420,
        legend=dict(orientation="h", y=-0.18, x=0),
        **CHART_LAYOUT,
    )
    return fig

def fig_grado(d: pl.DataFrame) -> go.Figure:
    """Especialización lidera el posgrado; doctorado se mantiene <1000/año."""
    fig = go.Figure()
    for grado in GRADO_KEEP:
        sub = d.filter(pl.col("categoria") == grado).sort("año")
        if sub.is_empty():
            continue
        fig.add_trace(go.Scatter(
            x=sub["año"].to_list(),
            y=sub["valor"].to_list(),
            mode="lines+markers",
            name=grado,
            line=dict(color=GRADO_COLORS[grado], width=2.5),
            marker=dict(size=7),
            hovertemplate=f"<b>{grado}</b><br>%{{x}}: %{{y:,}}<extra></extra>",
        ))
    fig.update_layout(
        title="Exámenes de grado por nivel",
        height=420,
        legend=dict(orientation="h", y=-0.18, x=0),
        **CHART_LAYOUT,
    )
    return fig

def fig_ingreso(d_pob: pl.DataFrame) -> go.Figure:
    """Primer ingreso anual por nivel (Posgrado, Licenciatura, Bachillerato)."""
    colors = {"Licenciatura": "#2E86AB", "Bachillerato": "#3BB273", "Posgrado": "#F4A261"}
    fig = go.Figure()
    for nivel in ["Licenciatura", "Bachillerato", "Posgrado"]:
        sub = d_pob.filter(pl.col("categoria") == nivel).sort("año")
        fig.add_trace(go.Scatter(
            x=sub["año"].to_list(),
            y=sub["primer_ingreso"].to_list(),
            mode="lines+markers", name=nivel,
            line=dict(color=colors[nivel], width=2.5),
            marker=dict(size=7),
            hovertemplate=f"<b>{nivel}</b><br>%{{x}}: %{{y:,}}<extra></extra>",
        ))
    fig.update_layout(
        title="Primer ingreso anual por nivel",
        height=420,
        legend=dict(orientation="h", y=-0.18, x=0),
        **CHART_LAYOUT,
    )
    return fig

def _pipeline_ratios(yr0: int, yr1: int) -> pl.DataFrame:
    """Tabla año → ratios del pipeline ingreso→egreso→título de licenciatura."""
    lag = df_lic_ingreso.with_columns((pl.col("año") + LAG_LIC).alias("año")) \
                        .rename({"ingreso_lic": "ingreso_lag"})
    base = (
        df_lic_egreso
        .join(df_lic_titulos, on="año", how="full", coalesce=True)
        .join(lag, on="año", how="left")
        .with_columns([
            (pl.col("titulos_expedidos") / pl.col("egreso_lic") * 100).alias("conv_corta"),
            (pl.col("titulos_expedidos") / pl.col("ingreso_lag") * 100).alias("conv_cohorte"),
            (pl.col("egreso_lic") / pl.col("ingreso_lag") * 100).alias("egreso_cohorte"),
        ])
        .filter(pl.col("año").is_between(yr0, yr1))
        .sort("año")
    )
    return base

def fig_pipeline(yr0: int, yr1: int) -> go.Figure:
    """Eficiencia del pipeline ingreso → egreso → título (licenciatura)."""
    base = _pipeline_ratios(yr0, yr1)
    xs = base["año"].to_list()
    fig = go.Figure()
    series = [
        ("conv_corta",     "Títulos / Egreso (mismo año)",                "#2E86AB"),
        ("conv_cohorte",   f"Títulos / Ingreso (lag {LAG_LIC}y, cohorte)", "#3BB273"),
        ("egreso_cohorte", f"Egreso / Ingreso (lag {LAG_LIC}y, cohorte)",  "#F4A261"),
    ]
    for col, name, color in series:
        ys = base[col].to_list()
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines+markers", name=name,
            line=dict(color=color, width=2.5), marker=dict(size=7),
            connectgaps=False,
            hovertemplate=f"<b>{name}</b><br>%{{x}}: %{{y:.1f}}%<extra></extra>",
        ))
    fig.update_layout(
        title=f"Eficiencia del pipeline licenciatura  ·  cohorte = lag {LAG_LIC} años",
        height=420,
        yaxis=dict(title="% de conversión", gridcolor="#334155", ticksuffix="%"),
        xaxis=dict(gridcolor="#334155"),
        legend=dict(orientation="h", y=-0.22, x=0),
        margin=dict(t=50, b=70, l=10, r=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#CBD5E1",
    )
    return fig

def fig_trayectorias(d: pl.DataFrame) -> go.Figure:
    """Small multiples: trayectoria anual de cada método de titulación."""
    agg = d.group_by(["categoria", "año"]).agg(pl.col("valor").sum())
    totals = agg.group_by("categoria").agg(pl.col("valor").sum()).sort("valor", descending=True)
    methods = totals["categoria"].to_list()
    if not methods:
        return go.Figure().update_layout(
            title="Sin datos para la selección actual", **CHART_LAYOUT,
        )

    cols = 4
    rows = (len(methods) + cols - 1) // cols
    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=methods,
        horizontal_spacing=0.06, vertical_spacing=0.12,
    )

    for i, m in enumerate(methods):
        r, c = i // cols + 1, i % cols + 1
        sub = agg.filter(pl.col("categoria") == m).sort("año")
        xs = sub["año"].to_list()
        ys = sub["valor"].to_list()
        color = "#3BB273" if len(ys) >= 2 and ys[-1] >= ys[0] else "#E84855"
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines+markers",
            line=dict(color=color, width=2),
            marker=dict(color=color, size=5),
            showlegend=False,
            hovertemplate=f"<b>{m}</b><br>%{{x}}: %{{y:,}}<extra></extra>",
        ), row=r, col=c)

    fig.update_xaxes(gridcolor="#334155", showticklabels=True, dtick=2)
    fig.update_yaxes(gridcolor="#334155", rangemode="tozero", tickformat=",")
    for ann in fig.layout.annotations:
        ann.font = dict(size=11, color="#CBD5E1")

    fig.update_layout(
        title="Trayectoria anual por método de titulación (licenciatura)",
        height=rows * 190 + 90,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#CBD5E1",
        margin=dict(t=80, b=30, l=10, r=10),
    )
    return fig

def fig_share(d: pl.DataFrame) -> go.Figure:
    """100% stacked area: composición porcentual por método y año."""
    agg = d.group_by(["categoria", "año"]).agg(pl.col("valor").sum())
    totals = agg.group_by("categoria").agg(pl.col("valor").sum()).sort("valor", descending=True)
    methods = totals["categoria"].to_list()
    years = sorted(agg["año"].unique().to_list())

    palette = (px.colors.qualitative.Plotly + px.colors.qualitative.Set2)
    color_map = {m: palette[i % len(palette)] for i, m in enumerate(methods)}

    fig = go.Figure()
    for m in methods:
        sub = agg.filter(pl.col("categoria") == m).sort("año")
        d_dict = dict(zip(sub["año"].to_list(), sub["valor"].to_list()))
        ys = [d_dict.get(y, 0) for y in years]
        fig.add_trace(go.Scatter(
            x=years, y=ys, mode="lines",
            name=m,
            stackgroup="one", groupnorm="percent",
            line=dict(width=0.5, color=color_map[m]),
            fillcolor=color_map[m],
            hovertemplate=f"<b>{m}</b><br>%{{x}}: %{{y:.1f}}%<extra></extra>",
        ))
    fig.update_layout(
        title="Composición porcentual de métodos de titulación (licenciatura)",
        height=520,
        yaxis=dict(title="% del total", range=[0, 100], gridcolor="#334155", ticksuffix="%"),
        xaxis=dict(gridcolor="#334155", dtick=1),
        legend=dict(orientation="h", y=-0.22, x=0, font=dict(size=10)),
        margin=dict(t=50, b=110, l=10, r=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#CBD5E1",
    )
    return fig

def fig_top_method_per_entidad(yr0: int, yr1: int) -> go.Figure:
    """Para cada entidad, el método de titulación más popular (% del total)."""
    base = (
        _titulos_raw.filter(pl.col("año").is_between(yr0, yr1))
        .group_by(["entidad", "opcion_titulacion"]).agg(pl.col("total").sum())
    )
    ent_total = base.group_by("entidad").agg(pl.col("total").sum().alias("ent_total"))
    base = (
        base.join(ent_total, on="entidad")
        .with_columns((pl.col("total") / pl.col("ent_total") * 100).alias("pct"))
    )
    top = (
        base.sort(["entidad", "pct"], descending=[False, True])
        .group_by("entidad", maintain_order=True).head(1)
        # ordering: agrupar entidades por método dominante, descendiendo dentro de cada grupo
        .sort(["opcion_titulacion", "pct"], descending=[False, True])
    )
    if top.is_empty():
        return go.Figure().update_layout(
            title="Sin datos para el rango seleccionado", **CHART_LAYOUT,
        )

    fig = go.Figure()
    # una traza por método (para que aparezca en la leyenda con su color)
    for method in top["opcion_titulacion"].unique(maintain_order=True).to_list():
        sub = top.filter(pl.col("opcion_titulacion") == method)
        fig.add_trace(go.Bar(
            x=sub["pct"].to_list(),
            y=sub["entidad"].to_list(),
            orientation="h",
            name=method,
            marker_color=METHOD_COLORS.get(method, "#94A3B8"),
            customdata=sub["total"].to_list(),
            text=[f"{p:.0f}%" for p in sub["pct"].to_list()],
            textposition="outside",
            textfont=dict(color="#CBD5E1", size=10),
            hovertemplate=(
                "<b>%{y}</b><br>"
                f"Método: {method}<br>"
                "% del total: %{x:.1f}%<br>"
                "Títulos: %{customdata:,}<extra></extra>"
            ),
            cliponaxis=False,
        ))
    n = top.height
    fig.update_layout(
        title=f"Método de titulación más popular por entidad · {yr0}–{yr1}",
        height=max(440, n * 20 + 140),
        xaxis=dict(title="% de los títulos de la entidad",
                   range=[0, 100], ticksuffix="%", gridcolor="#334155"),
        yaxis=dict(categoryorder="array",
                   categoryarray=top["entidad"].to_list(),
                   gridcolor="#334155", automargin=True),
        legend=dict(orientation="h", y=-0.05, x=0, font=dict(size=10)),
        margin=dict(t=60, b=80, l=10, r=60),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#CBD5E1",
    )
    return fig

def _ratio_per_method(year: int) -> pl.DataFrame:
    """Return per-method title/egreso ratio (%) for a single año."""
    eg = df_lic_egreso.filter(pl.col("año") == year)["egreso_lic"].sum()
    if not eg:
        return pl.DataFrame({"categoria": [], "rate": []})
    return (
        df_titulos.filter(pl.col("año") == year)
        .with_columns((pl.col("valor") / eg * 100).alias("rate"))
        .select(["categoria", "rate"])
    )

def fig_contribution(yr0: int, yr1: int) -> go.Figure:
    """Bar chart: cada método y cuánto sumó (o restó) al ratio título/egreso entre yr0 y yr1.

    Δ ratio_metodo = títulos_metodo(yr1)/egreso(yr1) − títulos_metodo(yr0)/egreso(yr0)
    La suma de Δ por método = Δ del ratio total título/egreso.
    """
    r0 = _ratio_per_method(yr0).rename({"rate": "r0"})
    r1 = _ratio_per_method(yr1).rename({"rate": "r1"})
    d = (
        r0.join(r1, on="categoria", how="full", coalesce=True)
        .with_columns([
            pl.col("r0").fill_null(0),
            pl.col("r1").fill_null(0),
            (pl.col("r1").fill_null(0) - pl.col("r0").fill_null(0)).alias("delta"),
        ])
        .sort("delta", descending=False)  # ascending → biggest on top after y-axis reversal
    )
    if d.is_empty() or (d["r0"].sum() == 0 and d["r1"].sum() == 0):
        return go.Figure().update_layout(
            title=f"Sin datos para {yr0} → {yr1}", **CHART_LAYOUT,
        )
    methods = d["categoria"].to_list()
    deltas = d["delta"].to_list()
    r0s = d["r0"].to_list()
    r1s = d["r1"].to_list()
    colors = ["#3BB273" if v >= 0 else "#E84855" for v in deltas]
    total_delta = sum(deltas)
    total_r0 = sum(r0s)
    total_r1 = sum(r1s)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=deltas, y=methods, orientation="h",
        marker_color=colors,
        text=[f"{v:+.1f} pp" for v in deltas],
        textposition="outside",
        textfont=dict(color="#CBD5E1"),
        customdata=list(zip(r0s, r1s)),
        hovertemplate=(
            "<b>%{y}</b><br>"
            f"{yr0}: %{{customdata[0]:.1f}}%<br>"
            f"{yr1}: %{{customdata[1]:.1f}}%<br>"
            "Δ: %{x:+.2f} pp<extra></extra>"
        ),
        cliponaxis=False,
    ))
    fig.update_layout(
        title=(
            f"Contribución por método al cambio en tasa título/egreso · {yr0} → {yr1}"
            f"<br><span style='font-size:0.85em;color:#94A3B8'>"
            f"Tasa total: {total_r0:.1f}% → {total_r1:.1f}%  "
            f"(Δ {total_delta:+.1f} pp)</span>"
        ),
        height=440,
        xaxis=dict(
            title="Δ títulos / egreso licenciatura (puntos porcentuales)",
            gridcolor="#334155", ticksuffix=" pp", zerolinecolor="#64748B",
        ),
        yaxis=dict(gridcolor="#334155"),
        margin=dict(t=80, b=50, l=10, r=80),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#CBD5E1",
        showlegend=False,
    )
    return fig

Y_METRIC_LABEL = {
    "egreso":  "Egreso licenciatura",
    "cohorte": f"Títulos / Ingreso (lag {LAG_LIC}y, %)",
}

def _trajectory_y_table(metric: str) -> tuple[pl.DataFrame, str, str, str]:
    """Returns (df[año,y], y-title, hover-fmt, hover-suffix)."""
    if metric == "cohorte":
        lag = df_lic_ingreso.with_columns((pl.col("año") + LAG_LIC).alias("año")) \
                            .rename({"ingreso_lic": "ingreso_lag"})
        t = (
            df_lic_titulos.join(lag, on="año", how="left")
            .with_columns((pl.col("titulos_expedidos") / pl.col("ingreso_lag") * 100).alias("y"))
            .filter(pl.col("y").is_not_null())
            .select(["año", "y"])
        )
        return t, Y_METRIC_LABEL["cohorte"], ".1f", "%"
    t = df_lic_egreso.rename({"egreso_lic": "y"})
    return t, Y_METRIC_LABEL["egreso"], ",", ""

def fig_trajectory(d_methods: pl.DataFrame, method: str, y_metric: str) -> go.Figure:
    """Connected scatter: % del método elegido vs métrica de egreso/cohorte, año por año."""
    year_total = d_methods.group_by("año").agg(pl.col("valor").sum().alias("total"))
    method_year = (
        d_methods.filter(pl.col("categoria") == method)
        .group_by("año").agg(pl.col("valor").sum().alias("v"))
    )
    y_table, y_title, fmt, suffix = _trajectory_y_table(y_metric)
    share = (
        year_total.join(method_year, on="año", how="left")
        .with_columns((pl.col("v").fill_null(0) / pl.col("total") * 100).alias("share"))
        .join(y_table, on="año", how="inner")
        .sort("año")
    )
    xs = share["share"].to_list()
    ys = share["y"].to_list()
    years = share["año"].to_list()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="lines",
        line=dict(color="#475569", width=1.5, dash="dot"),
        showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="markers+text",
        marker=dict(color="#2E86AB", size=11, line=dict(color="#0F172A", width=1)),
        text=[str(y) for y in years],
        textposition="top center",
        textfont=dict(size=10, color="#CBD5E1"),
        showlegend=False,
        hovertemplate=(
            f"<b>%{{text}}</b><br>{method}: %{{x:.1f}}%"
            f"<br>{y_title}: %{{y:{fmt}}}{suffix}<extra></extra>"
        ),
    ))
    yaxis = dict(title=y_title, gridcolor="#334155")
    if y_metric == "cohorte":
        yaxis["ticksuffix"] = "%"
    else:
        yaxis["tickformat"] = ","
    fig.update_layout(
        title=f"Trayectoria año a año: «{method}» vs {y_title.lower()}",
        height=480,
        xaxis=dict(title=f"% de «{method}» en títulos expedidos",
                   gridcolor="#334155", ticksuffix="%"),
        yaxis=yaxis,
        margin=dict(t=60, b=50, l=10, r=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#CBD5E1",
    )
    return fig

def fig_eficiencia(yr0: int, yr1: int) -> go.Figure:
    """Egreso vs títulos vs exámenes (licenciatura) — eficiencia terminal."""
    j = (
        df_lic_egreso.join(df_lic_examenes, on="año", how="full", coalesce=True)
        .join(df_lic_titulos, on="año", how="full", coalesce=True)
        .filter(pl.col("año").is_between(yr0, yr1))
        .sort("año")
        .fill_null(0)
    )
    fig = go.Figure()
    series = [
        ("egreso_lic",          "Egreso licenciatura",      "#2E86AB"),
        ("examenes_profesionales", "Exámenes profesionales", "#F4A261"),
        ("titulos_expedidos",   "Títulos expedidos",        "#3BB273"),
    ]
    for col, name, color in series:
        fig.add_trace(go.Scatter(
            x=j["año"].to_list(), y=j[col].to_list(),
            mode="lines+markers", name=name,
            line=dict(color=color, width=2.5),
            marker=dict(size=7),
            hovertemplate=f"<b>{name}</b><br>%{{x}}: %{{y:,}}<extra></extra>",
        ))
    fig.update_layout(
        title="Egreso vs titulación (licenciatura)",
        height=420,
        legend=dict(orientation="h", y=-0.18, x=0),
        **CHART_LAYOUT,
    )
    return fig

def _flujo_year(carrera: str | None, year: int) -> dict:
    """Aggregate one (carrera, año) for the Sankey.

    `carrera=None` → todas las carreras válidas (intersección de las 3 fuentes).
    """
    if carrera is None:
        pob = df_carr_pob.filter(pl.col("carrera").is_in(CARRERAS_FLUJO))
        eg  = df_carr_eg.filter(pl.col("carrera").is_in(CARRERAS_FLUJO))
        tit = df_carr_tit.filter(pl.col("carrera").is_in(CARRERAS_FLUJO))
    else:
        pob = df_carr_pob.filter(pl.col("carrera") == carrera)
        eg  = df_carr_eg.filter(pl.col("carrera") == carrera)
        tit = df_carr_tit.filter(pl.col("carrera") == carrera)

    pi       = int(pob.filter(pl.col("año") == year)["pi"].sum())
    rei      = int(pob.filter(pl.col("año") == year)["rei"].sum())
    rei_next = int(pob.filter(pl.col("año") == year + 1)["rei"].sum())
    egreso   = int(eg.filter(pl.col("año") == year)["egreso"].sum())
    tit_y    = int(tit.filter(pl.col("año") == year)["titulos"].sum())

    inscritos = pi + rei
    # cap continúan al saldo de no-egresados para mantener conservación en el sankey
    no_egresan = max(0, inscritos - egreso)
    continuan  = min(rei_next, no_egresan)
    abandono   = max(0, no_egresan - continuan)
    # cap titulados al egreso del año; el resto (cohortes anteriores) se reporta aparte
    tit_link = min(tit_y, egreso)
    sin_tit  = max(0, egreso - tit_link)

    return dict(
        pi=pi, rei=rei, inscritos=inscritos, egreso=egreso,
        continuan=continuan, abandono=abandono,
        tit_link=tit_link, sin_tit=sin_tit, tit_total=tit_y,
    )

def fig_flujo(carrera: str | None, year: int) -> go.Figure:
    """Sankey: ingreso → inscritos → {egresan, continúan, abandono} → titulación."""
    f = _flujo_year(carrera, year)
    label = "Todas las carreras" if carrera is None else carrera

    if f["inscritos"] == 0:
        return go.Figure().update_layout(
            title=f"Sin datos para «{label}» en {year}", **CHART_LAYOUT,
        )

    nodes = [
        f"Primer ingreso · {f['pi']:,}",          # 0
        f"Reingreso · {f['rei']:,}",              # 1
        f"Inscritos · {f['inscritos']:,}",        # 2
        f"Egresan · {f['egreso']:,}",             # 3
        f"Continúan en {year+1} · {f['continuan']:,}",  # 4
        f"Abandono · {f['abandono']:,}",          # 5
        f"Titulados ese año · {f['tit_link']:,}", # 6
        f"Sin título inmediato · {f['sin_tit']:,}",     # 7
    ]
    node_colors = ["#2E86AB", "#3BB273", "#475569",
                   "#F4A261", "#A78BFA", "#E84855",
                   "#3BB273", "#F4A261"]
    src   = [0, 1, 2, 2, 2, 3, 3]
    tgt   = [2, 2, 3, 4, 5, 6, 7]
    value = [f["pi"], f["rei"], f["egreso"], f["continuan"], f["abandono"],
             f["tit_link"], f["sin_tit"]]
    link_colors = [
        "rgba(46,134,171,0.35)", "rgba(59,178,115,0.35)",
        "rgba(244,162,97,0.35)", "rgba(167,139,250,0.35)", "rgba(232,72,85,0.35)",
        "rgba(59,178,115,0.45)", "rgba(244,162,97,0.35)",
    ]
    # Eliminar enlaces con value=0 para que el sankey no muestre nodos vacíos
    keep = [i for i, v in enumerate(value) if v > 0]
    src, tgt, value, link_colors = ([x[i] for i in keep] for x in (src, tgt, value, link_colors))

    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(
            label=nodes, color=node_colors,
            pad=22, thickness=18,
            line=dict(color="#0F172A", width=0.5),
        ),
        link=dict(source=src, target=tgt, value=value, color=link_colors,
                  hovertemplate="%{source.label} → %{target.label}<br>%{value:,}<extra></extra>"),
    ))

    sub = ""
    if f["tit_total"] > f["tit_link"]:
        extra = f["tit_total"] - f["tit_link"]
        sub = (
            f"<br><span style='font-size:0.78em;color:#94A3B8'>"
            f"Titulados totales en {year}: {f['tit_total']:,} "
            f"(+{extra:,} de cohortes anteriores, fuera del flujo)</span>"
        )
    fig.update_layout(
        title=f"Flujo · {label} · {year} → {year+1}{sub}",
        height=540,
        font_color="#CBD5E1",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=90, b=20, l=10, r=10),
    )
    return fig

SEX_COLORS = {"h": "#2E86AB", "m": "#F4A261"}  # hombres / mujeres
MIN_INSCRITOS = 500  # excluir carreras con muestra pequeña

def fig_abandono_sexo(yr0: int, yr1: int, top_n: int) -> go.Figure:
    """Dumbbell: carreras ordenadas por tasa de abandono, con marcadores H/M."""
    d = df_drop_carr_year.filter(pl.col("año").is_between(yr0, yr1))
    agg = (
        d.group_by("carrera")
        .agg([
            pl.col("insc_h").sum(), pl.col("insc_m").sum(),
            pl.col("abandono_h").sum(), pl.col("abandono_m").sum(),
        ])
        .with_columns([
            (pl.col("insc_h") + pl.col("insc_m")).alias("insc_total"),
            (pl.col("abandono_h") + pl.col("abandono_m")).alias("ab_total"),
        ])
        .filter(
            (pl.col("insc_total") >= MIN_INSCRITOS)
            & (pl.col("insc_h") > 0) & (pl.col("insc_m") > 0)
        )
        .with_columns([
            (pl.col("abandono_h") / pl.col("insc_h") * 100).alias("rate_h"),
            (pl.col("abandono_m") / pl.col("insc_m") * 100).alias("rate_m"),
            (pl.col("ab_total")   / pl.col("insc_total") * 100).alias("rate_total"),
        ])
        .sort("rate_total", descending=False)  # ascendente → mayor abandono arriba
        .tail(top_n)
    )
    if agg.is_empty():
        return go.Figure().update_layout(
            title=f"Sin datos para {yr0}–{yr1}", **CHART_LAYOUT,
        )

    carreras = agg["carrera"].to_list()
    rh = agg["rate_h"].to_list()
    rm = agg["rate_m"].to_list()
    ih = agg["insc_h"].to_list()
    im = agg["insc_m"].to_list()

    # Conectores entre H y M por carrera (truco de None-separator)
    x_lines, y_lines = [], []
    for h, m, c in zip(rh, rm, carreras):
        x_lines += [h, m, None]
        y_lines += [c, c, None]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_lines, y=y_lines, mode="lines",
        line=dict(color="#475569", width=1.5),
        showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=rh, y=carreras, mode="markers", name="Hombres",
        marker=dict(color=SEX_COLORS["h"], size=11,
                    line=dict(color="#0F172A", width=1)),
        customdata=ih,
        hovertemplate=("<b>%{y}</b><br>Hombres: %{x:.1f}%"
                       "  (inscritos H: %{customdata:,})<extra></extra>"),
    ))
    fig.add_trace(go.Scatter(
        x=rm, y=carreras, mode="markers", name="Mujeres",
        marker=dict(color=SEX_COLORS["m"], size=11,
                    line=dict(color="#0F172A", width=1)),
        customdata=im,
        hovertemplate=("<b>%{y}</b><br>Mujeres: %{x:.1f}%"
                       "  (inscritas M: %{customdata:,})<extra></extra>"),
    ))
    n = len(carreras)
    fig.update_layout(
        title=(
            f"Tasa de abandono por carrera y sexo · {yr0}–{yr1}"
            f"<br><span style='font-size:0.8em;color:#94A3B8'>"
            f"abandono = inscritos − egreso − reingreso al año siguiente · "
            f"muestra ≥ {MIN_INSCRITOS:,} inscritos · top {top_n}</span>"
        ),
        height=max(360, n * 22 + 140),
        xaxis=dict(title="% de abandono", ticksuffix="%", gridcolor="#334155"),
        yaxis=dict(categoryorder="array", categoryarray=carreras,
                   gridcolor="rgba(0,0,0,0)", automargin=True),
        legend=dict(orientation="h", y=-0.05, x=0),
        margin=dict(t=90, b=60, l=10, r=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#CBD5E1",
    )
    return fig

# ── Demanda por carrera (primer ingreso) ─────────────────────────────────────
# df_carr_pob ya trae pi = pi_h+pi_m por (carrera, año). La demanda de una
# carrera es su primer ingreso anual; el cambio entre dos años dice si crece o
# decae. Tras el _CARRERA_RENAME, los renombres 2020/2021 ya no fingen muertes.
MIN_PI_START = 100  # intake mínimo en el año inicial para evitar ruido de % change

_WINDOW = 3  # años promediados en cada extremo del rango

def fig_demanda_ranking(yr0: int, yr1: int, top_n: int) -> go.Figure:
    """Barra divergente: carreras que más crecieron/decayeron en primer ingreso yr0→yr1.

    Usa promedios de ventana de _WINDOW años en cada extremo para absorber picos anuales.
    """
    w = _WINDOW
    lbl0 = f"{yr0}–{yr0 + w - 1}"
    lbl1 = f"{yr1 - w + 1}–{yr1}"
    p0 = (df_carr_pob.filter(pl.col("año").is_between(yr0, yr0 + w - 1))
          .group_by("carrera").agg(pl.col("pi").mean().alias("pi0")))
    p1 = (df_carr_pob.filter(pl.col("año").is_between(yr1 - w + 1, yr1))
          .group_by("carrera").agg(pl.col("pi").mean().alias("pi1")))
    d = (
        p0.join(p1, on="carrera", how="inner")
        .filter(pl.col("pi0") >= MIN_PI_START)
        .with_columns(((pl.col("pi1") - pl.col("pi0")) / pl.col("pi0") * 100).alias("chg"))
        .sort("chg", descending=True)
    )
    if d.is_empty():
        return go.Figure().update_layout(
            title=f"Sin datos para {yr0}→{yr1}", **CHART_LAYOUT,
        )
    growers = d.head(top_n)
    shrinkers = d.tail(top_n)
    sel = pl.concat([shrinkers, growers]).unique(subset=["carrera"], maintain_order=True) \
            .sort("chg", descending=False)

    carreras = sel["carrera"].to_list()
    chg = sel["chg"].to_list()
    pi0 = sel["pi0"].to_list()
    pi1 = sel["pi1"].to_list()
    colors = ["#3BB273" if v >= 0 else "#E84855" for v in chg]

    fig = go.Figure(go.Bar(
        x=chg, y=carreras, orientation="h",
        marker_color=colors,
        customdata=list(zip(pi0, pi1)),
        text=[f"{v:+.0f}%" for v in chg],
        textposition="outside", textfont=dict(color="#CBD5E1", size=10),
        hovertemplate=("<b>%{y}</b><br>"
                       f"Promedio {lbl0}: %{{customdata[0]:,.0f}} → {lbl1}: %{{customdata[1]:,.0f}}<br>"
                       "Cambio: %{x:+.0f}%<extra></extra>"),
        cliponaxis=False,
    ))
    n = len(carreras)
    fig.update_layout(
        title=(
            f"Carreras con mayor crecimiento y caída en demanda · primer ingreso {yr0}→{yr1}"
            f"<br><span style='font-size:0.8em;color:#94A3B8'>"
            f"promedio {lbl0} vs {lbl1} · verde = crece · rojo = decae · "
            f"solo carreras con ≥ {MIN_PI_START} de primer ingreso promedio en {lbl0}</span>"
        ),
        height=max(360, n * 22 + 150),
        xaxis=dict(title="% de cambio en primer ingreso", ticksuffix="%",
                   gridcolor="#334155", zerolinecolor="#64748B"),
        yaxis=dict(categoryorder="array", categoryarray=carreras,
                   gridcolor="rgba(0,0,0,0)", automargin=True),
        margin=dict(t=80, b=50, l=10, r=60),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#CBD5E1", showlegend=False,
    )
    return fig

_DEMANDA_PALETTE = px.colors.qualitative.Plotly + px.colors.qualitative.Set2

def fig_demanda_trend(carreras: list[str], yr0: int, yr1: int) -> go.Figure:
    """Líneas de primer ingreso por carrera. Vacío = top 6 por ingreso en yr1."""
    base = df_carr_pob.filter(pl.col("año").is_between(yr0, yr1))
    if not carreras:
        carreras = (
            base.filter(pl.col("año") == yr1)
            .sort("pi", descending=True).head(6)["carrera"].to_list()
        )
    fig = go.Figure()
    for i, c in enumerate(sorted(carreras)):
        sub = base.filter(pl.col("carrera") == c).sort("año")
        if sub.is_empty():
            continue
        fig.add_trace(go.Scatter(
            x=sub["año"].to_list(), y=sub["pi"].to_list(),
            mode="lines+markers", name=c,
            line=dict(color=_DEMANDA_PALETTE[i % len(_DEMANDA_PALETTE)], width=2.5),
            marker=dict(size=6),
            hovertemplate=f"<b>{c}</b><br>%{{x}}: %{{y:,}}<extra></extra>",
        ))
    fig.update_layout(
        title="Primer ingreso anual por carrera",
        height=460,
        yaxis=dict(title="Primer ingreso", gridcolor="#334155", tickformat=",", rangemode="tozero"),
        xaxis=dict(gridcolor="#334155", dtick=2),
        legend=dict(orientation="h", y=-0.22, x=0, font=dict(size=10)),
        margin=dict(t=50, b=90, l=10, r=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#CBD5E1",
    )
    return fig

def fig_gender_scatter(yr0: int, yr1: int) -> go.Figure:
    """Dispersión: %mujeres inscritas vs brecha de abandono (M−H) por carrera.

    Prueba el efecto «el sexo minoritario abandona más»: entre más femenina la
    carrera, más se inclina la brecha hacia que los hombres abandonen (y al revés).
    """
    agg = (
        df_drop_carr_year.filter(pl.col("año").is_between(yr0, yr1))
        .group_by("carrera")
        .agg([
            pl.col("insc_h").sum(), pl.col("insc_m").sum(),
            pl.col("abandono_h").sum(), pl.col("abandono_m").sum(),
        ])
        .with_columns((pl.col("insc_h") + pl.col("insc_m")).alias("insc_total"))
        .filter((pl.col("insc_total") >= MIN_INSCRITOS)
                & (pl.col("insc_h") > 0) & (pl.col("insc_m") > 0))
        .with_columns([
            (pl.col("abandono_h") / pl.col("insc_h") * 100).alias("rate_h"),
            (pl.col("abandono_m") / pl.col("insc_m") * 100).alias("rate_m"),
            (pl.col("insc_m") / pl.col("insc_total") * 100).alias("pct_fem"),
        ])
        .with_columns((pl.col("rate_m") - pl.col("rate_h")).alias("gap"))
    )
    if agg.height < 3:
        return go.Figure().update_layout(
            title=f"Muestra insuficiente para {yr0}–{yr1}", **CHART_LAYOUT,
        )

    x = agg["pct_fem"].to_numpy()
    y = agg["gap"].to_numpy()
    # ajuste OLS y r de Pearson (scipy no disponible)
    slope, intercept = np.polyfit(x, y, 1)
    r = float(np.corrcoef(x, y)[0, 1])
    xs_fit = np.array([x.min(), x.max()])
    ys_fit = slope * xs_fit + intercept

    colors = ["#E84855" if g > 0 else "#3BB273" for g in y]  # rojo = mujeres abandonan más
    sizes = (agg["insc_total"].to_numpy()) ** 0.5
    sizes = 6 + 24 * (sizes - sizes.min()) / (np.ptp(sizes) or 1)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=xs_fit, y=ys_fit, mode="lines",
        line=dict(color="#94A3B8", width=2, dash="dash"),
        name=f"Ajuste · r = {r:.2f}", hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="markers",
        marker=dict(color=colors, size=sizes, opacity=0.75,
                    line=dict(color="#0F172A", width=1)),
        customdata=list(zip(agg["carrera"].to_list(),
                            agg["rate_m"].to_list(), agg["rate_h"].to_list(),
                            agg["insc_total"].to_list())),
        hovertemplate=("<b>%{customdata[0]}</b><br>"
                       "%mujeres: %{x:.0f}%<br>"
                       "Abandono M: %{customdata[1]:.1f}%  ·  H: %{customdata[2]:.1f}%<br>"
                       "Brecha (M−H): %{y:+.1f} pp  ·  inscritos: %{customdata[3]:,}<extra></extra>"),
        showlegend=False,
    ))
    fig.add_hline(y=0, line=dict(color="#64748B", width=1))
    fig.update_layout(
        title=(
            f"¿El sexo minoritario abandona más? · {yr0}–{yr1}"
            f"<br><span style='font-size:0.8em;color:#94A3B8'>"
            f"cada punto = una carrera (tamaño = inscritos) · arriba/rojo = mujeres abandonan más · "
            f"pendiente negativa ⇒ efecto minoría</span>"
        ),
        height=480,
        xaxis=dict(title="% de mujeres inscritas", ticksuffix="%", gridcolor="#334155"),
        yaxis=dict(title="Brecha de abandono M − H (pp)", ticksuffix=" pp", gridcolor="#334155"),
        legend=dict(orientation="h", y=-0.18, x=0),
        margin=dict(t=80, b=60, l=10, r=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#CBD5E1",
    )
    return fig

# Carreras disponibles para el selector de tendencia de demanda
DEMANDA_CARRERAS = sorted(df_carr_pob["carrera"].unique().to_list())

MIN_AVG_EGRESO = 10  # promedio anual mínimo de egresados para filtrar ruido

def fig_pipeline_ratios_carrera(yr0: int, yr1: int) -> go.Figure:
    """3-dot connected chart: tres ratios del pipeline por carrera, eje compartido 0-100%.

    Base = reinscritos (rei), no inscritos totales (pi+rei), para reflejar estudiantes
    en continuación sin contaminar con el primer ingreso del año.
    - r_rei_eg  = egresados / reinscritos
    - r_rei_tit = titulados / reinscritos
    - r_eg_tit  = titulados / egresados
    Todas las carreras con >= MIN_AVG_EGRESO, ordenadas por r_rei_eg ascendente.
    """
    n_yrs = max(yr1 - yr0, 1)
    rei = (
        df_carr_pob.filter(pl.col("año").is_between(yr0, yr1))
        .group_by("carrera")
        .agg((pl.col("rei").sum() / n_yrs).alias("avg_rei"))
    )
    eg = (
        df_carr_eg.filter(pl.col("año").is_between(yr0, yr1))
        .group_by("carrera")
        .agg((pl.col("egreso").sum() / n_yrs).alias("avg_eg"))
    )
    tit = (
        df_carr_tit.filter(pl.col("año").is_between(yr0, yr1))
        .group_by("carrera")
        .agg((pl.col("titulos").sum() / n_yrs).alias("avg_tit"))
    )
    agg = (
        rei.join(eg, on="carrera", how="full", coalesce=True)
        .join(tit, on="carrera", how="full", coalesce=True)
        .with_columns([
            pl.col("avg_rei").fill_null(0),
            pl.col("avg_eg").fill_null(0),
            pl.col("avg_tit").fill_null(0),
        ])
        .filter(pl.col("avg_eg") >= MIN_AVG_EGRESO)
        .with_columns([
            (pl.col("avg_eg")  / pl.col("avg_rei").replace(0, None) * 100).alias("r_rei_eg"),
            (pl.col("avg_tit") / pl.col("avg_rei").replace(0, None) * 100).alias("r_rei_tit"),
            (pl.col("avg_tit") / pl.col("avg_eg").replace(0, None)  * 100).alias("r_eg_tit"),
        ])
        .fill_null(0)
        .sort("r_rei_eg", descending=False)
    )

    if agg.is_empty():
        return go.Figure().update_layout(title=f"Sin datos para {yr0}–{yr1}", **CHART_LAYOUT)

    carreras  = agg["carrera"].to_list()
    r_rei_eg  = agg["r_rei_eg"].to_list()
    r_rei_tit = agg["r_rei_tit"].to_list()
    r_eg_tit  = agg["r_eg_tit"].to_list()
    avg_rei   = agg["avg_rei"].to_list()
    avg_eg    = agg["avg_eg"].to_list()
    avg_tit   = agg["avg_tit"].to_list()

    x_lines, y_lines = [], []
    for a, b, c_, car in zip(r_rei_eg, r_rei_tit, r_eg_tit, carreras):
        lo, hi = min(a, b, c_), max(a, b, c_)
        x_lines += [lo, hi, None]
        y_lines += [car, car, None]

    traces = [
        (r_rei_eg,  "Egresados / Reinscritos", "#F4A261",
         list(zip(avg_rei, avg_eg)),
         "<b>%{y}</b><br>Egresados/Reinscritos: %{x:.1f}%"
         "<br>Reinscritos/año: %{customdata[0]:.0f}  ·  Egresados/año: %{customdata[1]:.0f}<extra></extra>"),
        (r_rei_tit, "Titulados / Reinscritos",  "#2E86AB",
         list(zip(avg_rei, avg_tit)),
         "<b>%{y}</b><br>Titulados/Reinscritos: %{x:.1f}%"
         "<br>Reinscritos/año: %{customdata[0]:.0f}  ·  Titulados/año: %{customdata[1]:.0f}<extra></extra>"),
        (r_eg_tit,  "Titulados / Egresados",    "#3BB273",
         list(zip(avg_eg, avg_tit)),
         "<b>%{y}</b><br>Titulados/Egresados: %{x:.1f}%"
         "<br>Egresados/año: %{customdata[0]:.0f}  ·  Titulados/año: %{customdata[1]:.0f}<extra></extra>"),
    ]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_lines, y=y_lines, mode="lines",
        line=dict(color="#334155", width=1.5),
        showlegend=False, hoverinfo="skip",
    ))
    for vals, name, color, cdata, tmpl in traces:
        fig.add_trace(go.Scatter(
            x=vals, y=carreras, mode="markers", name=name,
            marker=dict(color=color, size=10, line=dict(color="#0F172A", width=1)),
            customdata=cdata,
            hovertemplate=tmpl,
        ))

    n = len(carreras)
    fig.update_layout(
        title=(
            f"Eficiencia del pipeline por carrera · promedio anual {yr0}–{yr1}"
            f"<br><span style='font-size:0.8em;color:#94A3B8'>"
            f"ordenado por Egresados/Reinscritos · mínimo {MIN_AVG_EGRESO} egresados/año</span>"
        ),
        height=max(400, n * 22 + 150),
        xaxis=dict(title="% del grupo base", ticksuffix="%",
                   gridcolor="#334155", range=[0, 105]),
        yaxis=dict(categoryorder="array", categoryarray=carreras,
                   gridcolor="rgba(0,0,0,0)", automargin=True),
        legend=dict(orientation="h", y=-0.05, x=0),
        margin=dict(t=90, b=70, l=10, r=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#CBD5E1",
    )
    return fig

# ── Layout ────────────────────────────────────────────────────────────────────

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
app.title = "UNAM · Egreso y Titulación"

def kpi(title: str, value: str, sub: str = "") -> dbc.Col:
    return dbc.Col(html.Div([
        html.Div(title, style={"color": "#94A3B8", "fontSize": "0.85rem"}),
        html.Div(value, style={"color": "#F8FAFC", "fontSize": "1.8rem", "fontWeight": 700}),
        html.Div(sub, style={"color": "#64748B", "fontSize": "0.75rem"}) if sub else None,
    ], style=CARD_STYLE), md=3)

app.layout = dbc.Container([
    html.H2("UNAM · Egreso y Titulación", style={"color": "#F8FAFC", "marginTop": "18px"}),
    html.P(f"Datos del Anuario Estadístico UNAM, hojas «egr y tit» y «lic x car op» "
           f"({Y_MIN}–{Y_MAX})",
           style={"color": "#94A3B8"}),

    dbc.Row([
        dbc.Col([
            html.Label("Rango de años", style={"color": "#CBD5E1"}),
            dcc.RangeSlider(
                id="year-range",
                min=Y_MIN, max=Y_MAX, step=1,
                value=[Y_MIN, Y_MAX],
                marks={y: str(y) for y in YEARS},
                tooltip={"placement": "bottom"},
            ),
        ], md=12),
    ], className="mb-3"),

    dbc.Tabs([
        dbc.Tab(label="Vista general", tab_id="tab-overview", children=[
            dbc.Row(id="kpis", className="mb-3 mt-3"),
            dbc.Row([
                dbc.Col(dcc.Graph(id="g-egreso"), md=6),
                dbc.Col(dcc.Graph(id="g-grado"), md=6),
            ], className="mb-3"),
            dbc.Row([
                dbc.Col(dcc.Graph(id="g-ingreso"), md=6),
                dbc.Col(dcc.Graph(id="g-eficiencia"), md=6),
            ], className="mb-3"),
            dbc.Row([
                dbc.Col(dcc.Graph(id="g-pipeline"), md=12),
            ], className="mb-3"),
            dbc.Row([
                dbc.Col(dcc.Graph(id="g-share"), md=12),
            ], className="mb-3"),
            dbc.Row([
                dbc.Col(dcc.Graph(id="g-top-method"), md=12),
            ], className="mb-3"),
            dbc.Row([
                dbc.Col(dcc.Graph(id="g-contribution"), md=12),
            ], className="mb-3"),
            dbc.Row([
                dbc.Col([
                    html.Label("Método a comparar", style={"color": "#CBD5E1"}),
                    dcc.Dropdown(
                        id="method-pick",
                        options=[{"label": m, "value": m} for m in METHODS_SORTED],
                        value=METHODS_SORTED[0],
                        clearable=False,
                        style={"backgroundColor": "#1E293B", "color": "#0F172A"},
                    ),
                ], md=6),
                dbc.Col([
                    html.Label("Eje Y", style={"color": "#CBD5E1"}),
                    dcc.RadioItems(
                        id="trajectory-y",
                        options=[
                            {"label": "  Egreso licenciatura", "value": "egreso"},
                            {"label": f"  Títulos / Ingreso (lag {LAG_LIC}y)", "value": "cohorte"},
                        ],
                        value="egreso",
                        inline=True,
                        inputStyle={"marginLeft": "12px", "marginRight": "4px"},
                        style={"color": "#CBD5E1"},
                    ),
                ], md=6),
            ], className="mb-2"),
            dbc.Row([
                dbc.Col(dcc.Graph(id="g-trajectory"), md=12),
            ]),
        ]),
        dbc.Tab(label="Flujo por carrera", tab_id="tab-flujo", children=[
            dbc.Row([
                dbc.Col([
                    html.Label("Carrera", style={"color": "#CBD5E1"}),
                    dcc.Dropdown(
                        id="flujo-carrera",
                        options=([{"label": "— Todas las carreras —", "value": "__ALL__"}]
                                 + [{"label": c, "value": c} for c in CARRERAS_FLUJO]),
                        value="__ALL__",
                        clearable=False,
                        style={"backgroundColor": "#1E293B", "color": "#0F172A"},
                    ),
                ], md=8),
                dbc.Col([
                    html.Label("Año base", style={"color": "#CBD5E1"}),
                    dcc.Dropdown(
                        id="flujo-year",
                        options=[{"label": str(y), "value": y} for y in FLUJO_YEARS],
                        value=FLUJO_YEARS[-1],
                        clearable=False,
                        style={"backgroundColor": "#1E293B", "color": "#0F172A"},
                    ),
                ], md=4),
            ], className="mb-2 mt-3"),
            dbc.Row([
                dbc.Col(html.Div(
                    "Inscritos = primer ingreso + reingreso del año. "
                    "Continúan = reingreso registrado al año siguiente (acotado al saldo no-egresado). "
                    "Abandono = inscritos − egresan − continúan. "
                    "Titulados ese año se acota al egreso del año; el excedente proviene de cohortes previas.",
                    style={"color": "#94A3B8", "fontSize": "0.8rem", "marginBottom": "8px"},
                ), md=12),
            ]),
            dbc.Row([
                dbc.Col(dcc.Graph(id="g-flujo"), md=12),
            ]),
        ]),
        dbc.Tab(label="Abandono por sexo", tab_id="tab-abandono", children=[
            dbc.Row([
                dbc.Col([
                    html.Label("Mostrar top", style={"color": "#CBD5E1"}),
                    dcc.Dropdown(
                        id="abandono-topn",
                        options=[{"label": f"Top {n}", "value": n} for n in [15, 30, 50, 100]],
                        value=30,
                        clearable=False,
                        style={"backgroundColor": "#1E293B", "color": "#0F172A"},
                    ),
                ], md=4),
            ], className="mb-2 mt-3"),
            dbc.Row([
                dbc.Col(html.Div(
                    "Ranking por tasa global (H+M). Cada par de puntos compara la "
                    "tasa de abandono de hombres (azul) y mujeres (naranja) en la misma carrera. "
                    "Una brecha grande indica que un sexo abandona mucho más que el otro. "
                    f"Se excluyen carreras con < {MIN_INSCRITOS:,} inscritos en el rango.",
                    style={"color": "#94A3B8", "fontSize": "0.8rem", "marginBottom": "8px"},
                ), md=12),
            ]),
            dbc.Row([
                dbc.Col(dcc.Graph(id="g-abandono"), md=12),
            ], className="mb-3"),
            dbc.Row([
                dbc.Col(dcc.Graph(id="g-gender-scatter"), md=12),
            ]),
        ]),
        dbc.Tab(label="Demanda por carrera", tab_id="tab-demanda", children=[
            dbc.Row([
                dbc.Col([
                    html.Label("Mostrar top", style={"color": "#CBD5E1"}),
                    dcc.Dropdown(
                        id="demanda-topn",
                        options=[{"label": f"Top {n} ↑ / {n} ↓", "value": n}
                                 for n in [8, 12, 20]],
                        value=12,
                        clearable=False,
                        style={"backgroundColor": "#1E293B", "color": "#0F172A"},
                    ),
                ], md=4),
            ], className="mb-2 mt-3"),
            dbc.Row([
                dbc.Col(html.Div(
                    "Cambio en primer ingreso entre los años extremos del rango seleccionado arriba. "
                    "Carreras renombradas (p. ej. «Química Farmacéutico Biológica») se consolidan; "
                    "Ingeniería Mecatrónica es un cierre real en 2021.",
                    style={"color": "#94A3B8", "fontSize": "0.8rem", "marginBottom": "8px"},
                ), md=12),
            ]),
            dbc.Row([
                dbc.Col(dcc.Graph(id="g-demanda-ranking"), md=12),
            ], className="mb-3"),
            dbc.Row([
                dbc.Col([
                    html.Label("Carreras a comparar (vacío = top 6 por ingreso)",
                               style={"color": "#CBD5E1"}),
                    dcc.Dropdown(
                        id="demanda-carrera",
                        options=[{"label": c, "value": c} for c in DEMANDA_CARRERAS],
                        value=[], multi=True,
                        placeholder="Seleccionar carreras…",
                        style={"backgroundColor": "#1E293B", "color": "#0F172A"},
                    ),
                ], md=12),
            ], className="mb-2"),
            dbc.Row([
                dbc.Col(dcc.Graph(id="g-demanda-trend"), md=12),
            ], className="mb-3"),
            html.Hr(style={"borderColor": "#334155"}),
            dbc.Row([
                dbc.Col(html.Div(
                    "Naranja = Egresados/Reinscritos · Azul = Titulados/Reinscritos · Verde = Titulados/Egresados. "
                    "Base: reinscritos (estudiantes en continuación), no inscritos totales. "
                    "Ordenado por Egresados/Reinscritos ascendente.",
                    style={"color": "#94A3B8", "fontSize": "0.8rem", "marginBottom": "6px"},
                ), md=12),
            ]),
            dbc.Row([
                dbc.Col(dcc.Graph(id="g-et-carrera"), md=12),
            ]),
        ]),
        dbc.Tab(label="Trayectorias por método", tab_id="tab-trayectorias", children=[
            dbc.Row([
                dbc.Col([
                    html.Label("Carreras (vacío = todas)", style={"color": "#CBD5E1"}),
                    dcc.Dropdown(
                        id="carrera-pick",
                        options=[{"label": c, "value": c} for c in CARRERAS],
                        value=[],
                        multi=True,
                        placeholder="Seleccionar carreras…",
                        style={"backgroundColor": "#1E293B", "color": "#0F172A"},
                    ),
                ], md=12),
            ], className="mb-3 mt-3"),
            dbc.Row([
                dbc.Col(dcc.Graph(id="g-trayectorias"), md=12),
            ]),
        ]),
    ], id="tabs", active_tab="tab-overview"),
], fluid=True, style={"backgroundColor": "#0F172A", "minHeight": "100vh", "padding": "20px"})

# ── Callback ─────────────────────────────────────────────────────────────────

@app.callback(
    Output("kpis", "children"),
    Output("g-egreso", "figure"),
    Output("g-grado", "figure"),
    Output("g-ingreso", "figure"),
    Output("g-eficiencia", "figure"),
    Output("g-pipeline", "figure"),
    Output("g-share", "figure"),
    Output("g-top-method", "figure"),
    Output("g-contribution", "figure"),
    Input("year-range", "value"),
)
def update_all(year_range):
    yr0, yr1 = year_range
    eg = df_egreso.filter(pl.col("año").is_between(yr0, yr1))
    gr = df_grado.filter(pl.col("año").is_between(yr0, yr1))
    me = df_titulos.filter(pl.col("año").is_between(yr0, yr1))
    ig = df_ingreso_nivel.filter(pl.col("año").is_between(yr0, yr1))

    k = compute_kpis(yr0, yr1)
    kpi_row = [
        kpi("Egreso total", f"{k['total']:,}", f"Año {k['año']}"),
        kpi("Doctorados otorgados", f"{k['doctorado']:,}", f"Año {k['año']}"),
        kpi("Títulos expedidos (lic.)", f"{k['titulos']:,}", f"Año {k['año']}"),
        kpi("Tasa de titulación lic.", f"{k['tasa']:.1f}%",
            f"títulos / egreso · {k['año']}"),
    ]
    return (
        kpi_row,
        fig_egreso(eg),
        fig_grado(gr),
        fig_ingreso(ig),
        fig_eficiencia(yr0, yr1),
        fig_pipeline(yr0, yr1),
        fig_share(me),
        fig_top_method_per_entidad(yr0, yr1),
        fig_contribution(yr0, yr1),
    )

@app.callback(
    Output("g-trajectory", "figure"),
    Input("year-range", "value"),
    Input("method-pick", "value"),
    Input("trajectory-y", "value"),
)
def update_trajectory(year_range, method, y_metric):
    yr0, yr1 = year_range
    d = df_titulos.filter(pl.col("año").is_between(yr0, yr1))
    return fig_trajectory(d, method, y_metric)

@app.callback(
    Output("g-trayectorias", "figure"),
    Input("year-range", "value"),
    Input("carrera-pick", "value"),
)
def update_trayectorias(year_range, carreras):
    yr0, yr1 = year_range
    d = _titulos_raw.filter(pl.col("año").is_between(yr0, yr1))
    if carreras:
        d = d.filter(pl.col("carrera").is_in(carreras))
    agg = (
        d.group_by(["año", "opcion_titulacion"]).agg(pl.col("total").sum().alias("valor"))
        .rename({"opcion_titulacion": "categoria"})
    )
    return fig_trayectorias(agg)

@app.callback(
    Output("g-flujo", "figure"),
    Input("flujo-carrera", "value"),
    Input("flujo-year", "value"),
)
def update_flujo(carrera, year):
    return fig_flujo(None if carrera == "__ALL__" else carrera, int(year))

@app.callback(
    Output("g-abandono", "figure"),
    Input("year-range", "value"),
    Input("abandono-topn", "value"),
)
def update_abandono(year_range, top_n):
    yr0, yr1 = year_range
    return fig_abandono_sexo(int(yr0), int(yr1), int(top_n))

@app.callback(
    Output("g-gender-scatter", "figure"),
    Input("year-range", "value"),
)
def update_gender_scatter(year_range):
    yr0, yr1 = year_range
    return fig_gender_scatter(int(yr0), int(yr1))

@app.callback(
    Output("g-demanda-ranking", "figure"),
    Input("year-range", "value"),
    Input("demanda-topn", "value"),
)
def update_demanda_ranking(year_range, top_n):
    yr0, yr1 = year_range
    return fig_demanda_ranking(int(yr0), int(yr1), int(top_n))

@app.callback(
    Output("g-demanda-trend", "figure"),
    Input("demanda-carrera", "value"),
    Input("year-range", "value"),
)
def update_demanda_trend(carreras, year_range):
    yr0, yr1 = year_range
    return fig_demanda_trend(carreras or [], int(yr0), int(yr1))

@app.callback(
    Output("g-et-carrera", "figure"),
    Input("year-range", "value"),
)
def update_et_carrera(year_range):
    yr0, yr1 = year_range
    return fig_pipeline_ratios_carrera(int(yr0), int(yr1))

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)
