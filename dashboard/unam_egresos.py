"""Dashboard: UNAM — Egreso y Titulación

Source: data/unam_egresos_y_titulacion/{egreso, examenes_de_grado,
examenes_profesionales, titulacion_carreras}.csv

Títulos de licenciatura provienen del desglose por carrera/opción
(`titulacion_carreras.csv`). El agregado «titulos_expedidos.csv» queda
deprecated: tiene 2025 desactualizado (réplica de 2024) e incluye SUAYED,
mientras que el desglose por carrera es escolarizada únicamente y se
extiende hasta 2025 con datos correctos.
"""

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
_titulos_raw = pl.read_csv(f"{DATA}/titulacion_carreras.csv")
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
df_lic_egreso = (
    df_egreso.filter(pl.col("categoria") == "Licenciatura")
    .select(["año", pl.col("valor").alias("egreso_lic")])
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

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)
