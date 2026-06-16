"""
Dashboard: SEDENA — Actos Jurídicos de Transparencia (2023–2025)
Fuente: Plataforma Nacional de Transparencia, Art. 70 Frac. XXVII

Run: uv run python dashboard/sedena.py
"""

import polars as pl
import plotly.graph_objects as go
from dash import Dash, dcc, html
import dash_bootstrap_components as dbc

# ─── Column names ─────────────────────────────────────────────────────────────
C_AÑO     = "Ejercicio"
C_TIPO    = "Tipo de acto jurídico (catálogo)"
C_OBJETO  = "Objeto de la realización del acto jurídico"
C_UNIDAD  = "Unidad(es) o área(s) responsable(s) de instrumentación"
C_SECTOR  = "Sector al cual se otorgó el acto jurídico (catálogo)"
C_MONTO   = "Monto total o beneficio, servicio y/o recurso público aprovechado"
C_PERIODO = "Fecha de inicio del periodo que se informa"
C_RAZON_23 = "Razón social del titular al cual se otorgó el acto jurídico"
C_RAZON_24 = "Razón social de la persona moral titular a quien se otorgó el acto jurídico"

# ─── Load ─────────────────────────────────────────────────────────────────────
def _load(path: str, razon_col: str) -> pl.DataFrame:
    return (
        pl.read_csv(path, encoding="utf-8-sig", infer_schema_length=0)
        .rename({
            C_AÑO:     "año",
            C_TIPO:    "tipo_acto",
            C_OBJETO:  "objeto",
            C_UNIDAD:  "unidad",
            C_SECTOR:  "sector",
            C_MONTO:   "monto_raw",
            C_PERIODO: "periodo_inicio",
            razon_col: "razon_social",
        })
        .select(["año", "tipo_acto", "objeto", "unidad", "sector", "monto_raw", "periodo_inicio", "razon_social"])
    )

df = pl.concat([
    _load("data/sedena/buscador_solicitudes_43334_2023.csv", C_RAZON_23),
    _load("data/sedena/buscador_solicitudes_43334_2024.csv", C_RAZON_24),
    _load("data/sedena/buscador_solicitudes_43334_2025.csv", C_RAZON_24),
], how="diagonal")

# ─── Clean ────────────────────────────────────────────────────────────────────
df = df.with_columns([
    pl.col("monto_raw")
      .str.replace_all(",", "", literal=True)
      .str.replace_all("$", "", literal=True)
      .str.strip_chars()
      .cast(pl.Float64, strict=False)
      .alias("monto_total"),
    pl.col("unidad").str.to_lowercase().str.contains("ingenieros").alias("es_dgi"),
    pl.col("objeto").str.to_lowercase().alias("obj_lc"),
])

df = df.with_columns(
    pl.when(pl.col("obj_lc").str.contains("pirotec|artificios|fuegos artificiales"))
    .then(pl.lit("Pirotecnia"))
    .when(pl.col("obj_lc").str.contains("explosivo"))
    .then(pl.lit("Explosivos"))
    .when(pl.col("obj_lc").str.contains("seguridad"))
    .then(pl.lit("Seguridad"))
    .when(pl.col("obj_lc").str.contains("fabricaci|comercializ"))
    .then(pl.lit("Fabricación y Comercio"))
    .otherwise(pl.lit("Otro"))
    .alias("cat_objeto")
)
df = df.with_columns(
    pl.when(pl.col("es_dgi"))
    .then(pl.lit("Contratos DGI"))
    .otherwise(pl.col("cat_objeto"))
    .alias("cat_objeto")
)

# ─── Pre-aggregate ────────────────────────────────────────────────────────────
AÑOS = ["2023", "2024", "2025"]

vol_by_year = (
    df.filter(pl.col("tipo_acto").is_in(["Permiso", "Licencia"]))
    .group_by("año", "tipo_acto")
    .agg(pl.len().alias("n"))
    .sort("año", "tipo_acto")
)

sector_by_year = (
    df.filter(
        pl.col("sector").is_in(["Público", "Privado"]) & ~pl.col("es_dgi")
    )
    .group_by("año", "sector")
    .agg(pl.len().alias("n"))
    .sort("año", "sector")
)

obj_by_year = (
    df.filter(~pl.col("es_dgi"))
    .group_by("año", "cat_objeto")
    .agg(pl.len().alias("n"))
    .sort("año", "cat_objeto")
)

dgi_df = df.filter(pl.col("es_dgi") & pl.col("monto_total").is_not_null())
dgi_by_obj = (
    dgi_df
    .with_columns(pl.col("objeto").str.slice(0, 38).alias("obj_label"))
    .group_by("obj_label")
    .agg(
        pl.col("monto_total").sum().alias("total_mxn"),
        pl.len().alias("n"),
    )
    .sort("total_mxn", descending=True)
    .head(15)
)

fees_2023 = df.filter((pl.col("año") == "2023") & pl.col("monto_total").is_not_null())

# ─── Pirotecnia pre-aggregations ──────────────────────────────────────────────
# Broader pattern captures "pirotécnico" (2024, accented) + "pirotecnicos" (2025)
PIRO_PAT = "pirotec|pirotéc|artificios|fuegos artificiales"

piro_df = (
    df.filter(~pl.col("es_dgi") & pl.col("obj_lc").str.contains(PIRO_PAT))
    .with_columns([
        pl.when(pl.col("obj_lc").str.contains("fabricac"))
        .then(pl.lit("Fabricación y venta"))
        .when(pl.col("obj_lc").str.contains(r"compra.?venta|compraventa"))
        .then(pl.lit("Compra-venta"))
        .when(pl.col("obj_lc").str.contains("venta"))
        .then(pl.lit("Compra, almac. y venta"))
        .otherwise(pl.lit("Compra y consumo"))
        .alias("actividad"),
        pl.col("periodo_inicio").str.slice(3, 2).cast(pl.Int32, strict=False).alias("_mes"),
    ])
    .with_columns(
        pl.when(pl.col("_mes") <= 3).then(pl.lit("Q1"))
        .when(pl.col("_mes") <= 6).then(pl.lit("Q2"))
        .when(pl.col("_mes") <= 9).then(pl.lit("Q3"))
        .otherwise(pl.lit("Q4"))
        .alias("trimestre")
    )
)

piro_act_año = (
    piro_df.group_by("año", "actividad")
    .agg(pl.len().alias("n"))
    .sort("año", "actividad")
)

piro_sector_año = (
    piro_df.filter(pl.col("sector").is_in(["Público", "Privado"]))
    .group_by("año", "sector")
    .agg(pl.len().alias("n"))
    .sort("año", "sector")
)

piro_quarterly = (
    piro_df.group_by("año", "trimestre")
    .agg(pl.len().alias("n"))
    .sort("año", "trimestre")
)

piro_empresas = (
    piro_df.filter(pl.col("razon_social").is_not_null())
    .group_by("razon_social")
    .agg(pl.len().alias("n"))
    .sort("n", descending=True)
    .head(20)
)

piro_n_2024   = piro_df.filter(pl.col("año") == "2024").height
piro_n_2025   = piro_df.filter(pl.col("año") == "2025").height
piro_n_total  = piro_df.height
piro_n_emps   = piro_df["razon_social"].drop_nulls().n_unique()
piro_pct_anom = piro_df["razon_social"].is_null().sum() / piro_n_total * 100

# ─── KPI values ───────────────────────────────────────────────────────────────
n_2023      = df.filter(pl.col("año") == "2023").height
n_2024      = df.filter(pl.col("año") == "2024").height
drop_pct    = (n_2024 - n_2023) / n_2023 * 100
dgi_total_b = float(dgi_df["monto_total"].sum()) / 1e9 if dgi_df.height > 0 else 0
avg_fee     = float(fees_2023["monto_total"].mean() or 0)

# ─── Theme ────────────────────────────────────────────────────────────────────
CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font_color="#CBD5E1",
)
GRID = "#334155"
CARD_STYLE = {
    "background": "#1E293B", "border": "1px solid #334155",
    "borderRadius": "8px", "padding": "16px", "textAlign": "center",
}

TAB_STYLE = {"backgroundColor": "#0F172A", "color": "#94A3B8", "borderTop": "none"}
TAB_SEL   = {"backgroundColor": "#1E293B", "color": "#F8FAFC",
             "borderTop": "2px solid #2E86AB", "fontWeight": "600"}

PERMISO_C  = "#2E86AB"
LICENCIA_C = "#F4A261"
PUBLICO_C  = "#3BB273"
PRIVADO_C  = "#E84855"

CAT_COLORS = {
    "Explosivos":             "#E84855",
    "Pirotecnia":             "#F4A261",
    "Seguridad":              "#2E86AB",
    "Fabricación y Comercio": "#3BB273",
    "Contratos DGI":          "#A78BFA",
    "Otro":                   "#475569",
}

def _title(main: str, sub: str) -> dict:
    return dict(
        text=f"<b>{main}</b><br><sup style='color:#94A3B8'>{sub}</sup>",
        font=dict(size=14),
    )

# ─── Figure factories ─────────────────────────────────────────────────────────

def fig_volume(d: pl.DataFrame) -> go.Figure:
    permisos  = {r["año"]: r["n"] for r in d.filter(pl.col("tipo_acto") == "Permiso").iter_rows(named=True)}
    licencias = {r["año"]: r["n"] for r in d.filter(pl.col("tipo_acto") == "Licencia").iter_rows(named=True)}

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=AÑOS, y=[permisos.get(a, 0) for a in AÑOS],
        name="Permisos", marker_color=PERMISO_C,
        hovertemplate="<b>Permisos %{x}</b>: %{y:,}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        x=AÑOS, y=[licencias.get(a, 0) for a in AÑOS],
        name="Licencias", marker_color=LICENCIA_C,
        hovertemplate="<b>Licencias %{x}</b>: %{y:,}<extra></extra>",
    ))
    fig.add_annotation(
        x=0.01, y=1.0,
        xref="paper", yref="paper",
        text="⚠ 2023 incluye duplicados<br>de reporte semestral",
        font=dict(color="#94A3B8", size=9), showarrow=False,
        xanchor="left", yanchor="top",
    )
    fig.update_layout(
        **CHART_LAYOUT,
        barmode="group",
        height=400,
        title=_title(
            "Los actos jurídicos cayeron 73% de 2023 a 2024",
            "Permisos y licencias por año | DGRMFA-SEDENA",
        ),
        yaxis=dict(gridcolor=GRID, title="Actos"),
        xaxis=dict(gridcolor=GRID),
        legend=dict(orientation="h", y=-0.18, x=0),
        margin=dict(t=70, b=60, l=10, r=10),
    )
    return fig


def fig_sector(d: pl.DataFrame) -> go.Figure:
    totals: dict[str, int] = {}
    pub:    dict[str, int] = {}
    priv:   dict[str, int] = {}

    for r in d.iter_rows(named=True):
        a, n = r["año"], r["n"]
        if a not in AÑOS:
            continue
        totals[a] = totals.get(a, 0) + n
        if r["sector"] == "Público":
            pub[a] = n
        elif r["sector"] == "Privado":
            priv[a] = n

    fig = go.Figure()
    for label, vals, color in [("Público", pub, PUBLICO_C), ("Privado", priv, PRIVADO_C)]:
        pcts = [vals.get(a, 0) / max(totals.get(a, 1), 1) * 100 for a in AÑOS]
        fig.add_trace(go.Bar(
            x=pcts, y=AÑOS,
            orientation="h", name=label,
            marker_color=color,
            text=[f"{p:.0f}%" for p in pcts],
            textposition="inside", insidetextanchor="middle",
            customdata=[[vals.get(a, 0)] for a in AÑOS],
            hovertemplate=f"<b>{label} %{{y}}</b>: %{{x:.0f}}%  (n=%{{customdata[0]:,}})<extra></extra>",
        ))
    fig.update_layout(
        **CHART_LAYOUT,
        barmode="stack",
        height=280,
        xaxis=dict(range=[0, 100], visible=False, gridcolor=GRID),
        yaxis=dict(gridcolor=GRID),
        title=_title(
            "El sector privado dominó en 2024; el público recupera en 2025",
            "Distribución público/privado por año (excluye contratos DGI)",
        ),
        legend=dict(orientation="h", y=-0.28, x=0),
        margin=dict(t=70, b=70, l=10, r=10),
    )
    return fig


def fig_objeto(d: pl.DataFrame) -> go.Figure:
    cats = ["Explosivos", "Pirotecnia", "Seguridad", "Fabricación y Comercio", "Otro"]
    totals: dict[str, int] = {}
    vals: dict[str, dict[str, int]] = {c: {} for c in cats}

    for r in d.iter_rows(named=True):
        a, c, n = r["año"], r["cat_objeto"], r["n"]
        if a not in AÑOS or c not in cats:
            continue
        totals[a] = totals.get(a, 0) + n
        vals[c][a] = n

    fig = go.Figure()
    for cat in cats:
        pcts = [vals[cat].get(a, 0) / max(totals.get(a, 1), 1) * 100 for a in AÑOS]
        fig.add_trace(go.Bar(
            x=pcts, y=AÑOS,
            orientation="h", name=cat,
            marker_color=CAT_COLORS[cat],
            text=[f"{p:.0f}%" if p >= 5 else "" for p in pcts],
            textposition="inside", insidetextanchor="middle",
            customdata=[[vals[cat].get(a, 0)] for a in AÑOS],
            hovertemplate=f"<b>{cat} %{{y}}</b>: %{{x:.1f}}%  (n=%{{customdata[0]:,}})<extra></extra>",
        ))
    fig.update_layout(
        **CHART_LAYOUT,
        barmode="stack",
        height=280,
        xaxis=dict(range=[0, 100], visible=False, gridcolor=GRID),
        yaxis=dict(gridcolor=GRID),
        title=_title(
            "La pirotecnia desplazó a los explosivos como actividad principal desde 2024",
            "% de actos por categoría | DGRMFA únicamente",
        ),
        legend=dict(orientation="h", y=-0.28, x=0),
        margin=dict(t=70, b=70, l=10, r=10),
    )
    return fig


def fig_dgi(d: pl.DataFrame) -> go.Figure:
    sorted_d = d.sort("total_mxn", descending=False)
    height = max(320, sorted_d.height * 30 + 80)

    fig = go.Figure(go.Bar(
        x=sorted_d["total_mxn"].to_list(),
        y=sorted_d["obj_label"].to_list(),
        orientation="h",
        marker_color=CAT_COLORS["Contratos DGI"],
        customdata=sorted_d["n"].to_list(),
        hovertemplate="<b>%{y}</b><br>$%{x:,.0f} MXN  (%{customdata} contratos)<extra></extra>",
    ))
    fig.update_layout(
        **CHART_LAYOUT,
        height=height,
        title=_title(
            f"En 2025, SEDENA reveló contratos de ingeniería por ${dgi_total_b:.1f} mil millones",
            f"Top {sorted_d.height} rubros de adquisición | DGI-SEDENA 2025",
        ),
        xaxis=dict(gridcolor=GRID, title="Monto total (MXN)", tickformat="$,.0f"),
        yaxis=dict(gridcolor="rgba(0,0,0,0)"),
        margin=dict(t=70, b=50, l=230, r=20),
    )
    return fig


def fig_fees(d: pl.DataFrame) -> go.Figure:
    fees = d["monto_total"].to_list()
    avg  = float(d["monto_total"].mean() or 0)

    fig = go.Figure(go.Histogram(
        x=fees, nbinsx=40,
        marker_color=PERMISO_C, opacity=0.85,
        hovertemplate="Cuota ~$%{x:,.0f}<br>Permisos: %{y}<extra></extra>",
    ))
    fig.add_vline(
        x=avg, line_dash="dash", line_color="#94A3B8",
        annotation_text=f"Promedio: ${avg:,.0f}",
        annotation_font_color="#94A3B8",
        annotation_position="top right",
    )
    fig.update_layout(
        **CHART_LAYOUT,
        height=340,
        title=_title(
            f"El permiso de explosivos promedio costó ${avg:,.0f} MXN en 2023",
            f"Distribución de cuotas | {len(fees):,} permisos con monto declarado",
        ),
        xaxis=dict(gridcolor=GRID, title="Monto (MXN)", tickformat="$,.0f"),
        yaxis=dict(gridcolor=GRID, title="Permisos"),
        margin=dict(t=70, b=50, l=60, r=10),
    )
    return fig


# ─── Pirotecnia figure factories ─────────────────────────────────────────────

ACTIVIDAD_CATS   = ["Fabricación y venta", "Compra-venta", "Compra, almac. y venta", "Compra y consumo"]
ACTIVIDAD_COLORS = {
    "Fabricación y venta":    "#2E86AB",
    "Compra-venta":           "#F4A261",
    "Compra, almac. y venta": "#3BB273",
    "Compra y consumo":       "#64748B",
}


def fig_piro_actividad(d: pl.DataFrame) -> go.Figure:
    años = ["2024", "2025"]
    totals: dict[str, int] = {a: 0 for a in años}
    vals: dict[str, dict[str, int]] = {c: {a: 0 for a in años} for c in ACTIVIDAD_CATS}

    for r in d.iter_rows(named=True):
        a, c, n = r["año"], r["actividad"], r["n"]
        if a not in años or c not in ACTIVIDAD_CATS:
            continue
        totals[a] += n
        vals[c][a] = n

    fig = go.Figure()
    for cat in ACTIVIDAD_CATS:
        pcts = [vals[cat][a] / max(totals[a], 1) * 100 for a in años]
        fig.add_trace(go.Bar(
            x=pcts, y=años,
            orientation="h", name=cat,
            marker_color=ACTIVIDAD_COLORS[cat],
            text=[f"{p:.0f}%" if p >= 5 else "" for p in pcts],
            textposition="inside", insidetextanchor="middle",
            customdata=[[vals[cat][a]] for a in años],
            hovertemplate=f"<b>{cat} %{{y}}</b>: %{{x:.0f}}%  (n=%{{customdata[0]:,}})<extra></extra>",
        ))
    fig.update_layout(
        **CHART_LAYOUT,
        barmode="stack",
        height=220,
        xaxis=dict(range=[0, 100], visible=False, gridcolor=GRID),
        yaxis=dict(gridcolor=GRID),
        title=_title(
            "En 2025, la fabricación y venta desplazó al simple consumo de químicos",
            "% de permisos pirotécnicos por tipo de actividad | DGRMFA-SEDENA",
        ),
        legend=dict(orientation="h", y=-0.38, x=0),
        margin=dict(t=70, b=90, l=10, r=10),
    )
    return fig


def fig_piro_sector(d: pl.DataFrame) -> go.Figure:
    años = ["2024", "2025"]
    totals: dict[str, int] = {a: 0 for a in años}
    pub:  dict[str, int]   = {a: 0 for a in años}
    priv: dict[str, int]   = {a: 0 for a in años}

    for r in d.iter_rows(named=True):
        a, s, n = r["año"], r["sector"], r["n"]
        if a not in años:
            continue
        totals[a] += n
        if s == "Público":
            pub[a]  = n
        elif s == "Privado":
            priv[a] = n

    fig = go.Figure()
    for label, vals, color in [("Privado", priv, PRIVADO_C), ("Público", pub, PUBLICO_C)]:
        pcts = [vals[a] / max(totals[a], 1) * 100 for a in años]
        fig.add_trace(go.Bar(
            x=pcts, y=años,
            orientation="h", name=label,
            marker_color=color,
            text=[f"{p:.0f}%" if p >= 5 else "" for p in pcts],
            textposition="inside", insidetextanchor="middle",
            customdata=[[vals[a]] for a in años],
            hovertemplate=f"<b>{label} %{{y}}</b>: %{{x:.0f}}%  (n=%{{customdata[0]:,}})<extra></extra>",
        ))
    fig.update_layout(
        **CHART_LAYOUT,
        barmode="stack",
        height=220,
        xaxis=dict(range=[0, 100], visible=False, gridcolor=GRID),
        yaxis=dict(gridcolor=GRID),
        title=_title(
            "El sector público apareció en 2025 con 32% de los permisos — ausente en 2024",
            "Distribución público / privado | 2024–2025",
        ),
        legend=dict(orientation="h", y=-0.38, x=0),
        margin=dict(t=70, b=90, l=10, r=10),
    )
    return fig


def fig_piro_quarterly(d: pl.DataFrame) -> go.Figure:
    años  = ["2024", "2025"]
    trims = ["Q1", "Q2", "Q3", "Q4"]
    vals: dict[str, dict[str, int]] = {a: {t: 0 for t in trims} for a in años}

    for r in d.iter_rows(named=True):
        a, t, n = r["año"], r["trimestre"], r["n"]
        if a in años and t in trims:
            vals[a][t] = n

    fig = go.Figure()
    colors_años = {"2024": PERMISO_C, "2025": LICENCIA_C}
    for a in años:
        fig.add_trace(go.Bar(
            x=trims,
            y=[vals[a][t] for t in trims],
            name=a,
            marker_color=colors_años[a],
            hovertemplate=f"<b>{a} %{{x}}</b>: %{{y:,}} permisos<extra></extra>",
        ))
    fig.update_layout(
        **CHART_LAYOUT,
        barmode="group",
        height=320,
        title=_title(
            "99% de los permisos se reportan en Q1: registro anual, no actividad trimestral",
            "Permisos pirotécnicos por trimestre de reporte | 2024–2025",
        ),
        yaxis=dict(gridcolor=GRID, title="Permisos"),
        xaxis=dict(gridcolor=GRID),
        legend=dict(orientation="h", y=-0.2, x=0),
        margin=dict(t=70, b=60, l=60, r=10),
    )
    return fig


def fig_piro_empresas(d: pl.DataFrame) -> go.Figure:
    sorted_d = d.sort("n", descending=False)
    height   = max(300, sorted_d.height * 26 + 60)
    labels   = [s[:48] + "…" if len(s) > 48 else s for s in sorted_d["razon_social"].to_list()]

    fig = go.Figure(go.Bar(
        x=sorted_d["n"].to_list(),
        y=labels,
        orientation="h",
        marker_color=CAT_COLORS["Pirotecnia"],
        hovertemplate="<b>%{y}</b><br>%{x} permisos<extra></extra>",
    ))
    fig.update_layout(
        **CHART_LAYOUT,
        height=height,
        title=_title(
            "141 empresas identificadas — el mismo nombre puede aparecer con distintas grafías",
            "Top 20 razones sociales | 2024–2025",
        ),
        xaxis=dict(gridcolor=GRID, title="Permisos"),
        yaxis=dict(gridcolor="rgba(0,0,0,0)"),
        margin=dict(t=70, b=50, l=310, r=10),
    )
    return fig


# ─── KPI card helper ──────────────────────────────────────────────────────────
def kpi_card(label: str, value: str, note: str) -> dbc.Col:
    return dbc.Col(html.Div([
        html.P(label, style={"color": "#94A3B8", "fontSize": "0.8rem", "marginBottom": "4px"}),
        html.P(value, style={"color": "#F8FAFC", "fontSize": "1.6rem", "fontWeight": "700", "marginBottom": "4px"}),
        html.P(note,  style={"color": "#64748B", "fontSize": "0.75rem", "marginBottom": 0}),
    ], style=CARD_STYLE), md=4)


# ─── Layout ───────────────────────────────────────────────────────────────────
app = Dash(__name__, external_stylesheets=[dbc.themes.SLATE], title="SEDENA — Actos Jurídicos")
app.layout = dbc.Container([

    html.H2("SEDENA — Actos Jurídicos de Transparencia",
            style={"color": "#F8FAFC", "paddingTop": "1.2rem", "marginBottom": "0.2rem"}),
    html.P("Permisos, licencias y contratos bajo Art. 70 Frac. XXVII | 2023–2025",
           style={"color": "#94A3B8", "fontSize": "0.85rem", "marginBottom": "1rem"}),

    dcc.Tabs([

        dcc.Tab(label="Resumen General", style=TAB_STYLE, selected_style=TAB_SEL, children=[
            dbc.Row([
                kpi_card("Caída 2023 → 2024",
                         f"{drop_pct:.0f}%",
                         f"de {n_2023:,} a {n_2024:,} actos jurídicos"),
                kpi_card("Licencias otorgadas",
                         "212 → 149 → 109",
                         "Caída del 49% en dos años"),
                kpi_card("Contratos DGI revelados en 2025",
                         f"${dgi_total_b:.1f} mil millones MXN",
                         "607 contratos de ingeniería — primera vez en el dataset"),
            ], className="g-3 mb-4 mt-3"),

            dbc.Row([
                dbc.Col(dcc.Graph(figure=fig_volume(vol_by_year), config={"displayModeBar": False}), md=7),
                dbc.Col(dcc.Graph(figure=fig_sector(sector_by_year), config={"displayModeBar": False}), md=5),
            ], className="g-3 mb-3"),

            dbc.Row([
                dbc.Col(dcc.Graph(figure=fig_objeto(obj_by_year), config={"displayModeBar": False}), md=12),
            ], className="g-3 mb-3"),

            dbc.Row([
                dbc.Col(dcc.Graph(figure=fig_dgi(dgi_by_obj), config={"displayModeBar": False}), md=7),
                dbc.Col(dcc.Graph(figure=fig_fees(fees_2023), config={"displayModeBar": False}), md=5),
            ], className="g-3 mb-4"),
        ]),

        dcc.Tab(label="Pirotecnia", style=TAB_STYLE, selected_style=TAB_SEL, children=[
            dbc.Row([
                kpi_card("Permisos pirotécnicos 2024–2025",
                         f"{piro_n_total:,}",
                         f"{piro_n_2024:,} en 2024 → {piro_n_2025:,} en 2025  |  cero en 2023"),
                kpi_card("Personas físicas sin nombre publicado",
                         f"{piro_pct_anom:.0f}%",
                         f"Solo {piro_n_emps} empresas identificadas de {piro_n_total:,} permisos"),
                kpi_card("Primera regulación formal",
                         "2024",
                         "La pirotecnia no aparecía en los reportes de transparencia de 2023"),
            ], className="g-3 mb-4 mt-3"),

            dbc.Row([
                dbc.Col(dcc.Graph(figure=fig_piro_actividad(piro_act_año), config={"displayModeBar": False}), md=6),
                dbc.Col(dcc.Graph(figure=fig_piro_sector(piro_sector_año), config={"displayModeBar": False}), md=6),
            ], className="g-3 mb-3"),

            dbc.Row([
                dbc.Col(dcc.Graph(figure=fig_piro_quarterly(piro_quarterly), config={"displayModeBar": False}), md=5),
                dbc.Col(dcc.Graph(figure=fig_piro_empresas(piro_empresas), config={"displayModeBar": False}), md=7),
            ], className="g-3 mb-4"),
        ]),

    ]),

], fluid=True, style={"backgroundColor": "#0F172A", "minHeight": "100vh"})

if __name__ == "__main__":
    app.run(debug=True)
