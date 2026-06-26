"""
Dashboard: Gasto Programable Federal — México (2012–2026)
Fuente: CGPE / SHCP (normalizado por scripts/normalize_cgpe.py)

Tab 1 — Tendencia: evolución del gasto total y por grupo
Tab 2 — Concentración: qué ramos absorben el gasto
Tab 3 — Propuesto vs Aprobado: qué tanto modifica el Congreso

Run: uv run python dashboard/cgpe.py
"""

import numpy as np
import polars as pl
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc

# ── Data loading ──────────────────────────────────────────────────────────────

_df = pl.read_csv("data/cgpe_normalized/gasto_programable_normalizado.csv")

# Leaf-level rows: safe to sum, no double-counting
admin_det = _df.filter(
    (pl.col("clasificacion") == "administrativa") &
    (pl.col("tipo") == "detalle")
)

# Published annual totals (14 rows — 2021 missing)
_admin_tot = _df.filter(
    (pl.col("clasificacion") == "administrativa") &
    (pl.col("tipo") == "total")
).select(["año", "proyectado"])

# 2021 reconstructed from detalle (subtotal-sum double-counts nested Generales subgroups)
_tot_2021 = float(admin_det.filter(pl.col("año") == 2021)["proyectado"].sum())

totals_series = (
    _admin_tot
    .vstack(pl.DataFrame({"año": [2021], "proyectado": [_tot_2021]}))
    .sort("año")
)

# KPI constants
_val_2026 = float(totals_series.filter(pl.col("año") == 2026)["proyectado"][0])
_val_2025 = float(totals_series.filter(pl.col("año") == 2025)["proyectado"][0])
_val_2012 = float(totals_series.filter(pl.col("año") == 2012)["proyectado"][0])
_yoy_pct = (_val_2026 - _val_2025) / _val_2025 * 100
_mult = _val_2026 / _val_2012

# IMSS + Aportaciones share in 2026 (both from detalle for consistency)
_det_2026 = admin_det.filter(pl.col("año") == 2026)
_total_det_2026 = float(_det_2026["proyectado"].sum())
_imss_aport_val = float(
    _det_2026.filter(
        pl.col("ramo").is_in([
            "Instituto Mexicano del Seguro Social",
            "Aportaciones, subsidios y transferencias",
        ])
    )["proyectado"].sum()
)
_imss_aport_pct = _imss_aport_val / _total_det_2026 * 100

RANKING_YEARS = sorted(admin_det["año"].unique().to_list(), reverse=True)

# Pre-compute ramo % share of annual total — used by the participation line chart
_yr_totals = admin_det.group_by("año").agg(pl.col("proyectado").sum().alias("total_yr"))
share_by_ramo = (
    admin_det.group_by(["año", "ramo"])
    .agg(pl.col("proyectado").sum().alias("val"))
    .join(_yr_totals, on="año")
    .with_columns((pl.col("val") / pl.col("total_yr") * 100).alias("share"))
    .sort(["ramo", "año"])
)
# Dropdown options: ordered by 2026 share descending, then alphabetically for the rest
_ramos_by_share_2026 = (
    share_by_ramo.filter(pl.col("año") == 2026)
    .sort("share", descending=True)["ramo"].to_list()
)
_ramos_all = share_by_ramo["ramo"].unique().to_list()
_ramos_rest = sorted(r for r in _ramos_all if r not in _ramos_by_share_2026)
RAMO_OPTIONS = [{"label": r, "value": r} for r in _ramos_by_share_2026 + _ramos_rest]
DEFAULT_RAMOS = _ramos_by_share_2026[:5]

# ── Theme ─────────────────────────────────────────────────────────────────────

CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font_color="#CBD5E1",
)
CARD_STYLE = {
    "background": "#1E293B", "border": "1px solid #334155",
    "borderRadius": "8px", "padding": "16px",
}
TAB_STYLE = {"backgroundColor": "#0F172A", "color": "#94A3B8", "borderTop": "none"}
TAB_SEL = {
    "backgroundColor": "#1E293B", "color": "#F8FAFC",
    "borderTop": "2px solid #2E86AB", "fontWeight": "600",
}
GRAPH_CONFIG = {"displayModeBar": False}
FOCUS = "#2E86AB"
CONTEXT = "#475569"
GRUPO_COLORS = {
    "Generales": "#2E86AB",
    "Administrativos": "#F4A261",
    "Autónomos": "#3BB273",
}

# Short labels for slope chart annotations
_RAMO_SHORT = {
    "Instituto Mexicano del Seguro Social": "IMSS",
    "Aportaciones, subsidios y transferencias": "Aportaciones",
    "Petróleos Mexicanos": "Pemex",
    "Comisión Federal de Electricidad": "CFE",
    "Instituto de Seguridad y Servicios Sociales de los Trabajadores del Estado": "ISSSTE",
    "Bienestar": "Bienestar",
    "Educación Pública": "SEP",
    "Energía": "Energía",
    "Defensa Nacional": "SEDENA",
    "Servicios de Salud del Instituto Mexicano del Seguro Social para el Bienestar": "IMSS-Bienestar",
}


# ── Figure factories ──────────────────────────────────────────────────────────

def fig_trend_total(d_totals: pl.DataFrame) -> go.Figure:
    years = d_totals["año"].to_list()
    vals_b = [v / 1_000_000 for v in d_totals["proyectado"].to_list()]  # billones

    coef = np.polyfit(years, vals_b, 1)
    trend_y = [float(np.polyval(coef, y)) for y in years]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=years, y=vals_b,
        mode="lines+markers",
        line=dict(color=FOCUS, width=2.5),
        marker=dict(color=FOCUS, size=7),
        name="Gasto programable",
        hovertemplate="<b>%{x}</b>: %{y:.2f} billones de pesos<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=years, y=trend_y,
        mode="lines",
        line=dict(color="#64748B", width=1.2, dash="dot"),
        name="Tendencia ~6.7%/año",
        hoverinfo="skip",
    ))
    fig.add_vline(x=2019, line_dash="dot", line_color="#94A3B8",
                  annotation_text="2019<br>nueva adm.",
                  annotation_font_color="#94A3B8",
                  annotation_font_size=10,
                  annotation_position="top left")
    fig.add_vline(x=2025, line_dash="dot", line_color="#94A3B8",
                  annotation_text="2025<br>nueva adm.",
                  annotation_font_color="#94A3B8",
                  annotation_font_size=10,
                  annotation_position="top left")
    fig.update_layout(
        **CHART_LAYOUT, height=400,
        title=dict(
            text=(
                "<b>El gasto se multiplicó ×2.5 en 14 años — de 2.8 a 7.1 billones de pesos</b>"
                "<br><sup style='color:#94A3B8'>Gasto programable federal · clasificación administrativa"
                " · billones de pesos corrientes · 2012–2026</sup>"
            ),
        ),
        xaxis=dict(gridcolor="#334155", tickmode="linear", dtick=1),
        yaxis=dict(gridcolor="#334155", ticksuffix=" B"),
        legend=dict(orientation="h", y=-0.15, x=0),
        margin=dict(t=80, b=60, l=20, r=20),
    )
    return fig


def fig_grupo_share(d_det: pl.DataFrame) -> go.Figure:
    agg = (
        d_det.group_by(["año", "grupo"])
        .agg(pl.col("proyectado").sum())
        .sort(["año", "grupo"])
    )
    years_u = sorted(agg["año"].unique().to_list())
    totals_by_yr = {
        y: float(agg.filter(pl.col("año") == y)["proyectado"].sum())
        for y in years_u
    }

    fig = go.Figure()
    for grupo in ["Generales", "Administrativos", "Autónomos"]:
        g = agg.filter(pl.col("grupo") == grupo).sort("año")
        g_by_yr = {int(r["año"]): float(r["proyectado"]) for r in g.iter_rows(named=True)}
        pcts = [g_by_yr.get(y, 0) / totals_by_yr[y] * 100 for y in years_u]
        abs_b = [g_by_yr.get(y, 0) / 1_000_000 for y in years_u]
        fig.add_trace(go.Bar(
            x=years_u, y=pcts, name=grupo,
            marker_color=GRUPO_COLORS[grupo],
            customdata=abs_b,
            hovertemplate=f"<b>{grupo}</b> %{{x}}<br>%{{y:.1f}}% · %{{customdata:.2f}} B<extra></extra>",
        ))

    fig.update_layout(
        **CHART_LAYOUT, height=380,
        barmode="stack",
        title=dict(
            text=(
                "<b>Los Ramos Generales (IMSS, pensiones, aportaciones) "
                "crecieron 3× y concentran más del 65% del gasto</b>"
                "<br><sup style='color:#94A3B8'>% del gasto programable por tipo de ramo"
                " · clasificación administrativa</sup>"
            ),
        ),
        xaxis=dict(gridcolor="#334155", tickmode="linear", dtick=1),
        yaxis=dict(gridcolor="#334155", range=[0, 100], ticksuffix="%"),
        legend=dict(orientation="h", y=-0.18, x=0),
        margin=dict(t=80, b=70, l=20, r=20),
    )
    return fig


def fig_ranking(d_det: pl.DataFrame, year: int, n: int = 15) -> go.Figure:
    yr_data = d_det.filter(pl.col("año") == year)
    total_yr = float(yr_data["proyectado"].sum())

    highlight = {
        "Instituto Mexicano del Seguro Social",
        "Aportaciones, subsidios y transferencias",
    }
    imss_aport_yr = float(yr_data.filter(pl.col("ramo").is_in(highlight))["proyectado"].sum())
    ia_pct = imss_aport_yr / total_yr * 100

    agg = (
        yr_data.group_by("ramo")
        .agg(pl.col("proyectado").sum())
        .sort("proyectado", descending=True)
        .head(n)
        .sort("proyectado")
    )
    ramos = agg["ramo"].to_list()
    vals_b = (agg["proyectado"] / 1_000_000).to_list()
    pcts = [v * 1_000_000 / total_yr * 100 for v in vals_b]
    colors = [FOCUS if r in highlight else CONTEXT for r in ramos]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=vals_b, y=ramos, orientation="h",
        marker_color=colors,
        customdata=pcts,
        hovertemplate="<b>%{y}</b><br>%{x:.1f} B  (%{customdata:.1f}% del total)<extra></extra>",
    ))
    h = max(420, n * 32 + 80)
    fig.update_layout(
        **CHART_LAYOUT, height=h,
        title=dict(
            text=(
                f"<b>IMSS y Aportaciones Federales absorben el {ia_pct:.1f}% del gasto — {year}</b>"
                f"<br><sup style='color:#94A3B8'>Top {n} dependencias por presupuesto propuesto"
                f" · billones de pesos corrientes</sup>"
            ),
        ),
        xaxis=dict(gridcolor="#334155", ticksuffix=" B"),
        yaxis=dict(gridcolor="rgba(0,0,0,0)", tickfont=dict(size=11)),
        margin=dict(t=80, b=40, l=270, r=20),
    )
    return fig


def fig_slope(d_det: pl.DataFrame, y0: int = 2012, y1: int = 2026) -> go.Figure:
    r0 = d_det.filter(pl.col("año") == y0).group_by("ramo").agg(pl.col("proyectado").sum())
    r1 = d_det.filter(pl.col("año") == y1).group_by("ramo").agg(pl.col("proyectado").sum())
    t0 = float(r0["proyectado"].sum())
    t1 = float(r1["proyectado"].sum())

    joined = (
        r0.rename({"proyectado": "v0"})
        .join(r1.rename({"proyectado": "v1"}), on="ramo", how="inner")
        .with_columns([
            (pl.col("v0") / t0 * 100).alias("s0"),
            (pl.col("v1") / t1 * 100).alias("s1"),
            (pl.col("v1") / t1 * 100 - pl.col("v0") / t0 * 100).alias("delta"),
        ])
        .sort("s1", descending=True)
        .head(20)
    )

    # Find the top ramo in y0 and its y1 rank for the title
    top_y0 = joined.sort("s0", descending=True)["ramo"][0]
    top_y0_short = _RAMO_SHORT.get(top_y0, top_y0[:15])
    r1_ranked = joined.sort("s1", descending=True).with_row_index("rank")
    top_y0_rank_rows = r1_ranked.filter(pl.col("ramo") == top_y0)
    top_y0_rank_y1 = int(top_y0_rank_rows["rank"][0]) + 1 if len(top_y0_rank_rows) > 0 else "?"

    n_up = int((joined["delta"] > 0).sum())
    n_dn = int((joined["delta"] <= 0).sum())

    fig = go.Figure()
    # Use numeric x positions (0 = y0, 1 = y1) — category axis misplaces multi-year string keys
    for row in joined.iter_rows(named=True):
        grew = row["delta"] > 0
        color = "#3BB273" if grew else "#E84855"
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[row["s0"], row["s1"]],
            mode="lines+markers",
            line=dict(color=color, width=1.5),
            marker=dict(color=color, size=7),
            showlegend=False,
            name=row["ramo"],
            hovertemplate=(
                f"<b>{row['ramo']}</b><br>"
                f"{y0}: {row['s0']:.1f}% del total<br>"
                f"{y1}: {row['s1']:.1f}% del total<br>"
                f"Δ: {row['delta']:+.1f} pp<extra></extra>"
            ),
        ))

    # Annotate top 8 ramos at the y1 end (x=1)
    for row in joined.head(8).iter_rows(named=True):
        label = _RAMO_SHORT.get(row["ramo"], row["ramo"][:16])
        fig.add_annotation(
            x=1.02, y=row["s1"],
            text=label,
            showarrow=False, xanchor="left",
            font=dict(color="#CBD5E1", size=10),
            xref="x", yref="y",
        )

    fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines+markers",
                              line=dict(color="#3BB273"), marker=dict(color="#3BB273"),
                              name=f"▲ Ganó participación ({n_up})"))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines+markers",
                              line=dict(color="#E84855"), marker=dict(color="#E84855"),
                              name=f"▼ Perdió participación ({n_dn})"))

    fig.update_layout(
        **CHART_LAYOUT, height=520,
        title=dict(
            text=(
                f"<b>{top_y0_short} cayó del 1° al {top_y0_rank_y1}° lugar;"
                f" IMSS y Aportaciones acaparan el crecimiento</b>"
                f"<br><sup style='color:#94A3B8'>Participación en el gasto programable"
                f" · top 20 ramos · {y0} → {y1}</sup>"
            ),
        ),
        xaxis=dict(
            tickvals=[0, 1],
            ticktext=[str(y0), str(y1)],
            range=[-0.15, 1.4],
            gridcolor="rgba(0,0,0,0)",
            showgrid=False,
            zeroline=False,
        ),
        yaxis=dict(gridcolor="#334155", ticksuffix="%"),
        legend=dict(orientation="h", y=-0.1, x=0),
        margin=dict(t=80, b=60, l=20, r=160),
    )
    return fig


def fig_ppef_vs_pef(d_det: pl.DataFrame) -> go.Figure:
    agg = (
        d_det.group_by("año")
        .agg([
            pl.col("ppef_anterior").sum().alias("ppef"),
            pl.col("pef_anterior").sum().alias("pef"),
        ])
        .with_columns((pl.col("año") - 1).alias("budget_year"))
        .sort("budget_year")
    )
    byr = agg["budget_year"].to_list()
    ppef_b = (agg["ppef"] / 1_000_000).to_list()
    pef_b = (agg["pef"] / 1_000_000).to_list()

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=byr, y=ppef_b, name="PPEF (Propuesto)",
        marker_color="#64748B",
        hovertemplate="<b>%{x}</b><br>PPEF: %{y:.2f} B<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        x=byr, y=pef_b, name="PEF (Aprobado)",
        marker_color=FOCUS,
        hovertemplate="<b>%{x}</b><br>PEF: %{y:.2f} B<extra></extra>",
    ))
    fig.add_vline(x=2015, line_dash="dot", line_color="#94A3B8",
                  annotation_text="2015: inicio<br>convergencia",
                  annotation_font_color="#94A3B8",
                  annotation_font_size=10,
                  annotation_position="top left")
    fig.update_layout(
        **CHART_LAYOUT, height=380,
        barmode="group",
        title=dict(
            text=(
                "<b>Desde 2015, el Congreso aprueba el presupuesto casi sin cambios</b>"
                "<br><sup style='color:#94A3B8'>PPEF (propuesto) vs PEF (aprobado)"
                " · gasto programable total · billones de pesos corrientes</sup>"
            ),
        ),
        xaxis=dict(gridcolor="#334155", tickmode="linear", dtick=1),
        yaxis=dict(gridcolor="#334155", ticksuffix=" B"),
        legend=dict(orientation="h", y=-0.18, x=0),
        margin=dict(t=80, b=70, l=20, r=20),
    )
    return fig


def fig_mod_pct(d_det: pl.DataFrame) -> go.Figure:
    agg = (
        d_det.group_by("año")
        .agg([
            pl.col("ppef_anterior").sum().alias("ppef"),
            pl.col("pef_anterior").sum().alias("pef"),
        ])
        .with_columns([
            (pl.col("año") - 1).alias("budget_year"),
            ((pl.col("pef") - pl.col("ppef")) / pl.col("ppef") * 100).alias("mod_pct"),
        ])
        .sort("budget_year")
    )
    byr = agg["budget_year"].to_list()
    mod = agg["mod_pct"].to_list()
    colors = [FOCUS if v >= 0 else "#E84855" for v in mod]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=byr, y=mod,
        marker_color=colors,
        hovertemplate="<b>%{x}</b><br>Modificación: %{y:+.2f}%<extra></extra>",
    ))
    fig.add_hline(y=0, line_color="#64748B", line_width=1)
    fig.add_vline(x=2015, line_dash="dot", line_color="#94A3B8",
                  annotation_text="2015",
                  annotation_font_color="#94A3B8",
                  annotation_font_size=10)
    fig.update_layout(
        **CHART_LAYOUT, height=340,
        title=dict(
            text=(
                "<b>El margen de modificación cayó de +2.5% a casi 0% desde 2015</b>"
                "<br><sup style='color:#94A3B8'>% de cambio PPEF→PEF"
                " · presupuesto agregado de gasto programable</sup>"
            ),
        ),
        xaxis=dict(gridcolor="#334155", tickmode="linear", dtick=1),
        yaxis=dict(gridcolor="#334155", ticksuffix="%"),
        margin=dict(t=80, b=40, l=20, r=20),
    )
    return fig


_LINE_PALETTE = [
    "#2E86AB", "#F4A261", "#3BB273", "#E84855", "#A855F7",
    "#06B6D4", "#F59E0B", "#10B981", "#EC4899", "#8B5CF6",
    "#94A3B8", "#EF4444", "#22D3EE", "#FB923C", "#4ADE80",
]


def fig_participacion_linea(ramos_sel: list) -> go.Figure:
    if not ramos_sel:
        fig = go.Figure()
        fig.update_layout(**CHART_LAYOUT, height=420,
                          xaxis=dict(gridcolor="#334155"),
                          yaxis=dict(gridcolor="#334155"))
        return fig

    d = share_by_ramo.filter(pl.col("ramo").is_in(ramos_sel))
    fig = go.Figure()
    for i, ramo in enumerate(ramos_sel):
        rd = d.filter(pl.col("ramo") == ramo).sort("año")
        if len(rd) == 0:
            continue
        color = _LINE_PALETTE[i % len(_LINE_PALETTE)]
        fig.add_trace(go.Scatter(
            x=rd["año"].to_list(),
            y=rd["share"].to_list(),
            mode="lines+markers",
            name=_RAMO_SHORT.get(ramo, ramo),
            line=dict(color=color, width=2),
            marker=dict(color=color, size=6),
            hovertemplate=f"<b>{ramo}</b><br>%{{x}}: %{{y:.2f}}% del gasto<extra></extra>",
        ))

    fig.add_vline(x=2019, line_dash="dot", line_color="#475569",
                  annotation_text="2019", annotation_font_color="#64748B",
                  annotation_font_size=9, annotation_position="top left")
    fig.add_vline(x=2025, line_dash="dot", line_color="#475569",
                  annotation_text="2025", annotation_font_color="#64748B",
                  annotation_font_size=9, annotation_position="top left")
    fig.update_layout(
        **CHART_LAYOUT, height=420,
        title=dict(
            text=(
                "<b>Participación en el gasto programable por dependencia</b>"
                "<br><sup style='color:#94A3B8'>% del total · clasificación administrativa"
                " · 2012–2026</sup>"
            ),
        ),
        xaxis=dict(gridcolor="#334155", tickmode="linear", dtick=1),
        yaxis=dict(gridcolor="#334155", ticksuffix="%"),
        legend=dict(orientation="h", y=-0.18, x=0),
        margin=dict(t=80, b=80, l=20, r=20),
    )
    return fig


# ── Layout helpers ────────────────────────────────────────────────────────────

def _panel(child) -> html.Div:
    return html.Div(child, style={
        "background": "#1E293B", "borderRadius": "8px",
        "padding": "16px", "marginBottom": "20px",
    })


def _kpi_cards() -> dbc.Row:
    yoy_color = "#3BB273" if _yoy_pct >= 0 else "#E84855"
    yoy_arrow = "▲" if _yoy_pct >= 0 else "▼"
    return dbc.Row([
        dbc.Col(html.Div([
            html.P("Gasto programable 2026",
                   style={"color": "#94A3B8", "fontSize": "13px", "margin": "0 0 6px"}),
            html.H2(f"{_val_2026 / 1_000_000:.2f}",
                    style={"color": "#F8FAFC", "margin": "0", "fontSize": "2.2rem"}),
            html.P("billones de pesos corrientes",
                   style={"color": "#64748B", "fontSize": "12px", "margin": "0 0 4px"}),
            html.P(f"{yoy_arrow} {abs(_yoy_pct):.1f}% vs 2025",
                   style={"color": yoy_color, "fontSize": "13px",
                          "fontWeight": "600", "margin": "0"}),
        ], style=CARD_STYLE), md=4),

        dbc.Col(html.Div([
            html.P("Crecimiento nominal 2012–2026",
                   style={"color": "#94A3B8", "fontSize": "13px", "margin": "0 0 6px"}),
            html.H2(f"×{_mult:.2f}",
                    style={"color": "#F8FAFC", "margin": "0", "fontSize": "2.2rem"}),
            html.P("pesos corrientes",
                   style={"color": "#64748B", "fontSize": "12px", "margin": "0 0 4px"}),
            html.P("≈ 6.7% anual sostenido",
                   style={"color": "#94A3B8", "fontSize": "13px",
                          "fontWeight": "600", "margin": "0"}),
        ], style=CARD_STYLE), md=4),

        dbc.Col(html.Div([
            html.P("IMSS + Aportaciones — 2026",
                   style={"color": "#94A3B8", "fontSize": "13px", "margin": "0 0 6px"}),
            html.H2(f"{_imss_aport_pct:.1f}%",
                    style={"color": "#F8FAFC", "margin": "0", "fontSize": "2.2rem"}),
            html.P("del gasto programable total",
                   style={"color": "#64748B", "fontSize": "12px", "margin": "0 0 4px"}),
            html.P("4 de cada 10 pesos del presupuesto",
                   style={"color": "#F4A261", "fontSize": "13px",
                          "fontWeight": "600", "margin": "0"}),
        ], style=CARD_STYLE), md=4),
    ], class_name="g-3", style={"marginBottom": "24px"})


# ── App ───────────────────────────────────────────────────────────────────────

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Gasto Programable Federal — México"

app.layout = html.Div(
    style={"backgroundColor": "#0F172A", "minHeight": "100vh",
           "padding": "24px", "fontFamily": "Inter, sans-serif"},
    children=[
        html.H1("Gasto Programable Federal — México",
                style={"color": "#F8FAFC", "marginBottom": "4px", "fontSize": "1.8rem"}),
        html.P(
            "Clasificación administrativa · PPEF · 2012–2026 · Fuente: CGPE / SHCP",
            style={"color": "#64748B", "marginBottom": "24px"},
        ),

        _kpi_cards(),

        dcc.Tabs(id="tabs", value="tendencia", style={"marginBottom": "16px"}, children=[
            dcc.Tab(label="Tendencia", value="tendencia",
                    style=TAB_STYLE, selected_style=TAB_SEL),
            dcc.Tab(label="Concentración", value="concentracion",
                    style=TAB_STYLE, selected_style=TAB_SEL),
            dcc.Tab(label="Propuesto vs Aprobado", value="ppef_pef",
                    style=TAB_STYLE, selected_style=TAB_SEL),
        ]),

        html.Div(id="tab-tendencia", children=[
            _panel(dcc.Graph(id="trend-total", config=GRAPH_CONFIG)),
            _panel(dcc.Graph(id="grupo-share", config=GRAPH_CONFIG)),
        ]),

        html.Div(id="tab-concentracion", children=[
            html.Div([
                html.Label("Año:", style={
                    "color": "#94A3B8", "marginRight": "12px", "fontWeight": "600",
                }),
                dcc.Dropdown(
                    id="ranking-year", value=2026,
                    options=[{"label": str(y), "value": y} for y in RANKING_YEARS],
                    clearable=False,
                    style={"width": "120px", "color": "#0F172A"},
                ),
            ], style={"display": "flex", "alignItems": "center", "marginBottom": "16px"}),
            _panel(dcc.Graph(id="ranking-bar", config=GRAPH_CONFIG)),
            _panel(dcc.Graph(id="slope-chart", config=GRAPH_CONFIG)),
            html.Div([
                html.Label("Dependencias:", style={
                    "color": "#94A3B8", "fontWeight": "600",
                    "marginBottom": "8px", "display": "block",
                }),
                dcc.Dropdown(
                    id="participacion-ramos",
                    options=RAMO_OPTIONS,
                    value=DEFAULT_RAMOS,
                    multi=True,
                    placeholder="Selecciona una o más dependencias…",
                    style={"color": "#0F172A"},
                ),
            ], style={"marginBottom": "16px"}),
            _panel(dcc.Graph(id="participacion-chart", config=GRAPH_CONFIG)),
        ]),

        html.Div(id="tab-ppef-pef", children=[
            _panel(dcc.Graph(id="ppef-pef-bar", config=GRAPH_CONFIG)),
            _panel(dcc.Graph(id="mod-pct-bar", config=GRAPH_CONFIG)),
        ]),
    ],
)


# ── Callbacks ─────────────────────────────────────────────────────────────────

@app.callback(
    Output("tab-tendencia", "style"),
    Output("tab-concentracion", "style"),
    Output("tab-ppef-pef", "style"),
    Input("tabs", "value"),
)
def show_tab(tab):
    show = {"display": "block"}
    hide = {"display": "none"}
    return (
        show if tab == "tendencia" else hide,
        show if tab == "concentracion" else hide,
        show if tab == "ppef_pef" else hide,
    )


@app.callback(
    Output("trend-total", "figure"),
    Output("grupo-share", "figure"),
    Input("tabs", "value"),
)
def update_tendencia(tab):
    if tab != "tendencia":
        return go.Figure(), go.Figure()
    return fig_trend_total(totals_series), fig_grupo_share(admin_det)


@app.callback(
    Output("ranking-bar", "figure"),
    Output("slope-chart", "figure"),
    Input("tabs", "value"),
    Input("ranking-year", "value"),
)
def update_concentracion(tab, year):
    if tab != "concentracion":
        return go.Figure(), go.Figure()
    return fig_ranking(admin_det, year), fig_slope(admin_det)


@app.callback(
    Output("ppef-pef-bar", "figure"),
    Output("mod-pct-bar", "figure"),
    Input("tabs", "value"),
)
def update_ppef_pef(tab):
    if tab != "ppef_pef":
        return go.Figure(), go.Figure()
    return fig_ppef_vs_pef(admin_det), fig_mod_pct(admin_det)


@app.callback(
    Output("participacion-chart", "figure"),
    Input("participacion-ramos", "value"),
    Input("tabs", "value"),
)
def update_participacion(ramos_sel, tab):
    if tab != "concentracion":
        return go.Figure()
    return fig_participacion_linea(ramos_sel or [])


if __name__ == "__main__":
    app.run(debug=True, port=8050)
