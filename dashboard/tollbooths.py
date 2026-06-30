#!/usr/bin/env python3
"""Casetas de cobro México 2021-2025 — crecimiento de tarifas vs inflación."""

import polars as pl
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc

# ── Style ──────────────────────────────────────────────────────────────────────

CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font_color="#CBD5E1",
)
CARD_STYLE = {
    "background": "#1E293B", "border": "1px solid #334155",
    "borderRadius": "8px", "padding": "16px", "textAlign": "center",
}
TAB_STYLE = {"backgroundColor": "#0F172A", "color": "#94A3B8", "borderTop": "none"}
TAB_SEL   = {"backgroundColor": "#1E293B", "color": "#F8FAFC",
              "borderTop": "2px solid #2E86AB", "fontWeight": "600"}

ABOVE = "#3BB273"    # beat inflation
BELOW = "#E84855"    # below inflation
CAPUFE_C = "#E84855"
CONC_C   = "#2E86AB"
NEUTRAL  = "#475569"

GATE_LABELS = {
    "urban_road":           "Vía urbana",
    "bypass":               "Libramiento",
    "international_bridge": "Puente internacional",
    "bridge":               "Puente",
    "tunnel":               "Túnel",
    "otro":                 "Otro / sin clasificar",
}

INFLATION_REF = 5.08   # CAGR 2021-2025 benchmark (constant across all rows)
ANNUAL_INF = {2022: 7.82, 2023: 4.66, 2024: 4.21, 2025: 3.69}

ROUND_COLS  = ["tdpa_round_2021", "tdpa_round_2022", "tdpa_round_2023",
               "tdpa_round_2024", "tdpa_round_2025"]
GROWTH_COLS = ["tdpa_growth_rate_2022", "tdpa_growth_rate_2023",
               "tdpa_growth_rate_2024", "tdpa_growth_rate_2025"]

# ── Data loading ───────────────────────────────────────────────────────────────

def load_data():
    df = pl.read_csv(
        "data/tollbooths/growth_rate_car_2021_2025.csv",
        encoding="latin1",
    )

    # Primary working set: rows with valid toll CAGR (1,529 rows after artifact drop)
    toll = (
        df.filter(
            pl.col("toll_cagr_growth_rate_2021_2025").is_not_null()
            & (pl.col("toll_cagr_growth_rate_2021_2025") > -50)
        )
        .with_columns([
            pl.when(pl.col("toll_inflation_diff") >= 0)
              .then(pl.lit("Por encima"))
              .otherwise(pl.lit("Por debajo"))
              .alias("beat_inflation"),
            pl.when(pl.col("tb_manage").str.to_lowercase().str.contains("capufe"))
              .then(pl.lit("CAPUFE"))
              .otherwise(pl.lit("Concesionario"))
              .alias("operador"),
        ])
    )

    # Scatter subset: rows with valid toll AND tdpa CAGR (719 rows)
    scatter = toll.filter(
        pl.col("tdpa_cagr_growth_rate_2021_2025").is_not_null()
        & pl.col("tdpa_cagr_growth_rate_2021_2025").is_not_nan()
        & (pl.col("tdpa_cagr_growth_rate_2021_2025") > -100)
    )

    # Traffic panel: rows with at least one tdpa_round value
    tdpa_df = (
        df.filter(pl.any_horizontal([pl.col(c).is_not_null() for c in ROUND_COLS]))
        .select(["road_name", "tollbooth_name", "state"] + ROUND_COLS + GROWTH_COLS)
    )

    return toll, scatter, tdpa_df


toll, scatter, tdpa_df = load_data()

# ── Pre-computed aggregates ────────────────────────────────────────────────────

# Operator stats (CAPUFE vs Concesionario)
op_stats = (
    toll.group_by("operador").agg([
        pl.col("toll_cagr_growth_rate_2021_2025").median().alias("median"),
        pl.col("toll_cagr_growth_rate_2021_2025").quantile(0.25).alias("p25"),
        pl.col("toll_cagr_growth_rate_2021_2025").quantile(0.75).alias("p75"),
        (pl.col("beat_inflation") == "Por encima").mean().alias("pct_beat"),
        pl.len().alias("n"),
    ])
    .sort("median")   # CAPUFE first (lower)
)

# State stats
state_stats = (
    toll.group_by("state").agg([
        pl.col("toll_cagr_growth_rate_2021_2025").median().alias("median_cagr"),
        pl.len().alias("n"),
        (pl.col("beat_inflation") == "Por encima").sum().alias("n_above"),
        (pl.col("beat_inflation") == "Por debajo").sum().alias("n_below"),
    ])
    .sort("median_cagr", descending=True)
)

# tb_manage bar (top 15 by count)
tb_stats = (
    toll.group_by("tb_manage").agg([
        pl.col("toll_cagr_growth_rate_2021_2025").median().alias("median_cagr"),
        pl.len().alias("n"),
        (pl.col("beat_inflation") == "Por encima").mean().alias("pct_beat"),
    ])
    .sort("n", descending=True)
    .head(15)
    .sort("median_cagr")
)

# Annual means
annual_means = {}
for yr in [2022, 2023, 2024, 2025]:
    col = f"toll_growth_rate_{yr}"
    valid = toll.filter(pl.col(col).is_not_null() & (pl.col(col) > -50))
    annual_means[yr] = float(valid[col].mean()) if len(valid) > 0 else None

# gate_to stats — infrastructure type vs toll CAGR (filter tunnel: n=3 too small)
gate_toll = (
    toll.filter(pl.col("gate_to").is_not_null())
    .group_by("gate_to").agg([
        pl.col("toll_cagr_growth_rate_2021_2025").median().alias("median_cagr"),
        pl.col("toll_cagr_growth_rate_2021_2025").quantile(0.25).alias("p25"),
        pl.col("toll_cagr_growth_rate_2021_2025").quantile(0.75).alias("p75"),
        pl.len().alias("n"),
        (pl.col("beat_inflation") == "Por encima").mean().alias("pct_beat"),
    ])
    .filter(pl.col("n") >= 10)
    .sort("median_cagr", descending=True)
    .with_columns(
        pl.col("gate_to").replace(GATE_LABELS).alias("gate_label")
    )
)

# fonadin stats — FONADIN=1 vs FONADIN=0
fonadin_toll = (
    toll.filter(pl.col("fonadin").is_not_null())
    .with_columns(pl.col("fonadin").cast(pl.Int8).alias("fonadin_int"))
    .group_by("fonadin_int").agg([
        pl.col("toll_cagr_growth_rate_2021_2025").median().alias("median"),
        pl.col("toll_cagr_growth_rate_2021_2025").quantile(0.25).alias("p25"),
        pl.col("toll_cagr_growth_rate_2021_2025").quantile(0.75).alias("p75"),
        pl.len().alias("n"),
        (pl.col("beat_inflation") == "Por encima").mean().alias("pct_beat"),
    ])
    .sort("fonadin_int")
    .with_columns(
        pl.when(pl.col("fonadin_int") == 1)
          .then(pl.lit("Con FONADIN"))
          .otherwise(pl.lit("Sin FONADIN"))
          .alias("label")
    )
)

# cost scatter: stretch_length_km vs km_cost, colored by gate_to; cap km_cost at 99th pct
_km_cost_p99 = float(toll["km_cost"].drop_nulls().quantile(0.99))
cost_scatter = (
    toll.filter(
        pl.col("stretch_length_km").is_not_null()
        & pl.col("km_cost").is_not_null()
        & (pl.col("km_cost") > 0)
        & (pl.col("km_cost") <= _km_cost_p99)
    )
    .with_columns(
        pl.col("gate_to").fill_null("otro")
          .replace(GATE_LABELS).alias("tipo")
    )
)

# bond year scatter: first date from bond_issuance_date vs toll CAGR
bond_scatter = (
    toll.filter(pl.col("bond_issuance_date").is_not_null())
    .with_columns(
        pl.col("bond_issuance_date")
          .str.split(",").list.first()
          .str.strip_chars()
          .str.slice(0, 4)
          .cast(pl.Int32, strict=False)
          .alias("bond_year")
    )
    .filter(pl.col("bond_year").is_not_null() & pl.col("bond_year").is_between(2000, 2025))
)

# KPI values
n_toll = len(toll)
med_cagr = float(toll["toll_cagr_growth_rate_2021_2025"].median())
pct_above = float((toll["beat_inflation"] == "Por encima").mean()) * 100
med_diff = float(toll["toll_inflation_diff"].median())
n_scatter = len(scatter)
pct_neg_tdpa = float((scatter["tdpa_cagr_growth_rate_2021_2025"] < 0).mean()) * 100
pct_both_pos = float(
    ((scatter["tdpa_cagr_growth_rate_2021_2025"] > 0) &
     (scatter["toll_cagr_growth_rate_2021_2025"] > 0)).mean()
) * 100

# Road dropdown options (alphabetical by display label)
road_options = sorted([
    {"label": r.replace("_", " ").title(), "value": r}
    for r in tdpa_df["road_name"].drop_nulls().unique().to_list()
], key=lambda x: x["label"])
# Default: road with median tollbooth count (good for demo)
_road_counts = (
    tdpa_df.group_by("road_name").len()
    .sort("len").filter(pl.col("road_name").is_not_null())
)
default_road = _road_counts[len(_road_counts) // 2]["road_name"][0]

# ── Helpers ────────────────────────────────────────────────────────────────────

def kpi(title, value, sub=""):
    return dbc.Col(html.Div([
        html.Div(str(value), style={"fontSize": "2rem", "fontWeight": "700", "color": "#F8FAFC"}),
        html.Div(title, style={"fontSize": "0.85rem", "color": "#94A3B8", "marginTop": "2px"}),
        html.Div(sub, style={"fontSize": "0.75rem", "color": "#64748B"}) if sub else None,
    ], style=CARD_STYLE), md=3)


# ── Figure factories ───────────────────────────────────────────────────────────

def fig_histogram() -> go.Figure:
    """Distribution of toll_inflation_diff (toll CAGR minus 5.08% benchmark)."""
    above = toll.filter(pl.col("beat_inflation") == "Por encima")["toll_inflation_diff"].to_list()
    below = toll.filter(pl.col("beat_inflation") == "Por debajo")["toll_inflation_diff"].to_list()
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=below, name="Por debajo de inflación", marker_color=BELOW, opacity=0.85,
        xbins=dict(size=0.5),
        hovertemplate="Diff %{x:.1f}pp<br>%{y} casetas<extra></extra>",
    ))
    fig.add_trace(go.Histogram(
        x=above, name="Por encima de inflación", marker_color=ABOVE, opacity=0.85,
        xbins=dict(size=0.5),
        hovertemplate="Diff %{x:.1f}pp<br>%{y} casetas<extra></extra>",
    ))
    fig.add_vline(x=0, line_dash="dash", line_color="#94A3B8",
                  annotation_text="Benchmark inflación 5.08%",
                  annotation_font_color="#94A3B8", annotation_position="top right")
    fig.update_layout(
        **CHART_LAYOUT,
        title="<b>Solo el 53.9% de las casetas supera la inflación — una moneda al aire</b>"
              "<br><sup style='color:#94A3B8'>Distribución de la diferencia CAGR tarifa − inflación 2021-2025 (pp)</sup>",
        barmode="overlay",
        xaxis=dict(gridcolor="#334155", title="CAGR tarifa − inflación (pp)", zeroline=False),
        yaxis=dict(gridcolor="#334155", title="Casetas"),
        legend=dict(orientation="h", y=-0.18, x=0),
        margin=dict(t=70, b=80, l=10, r=10),
        height=380,
    )
    return fig


def fig_annual_vs_inflation() -> go.Figure:
    """Bar chart: mean annual toll growth rate vs inflation per year (2022-2025)."""
    years = [2022, 2023, 2024, 2025]
    toll_vals = [annual_means[y] for y in years]
    inf_vals  = [ANNUAL_INF[y] for y in years]
    diffs     = [t - i for t, i in zip(toll_vals, inf_vals)]
    bar_colors = [ABOVE if d >= 0 else BELOW for d in diffs]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[str(y) for y in years], y=toll_vals,
        name="CAGR tarifa (media)",
        marker_color=bar_colors,
        hovertemplate="%{x}<br>Tarifa: %{y:.2f}%<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=[str(y) for y in years], y=inf_vals,
        mode="lines+markers", name="Inflación anual",
        line=dict(color="#94A3B8", width=2, dash="dot"),
        marker=dict(size=8, color="#94A3B8"),
        hovertemplate="%{x}<br>Inflación: %{y:.2f}%<extra></extra>",
    ))
    for i, (yr, tv, iv) in enumerate(zip(years, toll_vals, inf_vals)):
        d = tv - iv
        fig.add_annotation(
            x=str(yr), y=max(tv, iv) + 0.4,
            text=f"{d:+.1f}pp",
            font=dict(color=ABOVE if d >= 0 else BELOW, size=11),
            showarrow=False,
        )
    fig.update_layout(
        **CHART_LAYOUT,
        title="<b>Ajuste de tarifas desigual — rezago en 2022 y 2024, rebote en 2023</b>"
              "<br><sup style='color:#94A3B8'>Crecimiento medio anual de tarifa vs inflación por año</sup>",
        xaxis=dict(gridcolor="#334155", title="Año"),
        yaxis=dict(gridcolor="#334155", title="%", range=[0, 11]),
        legend=dict(orientation="h", y=-0.18, x=0),
        margin=dict(t=70, b=80, l=10, r=10),
        height=360,
    )
    return fig


def fig_beat_pct_bar() -> go.Figure:
    """Stacked 100% horizontal bar: % above/below inflation per operator."""
    rows = {r["operador"]: r for r in op_stats.iter_rows(named=True)}
    ops  = ["CAPUFE", "Concesionario"]
    fig  = go.Figure()
    for label, color in [("Por debajo", BELOW), ("Por encima", ABOVE)]:
        vals = []
        for op in ops:
            r = rows.get(op, {})
            pct = r.get("pct_beat", 0) * 100 if label == "Por encima" else (1 - r.get("pct_beat", 0)) * 100
            vals.append(round(pct, 1))
        fig.add_trace(go.Bar(
            y=ops, x=vals, orientation="h", name=label,
            marker_color=color,
            text=[f"{v:.1f}%" for v in vals], textposition="inside", insidetextanchor="middle",
            hovertemplate=f"<b>%{{y}}</b><br>{label}: %{{x:.1f}}%<extra></extra>",
        ))
    fig.update_layout(
        **CHART_LAYOUT,
        title="<b>CAPUFE: 30.5% supera inflación · Concesionarios: 59.7%</b>"
              "<br><sup style='color:#94A3B8'>% de casetas por encima / debajo del benchmark 5.08%</sup>",
        barmode="stack",
        xaxis=dict(range=[0, 100], visible=False),
        yaxis=dict(gridcolor="rgba(0,0,0,0)"),
        legend=dict(orientation="h", y=-0.3, x=0),
        margin=dict(t=60, b=60, l=110, r=10),
        height=220,
    )
    return fig


def fig_dot_range() -> go.Figure:
    """Dot-and-range: median + IQR per operator for toll CAGR."""
    rows   = sorted(op_stats.iter_rows(named=True), key=lambda r: r["median"])
    ops    = [r["operador"] for r in rows]
    meds   = [r["median"]   for r in rows]
    p25s   = [r["p25"]      for r in rows]
    p75s   = [r["p75"]      for r in rows]
    colors = [CAPUFE_C if op == "CAPUFE" else CONC_C for op in ops]

    x_lines, y_lines = [], []
    for lo, hi, op in zip(p25s, p75s, ops):
        x_lines += [lo, hi, None]
        y_lines += [op, op, None]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_lines, y=y_lines, mode="lines",
        line=dict(color="#334155", width=3), showlegend=False, hoverinfo="skip",
    ))
    # IQR end ticks
    for x_vals, sym in [(p25s, "line-ew"), (p75s, "line-ew")]:
        fig.add_trace(go.Scatter(
            x=x_vals, y=ops, mode="markers", showlegend=False,
            marker=dict(symbol=sym, size=10, color="#475569",
                        line=dict(width=2, color="#475569")),
            hoverinfo="skip",
        ))
    fig.add_trace(go.Scatter(
        x=meds, y=ops, mode="markers", name="Mediana",
        marker=dict(color=colors, size=14, line=dict(color="#0F172A", width=1)),
        customdata=[r["n"] for r in rows],
        hovertemplate="<b>%{y}</b><br>Mediana: %{x:.2f}%<br>n=%{customdata:,}<extra></extra>",
    ))
    fig.add_vline(x=INFLATION_REF, line_dash="dash", line_color="#94A3B8",
                  annotation_text=f"Inflación {INFLATION_REF}%",
                  annotation_font_color="#94A3B8", annotation_position="top right")
    fig.update_layout(
        **CHART_LAYOUT,
        title="<b>Brecha de 1.35pp entre concesionarios y CAPUFE — estadísticamente sólida</b>"
              "<br><sup style='color:#94A3B8'>Mediana e IQR del CAGR de tarifa 2021-2025 por tipo de operador</sup>",
        xaxis=dict(gridcolor="#334155", title="CAGR tarifa 2021-2025 (%)"),
        yaxis=dict(gridcolor="rgba(0,0,0,0)"),
        margin=dict(t=70, b=50, l=120, r=10),
        height=200,
        showlegend=False,
    )
    return fig


def fig_operator_bar() -> go.Figure:
    """Horizontal bar: top-15 tb_manage groups by median toll CAGR."""
    rows   = tb_stats.iter_rows(named=True)
    groups = tb_stats["tb_manage"].to_list()
    meds   = tb_stats["median_cagr"].to_list()
    counts = tb_stats["n"].to_list()
    colors = [CAPUFE_C if g and "capufe" in g.lower() else
              (ABOVE if m >= INFLATION_REF else NEUTRAL)
              for g, m in zip(groups, meds)]

    fig = go.Figure(go.Bar(
        y=groups, x=meds, orientation="h",
        marker_color=colors,
        customdata=counts,
        hovertemplate="<b>%{y}</b><br>Mediana CAGR: %{x:.2f}%<br>n=%{customdata:,}<extra></extra>",
    ))
    fig.add_vline(x=INFLATION_REF, line_dash="dash", line_color="#94A3B8",
                  annotation_text=f"Inflación {INFLATION_REF}%",
                  annotation_font_color="#94A3B8")
    fig.update_layout(
        **CHART_LAYOUT,
        title="<b>CAPUFE y arconorte son los operadores con mayor rezago tarifario</b>"
              "<br><sup style='color:#94A3B8'>Top 15 administradores por número de casetas · mediana CAGR 2021-2025</sup>",
        xaxis=dict(gridcolor="#334155", title="Mediana CAGR (%)"),
        yaxis=dict(gridcolor="rgba(0,0,0,0)"),
        margin=dict(t=70, b=50, l=175, r=10),
        height=max(320, 15 * 28 + 80),
    )
    return fig


def fig_state_bar() -> go.Figure:
    """Horizontal bar: states sorted by median toll CAGR, colored above/below inflation."""
    states = state_stats["state"].to_list()
    meds   = state_stats["median_cagr"].to_list()
    counts = state_stats["n"].to_list()
    colors = [ABOVE if m >= INFLATION_REF else BELOW for m in meds]

    fig = go.Figure(go.Bar(
        y=states, x=meds, orientation="h",
        marker_color=colors,
        customdata=counts,
        hovertemplate="<b>%{y}</b><br>Mediana CAGR: %{x:.2f}%<br>n=%{customdata:,}<extra></extra>",
    ))
    fig.add_vline(x=INFLATION_REF, line_dash="dash", line_color="#94A3B8",
                  annotation_text=f"Inflación {INFLATION_REF}%",
                  annotation_font_color="#94A3B8")
    n_states = len(states)
    fig.update_layout(
        **CHART_LAYOUT,
        title="<b>CDMX lidera el crecimiento tarifario (96% supera inflación) — Edomx es la mitad del universo</b>"
              "<br><sup style='color:#94A3B8'>Mediana CAGR de tarifa 2021-2025 por estado · verde = supera inflación</sup>",
        xaxis=dict(gridcolor="#334155", title="Mediana CAGR (%)"),
        yaxis=dict(gridcolor="rgba(0,0,0,0)"),
        margin=dict(t=70, b=50, l=140, r=10),
        height=max(420, n_states * 22 + 80),
    )
    return fig


def fig_state_count() -> go.Figure:
    """Stacked horizontal bar: count above/below inflation per state (same sort as fig_state_bar)."""
    states  = state_stats["state"].to_list()
    n_above = state_stats["n_above"].to_list()
    n_below = state_stats["n_below"].to_list()

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=states, x=n_below, orientation="h", name="Por debajo",
        marker_color=BELOW,
        hovertemplate="<b>%{y}</b><br>Por debajo: %{x}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        y=states, x=n_above, orientation="h", name="Por encima",
        marker_color=ABOVE,
        hovertemplate="<b>%{y}</b><br>Por encima: %{x}<extra></extra>",
    ))
    n_states = len(states)
    fig.update_layout(
        **CHART_LAYOUT,
        title="<b>Volumen de casetas por estado — Edomx concentra 545 de 1,529</b>"
              "<br><sup style='color:#94A3B8'>Casetas por encima / debajo del benchmark de inflación</sup>",
        barmode="stack",
        xaxis=dict(gridcolor="#334155", title="Número de casetas"),
        yaxis=dict(gridcolor="rgba(0,0,0,0)"),
        legend=dict(orientation="h", y=-0.06, x=0),
        margin=dict(t=70, b=60, l=140, r=10),
        height=max(420, n_states * 22 + 80),
    )
    return fig


def fig_scatter() -> go.Figure:
    """Scatter: tdpa_cagr (traffic) vs toll_cagr, colored by operator."""
    op_map    = {"CAPUFE": CAPUFE_C, "Concesionario": CONC_C}
    label_col = scatter["tollbooth_name"].fill_null("—").to_list()
    road_col  = scatter["road_name"].fill_null("—").to_list()

    fig = go.Figure()
    for op, color in op_map.items():
        sub = scatter.filter(pl.col("operador") == op)
        fig.add_trace(go.Scatter(
            x=sub["tdpa_cagr_growth_rate_2021_2025"].to_list(),
            y=sub["toll_cagr_growth_rate_2021_2025"].to_list(),
            mode="markers", name=op,
            marker=dict(color=color, size=5, opacity=0.55,
                        line=dict(color="#0F172A", width=0.3)),
            customdata=list(zip(
                sub["tollbooth_name"].fill_null("—").to_list(),
                sub["road_name"].fill_null("—").to_list(),
                sub["state"].fill_null("—").to_list(),
            )),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "%{customdata[1]} · %{customdata[2]}<br>"
                "Tráfico CAGR: %{x:.2f}%<br>"
                "Tarifa CAGR: %{y:.2f}%<extra></extra>"
            ),
        ))
    fig.add_vline(x=0, line_dash="dot", line_color="#475569")
    fig.add_hline(y=0, line_dash="dot", line_color="#475569")
    fig.add_hline(y=INFLATION_REF, line_dash="dash", line_color="#94A3B8",
                  annotation_text=f"Inflación {INFLATION_REF}%",
                  annotation_font_color="#94A3B8", annotation_position="right")
    # Quadrant labels
    for txt, x, y in [
        ("Tráfico ↑ · Tarifa ↑",  30,  30),
        ("Tráfico ↓ · Tarifa ↑", -40,  30),
        ("Tráfico ↑ · Tarifa ↓",  30, -12),
        ("Tráfico ↓ · Tarifa ↓", -40, -12),
    ]:
        fig.add_annotation(
            x=x, y=y, text=txt,
            font=dict(color="#475569", size=10),
            showarrow=False, bgcolor="rgba(15,23,42,0.6)",
        )
    fig.update_layout(
        **CHART_LAYOUT,
        title="<b>Tráfico y tarifa están desacoplados — Spearman ρ = −0.02 (p = 0.65)</b>"
              "<br><sup style='color:#94A3B8'>CAGR de tráfico diario vs CAGR de tarifa · 719 casetas con ambos datos</sup>",
        xaxis=dict(gridcolor="#334155", title="CAGR tráfico diario TDPA 2021-2025 (%)"),
        yaxis=dict(gridcolor="#334155", title="CAGR tarifa 2021-2025 (%)"),
        legend=dict(orientation="h", y=-0.12, x=0),
        margin=dict(t=70, b=80, l=10, r=10),
        height=480,
    )
    return fig


def fig_tdpa_round(sub: pl.DataFrame, road: str) -> go.Figure:
    """Line chart: tdpa_round 2021-2025 per tollbooth for a given road."""
    long = (
        sub.unpivot(on=ROUND_COLS, index=["tollbooth_name"],
                    variable_name="year_col", value_name="tdpa_round")
        .filter(pl.col("tdpa_round").is_not_null())
        .with_columns(pl.col("year_col").str.slice(-4).cast(pl.Int32).alias("year"))
        .sort("year")
    )
    n_booths = sub["tollbooth_name"].n_unique()
    opacity  = max(0.2, 1.0 - (n_booths - 5) * 0.025) if n_booths > 5 else 0.9
    show_leg = n_booths <= 15

    fig = go.Figure()
    for name in sub["tollbooth_name"].to_list():
        row = long.filter(pl.col("tollbooth_name") == name)
        if len(row) == 0:
            continue
        fig.add_trace(go.Scatter(
            x=row["year"].to_list(), y=row["tdpa_round"].to_list(),
            mode="lines+markers", name=name, opacity=opacity,
            line=dict(width=1.5), showlegend=show_leg,
            hovertemplate=f"<b>{name}</b><br>%{{x}}: %{{y:,.0f}} veh/día<extra></extra>",
        ))
    if n_booths > 15:
        med = long.group_by("year").agg(pl.col("tdpa_round").median().alias("m")).sort("year")
        fig.add_trace(go.Scatter(
            x=med["year"].to_list(), y=med["m"].to_list(),
            mode="lines+markers", name="Mediana",
            line=dict(color="#F8FAFC", width=3),
            marker=dict(size=9, color="#F8FAFC"),
            hovertemplate="Mediana %{x}: %{y:,.0f} veh/día<extra></extra>",
        ))
    road_label = road.replace("_", " ").title()
    fig.update_layout(
        **CHART_LAYOUT,
        title=f"<b>TDPA · {road_label}</b>"
              f"<br><sup style='color:#94A3B8'>Tráfico diario promedio anual por caseta 2021-2025 · n={n_booths} casetas</sup>",
        xaxis=dict(gridcolor="#334155", title="Año", dtick=1),
        yaxis=dict(gridcolor="#334155", title="Vehículos/día"),
        legend=dict(orientation="h", y=-0.15, x=0) if show_leg else {},
        margin=dict(t=70, b=70 if show_leg else 40, l=10, r=10),
        height=380,
    )
    return fig


def fig_tdpa_growth(sub: pl.DataFrame, road: str) -> go.Figure:
    """Heatmap: tdpa_growth_rate 2022-2025 per tollbooth, Y=tollbooth, X=year."""
    years  = [2022, 2023, 2024, 2025]
    booths = sub["tollbooth_name"].to_list()
    n_booths = len(booths)

    z = []
    for yr in years:
        col = f"tdpa_growth_rate_{yr}"
        vals = sub[col].to_list()
        # Cap extreme outliers for colorscale stability
        z.append([min(max(v, -100), 100) if v is not None else None for v in vals])

    show_xlabels = n_booths <= 40
    road_label = road.replace("_", " ").title()
    fig = go.Figure(go.Heatmap(
        x=booths,
        y=[str(yr) for yr in years],
        z=z,
        colorscale=[[0.0, "#E84855"], [0.5, "#1E293B"], [1.0, "#3BB273"]],
        zmid=0, zmin=-100, zmax=100,
        colorbar=dict(title="%", ticksuffix="%", len=0.7),
        hovertemplate="<b>%{x}</b><br>%{y}: %{z:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        **CHART_LAYOUT,
        title=f"<b>Crecimiento de tráfico · {road_label}</b>"
              f"<br><sup style='color:#94A3B8'>Tasa de crecimiento TDPA anual · verde = crecimiento · valores >±100% recortados</sup>",
        xaxis=dict(showticklabels=show_xlabels,
                   tickangle=-45 if n_booths > 8 else 0,
                   gridcolor="rgba(0,0,0,0)"),
        yaxis=dict(gridcolor="rgba(0,0,0,0)"),
        margin=dict(t=70, b=100 if show_xlabels and n_booths > 5 else 40, l=60, r=10),
        height=300,
    )
    return fig


def fig_gate_type_bar() -> go.Figure:
    """Horizontal bar: median toll CAGR by infrastructure type (gate_to), n≥10."""
    labels   = gate_toll["gate_label"].to_list()
    meds     = gate_toll["median_cagr"].to_list()
    ns       = gate_toll["n"].to_list()
    colors   = [ABOVE if m >= INFLATION_REF else BELOW for m in meds]

    fig = go.Figure(go.Bar(
        y=labels, x=meds, orientation="h",
        marker_color=colors,
        customdata=ns,
        text=[f"n={n:,}" for n in ns],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Mediana CAGR: %{x:.2f}%<br>n=%{customdata:,}<extra></extra>",
    ))
    fig.add_vline(x=INFLATION_REF, line_dash="dash", line_color="#94A3B8",
                  annotation_text=f"Inflación {INFLATION_REF}%",
                  annotation_font_color="#94A3B8", annotation_position="top right")
    fig.update_layout(
        **CHART_LAYOUT,
        title="<b>Puentes internacionales crecen a 1.6% — regulación bilateral frena la tarifa a ¼ de las vías urbanas</b>"
              "<br><sup style='color:#94A3B8'>Mediana CAGR de tarifa 2021-2025 por tipo de infraestructura</sup>",
        xaxis=dict(gridcolor="#334155", title="Mediana CAGR (%)", range=[0, max(meds) * 1.35]),
        yaxis=dict(gridcolor="rgba(0,0,0,0)"),
        margin=dict(t=80, b=40, l=145, r=60),
        height=280,
        showlegend=False,
    )
    return fig


def fig_fonadin_dot_range() -> go.Figure:
    """Dot-and-range: median + IQR for Con FONADIN vs Sin FONADIN."""
    rows   = fonadin_toll.iter_rows(named=True)
    data   = sorted(rows, key=lambda r: r["median"])
    labels = [r["label"]  for r in data]
    meds   = [r["median"] for r in data]
    p25s   = [r["p25"]    for r in data]
    p75s   = [r["p75"]    for r in data]
    ns     = [r["n"]      for r in data]
    colors = [ABOVE if m >= INFLATION_REF else BELOW for m in meds]

    x_lines, y_lines = [], []
    for lo, hi, lbl in zip(p25s, p75s, labels):
        x_lines += [lo, hi, None]
        y_lines += [lbl, lbl, None]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_lines, y=y_lines, mode="lines",
        line=dict(color="#334155", width=3), showlegend=False, hoverinfo="skip",
    ))
    for x_vals in [p25s, p75s]:
        fig.add_trace(go.Scatter(
            x=x_vals, y=labels, mode="markers", showlegend=False,
            marker=dict(symbol="line-ew", size=10, color="#475569",
                        line=dict(width=2, color="#475569")),
            hoverinfo="skip",
        ))
    fig.add_trace(go.Scatter(
        x=meds, y=labels, mode="markers",
        marker=dict(color=colors, size=14, line=dict(color="#0F172A", width=1)),
        customdata=ns,
        hovertemplate="<b>%{y}</b><br>Mediana: %{x:.2f}%<br>n=%{customdata:,}<extra></extra>",
        showlegend=False,
    ))
    # Count annotations
    for lbl, n in zip(labels, ns):
        fig.add_annotation(
            x=max(p75s) + 0.4, y=lbl, text=f"n={n:,}",
            font=dict(color="#64748B", size=11), showarrow=False, xanchor="left",
        )
    fig.add_vline(x=INFLATION_REF, line_dash="dash", line_color="#94A3B8",
                  annotation_text=f"Inflación {INFLATION_REF}%",
                  annotation_font_color="#94A3B8", annotation_position="top right")
    fig.update_layout(
        **CHART_LAYOUT,
        title="<b>Casetas con FONADIN crecen +0.55pp más — el respaldo público amplía el margen tarifario</b>"
              "<br><sup style='color:#94A3B8'>Mediana e IQR del CAGR de tarifa 2021-2025 · Con vs Sin FONADIN</sup>",
        xaxis=dict(gridcolor="#334155", title="CAGR tarifa 2021-2025 (%)"),
        yaxis=dict(gridcolor="rgba(0,0,0,0)"),
        margin=dict(t=80, b=40, l=120, r=80),
        height=200,
    )
    return fig


def fig_cost_scatter() -> go.Figure:
    """Scatter: stretch_length_km vs km_cost, colored by gate type."""
    tipo_colors = {
        "Vía urbana":           "#2E86AB",
        "Libramiento":          "#3BB273",
        "Puente internacional": "#F4A261",
        "Puente":               "#FFD700",
        "Otro / sin clasificar":"#475569",
    }
    fig = go.Figure()
    for tipo, color in tipo_colors.items():
        sub = cost_scatter.filter(pl.col("tipo") == tipo)
        if len(sub) == 0:
            continue
        fig.add_trace(go.Scatter(
            x=sub["stretch_length_km"].to_list(),
            y=sub["km_cost"].to_list(),
            mode="markers", name=tipo,
            marker=dict(color=color, size=5, opacity=0.6,
                        line=dict(color="#0F172A", width=0.3)),
            customdata=list(zip(
                sub["tollbooth_name"].fill_null("—").to_list(),
                sub["road_name"].fill_null("—").to_list(),
                sub["state"].fill_null("—").to_list(),
            )),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "%{customdata[1]} · %{customdata[2]}<br>"
                "Longitud: %{x:.1f} km<br>"
                "Costo/km: $%{y:,.1f}M<extra></extra>"
            ),
        ))
    fig.update_layout(
        **CHART_LAYOUT,
        title="<b>Carreteras urbanas cortas concentran el mayor costo/km — infraestructura especializada en ciudad</b>"
              f"<br><sup style='color:#94A3B8'>Longitud vs costo por km · valores atípicos >p99 (${_km_cost_p99:,.0f}M/km) excluidos</sup>",
        xaxis=dict(gridcolor="#334155", title="Longitud tramo (km)"),
        yaxis=dict(gridcolor="#334155", title="Costo por km ($M)"),
        legend=dict(orientation="h", y=-0.15, x=0),
        margin=dict(t=80, b=90, l=10, r=10),
        height=420,
    )
    return fig


def fig_bond_vintage_scatter() -> go.Figure:
    """Scatter: bond issuance year vs toll CAGR, colored by operador."""
    op_map = {"CAPUFE": CAPUFE_C, "Concesionario": CONC_C}
    n_bond = len(bond_scatter)
    fig = go.Figure()
    for op, color in op_map.items():
        sub = bond_scatter.filter(pl.col("operador") == op)
        fig.add_trace(go.Scatter(
            x=sub["bond_year"].to_list(),
            y=sub["toll_cagr_growth_rate_2021_2025"].to_list(),
            mode="markers", name=op,
            marker=dict(color=color, size=6, opacity=0.6,
                        line=dict(color="#0F172A", width=0.3)),
            customdata=list(zip(
                sub["tollbooth_name"].fill_null("—").to_list(),
                sub["state"].fill_null("—").to_list(),
            )),
            hovertemplate=(
                "<b>%{customdata[0]}</b> · %{customdata[1]}<br>"
                "Año bono: %{x}<br>"
                "Tarifa CAGR: %{y:.2f}%<extra></extra>"
            ),
        ))
    fig.add_hline(y=INFLATION_REF, line_dash="dash", line_color="#94A3B8",
                  annotation_text=f"Inflación {INFLATION_REF}%",
                  annotation_font_color="#94A3B8", annotation_position="right")
    fig.update_layout(
        **CHART_LAYOUT,
        title="<b>La antigüedad del bono no predice la dinámica tarifaria — sin patrón por vintage</b>"
              f"<br><sup style='color:#94A3B8'>Año de emisión del bono vs CAGR tarifa · {n_bond:,} casetas con fecha de bono parseable</sup>",
        xaxis=dict(gridcolor="#334155", title="Año emisión bono"),
        yaxis=dict(gridcolor="#334155", title="CAGR tarifa 2021-2025 (%)"),
        legend=dict(orientation="h", y=-0.12, x=0),
        margin=dict(t=80, b=80, l=10, r=10),
        height=380,
    )
    return fig


# ── App layout ─────────────────────────────────────────────────────────────────

app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],
    title="Casetas de Cobro México",
)

app.layout = html.Div(
    style={"backgroundColor": "#0F172A", "minHeight": "100vh", "padding": "24px"},
    children=[
        html.H1("Casetas de Cobro México 2021–2025",
                style={"color": "#F8FAFC", "fontWeight": "700", "marginBottom": "4px"}),
        html.P("Crecimiento de tarifas, tráfico e ingresos vs inflación · SCT / CAPUFE / Concesionarios",
               style={"color": "#94A3B8", "marginBottom": "24px"}),

        dcc.Tabs(style={"marginBottom": "16px"}, children=[

            # ── Tab 1: Panorama ───────────────────────────────────────────────
            dcc.Tab(label="Panorama", style=TAB_STYLE, selected_style=TAB_SEL, children=[
                dbc.Row([
                    kpi("Casetas con datos", f"{n_toll:,}"),
                    kpi("Mediana CAGR tarifa", f"{med_cagr:.2f}%", "2021-2025"),
                    kpi("Superan inflación (5.08%)", f"{pct_above:.1f}%", "de las casetas"),
                    kpi("Margen real mediano", f"{med_diff:+.2f} pp", "CAGR tarifa − inflación"),
                ], className="g-2 mb-4"),
                dbc.Row([
                    dbc.Col(dcc.Graph(figure=fig_histogram(),
                                     config={"displayModeBar": False}), md=7),
                    dbc.Col(dcc.Graph(figure=fig_annual_vs_inflation(),
                                     config={"displayModeBar": False}), md=5),
                ], className="mt-2"),
                dbc.Row([
                    dbc.Col(dcc.Graph(figure=fig_beat_pct_bar(),
                                     config={"displayModeBar": False}), md=12),
                ], className="mt-2"),
            ]),

            # ── Tab 2: Operador ───────────────────────────────────────────────
            dcc.Tab(label="Operador", style=TAB_STYLE, selected_style=TAB_SEL, children=[
                dbc.Row([
                    dbc.Col(dcc.Graph(figure=fig_dot_range(),
                                     config={"displayModeBar": False}), md=12),
                ], className="mt-3"),
                dbc.Row([
                    dbc.Col(dcc.Graph(figure=fig_operator_bar(),
                                     config={"displayModeBar": False}), md=12),
                ], className="mt-2"),
            ]),

            # ── Tab 3: Por Estado ─────────────────────────────────────────────
            dcc.Tab(label="Por Estado", style=TAB_STYLE, selected_style=TAB_SEL, children=[
                dbc.Row([
                    dbc.Col(dcc.Graph(figure=fig_state_bar(),
                                     config={"displayModeBar": False}), md=7),
                    dbc.Col(dcc.Graph(figure=fig_state_count(),
                                     config={"displayModeBar": False}), md=5),
                ], className="mt-3"),
            ]),

            # ── Tab 4: Tráfico vs Tarifa ──────────────────────────────────────
            dcc.Tab(label="Tráfico vs Tarifa", style=TAB_STYLE, selected_style=TAB_SEL, children=[
                dbc.Row([
                    kpi("Casetas con datos de tráfico", f"{n_scatter:,}"),
                    kpi("Con tráfico decreciente", f"{pct_neg_tdpa:.1f}%"),
                    kpi("Correlación ρ (Spearman)", "−0.02", "p = 0.65 · no significativa"),
                    kpi("Con ambos positivos", f"{pct_both_pos:.1f}%"),
                ], className="g-2 mb-3"),
                dbc.Row([
                    dbc.Col(dcc.Graph(figure=fig_scatter(),
                                     config={"displayModeBar": False}), md=12),
                ], className="mt-2"),
            ]),

            # ── Tab 5: Costos e Infraestructura ──────────────────────────────
            dcc.Tab(label="Costos e Infraestructura", style=TAB_STYLE, selected_style=TAB_SEL, children=[
                dbc.Row([
                    dbc.Col(dcc.Graph(figure=fig_gate_type_bar(),
                                     config={"displayModeBar": False}), md=5),
                    dbc.Col(dcc.Graph(figure=fig_fonadin_dot_range(),
                                     config={"displayModeBar": False}), md=7),
                ], className="mt-3"),
                dbc.Row([
                    dbc.Col(dcc.Graph(figure=fig_cost_scatter(),
                                     config={"displayModeBar": False}), md=6),
                    dbc.Col(dcc.Graph(figure=fig_bond_vintage_scatter(),
                                     config={"displayModeBar": False}), md=6),
                ], className="mt-2"),
            ]),

            # ── Tab 6: Tráfico por Tramo ──────────────────────────────────────
            dcc.Tab(label="Tráfico por Tramo", style=TAB_STYLE, selected_style=TAB_SEL, children=[
                dbc.Row([
                    dbc.Col([
                        html.Label("Tramo / carretera:",
                                   style={"color": "#94A3B8", "marginBottom": "6px",
                                          "display": "block"}),
                        dcc.Dropdown(
                            id="road-dropdown",
                            options=road_options,
                            value=default_road,
                            clearable=False,
                            style={"color": "#0F172A"},
                        ),
                    ], md=6),
                ], className="mt-3 mb-3"),
                dbc.Row([
                    dbc.Col(dcc.Graph(id="tdpa-round-fig",
                                     config={"displayModeBar": False}), md=12),
                ]),
                dbc.Row([
                    dbc.Col(dcc.Graph(id="tdpa-growth-fig",
                                     config={"displayModeBar": False}), md=12),
                ], className="mt-2"),
            ]),
        ]),
    ],
)


@app.callback(
    Output("tdpa-round-fig", "figure"),
    Output("tdpa-growth-fig", "figure"),
    Input("road-dropdown", "value"),
)
def update_tdpa(road: str):
    sub = tdpa_df.filter(pl.col("road_name") == road)
    return fig_tdpa_round(sub, road), fig_tdpa_growth(sub, road)


if __name__ == "__main__":
    app.run(debug=True, port=8061)
