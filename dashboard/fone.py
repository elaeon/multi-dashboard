import glob
import json
import math
import polars as pl
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc

# ── Data loading ──────────────────────────────────────────────────────────────

_paths = sorted(glob.glob("data/fone/*/PlazasDocAdmtvasDirec_*.parquet"))

# Pass 1: global min/max of positive per-CURP annual totals — defines log-spaced bins.
_stats = (
    pl.scan_parquet(_paths)
    .group_by(["CURP", "ENTIDAD_FEDERATIVA", "YEAR"])
    .agg(pl.col("PERCEPCIONES_TRIMESTRALES").sum())
    .filter(pl.col("PERCEPCIONES_TRIMESTRALES") > 0)
    .select([
        pl.col("PERCEPCIONES_TRIMESTRALES").min().alias("pmin"),
        pl.col("PERCEPCIONES_TRIMESTRALES").max().alias("pmax"),
    ])
    .collect(engine="streaming")
)
_pmin = float(_stats["pmin"][0])
_pmax = float(_stats["pmax"][0])

_N_BINS    = 20
_LOG_MIN   = math.log10(_pmin)
_LOG_MAX   = math.log10(_pmax)
_LOG_RANGE = _LOG_MAX - _LOG_MIN
_EDGES     = [_pmin * (_pmax / _pmin) ** (i / _N_BINS) for i in range(_N_BINS + 1)]
_MIDPOINTS = [math.sqrt(_EDGES[i] * _EDGES[i + 1]) for i in range(_N_BINS)]



def _bin_lf(lf: pl.LazyFrame, keys: list) -> pl.LazyFrame:
    """Sum PERCEPCIONES per CURP, assign log-spaced bin index, count per keys×bin."""
    return (
        lf.group_by(["CURP"] + keys)
        .agg(pl.col("PERCEPCIONES_TRIMESTRALES").sum())
        .filter(pl.col("PERCEPCIONES_TRIMESTRALES") > 0)
        .with_columns(
            ((pl.col("PERCEPCIONES_TRIMESTRALES").log(10) - _LOG_MIN) / _LOG_RANGE * _N_BINS)
            .floor().cast(pl.Int32).clip(0, _N_BINS - 1)
            .alias("bin_idx")
        )
        .group_by(keys + ["bin_idx"])
        .agg(pl.len().alias("count"))
    )


_lf = pl.scan_parquet(_paths)

_df_raw = _bin_lf(_lf, ["ENTIDAD_FEDERATIVA", "YEAR", "TIPO_PLAZA"]).collect(engine="streaming")

# Per-CURP annual totals — base for percentile charts
_df_curp_totals = (
    _lf.group_by(["CURP", "ENTIDAD_FEDERATIVA", "YEAR", "TIPO_PLAZA"])
    .agg(pl.col("PERCEPCIONES_TRIMESTRALES").sum())
    .filter(pl.col("PERCEPCIONES_TRIMESTRALES") > 0)
    .collect(engine="streaming")
)

YEARS  = sorted(_df_raw["YEAR"].cast(pl.Int32).unique().to_list())
TIPOS  = ["Todos"] + sorted(_df_raw["TIPO_PLAZA"].unique().to_list())
STATES = sorted(_df_raw["ENTIDAD_FEDERATIVA"].unique().to_list())

with open("data/mexico_states.geojson") as _f:
    _MEXICO_GEOJSON = json.load(_f)

# ── Theme ─────────────────────────────────────────────────────────────────────

CARD_STYLE = {
    "background": "#1E293B", "border": "1px solid #334155",
    "borderRadius": "8px", "padding": "12px",
}

_DARK  = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#CBD5E1")
_XAXIS = dict(tickfont=dict(size=9, color="#94A3B8"), gridcolor="#334155", showgrid=True, zeroline=False)
_YAXIS = dict(tickfont=dict(size=9, color="#CBD5E1"), showgrid=False)

# ── Figure factories ──────────────────────────────────────────────────────────

_PCTILE_OPTS = {
    "p50": (0.50, "Top 50% · mediana",    "mediana"),
    "p90": (0.90, "Top 10% · umbral P90", "P90"),
    "p95": (0.95, "Top 5%  · umbral P95", "P95"),
    "p99": (0.99, "Top 1%  · umbral P99", "P99"),
}


def fig_percentile_bar(d_totals: pl.DataFrame, pctile_key: str) -> go.Figure:
    q, title_long, label = _PCTILE_OPTS[pctile_key]
    df = (
        d_totals
        .group_by("ENTIDAD_FEDERATIVA")
        .agg(pl.col("PERCEPCIONES_TRIMESTRALES").quantile(q).alias("val"))
        .sort("val", descending=True)
    )
    states = df["ENTIDAD_FEDERATIVA"].to_list()
    vals   = df["val"].to_list()

    v_min, v_max = vals[-1], vals[0]
    span = (v_max - v_min) or 1

    colors = []
    for v in vals:
        t = (v - v_min) / span
        r  = int(0x2E + t * (0xE8 - 0x2E))
        gv = int(0x86 + t * (0x48 - 0x86))
        b  = int(0xAB + t * (0x55 - 0xAB))
        colors.append(f"rgb({r},{gv},{b})")

    fig = go.Figure(go.Bar(
        x=vals, y=states, orientation="h",
        marker_color=colors, marker_line_width=0,
        hovertemplate=f"<b>%{{y}}</b><br>{label} = %{{x:,.0f}} MXN<extra></extra>",
    ))
    fig.update_layout(
        **_DARK, height=600,
        margin=dict(t=50, b=30, l=10, r=20),
        title=dict(text=f"Umbral por estado · {title_long}", font=dict(size=13, color="#94A3B8"), x=0),
        xaxis=dict(**_XAXIS, title="MXN", tickformat=",.0f"),
        yaxis=dict(**_YAXIS),
    )
    return fig


def fig_hist_state(year_range: list[int], tipo: str, state: str, range_opt: str) -> go.Figure:
    lf = _lf.filter(pl.col("YEAR").cast(pl.Int32).is_between(year_range[0], year_range[1]))
    if tipo != "Todos":
        lf = lf.filter(pl.col("TIPO_PLAZA") == tipo)
    if state != "Nacional":
        lf = lf.filter(pl.col("ENTIDAD_FEDERATIVA") == state)

    vals = (
        lf.group_by("CURP").agg(pl.sum("PERCEPCIONES_TRIMESTRALES"))
        .filter(pl.col("PERCEPCIONES_TRIMESTRALES") > 0)
        .collect(engine="streaming")
        ["PERCEPCIONES_TRIMESTRALES"]
    )

    p95 = float(vals.quantile(0.95))
    p99 = float(vals.quantile(0.99))

    if range_opt == "p99":
        vals = vals.filter(vals <= p99)
        lo, hi = 0.0, p99
    elif range_opt == "p95":
        vals = vals.filter(vals <= p95)
        lo, hi = 0.0, p95
    elif range_opt == "top1":
        vals = vals.filter(vals > p99)
        lo, hi = float(vals.min()), float(vals.max())
    else:  # "all"
        lo, hi = 0.0, float(vals.max())

    n_bins   = _N_BINS
    bin_size = (hi - lo) / n_bins if hi > lo else 1.0
    binned   = (
        vals.to_frame()
        .with_columns(
            ((pl.col("PERCEPCIONES_TRIMESTRALES") - lo) / bin_size)
            .floor().cast(pl.Int32).clip(0, n_bins - 1).alias("bin_idx")
        )
        .group_by("bin_idx").agg(pl.len().alias("count"))
    )
    midpoints = [lo + (i + 0.5) * bin_size for i in range(n_bins)]
    counts    = [0] * n_bins
    for idx, cnt in binned.iter_rows():
        counts[idx] = cnt

    total = sum(counts) or 1
    running, cum_pct = 0, []
    for c in counts:
        running += c
        cum_pct.append(running / total * 100)

    total_inc = sum(m * c for m, c in zip(midpoints, counts))
    gini = 0.0
    if total > 0 and total_inc > 0:
        cum_n, cum_inc, area = 0, 0.0, 0.0
        prev_f, prev_l = 0.0, 0.0
        for c, m in zip(counts, midpoints):
            cum_n += c
            cum_inc += m * c
            f = cum_n / total
            l = cum_inc / total_inc
            area += (prev_l + l) / 2 * (f - prev_f)
            prev_f, prev_l = f, l
        gini = 1 - 2 * area

    fig = go.Figure(go.Bar(
        x=midpoints, y=counts, customdata=cum_pct,
        width=bin_size, marker_color="#2E86AB", marker_line_width=0,
        hovertemplate="%{x:,.0f} MXN<br>n=%{y:,}<br>≤ %{customdata:.1f}%<extra></extra>",
    ))

    if range_opt == "all":
        fig.add_vline(x=p95, line=dict(color="#F5C518", width=1.5, dash="dot"),
                      annotation_text=f"P95: {p95:,.0f}",
                      annotation_position="top left",
                      annotation_font=dict(size=9, color="#F5C518"))
        fig.add_vline(x=p99, line=dict(color="#E84855", width=1.5, dash="dash"),
                      annotation_text=f"P99: {p99:,.0f}",
                      annotation_position="top right",
                      annotation_font=dict(size=9, color="#E84855"))
    elif range_opt == "p99":
        fig.add_vline(x=p99, line=dict(color="#E84855", width=1.5, dash="dash"),
                      annotation_text=f"P99: {p99:,.0f}",
                      annotation_position="top right",
                      annotation_font=dict(size=9, color="#E84855"))
    elif range_opt == "p95":
        fig.add_vline(x=p95, line=dict(color="#F5C518", width=1.5, dash="dot"),
                      annotation_text=f"P95: {p95:,.0f}",
                      annotation_position="top right",
                      annotation_font=dict(size=9, color="#F5C518"))

    fig.update_layout(
        **_DARK, bargap=0, height=450,
        margin=dict(t=50, b=40, l=10, r=20),
        title=dict(
            text=f"PERCEPCIONES_TRIMESTRALES · {state} · Gini = {gini:.3f}",
            font=dict(size=13, color="#94A3B8"), x=0,
        ),
        xaxis=dict(**_XAXIS, title="MXN", tickformat=",.0f"),
        yaxis={**_YAXIS, "title": "Registros", "gridcolor": "#334155", "showgrid": True},
    )
    return fig


def fig_pctile_scatter(d_totals: pl.DataFrame) -> go.Figure:
    state_stats = (
        d_totals
        .group_by("ENTIDAD_FEDERATIVA")
        .agg([
            pl.len().alias("total"),
            pl.col("PERCEPCIONES_TRIMESTRALES").quantile(0.50).alias("p50"),
            pl.col("PERCEPCIONES_TRIMESTRALES").quantile(0.90).alias("p90"),
            pl.col("PERCEPCIONES_TRIMESTRALES").quantile(0.95).alias("p95"),
            pl.col("PERCEPCIONES_TRIMESTRALES").quantile(0.99).alias("p99"),
        ])
    )

    states = state_stats["ENTIDAD_FEDERATIVA"].to_list()
    totals = state_stats["total"].to_list()

    pctile_defs = [
        ("P50", "p50", 0.50, "#4E9AF1"),
        ("P90", "p90", 0.10, "#F7B731"),
        ("P95", "p95", 0.05, "#FF6B35"),
        ("P99", "p99", 0.01, "#E84855"),
    ]

    fig = go.Figure()
    for label, col, frac_above, color in pctile_defs:
        xs = state_stats[col].to_list()
        ys = [int(t * frac_above) for t in totals]
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="markers", name=label,
            text=states,
            marker=dict(color=color, size=9, opacity=0.85, line=dict(width=0)),
            hovertemplate=f"<b>%{{text}}</b><br>{label} = %{{x:,.0f}} MXN<br>Trabajadores ≥ {label}: %{{y:,}}<extra></extra>",
        ))

    fig.update_layout(
        **_DARK, height=550,
        margin=dict(t=50, b=40, l=10, r=20),
        title=dict(text="Percentiles salariales por estado · P50 / P90 / P95 / P99", font=dict(size=13, color="#94A3B8"), x=0),
        xaxis=dict(**_XAXIS, title="MXN", tickformat=",.0f"),
        yaxis={**_YAXIS, "title": "Trabajadores", "showgrid": True, "gridcolor": "#334155"},
        legend=dict(orientation="h", y=1.06, x=0, font=dict(size=10), bgcolor="rgba(0,0,0,0)"),
    )
    return fig


_BOX_YEAR_COLORS = {2024: "#4E9AF1", 2025: "#F7B731", 2026: "#3BB273"}
_PCTILE_COLORS   = {"P50": "#4E9AF1", "P90": "#F7B731", "P95": "#FF6B35", "P99": "#E84855"}
_DASH_CYCLE      = ["solid", "dash", "dot", "dashdot", "longdash"]
_GINI_LINE_COLORS = [
    "#4E9AF1", "#F7B731", "#3BB273", "#E84855", "#A78BFA",
    "#F97316", "#06B6D4", "#EC4899", "#84CC16", "#F59E0B",
]


def fig_progress_lines(tipo: str, selected_states: list[str], pctiles: list[str], year_range: list[int]) -> go.Figure:
    if not selected_states or not pctiles:
        fig = go.Figure()
        fig.update_layout(**_DARK, height=500, margin=dict(t=50, b=40, l=10, r=20),
                          title=dict(text="Selecciona al menos un estado", font=dict(size=13, color="#94A3B8"), x=0))
        return fig

    d = _df_curp_totals
    if tipo != "Todos":
        d = d.filter(pl.col("TIPO_PLAZA") == tipo)
    d = d.filter(pl.col("YEAR").cast(pl.Int32).is_between(year_range[0], year_range[1]))

    df = (
        d.filter(pl.col("ENTIDAD_FEDERATIVA").is_in(selected_states))
        .group_by(["ENTIDAD_FEDERATIVA", "YEAR"])
        .agg([
            pl.col("PERCEPCIONES_TRIMESTRALES").quantile(0.50).alias("P50"),
            pl.col("PERCEPCIONES_TRIMESTRALES").quantile(0.90).alias("P90"),
            pl.col("PERCEPCIONES_TRIMESTRALES").quantile(0.95).alias("P95"),
            pl.col("PERCEPCIONES_TRIMESTRALES").quantile(0.99).alias("P99"),
        ])
        .sort(["ENTIDAD_FEDERATIVA", "YEAR"])
    )

    all_years = sorted(df["YEAR"].unique().to_list())
    fig = go.Figure()

    for si, state in enumerate(selected_states):
        dash = _DASH_CYCLE[si % len(_DASH_CYCLE)]
        d_s  = df.filter(pl.col("ENTIDAD_FEDERATIVA") == state).sort("YEAR")
        years = d_s["YEAR"].to_list()
        for pct, color in _PCTILE_COLORS.items():
            if pct not in pctiles:
                continue
            vals = d_s[pct].to_list()
            fig.add_trace(go.Scatter(
                x=years, y=vals, mode="lines+markers",
                name=f"{state} · {pct}",
                line=dict(color=color, dash=dash, width=2),
                marker=dict(size=6),
                hovertemplate=f"<b>{state} · {pct}</b><br>Año: %{{x}}<br>%{{y:,.0f}} MXN<extra></extra>",
            ))

    fig.update_layout(
        **_DARK, height=520,
        margin=dict(t=50, b=40, l=10, r=20),
        title=dict(text="Evolución de percentiles salariales por estado", font=dict(size=13, color="#94A3B8"), x=0),
        xaxis=dict(**_XAXIS, title="Año", tickmode="array", tickvals=all_years, dtick=1),
        yaxis={**_YAXIS, "title": "MXN", "tickformat": ",.0f", "showgrid": True, "gridcolor": "#334155"},
        legend=dict(font=dict(size=10), bgcolor="rgba(0,0,0,0)"),
    )
    return fig


def fig_box_pctiles(d_totals: pl.DataFrame, selected_states: list[str]) -> go.Figure:
    if selected_states:
        d_totals = d_totals.filter(pl.col("ENTIDAD_FEDERATIVA").is_in(selected_states))
    pctile_defs = [("P50", 0.50), ("P90", 0.90), ("P95", 0.95), ("P99", 0.99)]
    pctile_cols = [k for k, _ in pctile_defs]
    agg = (
        d_totals
        .group_by(["ENTIDAD_FEDERATIVA", "YEAR"])
        .agg([pl.col("PERCEPCIONES_TRIMESTRALES").quantile(q).alias(k) for k, q in pctile_defs])
        .sort(["YEAR", "ENTIDAD_FEDERATIVA"])
    )
    years = sorted(agg["YEAR"].unique().to_list())
    year_label = str(years[0]) if len(years) == 1 else f"{years[0]}–{years[-1]}"
    fig = go.Figure()
    for year in years:
        d_y = agg.filter(pl.col("YEAR") == year)
        states = d_y["ENTIDAD_FEDERATIVA"].to_list()
        fig.add_trace(go.Box(
            name=str(year),
            x=[p for p in pctile_cols for _ in states],
            y=[v for p in pctile_cols for v in d_y[p].to_list()],
            text=[s for _ in pctile_cols for s in states],
            marker_color=_BOX_YEAR_COLORS.get(year, "#CBD5E1"),
            boxmean=True,
            boxpoints="outliers",
            hovertemplate="<b>%{text}</b><br>%{y:,.0f} MXN<extra></extra>",
        ))
    fig.update_layout(
        **_DARK, boxmode="group", height=460,
        margin=dict(t=60, b=40, l=10, r=20),
        title=dict(
            text=f"<b>Dispersión entre estados por percentil salarial · {year_label}</b>"
                 "<br><sup style='color:#94A3B8'>Percepción anual por CURP (MXN). 2026 = Q1 únicamente.</sup>",
            font=dict(size=13, color="#F8FAFC"), x=0,
        ),
        xaxis=dict(**_XAXIS, title=""),
        yaxis={**_YAXIS, "title": "MXN", "tickformat": ",.0f",
               "showgrid": True, "gridcolor": "#334155"},
        legend=dict(title="Año", orientation="v", font=dict(size=11), bgcolor="rgba(0,0,0,0)"),
    )
    return fig


def fig_gini_map(year_range: list[int], tipo: str) -> go.Figure:
    d = _df_raw.filter(pl.col("YEAR").cast(pl.Int32).is_between(year_range[0], year_range[1]))
    if tipo != "Todos":
        d = d.filter(pl.col("TIPO_PLAZA") == tipo)

    state_bins = (
        d.group_by(["ENTIDAD_FEDERATIVA", "bin_idx"])
        .agg(pl.col("count").sum())
    )

    names, ginis, labels = [], [], []
    for state in STATES:
        d_s = state_bins.filter(pl.col("ENTIDAD_FEDERATIVA") == state)
        counts_map = dict(zip(d_s["bin_idx"].to_list(), d_s["count"].to_list()))
        counts = [counts_map.get(i, 0) for i in range(_N_BINS)]
        total_n   = sum(counts)
        total_inc = sum(m * c for m, c in zip(_MIDPOINTS, counts))
        gini = 0.0
        if total_n > 0 and total_inc > 0:
            cum_n, cum_inc, area = 0, 0.0, 0.0
            prev_f, prev_l = 0.0, 0.0
            for c, m in zip(counts, _MIDPOINTS):
                cum_n   += c
                cum_inc += m * c
                f = cum_n   / total_n
                l = cum_inc / total_inc
                area   += (prev_l + l) / 2 * (f - prev_f)
                prev_f, prev_l = f, l
            gini = 1 - 2 * area
        names.append(state.title())
        ginis.append(round(gini, 4))
        labels.append(f"{state.title()}: {gini:.3f}")

    g_min = min(ginis)
    g_max = max(ginis)
    g_pad = (g_max - g_min) * 0.1 or 0.01

    fig = go.Figure(go.Choroplethmap(
        geojson=_MEXICO_GEOJSON,
        locations=names,
        z=ginis,
        featureidkey="properties.name",
        colorscale="RdYlGn_r",
        zmin=round(g_min - g_pad, 3),
        zmax=round(g_max + g_pad, 3),
        text=labels,
        hovertemplate="<b>%{text}</b><extra></extra>",
        marker_line_color="#334155",
        marker_line_width=0.5,
        colorbar=dict(
            title=dict(text="Gini", font=dict(color="#CBD5E1", size=11)),
            tickfont=dict(color="#94A3B8", size=9),
            tickformat=".3f",
            thickness=12, len=0.6,
        ),
    ))
    fig.update_layout(
        **_DARK, height=580,
        margin=dict(t=60, b=10, l=0, r=0),
        title=dict(
            text="<b>Índice Gini salarial por estado</b>"
                 "<br><sup style='color:#94A3B8'>Percepción anual por CURP (MXN). CDMX sin datos FONE.</sup>",
            font=dict(size=13, color="#F8FAFC"), x=0,
        ),
        map=dict(
            style="carto-darkmatter",
            center=dict(lat=23.6, lon=-102.5),
            zoom=3.8,
        ),
    )
    return fig


def fig_gini_lines(tipo: str, selected_states: list[str], year_range: list[int]) -> go.Figure:
    d = _df_raw.filter(pl.col("YEAR").cast(pl.Int32).is_between(year_range[0], year_range[1]))
    if tipo != "Todos":
        d = d.filter(pl.col("TIPO_PLAZA") == tipo)

    states = selected_states if selected_states else STATES
    years  = sorted(d["YEAR"].cast(pl.Int32).unique().to_list())

    state_bins = d.group_by(["ENTIDAD_FEDERATIVA", "YEAR", "bin_idx"]).agg(pl.col("count").sum())

    fig = go.Figure()
    for si, state in enumerate(states):
        color = _GINI_LINE_COLORS[si % len(_GINI_LINE_COLORS)]
        ginis = []
        for yr in years:
            d_s = state_bins.filter(
                (pl.col("ENTIDAD_FEDERATIVA") == state) & (pl.col("YEAR").cast(pl.Int32) == yr)
            )
            counts_map = dict(zip(d_s["bin_idx"].to_list(), d_s["count"].to_list()))
            counts    = [counts_map.get(i, 0) for i in range(_N_BINS)]
            total_n   = sum(counts)
            total_inc = sum(m * c for m, c in zip(_MIDPOINTS, counts))
            gini = 0.0
            if total_n > 0 and total_inc > 0:
                cum_n, cum_inc, area = 0, 0.0, 0.0
                prev_f, prev_l = 0.0, 0.0
                for c, m in zip(counts, _MIDPOINTS):
                    cum_n   += c
                    cum_inc += m * c
                    f = cum_n   / total_n
                    l = cum_inc / total_inc
                    area   += (prev_l + l) / 2 * (f - prev_f)
                    prev_f, prev_l = f, l
                gini = 1 - 2 * area
            ginis.append(round(gini, 4))

        fig.add_trace(go.Scatter(
            x=years, y=ginis, mode="lines+markers",
            name=state.title(),
            line=dict(color=color, width=2),
            marker=dict(size=6),
            hovertemplate=f"<b>{state.title()}</b><br>Año: %{{x}}<br>Gini: %{{y:.3f}}<extra></extra>",
        ))

    fig.update_layout(
        **_DARK, height=400,
        margin=dict(t=60, b=40, l=10, r=20),
        title=dict(
            text="<b>Evolución del índice Gini por estado</b>"
                 "<br><sup style='color:#94A3B8'>Percepción anual por CURP (MXN). CDMX sin datos FONE.</sup>",
            font=dict(size=13, color="#F8FAFC"), x=0,
        ),
        xaxis=dict(**_XAXIS, title="Año", tickmode="array", tickvals=years, dtick=1),
        yaxis={**_YAXIS, "title": "Gini", "tickformat": ".3f", "showgrid": True, "gridcolor": "#334155"},
        legend=dict(font=dict(size=10), bgcolor="rgba(0,0,0,0)"),
    )
    return fig


# ── App ───────────────────────────────────────────────────────────────────────

app = Dash(__name__, external_stylesheets=[dbc.themes.SLATE])

_TAB_STYLE     = {"color": "#94A3B8", "backgroundColor": "#1E293B", "borderColor": "#334155"}
_TAB_SEL_STYLE = {"color": "#F8FAFC", "backgroundColor": "#0F172A", "borderColor": "#2E86AB", "borderTop": "2px solid #2E86AB"}

app.layout = html.Div(
    style={"backgroundColor": "#0F172A", "minHeight": "100vh", "padding": "24px", "fontFamily": "sans-serif"},
    children=[
        html.H2("Nómina Educativa Federalizada · FONE", style={"color": "#F8FAFC", "marginBottom": "4px"}),
        html.P(
            "Distribución salarial por entidad · PERCEPCIONES_TRIMESTRALES por CURP · SEP/DGSANEF · 2024–2026",
            style={"color": "#64748B", "marginBottom": "20px"},
        ),

        dbc.Row([
            dbc.Col([
                html.Label("Rango de años", style={"color": "#94A3B8", "fontSize": "12px"}),
                dcc.RangeSlider(
                    id="sl-year",
                    min=YEARS[0], max=YEARS[-1], step=1,
                    marks={y: {"label": str(y), "style": {"color": "#94A3B8"}} for y in YEARS},
                    value=[YEARS[0], YEARS[-1]],
                    tooltip={"placement": "bottom", "always_visible": True},
                ),
            ], md=5),
            dbc.Col([
                html.Label("Tipo de plaza", style={"color": "#94A3B8", "fontSize": "12px"}),
                dcc.Dropdown(
                    id="dd-tipo",
                    options=[{"label": t, "value": t} for t in TIPOS],
                    value="Todos", clearable=False,
                    style={"backgroundColor": "#1E293B", "color": "#0F172A"},
                ),
            ], md=3),
        ], className="mb-3"),

        dcc.Tabs(
            id="tabs", value="tab-comp",
            colors={"border": "#334155", "primary": "#2E86AB", "background": "#1E293B"},
            children=[
                dcc.Tab(
                    label="Comparativo estatal", value="tab-comp",
                    style=_TAB_STYLE, selected_style=_TAB_SEL_STYLE,
                    children=[
                        dbc.Row([
                            dbc.Col([
                                html.Label("Estados", style={"color": "#94A3B8", "fontSize": "12px"}),
                                dcc.Dropdown(
                                    id="dd-states",
                                    options=[{"label": s, "value": s} for s in STATES],
                                    value=STATES[:3], multi=True, clearable=False,
                                    style={"backgroundColor": "#1E293B", "color": "#0F172A"},
                                ),
                            ], md=7),
                            dbc.Col([
                                html.Label("Percentiles", style={"color": "#94A3B8", "fontSize": "12px"}),
                                dcc.Checklist(
                                    id="ck-pctiles",
                                    options=[
                                        {"label": " P50", "value": "P50"},
                                        {"label": " P90", "value": "P90"},
                                        {"label": " P95", "value": "P95"},
                                        {"label": " P99", "value": "P99"},
                                    ],
                                    value=["P50", "P90", "P95", "P99"],
                                    inline=True,
                                    style={"color": "#CBD5E1", "fontSize": "13px", "marginTop": "6px"},
                                    inputStyle={"marginRight": "4px", "marginLeft": "12px"},
                                ),
                            ], md=5),
                        ], className="mt-3 mb-3"),
                        dbc.Row([
                            dbc.Col(html.Div(dcc.Graph(id="chart-progress"), style=CARD_STYLE), md=12),
                        ]),
                    ],
                ),
                dcc.Tab(
                    label="Distribución por estado", value="tab-dist",
                    style=_TAB_STYLE, selected_style=_TAB_SEL_STYLE,
                    children=[
                        dbc.Row([
                            dbc.Col([
                                html.Label("Estado", style={"color": "#94A3B8", "fontSize": "12px"}),
                                dcc.Dropdown(
                                    id="dd-state",
                                    options=[{"label": "Nacional", "value": "Nacional"}]
                                            + [{"label": s, "value": s} for s in STATES],
                                    value="Nacional", clearable=False,
                                    style={"backgroundColor": "#1E293B", "color": "#0F172A"},
                                ),
                            ], md=4),
                            dbc.Col([
                                html.Label("Rango", style={"color": "#94A3B8", "fontSize": "12px"}),
                                dcc.RadioItems(
                                    id="rd-range",
                                    options=[
                                        {"label": " Hasta P95 (95%)", "value": "p95"},
                                        {"label": " Hasta P99 (99%)", "value": "p99"},
                                        {"label": " Todo (100%)",      "value": "all"},
                                        {"label": " Solo top 1%",      "value": "top1"},
                                    ],
                                    value="p99", inline=True,
                                    style={"color": "#CBD5E1", "fontSize": "13px", "marginTop": "6px"},
                                    inputStyle={"marginRight": "4px", "marginLeft": "12px"},
                                ),
                            ], md=8),
                        ], className="mt-3 mb-3"),
                        dbc.Row([
                            dbc.Col(html.Div(dcc.Graph(id="chart-hist-state"), style=CARD_STYLE), md=12),
                        ]),
                    ],
                ),
                dcc.Tab(
                    label="Percentiles por estado", value="tab-scatter",
                    style=_TAB_STYLE, selected_style=_TAB_SEL_STYLE,
                    children=[
                        dbc.Row([
                            dbc.Col(html.Div(dcc.Graph(id="chart-scatter"), style=CARD_STYLE), md=12),
                        ], className="mt-3"),
                    ],
                ),
                dcc.Tab(
                    label="Dispersión entre estados", value="tab-box",
                    style=_TAB_STYLE, selected_style=_TAB_SEL_STYLE,
                    children=[
                        dbc.Row([
                            dbc.Col([
                                html.Label("Estados", style={"color": "#94A3B8", "fontSize": "12px"}),
                                dcc.Dropdown(
                                    id="dd-states-box",
                                    options=[{"label": s, "value": s} for s in STATES],
                                    value=[], multi=True, clearable=True,
                                    placeholder="Todos los estados",
                                    style={"backgroundColor": "#1E293B", "color": "#0F172A"},
                                ),
                            ], md=8),
                        ], className="mt-3 mb-3"),
                        dbc.Row([
                            dbc.Col(html.Div(dcc.Graph(id="chart-box-pctiles"), style=CARD_STYLE), md=12),
                        ]),
                    ],
                ),
                dcc.Tab(
                    label="Mapa Gini", value="tab-gini",
                    style=_TAB_STYLE, selected_style=_TAB_SEL_STYLE,
                    children=[
                        dbc.Row([
                            dbc.Col(html.Div(dcc.Graph(id="chart-gini-map"), style=CARD_STYLE), md=12),
                        ], className="mt-3"),
                        dbc.Row([
                            dbc.Col([
                                html.Label("Estados (líneas)", style={"color": "#94A3B8", "fontSize": "12px"}),
                                dcc.Dropdown(
                                    id="dd-states-gini",
                                    options=[{"label": s.title(), "value": s} for s in STATES],
                                    value=[], multi=True, clearable=True,
                                    placeholder="Todos los estados",
                                    style={"backgroundColor": "#1E293B", "color": "#0F172A"},
                                ),
                            ], md=8),
                        ], className="mt-3 mb-2"),
                        dbc.Row([
                            dbc.Col(html.Div(dcc.Graph(id="chart-gini-lines"), style=CARD_STYLE), md=12),
                        ]),
                    ],
                ),
                dcc.Tab(
                    label="Umbral por estado", value="tab-umbral",
                    style=_TAB_STYLE, selected_style=_TAB_SEL_STYLE,
                    children=[
                        dbc.Row([
                            dbc.Col([
                                html.Label("Percentil", style={"color": "#94A3B8", "fontSize": "12px"}),
                                dcc.RadioItems(
                                    id="rd-pctile",
                                    options=[
                                        {"label": " Top 50% (mediana)", "value": "p50"},
                                        {"label": " Top 10% (P90)",     "value": "p90"},
                                        {"label": " Top 5%  (P95)",     "value": "p95"},
                                        {"label": " Top 1%  (P99)",     "value": "p99"},
                                    ],
                                    value="p90", inline=True,
                                    style={"color": "#CBD5E1", "fontSize": "13px", "marginTop": "6px"},
                                    inputStyle={"marginRight": "4px", "marginLeft": "12px"},
                                ),
                            ], md=12),
                        ], className="mt-3 mb-3"),
                        dbc.Row([
                            dbc.Col(html.Div(dcc.Graph(id="chart-umbral"), style=CARD_STYLE), md=12),
                        ]),
                    ],
                ),
            ],
        ),

        html.P(
            "Fuente: SEP/DGSANEF · Nómina Educativa Federalizada · Art. 73 LGCG. "
            "Incluye todos los valores. 2026 contiene únicamente Q1.",
            style={"color": "#475569", "fontSize": "11px", "textAlign": "center", "marginTop": "16px"},
        ),
    ],
)

# ── Callbacks ─────────────────────────────────────────────────────────────────

@app.callback(
    Output("chart-umbral", "figure"),
    Input("sl-year",   "value"),
    Input("dd-tipo",   "value"),
    Input("rd-pctile", "value"),
)
def update_umbral(year_range: list, tipo: str, pctile_key: str):
    y_min, y_max = year_range
    d = _df_curp_totals.filter(pl.col("YEAR").cast(pl.Int32).is_between(y_min, y_max))
    if tipo != "Todos":
        d = d.filter(pl.col("TIPO_PLAZA") == tipo)
    return fig_percentile_bar(d, pctile_key)


@app.callback(
    Output("chart-progress", "figure"),
    Input("sl-year",    "value"),
    Input("dd-tipo",    "value"),
    Input("dd-states",  "value"),
    Input("ck-pctiles", "value"),
)
def update_progress(year_range: list, tipo: str, selected_states: list, pctiles: list):
    return fig_progress_lines(tipo, selected_states or [], pctiles or [], year_range)


@app.callback(
    Output("chart-hist-state", "figure"),
    Input("sl-year",  "value"),
    Input("dd-tipo",  "value"),
    Input("dd-state", "value"),
    Input("rd-range", "value"),
)
def update_hist_state(year_range: list, tipo: str, state: str, range_opt: str):
    return fig_hist_state(year_range, tipo, state, range_opt)


@app.callback(
    Output("chart-scatter", "figure"),
    Input("sl-year", "value"),
    Input("dd-tipo", "value"),
)
def update_scatter(year_range: list, tipo: str):
    y_min, y_max = year_range
    d = _df_curp_totals.filter(pl.col("YEAR").cast(pl.Int32).is_between(y_min, y_max))
    if tipo != "Todos":
        d = d.filter(pl.col("TIPO_PLAZA") == tipo)
    return fig_pctile_scatter(d)


@app.callback(
    Output("chart-box-pctiles", "figure"),
    Input("sl-year",       "value"),
    Input("dd-tipo",       "value"),
    Input("dd-states-box", "value"),
)
def update_box_pctiles(year_range: list, tipo: str, selected_states: list):
    y_min, y_max = year_range
    d = _df_curp_totals.filter(pl.col("YEAR").cast(pl.Int32).is_between(y_min, y_max))
    if tipo != "Todos":
        d = d.filter(pl.col("TIPO_PLAZA") == tipo)
    return fig_box_pctiles(d, selected_states or [])


@app.callback(
    Output("chart-gini-map", "figure"),
    Input("sl-year", "value"),
    Input("dd-tipo", "value"),
)
def update_gini_map(year_range: list, tipo: str):
    return fig_gini_map(year_range, tipo)


@app.callback(
    Output("chart-gini-lines", "figure"),
    Input("sl-year",        "value"),
    Input("dd-tipo",        "value"),
    Input("dd-states-gini", "value"),
)
def update_gini_lines(year_range: list, tipo: str, selected_states: list):
    return fig_gini_lines(tipo, selected_states or [], year_range)


if __name__ == "__main__":
    app.run(debug=True)
