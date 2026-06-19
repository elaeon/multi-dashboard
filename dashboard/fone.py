import glob
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



def _approx_quantile(counts: list[int], p: float) -> float:
    total = sum(counts)
    target = total * p
    cumsum = 0
    for i, c in enumerate(counts):
        cumsum += c
        if cumsum >= target:
            return _MIDPOINTS[i]
    return _MIDPOINTS[-1]


def _gini(counts: list[int]) -> float:
    total_n = sum(counts)
    total_inc = sum(m * c for m, c in zip(_MIDPOINTS, counts))
    if total_n == 0 or total_inc == 0:
        return 0.0
    cum_n, cum_inc, area = 0, 0.0, 0.0
    prev_f, prev_l = 0.0, 0.0
    for c, m in zip(counts, _MIDPOINTS):
        cum_n += c
        cum_inc += m * c
        f = cum_n / total_n
        l = cum_inc / total_inc
        area += (prev_l + l) / 2 * (f - prev_f)  # trapezoidal rule
        prev_f, prev_l = f, l
    return 1 - 2 * area


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

# Binned frames — used by Gini chart
_df_raw   = _bin_lf(_lf, ["ENTIDAD_FEDERATIVA", "YEAR", "TIPO_PLAZA"]).collect(engine="streaming")
_df_todos = _bin_lf(_lf, ["ENTIDAD_FEDERATIVA", "YEAR"]).collect(engine="streaming")

_FRAMES = {
    "Todos": _df_todos,
    "PLAZA": _df_raw.filter(pl.col("TIPO_PLAZA") == "PLAZA"),
    "HORA":  _df_raw.filter(pl.col("TIPO_PLAZA") == "HORA"),
}

# Per-CURP annual totals — base for all threshold frames
_df_curp_totals = (
    _lf.group_by(["CURP", "ENTIDAD_FEDERATIVA", "YEAR", "TIPO_PLAZA"])
    .agg(pl.col("PERCEPCIONES_TRIMESTRALES").sum())
    .filter(pl.col("PERCEPCIONES_TRIMESTRALES") > 0)
    .collect(engine="streaming")
)


# National heptile cutoffs (P1/7 … P6/7) from all per-CURP annual totals
_vals_all = _df_curp_totals["PERCEPCIONES_TRIMESTRALES"]
_Q = [float(_vals_all.quantile(i / 7)) for i in range(1, 7)]


def _fmt_mxn(x: float) -> str:
    return f"{x/1_000_000:.1f}M" if x >= 1_000_000 else f"{x/1_000:.0f}k"


_BAND_LABELS = [
    f"H1 · < {_fmt_mxn(_Q[0])}",
    f"H2 · {_fmt_mxn(_Q[0])}–{_fmt_mxn(_Q[1])}",
    f"H3 · {_fmt_mxn(_Q[1])}–{_fmt_mxn(_Q[2])}",
    f"H4 · {_fmt_mxn(_Q[2])}–{_fmt_mxn(_Q[3])}",
    f"H5 · {_fmt_mxn(_Q[3])}–{_fmt_mxn(_Q[4])}",
    f"H6 · {_fmt_mxn(_Q[4])}–{_fmt_mxn(_Q[5])}",
    f"H7 · ≥ {_fmt_mxn(_Q[5])}",
]
_BAND_COLORS = ["#4E9AF1", "#45B7D1", "#4ECDC4", "#95E06C", "#F7B731", "#FF6B35", "#E84855"]

_BAND_EXPR = (
    pl.when(pl.col("PERCEPCIONES_TRIMESTRALES") < _Q[0]).then(pl.lit(0))
    .when(pl.col("PERCEPCIONES_TRIMESTRALES") < _Q[1]).then(pl.lit(1))
    .when(pl.col("PERCEPCIONES_TRIMESTRALES") < _Q[2]).then(pl.lit(2))
    .when(pl.col("PERCEPCIONES_TRIMESTRALES") < _Q[3]).then(pl.lit(3))
    .when(pl.col("PERCEPCIONES_TRIMESTRALES") < _Q[4]).then(pl.lit(4))
    .when(pl.col("PERCEPCIONES_TRIMESTRALES") < _Q[5]).then(pl.lit(5))
    .otherwise(pl.lit(6))
    .alias("band")
)

_df_raw_bands   = (
    _df_curp_totals
    .with_columns(_BAND_EXPR)
    .group_by(["ENTIDAD_FEDERATIVA", "YEAR", "TIPO_PLAZA", "band"])
    .agg(pl.len().alias("count"))
)
_df_todos_bands = (
    _df_curp_totals
    .with_columns(_BAND_EXPR)
    .group_by(["ENTIDAD_FEDERATIVA", "YEAR", "band"])
    .agg(pl.len().alias("count"))
)

_FRAMES_BANDS = {
    "Todos": _df_todos_bands,
    "PLAZA": _df_raw_bands.filter(pl.col("TIPO_PLAZA") == "PLAZA"),
    "HORA":  _df_raw_bands.filter(pl.col("TIPO_PLAZA") == "HORA"),
}

YEARS  = sorted(_df_raw["YEAR"].cast(pl.Int32).unique().to_list())
TIPOS  = ["Todos"] + sorted(_df_raw["TIPO_PLAZA"].unique().to_list())
STATES = sorted(_df_raw["ENTIDAD_FEDERATIVA"].unique().to_list())

# ── Theme ─────────────────────────────────────────────────────────────────────

CARD_STYLE = {
    "background": "#1E293B", "border": "1px solid #334155",
    "borderRadius": "8px", "padding": "12px",
}

_DARK  = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#CBD5E1")
_XAXIS = dict(tickfont=dict(size=9, color="#94A3B8"), gridcolor="#334155", showgrid=True, zeroline=False)
_YAXIS = dict(tickfont=dict(size=9, color="#CBD5E1"), showgrid=False)

# ── Figure factories ──────────────────────────────────────────────────────────

def fig_million_split(d_bands: pl.DataFrame, d_bins: pl.DataFrame) -> go.Figure:
    # Gini per state from binned data (for sorting)
    state_gini: dict[str, float] = {}
    for row in (
        d_bins.group_by("ENTIDAD_FEDERATIVA")
        .agg(pl.col("bin_idx"), pl.col("count"))
        .iter_rows(named=True)
    ):
        counts = [0] * _N_BINS
        for idx, c in zip(row["bin_idx"], row["count"]):
            counts[idx] = c
        state_gini[row["ENTIDAD_FEDERATIVA"]] = _gini(counts)

    state_counts: dict[str, list[int]] = {}
    for row in (
        d_bands.group_by("ENTIDAD_FEDERATIVA")
        .agg(pl.col("band"), pl.col("count"))
        .iter_rows(named=True)
    ):
        counts = [0] * 7
        for b, c in zip(row["band"], row["count"]):
            counts[b] = c
        state_counts[row["ENTIDAD_FEDERATIVA"]] = counts

    # sort by Gini descending (most unequal at top → Q5-heavy states at bottom)
    states = sorted(state_counts, key=lambda s: state_gini.get(s, 0), reverse=True)
    totals = [sum(state_counts[s]) or 1 for s in states]

    ginis = [state_gini.get(s, 0) for s in states]

    fig = go.Figure()
    for band_idx, (label, color) in enumerate(zip(_BAND_LABELS, _BAND_COLORS)):
        ns   = [state_counts[s][band_idx] for s in states]
        pcts = [n / t * 100 for n, t in zip(ns, totals)]
        fig.add_trace(go.Bar(
            x=pcts, y=states, orientation="h", name=label,
            marker_color=color, marker_line_width=0,
            customdata=list(zip(ns, ginis)),
            hovertemplate=f"<b>%{{y}}</b><br>{label} MXN<br>n=%{{customdata[0]:,}}<br>%{{x:.1f}}%<br>Gini=%{{customdata[1]:.3f}}<extra></extra>",
        ))

    fig.update_layout(
        **_DARK, barmode="stack", height=600,
        margin=dict(t=50, b=30, l=10, r=20),
        title=dict(text="Distribución salarial · heptiles nacionales · ordenado por Gini", font=dict(size=13, color="#94A3B8"), x=0),
        xaxis=dict(**_XAXIS, title="%", range=[0, 100], ticksuffix="%"),
        yaxis=dict(**_YAXIS),
        legend=dict(orientation="h", y=1.06, x=0, font=dict(size=10), bgcolor="rgba(0,0,0,0)"),
    )
    return fig


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


def fig_hist_state(year: int, tipo: str, state: str, range_opt: str) -> go.Figure:
    lf = _lf.filter(pl.col("YEAR").cast(pl.Int32) == year)
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
                html.Label("Año", style={"color": "#94A3B8", "fontSize": "12px"}),
                dcc.Dropdown(
                    id="dd-year",
                    options=[{"label": str(y), "value": y} for y in YEARS],
                    value=YEARS[0], clearable=False,
                    style={"backgroundColor": "#1E293B", "color": "#0F172A"},
                ),
            ], md=3),
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
                            dbc.Col(html.Div(dcc.Graph(id="chart-million"), style=CARD_STYLE), md=12),
                        ], className="mt-3"),
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
    Input("dd-year",   "value"),
    Input("dd-tipo",   "value"),
    Input("rd-pctile", "value"),
)
def update_umbral(year: int, tipo: str, pctile_key: str):
    d = _df_curp_totals.filter(pl.col("YEAR").cast(pl.Int32) == int(year))
    if tipo != "Todos":
        d = d.filter(pl.col("TIPO_PLAZA") == tipo)
    return fig_percentile_bar(d, pctile_key)


@app.callback(
    Output("chart-million", "figure"),
    Input("dd-year", "value"),
    Input("dd-tipo", "value"),
)
def update_million(year: int, tipo: str):
    yr = int(year)
    d_bands = _FRAMES_BANDS[tipo].filter(pl.col("YEAR").cast(pl.Int32) == yr)
    d_bins  = _FRAMES[tipo].filter(pl.col("YEAR").cast(pl.Int32) == yr)
    return fig_million_split(d_bands, d_bins)


@app.callback(
    Output("chart-hist-state", "figure"),
    Input("dd-year",  "value"),
    Input("dd-tipo",  "value"),
    Input("dd-state", "value"),
    Input("rd-range", "value"),
)
def update_hist_state(year: int, tipo: str, state: str, range_opt: str):
    return fig_hist_state(int(year), tipo, state, range_opt)


@app.callback(
    Output("chart-scatter", "figure"),
    Input("dd-year", "value"),
    Input("dd-tipo", "value"),
)
def update_scatter(year: int, tipo: str):
    d = _df_curp_totals.filter(pl.col("YEAR").cast(pl.Int32) == int(year))
    if tipo != "Todos":
        d = d.filter(pl.col("TIPO_PLAZA") == tipo)
    return fig_pctile_scatter(d)


if __name__ == "__main__":
    app.run(debug=True)
