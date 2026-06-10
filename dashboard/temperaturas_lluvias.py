import polars as pl
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc

# ── Data loading ──────────────────────────────────────────────────────────────

_RAW = pl.read_csv("data/temperaturas_lluvias.csv")

df = _RAW.with_columns(
    pl.col("PERIODO").str.slice(0, 4).cast(pl.Int32).alias("anio"),
    pl.col("PERIODO").str.slice(5, 2).cast(pl.Int32).alias("mes"),
)

ESTADOS = sorted(
    df.filter(pl.col("CVE_ENT") != 0)["ENTIDAD"].unique().to_list()
)

MES_NOMBRES = ["Ene", "Feb", "Mar", "Abr", "May", "Jun",
               "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]

# Baseline (1985–1994) per entity — used for warming delta KPIs and dumbbell
_baseline = (
    df.filter(pl.col("anio").is_between(1985, 1994))
    .group_by("ENTIDAD")
    .agg(
        pl.col("MEDIA").mean().alias("baseline_media"),
        pl.col("MAXIMA").mean().alias("baseline_maxima"),
    )
)

# ── Theme ─────────────────────────────────────────────────────────────────────

CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font_color="#CBD5E1",
    xaxis=dict(gridcolor="#334155"),
    yaxis=dict(gridcolor="#334155"),
    margin=dict(t=40, b=40, l=10, r=10),
)
CARD_STYLE = {
    "background": "#1E293B", "border": "1px solid #334155",
    "borderRadius": "8px", "padding": "16px", "textAlign": "center",
}
TAB_STYLE = {"backgroundColor": "#0F172A", "color": "#94A3B8", "borderTop": "none",
             "borderBottom": "1px solid #334155", "padding": "10px 18px"}
TAB_SEL   = {"backgroundColor": "#1E293B", "color": "#F8FAFC",
             "borderTop": "2px solid #2E86AB", "borderBottom": "none",
             "fontWeight": "600", "padding": "10px 18px"}


def kpi_card(title: str, value: str, color: str = "#CBD5E1",
             delta: str | None = None, delta_color: str | None = None) -> dbc.Col:
    children = [
        html.P(title, style={"color": "#94A3B8", "fontSize": "12px", "margin": 0}),
        html.H3(value, style={"color": color, "margin": "4px 0 0"}),
    ]
    if delta is not None:
        children.append(
            html.P(delta, style={"color": delta_color or "#94A3B8",
                                 "fontSize": "11px", "margin": "3px 0 0"})
        )
    return dbc.Col(html.Div(children, style=CARD_STYLE), xs=12, sm=6, md=3)


# ── Figure factories ──────────────────────────────────────────────────────────

def fig_temp_trend(d: pl.DataFrame) -> go.Figure:
    anual = (
        d.filter(pl.col("anio").is_between(1985, 2025))
        .group_by("anio")
        .agg(
            pl.col("MINIMA").mean().round(2),
            pl.col("MEDIA").mean().round(2),
            pl.col("MAXIMA").mean().round(2),
        )
        .sort("anio")
    )
    years = anual["anio"].to_list()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=years, y=anual["MAXIMA"].to_list(),
        mode="lines", name="Máxima",
        line=dict(color="#E84855", width=1.5),
        fill="tonexty", fillcolor="rgba(232,72,85,0.07)",
    ))
    fig.add_trace(go.Scatter(
        x=years, y=anual["MEDIA"].to_list(),
        mode="lines+markers", name="Media",
        line=dict(color="#F4A261", width=2.5),
        marker=dict(size=4),
    ))
    fig.add_trace(go.Scatter(
        x=years, y=anual["MINIMA"].to_list(),
        mode="lines", name="Mínima",
        line=dict(color="#2E86AB", width=1.5),
    ))
    # Annotate 2023: hottest year on record
    media_2023 = anual.filter(pl.col("anio") == 2023)["MEDIA"]
    if len(media_2023) > 0:
        fig.add_annotation(
            x=2023, y=float(media_2023[0]),
            text="<b>2023</b><br>más cálido<br>y más seco",
            font=dict(color="#F4A261", size=11),
            arrowcolor="#64748B", ax=36, ay=-38,
            bgcolor="rgba(30,41,59,0.85)", bordercolor="#475569",
        )
    # Reference line: baseline mean (1985-1994) — only for Nacional
    baseline_row = _baseline.filter(pl.col("ENTIDAD") == d["ENTIDAD"][0])
    if len(baseline_row) > 0:
        base_val = float(baseline_row["baseline_media"][0])
        fig.add_hline(
            y=base_val, line=dict(color="#64748B", width=1, dash="dot"),
            annotation_text="media 1985–1994",
            annotation_font_color="#64748B",
            annotation_position="bottom right",
        )
    fig.update_layout(
        title=dict(
            text="<b>La temperatura media sube +0.5°C por década (1985–2025)</b>"
                 "<br><sup style='color:#94A3B8'>Temperatura mínima, media y máxima anual — las noches se calientan ligeramente más rápido que los días</sup>",
        ),
        height=400,
        xaxis=dict(gridcolor="#334155", title="Año", dtick=5),
        yaxis=dict(gridcolor="#334155", ticksuffix="°C"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font_color="#94A3B8",
                    orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        **{k: v for k, v in CHART_LAYOUT.items() if k not in ("xaxis", "yaxis", "margin")},
        margin=dict(t=70, b=40, l=10, r=10),
    )
    return fig


def fig_temp_seasonality(d: pl.DataFrame) -> go.Figure:
    seasonal = (
        d.group_by("mes")
        .agg(
            pl.col("MINIMA").mean().round(1),
            pl.col("MEDIA").mean().round(1),
            pl.col("MAXIMA").mean().round(1),
        )
        .sort("mes")
    )
    meses = [MES_NOMBRES[m - 1] for m in seasonal["mes"].to_list()]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=meses, y=seasonal["MAXIMA"].to_list(),
        name="Máxima", marker_color="#E84855", opacity=0.7,
    ))
    fig.add_trace(go.Bar(
        x=meses, y=seasonal["MEDIA"].to_list(),
        name="Media", marker_color="#F4A261",
    ))
    fig.add_trace(go.Bar(
        x=meses, y=seasonal["MINIMA"].to_list(),
        name="Mínima", marker_color="#2E86AB", opacity=0.7,
    ))
    fig.update_layout(
        barmode="group",
        title=dict(
            text="<b>Mayo–Sep concentra el calor; Dic–Feb los meses más fríos</b>"
                 "<br><sup style='color:#94A3B8'>Temperatura promedio por mes (todo el período)</sup>",
        ),
        height=360,
        xaxis=dict(gridcolor="#334155"),
        yaxis=dict(gridcolor="#334155", ticksuffix="°C"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font_color="#94A3B8",
                    orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        **{k: v for k, v in CHART_LAYOUT.items() if k not in ("xaxis", "yaxis", "margin")},
        margin=dict(t=60, b=40, l=10, r=10),
    )
    return fig


def fig_warming_delta() -> go.Figure:
    """Dumbbell: baseline 1985–1994 vs 2015–2024 mean temperature per state."""
    recent = (
        df.filter(pl.col("CVE_ENT") != 0)
        .filter(pl.col("anio").is_between(2015, 2024))
        .group_by("ENTIDAD")
        .agg(pl.col("MEDIA").mean().alias("recent"))
    )
    result = (
        _baseline.filter(pl.col("ENTIDAD") != "Nacional")
        .join(recent, on="ENTIDAD")
        .with_columns((pl.col("recent") - pl.col("baseline_media")).round(2).alias("delta"))
        .sort("delta")
    )
    states_list = result["ENTIDAD"].to_list()
    baselines   = result["baseline_media"].to_list()
    recents     = result["recent"].to_list()
    deltas      = result["delta"].to_list()
    colors      = ["#E84855" if d >= 2 else "#F4A261" if d >= 1 else "#3BB273" for d in deltas]

    x_lines, y_lines = [], []
    for b, r, s in zip(baselines, recents, states_list):
        x_lines += [b, r, None]
        y_lines += [s, s, None]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_lines, y=y_lines,
        mode="lines",
        line=dict(color="#475569", width=1.5),
        showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=baselines, y=states_list,
        mode="markers", name="1985–1994",
        marker=dict(color="#64748B", size=10, symbol="circle-open",
                    line=dict(color="#64748B", width=2)),
        hovertemplate="<b>%{y}</b><br>Base 1985–1994: %{x:.2f}°C<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=recents, y=states_list,
        mode="markers", name="2015–2024",
        marker=dict(color=colors, size=10),
        customdata=deltas,
        hovertemplate="<b>%{y}</b><br>Rec. 2015–2024: %{x:.2f}°C<br>Δ: +%{customdata:.2f}°C<extra></extra>",
    ))
    # Callout: most and least warmed
    slp_row = result.filter(pl.col("ENTIDAD") == "San Luis Potosí")
    gro_row = result.filter(pl.col("ENTIDAD") == "Guerrero")
    if len(slp_row) > 0:
        fig.add_annotation(
            x=float(slp_row["recent"][0]), y="San Luis Potosí",
            text="<b>+3.3°C</b> — mayor calentamiento",
            font=dict(color="#E84855", size=11),
            arrowcolor="#64748B", ax=60, ay=0,
            xanchor="left",
        )
    if len(gro_row) > 0:
        fig.add_annotation(
            x=float(gro_row["recent"][0]), y="Guerrero",
            text="<b>+0.4°C</b> — menor calentamiento",
            font=dict(color="#3BB273", size=11),
            arrowcolor="#64748B", ax=60, ay=0,
            xanchor="left",
        )
    fig.update_layout(
        title=dict(
            text="<b>San Luis Potosí calentó +3.3°C; Guerrero, solo +0.4°C — una brecha de 7.5×</b>"
                 "<br><sup style='color:#94A3B8'>Temperatura media por estado: 1985–1994 → 2015–2024</sup>",
        ),
        height=max(380, len(states_list) * 22 + 120),
        xaxis=dict(gridcolor="#334155", ticksuffix="°C", title="Temperatura media (°C)"),
        yaxis=dict(gridcolor="rgba(0,0,0,0)"),
        legend=dict(orientation="h", y=1.05, x=0, bgcolor="rgba(0,0,0,0)"),
        **{k: v for k, v in CHART_LAYOUT.items() if k not in ("xaxis", "yaxis", "margin")},
        margin=dict(t=70, b=40, l=10, r=200),
    )
    return fig


def fig_prec_trend(d: pl.DataFrame) -> go.Figure:
    anual = (
        d.filter(pl.col("anio").is_between(1985, 2025))
        .group_by("anio")
        .agg(pl.col("PRECIPITACION").sum().round(1).alias("prec"))
        .sort("anio")
    )
    years = anual["anio"].to_list()
    precs = anual["prec"].to_list()
    avg   = sum(precs) / len(precs)
    colors = ["#2E86AB" if p >= avg else "#F4A261" for p in precs]
    fig = go.Figure()
    fig.add_hrect(y0=avg - 30, y1=avg + 30,
                  fillcolor="rgba(59,178,115,0.06)", line_width=0)
    fig.add_hline(y=avg, line=dict(color="#3BB273", width=1.5, dash="dash"),
                  annotation_text=f"Promedio histórico {avg:.0f}mm",
                  annotation_font_color="#3BB273",
                  annotation_position="top right")
    fig.add_trace(go.Bar(
        x=years, y=precs,
        marker_color=colors,
        hovertemplate="<b>%{x}</b><br>Precipitación: %{y:.0f}mm<extra></extra>",
    ))
    # Annotate 2023 (driest) and 2010 (wettest)
    prec_2023 = anual.filter(pl.col("anio") == 2023)["prec"]
    prec_2010 = anual.filter(pl.col("anio") == 2010)["prec"]
    if len(prec_2023) > 0:
        fig.add_annotation(
            x=2023, y=float(prec_2023[0]),
            text="<b>2023</b><br>590mm<br>más seco",
            font=dict(color="#F4A261", size=11),
            arrowcolor="#64748B", ax=0, ay=40,
            yanchor="bottom",
        )
    if len(prec_2010) > 0:
        fig.add_annotation(
            x=2010, y=float(prec_2010[0]),
            text="<b>2010</b><br>962mm<br>más lluvioso",
            font=dict(color="#2E86AB", size=11),
            arrowcolor="#64748B", ax=0, ay=-40,
            yanchor="top",
        )
    fig.update_layout(
        title=dict(
            text="<b>Sin tendencia a largo plazo, pero alta variabilidad: 2023 fue 24% más seco que el promedio</b>"
                 "<br><sup style='color:#94A3B8'>Precipitación anual total (mm) — promedio histórico 1985–2025</sup>",
        ),
        height=400,
        xaxis=dict(gridcolor="#334155", title="Año", dtick=5),
        yaxis=dict(gridcolor="#334155", ticksuffix="mm"),
        showlegend=False,
        **{k: v for k, v in CHART_LAYOUT.items() if k not in ("xaxis", "yaxis", "margin")},
        margin=dict(t=70, b=40, l=10, r=10),
    )
    return fig


def fig_prec_seasonality(d: pl.DataFrame) -> go.Figure:
    seasonal = (
        d.group_by("mes")
        .agg(pl.col("PRECIPITACION").mean().round(1))
        .sort("mes")
    )
    meses = [MES_NOMBRES[m - 1] for m in seasonal["mes"].to_list()]
    precs = seasonal["PRECIPITACION"].to_list()
    max_p = max(precs)
    colors = [
        f"rgba(46,134,171,{0.4 + 0.6 * (p / max_p):.2f})" for p in precs
    ]
    fig = go.Figure(go.Bar(
        x=meses, y=precs,
        marker_color=colors,
        text=[f"{p:.0f}" for p in precs],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Precipitación: %{y:.1f}mm<extra></extra>",
    ))
    fig.update_layout(
        title=dict(
            text="<b>Jun–Sep concentra el 70% de la lluvia anual</b>"
                 "<br><sup style='color:#94A3B8'>Precipitación mensual promedio (mm) — todo el período</sup>",
        ),
        height=360,
        xaxis=dict(gridcolor="#334155"),
        yaxis=dict(gridcolor="#334155", ticksuffix="mm"),
        showlegend=False,
        **{k: v for k, v in CHART_LAYOUT.items() if k not in ("xaxis", "yaxis", "margin")},
        margin=dict(t=60, b=40, l=10, r=10),
    )
    return fig


def fig_prec_state_ranking() -> go.Figure:
    """Static: states ranked by avg monthly precipitation."""
    agg = (
        df.filter(pl.col("CVE_ENT") != 0)
        .group_by("ENTIDAD")
        .agg(pl.col("PRECIPITACION").mean().round(1).alias("avg"))
        .sort("avg", descending=True)
    )
    states = agg["ENTIDAD"].to_list()
    avgs   = agg["avg"].to_list()
    max_p  = max(avgs)
    colors = [f"rgba(46,134,171,{0.35 + 0.65 * (p / max_p):.2f})" for p in avgs]
    fig = go.Figure(go.Bar(
        x=avgs, y=states, orientation="h",
        marker_color=colors,
        text=[f"{p:.0f}mm" for p in avgs],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Precipitación media: %{x:.1f}mm/mes<extra></extra>",
    ))
    fig.update_layout(
        title=dict(
            text="<b>Tabasco recibe 12× más lluvia mensual que Baja California (185mm vs 15mm)</b>"
                 "<br><sup style='color:#94A3B8'>Precipitación media mensual por estado — promedio histórico</sup>",
        ),
        height=max(340, len(states) * 28 + 100),
        xaxis=dict(gridcolor="#334155", ticksuffix="mm"),
        yaxis=dict(gridcolor="rgba(0,0,0,0)", autorange="reversed"),
        **{k: v for k, v in CHART_LAYOUT.items() if k not in ("xaxis", "yaxis", "margin")},
        margin=dict(t=70, b=40, l=10, r=80),
    )
    return fig


def fig_seasonal_shift() -> go.Figure:
    """Static: monthly precipitation profiles across 3 decades to show the rainy-season shift."""
    nat = df.filter(pl.col("CVE_ENT") == 0)
    periods = [("1985–1994", 1985, 1994, "#64748B"), ("2000–2009", 2000, 2009, "#F4A261"), ("2015–2024", 2015, 2024, "#2E86AB")]
    fig = go.Figure()
    for label, y0, y1, color in periods:
        seas = (
            nat.filter(pl.col("anio").is_between(y0, y1))
            .group_by("mes").agg(pl.col("PRECIPITACION").mean().round(1).alias("p"))
            .sort("mes")
        )
        meses = [MES_NOMBRES[m - 1] for m in seas["mes"].to_list()]
        precs = seas["p"].to_list()
        fig.add_trace(go.Scatter(
            x=meses, y=precs,
            mode="lines+markers",
            name=label,
            line=dict(color=color, width=2.5),
            marker=dict(color=color, size=7),
            hovertemplate=f"<b>{label}</b><br>%{{x}}: %{{y:.0f}}mm<extra></extra>",
        ))
    # Mark the shift: July declined, September rose
    fig.add_annotation(
        x="Jul", y=140.3,
        text="Jul: 140→117mm<br>−23mm (−16%)",
        font=dict(color="#E84855", size=11),
        arrowcolor="#64748B", ax=50, ay=-30,
        bgcolor="rgba(30,41,59,0.85)", bordercolor="#475569",
    )
    fig.add_annotation(
        x="Sep", y=137.0,
        text="Sep: ahora el<br>mes más lluvioso",
        font=dict(color="#2E86AB", size=11),
        arrowcolor="#64748B", ax=-60, ay=-35,
        bgcolor="rgba(30,41,59,0.85)", bordercolor="#475569",
    )
    fig.update_layout(
        title=dict(
            text="<b>El pico de lluvias se desplazó de julio a septiembre — la temporada lluviosa llega más tarde</b>"
                 "<br><sup style='color:#94A3B8'>Precipitación mensual promedio (mm) — comparación entre tres décadas</sup>",
        ),
        height=400,
        xaxis=dict(gridcolor="#334155", title="Mes"),
        yaxis=dict(gridcolor="#334155", ticksuffix="mm"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font_color="#94A3B8",
                    orientation="h", y=1.05, x=0),
        **{k: v for k, v in CHART_LAYOUT.items() if k not in ("xaxis", "yaxis", "margin")},
        margin=dict(t=70, b=40, l=10, r=10),
    )
    return fig


def compute_kpis(d: pl.DataFrame, entidad: str):
    recent = d.filter(pl.col("anio").is_between(2015, 2024))
    media_r  = float(recent["MEDIA"].mean())
    maxima_r = float(recent["MAXIMA"].mean())

    ann = d.filter(pl.col("anio").is_between(1985, 2024)).group_by("anio").agg(
        pl.col("PRECIPITACION").sum().alias("prec")
    )
    prec_hist   = float(ann.filter(pl.col("anio") < 2015)["prec"].mean())
    prec_recent = float(ann.filter(pl.col("anio") >= 2015)["prec"].mean())

    base = _baseline.filter(pl.col("ENTIDAD") == entidad)
    if len(base) > 0:
        delta_media  = media_r - float(base["baseline_media"][0])
        delta_maxima = maxima_r - float(base["baseline_maxima"][0])
    else:
        delta_media = delta_maxima = float("nan")
    return media_r, maxima_r, prec_recent, prec_hist, delta_media, delta_maxima


# ── Pre-render static charts ──────────────────────────────────────────────────

_fig_warming        = fig_warming_delta()
_fig_prec_ranking   = fig_prec_state_ranking()
_fig_seasonal_shift = fig_seasonal_shift()


# ── Layout ────────────────────────────────────────────────────────────────────

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP],
           title="Temperaturas y Lluvias México")

estado_options = (
    [{"label": "Nacional", "value": "Nacional"}]
    + [{"label": e, "value": e} for e in ESTADOS]
)

app.layout = html.Div(
    style={"backgroundColor": "#0F172A", "minHeight": "100vh", "padding": "24px"},
    children=[
        html.H2("Temperaturas y Precipitaciones — México 1985–2025",
                style={"color": "#F8FAFC", "fontWeight": "700", "marginBottom": "4px"}),
        html.P("Temperatura mínima, media y máxima, y precipitación mensual por estado",
               style={"color": "#94A3B8", "marginBottom": "24px"}),

        # Filter
        dbc.Row(style={"marginBottom": "20px"}, children=[
            dbc.Col([
                html.Label("Estado / Región", style={"color": "#94A3B8", "fontSize": "12px"}),
                dcc.Dropdown(
                    id="estado-filter",
                    options=estado_options,
                    value="Nacional",
                    clearable=False,
                    style={"backgroundColor": "#1E293B", "color": "#CBD5E1",
                           "border": "1px solid #334155"},
                ),
            ], md=4),
        ]),

        # KPI row
        dbc.Row(id="kpi-row", className="g-3", style={"marginBottom": "20px"}),

        # Tabs
        dcc.Tabs(style={"marginBottom": "0"}, children=[

            dcc.Tab(label="Temperatura", style=TAB_STYLE, selected_style=TAB_SEL, children=[
                html.Div(style={"paddingTop": "20px"}, children=[
                    dbc.Row(className="g-3", style={"marginBottom": "20px"}, children=[
                        dbc.Col(dcc.Graph(id="graph-temp-trend"), md=7),
                        dbc.Col(dcc.Graph(id="graph-temp-seasonal"), md=5),
                    ]),
                    dbc.Row(className="g-3", children=[
                        dbc.Col(dcc.Graph(id="graph-warming-delta"), md=12),
                    ]),
                ]),
            ]),

            dcc.Tab(label="Precipitación", style=TAB_STYLE, selected_style=TAB_SEL, children=[
                html.Div(style={"paddingTop": "20px"}, children=[
                    dbc.Row(className="g-3", style={"marginBottom": "20px"}, children=[
                        dbc.Col(dcc.Graph(id="graph-prec-trend"), md=7),
                        dbc.Col(dcc.Graph(id="graph-prec-seasonal"), md=5),
                    ]),
                    dbc.Row(className="g-3", style={"marginBottom": "20px"}, children=[
                        dbc.Col(dcc.Graph(id="graph-seasonal-shift"), md=12),
                    ]),
                    dbc.Row(className="g-3", children=[
                        dbc.Col(dcc.Graph(id="graph-prec-ranking"), md=12),
                    ]),
                ]),
            ]),
        ]),
    ],
)


# ── Callbacks ─────────────────────────────────────────────────────────────────

@app.callback(
    Output("kpi-row", "children"),
    Output("graph-temp-trend", "figure"),
    Output("graph-temp-seasonal", "figure"),
    Output("graph-prec-trend", "figure"),
    Output("graph-prec-seasonal", "figure"),
    Input("estado-filter", "value"),
)
def update_all(estado: str):
    d = df.filter(pl.col("ENTIDAD") == estado)
    media, maxima, prec_recent, prec_hist, delta_media, delta_maxima = compute_kpis(d, estado)

    sign = lambda v: ("+" if v >= 0 else "") + f"{v:.2f}°C"
    prec_pct = (prec_recent - prec_hist) / prec_hist * 100 if prec_hist else float("nan")
    prec_sign = "+" if prec_pct >= 0 else ""

    kpis = [
        kpi_card(
            "Temp. media (2015–2024)", f"{media:.1f}°C", "#F4A261",
            delta=f"{sign(delta_media)} vs 1985–1994",
            delta_color="#E84855" if delta_media >= 2 else "#F4A261" if delta_media >= 1 else "#3BB273",
        ),
        kpi_card(
            "Temp. máxima (2015–2024)", f"{maxima:.1f}°C", "#E84855",
            delta=f"{sign(delta_maxima)} vs 1985–1994",
            delta_color="#E84855" if delta_maxima >= 2 else "#F4A261" if delta_maxima >= 1 else "#3BB273",
        ),
        kpi_card(
            "Precipitación anual (2015–2024)", f"{prec_recent:.0f}mm", "#2E86AB",
            delta=f"{prec_sign}{prec_pct:.1f}% vs 1985–2014",
            delta_color="#E84855" if prec_pct < -10 else "#F4A261" if prec_pct < 0 else "#3BB273",
        ),
        kpi_card(
            "Calentamiento total vs 1985–1994",
            f"{'+'if delta_media>=0 else ''}{delta_media:.2f}°C",
            "#E84855" if delta_media >= 2 else "#F4A261" if delta_media >= 1 else "#3BB273",
        ),
    ]
    return (
        kpis,
        fig_temp_trend(d),
        fig_temp_seasonality(d),
        fig_prec_trend(d),
        fig_prec_seasonality(d),
    )


@app.callback(
    Output("graph-warming-delta", "figure"),
    Output("graph-prec-ranking", "figure"),
    Output("graph-seasonal-shift", "figure"),
    Input("estado-filter", "value"),
)
def update_static(_):
    return _fig_warming, _fig_prec_ranking, _fig_seasonal_shift


if __name__ == "__main__":
    app.run(debug=True, port=8056)
