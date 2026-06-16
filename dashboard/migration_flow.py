import polars as pl
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc

# ── CONAPO Migration Intensity Index ─────────────────────────────────────────
_IIM_BASE = "data/conapo/intensidad_migratoria"
_IIM_FILES = [
    (f"{_IIM_BASE}/04_iim_mex_eeuu_2000_municipio.csv", 2000),
    (f"{_IIM_BASE}/05_iim_mex_eeuu_2010_municipio.csv", 2010),
    (f"{_IIM_BASE}/06_iim_mex_eeuu_2020_municipio.csv", 2020),
]

ESTADOS = {
    1: "Aguascalientes", 2: "Baja California", 3: "Baja California Sur",
    4: "Campeche", 5: "Coahuila", 6: "Colima", 7: "Chiapas",
    8: "Chihuahua", 9: "CDMX", 10: "Durango", 11: "Guanajuato",
    12: "Guerrero", 13: "Hidalgo", 14: "Jalisco", 15: "Estado de México",
    16: "Michoacán", 17: "Morelos", 18: "Nayarit", 19: "Nuevo León",
    20: "Oaxaca", 21: "Puebla", 22: "Querétaro", 23: "Quintana Roo",
    24: "San Luis Potosí", 25: "Sinaloa", 26: "Sonora", 27: "Tabasco",
    28: "Tamaulipas", 29: "Tlaxcala", 30: "Veracruz", 31: "Yucatán",
    32: "Zacatecas",
}

VALID_GRADES = ["muy alto", "alto", "medio", "bajo", "muy bajo"]

GRADE_COLORS = {
    "muy alto": "#E84855",
    "alto":     "#F4A261",
    "medio":    "#CBD5E1",
    "bajo":     "#2E86AB",
    "muy bajo": "#475569",
}


def _load_iim(path: str, year: int) -> pl.DataFrame:
    df = pl.read_csv(path)
    renames = {}
    if "tot_viv" in df.columns: renames["tot_viv"] = "viv_tot"
    if "iaim"    in df.columns: renames["iaim"]    = "iim_dp2"
    if "gaim"    in df.columns: renames["gaim"]    = "gim_dp2"
    if renames:
        df = df.rename(renames)
    return df.with_columns([
        pl.lit(year).cast(pl.Int32).alias("year"),
        pl.col("gim_dp2").str.to_lowercase().alias("gim_dp2"),
        pl.col("viv_tot").cast(pl.Float64).alias("viv_tot"),
    ])


iim_panel = pl.concat([_load_iim(p, y) for p, y in _IIM_FILES]).filter(
    pl.col("gim_dp2").is_in(VALID_GRADES)
)

# National grade distribution (% per year)
_grade_counts = (
    iim_panel.group_by(["year", "gim_dp2"])
    .agg(pl.len().alias("n"))
    .with_columns(
        (pl.col("n") / pl.col("n").sum().over("year") * 100).alias("pct")
    )
    .sort(["year", "gim_dp2"])
)

# Per-state % "alto"+"muy alto" for 2010 and 2020
_state_agg = (
    iim_panel.filter(pl.col("year").is_in([2010, 2020]))
    .with_columns(
        pl.col("gim_dp2").is_in(["alto", "muy alto"]).cast(pl.Int32).alias("high")
    )
    .group_by(["cve_ent", "year"])
    .agg(pl.col("high").mean().alias("pct_high"))
    .with_columns((pl.col("pct_high") * 100).alias("pct_high"))
)

# KPI values for IIM mini-cards
_muy_alto_2020 = int(
    iim_panel.filter((pl.col("year") == 2020) & (pl.col("gim_dp2") == "muy alto")).height
)
_muy_alto_2010 = int(
    iim_panel.filter((pl.col("year") == 2010) & (pl.col("gim_dp2") == "muy alto")).height
)

# ── Banxico — Remesas por entidad federativa ─────────────────────────────────
_BNX_FILE = "data/banxico/Consulta_20260615-174730348.csv"

def _load_banxico() -> pl.DataFrame:
    raw = pl.read_csv(_BNX_FILE, skip_rows=9, encoding="latin1")
    date_cols = [c for c in raw.columns if c.startswith("01/")]
    return (
        raw.rename({raw.columns[0]: "titulo"})
        .with_columns(pl.col("titulo").str.split(", ").list.last().alias("estado"))
        .select(["estado"] + date_cols)
        .unpivot(index="estado", variable_name="fecha", value_name="mdd")
        .with_columns([
            pl.col("mdd").cast(pl.Float64),
            pl.col("fecha").str.to_date("%d/%m/%Y").dt.year().alias("year"),
        ])
    )

_bnx_long = _load_banxico()

# Annual totals (complete years 2003–2025; filter partial 2026)
bnx_annual = (
    _bnx_long.filter(pl.col("year").is_between(2003, 2025))
    .group_by(["estado", "year"])
    .agg(pl.col("mdd").sum())
    .sort(["estado", "year"])
)

bnx_national = bnx_annual.filter(pl.col("estado") == "TOTAL").sort("year")
bnx_states   = bnx_annual.filter(pl.col("estado") != "TOTAL")

# KPI: total 2024
_bnx_2024_b = float(
    bnx_national.filter(pl.col("year") == 2024)["mdd"][0]
) / 1000  # billions

# ── Theme ─────────────────────────────────────────────────────────────────────
CARD_STYLE = {
    "background": "#1E293B", "border": "1px solid #334155",
    "borderRadius": "8px", "padding": "16px", "textAlign": "center",
}
TAB_STYLE = {"backgroundColor": "#0F172A", "color": "#94A3B8", "borderTop": "none"}
TAB_SEL = {
    "backgroundColor": "#1E293B", "color": "#F8FAFC",
    "borderTop": "2px solid #2E86AB", "fontWeight": "600",
}
FOCUS, CONTEXT = "#2E86AB", "#475569"


def _theme(fig: go.Figure, height: int = 420, **kwargs) -> go.Figure:
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#CBD5E1", height=height, **kwargs,
    )
    fig.update_xaxes(gridcolor="#334155")
    fig.update_yaxes(gridcolor="#334155")
    return fig


# ── Data loading (lazy for 138 MB bilateral file) ─────────────────────────────
_BI = "data/migration_flows/mig_bilateral.csv"
_UNI = "data/migration_flows/mig_unilateral.csv"

lf_bi = (
    pl.scan_csv(_BI)
    .filter(
        ((pl.col("Origin ISO") == "MEX") | (pl.col("Destination ISO") == "MEX"))
        & ~((pl.col("Origin ISO") == "MEX") & (pl.col("Destination ISO") == "MEX"))
    )
    .select(["Origin ISO", "Destination ISO", "Year", "stock_mean"])
)

# Two aggregation queries — collect_all scans the file once
q_emi = (
    lf_bi.filter(pl.col("Origin ISO") == "MEX")
    .group_by(["Destination ISO", "Year"])
    .agg(pl.col("stock_mean").sum())
)
q_imm = (
    lf_bi.filter(pl.col("Destination ISO") == "MEX")
    .group_by(["Origin ISO", "Year"])
    .agg(pl.col("stock_mean").sum())
)
emi_dest_yr, imm_orig_yr = pl.collect_all([q_emi, q_imm])

uni = pl.read_csv(_UNI).filter(pl.col("Country ISO") == "MEX").sort("Year")

# Pre-compute US share trend
_total_by_yr = emi_dest_yr.group_by("Year").agg(pl.col("stock_mean").sum().alias("total"))
_us_by_yr = emi_dest_yr.filter(pl.col("Destination ISO") == "USA").rename({"stock_mean": "us"}).select(["Year", "us"])
us_share_trend = (
    _total_by_yr.join(_us_by_yr, on="Year")
    .with_columns((pl.col("us") / pl.col("total") * 100).alias("share"))
    .sort("Year")
)

MIN_YEAR = int(uni["Year"].min())
MAX_YEAR = int(uni["Year"].max())

# ── KPI values ────────────────────────────────────────────────────────────────
def _val(col: str, year: int) -> float:
    rows = uni.filter(pl.col("Year") == year)[col]
    return float(rows[0]) if len(rows) > 0 and rows[0] is not None else float("nan")


emi_pop_2024 = _val("emi_pop", 2024)
imm_pop_2024 = _val("imm_pop", 2024)
imm_pop_1990 = _val("imm_pop", 1990)
net_2023 = _val("net", 2023)
us_share_2024 = float(us_share_trend.filter(pl.col("Year") == 2024)["share"][0])
us_share_1990 = float(us_share_trend.filter(pl.col("Year") == 1990)["share"][0])


# ── Figure factories ──────────────────────────────────────────────────────────
def fig_flows() -> go.Figure:
    d = uni.filter(pl.col("emi").is_not_null()).sort("Year")
    emi_2022 = float(d.filter(pl.col("Year") == 2022)["emi"][0])
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=d["Year"].to_list(), y=(d["emi"] / 1e3).to_list(),
        name="Emigración", mode="lines", line=dict(color="#E84855", width=2.5),
        hovertemplate="<b>%{x}</b>: %{y:.0f}k emi<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=d["Year"].to_list(), y=(d["imm"] / 1e3).to_list(),
        name="Inmigración", mode="lines", line=dict(color="#3BB273", width=2.5),
        hovertemplate="<b>%{x}</b>: %{y:.0f}k imm<extra></extra>",
    ))
    fig.add_vline(x=2020, line_dash="dot", line_color="#94A3B8",
                  annotation_text="2020: mínimo histórico\nemi (280k)",
                  annotation_font_color="#94A3B8", annotation_position="top left")
    fig.add_annotation(
        x=2022, y=emi_2022 / 1e3 + 40,
        text="<b>2022: récord histórico (765k)</b>",
        font=dict(color="#E84855", size=11), showarrow=True,
        arrowcolor="#94A3B8", ax=-60, ay=-30,
    )
    return _theme(fig,
        title=dict(text=(
            "<b>Emigración alcanzó récord histórico en 2022, 2.7× el mínimo de 2020</b>"
            "<br><sup style='color:#94A3B8'>Flujos anuales, miles de personas, 1990–2023</sup>"
        )),
        yaxis_title="Miles",
        legend=dict(orientation="h", y=-0.15, x=0),
        margin=dict(t=80, b=70, l=60, r=20),
    )


def fig_net() -> go.Figure:
    d = uni.filter(pl.col("net").is_not_null()).sort("Year")
    nets_k = (d["net"] / 1e3).to_list()
    colors = ["#3BB273" if v > 0 else "#E84855" for v in nets_k]
    fig = go.Figure()
    fig.add_vrect(x0=2015, x1=2020, fillcolor="rgba(59,178,115,0.08)", line_width=0)
    fig.add_hline(y=0, line_dash="dash", line_color="#64748B",
                  annotation_text="Equilibrio", annotation_font_color="#94A3B8",
                  annotation_position="right")
    fig.add_trace(go.Scatter(
        x=d["Year"].to_list(), y=nets_k,
        mode="lines+markers",
        line=dict(color="#2E86AB", width=2.5),
        marker=dict(color=colors, size=7),
        hovertemplate="<b>%{x}</b>: %{y:.0f}k<extra></extra>",
        showlegend=False,
    ))
    fig.add_annotation(x=2017, y=2, text="Era receptora\n2015–2020",
                       font=dict(color="#3BB273", size=10), showarrow=False)
    return _theme(fig,
        title=dict(text=(
            "<b>México fue receptor neto 2015–2020; revirtió a emigración neta en 2021</b>"
            "<br><sup style='color:#94A3B8'>Migración neta anual, miles de personas</sup>"
        )),
        yaxis_title="Miles",
        margin=dict(t=80, b=40, l=60, r=60),
    )


def fig_top_destinations(year: int, n: int = 15) -> go.Figure:
    year_data = emi_dest_yr.filter(pl.col("Year") == year)
    total_all = float(year_data["stock_mean"].sum())
    d = year_data.sort("stock_mean", descending=True).head(n)
    entities = d["Destination ISO"].to_list()
    values_m = (d["stock_mean"] / 1e6).to_list()
    colors = [FOCUS if e == "USA" else CONTEXT for e in entities]
    labels = [f"{v:.2f}M" if v >= 0.1 else f"{v*1000:.0f}k" for v in values_m]
    top3 = float(d.head(3)["stock_mean"].sum()) / total_all * 100
    fig = go.Figure(go.Bar(
        x=values_m, y=entities, orientation="h",
        marker_color=colors,
        text=labels, textposition="inside", insidetextanchor="middle",
        hovertemplate="<b>%{y}</b>: %{x:.3f}M<extra></extra>",
    ))
    return _theme(fig, height=max(300, n * 28 + 100),
        title=dict(text=(
            f"<b>{top3:.0f}% de los mexicanos en el exterior viven en solo 3 países ({year})</b>"
            f"<br><sup style='color:#94A3B8'>Stock migratorio, millones de personas</sup>"
        )),
        xaxis_title="Millones",
        yaxis=dict(autorange="reversed", gridcolor="rgba(0,0,0,0)"),
        margin=dict(t=80, b=40, l=60, r=20),
    )


def fig_us_share() -> go.Figure:
    d = us_share_trend.sort("Year")
    fig = go.Figure(go.Scatter(
        x=d["Year"].to_list(), y=d["share"].to_list(),
        mode="lines+markers", line=dict(color="#2E86AB", width=2.5),
        marker=dict(size=5),
        hovertemplate="<b>%{x}</b>: %{y:.1f}%<extra></extra>",
        showlegend=False,
    ))
    fig.update_yaxes(range=[90, 100], title_text="%")
    return _theme(fig, height=340,
        title=dict(text=(
            "<b>La dependencia de EUA bajó solo 3 puntos en 34 años (98.8% → 95.6%)</b>"
            "<br><sup style='color:#94A3B8'>% del stock emigrante total residente en EUA</sup>"
        )),
        margin=dict(t=80, b=40, l=60, r=20),
    )


_NON_US = ["CAN", "ESP", "DEU", "GTM", "FRA", "GBR"]
_NON_US_COLORS = {
    "CAN": "#2E86AB", "ESP": "#F4A261", "DEU": "#3BB273",
    "GTM": "#E84855", "FRA": "#94A3B8", "GBR": "#64748B",
}


def fig_non_us_growth() -> go.Figure:
    fig = go.Figure()
    for iso in _NON_US:
        sub = emi_dest_yr.filter(pl.col("Destination ISO") == iso).sort("Year")
        fig.add_trace(go.Scatter(
            x=sub["Year"].to_list(), y=(sub["stock_mean"] / 1e3).to_list(),
            name=iso, mode="lines+markers",
            line=dict(color=_NON_US_COLORS.get(iso, CONTEXT), width=1.8),
            marker=dict(size=4),
            hovertemplate=f"<b>{iso}</b> %{{x}}: %{{y:.1f}}k<extra></extra>",
        ))
    return _theme(fig, height=320,
        title=dict(text=(
            "<b>Canadá y España lideran destinos alternativos — pero suman < 2% del total</b>"
            "<br><sup style='color:#94A3B8'>Stock migratorio, miles de personas, excluyendo EUA</sup>"
        )),
        yaxis_title="Miles",
        legend=dict(orientation="h", y=-0.2, x=0),
        margin=dict(t=80, b=70, l=60, r=20),
    )


def fig_top_sources(year: int, n: int = 15) -> go.Figure:
    d = (
        imm_orig_yr.filter(pl.col("Year") == year)
        .sort("stock_mean", descending=True)
        .head(n)
    )
    entities = d["Origin ISO"].to_list()
    values_k = (d["stock_mean"] / 1e3).to_list()
    highlight = {"GTM", "HND", "VEN", "HTI"}
    colors = [FOCUS if e in highlight else CONTEXT for e in entities]
    fig = go.Figure(go.Bar(
        x=values_k, y=entities, orientation="h",
        marker_color=colors,
        text=[f"{v:.0f}k" for v in values_k],
        textposition="inside", insidetextanchor="middle",
        hovertemplate="<b>%{y}</b>: %{x:.1f}k<extra></extra>",
    ))
    return _theme(fig, height=max(300, n * 28 + 100),
        title=dict(text=(
            f"<b>La población inmigrante en México se cuadruplicó desde 1990 ({year})</b>"
            f"<br><sup style='color:#94A3B8'>Stock de inmigrantes, miles de personas</sup>"
        )),
        xaxis_title="Miles",
        yaxis=dict(autorange="reversed", gridcolor="rgba(0,0,0,0)"),
        margin=dict(t=80, b=40, l=60, r=20),
    )


_SURGE = ["HTI", "VEN", "HND", "GTM", "CUB"]
_SURGE_COLORS = {
    "HTI": "#E84855", "VEN": "#F4A261", "HND": "#2E86AB",
    "GTM": "#3BB273", "CUB": "#94A3B8",
}


def fig_surge_sources() -> go.Figure:
    fig = go.Figure()
    for iso in _SURGE:
        sub = imm_orig_yr.filter(pl.col("Origin ISO") == iso).sort("Year")
        fig.add_trace(go.Scatter(
            x=sub["Year"].to_list(), y=(sub["stock_mean"] / 1e3).to_list(),
            name=iso, mode="lines+markers",
            line=dict(color=_SURGE_COLORS.get(iso, CONTEXT), width=1.8),
            marker=dict(size=5),
            hovertemplate=f"<b>{iso}</b> %{{x}}: %{{y:.1f}}k<extra></extra>",
        ))
    fig.add_vline(x=2015, line_dash="dot", line_color="#94A3B8",
                  annotation_text="2015: aceleración",
                  annotation_font_color="#94A3B8", annotation_position="top left")
    return _theme(fig, height=420,
        title=dict(text=(
            "<b>Haití creció ×15 en 9 años (2015–2024); Venezuela ×8, Honduras ×7</b>"
            "<br><sup style='color:#94A3B8'>Stock de inmigrantes en México, miles de personas</sup>"
        )),
        yaxis_title="Miles",
        legend=dict(orientation="h", y=-0.15, x=0),
        margin=dict(t=80, b=70, l=60, r=20),
    )


# ── Banxico figure factories ──────────────────────────────────────────────────
def fig_bnx_trend() -> go.Figure:
    d = bnx_national.sort("year")
    years = d["year"].to_list()
    vals_b = (d["mdd"] / 1000).to_list()  # billions
    val_2003 = vals_b[0]
    val_2024 = float(bnx_national.filter(pl.col("year") == 2024)["mdd"][0]) / 1000

    fig = go.Figure(go.Scatter(
        x=years, y=vals_b,
        mode="lines+markers",
        line=dict(color="#2E86AB", width=2.5),
        marker=dict(size=4),
        hovertemplate="<b>%{x}</b>: $%{y:.1f}B<extra></extra>",
        showlegend=False,
    ))
    # 2008-2009 dip
    fig.add_vrect(x0=2008, x1=2010, fillcolor="rgba(232,72,85,0.08)", line_width=0,
                  annotation_text="Crisis 2008", annotation_font_color="#94A3B8",
                  annotation_position="top left")
    # 2020 COVID dip
    fig.add_vline(x=2020, line_dash="dot", line_color="#94A3B8",
                  annotation_text="2020: COVID", annotation_font_color="#94A3B8",
                  annotation_position="top right")
    # 2024 annotation
    fig.add_annotation(
        x=2024, y=val_2024, text=f"<b>2024: ${val_2024:.1f}B</b>",
        font=dict(color="#2E86AB", size=11), showarrow=True,
        arrowcolor="#94A3B8", ax=-50, ay=-30,
    )
    return _theme(fig, height=320,
        title=dict(text=(
            f"<b>Las remesas se multiplicaron 4× en 20 años: de ${val_2003:.1f}B (2003) a ${val_2024:.1f}B (2024)</b>"
            "<br><sup style='color:#94A3B8'>Remesas familiares anuales, miles de millones de USD · Banxico CA79</sup>"
        )),
        yaxis_title="Miles de millones USD",
        margin=dict(t=80, b=40, l=70, r=20),
    )


def fig_bnx_states() -> go.Figure:
    d = bnx_states.filter(pl.col("year") == 2024).sort("mdd", descending=True)
    states = d["estado"].to_list()
    vals_b = (d["mdd"] / 1000).to_list()
    top3 = {"Guanajuato", "Michoacán", "Jalisco"}
    colors = [FOCUS if s in top3 else CONTEXT for s in states]
    pct_top3 = sum(vals_b[:3]) / sum(vals_b) * 100

    fig = go.Figure(go.Bar(
        x=vals_b, y=states, orientation="h",
        marker_color=colors,
        text=[f"${v:.1f}B" for v in vals_b],
        textposition="inside", insidetextanchor="middle",
        hovertemplate="<b>%{y}</b>: $%{x:.2f}B<extra></extra>",
    ))
    return _theme(fig, height=max(300, len(states) * 24 + 80),
        title=dict(text=(
            f"<b>Guanajuato, Michoacán y Jalisco captaron {pct_top3:.0f}% de las remesas en 2024</b>"
            "<br><sup style='color:#94A3B8'>Remesas anuales por entidad federativa, miles de millones USD · Banxico 2024</sup>"
        )),
        xaxis_title="Miles de millones USD",
        yaxis=dict(autorange="reversed", gridcolor="rgba(0,0,0,0)"),
        margin=dict(t=80, b=40, l=140, r=20),
    )


# ── CONAPO figure factories ───────────────────────────────────────────────────
def fig_grade_dist() -> go.Figure:
    fig = go.Figure()
    years = [2000, 2010, 2020]
    for grade in VALID_GRADES:
        x_vals, y_vals, n_vals = [], [], []
        for yr in years:
            row = _grade_counts.filter(
                (pl.col("year") == yr) & (pl.col("gim_dp2") == grade)
            )
            pct = float(row["pct"][0]) if row.height > 0 else 0.0
            n   = int(row["n"][0])     if row.height > 0 else 0
            x_vals.append(pct)
            y_vals.append(str(yr))
            n_vals.append(n)
        label = grade.title()
        fig.add_trace(go.Bar(
            x=x_vals, y=y_vals, orientation="h", name=label,
            marker_color=GRADE_COLORS[grade],
            text=[f"{v:.1f}%" for v in x_vals],
            textposition="inside", insidetextanchor="middle",
            customdata=n_vals,
            hovertemplate=f"<b>{label}</b>: %{{x:.1f}}%  (n=%{{customdata[0]:,}})<extra></extra>",
        ))
    return _theme(fig, height=220,
        title=dict(text=(
            "<b>Los municipios de muy alta intensidad cayeron 22% entre 2010 y 2020 (177→137)</b>"
            "<br><sup style='color:#94A3B8'>% de municipios por grado de intensidad migratoria hacia EE.UU., CONAPO</sup>"
        )),
        barmode="stack",
        xaxis=dict(range=[0, 100], visible=False, gridcolor="rgba(0,0,0,0)"),
        yaxis=dict(gridcolor="rgba(0,0,0,0)"),
        legend=dict(orientation="h", y=-0.25, x=0),
        margin=dict(t=80, b=70, l=50, r=20),
    )


def fig_state_intensity() -> go.Figure:
    st10 = _state_agg.filter(pl.col("year") == 2010).sort("cve_ent")
    st20 = _state_agg.filter(pl.col("year") == 2020).sort("cve_ent")

    # Build aligned table sorted by 2020 value
    joined = (
        st20.rename({"pct_high": "p20"})
        .join(st10.rename({"pct_high": "p10"}).drop("year"), on="cve_ent", how="left")
        .drop("year")
        .with_columns(
            pl.col("cve_ent").cast(pl.Utf8)
            .replace({str(k): v for k, v in ESTADOS.items()})
            .alias("estado")
        )
        .sort("p20")
    )

    states = joined["estado"].to_list()
    p10s   = joined["p10"].fill_null(0).to_list()
    p20s   = joined["p20"].to_list()
    deltas = [p20 - p10 for p20, p10 in zip(p20s, p10s)]

    # Connector lines
    x_lines, y_lines = [], []
    for p10, p20, st in zip(p10s, p20s, states):
        x_lines += [p10, p20, None]
        y_lines += [st, st, None]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_lines, y=y_lines, mode="lines",
        line=dict(color="#334155", width=1.5),
        showlegend=False, hoverinfo="skip",
    ))
    # 2010 dots (open)
    fig.add_trace(go.Scatter(
        x=p10s, y=states, mode="markers", name="2010",
        marker=dict(color="#64748B", size=9, symbol="circle-open",
                    line=dict(color="#64748B", width=2)),
        hovertemplate="<b>%{y}</b> 2010: %{x:.1f}%<extra></extra>",
    ))
    # 2020 dots (filled, direction-colored)
    dot_colors = ["#E84855" if d > 0 else "#3BB273" for d in deltas]
    fig.add_trace(go.Scatter(
        x=p20s, y=states, mode="markers", name="2020",
        marker=dict(color=dot_colors, size=9),
        customdata=deltas,
        hovertemplate="<b>%{y}</b> 2020: %{x:.1f}%  Δ: %{customdata:.1f} pp<extra></extra>",
    ))

    n_up = sum(1 for d in deltas if d > 0)
    n_dn = sum(1 for d in deltas if d <= 0)
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
                              marker=dict(color="#E84855", size=9),
                              name=f"▲ Aumentó ({n_up})"))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
                              marker=dict(color="#3BB273", size=9),
                              name=f"▼ Disminuyó ({n_dn})"))

    return _theme(fig, height=max(300, 32 * 22 + 80),
        title=dict(text=(
            "<b>Zacatecas lidera la intensidad migratoria; la mayoría de estados la redujo 2010–2020</b>"
            "<br><sup style='color:#94A3B8'>% de municipios con intensidad Alta o Muy alta, CONAPO 2010 vs 2020</sup>"
        )),
        xaxis=dict(title_text="% municipios", ticksuffix="%", gridcolor="#334155"),
        yaxis=dict(gridcolor="rgba(0,0,0,0)"),
        legend=dict(orientation="h", y=-0.08, x=0),
        margin=dict(t=80, b=60, l=150, r=20),
    )


def fig_top_mun_remesas(year: int) -> go.Figure:
    d = (
        iim_panel.filter(pl.col("year") == year)
        .sort("viv_rem", descending=True)
        .head(25)
        .with_columns(
            (pl.col("nom_mun") + " (" +
             pl.col("cve_ent").cast(pl.Utf8).replace({str(k): v for k, v in ESTADOS.items()}) +
             ")"
            ).alias("label")
        )
        .sort("viv_rem")  # ascending so highest is at top in horizontal bar
    )
    labels  = d["label"].to_list()
    values  = d["viv_rem"].to_list()

    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker_color=FOCUS,
        text=[f"{v:.1f}%" for v in values],
        textposition="inside", insidetextanchor="middle",
        hovertemplate="<b>%{y}</b>: %{x:.1f}% de viviendas<extra></extra>",
    ))
    return _theme(fig, height=max(300, 25 * 28 + 100),
        title=dict(text=(
            f"<b>Los 25 municipios con mayor proporción de viviendas que reciben remesas ({year})</b>"
            f"<br><sup style='color:#94A3B8'>% de viviendas con remesas de EE.UU., CONAPO Índice de Intensidad Migratoria</sup>"
        )),
        xaxis=dict(title_text="%", ticksuffix="%"),
        yaxis=dict(gridcolor="rgba(0,0,0,0)"),
        margin=dict(t=80, b=40, l=230, r=20),
    )


# ── KPI card ──────────────────────────────────────────────────────────────────
def kpi_card(label: str, value: str, delta: str, color: str) -> dbc.Card:
    return dbc.Card(
        dbc.CardBody([
            html.P(label, style={"color": "#94A3B8", "fontSize": "0.8rem", "marginBottom": "4px"}),
            html.H3(value, style={"color": "#F8FAFC", "margin": "0", "fontSize": "1.5rem"}),
            html.Span(delta, style={"color": color, "fontSize": "0.82rem"}),
        ]),
        style=CARD_STYLE,
    )


# ── Layout ────────────────────────────────────────────────────────────────────
app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
app.title = "Migración México"

_slider_marks = {y: str(y) for y in range(MIN_YEAR, MAX_YEAR + 1, 5)}

kpis = dbc.Row([
    dbc.Col(kpi_card(
        "Mexicanos en el exterior (2024)",
        f"{emi_pop_2024/1e6:.1f}M",
        "vs 5.4M en 1990 (+110%)", "#3BB273",
    ), md=True),
    dbc.Col(kpi_card(
        "Inmigrantes en México (2024)",
        f"{imm_pop_2024/1e6:.2f}M",
        f"vs 454k en 1990 (×{imm_pop_2024/imm_pop_1990:.1f})", "#3BB273",
    ), md=True),
    dbc.Col(kpi_card(
        "Migración neta (2023)",
        f"{net_2023/1e3:+.0f}k",
        "▼ Revirtió a emigración neta en 2021", "#E84855",
    ), md=True),
    dbc.Col(kpi_card(
        "% emigrantes en EUA (2024)",
        f"{us_share_2024:.1f}%",
        f"vs {us_share_1990:.1f}% en 1990 (−{us_share_1990-us_share_2024:.1f} pp)", "#F4A261",
    ), md=True),
    dbc.Col(kpi_card(
        "Remesas familiares (2024)",
        f"${_bnx_2024_b:.0f}B USD",
        f"vs $15B en 2003 (×{_bnx_2024_b/15.1:.1f}) · Banxico CA79", "#2E86AB",
    ), md=True),
], className="g-3 mb-4")

app.layout = html.Div(
    style={"backgroundColor": "#0F172A", "minHeight": "100vh", "padding": "24px"},
    children=[
        html.H2("Flujos Migratorios de México 1990–2024",
                style={"color": "#F8FAFC", "marginBottom": "4px"}),
        html.P(
            "Estimaciones de stock y flujos migratorios · Fuente: ThGaskin/Migration_flows (HuggingFace)",
            style={"color": "#64748B", "marginBottom": "24px", "fontSize": "0.85rem"},
        ),
        kpis,
        dcc.Tabs([
            dcc.Tab(label="Balance Migratorio", style=TAB_STYLE, selected_style=TAB_SEL, children=[
                dbc.Row([
                    dbc.Col(dcc.Graph(id="fig-flows", figure=fig_flows()), md=6),
                    dbc.Col(dcc.Graph(id="fig-net", figure=fig_net()), md=6),
                ], className="g-3 mt-3"),
            ]),
            dcc.Tab(label="Mexicanos en el Mundo", style=TAB_STYLE, selected_style=TAB_SEL, children=[
                dbc.Row([
                    dbc.Col([
                        html.Label("Año:", style={"color": "#94A3B8", "fontSize": "0.85rem"}),
                        dcc.Slider(
                            id="slider-mundo", min=MIN_YEAR, max=MAX_YEAR, step=1,
                            value=MAX_YEAR, marks=_slider_marks,
                            tooltip={"placement": "bottom"},
                        ),
                    ], md=8, className="mt-3 mb-1"),
                ]),
                dbc.Row([
                    dbc.Col(dcc.Graph(id="fig-top-dests"), md=5),
                    dbc.Col([
                        dcc.Graph(id="fig-us-share", figure=fig_us_share()),
                        dcc.Graph(id="fig-non-us", figure=fig_non_us_growth()),
                    ], md=7),
                ], className="g-3"),
                dbc.Row([
                    dbc.Col(dcc.Graph(figure=fig_bnx_trend()), md=12),
                ], className="g-3 mt-2"),
            ]),
            dcc.Tab(label="Migrantes en México", style=TAB_STYLE, selected_style=TAB_SEL, children=[
                dbc.Row([
                    dbc.Col([
                        html.Label("Año:", style={"color": "#94A3B8", "fontSize": "0.85rem"}),
                        dcc.Slider(
                            id="slider-mex", min=MIN_YEAR, max=MAX_YEAR, step=1,
                            value=MAX_YEAR, marks=_slider_marks,
                            tooltip={"placement": "bottom"},
                        ),
                    ], md=8, className="mt-3 mb-1"),
                ]),
                dbc.Row([
                    dbc.Col(dcc.Graph(id="fig-top-sources"), md=5),
                    dbc.Col(dcc.Graph(id="fig-surge", figure=fig_surge_sources()), md=7),
                ], className="g-3"),
            ]),
            dcc.Tab(label="Origen Territorial", style=TAB_STYLE, selected_style=TAB_SEL, children=[
                dbc.Row([
                    dbc.Col(dbc.Card(dbc.CardBody([
                        html.P("Municipios Muy alta intensidad (2020)",
                               style={"color": "#94A3B8", "fontSize": "0.8rem", "marginBottom": "4px"}),
                        html.H3(str(_muy_alto_2020),
                                style={"color": "#F8FAFC", "margin": "0", "fontSize": "1.5rem"}),
                        html.Span(f"vs {_muy_alto_2010} en 2010 (−{_muy_alto_2010 - _muy_alto_2020})",
                                  style={"color": "#E84855", "fontSize": "0.82rem"}),
                    ]), style=CARD_STYLE), md=4),
                    dbc.Col(dbc.Card(dbc.CardBody([
                        html.P("Municipios activos (con migración a EE.UU., 2020)",
                               style={"color": "#94A3B8", "fontSize": "0.8rem", "marginBottom": "4px"}),
                        html.H3(
                            str(iim_panel.filter(pl.col("year") == 2020).height),
                            style={"color": "#F8FAFC", "margin": "0", "fontSize": "1.5rem"},
                        ),
                        html.Span("de 2,469 municipios del país",
                                  style={"color": "#94A3B8", "fontSize": "0.82rem"}),
                    ]), style=CARD_STYLE), md=4),
                    dbc.Col(dbc.Card(dbc.CardBody([
                        html.P("Estados con mayor % municipios de alta intensidad (2020)",
                               style={"color": "#94A3B8", "fontSize": "0.8rem", "marginBottom": "4px"}),
                        html.H3("Zacatecas · Nayarit · Durango",
                                style={"color": "#F8FAFC", "margin": "0", "fontSize": "1.1rem"}),
                        html.Span("corredor migratorio tradicional",
                                  style={"color": "#94A3B8", "fontSize": "0.82rem"}),
                    ]), style=CARD_STYLE), md=4),
                ], className="g-3 mt-3 mb-3"),
                dbc.Row([
                    dbc.Col(dcc.Graph(figure=fig_grade_dist()), md=5),
                    dbc.Col(dcc.Graph(figure=fig_state_intensity()), md=7),
                ], className="g-3"),
                dbc.Row([
                    dbc.Col([
                        html.Label("Año del censo:", style={"color": "#94A3B8", "fontSize": "0.85rem", "marginTop": "16px"}),
                        dcc.RadioItems(
                            id="radio-iim-year",
                            options=[{"label": str(y), "value": y} for y in [2000, 2010, 2020]],
                            value=2020,
                            inline=True,
                            style={"color": "#CBD5E1", "fontSize": "0.9rem", "marginLeft": "12px"},
                        ),
                    ], md=4, className="mt-3 mb-2"),
                ]),
                dbc.Row([
                    dbc.Col(dcc.Graph(id="fig-top-mun", figure=fig_top_mun_remesas(2020)), md=12),
                ], className="g-3"),
                dbc.Row([
                    dbc.Col(dcc.Graph(figure=fig_bnx_states()), md=12),
                ], className="g-3 mt-2"),
            ]),
        ]),
    ],
)


# ── Callbacks ─────────────────────────────────────────────────────────────────
@app.callback(Output("fig-top-dests", "figure"), Input("slider-mundo", "value"))
def update_top_dests(year: int) -> go.Figure:
    return fig_top_destinations(year)


@app.callback(Output("fig-top-sources", "figure"), Input("slider-mex", "value"))
def update_top_sources(year: int) -> go.Figure:
    return fig_top_sources(year)


@app.callback(Output("fig-top-mun", "figure"), Input("radio-iim-year", "value"))
def update_top_mun(year: int) -> go.Figure:
    return fig_top_mun_remesas(year)


if __name__ == "__main__":
    app.run(debug=True, port=8060)
