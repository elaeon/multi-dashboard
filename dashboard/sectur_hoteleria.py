import plotly.graph_objects as go
from dash import Dash, dcc, html, callback, Input, Output
import dash_bootstrap_components as dbc
import polars as pl

# ── Datos ─────────────────────────────────────────────────────────────────────
panel = pl.read_parquet("dashboard_data/sectur_hoteleria_panel.parquet")
mensual = pl.read_parquet("dashboard_data/sectur_extranjeros_mensual.parquet")
aeropuerto = pl.read_parquet("dashboard_data/sectur_extranjeros_aeropuerto.parquet")
evi = pl.read_parquet("dashboard_data/sectur_evi.parquet")
nac_mensual = pl.read_parquet("dashboard_data/sectur_nacionalidad_mensual.parquet")
nac_aeropuerto = pl.read_parquet("dashboard_data/sectur_nacionalidad_aeropuerto.parquet")

TOP_N_DESTINOS = 15
TOP_N_SLOPE = 8
TOP_N_PAISES = 10
TOP_N_AEROPUERTOS = 6
TODOS = "Todas"
ACUMULADO = "Acumulado"

CATEGORIA_OPTIONS = [TODOS, "Playa", "Ciudad", "Frontera"]
AÑOS_EXTRANJEROS = sorted(mensual["Año"].unique().to_list())
AÑO_OPTIONS = [ACUMULADO] + [str(a) for a in AÑOS_EXTRANJEROS]

AIR_MODES = {"Vía aérea"}
LAND_MODES = {"En automóviles", "Peatones", "Vía terrestre"}
MAR_MODES = {"Excursionistas vía marítima"}
MES_NOMBRE = {1: "Ene", 2: "Feb", 3: "Mar", 4: "Abr", 5: "May", 6: "Jun",
              7: "Jul", 8: "Ago", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dic"}
DIASPORA_PAISES = ["China", "Filipinas", "Colombia", "India", "Reunión, Islas", "Puerto Rico (EUA)"]


def _modo(nivel06: str) -> str:
    if nivel06 in AIR_MODES:
        return "Aéreo"
    if nivel06 in LAND_MODES:
        return "Terrestre"
    return "Marítimo (cruceros)"

FOCUS = "#2E86AB"
CONTEXT = "#475569"
GOOD = "#3BB273"
BAD = "#E84855"

CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font_color="#CBD5E1",
)
AXIS_STYLE = dict(gridcolor="#334155")

CARD_STYLE = {
    "background": "#1E293B", "border": "1px solid #334155",
    "borderRadius": "8px", "padding": "16px", "textAlign": "center",
}

TAB_STYLE = {"backgroundColor": "#0F172A", "color": "#94A3B8", "borderTop": "none"}
TAB_SEL = {"backgroundColor": "#1E293B", "color": "#F8FAFC",
           "borderTop": "2px solid #2E86AB", "fontWeight": "600"}


# ── Hotelería: agregados ──────────────────────────────────────────────────────

def national_weekly() -> pl.DataFrame:
    return (
        panel.group_by(["año", "semana", "fecha"])
        .agg(pl.col("cuartos_ocup").sum().alias("ocup_sum"), pl.col("cuartos_disp").sum().alias("disp_sum"))
        .with_columns((pl.col("ocup_sum") / pl.col("disp_sum") * 100).alias("ocupacion"))
        .sort(["año", "semana"])
    )


def categoria_weekly(categorias: list[str]) -> pl.DataFrame:
    return (
        panel.filter(pl.col("categoria").is_in(categorias))
        .group_by(["fecha", "categoria"])
        .agg(pl.col("cuartos_ocup").sum().alias("ocup_sum"), pl.col("cuartos_disp").sum().alias("disp_sum"))
        .with_columns((pl.col("ocup_sum") / pl.col("disp_sum") * 100).alias("ocupacion"))
        .sort("fecha")
    )


def destinos_ranking(categoria: str) -> pl.DataFrame:
    d = panel if categoria == TODOS else panel.filter(pl.col("categoria") == categoria)
    return (
        d.group_by("centro").agg(pl.col("cuartos_ocup").sum())
        .sort("cuartos_ocup", descending=True)
    )


def yoy_slope_destinos() -> pl.DataFrame:
    d24 = panel.filter(pl.col("año") == 2024).group_by("centro").agg(pl.col("ocupacion").mean().alias("o2024"))
    d25 = panel.filter(pl.col("año") == 2025).group_by("centro").agg(pl.col("ocupacion").mean().alias("o2025"))
    j = d24.join(d25, on="centro").with_columns((pl.col("o2025") - pl.col("o2024")).alias("delta")).sort("delta", descending=True)
    top = j.head(TOP_N_SLOPE)
    bottom = j.tail(TOP_N_SLOPE).sort("delta", descending=True)
    return pl.concat([top, bottom])


# ── Extranjeros: agregados ────────────────────────────────────────────────────

def anual_total() -> pl.DataFrame:
    return mensual.group_by("Año").agg(pl.col("Valor").sum()).sort("Año")


def region_recovery() -> pl.DataFrame:
    d = mensual.filter(pl.col("Región") != "No especificado").group_by(["Región", "Año"]).agg(pl.col("Valor").sum())
    base = d.filter(pl.col("Año") == 2019).select("Región", pl.col("Valor").alias("v2019"))
    return d.join(base, on="Región").with_columns((pl.col("Valor") / pl.col("v2019") * 100).alias("idx")).sort(["Región", "Año"])


def paises_top(año_sel: str) -> pl.DataFrame:
    d = mensual if año_sel == ACUMULADO else mensual.filter(pl.col("Año") == int(año_sel))
    return d.group_by("Pais").agg(pl.col("Valor").sum()).sort("Valor", descending=True).head(TOP_N_PAISES)


def aeropuertos_usa_share() -> pl.DataFrame:
    totals = aeropuerto.group_by("Aeropuerto").agg(pl.col("Valor").sum().alias("total")).sort("total", descending=True)
    top = totals.head(TOP_N_AEROPUERTOS)
    usa = aeropuerto.filter(pl.col("Pais") == "Estados Unidos").group_by("Aeropuerto").agg(pl.col("Valor").sum().alias("usa"))
    j = top.join(usa, on="Aeropuerto", how="left").with_columns((pl.col("usa") / pl.col("total") * 100).alias("pct_usa"))
    return j.sort("pct_usa", descending=True)


# ── EVI (Cuenta de Viajeros): agregados ───────────────────────────────────────

def evi_headcount_anual() -> pl.DataFrame:
    d = evi.filter((pl.col("Tipo") == "Ingresos") & (pl.col("DescripcionNivel03") == "Número de Viajeros"))
    return d.group_by("kano").agg(pl.col("Valor").sum().alias("total")).sort("kano")


def evi_headcount_ytd(kano: int, hasta_mes: int) -> float:
    d = evi.filter(
        (pl.col("Tipo") == "Ingresos") & (pl.col("DescripcionNivel03") == "Número de Viajeros")
        & (pl.col("kano") == kano) & (pl.col("mes") <= hasta_mes)
    )
    return float(d["Valor"].sum())


def evi_balance_anual() -> pl.DataFrame:
    ing = (
        evi.filter((pl.col("Tipo") == "Ingresos") & (pl.col("DescripcionNivel03") == "Ingresos"))
        .group_by("kano").agg(pl.col("Valor").sum().alias("ingresos"))
    )
    egr = (
        evi.filter((pl.col("Tipo") == "Egresos") & (pl.col("DescripcionNivel03") == "Egresos"))
        .group_by("kano").agg(pl.col("Valor").sum().alias("egresos"))
    )
    d = ing.join(egr, on="kano").sort("kano")
    return d.with_columns(
        (pl.col("ingresos") - pl.col("egresos")).alias("balance"),
        (pl.col("ingresos") / pl.col("egresos")).alias("cobertura"),
    )


def evi_modalidad(kano: int) -> pl.DataFrame:
    hc = (
        evi.filter((pl.col("Tipo") == "Ingresos") & (pl.col("DescripcionNivel03") == "Número de Viajeros") & (pl.col("kano") == kano))
        .with_columns(pl.col("DescripcionNivel06").map_elements(_modo, return_dtype=pl.String).alias("modo"))
        .group_by("modo").agg(pl.col("Valor").sum().alias("viajeros"))
    )
    rev = (
        evi.filter((pl.col("Tipo") == "Ingresos") & (pl.col("DescripcionNivel03") == "Ingresos") & (pl.col("kano") == kano))
        .with_columns(pl.col("DescripcionNivel06").map_elements(_modo, return_dtype=pl.String).alias("modo"))
        .group_by("modo").agg(pl.col("Valor").sum().alias("ingresos"))
    )
    d = hc.join(rev, on="modo")
    return d.with_columns(
        (pl.col("viajeros") / pl.col("viajeros").sum() * 100).alias("pct_viajeros"),
        (pl.col("ingresos") / pl.col("ingresos").sum() * 100).alias("pct_ingresos"),
    )


def evi_estacionalidad() -> pl.DataFrame:
    hc = evi.filter((pl.col("Tipo") == "Ingresos") & (pl.col("DescripcionNivel03") == "Número de Viajeros"))
    monthly = hc.group_by(["kano", "mes"]).agg(pl.col("Valor").sum().alias("total"))
    yearmean = monthly.group_by("kano").agg(pl.col("total").mean().alias("ymean"))
    idx = monthly.join(yearmean, on="kano").with_columns((pl.col("total") / pl.col("ymean") * 100).alias("idx"))

    pre = idx.filter(pl.col("kano").is_in([2018, 2019])).group_by("mes").agg(pl.col("idx").mean().alias("Pre-COVID (2018–19)"))
    post = idx.filter(pl.col("kano").is_in([2023, 2024, 2025])).group_by("mes").agg(pl.col("idx").mean().alias("Post-COVID (2023–25)"))
    return pre.join(post, on="mes").sort("mes")


# ── Figuras: Hotelería ────────────────────────────────────────────────────────

def fig_ocupacion_semanal() -> go.Figure:
    d = national_weekly()
    fig = go.Figure()
    colors = {2024: CONTEXT, 2025: "#94A3B8", 2026: FOCUS}
    for año in sorted(d["año"].unique().to_list()):
        sub = d.filter(pl.col("año") == año)
        fig.add_trace(go.Scatter(
            x=sub["semana"], y=sub["ocupacion"], mode="lines",
            name=str(año), line=dict(color=colors.get(año, CONTEXT), width=2.5 if año == 2026 else 1.8),
        ))

    fig.add_vrect(x0=36, x1=40, fillcolor="rgba(148,163,184,0.12)", line_width=0,
                  annotation_text="valle estacional", annotation_font_color="#94A3B8")
    fig.add_vline(x=30, line_dash="dot", line_color="#94A3B8",
                  annotation_text="semana 30: pico estacional",
                  annotation_font_color="#94A3B8", annotation_position="top left")

    fig.update_layout(
        **CHART_LAYOUT,
        xaxis=dict(**AXIS_STYLE, title="Semana ISO"),
        yaxis=dict(**AXIS_STYLE, title="Ocupación nacional (%)"),
        height=420,
        legend=dict(orientation="h", y=-0.18, x=0),
        margin=dict(t=60, b=70, l=60, r=20),
        title=dict(text=(
            "<b>2026 arranca por debajo de 2024 y 2025</b>"
            "<br><sup style='color:#94A3B8'>Ocupación nacional ponderada por semana ISO, 2024–2026</sup>"
        )),
    )
    return fig


def fig_playa_ciudad() -> go.Figure:
    d = categoria_weekly(["Playa", "Ciudad"])
    playa = d.filter(pl.col("categoria") == "Playa")
    ciudad = d.filter(pl.col("categoria") == "Ciudad")
    gap = playa["ocupacion"].mean() - ciudad["ocupacion"].mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=playa["fecha"], y=playa["ocupacion"], mode="lines",
                              name="Centros de playa", line=dict(color=FOCUS, width=2)))
    fig.add_trace(go.Scatter(x=ciudad["fecha"], y=ciudad["ocupacion"], mode="lines",
                              name="Ciudades", line=dict(color=CONTEXT, width=2)))

    fig.update_layout(
        **CHART_LAYOUT,
        xaxis=dict(**AXIS_STYLE, title="Semana"),
        yaxis=dict(**AXIS_STYLE, title="Ocupación (%)"),
        height=380,
        legend=dict(orientation="h", y=-0.2, x=0),
        margin=dict(t=60, b=70, l=60, r=20),
        title=dict(text=(
            f"<b>Los destinos de playa registran {gap:.0f} puntos más de ocupación que las ciudades</b>"
            "<br><sup style='color:#94A3B8'>Ocupación ponderada semanal, 2024–2026</sup>"
        )),
    )
    return fig


def fig_destinos_bar(categoria: str) -> go.Figure:
    d = destinos_ranking(categoria).head(TOP_N_DESTINOS)
    total = destinos_ranking(categoria)["cuartos_ocup"].sum()
    top5_share = destinos_ranking(categoria).head(5)["cuartos_ocup"].sum() / total * 100
    d = d.sort("cuartos_ocup", descending=False)
    colors = [FOCUS if i >= len(d) - 5 else CONTEXT for i in range(len(d))]

    fig = go.Figure(go.Bar(
        x=d["cuartos_ocup"], y=d["centro"], orientation="h",
        marker_color=colors,
        hovertemplate="<b>%{y}</b><br>Cuartos ocupados (suma 2024–2026): %{x:,.0f}<extra></extra>",
    ))
    fig.update_layout(
        **CHART_LAYOUT,
        xaxis=dict(**AXIS_STYLE, title="Cuartos ocupados (suma, 2024–2026)"),
        yaxis=dict(**AXIS_STYLE),
        height=max(320, TOP_N_DESTINOS * 26 + 90),
        margin=dict(t=60, b=50, l=10, r=20),
        title=dict(text=(
            f"<b>El top 5 concentra {top5_share:.0f}% de los cuartos ocupados</b>"
            f"<br><sup style='color:#94A3B8'>Top {TOP_N_DESTINOS} destinos · categoría: {categoria}</sup>"
        )),
        showlegend=False,
    )
    return fig, top5_share


def fig_slope_destinos() -> go.Figure:
    d = yoy_slope_destinos()
    fig = go.Figure()
    for row in d.iter_rows(named=True):
        color = GOOD if row["delta"] > 0 else BAD
        fig.add_trace(go.Scatter(
            x=["2024", "2025"], y=[row["o2024"], row["o2025"]],
            mode="lines+markers", line=dict(color=color, width=1.5), marker=dict(color=color, size=7),
            showlegend=False,
            hovertemplate=f"<b>{row['centro']}</b><br>%{{x}}: %{{y:.1f}}%<extra></extra>",
        ))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines+markers",
                              line=dict(color=GOOD), marker=dict(color=GOOD), name="▲ Mejoró"))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines+markers",
                              line=dict(color=BAD), marker=dict(color=BAD), name="▼ Empeoró"))

    fig.update_xaxes(type="category", gridcolor="rgba(0,0,0,0)")
    fig.update_layout(
        **CHART_LAYOUT,
        yaxis=dict(**AXIS_STYLE, title="Ocupación media anual (%)"),
        height=420,
        legend=dict(orientation="h", y=-0.15, x=0),
        margin=dict(t=60, b=60, l=60, r=20),
        title=dict(text=(
            "<b>Acapulco lidera la recuperación; Culiacán y Villahermosa la mayor caída</b>"
            f"<br><sup style='color:#94A3B8'>Ocupación media 2024→2025 · {TOP_N_SLOPE} mayores ganancias y caídas</sup>"
        )),
    )
    return fig


# ── Figuras: Extranjeros ──────────────────────────────────────────────────────

def fig_llegadas_anuales() -> go.Figure:
    d = anual_total()
    completos = d.filter(pl.col("Año") <= 2025)
    parcial = d.filter(pl.col("Año") >= 2025)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=completos["Año"], y=completos["Valor"], mode="lines+markers",
                              name="Años completos", line=dict(color=FOCUS, width=2.5)))
    fig.add_trace(go.Scatter(x=parcial["Año"], y=parcial["Valor"], mode="lines+markers",
                              name="2026 (parcial, ene–abr)", line=dict(color="#94A3B8", width=2, dash="dash")))

    fig.add_vrect(x0=2019.5, x1=2020.5, fillcolor="rgba(232,72,85,0.12)", line_width=0,
                  annotation_text="COVID-19", annotation_font_color="#94A3B8")
    fig.add_annotation(x=2025, y=float(d.filter(pl.col("Año") == 2025)["Valor"][0]),
                        text="<b>2025: récord</b>", showarrow=True, arrowcolor="#94A3B8",
                        font=dict(color="#F8FAFC", size=11), ax=0, ay=-35)

    fig.update_layout(
        **CHART_LAYOUT,
        xaxis=dict(**AXIS_STYLE, title="Año", dtick=2),
        yaxis=dict(**AXIS_STYLE, title="Llegadas de extranjeros por vía aérea"),
        height=420,
        legend=dict(orientation="h", y=-0.18, x=0),
        margin=dict(t=60, b=70, l=70, r=20),
        title=dict(text=(
            "<b>Las llegadas marcaron récord en 2024–2025 tras el colapso de 2020</b>"
            "<br><sup style='color:#94A3B8'>Extranjeros por país de residencia, entrada aérea, 2012–2026</sup>"
        )),
    )
    return fig


def fig_recuperacion_regional() -> go.Figure:
    d = region_recovery()
    fig = go.Figure()
    for region in d["Región"].unique().to_list():
        sub = d.filter(pl.col("Región") == region)
        is_focus = region == "América del Norte"
        fig.add_trace(go.Scatter(
            x=sub["Año"], y=sub["idx"], mode="lines",
            name=region,
            line=dict(color=FOCUS if is_focus else CONTEXT, width=3 if is_focus else 1.3),
            opacity=1 if is_focus else 0.7,
        ))
    fig.add_hline(y=100, line_dash="dash", line_color="#64748B",
                  annotation_text="nivel 2019", annotation_font_color="#94A3B8")

    fig.update_layout(
        **CHART_LAYOUT,
        xaxis=dict(**AXIS_STYLE, title="Año"),
        yaxis=dict(**AXIS_STYLE, title="Índice (2019 = 100)"),
        height=440,
        legend=dict(orientation="h", y=-0.25, x=0, font=dict(size=10)),
        margin=dict(t=60, b=90, l=60, r=20),
        title=dict(text=(
            "<b>La recuperación es solo de Norteamérica — el resto del mundo sigue por debajo de 2019</b>"
            "<br><sup style='color:#94A3B8'>Llegadas por región, indexadas a 2019 = 100</sup>"
        )),
    )
    return fig


def fig_paises_bar(año_sel: str) -> go.Figure:
    d = paises_top(año_sel).sort("Valor", descending=False)
    total = (mensual if año_sel == ACUMULADO else mensual.filter(pl.col("Año") == int(año_sel)))["Valor"].sum()
    top10_share = d["Valor"].sum() / total * 100
    colors = [FOCUS if p == "Estados Unidos" else CONTEXT for p in d["Pais"]]

    fig = go.Figure(go.Bar(
        x=d["Valor"], y=d["Pais"], orientation="h", marker_color=colors,
        hovertemplate="<b>%{y}</b>: %{x:,.0f}<extra></extra>",
    ))
    fig.update_layout(
        **CHART_LAYOUT,
        xaxis=dict(**AXIS_STYLE, title="Llegadas"),
        yaxis=dict(**AXIS_STYLE),
        height=max(320, TOP_N_PAISES * 26 + 90),
        margin=dict(t=60, b=50, l=10, r=20),
        title=dict(text=(
            f"<b>Top {TOP_N_PAISES} países = {top10_share:.0f}% de las llegadas</b>"
            f"<br><sup style='color:#94A3B8'>{año_sel}</sup>"
        )),
        showlegend=False,
    )
    return fig, top10_share


def fig_aeropuertos_usa() -> go.Figure:
    d = aeropuertos_usa_share().sort("pct_usa", descending=False)
    colors = [FOCUS if a.startswith("Cancún") or a.startswith("Ciudad de México") else CONTEXT for a in d["Aeropuerto"]]

    fig = go.Figure(go.Bar(
        x=d["pct_usa"], y=d["Aeropuerto"], orientation="h", marker_color=colors,
        hovertemplate="<b>%{y}</b>: %{x:.1f}%% de EUA<extra></extra>",
    ))
    fig.update_layout(
        **CHART_LAYOUT,
        xaxis=dict(**AXIS_STYLE, title="% de llegadas desde Estados Unidos", range=[0, 100]),
        yaxis=dict(**AXIS_STYLE),
        height=max(300, TOP_N_AEROPUERTOS * 30 + 90),
        margin=dict(t=70, b=50, l=10, r=20),
        title=dict(text=(
            "<b>Guadalajara y Los Cabos dependen de EUA mucho más que Cancún</b>"
            "<br><sup style='color:#94A3B8'>% de llegadas con origen EUA por aeropuerto, top 6 por volumen, 2012–2026</sup>"
        )),
        showlegend=False,
    )
    return fig


# ── Figuras: EVI (Cuenta de Viajeros) ─────────────────────────────────────────

def fig_evi_headcount() -> go.Figure:
    d = evi_headcount_anual()
    completos = d.filter(pl.col("kano") <= 2025)
    v2025 = float(completos.filter(pl.col("kano") == 2025)["total"][0])
    v2019 = float(completos.filter(pl.col("kano") == 2019)["total"][0])
    ytd_kano, ytd_mes = 2026, int(evi.filter(pl.col("kano") == 2026)["mes"].max())
    ytd_valor = evi_headcount_ytd(ytd_kano, ytd_mes)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=completos["kano"], y=completos["total"], mode="lines+markers",
                              name="Años completos", line=dict(color=FOCUS, width=2.5)))
    fig.add_trace(go.Scatter(x=[2025, 2026], y=[v2025, ytd_valor], mode="lines+markers",
                              name=f"2026 (parcial, ene–{MES_NOMBRE[ytd_mes].lower()})",
                              line=dict(color="#94A3B8", width=2, dash="dash")))

    fig.add_vrect(x0=2019.5, x1=2020.5, fillcolor="rgba(232,72,85,0.12)", line_width=0,
                  annotation_text="COVID-19 (−47.5%)", annotation_font_color="#94A3B8")
    fig.add_annotation(x=2025, y=v2025, text="<b>2025: recuperación completa</b>",
                        showarrow=True, arrowcolor="#94A3B8", font=dict(color="#F8FAFC", size=11), ax=-10, ay=-35)

    fig.update_layout(
        **CHART_LAYOUT,
        xaxis=dict(**AXIS_STYLE, title="Año", dtick=2),
        yaxis=dict(**AXIS_STYLE, title="Viajeros internacionales (todas las modalidades)"),
        height=420,
        legend=dict(orientation="h", y=-0.18, x=0),
        margin=dict(t=60, b=70, l=70, r=20),
        title=dict(text=(
            f"<b>El volumen de viajeros ya superó el nivel pre-pandemia ({(v2025/v2019-1)*100:+.1f}% vs. 2019)</b>"
            "<br><sup style='color:#94A3B8'>Total de viajeros internacionales (Ingresos, todas las modalidades), 2018–2026</sup>"
        )),
    )
    return fig


def fig_evi_balance() -> go.Figure:
    d = evi_balance_anual().filter(pl.col("kano") <= 2025)
    cob_2022 = float(d.filter(pl.col("kano") == 2022)["cobertura"][0])
    cob_2025 = float(d.filter(pl.col("kano") == 2025)["cobertura"][0])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d["kano"], y=d["ingresos"] / 1e9, mode="lines+markers",
                              name="Ingresos (turismo entrante)", line=dict(color=FOCUS, width=2.2)))
    fig.add_trace(go.Scatter(x=d["kano"], y=d["egresos"] / 1e9, mode="lines+markers",
                              name="Egresos (turismo saliente)", line=dict(color=CONTEXT, width=2.2)))
    fig.add_trace(go.Scatter(x=d["kano"], y=d["balance"] / 1e9, mode="lines+markers",
                              name="Superávit", line=dict(color=GOOD, width=1.8, dash="dot")))

    fig.add_annotation(x=2022, y=float(d.filter(pl.col("kano") == 2022)["balance"][0]) / 1e9,
                        text=f"cobertura {cob_2022:.2f}×", showarrow=True, arrowcolor="#94A3B8",
                        font=dict(color="#94A3B8", size=10), ax=-20, ay=-30)
    fig.add_annotation(x=2025, y=float(d.filter(pl.col("kano") == 2025)["balance"][0]) / 1e9,
                        text=f"cobertura {cob_2025:.2f}×", showarrow=True, arrowcolor="#94A3B8",
                        font=dict(color="#94A3B8", size=10), ax=20, ay=-30)

    fig.update_layout(
        **CHART_LAYOUT,
        xaxis=dict(**AXIS_STYLE, title="Año", dtick=1),
        yaxis=dict(**AXIS_STYLE, title="Miles de millones de USD"),
        height=420,
        legend=dict(orientation="h", y=-0.18, x=0),
        margin=dict(t=60, b=70, l=70, r=20),
        title=dict(text=(
            "<b>El superávit turístico se estancó: el gasto de mexicanos en el extranjero crece más rápido</b>"
            "<br><sup style='color:#94A3B8'>Ingresos, egresos y balance de la cuenta de viajeros, 2018–2025</sup>"
        )),
    )
    return fig


def fig_evi_modalidad() -> go.Figure:
    d = evi_modalidad(2024).sort("pct_ingresos", descending=False)

    fig = go.Figure()
    fig.add_trace(go.Bar(x=d["pct_viajeros"], y=d["modo"], orientation="h",
                          name="% de viajeros", marker_color=CONTEXT,
                          hovertemplate="<b>%{y}</b><br>Viajeros: %{x:.1f}%<extra></extra>"))
    fig.add_trace(go.Bar(x=d["pct_ingresos"], y=d["modo"], orientation="h",
                          name="% de ingresos", marker_color=FOCUS,
                          hovertemplate="<b>%{y}</b><br>Ingresos: %{x:.1f}%<extra></extra>"))

    fig.update_layout(
        **CHART_LAYOUT,
        xaxis=dict(**AXIS_STYLE, title="% del total nacional"),
        yaxis=dict(**AXIS_STYLE),
        barmode="group",
        height=340,
        legend=dict(orientation="h", y=-0.22, x=0),
        margin=dict(t=60, b=60, l=10, r=20),
        title=dict(text=(
            "<b>El dinero sigue a los turistas aéreos, no a la multitud</b>"
            "<br><sup style='color:#94A3B8'>Modalidad de entrada: % de viajeros vs. % de ingresos por turismo, 2024</sup>"
        )),
    )
    return fig


def fig_evi_estacionalidad() -> go.Figure:
    d = evi_estacionalidad().sort("mes")
    meses = [MES_NOMBRE[m] for m in d["mes"]]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=meses, y=d["Pre-COVID (2018–19)"], mode="lines+markers",
                              name="Pre-COVID (2018–19)", line=dict(color=CONTEXT, width=2, dash="dash")))
    fig.add_trace(go.Scatter(x=meses, y=d["Post-COVID (2023–25)"], mode="lines+markers",
                              name="Post-COVID (2023–25)", line=dict(color=FOCUS, width=2.5)))

    fig.add_annotation(x="Dic", y=float(d.filter(pl.col("mes") == 12)["Post-COVID (2023–25)"][0]),
                        text="<b>diciembre: nuevo pico</b><br>(+16.7 pts vs. antes)", showarrow=True,
                        arrowcolor="#94A3B8", font=dict(color="#F8FAFC", size=11), ax=-10, ay=-40)
    fig.add_annotation(x="Mar", y=float(d.filter(pl.col("mes") == 3)["Pre-COVID (2018–19)"][0]),
                        text="antes: marzo era el pico", showarrow=True,
                        arrowcolor="#94A3B8", font=dict(color="#94A3B8", size=10), ax=10, ay=-30)

    fig.update_layout(
        **CHART_LAYOUT,
        xaxis=dict(**AXIS_STYLE, title="Mes", categoryorder="array", categoryarray=meses),
        yaxis=dict(**AXIS_STYLE, title="Índice estacional (100 = promedio del año)"),
        height=380,
        legend=dict(orientation="h", y=-0.2, x=0),
        margin=dict(t=60, b=60, l=60, r=20),
        title=dict(text=(
            "<b>El pico de temporada se movió de marzo a diciembre tras la pandemia</b>"
            "<br><sup style='color:#94A3B8'>Índice estacional del total de viajeros internacionales</sup>"
        )),
    )
    return fig


# ── Nacionalidad: agregados ───────────────────────────────────────────────────

def nac_anual_total() -> pl.DataFrame:
    return nac_mensual.group_by("Año").agg(pl.col("Valor").sum()).sort("Año")


def nac_region_recovery() -> pl.DataFrame:
    d = nac_mensual.filter(~pl.col("Región").is_in(["No especificado", "Apátrida"])).group_by(["Región", "Año"]).agg(pl.col("Valor").sum())
    base = d.filter(pl.col("Año") == 2019).select("Región", pl.col("Valor").alias("v2019"))
    return d.join(base, on="Región").with_columns((pl.col("Valor") / pl.col("v2019") * 100).alias("idx")).sort(["Región", "Año"])


def nac_aeropuertos_usa_share() -> pl.DataFrame:
    totals = nac_aeropuerto.group_by("Aeropuerto").agg(pl.col("Valor").sum().alias("total")).sort("total", descending=True)
    top = totals.head(TOP_N_AEROPUERTOS)
    usa = nac_aeropuerto.filter(pl.col("Pais") == "Estados Unidos").group_by("Aeropuerto").agg(pl.col("Valor").sum().alias("usa"))
    j = top.join(usa, on="Aeropuerto", how="left").with_columns((pl.col("usa") / pl.col("total") * 100).alias("pct_usa"))
    return j.sort("pct_usa", descending=True)


def nac_vs_residencia_rank() -> pl.DataFrame:
    nac_tot = (
        nac_mensual.group_by("Pais").agg(pl.col("Valor").sum().alias("nacionalidad"))
        .with_columns(pl.col("nacionalidad").rank(method="ordinal", descending=True).cast(pl.Int32).alias("rank_nac"))
    )
    res_tot = (
        mensual.group_by("Pais").agg(pl.col("Valor").sum().alias("residencia"))
        .with_columns(pl.col("residencia").rank(method="ordinal", descending=True).cast(pl.Int32).alias("rank_res"))
    )
    return nac_tot.join(res_tot, on="Pais", how="inner").filter(pl.col("Pais").is_in(DIASPORA_PAISES))


# ── Figuras: Nacionalidad ──────────────────────────────────────────────────────

def fig_nac_llegadas_anuales() -> go.Figure:
    d = nac_anual_total()
    completos = d.filter(pl.col("Año") <= 2025)
    parcial = d.filter(pl.col("Año") >= 2025)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=completos["Año"], y=completos["Valor"], mode="lines+markers",
                              name="Años completos", line=dict(color=FOCUS, width=2.5)))
    fig.add_trace(go.Scatter(x=parcial["Año"], y=parcial["Valor"], mode="lines+markers",
                              name="2026 (parcial, ene–abr)", line=dict(color="#94A3B8", width=2, dash="dash")))

    fig.add_vrect(x0=2019.5, x1=2020.5, fillcolor="rgba(232,72,85,0.12)", line_width=0,
                  annotation_text="COVID-19", annotation_font_color="#94A3B8")
    fig.add_annotation(x=2024, y=float(d.filter(pl.col("Año") == 2024)["Valor"][0]),
                        text="<b>2024: récord</b>", showarrow=True, arrowcolor="#94A3B8",
                        font=dict(color="#F8FAFC", size=11), ax=0, ay=-35)

    fig.update_layout(
        **CHART_LAYOUT,
        xaxis=dict(**AXIS_STYLE, title="Año", dtick=2),
        yaxis=dict(**AXIS_STYLE, title="Llegadas de extranjeros por vía aérea"),
        height=420,
        legend=dict(orientation="h", y=-0.18, x=0),
        margin=dict(t=60, b=70, l=70, r=20),
        title=dict(text=(
            "<b>El récord de 2024 ya se enfrió: 2025 y el arranque de 2026 van a la baja</b>"
            "<br><sup style='color:#94A3B8'>Extranjeros por nacionalidad, entrada aérea, 2012–2026</sup>"
        )),
    )
    return fig


def fig_nac_recuperacion_regional() -> go.Figure:
    d = nac_region_recovery()
    fig = go.Figure()
    for region in d["Región"].unique().to_list():
        sub = d.filter(pl.col("Región") == region)
        is_focus = region == "América del Norte"
        fig.add_trace(go.Scatter(
            x=sub["Año"], y=sub["idx"], mode="lines",
            name=region,
            line=dict(color=FOCUS if is_focus else CONTEXT, width=3 if is_focus else 1.3),
            opacity=1 if is_focus else 0.7,
        ))
    fig.add_hline(y=100, line_dash="dash", line_color="#64748B",
                  annotation_text="nivel 2019", annotation_font_color="#94A3B8")

    fig.update_layout(
        **CHART_LAYOUT,
        xaxis=dict(**AXIS_STYLE, title="Año"),
        yaxis=dict(**AXIS_STYLE, title="Índice (2019 = 100)"),
        height=440,
        legend=dict(orientation="h", y=-0.25, x=0, font=dict(size=10)),
        margin=dict(t=60, b=90, l=60, r=20),
        title=dict(text=(
            "<b>Por nacionalidad, el patrón se repite: solo Norteamérica supera a 2019 (+31%)</b>"
            "<br><sup style='color:#94A3B8'>Llegadas por región de nacionalidad, indexadas a 2019 = 100</sup>"
        )),
    )
    return fig


def fig_nac_vs_residencia() -> go.Figure:
    d = nac_vs_residencia_rank().sort("rank_nac", descending=True)

    x_lines, y_lines = [], []
    for row in d.iter_rows(named=True):
        x_lines += [row["rank_res"], row["rank_nac"], None]
        y_lines += [row["Pais"], row["Pais"], None]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_lines, y=y_lines, mode="lines",
                              line=dict(color="#475569", width=1.5), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=d["rank_res"], y=d["Pais"], mode="markers", name="Posición por residencia",
                              marker=dict(color=CONTEXT, size=10, symbol="circle-open", line=dict(color=CONTEXT, width=2))))
    fig.add_trace(go.Scatter(x=d["rank_nac"], y=d["Pais"], mode="markers", name="Posición por nacionalidad",
                              marker=dict(color=FOCUS, size=10),
                              hovertemplate="<b>%{y}</b><br>Posición por nacionalidad: %{x}<extra></extra>"))

    fig.update_layout(
        **CHART_LAYOUT,
        xaxis=dict(**AXIS_STYLE, title="Posición en el ranking de países (1 = más visitantes)"),
        yaxis=dict(**AXIS_STYLE),
        height=340,
        legend=dict(orientation="h", y=-0.22, x=0),
        margin=dict(t=70, b=60, l=10, r=20),
        title=dict(text=(
            "<b>India sube 14 lugares por diáspora; Puerto Rico cae 145 por ser \"nacionalidad EUA\"</b>"
            "<br><sup style='color:#94A3B8'>Posición en el ranking de países: por nacionalidad vs. por residencia, acumulado 2012–2026</sup>"
        )),
    )
    return fig


def fig_nac_aeropuertos_usa() -> go.Figure:
    d = nac_aeropuertos_usa_share().sort("pct_usa", descending=False)
    colors = [FOCUS if a.startswith("Cancún") or a.startswith("Ciudad de México") else CONTEXT for a in d["Aeropuerto"]]

    fig = go.Figure(go.Bar(
        x=d["pct_usa"], y=d["Aeropuerto"], orientation="h", marker_color=colors,
        hovertemplate="<b>%{y}</b>: %{x:.1f}%% nacionalidad EUA<extra></extra>",
    ))
    fig.update_layout(
        **CHART_LAYOUT,
        xaxis=dict(**AXIS_STYLE, title="% de llegadas de nacionalidad estadounidense", range=[0, 100]),
        yaxis=dict(**AXIS_STYLE),
        height=max(300, TOP_N_AEROPUERTOS * 30 + 90),
        margin=dict(t=70, b=50, l=10, r=20),
        title=dict(text=(
            "<b>Cancún no es el monocultivo de EUA — Guadalajara y Los Cabos sí lo son</b>"
            "<br><sup style='color:#94A3B8'>% de llegadas de nacionalidad EUA por aeropuerto, top 6 por volumen, 2012–2026</sup>"
        )),
        showlegend=False,
    )
    return fig


# ── Layout ────────────────────────────────────────────────────────────────────

def kpi(value_id: str, value: str, label: str):
    return dbc.Col(html.Div([
        html.Div(value, id=value_id, style={"fontSize": "2rem", "fontWeight": "700", "color": "#F8FAFC"}),
        html.Div(label, style={"fontSize": "0.85rem", "color": "#64748B", "marginTop": "4px"}),
    ], style=CARD_STYLE), md=4, className="mb-3")


# KPI: ocupación últimas 4 semanas vs mismo periodo año anterior
_nw = national_weekly()
_max_año = _nw["año"].max()
_max_semana = _nw.filter(pl.col("año") == _max_año)["semana"].max()
_ventana = list(range(max(1, _max_semana - 3), _max_semana + 1))
_actual = _nw.filter((pl.col("año") == _max_año) & pl.col("semana").is_in(_ventana))["ocupacion"].mean()
_previo = _nw.filter((pl.col("año") == _max_año - 1) & pl.col("semana").is_in(_ventana))["ocupacion"].mean()
_delta_ocup = _actual - _previo
_ultima_semana = panel.filter((pl.col("año") == _max_año) & (pl.col("semana") == _max_semana))
_cuartos_ocup_ultima = _ultima_semana["cuartos_ocup"].sum()
_n_destinos_ultima = _ultima_semana["centro"].n_unique()

fig_destinos_bar_default, _top5_share_default = fig_destinos_bar(TODOS)

# KPI: extranjeros 2025 vs 2024, 2026 YTD vs 2025 mismo periodo
_anual = anual_total()
_v2025 = float(_anual.filter(pl.col("Año") == 2025)["Valor"][0])
_v2024 = float(_anual.filter(pl.col("Año") == 2024)["Valor"][0])
_delta_anual = (_v2025 - _v2024) / _v2024 * 100
_max_mes_2026 = mensual.filter(pl.col("Año") == 2026)["MesNum"].max()
_ytd_2026 = mensual.filter((pl.col("Año") == 2026) & (pl.col("MesNum") <= _max_mes_2026))["Valor"].sum()
_ytd_2025 = mensual.filter((pl.col("Año") == 2025) & (pl.col("MesNum") <= _max_mes_2026))["Valor"].sum()
_delta_ytd = (_ytd_2026 - _ytd_2025) / _ytd_2025 * 100

fig_paises_bar_default, _top10_share_default = fig_paises_bar(ACUMULADO)

# KPI: EVI viajeros/ingresos 2025 vs 2019, superávit 2025
_evi_hc = evi_headcount_anual()
_evi_h2025 = float(_evi_hc.filter(pl.col("kano") == 2025)["total"][0])
_evi_h2019 = float(_evi_hc.filter(pl.col("kano") == 2019)["total"][0])
_evi_delta_hc = (_evi_h2025 - _evi_h2019) / _evi_h2019 * 100
_evi_bal = evi_balance_anual()
_evi_ing2025 = float(_evi_bal.filter(pl.col("kano") == 2025)["ingresos"][0])
_evi_ing2019 = float(_evi_bal.filter(pl.col("kano") == 2019)["ingresos"][0])
_evi_delta_rev = (_evi_ing2025 - _evi_ing2019) / _evi_ing2019 * 100
_evi_superavit2025 = float(_evi_bal.filter(pl.col("kano") == 2025)["balance"][0])

# KPI: nacionalidad 2025 vs 2024, 2026 YTD vs 2025, USA share nacionalidad vs residencia
_nac_anual = nac_anual_total()
_nac_v2025 = float(_nac_anual.filter(pl.col("Año") == 2025)["Valor"][0])
_nac_v2024 = float(_nac_anual.filter(pl.col("Año") == 2024)["Valor"][0])
_nac_delta_anual = (_nac_v2025 - _nac_v2024) / _nac_v2024 * 100
_nac_max_mes_2026 = nac_mensual.filter(pl.col("Año") == 2026)["MesNum"].max()
_nac_ytd_2026 = nac_mensual.filter((pl.col("Año") == 2026) & (pl.col("MesNum") <= _nac_max_mes_2026))["Valor"].sum()
_nac_ytd_2025 = nac_mensual.filter((pl.col("Año") == 2025) & (pl.col("MesNum") <= _nac_max_mes_2026))["Valor"].sum()
_nac_delta_ytd = (_nac_ytd_2026 - _nac_ytd_2025) / _nac_ytd_2025 * 100
_nac_usa_share = nac_mensual.filter(pl.col("Pais") == "Estados Unidos")["Valor"].sum() / nac_mensual["Valor"].sum() * 100
_res_usa_share = mensual.filter(pl.col("Pais") == "Estados Unidos")["Valor"].sum() / mensual["Valor"].sum() * 100

app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    title="SECTUR Hotelería",
)

app.layout = dbc.Container([
    html.H1(
        "Turismo en México: ocupación hotelera y llegadas de extranjeros",
        style={"color": "#F8FAFC", "fontWeight": "700", "marginBottom": "4px", "fontSize": "1.8rem"},
    ),
    html.P(
        "SECTUR DataTur · hotelería semanal 2024–2026, extranjeros por país de residencia y "
        "nacionalidad 2012–2026, y cuenta de viajeros internacionales 2018–2026",
        style={"color": "#64748B", "marginBottom": "24px"},
    ),

    dbc.Tabs([
        dbc.Tab(label="Ocupación nacional", tab_style=TAB_STYLE, active_tab_style=TAB_SEL, children=[
            dbc.Row([
                kpi("kpi-ocup-actual", f"{_actual:.1f}%", "ocupación, últimas 4 semanas"),
                kpi("kpi-ocup-delta", f"{'+' if _delta_ocup >= 0 else ''}{_delta_ocup:.1f} pp", "vs. mismo periodo año anterior"),
                kpi("kpi-destinos-activos", f"{_n_destinos_ultima}", "destinos reportando (última semana)"),
            ], className="mb-4 mt-3"),
            dcc.Graph(figure=fig_ocupacion_semanal(), config={"displayModeBar": False}),
            dcc.Graph(figure=fig_playa_ciudad(), config={"displayModeBar": False}),
        ]),
        dbc.Tab(label="Destinos", tab_style=TAB_STYLE, active_tab_style=TAB_SEL, children=[
            dbc.Row([
                kpi("kpi-top5-destinos", f"{_top5_share_default:.0f}%", "cuartos ocupados en el top 5 de destinos"),
            ], className="mb-4 mt-3"),
            html.Div([
                html.Label("Categoría:", style={"color": "#94A3B8", "fontSize": "0.85rem", "marginBottom": "6px"}),
                dcc.Dropdown(
                    id="categoria-dropdown",
                    options=CATEGORIA_OPTIONS,
                    value=TODOS,
                    clearable=False,
                    style={"backgroundColor": "#1E293B", "color": "#0F172A"},
                ),
            ], style={"padding": "0 8px 16px", "maxWidth": "360px"}),
            dcc.Graph(id="graph-destinos-bar", figure=fig_destinos_bar_default, config={"displayModeBar": False}),
            dcc.Graph(figure=fig_slope_destinos(), config={"displayModeBar": False}),
        ]),
        dbc.Tab(label="Extranjeros: llegadas", tab_style=TAB_STYLE, active_tab_style=TAB_SEL, children=[
            dbc.Row([
                kpi("kpi-anual-delta", f"{'+' if _delta_anual >= 0 else ''}{_delta_anual:.1f}%", "llegadas 2025 vs. 2024"),
                kpi("kpi-ytd-delta", f"{'+' if _delta_ytd >= 0 else ''}{_delta_ytd:.1f}%", "2026 (ene–abr) vs. mismo periodo 2025"),
            ], className="mb-4 mt-3"),
            dcc.Graph(figure=fig_llegadas_anuales(), config={"displayModeBar": False}),
            dcc.Graph(figure=fig_recuperacion_regional(), config={"displayModeBar": False}),
        ]),
        dbc.Tab(label="Extranjeros: concentración", tab_style=TAB_STYLE, active_tab_style=TAB_SEL, children=[
            dbc.Row([
                kpi("kpi-top10-paises", f"{_top10_share_default:.0f}%", "llegadas en el top 10 de países"),
            ], className="mb-4 mt-3"),
            html.Div([
                html.Label("Año:", style={"color": "#94A3B8", "fontSize": "0.85rem", "marginBottom": "6px"}),
                dcc.Dropdown(
                    id="año-dropdown",
                    options=AÑO_OPTIONS,
                    value=ACUMULADO,
                    clearable=False,
                    style={"backgroundColor": "#1E293B", "color": "#0F172A"},
                ),
            ], style={"padding": "0 8px 16px", "maxWidth": "360px"}),
            dcc.Graph(id="graph-paises-bar", figure=fig_paises_bar_default, config={"displayModeBar": False}),
            dcc.Graph(figure=fig_aeropuertos_usa(), config={"displayModeBar": False}),
        ]),
        dbc.Tab(label="Viajeros: volumen e ingresos", tab_style=TAB_STYLE, active_tab_style=TAB_SEL, children=[
            dbc.Row([
                kpi("kpi-evi-hc-delta", f"{'+' if _evi_delta_hc >= 0 else ''}{_evi_delta_hc:.1f}%", "viajeros 2025 vs. 2019"),
                kpi("kpi-evi-rev-delta", f"{'+' if _evi_delta_rev >= 0 else ''}{_evi_delta_rev:.1f}%", "ingresos por turismo 2025 vs. 2019"),
                kpi("kpi-evi-superavit", f"${_evi_superavit2025/1e9:.1f}B", "superávit turístico 2025 (USD)"),
            ], className="mb-4 mt-3"),
            dcc.Graph(figure=fig_evi_headcount(), config={"displayModeBar": False}),
            dcc.Graph(figure=fig_evi_balance(), config={"displayModeBar": False}),
        ]),
        dbc.Tab(label="Viajeros: gasto y estacionalidad", tab_style=TAB_STYLE, active_tab_style=TAB_SEL, children=[
            dcc.Graph(figure=fig_evi_modalidad(), config={"displayModeBar": False}),
            dcc.Graph(figure=fig_evi_estacionalidad(), config={"displayModeBar": False}),
        ]),
        dbc.Tab(label="Nacionalidad: llegadas", tab_style=TAB_STYLE, active_tab_style=TAB_SEL, children=[
            dbc.Row([
                kpi("kpi-nac-anual-delta", f"{'+' if _nac_delta_anual >= 0 else ''}{_nac_delta_anual:.1f}%", "llegadas 2025 vs. 2024"),
                kpi("kpi-nac-ytd-delta", f"{'+' if _nac_delta_ytd >= 0 else ''}{_nac_delta_ytd:.1f}%", "2026 (ene–abr) vs. mismo periodo 2025"),
            ], className="mb-4 mt-3"),
            dcc.Graph(figure=fig_nac_llegadas_anuales(), config={"displayModeBar": False}),
            dcc.Graph(figure=fig_nac_recuperacion_regional(), config={"displayModeBar": False}),
        ]),
        dbc.Tab(label="Nacionalidad vs. residencia", tab_style=TAB_STYLE, active_tab_style=TAB_SEL, children=[
            dbc.Row([
                kpi("kpi-nac-usa-share", f"{_nac_usa_share:.1f}%", "de las llegadas son de nacionalidad EUA"),
                kpi("kpi-res-usa-share", f"{_res_usa_share:.1f}%", "vs. % con residencia en EUA"),
            ], className="mb-4 mt-3"),
            dcc.Graph(figure=fig_nac_vs_residencia(), config={"displayModeBar": False}),
            dcc.Graph(figure=fig_nac_aeropuertos_usa(), config={"displayModeBar": False}),
        ]),
    ], style={"marginTop": "8px"}),

], fluid=True, style={"backgroundColor": "#0F172A", "minHeight": "100vh", "padding": "24px 32px"})


# ── Callbacks ─────────────────────────────────────────────────────────────────

@callback(
    Output("graph-destinos-bar", "figure"),
    Output("kpi-top5-destinos", "children"),
    Input("categoria-dropdown", "value"),
)
def update_categoria(categoria):
    fig, top5_share = fig_destinos_bar(categoria)
    return fig, f"{top5_share:.0f}%"


@callback(
    Output("graph-paises-bar", "figure"),
    Output("kpi-top10-paises", "children"),
    Input("año-dropdown", "value"),
)
def update_año(año_sel):
    fig, top10_share = fig_paises_bar(año_sel)
    return fig, f"{top10_share:.0f}%"


if __name__ == "__main__":
    app.run(debug=True, port=8069)
