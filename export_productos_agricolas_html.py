"""
Exports a fully self-contained productos_agricolas.html for GitHub Pages.

The state dropdown works entirely client-side:
  - Per-state aggregates are pre-computed in Python and embedded as JSON
  - Plotly.newPlot() swaps chart data on selection — no server needed

Charts that respond to the dropdown (state-sensitive):
  annual trend, top crops, market trend, siniestro trend, top export crops, KPIs

Charts that stay as full-dataset context:
  map, treemap, scatter, technology charts, export-by-state, etc.

Run: uv run python export_productos_agricolas_html.py
"""

import json
import polars as pl
import plotly.io as pio

from productos_agricolas import (
    df, compute_kpis, YEAR_MIN, YEAR_MAX, _fmt_pesos, ALL_CROPS,
    fig_produccion_anual, fig_top_cultivos_valor,
    fig_mapa_estados, fig_top_estados, fig_rendimiento_estados,
    fig_treemap_cultivos, fig_scatter_rendimiento_precio, fig_evolucion_top_cultivos,
    fig_tecnologia_trend, fig_rendimiento_por_tecnologia, fig_produccion_tipo,
    fig_mercado_trend, fig_top_cultivos_exportacion, fig_exportacion_por_estado,
    fig_siniestro_anual, fig_siniestro_estados, fig_siniestro_scatter,
    fig_cultivo_trend, fig_cultivo_estados, fig_cultivo_precio,
)

d = df
valor, volumen, n_cult, rend = compute_kpis(d)

CHART_CFG = {"responsive": True, "displayModeBar": False}


# ── HTML helpers ──────────────────────────────────────────────────────────────

def div(fig, div_id: str, first=False) -> str:
    return pio.to_html(
        fig, full_html=False,
        include_plotlyjs="cdn" if first else False,
        config=CHART_CFG,
        div_id=div_id,
    )


def row(*cols) -> str:
    inner = "".join(f'<div class="col-{w} mb-3">{h}</div>' for h, w in cols)
    return f'<div class="row g-3">{inner}</div>'


def kpi(title, value, sub, kpi_id=None):
    vid = f' id="{kpi_id}"'     if kpi_id else ""
    sid = f' id="{kpi_id}-sub"' if kpi_id else ""
    return f"""<div class="col-3">
      <div class="kpi-card">
        <p class="kpi-label">{title}</p>
        <h3 class="kpi-value"{vid}>{value}</h3>
        <small class="kpi-sub"{sid}>{sub}</small>
      </div>
    </div>"""


def tab_btn(tab_id, label, active=False):
    cls = "nav-link active" if active else "nav-link"
    return (f'<button class="{cls}" id="btn-{tab_id}" data-bs-toggle="tab" '
            f'data-bs-target="#tab-{tab_id}" type="button" role="tab">{label}</button>')


def tab_pane(tab_id, content, active=False):
    cls = "tab-pane fade show active" if active else "tab-pane fade"
    return f'<div class="{cls}" id="tab-{tab_id}" role="tabpanel">{content}</div>'


def insight_card(title, body):
    return f"""<div class="col-4">
      <div class="insight-card">
        <strong class="insight-title">{title}</strong>
        <p class="insight-body">{body}</p>
      </div>
    </div>"""


# ── Pre-compute per-state data for client-side filtering ─────────────────────

def build_estado_data(df: pl.DataFrame) -> str:
    """
    Returns JSON keyed by state name (plus '__all__').
    Compact keys to keep payload small:
      t   → annual trend  {y: years, v: valor_B per year}
      mk  → market split  {y: years, n: nacional_B, e: exportacion_B}
      tc  → top 10 crops  {n: names, v: valor_B}
      ex  → top 10 export {n: names, v: valor_B}
      sin → siniestro     {y: years, v: tasa_pct}
      kpi → {vl: valor_str, vo: vol_M, cu: cultivos, re: rend}
    """
    print("  Pre-computing annual trend data…")
    trend = (
        df.group_by("ENTIDAD", "AÑO")
        .agg((pl.col("VALOR_PRODUCCION").sum() / 1e9).round(2).alias("v"))
        .sort("ENTIDAD", "AÑO")
        .group_by("ENTIDAD", maintain_order=True)
        .agg(pl.col("AÑO").alias("y"), pl.col("v"))
    )

    print("  Pre-computing market trend data…")
    mkt_wide = (
        df.group_by("ENTIDAD", "AÑO", "TIPO_MERCADO")
        .agg((pl.col("VALOR_PRODUCCION").sum() / 1e9).round(2).alias("v"))
        .pivot(values="v", index=["ENTIDAD", "AÑO"], on="TIPO_MERCADO")
        .fill_null(0)
        .sort("ENTIDAD", "AÑO")
        .group_by("ENTIDAD", maintain_order=True)
        .agg(pl.col("AÑO").alias("y"), pl.col("Nacional"), pl.col("Exportación"))
    )

    print("  Pre-computing top crops data…")
    top_crops = (
        df.group_by("ENTIDAD", "CULTIVO")
        .agg((pl.col("VALOR_PRODUCCION").sum() / 1e9).round(2).alias("v"))
        .sort("v", descending=True)
        .group_by("ENTIDAD", maintain_order=True)
        .agg(pl.col("CULTIVO").head(10).alias("n"), pl.col("v").head(10))
    )

    print("  Pre-computing top export crops data…")
    top_export = (
        df.filter(pl.col("TIPO_MERCADO") == "Exportación")
        .group_by("ENTIDAD", "CULTIVO")
        .agg((pl.col("VALOR_PRODUCCION").sum() / 1e9).round(2).alias("v"))
        .sort("v", descending=True)
        .group_by("ENTIDAD", maintain_order=True)
        .agg(pl.col("CULTIVO").head(10).alias("n"), pl.col("v").head(10))
    )

    print("  Pre-computing siniestro data…")
    siniestro = (
        df.group_by("ENTIDAD", "AÑO")
        .agg(
            pl.col("SUPERFICIE_SINIESTRADA").sum().alias("sin"),
            pl.col("SUPERFICIE_SEMBRADA").sum().alias("sem"),
        )
        .with_columns(
            (pl.col("sin") / pl.col("sem") * 100).round(2).alias("t")
        )
        .sort("ENTIDAD", "AÑO")
        .group_by("ENTIDAD", maintain_order=True)
        .agg(pl.col("AÑO").alias("y"), pl.col("t").alias("v"))
    )

    # Build indices keyed by state
    mkt_idx     = {r["ENTIDAD"]: r for r in mkt_wide.iter_rows(named=True)}
    crops_idx   = {r["ENTIDAD"]: r for r in top_crops.iter_rows(named=True)}
    export_idx  = {r["ENTIDAD"]: r for r in top_export.iter_rows(named=True)}
    sin_idx     = {r["ENTIDAD"]: r for r in siniestro.iter_rows(named=True)}

    result = {}
    for row in trend.iter_rows(named=True):
        e  = row["ENTIDAD"]
        de = df.filter(pl.col("ENTIDAD") == e)
        v, vol, nc, rd = compute_kpis(de)

        mk  = mkt_idx.get(e, {"y": [], "Nacional": [], "Exportación": []})
        tc  = crops_idx.get(e, {"n": [], "v": []})
        ex  = export_idx.get(e, {"n": [], "v": []})
        si  = sin_idx.get(e, {"y": [], "v": []})

        result[e] = {
            "t":   {"y": row["y"], "v": row["v"]},
            "mk":  {"y": mk["y"], "n": mk["Nacional"], "e": mk["Exportación"]},
            "tc":  {"n": tc["n"], "v": tc["v"]},
            "ex":  {"n": ex["n"], "v": ex["v"]},
            "sin": {"y": si["y"], "v": si["v"]},
            "kpi": {"vl": _fmt_pesos(v), "vo": f"{vol/1e6:.1f}M ton",
                    "cu": nc, "re": round(rd, 1)},
        }

    # __all__ entry — full dataset
    all_trend = (
        df.group_by("AÑO")
        .agg((pl.col("VALOR_PRODUCCION").sum() / 1e9).round(2).alias("v"))
        .sort("AÑO")
    )
    all_mkt = (
        df.group_by("AÑO", "TIPO_MERCADO")
        .agg((pl.col("VALOR_PRODUCCION").sum() / 1e9).round(2).alias("v"))
        .pivot(values="v", index="AÑO", on="TIPO_MERCADO")
        .fill_null(0)
        .sort("AÑO")
    )
    all_crops = (
        df.group_by("CULTIVO")
        .agg((pl.col("VALOR_PRODUCCION").sum() / 1e9).round(2).alias("v"))
        .sort("v", descending=True)
        .head(10)
    )
    all_export = (
        df.filter(pl.col("TIPO_MERCADO") == "Exportación")
        .group_by("CULTIVO")
        .agg((pl.col("VALOR_PRODUCCION").sum() / 1e9).round(2).alias("v"))
        .sort("v", descending=True)
        .head(10)
    )
    all_sin = (
        df.group_by("AÑO")
        .agg(
            pl.col("SUPERFICIE_SINIESTRADA").sum().alias("sin"),
            pl.col("SUPERFICIE_SEMBRADA").sum().alias("sem"),
        )
        .with_columns((pl.col("sin") / pl.col("sem") * 100).round(2).alias("t"))
        .sort("AÑO")
    )

    result["__all__"] = {
        "t":   {"y": all_trend["AÑO"].to_list(), "v": all_trend["v"].to_list()},
        "mk":  {"y": all_mkt["AÑO"].to_list(),
                "n": all_mkt["Nacional"].to_list(),
                "e": all_mkt["Exportación"].to_list()},
        "tc":  {"n": all_crops["CULTIVO"].to_list(), "v": all_crops["v"].to_list()},
        "ex":  {"n": all_export["CULTIVO"].to_list(), "v": all_export["v"].to_list()},
        "sin": {"y": all_sin["AÑO"].to_list(), "v": all_sin["t"].to_list()},
        "kpi": {"vl": _fmt_pesos(valor), "vo": f"{volumen/1e6:.1f}M ton",
                "cu": n_cult, "re": round(rend, 1)},
    }

    return json.dumps(result, ensure_ascii=False, separators=(",", ":"))


# ── Pre-compute per-crop data for the Explorador tab ─────────────────────────

def build_crop_data(df: pl.DataFrame) -> str:
    """
    Returns JSON keyed by crop name.
    Compact keys:
      t → trend by top 8 states  {s:[names], y:[years], v:[[vals_per_state]]}
      e → estados bar chart       {n:[names_asc], v:[vals_B], s:[sup], d:[desde], h:[hasta]}
      p → precio line             {y:[years], v:[prices]}
    """
    print("  Pre-computing crop top-8 states…")
    top8_per_crop = (
        df.group_by("CULTIVO", "ENTIDAD")
        .agg(pl.col("VALOR_PRODUCCION").sum().alias("tot"))
        .sort("CULTIVO", "tot", descending=[False, True])
        .group_by("CULTIVO", maintain_order=True)
        .agg(pl.col("ENTIDAD").head(8).alias("states"))
    )
    top8_idx = {r["CULTIVO"]: r["states"] for r in top8_per_crop.iter_rows(named=True)}

    print("  Pre-computing crop yearly trend…")
    trend_raw = (
        df.group_by("CULTIVO", "AÑO", "ENTIDAD")
        .agg((pl.col("VALOR_PRODUCCION").sum() / 1e9).round(3).alias("v"))
        .sort("CULTIVO", "AÑO")
    )
    trend_by_crop: dict = {}
    for r in trend_raw.iter_rows(named=True):
        c = r["CULTIVO"]
        if c not in trend_by_crop:
            trend_by_crop[c] = {}
        yr = r["AÑO"]
        if yr not in trend_by_crop[c]:
            trend_by_crop[c][yr] = {}
        trend_by_crop[c][yr][r["ENTIDAD"]] = r["v"]

    print("  Pre-computing crop estado aggregates…")
    estados_raw = (
        df.group_by("CULTIVO", "ENTIDAD")
        .agg(
            (pl.col("VALOR_PRODUCCION").sum() / 1e9).round(3).alias("v"),
            pl.col("SUPERFICIE_SEMBRADA").sum().alias("s"),
            pl.col("AÑO").min().alias("d"),
            pl.col("AÑO").max().alias("h"),
        )
        .sort("CULTIVO", "v")
    )
    estados_by_crop: dict = {}
    for r in estados_raw.iter_rows(named=True):
        c = r["CULTIVO"]
        if c not in estados_by_crop:
            estados_by_crop[c] = {"n": [], "v": [], "s": [], "d": [], "h": []}
        e = estados_by_crop[c]
        e["n"].append(r["ENTIDAD"])
        e["v"].append(r["v"])
        e["s"].append(int(r["s"]))
        e["d"].append(r["d"])
        e["h"].append(r["h"])

    print("  Pre-computing crop prices…")
    precio_raw = (
        df.filter(pl.col("PRECIO_MEDIO_RURAL") > 0)
        .group_by("CULTIVO", "AÑO")
        .agg(pl.col("PRECIO_MEDIO_RURAL").mean().round(2).alias("p"))
        .sort("CULTIVO", "AÑO")
    )
    precio_by_crop: dict = {}
    for r in precio_raw.iter_rows(named=True):
        c = r["CULTIVO"]
        if c not in precio_by_crop:
            precio_by_crop[c] = {"y": [], "v": []}
        precio_by_crop[c]["y"].append(r["AÑO"])
        precio_by_crop[c]["v"].append(r["p"])

    print("  Assembling crop JSON…")
    result = {}
    for cultivo in sorted(df["CULTIVO"].unique().to_list()):
        top8  = top8_idx.get(cultivo, [])
        cyear = trend_by_crop.get(cultivo, {})
        years = sorted(cyear.keys())
        state_vals = [
            [round(cyear.get(yr, {}).get(s, 0.0), 3) for yr in years]
            for s in top8
        ]
        result[cultivo] = {
            "t": {"s": top8, "y": years, "v": state_vals},
            "e": estados_by_crop.get(cultivo, {"n": [], "v": [], "s": [], "d": [], "h": []}),
            "p": precio_by_crop.get(cultivo, {"y": [], "v": []}),
        }

    return json.dumps(result, ensure_ascii=False, separators=(",", ":"))


# ── Build charts ──────────────────────────────────────────────────────────────

print("Rendering Plotly charts…")
chart_prod_anual  = div(fig_produccion_anual(d),       "chart-prod-anual",  first=True)
chart_top_cults   = div(fig_top_cultivos_valor(d),     "chart-top-cults")
chart_mapa        = div(fig_mapa_estados(d),           "chart-mapa")
chart_top_estados = div(fig_top_estados(d),            "chart-top-estados")
chart_rend_est    = div(fig_rendimiento_estados(d),    "chart-rend-est")
chart_treemap     = div(fig_treemap_cultivos(d),       "chart-treemap")
chart_scatter     = div(fig_scatter_rendimiento_precio(d), "chart-scatter")
chart_evolucion   = div(fig_evolucion_top_cultivos(d), "chart-evolucion")
chart_tech_trend  = div(fig_tecnologia_trend(d),       "chart-tech-trend")
chart_rend_tech   = div(fig_rendimiento_por_tecnologia(d), "chart-rend-tech")
chart_tipo_donut  = div(fig_produccion_tipo(d),        "chart-tipo-donut")
chart_mkt_trend   = div(fig_mercado_trend(d),          "chart-mkt-trend")
chart_top_export  = div(fig_top_cultivos_exportacion(d), "chart-top-export")
chart_exp_estado  = div(fig_exportacion_por_estado(d), "chart-exp-estado")
chart_sin_anual   = div(fig_siniestro_anual(d),        "chart-sin-anual")
chart_sin_est     = div(fig_siniestro_estados(d),      "chart-sin-est")
chart_sin_scatter = div(fig_siniestro_scatter(d),      "chart-sin-scatter")

DEFAULT_CROP = "Aguacate"
chart_cult_trend   = div(fig_cultivo_trend(d,   DEFAULT_CROP), "chart-cult-trend")
chart_cult_estados = div(fig_cultivo_estados(d, DEFAULT_CROP), "chart-cult-estados")
chart_cult_precio  = div(fig_cultivo_precio(d,  DEFAULT_CROP), "chart-cult-precio")

# ── Pre-compute JSON data ─────────────────────────────────────────────────────

print("Pre-computing per-state data…")
estado_data_json = build_estado_data(df)

print("Pre-computing per-crop data…")
cultivo_data_json = build_crop_data(df)

# ── Dropdown options ──────────────────────────────────────────────────────────

all_states = sorted(df["ENTIDAD"].unique().to_list())
opts_html = "\n".join(
    f'<option value="{s.replace(chr(34), "&quot;")}">{s}</option>'
    for s in all_states
)

crop_opts_html = "\n".join(
    f'<option value="{c.replace(chr(34), "&quot;")}"{"  selected" if c == DEFAULT_CROP else ""}>{c}</option>'
    for c in ALL_CROPS
)

# ── Assemble layout pieces ────────────────────────────────────────────────────

kpis_html = "".join([
    kpi("Valor total de producción", _fmt_pesos(valor),        f"{YEAR_MIN}–{YEAR_MAX}", "kpi-valor"),
    kpi("Volumen total producido",   f"{volumen/1e6:.1f}M ton","volumen producido total","kpi-vol"),
    kpi("Cultivos activos",          f"{n_cult:,}",            "cultivos únicos",        "kpi-cult"),
    kpi("Rendimiento promedio",      f"{rend:.1f} ton/ha",     "cultivos en toneladas",  "kpi-rend"),
])

tabs_nav = "".join([
    tab_btn("panorama",   "📈 Panorama",        active=True),
    tab_btn("geografia",  "🗺 Geografía"),
    tab_btn("cultivos",   "🌱 Cultivos"),
    tab_btn("tecnologia", "⚙ Tecnología"),
    tab_btn("mercado",    "🛒 Mercado"),
    tab_btn("riesgo",     "⚠ Riesgo Agrícola"),
    tab_btn("explorador", "🌾 Explorador"),
])

tabs_content = "".join([
    tab_pane("panorama", "".join([
        row((chart_prod_anual, 12)),
        row((chart_top_cults,  12)),
    ]), active=True),
    tab_pane("geografia", "".join([
        row((chart_mapa,        12)),
        row((chart_top_estados,  6), (chart_rend_est, 6)),
    ])),
    tab_pane("cultivos", "".join([
        row((chart_treemap, 12)),
        row((chart_scatter,  6), (chart_evolucion, 6)),
    ])),
    tab_pane("tecnologia", "".join([
        row((chart_tech_trend, 12)),
        row((chart_rend_tech,   6), (chart_tipo_donut, 6)),
    ])),
    tab_pane("mercado", "".join([
        row((chart_mkt_trend, 12)),
        row((chart_top_export, 6), (chart_exp_estado, 6)),
    ])),
    tab_pane("riesgo", "".join([
        row((chart_sin_anual,   12)),
        row((chart_sin_est,      6), (chart_sin_scatter, 6)),
    ])),
    tab_pane("explorador", "".join([
        f"""<div style="background:#1E293B;border:1px solid #334155;border-radius:8px;
                        padding:14px 18px;margin-bottom:16px;">
          <label style="color:#94A3B8;font-size:.82rem;margin-bottom:8px;display:block;">
            Cultivo
            <small style="color:#64748B"> — selecciona para ver su evolución y distribución geográfica</small>
          </label>
          <select id="cultivo-select" class="form-select dark-select">
            <option value="">— selecciona un cultivo —</option>
            {crop_opts_html}
          </select>
        </div>""",
        row((chart_cult_trend,   12)),
        row((chart_cult_estados,  6), (chart_cult_precio, 6)),
    ])),
])

insights_html = "".join([
    insight_card("Sinaloa, Jalisco y Sonora: el triángulo productivo",
        "Estos tres estados concentran más del 30% del valor agrícola nacional. "
        "Sinaloa domina en volumen de granos y hortalizas de exportación."),
    insight_card("Invernadero: hasta 5× más rendimiento",
        "Los cultivos bajo invernadero superan al campo abierto en rendimiento por hectárea. "
        "Su adopción ha crecido sostenidamente desde los 2000."),
    insight_card("Exportación concentrada en pocos cultivos",
        "Aguacate, tomate y berries dominan las exportaciones. "
        "El valor de exportación agrícola se multiplicó 10× desde los 90."),
])

# ── JavaScript ────────────────────────────────────────────────────────────────

js = """
const CFG = {responsive: true, displayModeBar: false};
const NAC_COLOR = "#3BB273";
const EXP_COLOR = "#F4A261";
const BLUE      = "#2E86AB";

function getLayout(id) { return document.getElementById(id).layout; }

function updateCharts(estado) {
    const D = ESTADO_DATA[estado];
    if (!D) { console.error("No data for:", JSON.stringify(estado)); return; }

    // Annual production trend
    Plotly.newPlot("chart-prod-anual", [{
        type:"scatter", mode:"lines+markers", x:D.t.y, y:D.t.v,
        line:{color:BLUE,width:2}, fill:"tozeroy", fillcolor:"rgba(46,134,171,0.15)",
        hovertemplate:"Año: %{x}<br>Valor: $%{y:.1f}B<extra></extra>"
    }], getLayout("chart-prod-anual"), CFG);

    // Top 10 crops bar (horizontal)
    const tcN = [...D.tc.n].reverse();
    const tcV = [...D.tc.v].reverse();
    Plotly.newPlot("chart-top-cults", [{
        type:"bar", orientation:"h", x:tcV, y:tcN,
        marker:{color:tcV, colorscale:[[0,"#1E3A5F"],[1,BLUE]]},
        text: tcV.map(v => "$" + v.toFixed(1) + "B"),
        textposition:"outside",
        hovertemplate:"%{y}<br>Valor: $%{x:.1f}B<extra></extra>"
    }], getLayout("chart-top-cults"), CFG);

    // Market trend area
    Plotly.newPlot("chart-mkt-trend", [
        {type:"scatter",mode:"lines",name:"Exportación",stackgroup:"1",
         x:D.mk.y,y:D.mk.e,line:{color:EXP_COLOR},fillcolor:"rgba(244,162,97,0.5)"},
        {type:"scatter",mode:"lines",name:"Nacional",stackgroup:"1",
         x:D.mk.y,y:D.mk.n,line:{color:NAC_COLOR},fillcolor:"rgba(59,178,115,0.5)"}
    ], getLayout("chart-mkt-trend"), CFG);

    // Top export crops (horizontal)
    const exN = [...D.ex.n].reverse();
    const exV = [...D.ex.v].reverse();
    Plotly.newPlot("chart-top-export", [{
        type:"bar", orientation:"h", x:exV, y:exN,
        marker:{color:exV, colorscale:[[0,"#5C2D00"],[1,EXP_COLOR]]},
        text: exV.map(v => "$" + v.toFixed(1) + "B"),
        textposition:"outside",
        hovertemplate:"%{y}<br>Valor: $%{x:.1f}B<extra></extra>"
    }], getLayout("chart-top-export"), CFG);

    // Siniestro anual
    Plotly.newPlot("chart-sin-anual", [{
        type:"scatter", mode:"lines+markers", x:D.sin.y, y:D.sin.v,
        line:{color:"#E84855",width:2}, fill:"tozeroy", fillcolor:"rgba(232,72,85,0.12)",
        hovertemplate:"Año: %{x}<br>Tasa: %{y:.2f}%<extra></extra>"
    }], getLayout("chart-sin-anual"), CFG);

    // KPIs
    document.getElementById("kpi-valor").textContent     = D.kpi.vl;
    document.getElementById("kpi-valor-sub").textContent = estado !== "__all__" ? estado : "1980–2024";
    document.getElementById("kpi-vol").textContent       = D.kpi.vo;
    document.getElementById("kpi-cult").textContent      = D.kpi.cu.toLocaleString();
    document.getElementById("kpi-rend").textContent      = D.kpi.re + " ton/ha";

    // Badge
    document.getElementById("filter-badge").textContent = estado;
    document.getElementById("filter-notice").style.display = "flex";
}

function applyWhenReady(estado, attempt) {
    attempt = attempt || 0;
    if (attempt > 100) { updateCharts(estado); return; }
    const el = document.getElementById("chart-prod-anual");
    if (el && el.querySelector("svg.main-svg")) {
        updateCharts(estado);
    } else {
        setTimeout(() => applyWhenReady(estado, attempt + 1), 100);
    }
}

// ── Crop explorer ─────────────────────────────────────────────────────────────

function updateCultivoCharts(cultivo) {
    const D = CULTIVO_DATA[cultivo];
    if (!D) { console.error("No data for crop:", cultivo); return; }

    // Stacked area: trend by top states
    const trendTraces = D.t.s.map((state, i) => ({
        type:"scatter", mode:"lines", stackgroup:"1",
        name: state, x: D.t.y, y: D.t.v[i],
        hovertemplate: state + "<br>Año: %{x}<br>$%{y:.3f}B<extra></extra>"
    }));
    const trendLayout = Object.assign({}, getLayout("chart-cult-trend"), {
        title: {text: "Evolución de " + cultivo + " — valor por estado (top 8)"}
    });
    Plotly.newPlot("chart-cult-trend", trendTraces, trendLayout, CFG);

    // Horizontal bar: estados
    const estLayout = Object.assign({}, getLayout("chart-cult-estados"), {
        title: {text: "Estados productores de " + cultivo},
        height: Math.max(300, D.e.n.length * 28 + 80),
    });
    Plotly.newPlot("chart-cult-estados", [{
        type:"bar", orientation:"h",
        x: D.e.v, y: D.e.n,
        marker: {color: D.e.v, colorscale:[[0,"#1E3A5F"],[1,BLUE]]},
        text: D.e.v.map(v => "$" + v.toFixed(2) + "B"),
        textposition: "outside",
        customdata: D.e.s.map((s, i) => [s, D.e.d[i], D.e.h[i]]),
        hovertemplate: "<b>%{y}</b><br>Valor: $%{x:.3f}B<br>Superficie: %{customdata[0]:,} ha<br>%{customdata[1]}–%{customdata[2]}<extra></extra>"
    }], estLayout, CFG);

    // Price line
    const precioLayout = Object.assign({}, getLayout("chart-cult-precio"), {
        title: {text: "Precio medio rural de " + cultivo + " ($/ton)"}
    });
    Plotly.newPlot("chart-cult-precio", [{
        type:"scatter", mode:"lines+markers",
        x: D.p.y, y: D.p.v,
        line:{color:"#F4A261", width:2},
        fill:"tozeroy", fillcolor:"rgba(244,162,97,0.12)",
        hovertemplate:"Año: %{x}<br>Precio: $%{y:,.0f}/ton<extra></extra>"
    }], precioLayout, CFG);
}

function applyWhenReadyCultivo(cultivo, attempt) {
    attempt = attempt || 0;
    if (attempt > 100) { updateCultivoCharts(cultivo); return; }
    const el = document.getElementById("chart-cult-trend");
    if (el && el.querySelector("svg.main-svg")) {
        updateCultivoCharts(cultivo);
    } else {
        setTimeout(() => applyWhenReadyCultivo(cultivo, attempt + 1), 100);
    }
}

// ── Event listeners ───────────────────────────────────────────────────────────

document.getElementById("estado-select").addEventListener("change", function () {
    const url = new URL(window.location.href);
    if (this.value) {
        url.searchParams.set("estado", this.value);
    } else {
        url.searchParams.delete("estado");
    }
    window.location.href = url.toString();
});

document.getElementById("cultivo-select").addEventListener("change", function () {
    const url = new URL(window.location.href);
    if (this.value) {
        url.searchParams.set("cultivo", this.value);
    } else {
        url.searchParams.delete("cultivo");
    }
    window.location.href = url.toString();
});

document.getElementById("clear-filter").addEventListener("click", function () {
    window.location.href = window.location.pathname.replace(/\\/+$/, "") + "/productos_agricolas.html";
});

window.addEventListener("load", function () {
    const params = new URLSearchParams(window.location.search);

    const estado = params.get("estado");
    if (estado && ESTADO_DATA[estado]) {
        document.getElementById("estado-select").value = estado;
        applyWhenReady(estado);
    }

    const cultivo = params.get("cultivo");
    if (cultivo && CULTIVO_DATA[cultivo]) {
        document.getElementById("cultivo-select").value = cultivo;
        applyWhenReadyCultivo(cultivo);
    }
});

document.querySelectorAll("button[data-bs-toggle=tab]").forEach(btn => {
    btn.addEventListener("shown.bs.tab",
        () => window.dispatchEvent(new Event("resize")));
});
"""

# ── Full HTML ─────────────────────────────────────────────────────────────────

html = f"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Producción Agrícola de México</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css"
        rel="stylesheet" crossorigin="anonymous">
  <style>
    body        {{ background:#0F172A; color:#CBD5E1; padding:24px;
                  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }}
    .nav-tabs   {{ border-bottom:1px solid #334155; }}
    .nav-link   {{ color:#94A3B8; background:transparent; border:none;
                  border-top:2px solid transparent; border-radius:0; padding:10px 16px; }}
    .nav-link:hover {{ color:#F8FAFC; background:#1E293B; }}
    .nav-link.active {{ color:#F8FAFC!important; background:#1E293B!important;
                        border-top:2px solid #2E86AB!important; font-weight:600; }}
    .tab-content {{ background:#0F172A; padding-top:16px; }}
    .kpi-card  {{ background:#1E293B; border:1px solid #334155; border-radius:8px;
                  padding:16px; text-align:center; height:100%; }}
    .kpi-label {{ color:#94A3B8; font-size:.8rem; margin-bottom:4px; }}
    .kpi-value {{ color:#F8FAFC; font-weight:700; margin:0; }}
    .kpi-sub   {{ color:#64748B; }}
    .filter-box {{ background:#1E293B; border:1px solid #334155; border-radius:8px;
                   padding:14px 18px; }}
    .filter-label {{ color:#94A3B8; font-size:.82rem; margin-bottom:6px; }}
    .dark-select {{ background-color:#1E293B!important; border-color:#334155!important;
                    color:#CBD5E1!important; }}
    .dark-select:focus {{ border-color:#2E86AB!important;
                          box-shadow:0 0 0 .2rem rgba(46,134,171,.25)!important; }}
    .dark-select option {{ background:#1E293B; color:#CBD5E1; }}
    #filter-notice {{ display:none; align-items:center; gap:10px;
                      background:rgba(46,134,171,.12); border:1px solid #2E86AB;
                      border-radius:6px; padding:8px 14px; font-size:.83rem; color:#94A3B8; }}
    #filter-badge  {{ background:#2E86AB; color:#fff; border-radius:4px;
                      padding:2px 8px; font-size:.78rem; }}
    #clear-filter  {{ cursor:pointer; color:#94A3B8; font-size:.75rem;
                      border:1px solid #334155; border-radius:4px; padding:2px 8px;
                      background:transparent; }}
    #clear-filter:hover {{ color:#F8FAFC; border-color:#64748B; }}
    .insight-card  {{ background:#1E293B; border:1px solid #334155; border-radius:8px;
                      padding:16px; height:100%; }}
    .insight-title {{ color:#F8FAFC; display:block; margin-bottom:4px; }}
    .insight-body  {{ color:#94A3B8; font-size:.83rem; margin:0; }}
    h6.section-title {{ color:#94A3B8; font-weight:600; margin-bottom:12px; }}
  </style>
</head>
<body>
<div class="container-fluid">

  <div class="mb-4">
    <h2 style="color:#F8FAFC;font-weight:700;margin-bottom:4px;">
      Producción Agrícola de México
    </h2>
    <p style="color:#64748B;font-size:.9rem;">
      México · 1980–2024 · 114,841 registros · 32 estados · 366 cultivos
    </p>
  </div>

  <div class="row g-3 mb-3">{kpis_html}</div>

  <div class="filter-box mb-3">
    <div class="row g-3 align-items-end">
      <div class="col-10 col-md-8">
        <p class="filter-label mb-1">Estado
          <small style="color:#64748B"> — los gráficos de tendencias y cultivos se actualizan con la selección</small>
        </p>
        <select id="estado-select" class="form-select dark-select">
          <option value="">Todos los estados</option>
          {opts_html}
        </select>
      </div>
      <div class="col-2 col-md-4 d-flex align-items-end">
        <div id="filter-notice">
          <span id="filter-badge"></span>
          <button id="clear-filter">✕ limpiar</button>
        </div>
      </div>
    </div>
  </div>

  <ul class="nav nav-tabs" role="tablist">{tabs_nav}</ul>
  <div class="tab-content">{tabs_content}</div>

  <div class="mt-4">
    <h6 class="section-title">Hallazgos clave</h6>
    <div class="row g-3">{insights_html}</div>
  </div>

</div>

<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"
        crossorigin="anonymous"></script>
<script>
const ESTADO_DATA   = {estado_data_json};
const CULTIVO_DATA  = {cultivo_data_json};
{js}
</script>
</body>
</html>"""

out = "site/productos_agricolas.html"
with open(out, "w", encoding="utf-8") as f:
    f.write(html)

size_kb = len(html.encode()) / 1024
print(f"\nWritten {out}  ({size_kb:.0f} KB)")
