"""
Exports a fully self-contained temperaturas_lluvias.html for GitHub Pages.

Estado dropdown works client-side:
  - Per-state aggregates are pre-computed in Python and embedded as JSON
  - Plotly.newPlot() swaps chart data on selection

Static charts (full dataset, never change):
  - Warming delta by state
  - Precipitation ranking by state

Dynamic charts (update on estado selection):
  - Temperature annual trend (min, media, max)
  - Temperature monthly seasonality
  - Precipitation annual total
  - Precipitation monthly seasonality
  - KPIs

Run: uv run python export_temperaturas_lluvias_html.py
"""

import json
import polars as pl
import plotly.io as pio

from dashboard.temperaturas_lluvias import (
    df, ESTADOS, MES_NOMBRES, _baseline,
    fig_warming_delta, fig_prec_state_ranking, compute_kpis,
)

CHART_CFG = {"responsive": True, "displayModeBar": False}

ALL_ENTIDADES = ["Nacional"] + ESTADOS


# ── HTML helper ───────────────────────────────────────────────────────────────

def chart_div(fig, div_id: str, first=False) -> str:
    return pio.to_html(
        fig, full_html=False,
        include_plotlyjs="cdn" if first else False,
        config=CHART_CFG,
        div_id=div_id,
    )


def kpi_card(title: str, kpi_id: str, color: str) -> str:
    return f"""<div class="col-6 col-md-3">
      <div class="kpi-card">
        <p class="kpi-label">{title}</p>
        <h3 class="kpi-value" id="{kpi_id}" style="color:{color}">—</h3>
      </div>
    </div>"""


# ── Pre-compute per-estado data ───────────────────────────────────────────────

def _temp_trend_data(d: pl.DataFrame) -> dict:
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
    return {
        "years":  anual["anio"].to_list(),
        "minima": anual["MINIMA"].to_list(),
        "media":  anual["MEDIA"].to_list(),
        "maxima": anual["MAXIMA"].to_list(),
    }


def _temp_seasonal_data(d: pl.DataFrame) -> dict:
    s = d.group_by("mes").agg(
        pl.col("MINIMA").mean().round(1),
        pl.col("MEDIA").mean().round(1),
        pl.col("MAXIMA").mean().round(1),
    ).sort("mes")
    return {
        "meses":  [MES_NOMBRES[m - 1] for m in s["mes"].to_list()],
        "minima": s["MINIMA"].to_list(),
        "media":  s["MEDIA"].to_list(),
        "maxima": s["MAXIMA"].to_list(),
    }


def _prec_trend_data(d: pl.DataFrame) -> dict:
    anual = (
        d.filter(pl.col("anio").is_between(1985, 2025))
        .group_by("anio")
        .agg(pl.col("PRECIPITACION").sum().round(1).alias("prec"))
        .sort("anio")
    )
    precs = anual["prec"].to_list()
    avg = round(sum(precs) / len(precs), 1)
    return {
        "years": anual["anio"].to_list(),
        "prec":  precs,
        "avg":   avg,
    }


def _prec_seasonal_data(d: pl.DataFrame) -> dict:
    s = d.group_by("mes").agg(pl.col("PRECIPITACION").mean().round(1)).sort("mes")
    return {
        "meses": [MES_NOMBRES[m - 1] for m in s["mes"].to_list()],
        "prec":  s["PRECIPITACION"].to_list(),
    }


def build_estado_data() -> str:
    result = {}
    for entidad in ALL_ENTIDADES:
        print(f"  Computing {entidad}…")
        d = df.filter(pl.col("ENTIDAD") == entidad)
        media, maxima, prec_anual, delta = compute_kpis(d, entidad)
        delta_sign = "+" if delta >= 0 else ""
        result[entidad] = {
            "kpi": {
                "media":      round(media, 1),
                "maxima":     round(maxima, 1),
                "prec_anual": round(prec_anual, 0),
                "delta":      round(delta, 2),
                "delta_str":  f"{delta_sign}{delta:.2f}°C",
            },
            "temp_trend":     _temp_trend_data(d),
            "temp_seasonal":  _temp_seasonal_data(d),
            "prec_trend":     _prec_trend_data(d),
            "prec_seasonal":  _prec_seasonal_data(d),
        }
    return json.dumps(result, ensure_ascii=False, separators=(",", ":"))


# ── Render static charts ──────────────────────────────────────────────────────

print("Rendering static charts…")
chart_warming  = chart_div(fig_warming_delta(),      "chart-warming",  first=True)
chart_prec_rank = chart_div(fig_prec_state_ranking(), "chart-prec-rank")

print("Pre-computing per-estado data…")
estado_data_json = build_estado_data()

# ── Estado dropdown ───────────────────────────────────────────────────────────

estado_opts = "\n".join(
    f'<option value="{e}">{e}</option>' for e in ALL_ENTIDADES
)

# ── JavaScript ────────────────────────────────────────────────────────────────

js = """
const CFG = {responsive: true, displayModeBar: false};
const BASE = {
    paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
    font: {color: "#CBD5E1"},
    legend: {bgcolor: "rgba(0,0,0,0)", font: {color: "#94A3B8"},
             orientation: "h", yanchor: "bottom", y: 1.02, xanchor: "right", x: 1},
};
const MARGIN_STD  = {t:50, b:40, l:10, r:10};
const MARGIN_WIDE = {t:40, b:40, l:10, r:80};
const AXIS = (extra) => Object.assign({gridcolor: "#334155"}, extra || {});

function deltaColor(d) {
    return d >= 2 ? "#E84855" : d >= 1 ? "#F4A261" : "#3BB273";
}

function updateCharts(estado) {
    const D = ESTADO_DATA[estado];
    if (!D) return;

    // KPIs
    document.getElementById("kpi-media").textContent    = D.kpi.media + "°C";
    document.getElementById("kpi-maxima").textContent   = D.kpi.maxima + "°C";
    document.getElementById("kpi-prec").textContent     = D.kpi.prec_anual + "mm";
    const kpiDelta = document.getElementById("kpi-delta");
    kpiDelta.textContent = D.kpi.delta_str;
    kpiDelta.style.color = deltaColor(D.kpi.delta);

    // Temperature annual trend
    Plotly.newPlot("chart-temp-trend", [
        {type:"scatter", mode:"lines", name:"Máxima", x:D.temp_trend.years, y:D.temp_trend.maxima,
         line:{color:"#E84855",width:1.5}, fill:"tonexty", fillcolor:"rgba(232,72,85,0.07)"},
        {type:"scatter", mode:"lines+markers", name:"Media", x:D.temp_trend.years, y:D.temp_trend.media,
         line:{color:"#F4A261",width:2.5}, marker:{size:4}},
        {type:"scatter", mode:"lines", name:"Mínima", x:D.temp_trend.years, y:D.temp_trend.minima,
         line:{color:"#2E86AB",width:1.5}},
    ], Object.assign({}, BASE, {
        title: {text: "Temperatura anual (1985–2025)"},
        height: 380, margin: MARGIN_STD,
        xaxis: AXIS({title:"Año", dtick:5}),
        yaxis: AXIS({ticksuffix:"°C"}),
    }), CFG);

    // Temperature monthly seasonality
    Plotly.newPlot("chart-temp-seasonal", [
        {type:"bar", name:"Máxima", x:D.temp_seasonal.meses, y:D.temp_seasonal.maxima,
         marker:{color:"#E84855"}, opacity:0.7},
        {type:"bar", name:"Media",  x:D.temp_seasonal.meses, y:D.temp_seasonal.media,
         marker:{color:"#F4A261"}},
        {type:"bar", name:"Mínima", x:D.temp_seasonal.meses, y:D.temp_seasonal.minima,
         marker:{color:"#2E86AB"}, opacity:0.7},
    ], Object.assign({}, BASE, {
        barmode: "group",
        title: {text: "Temperatura promedio por mes"},
        height: 360, margin: MARGIN_STD,
        xaxis: AXIS(),
        yaxis: AXIS({ticksuffix:"°C"}),
    }), CFG);

    // Precipitation annual
    const avg = D.prec_trend.avg;
    const barColors = D.prec_trend.prec.map(p => p >= avg ? "#2E86AB" : "#F4A261");
    Plotly.newPlot("chart-prec-trend", [
        {type:"bar", x:D.prec_trend.years, y:D.prec_trend.prec,
         marker:{color:barColors},
         hovertemplate:"<b>%{x}</b><br>Precipitación: %{y:.0f}mm<extra></extra>"},
    ], Object.assign({}, BASE, {
        shapes: [{type:"line", x0:D.prec_trend.years[0], x1:D.prec_trend.years[D.prec_trend.years.length-1],
                  y0:avg, y1:avg, line:{color:"#3BB273", width:1.5, dash:"dash"}}],
        annotations: [{xref:"paper", yref:"y", x:1, y:avg,
                       text:"Promedio " + avg + "mm", showarrow:false,
                       font:{color:"#3BB273", size:11}, xanchor:"right"}],
        title: {text: "Precipitación anual total (1985–2025)"},
        height: 380, margin: MARGIN_STD, showlegend: false,
        xaxis: AXIS({title:"Año", dtick:5}),
        yaxis: AXIS({ticksuffix:"mm"}),
    }), CFG);

    // Precipitation monthly seasonality
    const maxP = Math.max(...D.prec_seasonal.prec);
    const precColors = D.prec_seasonal.prec.map(p =>
        "rgba(46,134,171," + (0.4 + 0.6 * (p / maxP)).toFixed(2) + ")"
    );
    Plotly.newPlot("chart-prec-seasonal", [
        {type:"bar", x:D.prec_seasonal.meses, y:D.prec_seasonal.prec,
         marker:{color:precColors},
         text: D.prec_seasonal.prec.map(p => p.toFixed(0)),
         textposition:"outside",
         hovertemplate:"<b>%{x}</b><br>Precipitación: %{y:.1f}mm<extra></extra>"},
    ], Object.assign({}, BASE, {
        title: {text: "Precipitación mensual promedio"},
        height: 360, margin: MARGIN_STD, showlegend: false,
        xaxis: AXIS(),
        yaxis: AXIS({ticksuffix:"mm"}),
    }), CFG);
}

function applyWhenReady(estado, attempt) {
    attempt = attempt || 0;
    if (attempt > 120) { updateCharts(estado); return; }
    const el = document.getElementById("chart-warming");
    if (el && el.querySelector("svg.main-svg")) {
        updateCharts(estado);
    } else {
        setTimeout(() => applyWhenReady(estado, attempt + 1), 80);
    }
}

document.getElementById("estado-select").addEventListener("change", function () {
    const url = new URL(window.location.href);
    url.searchParams.set("estado", this.value);
    window.location.href = url.toString();
});

document.querySelectorAll("button[data-bs-toggle='tab']").forEach(btn => {
    btn.addEventListener("shown.bs.tab", () => window.dispatchEvent(new Event("resize")));
});

window.addEventListener("load", function () {
    const estado = new URLSearchParams(window.location.search).get("estado") || "Nacional";
    const sel = document.getElementById("estado-select");
    if (estado !== "Nacional") {
        sel.value = estado;
        document.getElementById("filter-notice").style.display = "flex";
        document.getElementById("filter-badge").textContent = estado;
    }
    const hash = window.location.hash;
    if (hash) {
        const btn = document.querySelector("button[data-bs-target='" + hash + "']");
        if (btn) bootstrap.Tab.getOrCreateInstance(btn).show();
    }
    applyWhenReady(estado);
});

document.getElementById("clear-filter").addEventListener("click", function () {
    window.location.href = window.location.pathname.split("?")[0];
});

document.querySelectorAll("button[data-bs-toggle='tab']").forEach(btn => {
    btn.addEventListener("shown.bs.tab", function () {
        history.replaceState(null, "", window.location.search + this.dataset.bsTarget);
    });
});
"""

# ── Assemble HTML ─────────────────────────────────────────────────────────────

kpis_html = "".join([
    kpi_card("Temp. media (2015–2025)",       "kpi-media",  "#F4A261"),
    kpi_card("Temp. máxima (2015–2025)",      "kpi-maxima", "#E84855"),
    kpi_card("Precipitación anual promedio",  "kpi-prec",   "#2E86AB"),
    kpi_card("Calentamiento vs 1985–1994",    "kpi-delta",  "#F4A261"),
])

html_out = f"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Temperaturas y Precipitaciones — México</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css"
        rel="stylesheet" crossorigin="anonymous">
  <style>
    body        {{ background:#0F172A; color:#CBD5E1; padding:24px;
                  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }}
    .kpi-card   {{ background:#1E293B; border:1px solid #334155; border-radius:8px;
                  padding:16px; text-align:center; height:100%; }}
    .kpi-label  {{ color:#94A3B8; font-size:.78rem; margin-bottom:4px; }}
    .kpi-value  {{ font-weight:700; margin:0; font-size:1.4rem; }}
    .filter-box {{ background:#1E293B; border:1px solid #334155; border-radius:8px;
                  padding:14px 18px; }}
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
    .nav-tabs   {{ border-bottom:1px solid #334155; }}
    .nav-link   {{ color:#94A3B8; background:transparent; border:none;
                  border-top:2px solid transparent; border-radius:0; padding:10px 18px; }}
    .nav-link:hover  {{ color:#F8FAFC; background:#1E293B; }}
    .nav-link.active {{ color:#F8FAFC!important; background:#1E293B!important;
                       border-top:2px solid #2E86AB!important; font-weight:600; }}
    .tab-content {{ background:#0F172A; padding-top:20px; }}
  </style>
</head>
<body>
<div class="container-fluid">

  <div class="mb-4">
    <h2 style="color:#F8FAFC;font-weight:700;margin-bottom:4px;">
      Temperaturas y Precipitaciones — México 1985–2025
    </h2>
    <p style="color:#64748B;font-size:.9rem;">
      Temperatura mínima, media y máxima, y precipitación mensual por estado · 16,368 registros
    </p>
  </div>

  <div class="row g-3 mb-3">{kpis_html}</div>

  <div class="filter-box mb-3">
    <div class="row g-3 align-items-end">
      <div class="col-10 col-md-4">
        <p style="color:#94A3B8;font-size:.82rem;margin-bottom:6px;">Estado / Región
          <small style="color:#64748B"> — filtra la tendencia y estacionalidad; los rankings siempre muestran todos los estados</small>
        </p>
        <select id="estado-select" class="form-select dark-select">
          {estado_opts}
        </select>
      </div>
      <div class="col-2 col-md-8 d-flex align-items-end">
        <div id="filter-notice">
          <span id="filter-badge"></span>
          <button id="clear-filter">✕ limpiar</button>
        </div>
      </div>
    </div>
  </div>

  <ul class="nav nav-tabs mb-0" role="tablist">
    <li class="nav-item" role="presentation">
      <button class="nav-link active" data-bs-toggle="tab" data-bs-target="#tab-temp"
              type="button" role="tab">Temperatura</button>
    </li>
    <li class="nav-item" role="presentation">
      <button class="nav-link" data-bs-toggle="tab" data-bs-target="#tab-prec"
              type="button" role="tab">Precipitación</button>
    </li>
  </ul>

  <div class="tab-content">

    <div class="tab-pane fade show active" id="tab-temp" role="tabpanel">
      <div class="row g-3 mb-3">
        <div class="col-12 col-md-7">
          <div id="chart-temp-trend" style="min-height:380px"></div>
        </div>
        <div class="col-12 col-md-5">
          <div id="chart-temp-seasonal" style="min-height:360px"></div>
        </div>
      </div>
      <div class="row g-3">
        <div class="col-12">{chart_warming}</div>
      </div>
    </div>

    <div class="tab-pane fade" id="tab-prec" role="tabpanel">
      <div class="row g-3 mb-3">
        <div class="col-12 col-md-7">
          <div id="chart-prec-trend" style="min-height:380px"></div>
        </div>
        <div class="col-12 col-md-5">
          <div id="chart-prec-seasonal" style="min-height:360px"></div>
        </div>
      </div>
      <div class="row g-3">
        <div class="col-12">{chart_prec_rank}</div>
      </div>
    </div>

  </div>

</div>

<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"
        crossorigin="anonymous"></script>
<script>
const ESTADO_DATA = {estado_data_json};
{js}
</script>
</body>
</html>"""

out = "site/temperaturas_lluvias.html"
with open(out, "w", encoding="utf-8") as f:
    f.write(html_out)

size_kb = len(html_out.encode()) / 1024
print(f"\nWritten {out}  ({size_kb:.0f} KB)")
