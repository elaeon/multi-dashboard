"""
Exports a fully self-contained suicidios_entidad.html for GitHub Pages.

Single filter: entidad (Nacional + 32 states). Year range fixed to 1990–2024.

Static  (always full dataset): state ranking bar, choropleth map (2024), extremes callout.
Dynamic (per entidad):         rate trend line, annual counts bar, KPI cards.

Run: uv run python static_dashboard/export_suicidios_entidad_html.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import polars as pl
import plotly.io as pio

from dashboard.suicidios_entidad import (
    df, STATE_LIST, MIN_YEAR, MAX_YEAR,
    fig_map, fig_states_rank, compute_kpis, state_extremes,
)

CHART_CFG = {"responsive": True, "displayModeBar": False}
ENTIDADES  = ["Nacional"] + STATE_LIST


# ── HTML helpers ──────────────────────────────────────────────────────────────

def chart_div(fig, div_id: str, first=False) -> str:
    return pio.to_html(
        fig, full_html=False,
        include_plotlyjs="cdn" if first else False,
        config=CHART_CFG,
        div_id=div_id,
    )


def kpi_card(title: str, kpi_id: str, color: str) -> str:
    return (
        f'<div class="col-6 col-md-3"><div class="kpi-card">'
        f'<p class="kpi-label">{title}</p>'
        f'<h3 class="kpi-value" id="{kpi_id}" style="color:{color}">—</h3>'
        f'</div></div>'
    )


# ── Pre-compute per-entidad payload ───────────────────────────────────────────

def build_payload(d: pl.DataFrame) -> dict:
    s = d.sort("AÑO")
    total, rate, rate_h, rate_m = compute_kpis(d)
    years         = s["AÑO"].to_list()
    first_row     = s.head(1)
    last_row      = s.tail(1)
    h0 = first_row["TASA_HOMBRES"].item() or 0
    m0 = first_row["TASA_MUJERES"].item() or 0.01
    h1 = last_row["TASA_HOMBRES"].item() or 0
    m1 = last_row["TASA_MUJERES"].item() or 0.01
    y0_yr = first_row["AÑO"].item()
    y1_yr = last_row["AÑO"].item()
    return {
        "kpi": {"total": total, "rate": rate, "rate_h": rate_h, "rate_m": rate_m},
        "trend": {
            "years":   years,
            "hombres": s["TASA_HOMBRES"].to_list(),
            "mujeres": s["TASA_MUJERES"].to_list(),
            "total":   s["TASA_TOTAL"].to_list(),
        },
        "bar": {
            "years":   years,
            "hombres": s["HOMBRES"].to_list(),
            "mujeres": s["MUJERES"].to_list(),
        },
        "gap_text": (
            f"Brecha H/M: <b>{h1/m1:.1f}x</b> en {y1_yr}"
            f"<br>(era {h0/m0:.1f}x en {y0_yr})"
        ),
    }


print("Pre-computing per-entidad data…")
all_data = {}
for ent in ENTIDADES:
    all_data[ent] = build_payload(df.filter(pl.col("ENTIDAD") == ent))
    print(f"  {ent}")
all_data_json = json.dumps(all_data, ensure_ascii=False, separators=(",", ":"))


# ── Static charts ─────────────────────────────────────────────────────────────

d_states = df.filter(
    ~pl.col("ENTIDAD").is_in(["Nacional", "Extranjero", "Not specified"])
)

print("Rendering static charts…")
# Ranking bar first (CDN) — regular SVG chart, anchors applyWhenReady
chart_states_rank = chart_div(fig_states_rank(d_states), "chart-states-rank", first=True)
# Map second — reuses CDN
chart_map_html    = chart_div(fig_map(d_states, MAX_YEAR), "chart-map")


# ── State extremes (static HTML) ──────────────────────────────────────────────

max_st, max_rate, min_st, min_rate = state_extremes(d_states)
extremes_html = (
    f'<div class="col-12 col-md-6"><div class="kpi-card">'
    f'<p class="kpi-label">Entidad con mayor tasa promedio ({MIN_YEAR}–{MAX_YEAR})</p>'
    f'<span style="color:#E84855;font-weight:700;font-size:1rem;">{max_st}</span>'
    f'<span style="color:#CBD5E1;font-size:.9rem;">&nbsp;&nbsp;{max_rate:.1f}/100k</span>'
    f'</div></div>'
    f'<div class="col-12 col-md-6"><div class="kpi-card">'
    f'<p class="kpi-label">Entidad con menor tasa promedio ({MIN_YEAR}–{MAX_YEAR})</p>'
    f'<span style="color:#3BB273;font-weight:700;font-size:1rem;">{min_st}</span>'
    f'<span style="color:#CBD5E1;font-size:.9rem;">&nbsp;&nbsp;{min_rate:.1f}/100k</span>'
    f'</div></div>'
)


# ── Dropdown options ──────────────────────────────────────────────────────────

entidad_opts = '<option value="Nacional">Nacional</option>\n' + "\n".join(
    f'<option value="{e}">{e}</option>' for e in STATE_LIST
)


# ── JavaScript ────────────────────────────────────────────────────────────────

js = """
const CFG  = {responsive: true, displayModeBar: false};
const BASE = {
    paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
    font: {color: "#CBD5E1"},
};

const CRISIS = [
    {x: 2020, label: "COVID-19",         color: "#F4A261"},
    {x: 2023, label: "Récord histórico", color: "#E84855"},
];

function crisisMarkers(years) {
    var ymin = Math.min.apply(null, years), ymax = Math.max.apply(null, years);
    var shapes = [], annotations = [];
    CRISIS.forEach(function(c) {
        if (c.x < ymin || c.x > ymax) return;
        shapes.push({
            type:"line", x0:c.x, x1:c.x, y0:0, y1:1, yref:"paper",
            line:{color:c.color, width:1, dash:"dash"}, opacity:0.7,
        });
        annotations.push({
            x:c.x, y:0.97, xref:"x", yref:"paper",
            text:c.label, showarrow:false, textangle:-90,
            font:{size:10, color:c.color}, yanchor:"top", xanchor:"right",
        });
    });
    return {shapes:shapes, annotations:annotations};
}

function updateCharts(entidad) {
    var D = ALL_DATA[entidad];
    if (!D) return;

    document.getElementById("kpi-total").textContent  = D.kpi.total.toLocaleString("es-MX");
    document.getElementById("kpi-rate").textContent   = D.kpi.rate   !== null ? D.kpi.rate.toFixed(2)   : "—";
    document.getElementById("kpi-rate-h").textContent = D.kpi.rate_h !== null ? D.kpi.rate_h.toFixed(2) : "—";
    document.getElementById("kpi-rate-m").textContent = D.kpi.rate_m !== null ? D.kpi.rate_m.toFixed(2) : "—";

    var years   = D.trend.years;
    var markers = crisisMarkers(years);

    Plotly.newPlot("chart-trend", [
        {type:"scatter", x:years, y:D.trend.hombres, name:"Hombres", line:{color:"#2E86AB", width:2}},
        {type:"scatter", x:years, y:D.trend.mujeres, name:"Mujeres", line:{color:"#F4A261", width:2}},
        {type:"scatter", x:years, y:D.trend.total,   name:"Total",   line:{color:"#3BB273", width:2, dash:"dot"}},
    ], Object.assign({}, BASE, {
        title: {text:"Tasa de suicidio por 100,000 habitantes"},
        height: 380,
        xaxis: {gridcolor:"#334155"},
        yaxis: {gridcolor:"#334155", title:"Tasa por 100k hab."},
        legend: {orientation:"h", y:-0.18, x:0},
        margin: {t:40, b:70, l:10, r:10},
        shapes: markers.shapes,
        annotations: markers.annotations.concat([{
            x:0.01, y:0.99, xref:"paper", yref:"paper",
            text:D.gap_text, showarrow:false, align:"left",
            bgcolor:"#0F172A", bordercolor:"#334155", borderwidth:1,
            font:{size:11, color:"#CBD5E1"}, xanchor:"left", yanchor:"top",
        }]),
    }), CFG);

    Plotly.newPlot("chart-bar", [
        {type:"bar", x:D.bar.years, y:D.bar.hombres, name:"Hombres", marker:{color:"#2E86AB"}},
        {type:"bar", x:D.bar.years, y:D.bar.mujeres, name:"Mujeres", marker:{color:"#F4A261"}},
    ], Object.assign({}, BASE, {
        title: {text:"Casos anuales por sexo"},
        barmode:"stack",
        height: 380,
        xaxis: {gridcolor:"#334155"},
        yaxis: {gridcolor:"#334155", title:"Número de suicidios"},
        legend: {orientation:"h", y:-0.18, x:0},
        margin: {t:40, b:70, l:10, r:10},
        shapes: markers.shapes,
        annotations: markers.annotations,
    }), CFG);
}

function applyWhenReady(entidad, attempt) {
    attempt = attempt || 0;
    if (attempt > 120) { updateCharts(entidad); return; }
    var el = document.getElementById("chart-states-rank");
    if (el && el.querySelector("svg.main-svg")) {
        updateCharts(entidad);
    } else {
        setTimeout(function() { applyWhenReady(entidad, attempt + 1); }, 80);
    }
}

document.getElementById("entidad-select").addEventListener("change", function() {
    var url = new URL(window.location.href);
    url.searchParams.set("entidad", this.value);
    window.location.href = url.toString();
});

document.getElementById("clear-filter").addEventListener("click", function() {
    window.location.href = window.location.pathname.split("?")[0];
});

window.addEventListener("load", function() {
    var params  = new URLSearchParams(window.location.search);
    var entidad = params.get("entidad") || "Nacional";
    if (entidad !== "Nacional") {
        document.getElementById("entidad-select").value = entidad;
        document.getElementById("filter-notice").style.display = "flex";
        document.getElementById("filter-badge").textContent = entidad;
    }
    applyWhenReady(entidad);
});
"""

# ── HTML assembly ─────────────────────────────────────────────────────────────

kpis_html = "".join([
    kpi_card("Total de suicidios",   "kpi-total",  "#CBD5E1"),
    kpi_card("Tasa total (prom.)",   "kpi-rate",   "#3BB273"),
    kpi_card("Tasa hombres (prom.)", "kpi-rate-h", "#2E86AB"),
    kpi_card("Tasa mujeres (prom.)", "kpi-rate-m", "#F4A261"),
])

html_out = f"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Suicidios por Entidad — México</title>
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
  </style>
</head>
<body>
<div class="container-fluid">

  <div class="mb-4">
    <h2 style="color:#F8FAFC;font-weight:700;margin-bottom:4px;">Suicidios por Entidad — México</h2>
    <p style="color:#64748B;font-size:.9rem;">INEGI · Estadísticas de mortalidad · {MIN_YEAR}–{MAX_YEAR}</p>
  </div>

  <div class="row g-3 mb-3">{kpis_html}</div>

  <div class="row g-3 mb-3">{extremes_html}</div>

  <div class="filter-box mb-4">
    <div class="row g-3 align-items-center">
      <div class="col-12 col-md-4">
        <p style="color:#94A3B8;font-size:.82rem;margin-bottom:6px;">Entidad</p>
        <select id="entidad-select" class="form-select dark-select">
          {entidad_opts}
        </select>
      </div>
      <div class="col-12 col-md-5">
        <span style="color:#64748B;font-size:.78rem;">
          El mapa y el ranking muestran siempre todos los estados (periodo completo)
        </span>
      </div>
      <div class="col-12 col-md-3 d-flex align-items-center">
        <div id="filter-notice">
          <span id="filter-badge"></span>
          <button id="clear-filter">&#x2715; limpiar</button>
        </div>
      </div>
    </div>
  </div>

  <div class="row g-4 mb-4">
    <div class="col-12 col-md-7">
      <div id="chart-trend" style="min-height:380px"></div>
    </div>
    <div class="col-12 col-md-5">
      <div id="chart-bar" style="min-height:380px"></div>
    </div>
  </div>

  <div class="row g-4">
    <div class="col-12 col-md-6">
      {chart_states_rank}
    </div>
    <div class="col-12 col-md-6">
      {chart_map_html}
    </div>
  </div>

</div>
<script>
const ALL_DATA = {all_data_json};
{js}
</script>
</body>
</html>"""

out = "site/suicidios_entidad.html"
with open(out, "w", encoding="utf-8") as f:
    f.write(html_out)

size_kb = len(html_out.encode()) / 1024
print(f"\nWritten {out}  ({size_kb:.0f} KB)")
