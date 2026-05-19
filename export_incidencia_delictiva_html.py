"""
Exports a self-contained incidencia_delictiva.html for GitHub Pages.

Single filter: Tipo de delito (49 types + "Todos los delitos").
Pre-computes trend, state ranking, and KPIs per filter value.

Static (full dataset):  choropleth map
Dynamic (per filter):   trend line, state ranking bar, KPIs

Run: uv run python export_incidencia_delictiva_html.py
"""

import json
import polars as pl
import plotly.io as pio

from incidencia_delictiva import (
    MONTHS, STATE_ISO, GEO,
    agg_yr_bien, agg_yr_tipo, agg_yr_ent, agg_yr_ent_tipo,
    TIPOS,
    fig_map, fig_trend,
    CHART_LAYOUT,
)

CHART_CFG = {"responsive": True, "displayModeBar": False}
# Exclude 2026 (partial year) from the default trend shown
DEFAULT_YRS = (2015, 2025)


# ── HTML helpers ──────────────────────────────────────────────────────────────

def chart_div(fig, div_id: str, first: bool = False) -> str:
    return pio.to_html(
        fig, full_html=False,
        include_plotlyjs="cdn" if first else False,
        config=CHART_CFG,
        div_id=div_id,
    )


def kpi_card_html(title: str, kpi_id: str, color: str) -> str:
    return f"""<div class="col-6 col-md-3">
  <div class="kpi-card">
    <p class="kpi-label">{title}</p>
    <h3 class="kpi-value" id="{kpi_id}" style="color:{color}">—</h3>
  </div>
</div>"""


# ── Pre-compute per-filter payloads ───────────────────────────────────────────

def _build_payload(tipo: str) -> dict:
    yr0, yr1 = DEFAULT_YRS
    yr_mask = pl.col('Año').is_between(yr0, yr1)

    if tipo == 'Todos los delitos':
        d_yr = agg_yr_bien.filter(yr_mask).group_by('Año').agg(pl.col('Casos').sum()).sort('Año')
        d_ent = agg_yr_ent.filter(yr_mask).group_by(['Entidad','Clave_Ent']).agg(pl.col('Casos').sum())
        d_tipo_totals = agg_yr_tipo.filter(yr_mask).group_by('Tipo de delito').agg(pl.col('Casos').sum())
    else:
        tipo_mask = pl.col('Tipo de delito') == tipo
        d_yr = agg_yr_tipo.filter(yr_mask & tipo_mask).group_by('Año').agg(pl.col('Casos').sum()).sort('Año')
        d_ent = agg_yr_ent_tipo.filter(yr_mask & tipo_mask).group_by(['Entidad','Clave_Ent']).agg(pl.col('Casos').sum())
        d_tipo_totals = agg_yr_tipo.filter(yr_mask & tipo_mask).group_by('Tipo de delito').agg(pl.col('Casos').sum())

    total = int(d_yr['Casos'].sum())

    # YoY vs prior year
    t_last  = int(d_yr.filter(pl.col('Año') == yr1)['Casos'].sum())
    t_prior = int(d_yr.filter(pl.col('Año') == yr1 - 1)['Casos'].sum())
    yoy_pct  = round((t_last - t_prior) / t_prior * 100, 1) if t_prior > 0 else None

    top_ent = d_ent.sort('Casos', descending=True).head(1)['Entidad'][0] if len(d_ent) > 0 else "—"

    if tipo == 'Todos los delitos':
        top_tipo = d_tipo_totals.sort('Casos', descending=True).head(1)['Tipo de delito'][0] \
            if len(d_tipo_totals) > 0 else "—"
    else:
        top_tipo = tipo

    # State ranking (sorted descending)
    ent_sorted = d_ent.sort('Casos', descending=True).head(15)

    return {
        "kpi": {
            "total":    total,
            "yoy_pct":  yoy_pct,
            "top_ent":  top_ent,
            "top_tipo": top_tipo[:35] + "…" if len(top_tipo) > 35 else top_tipo,
        },
        "trend": {
            "years": d_yr['Año'].cast(pl.Int32).to_list(),
            "casos": d_yr['Casos'].to_list(),
        },
        "ranking": {
            "states": ent_sorted['Entidad'].to_list(),
            "casos":  ent_sorted['Casos'].to_list(),
        },
    }


print("Pre-computing per-tipo payloads…")
ALL_DATA = {}
for i, tipo in enumerate(TIPOS, 1):
    print(f"  [{i}/{len(TIPOS)}] {tipo}")
    ALL_DATA[tipo] = _build_payload(tipo)
all_data_json = json.dumps(ALL_DATA, ensure_ascii=False, separators=(",", ":"))


# ── Static charts ─────────────────────────────────────────────────────────────

print("Rendering static map…")
yr_mask = pl.col('Año').is_between(*DEFAULT_YRS)
d_ent_all = agg_yr_ent.filter(yr_mask).group_by(['Entidad','Clave_Ent']).agg(pl.col('Casos').sum())

# First chart loads Plotly CDN — must be a regular SVG chart, not a map
d_yr_all = agg_yr_bien.filter(yr_mask).group_by('Año').agg(pl.col('Casos').sum()).sort('Año')
# Use a non-"Todos los delitos" string to render the simple line branch (matches JS output)
chart_trend_static = chart_div(fig_trend(d_yr_all, '__total__'), "chart-trend", first=True)
chart_map_static   = chart_div(fig_map(d_ent_all), "chart-map")


# ── Dropdown options ──────────────────────────────────────────────────────────

tipo_opts = "\n".join(
    f'<option value="{t}"{"" if t != "Todos los delitos" else " selected"}>{t}</option>'
    for t in TIPOS
)


# ── JavaScript ────────────────────────────────────────────────────────────────

js = r"""
const CFG  = {responsive: true, displayModeBar: false};
const BASE = {
    paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
    font: {color: "#CBD5E1"},
    margin: {t:40, b:40, l:10, r:10},
};
const AXIS = (extra) => Object.assign({gridcolor: "#334155"}, extra || {});

function updateCharts(tipo) {
    const D = ALL_DATA[tipo];
    if (!D) return;

    // KPIs
    document.getElementById("kpi-total").textContent   = D.kpi.total.toLocaleString("es-MX");
    const yoy = D.kpi.yoy_pct;
    const yoyEl = document.getElementById("kpi-yoy");
    if (yoy !== null) {
        yoyEl.textContent = (yoy > 0 ? "+" : "") + yoy + "%";
        yoyEl.style.color = yoy > 0 ? "#E84855" : "#3BB273";
    } else {
        yoyEl.textContent = "—";
        yoyEl.style.color = "#94A3B8";
    }
    document.getElementById("kpi-top-ent").textContent  = D.kpi.top_ent;
    document.getElementById("kpi-top-tipo").textContent = D.kpi.top_tipo;

    // Trend
    Plotly.newPlot("chart-trend", [{
        type: "scatter", mode: "lines+markers",
        x: D.trend.years, y: D.trend.casos,
        fill: "tozeroy",
        line: {color: "#2E86AB", width: 2},
        fillcolor: "rgba(46,134,171,0.15)",
        hovertemplate: "<b>%{x}</b><br>Casos: %{y:,}<extra></extra>",
    }], Object.assign({}, BASE, {
        title: {text: tipo === "Todos los delitos" ? "Total delitos · 2015–2025" : tipo + " · 2015–2025"},
        height: 380,
        xaxis: AXIS({title: "Año", dtick: 1}),
        yaxis: AXIS({title: "Casos registrados"}),
    }), CFG);

    // State ranking
    Plotly.newPlot("chart-ranking", [{
        type: "bar", orientation: "h",
        x: D.ranking.casos, y: D.ranking.states,
        marker: {color: "#F4A261"},
        hovertemplate: "<b>%{y}</b><br>Casos: %{x:,}<extra></extra>",
    }], Object.assign({}, BASE, {
        title: {text: "Top 15 estados"},
        height: Math.max(300, D.ranking.states.length * 28 + 80),
        margin: {t:40, b:40, l:10, r:10},
        yaxis: AXIS({autorange: "reversed"}),
        xaxis: AXIS({title: "Casos"}),
    }), CFG);
}

function applyWhenReady(tipo, attempt) {
    attempt = attempt || 0;
    if (attempt > 120) { updateCharts(tipo); return; }
    const el = document.getElementById("chart-trend");
    if (el && el.querySelector("svg.main-svg")) {
        updateCharts(tipo);
    } else {
        setTimeout(() => applyWhenReady(tipo, attempt + 1), 80);
    }
}

document.getElementById("tipo-select").addEventListener("change", function() {
    const url = new URL(window.location.href);
    url.searchParams.set("tipo", this.value);
    window.location.href = url.toString();
});

window.addEventListener("load", function() {
    const p    = new URLSearchParams(window.location.search);
    const tipo = p.get("tipo") || "Todos los delitos";
    const sel  = document.getElementById("tipo-select");
    if (ALL_DATA[tipo]) sel.value = tipo;
    applyWhenReady(tipo);
});
"""

# ── Assemble HTML ─────────────────────────────────────────────────────────────

kpis_html = "".join([
    kpi_card_html("Total de casos (2015–2025)", "kpi-total",    "#CBD5E1"),
    kpi_card_html("Variación 2025 vs 2024",     "kpi-yoy",      "#E84855"),
    kpi_card_html("Estado con más casos",        "kpi-top-ent",  "#F4A261"),
    kpi_card_html("Delito más frecuente",        "kpi-top-tipo", "#2E86AB"),
])

html_out = f"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Incidencia Delictiva · Fuero Común · México</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css"
        rel="stylesheet" crossorigin="anonymous">
  <style>
    body       {{ background:#0F172A; color:#CBD5E1; padding:24px;
                 font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }}
    .kpi-card  {{ background:#1E293B; border:1px solid #334155; border-radius:8px;
                 padding:16px; text-align:center; height:100%; }}
    .kpi-label {{ color:#94A3B8; font-size:.78rem; margin-bottom:4px; }}
    .kpi-value {{ font-weight:700; margin:0; font-size:1.4rem; }}
    .filter-box {{ background:#1E293B; border:1px solid #334155; border-radius:8px;
                  padding:14px 18px; }}
    .dark-select {{ background-color:#1E293B!important; border-color:#334155!important;
                   color:#CBD5E1!important; }}
    .dark-select:focus {{ border-color:#2E86AB!important;
                          box-shadow:0 0 0 .2rem rgba(46,134,171,.25)!important; }}
    .dark-select option {{ background:#1E293B; color:#CBD5E1; }}
    .nav-tabs  {{ border-bottom:1px solid #334155; }}
    .nav-link  {{ color:#94A3B8; background:transparent; border:none;
                 border-top:2px solid transparent; border-radius:0; padding:10px 18px; }}
    .nav-link:hover {{ color:#F8FAFC; background:#1E293B; }}
    .nav-link.active {{ color:#F8FAFC!important; background:#1E293B!important;
                        border-top:2px solid #2E86AB!important; font-weight:600; }}
    .tab-content {{ background:#0F172A; padding-top:20px; }}
  </style>
</head>
<body>
<div class="container-fluid">

  <div class="mb-4">
    <h2 style="color:#F8FAFC;font-weight:700;margin-bottom:4px;">
      Incidencia Delictiva del Fuero Común
    </h2>
    <p style="color:#64748B;font-size:.9rem;">
      Carpetas de investigación por municipio · México 2015–2025 · SESNSP
    </p>
  </div>

  <div class="row g-2 mb-3">{kpis_html}</div>

  <div class="filter-box mb-3">
    <div class="row g-3 align-items-end">
      <div class="col-12 col-md-6">
        <p style="color:#94A3B8;font-size:.82rem;margin-bottom:6px;">Tipo de delito</p>
        <select id="tipo-select" class="form-select dark-select">
          {tipo_opts}
        </select>
      </div>
      <div class="col-12 col-md-6">
        <p style="color:#64748B;font-size:.78rem;margin:0;">
          El mapa siempre muestra el total nacional (2015–2025).
          La tendencia y el ranking se actualizan según el tipo seleccionado.
        </p>
      </div>
    </div>
  </div>

  <ul class="nav nav-tabs mb-0" role="tablist">
    <li class="nav-item"><button class="nav-link active" data-bs-toggle="tab"
        data-bs-target="#tab-tendencia" type="button">Tendencia</button></li>
    <li class="nav-item"><button class="nav-link" data-bs-toggle="tab"
        data-bs-target="#tab-estados" type="button">Por Estado</button></li>
    <li class="nav-item"><button class="nav-link" data-bs-toggle="tab"
        data-bs-target="#tab-mapa" type="button">Mapa</button></li>
  </ul>

  <div class="tab-content">

    <div class="tab-pane fade show active" id="tab-tendencia" role="tabpanel">
      <div class="row g-3">
        <div class="col-12">{chart_trend_static}</div>
      </div>
    </div>

    <div class="tab-pane fade" id="tab-estados" role="tabpanel">
      <div id="chart-ranking" style="min-height:480px"></div>
    </div>

    <div class="tab-pane fade" id="tab-mapa" role="tabpanel">
      <div class="row g-3">
        <div class="col-12">{chart_map_static}</div>
      </div>
    </div>

  </div>

  <p style="color:#475569;font-size:11px;text-align:center;margin-top:24px;">
    Fuente: Secretariado Ejecutivo del Sistema Nacional de Seguridad Pública (SESNSP)
  </p>
</div>

<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"
        crossorigin="anonymous"></script>
<script>
const ALL_DATA = {all_data_json};
{js}
</script>
</body>
</html>"""

out = "site/incidencia_delictiva.html"
with open(out, "w", encoding="utf-8") as f:
    f.write(html_out)

size_kb = len(html_out.encode()) / 1024
print(f"\nWritten {out}  ({size_kb:.0f} KB)")
