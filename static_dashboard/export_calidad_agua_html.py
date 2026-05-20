"""
Exports a fully self-contained calidad_agua_escuelas.html for GitHub Pages.

Year dropdown works client-side:
  - Per-year aggregates are pre-computed in Python and embedded as JSON
  - Plotly.newPlot() swaps chart data on selection

Trend chart stays static (full dataset context).
KPIs, municipality bar, supply bar, and supply donut update on year selection.

Run: uv run python export_calidad_agua_html.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import polars as pl
import plotly.io as pio

from dashboard.calidad_agua_escuelas import (
    df, VALID_YEARS, RISK_COLORS, CHART_LAYOUT,
    fig_trend_year, fig_ecoli_dist,
    compute_kpis,
)

CHART_CFG = {"responsive": True, "displayModeBar": False}


# ── HTML helpers ──────────────────────────────────────────────────────────────

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


# ── Pre-compute per-year data ─────────────────────────────────────────────────

def _risk_bar_data(d: pl.DataFrame) -> dict:
    base = d.filter(pl.col("riesgo_bact").is_not_null() & pl.col("Municipio").is_not_null())
    totals = base.group_by("Municipio").agg(pl.len().alias("total"))
    alto = (
        base.filter(pl.col("riesgo_bact") == "Alto riesgo")
        .group_by("Municipio").agg(pl.len().alias("alto"))
    )
    result = (
        totals.join(alto, on="Municipio", how="left")
        .with_columns(pl.col("alto").fill_null(0))
        .filter(pl.col("total") >= 20)
        .with_columns((pl.col("alto") / pl.col("total") * 100).round(1).alias("pct"))
        .sort("pct", descending=True)
    )
    if len(result) == 0:
        return {"munis": [], "pcts": [], "ns": []}
    return {
        "munis": result["Municipio"].to_list(),
        "pcts":  result["pct"].to_list(),
        "ns":    result["total"].to_list(),
    }


def _supply_bar_data(d: pl.DataFrame) -> dict:
    base = d.filter(
        pl.col("riesgo_bact").is_not_null() & (pl.col("abastecimiento") != "Desconocido")
    )
    totals = base.group_by("abastecimiento").agg(pl.len().alias("total"))
    alto = (
        base.filter(pl.col("riesgo_bact") == "Alto riesgo")
        .group_by("abastecimiento").agg(pl.len().alias("alto"))
    )
    result = (
        totals.join(alto, on="abastecimiento", how="left")
        .with_columns(pl.col("alto").fill_null(0))
        .with_columns((pl.col("alto") / pl.col("total") * 100).round(1).alias("pct"))
        .sort("pct", descending=True)
    )
    return {
        "supplies": result["abastecimiento"].to_list(),
        "pcts":     result["pct"].to_list(),
        "ns":       result["total"].to_list(),
    }


def _supply_donut_data(d: pl.DataFrame) -> dict:
    counts = (
        d.filter(pl.col("abastecimiento") != "Desconocido")
        .group_by("abastecimiento").agg(pl.len().alias("n"))
        .sort("n", descending=True)
    )
    return {
        "labels": counts["abastecimiento"].to_list(),
        "values": counts["n"].to_list(),
    }


def build_year_data() -> str:
    keys = ["__all__"] + [str(y) for y in VALID_YEARS]
    result = {}
    for key in keys:
        label = "todos los años" if key == "__all__" else key
        print(f"  Computing {label}…")
        d = df if key == "__all__" else df.filter(pl.col("Año") == int(key))

        n_total, pct_alto, pct_bajo, n_munis = compute_kpis(d)
        result[key] = {
            "kpi": {
                "n":       n_total,
                "alto":    round(pct_alto, 1),
                "bajo":    round(pct_bajo, 1),
                "munis":   n_munis,
            },
            "muni":   _risk_bar_data(d),
            "supply": _supply_bar_data(d),
            "donut":  _supply_donut_data(d),
        }
    return json.dumps(result, ensure_ascii=False, separators=(",", ":"))


# ── Render static charts ──────────────────────────────────────────────────────

print("Rendering static charts…")
chart_trend  = chart_div(fig_trend_year(df), "chart-trend",  first=True)
chart_ecoli  = chart_div(fig_ecoli_dist(df), "chart-ecoli")

print("Pre-computing per-year data…")
year_data_json = build_year_data()

# ── Year dropdown options ─────────────────────────────────────────────────────

year_opts = "\n".join(f'<option value="{y}">{y}</option>' for y in VALID_YEARS)

# ── JavaScript ────────────────────────────────────────────────────────────────

RISK_RED    = RISK_COLORS["Alto riesgo"]
RISK_ORANGE = RISK_COLORS["Intermedio"]
RISK_GREEN  = RISK_COLORS["Bajo riesgo"]
DONUT_PALETTE = ["#2E86AB", "#3BB273", "#F4A261", "#E84855", "#94A3B8", "#CBD5E1", "#64748B"]
donut_palette_json = json.dumps(DONUT_PALETTE)

js = f"""
const CFG   = {{responsive: true, displayModeBar: false}};
const RED   = "{RISK_RED}";
const ORANGE= "{RISK_ORANGE}";
const GREEN = "{RISK_GREEN}";
const BASE_LAYOUT = {{
    paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
    font: {{color: "#CBD5E1"}},
    margin: {{t:40, b:40, l:10, r:120}},
}};

function riskColor(pct) {{
    return pct >= 60 ? RED : pct >= 30 ? ORANGE : GREEN;
}}

function updateCharts(key) {{
    const D = YEAR_DATA[key];
    if (!D) return;

    // KPIs
    document.getElementById("kpi-n").textContent    = D.kpi.n.toLocaleString("es-MX");
    document.getElementById("kpi-alto").textContent = D.kpi.alto + "%";
    document.getElementById("kpi-bajo").textContent = D.kpi.bajo + "%";
    document.getElementById("kpi-munis").textContent= D.kpi.munis.toLocaleString("es-MX");

    // Municipality risk bar
    const muniColors = D.muni.pcts.map(riskColor);
    const muniHeight = Math.max(320, D.muni.munis.length * 32 + 80);
    Plotly.newPlot("chart-muni", [{{
        type: "bar", orientation: "h",
        x: D.muni.pcts, y: D.muni.munis,
        marker: {{color: muniColors}},
        text: D.muni.pcts.map((p, i) => p.toFixed(0) + "%  (n=" + D.muni.ns[i] + ")"),
        textposition: "outside",
        hovertemplate: "<b>%{{y}}</b><br>Alto riesgo: %{{x:.1f}}%<extra></extra>",
    }}], Object.assign({{}}, BASE_LAYOUT, {{
        title: {{text: "% Alto riesgo por municipio (mín. 20 muestras)"}},
        height: muniHeight,
        xaxis: {{range:[0,115], gridcolor:"#334155", ticksuffix:"%"}},
        yaxis: {{gridcolor:"rgba(0,0,0,0)", autorange:"reversed"}},
    }}), CFG);

    // Supply risk bar
    const supplyColors = D.supply.pcts.map(riskColor);
    const supplyHeight = Math.max(280, D.supply.supplies.length * 40 + 80);
    Plotly.newPlot("chart-supply-risk", [{{
        type: "bar", orientation: "h",
        x: D.supply.pcts, y: D.supply.supplies,
        marker: {{color: supplyColors}},
        text: D.supply.pcts.map((p, i) => p.toFixed(0) + "%  (n=" + D.supply.ns[i] + ")"),
        textposition: "outside",
        hovertemplate: "<b>%{{y}}</b><br>Alto riesgo: %{{x:.1f}}%<extra></extra>",
    }}], Object.assign({{}}, BASE_LAYOUT, {{
        title: {{text: "% Alto riesgo por tipo de abastecimiento"}},
        height: supplyHeight,
        xaxis: {{range:[0,115], gridcolor:"#334155", ticksuffix:"%"}},
        yaxis: {{gridcolor:"rgba(0,0,0,0)", autorange:"reversed"}},
    }}), CFG);

    // Supply donut
    Plotly.newPlot("chart-supply-donut", [{{
        type: "pie", hole: 0.5,
        labels: D.donut.labels, values: D.donut.values,
        textinfo: "label+percent",
        marker: {{colors: {donut_palette_json}}},
    }}], Object.assign({{}}, BASE_LAYOUT, {{
        title: {{text: "Distribución por tipo de abastecimiento"}},
        height: 360,
        showlegend: false,
        margin: {{t:40, b:40, l:10, r:10}},
    }}), CFG);
}}

function applyWhenReady(key, attempt) {{
    attempt = attempt || 0;
    if (attempt > 120) {{ updateCharts(key); return; }}
    const el = document.getElementById("chart-trend");
    if (el && el.querySelector("svg.main-svg")) {{
        updateCharts(key);
    }} else {{
        setTimeout(() => applyWhenReady(key, attempt + 1), 80);
    }}
}}

document.getElementById("year-select").addEventListener("change", function () {{
    const url = new URL(window.location.href);
    url.searchParams.set("year", this.value);
    window.location.href = url.toString();
}});

document.getElementById("clear-filter").addEventListener("click", function () {{
    window.location.href = window.location.pathname.split("?")[0];
}});

document.querySelectorAll("button[data-bs-toggle='tab']").forEach(btn => {{
    btn.addEventListener("shown.bs.tab", () => window.dispatchEvent(new Event("resize")));
}});

window.addEventListener("load", function () {{
    const year = new URLSearchParams(window.location.search).get("year") || "__all__";
    if (year !== "__all__") {{
        document.getElementById("year-select").value = year;
        document.getElementById("filter-notice").style.display = "flex";
        document.getElementById("filter-badge").textContent = year;
    }}
    const hash = window.location.hash;
    if (hash) {{
        const btn = document.querySelector("button[data-bs-target='" + hash + "']");
        if (btn) bootstrap.Tab.getOrCreateInstance(btn).show();
    }}
    applyWhenReady(year);
}});

document.querySelectorAll("button[data-bs-toggle='tab']").forEach(btn => {{
    btn.addEventListener("shown.bs.tab", function () {{
        history.replaceState(null, "", window.location.search + this.dataset.bsTarget);
    }});
}});
"""

# ── Assemble HTML ─────────────────────────────────────────────────────────────

kpis_html = "".join([
    kpi_card("Total de muestras",              "kpi-n",    "#CBD5E1"),
    kpi_card("Alto riesgo bacteriológico",     "kpi-alto", "#E84855"),
    kpi_card("Agua segura (Bajo riesgo)",      "kpi-bajo", "#3BB273"),
    kpi_card("Municipios monitoreados",        "kpi-munis","#2E86AB"),
])

html_out = f"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Calidad del Agua en Escuelas</title>
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
    .nav-tabs   {{ border-bottom:1px solid #334155; }}
    .nav-link   {{ color:#94A3B8; background:transparent; border:none;
                  border-top:2px solid transparent; border-radius:0; padding:10px 18px; }}
    .nav-link:hover  {{ color:#F8FAFC; background:#1E293B; }}
    .nav-link.active {{ color:#F8FAFC!important; background:#1E293B!important;
                       border-top:2px solid #2E86AB!important; font-weight:600; }}
    .tab-content {{ background:#0F172A; padding-top:20px; }}
    .info-card  {{ background:#1E293B; border:1px solid #334155; border-radius:8px;
                  padding:16px; }}
  </style>
</head>
<body>
<div class="container-fluid">

  <div class="mb-4">
    <h2 style="color:#F8FAFC;font-weight:700;margin-bottom:4px;">
      Calidad del Agua en Escuelas — Chiapas y México
    </h2>
    <p style="color:#64748B;font-size:.9rem;">
      Análisis bacteriológico y fisicoquímico del agua en planteles educativos · 4,653 muestras · 2014–2026
    </p>
  </div>

  <div class="row g-3 mb-3">{kpis_html}</div>

  <div class="filter-box mb-3">
    <div class="row g-3 align-items-end">
      <div class="col-10 col-md-4">
        <p class="filter-label mb-1">Año
          <small style="color:#64748B"> — filtra KPIs y gráficas de municipio y abastecimiento; la tendencia siempre muestra todos los años</small>
        </p>
        <select id="year-select" class="form-select dark-select">
          <option value="__all__">Todos los años</option>
          {year_opts}
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
      <button class="nav-link active" data-bs-toggle="tab" data-bs-target="#tab-panorama"
              type="button" role="tab">Panorama General</button>
    </li>
    <li class="nav-item" role="presentation">
      <button class="nav-link" data-bs-toggle="tab" data-bs-target="#tab-abastecimiento"
              type="button" role="tab">Tipo de Abastecimiento</button>
    </li>
  </ul>

  <div class="tab-content">

    <div class="tab-pane fade show active" id="tab-panorama" role="tabpanel">
      <div class="row g-3">
        <div class="col-12 col-md-7">{chart_trend}</div>
        <div class="col-12 col-md-5"><div id="chart-muni" style="min-height:320px"></div></div>
      </div>
    </div>

    <div class="tab-pane fade" id="tab-abastecimiento" role="tabpanel">
      <div class="row g-3 mb-3">
        <div class="col-12 col-md-7"><div id="chart-supply-risk" style="min-height:280px"></div></div>
        <div class="col-12 col-md-5"><div id="chart-supply-donut" style="min-height:360px"></div></div>
      </div>
      <div class="row g-3">
        <div class="col-12 col-md-6">{chart_ecoli}</div>
        <div class="col-12 col-md-6">
          <div class="info-card">
            <p style="color:#F8FAFC;font-weight:600;margin-bottom:8px;">Nota metodológica</p>
            <p style="color:#94A3B8;font-size:13px;line-height:1.6;margin:0;">
              <strong style="color:#CBD5E1;">SAE</strong> = Sistema de Agua de Escuela
              (red de distribución intraescolar).<br>
              <strong style="color:#CBD5E1;">SCALL</strong> = Sistema Comunitario de Agua de Lluvia.<br><br>
              El umbral de riesgo bacteriológico sigue la clasificación del IMTA:
              <span style="color:#3BB273;">Bajo riesgo</span> (&lt;1 UFC E. coli/100mL),
              <span style="color:#F4A261;">Intermedio</span> (1–10 UFC) y
              <span style="color:#E84855;">Alto riesgo</span> (&gt;10 UFC).
              La gráfica de distribución excluye valores atípicos por encima de 200 UFC.
            </p>
          </div>
        </div>
      </div>
    </div>

  </div>

</div>

<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"
        crossorigin="anonymous"></script>
<script>
const YEAR_DATA = {year_data_json};
{js}
</script>
</body>
</html>"""

out = "site/calidad_agua_escuelas.html"
with open(out, "w", encoding="utf-8") as f:
    f.write(html_out)

size_kb = len(html_out.encode()) / 1024
print(f"\nWritten {out}  ({size_kb:.0f} KB)")
