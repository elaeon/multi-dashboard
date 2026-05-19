"""
Exports a fully self-contained desaparecidos.html for GitHub Pages.

Estado dropdown works client-side:
  - Per-state aggregates pre-computed in Python and embedded as JSON
  - Plotly.newPlot() swaps charts on selection

Static chart (always full dataset):
  - State ranking bar

Dynamic charts (update on estado selection):
  - Annual trend bar
  - Monthly distribution bar
  - Sex donut
  - Age group bar
  - KPIs

Run: uv run python export_desaparecidos_html.py
"""

import json
import polars as pl
import plotly.io as pio

from desaparecidos import (
    df, ESTADOS, MES_NOMBRES, AGE_LABELS, AGE_RANGES,
    fig_state_ranking, compute_kpis,
)

CHART_CFG  = {"responsive": True, "displayModeBar": False}
ALL_KEYS   = ["__all__"] + ESTADOS


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


# ── Pre-compute per-state data ────────────────────────────────────────────────

def _trend_data(d: pl.DataFrame) -> dict:
    anual = (
        d.filter(pl.col("anio_desap").is_not_null())
        .group_by("anio_desap").agg(pl.len().alias("n"))
        .sort("anio_desap")
    )
    return {"years": anual["anio_desap"].to_list(), "n": anual["n"].to_list()}


def _monthly_data(d: pl.DataFrame) -> dict:
    monthly = (
        d.filter(pl.col("mes_desap").is_not_null() & pl.col("anio_desap").is_between(2010, 2025))
        .group_by("mes_desap").agg(pl.len().alias("n"))
        .sort("mes_desap")
    )
    meses = [MES_NOMBRES[m - 1] for m in monthly["mes_desap"].to_list()]
    return {"meses": meses, "n": monthly["n"].to_list()}


def _sex_data(d: pl.DataFrame) -> dict:
    counts = (
        d.filter(pl.col("SEXO").is_in(["HOMBRE", "MUJER"]))
        .group_by("SEXO").agg(pl.len().alias("n"))
        .sort("SEXO")
    )
    return {"labels": counts["SEXO"].to_list(), "values": counts["n"].to_list()}


def _age_data(d: pl.DataFrame) -> dict:
    with_age = d.filter(
        pl.col("anio_nac").is_not_null() &
        pl.col("anio_desap").is_not_null() &
        pl.col("anio_nac").is_between(1930, 2020)
    ).with_columns(
        (pl.col("anio_desap") - pl.col("anio_nac")).alias("edad")
    ).filter(pl.col("edad").is_between(0, 100))

    total = len(with_age)
    pcts = []
    for lo, hi in AGE_RANGES:
        n = len(with_age.filter(pl.col("edad").is_between(lo, hi)))
        pcts.append(round(n / total * 100, 1) if total > 0 else 0)
    return {"labels": AGE_LABELS, "pcts": pcts}


def build_state_data() -> str:
    result = {}
    for key in ALL_KEYS:
        label = "todo el país" if key == "__all__" else key
        print(f"  Computing {label}…")
        d = df if key == "__all__" else df.filter(pl.col("ENTIDAD") == key)
        n_total, pct_h, pct_m, peak_year, peak_n = compute_kpis(d)
        result[key] = {
            "kpi": {
                "n":          n_total,
                "pct_h":      round(pct_h, 1),
                "pct_m":      round(pct_m, 1),
                "peak_year":  peak_year,
                "peak_n":     peak_n,
            },
            "trend":   _trend_data(d),
            "monthly": _monthly_data(d),
            "sex":     _sex_data(d),
            "age":     _age_data(d),
        }
    return json.dumps(result, ensure_ascii=False, separators=(",", ":"))


# ── Render static charts ──────────────────────────────────────────────────────

print("Rendering static charts…")
chart_states = chart_div(fig_state_ranking(), "chart-states", first=True)

print("Pre-computing per-state data…")
state_data_json = build_state_data()

# ── Estado dropdown options ───────────────────────────────────────────────────

estado_opts = '<option value="__all__">Todo el país</option>\n' + "\n".join(
    f'<option value="{e}">{e.title()}</option>' for e in ESTADOS
)

# ── JavaScript ────────────────────────────────────────────────────────────────

js = """
const CFG = {responsive: true, displayModeBar: false};
const BASE = {
    paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
    font: {color: "#CBD5E1"},
};
const AXIS = (extra) => Object.assign({gridcolor: "#334155"}, extra || {});
const MARGIN = {t:40, b:40, l:10, r:10};

function updateCharts(key) {
    const D = STATE_DATA[key];
    if (!D) return;

    // KPIs
    document.getElementById("kpi-n").textContent       = D.kpi.n.toLocaleString("es-MX");
    document.getElementById("kpi-hombre").textContent  = D.kpi.pct_h + "%";
    document.getElementById("kpi-mujer").textContent   = D.kpi.pct_m + "%";
    document.getElementById("kpi-pico").textContent    = D.kpi.peak_year + " (" + D.kpi.peak_n.toLocaleString("es-MX") + ")";

    // Annual trend
    Plotly.newPlot("chart-trend", [{
        type: "bar", x: D.trend.years, y: D.trend.n,
        marker: {color: "#E84855"},
        hovertemplate: "<b>%{x}</b><br>Casos: %{y:,}<extra></extra>",
    }], Object.assign({}, BASE, {
        title: {text: "Personas desaparecidas por año"},
        height: 380, margin: MARGIN, showlegend: false,
        xaxis: AXIS({title: "Año"}),
        yaxis: AXIS({title: "Casos registrados"}),
    }), CFG);

    // Monthly distribution
    const avg = D.monthly.n.reduce((a, b) => a + b, 0) / D.monthly.n.length;
    const mColors = D.monthly.n.map(n => n > avg ? "#E84855" : "#F4A261");
    Plotly.newPlot("chart-monthly", [{
        type: "bar", x: D.monthly.meses, y: D.monthly.n,
        marker: {color: mColors},
        hovertemplate: "<b>%{x}</b><br>Casos: %{y:,}<extra></extra>",
    }], Object.assign({}, BASE, {
        title: {text: "Distribución mensual (2010–2025)"},
        height: 360, margin: MARGIN, showlegend: false,
        xaxis: AXIS(),
        yaxis: AXIS(),
    }), CFG);

    // Sex donut
    Plotly.newPlot("chart-sex", [{
        type: "pie", hole: 0.5,
        labels: D.sex.labels, values: D.sex.values,
        textinfo: "label+percent",
        marker: {colors: ["#2E86AB", "#F4A261"]},
    }], Object.assign({}, BASE, {
        title: {text: "Distribución por sexo"},
        height: 360, showlegend: false,
        margin: {t:40, b:40, l:10, r:10},
    }), CFG);

    // Age group bar
    const maxPct = Math.max(...D.age.pcts);
    Plotly.newPlot("chart-age", [{
        type: "bar", x: D.age.labels, y: D.age.pcts,
        marker: {color: "#F4A261"},
        text: D.age.pcts.map(p => p.toFixed(1) + "%"),
        textposition: "outside",
        hovertemplate: "<b>%{x}</b><br>%{y:.1f}%<extra></extra>",
    }], Object.assign({}, BASE, {
        title: {text: "Distribución por grupo de edad"},
        height: 360, margin: MARGIN, showlegend: false,
        xaxis: AXIS({title: "Edad al desaparecer"}),
        yaxis: AXIS({ticksuffix: "%", range: [0, maxPct * 1.2 + 2]}),
    }), CFG);
}

function applyWhenReady(key, attempt) {
    attempt = attempt || 0;
    if (attempt > 120) { updateCharts(key); return; }
    const el = document.getElementById("chart-states");
    if (el && el.querySelector("svg.main-svg")) {
        updateCharts(key);
    } else {
        setTimeout(() => applyWhenReady(key, attempt + 1), 80);
    }
}

document.getElementById("estado-select").addEventListener("change", function () {
    const url = new URL(window.location.href);
    url.searchParams.set("estado", this.value);
    window.location.href = url.toString();
});

document.getElementById("clear-filter").addEventListener("click", function () {
    window.location.href = window.location.pathname.split("?")[0];
});

document.querySelectorAll("button[data-bs-toggle='tab']").forEach(btn => {
    btn.addEventListener("shown.bs.tab", () => window.dispatchEvent(new Event("resize")));
    btn.addEventListener("shown.bs.tab", function () {
        history.replaceState(null, "", window.location.search + this.dataset.bsTarget);
    });
});

window.addEventListener("load", function () {
    const key = new URLSearchParams(window.location.search).get("estado") || "__all__";
    if (key !== "__all__") {
        document.getElementById("estado-select").value = key;
        document.getElementById("filter-notice").style.display = "flex";
        document.getElementById("filter-badge").textContent = key.toLowerCase().replace(/\\b\\w/g, c => c.toUpperCase());
    }
    const hash = window.location.hash;
    if (hash) {
        const btn = document.querySelector("button[data-bs-target='" + hash + "']");
        if (btn) bootstrap.Tab.getOrCreateInstance(btn).show();
    }
    applyWhenReady(key);
});
"""

# ── Assemble HTML ─────────────────────────────────────────────────────────────

kpis_html = "".join([
    kpi_card("Total registrados",       "kpi-n",      "#CBD5E1"),
    kpi_card("Hombres (con dato)",      "kpi-hombre", "#2E86AB"),
    kpi_card("Mujeres (con dato)",      "kpi-mujer",  "#F4A261"),
    kpi_card("Año con más casos",       "kpi-pico",   "#E84855"),
])

html_out = f"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Personas Desaparecidas — México</title>
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
    .data-note  {{ background:#1E293B; border:1px solid #334155; border-radius:8px;
                  padding:12px 16px; }}
  </style>
</head>
<body>
<div class="container-fluid">

  <div class="mb-4">
    <h2 style="color:#F8FAFC;font-weight:700;margin-bottom:4px;">
      Personas Desaparecidas — México
    </h2>
    <p style="color:#64748B;font-size:.9rem;">
      Registro Nacional de Personas Desaparecidas y No Localizadas · 133,887 casos activos
    </p>
  </div>

  <div class="row g-3 mb-3">{kpis_html}</div>

  <div class="filter-box mb-3">
    <div class="row g-3 align-items-end">
      <div class="col-10 col-md-4">
        <p style="color:#94A3B8;font-size:.82rem;margin-bottom:6px;">Estado
          <small style="color:#64748B"> — filtra tendencia, perfil y KPIs; el ranking siempre muestra todos los estados</small>
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

  <div class="data-note mb-3">
    <p style="margin:0;">
      <strong style="color:#F4A261;">Nota: </strong>
      <span style="color:#94A3B8;font-size:13px;">
        El 43% de los registros tienen fecha confidencial y no aparecen en las gráficas de
        tendencia temporal. Los totales en los KPIs incluyen todos los registros.
        Fuente: RNPDNO — Registro Nacional de Personas Desaparecidas y No Localizadas.
      </span>
    </p>
  </div>

  <ul class="nav nav-tabs mb-0" role="tablist">
    <li class="nav-item" role="presentation">
      <button class="nav-link active" data-bs-toggle="tab" data-bs-target="#tab-tendencia"
              type="button" role="tab">Tendencia</button>
    </li>
    <li class="nav-item" role="presentation">
      <button class="nav-link" data-bs-toggle="tab" data-bs-target="#tab-perfil"
              type="button" role="tab">Perfil de Víctimas</button>
    </li>
    <li class="nav-item" role="presentation">
      <button class="nav-link" data-bs-toggle="tab" data-bs-target="#tab-estados"
              type="button" role="tab">Por Estado</button>
    </li>
  </ul>

  <div class="tab-content">

    <div class="tab-pane fade show active" id="tab-tendencia" role="tabpanel">
      <div class="row g-3">
        <div class="col-12 col-md-8">
          <div id="chart-trend" style="min-height:380px"></div>
        </div>
        <div class="col-12 col-md-4">
          <div id="chart-monthly" style="min-height:360px"></div>
        </div>
      </div>
    </div>

    <div class="tab-pane fade" id="tab-perfil" role="tabpanel">
      <div class="row g-3">
        <div class="col-12 col-md-4">
          <div id="chart-sex" style="min-height:360px"></div>
        </div>
        <div class="col-12 col-md-8">
          <div id="chart-age" style="min-height:360px"></div>
        </div>
      </div>
    </div>

    <div class="tab-pane fade" id="tab-estados" role="tabpanel">
      <div class="row g-3">
        <div class="col-12">{chart_states}</div>
      </div>
    </div>

  </div>

</div>

<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"
        crossorigin="anonymous"></script>
<script>
const STATE_DATA = {state_data_json};
{js}
</script>
</body>
</html>"""

out = "site/desaparecidos.html"
with open(out, "w", encoding="utf-8") as f:
    f.write(html_out)

size_kb = len(html_out.encode()) / 1024
print(f"\nWritten {out}  ({size_kb:.0f} KB)")
