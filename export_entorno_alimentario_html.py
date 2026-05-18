"""
Exports a fully self-contained entorno_alimentario_escuelas.html for GitHub Pages.

The ciclo_escolar dropdown works entirely client-side:
  - Per-cycle aggregates are pre-computed in Python and embedded as JSON
  - Plotly.newPlot() swaps chart data on selection — no server needed

Charts that update on cycle selection: KPIs, panorama bar, states bar, sellers bar, roles donut
Chart that stays static (full-dataset context): trend line

Run: uv run python export_entorno_alimentario_html.py
"""

import json
import polars as pl
import plotly.io as pio
from collections import Counter

from entorno_alimentario_escuelas import (
    df, Q_COLS, Q_LABELS, SELLER_COL, VALID_CICLOS, pct_yes,
    fig_trend, fig_panorama, fig_states, fig_sellers, fig_roles,
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


def kpi_card(title, kpi_id, color):
    return f"""<div class="col-6 col-md-4 col-lg-2">
      <div class="kpi-card">
        <p class="kpi-label">{title}</p>
        <h3 class="kpi-value" id="{kpi_id}" style="color:{color}">—</h3>
      </div>
    </div>"""


# ── Pre-compute per-cycle data ────────────────────────────────────────────────

def build_cycle_data(d: pl.DataFrame) -> str:
    result = {}
    keys_to_build = ["__all__"] + VALID_CICLOS

    for ciclo in keys_to_build:
        label = "todos los ciclos" if ciclo == "__all__" else ciclo
        print(f"  Computing {label}…")
        d_f = d if ciclo == "__all__" else d.filter(pl.col("ciclo_escolar") == ciclo)

        # KPIs
        kpis = {
            "n":   len(d_f),
            "ref": round(pct_yes(d_f, "refrescos"), 1),
            "cha": round(pct_yes(d_f, "chatarra"), 1),
            "beb": round(pct_yes(d_f, "bebederos"), 1),
            "com": round(pct_yes(d_f, "comite"), 1),
            "pra": round(pct_yes(d_f, "practicas"), 1),
        }

        # Panorama: all 11 questions sorted ascending
        pano_rows = [(Q_LABELS[k], round(pct_yes(d_f, k), 1)) for k in Q_COLS]
        pano_rows.sort(key=lambda x: x[1])
        p_labels, p_vals = zip(*pano_rows)

        # States: top 20 by junk-food rate (min 20 valid responses)
        valid_s = d_f.filter(
            pl.col("chatarra").is_not_null() & pl.col("state").str.len_chars().gt(3)
        )
        agg_s = (
            valid_s.group_by("state")
            .agg(pl.col("chatarra").mean().alias("p"), pl.col("chatarra").count().alias("cnt"))
            .filter(pl.col("cnt") >= 20)
            .sort("p")
            .head(20)
        )
        agg_s = agg_s.with_columns((pl.col("p") * 100).alias("p"))

        # Sellers: top 8 from multi-select field
        c: Counter = Counter()
        for v in d_f[SELLER_COL].drop_nulls().to_list():
            if isinstance(v, str) and v.strip():
                for part in v.split(","):
                    p = part.strip()
                    if p and not p.lstrip("-").replace(".", "").isdigit() and len(p) > 2:
                        c[p] += 1
        top_sellers = sorted(c.items(), key=lambda x: x[1])[-8:]

        # Roles donut
        role_counts = (
            d_f.filter(pl.col("rol").is_not_null())["rol"]
            .value_counts().sort("count", descending=True)
        )

        result[ciclo] = {
            "kpi": kpis,
            "pano": {"l": list(p_labels), "v": list(p_vals)},
            "states": {
                "n": agg_s["state"].to_list(),
                "v": [round(x, 1) for x in agg_s["p"].to_list()],
            },
            "sellers": {
                "n": [x[0] for x in top_sellers],
                "v": [x[1] for x in top_sellers],
            },
            "roles": {
                "n": role_counts["rol"].to_list(),
                "v": role_counts["count"].to_list(),
            },
        }

    return json.dumps(result, ensure_ascii=False, separators=(",", ":"))


# ── Render static charts ──────────────────────────────────────────────────────

print("Rendering static charts…")
d = df
chart_panorama = chart_div(fig_panorama(d), "chart-panorama", first=True)
chart_trend    = chart_div(fig_trend(d),    "chart-trend")
chart_states   = chart_div(fig_states(d),   "chart-states")
chart_sellers  = chart_div(fig_sellers(d),  "chart-sellers")
chart_roles    = chart_div(fig_roles(d),    "chart-roles")

print("Pre-computing per-cycle data…")
cycle_data_json = build_cycle_data(d)

# ── Cycle dropdown options ────────────────────────────────────────────────────

cycle_opts = "\n".join(
    f'<option value="{c}">{c}</option>' for c in VALID_CICLOS
)

# ── JavaScript ────────────────────────────────────────────────────────────────

js = """
const CFG   = {responsive: true, displayModeBar: false};
const RED   = "#E84855";
const GREEN = "#3BB273";
const BLUE  = "#2E86AB";

function getLayout(id) { return document.getElementById(id).layout || {}; }

function updateCharts(ciclo) {
    const D = CYCLE_DATA[ciclo];
    if (!D) return;

    // KPIs
    document.getElementById("kpi-n").textContent   = D.kpi.n.toLocaleString();
    document.getElementById("kpi-ref").textContent = D.kpi.ref + "%";
    document.getElementById("kpi-cha").textContent = D.kpi.cha + "%";
    document.getElementById("kpi-beb").textContent = D.kpi.beb + "%";
    document.getElementById("kpi-com").textContent = D.kpi.com + "%";
    document.getElementById("kpi-pra").textContent = D.kpi.pra + "%";

    // Panorama bar
    const colors = D.pano.v.map(v => v > 50 ? RED : GREEN);
    Plotly.newPlot("chart-panorama", [{
        type: "bar", orientation: "h",
        x: D.pano.v, y: D.pano.l,
        marker: { color: colors },
        text: D.pano.v.map(v => v.toFixed(1) + "%"),
        textposition: "outside",
    }], Object.assign({}, getLayout("chart-panorama"), {
        title: { text: "Panorama general — % de respuestas Sí" },
        xaxis: { range: [0, 110], ticksuffix: "%" },
    }), CFG);

    // States bar
    const stateColors = D.states.v.map(v => {
        const t = Math.min(1, Math.max(0, (v - 50) / 50));
        return "rgb(" + Math.round(255*t) + "," + Math.round(255*(1-t)) + ",80)";
    });
    Plotly.newPlot("chart-states", [{
        type: "bar", orientation: "h",
        x: D.states.v, y: D.states.n,
        marker: { color: stateColors },
        text: D.states.v.map(v => v.toFixed(1) + "%"),
        textposition: "outside",
    }], Object.assign({}, getLayout("chart-states"), {
        title: { text: "% Escuelas con venta de comida chatarra por estado" },
        xaxis: { range: [0, 110], ticksuffix: "%" },
        height: Math.max(340, D.states.n.length * 28 + 80),
    }), CFG);

    // Sellers bar
    Plotly.newPlot("chart-sellers", [{
        type: "bar", orientation: "h",
        x: D.sellers.v, y: D.sellers.n,
        marker: { color: BLUE },
        text: D.sellers.v.map(v => v.toLocaleString()),
        textposition: "outside",
    }], Object.assign({}, getLayout("chart-sellers"), {
        title: { text: "¿Quién vende alimentos dentro de la escuela?" },
        height: Math.max(280, D.sellers.n.length * 38 + 80),
    }), CFG);

    // Roles donut
    const DONUT_COLORS = ["#2E86AB","#3BB273","#F4A261","#E84855","#94A3B8","#CBD5E1","#64748B"];
    Plotly.newPlot("chart-roles", [{
        type: "pie", hole: 0.5,
        labels: D.roles.n, values: D.roles.v,
        textinfo: "label+percent",
        marker: { colors: DONUT_COLORS },
    }], Object.assign({}, getLayout("chart-roles"), {
        title: { text: "Rol del encuestado" },
        showlegend: false,
    }), CFG);
}

function applyWhenReady(ciclo, attempt) {
    attempt = attempt || 0;
    if (attempt > 120) { updateCharts(ciclo); return; }
    const el = document.getElementById("chart-panorama");
    if (el && el.querySelector("svg.main-svg")) {
        updateCharts(ciclo);
    } else {
        setTimeout(() => applyWhenReady(ciclo, attempt + 1), 80);
    }
}

document.getElementById("ciclo-select").addEventListener("change", function () {
    const url = new URL(window.location.href);
    url.searchParams.set("ciclo", this.value);
    window.location.href = url.toString();
});

document.getElementById("clear-filter").addEventListener("click", function () {
    window.location.href = window.location.pathname.split("?")[0];
});

window.addEventListener("load", function () {
    const ciclo = new URLSearchParams(window.location.search).get("ciclo") || "__all__";
    if (ciclo !== "__all__") {
        document.getElementById("ciclo-select").value = ciclo;
        document.getElementById("filter-notice").style.display = "flex";
        document.getElementById("filter-badge").textContent = ciclo;
    }
    applyWhenReady(ciclo);
});
"""

# ── Full HTML ─────────────────────────────────────────────────────────────────

kpis_html = "".join([
    kpi_card("Total respuestas",              "kpi-n",   "#CBD5E1"),
    kpi_card("Refrescos con azúcar",          "kpi-ref", "#E84855"),
    kpi_card("Venden comida chatarra",        "kpi-cha", "#E84855"),
    kpi_card("Bebederos funcionando",         "kpi-beb", "#3BB273"),
    kpi_card("Comité de vigilancia activo",   "kpi-com", "#F4A261"),
    kpi_card("Promueve alimentación sana",    "kpi-pra", "#2E86AB"),
])

html_out = f"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Entorno Alimentario en Escuelas de México</title>
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
  </style>
</head>
<body>
<div class="container-fluid">

  <div class="mb-4">
    <h2 style="color:#F8FAFC;font-weight:700;margin-bottom:4px;">
      Entorno Alimentario en Escuelas de México
    </h2>
    <p style="color:#64748B;font-size:.9rem;">
      Encuesta sobre hábitos de venta de alimentos y bebidas · 48,118 respuestas · 2014–2025
    </p>
  </div>

  <div class="row g-3 mb-3">{kpis_html}</div>

  <div class="filter-box mb-3">
    <div class="row g-3 align-items-end">
      <div class="col-10 col-md-6">
        <p class="filter-label mb-1">Ciclo escolar
          <small style="color:#64748B"> — filtra todos los gráficos excepto la tendencia</small>
        </p>
        <select id="ciclo-select" class="form-select dark-select">
          <option value="__all__">Todos los ciclos</option>
          {cycle_opts}
        </select>
      </div>
      <div class="col-2 col-md-6 d-flex align-items-end">
        <div id="filter-notice">
          <span id="filter-badge"></span>
          <button id="clear-filter">✕ limpiar</button>
        </div>
      </div>
    </div>
  </div>

  <div class="row g-3 mb-3">
    <div class="col-12 col-md-6">{chart_panorama}</div>
    <div class="col-12 col-md-6">{chart_trend}</div>
  </div>

  <div class="row g-3">
    <div class="col-12 col-md-6">{chart_states}</div>
    <div class="col-12 col-md-6">
      {chart_sellers}
      {chart_roles}
    </div>
  </div>

</div>

<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"
        crossorigin="anonymous"></script>
<script>
const CYCLE_DATA = {cycle_data_json};
{js}
</script>
</body>
</html>"""

out = "site/entorno_alimentario_escuelas.html"
with open(out, "w", encoding="utf-8") as f:
    f.write(html_out)

size_kb = len(html_out.encode()) / 1024
print(f"\nWritten {out}  ({size_kb:.0f} KB)")
