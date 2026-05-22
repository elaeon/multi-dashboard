"""
Exports a fully self-contained puertos.html for GitHub Pages.

All filter combinations (periodo × trafico) are pre-computed as JSON
and swapped client-side with Plotly.react — no server needed.

Run: uv run python static_dashboard/export_puertos_html.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import polars as pl
import plotly.io as pio

from dashboard.puertos import (
    df, compute_kpis, _fmt_num,
    fig_top_buques, fig_tipo_carga, fig_top_toneladas,
    fig_top_pasajeros, fig_top_teus,
)

CHART_CFG = {"responsive": True, "displayModeBar": False}
TOP_N = 15
TIPOS = ["Agricola", "Contenerizada", "Mineral", "Otros F.", "Petroleo", "Suelta"]
CARGO_COLORS = {
    "Agricola":      "#3BB273",
    "Contenerizada": "#2E86AB",
    "Mineral":       "#94A3B8",
    "Otros F.":      "#F4A261",
    "Petroleo":      "#E84855",
    "Suelta":        "#9B59B6",
}


# ── HTML helpers ──────────────────────────────────────────────────────────────

def chart_div(fig, div_id, first=False):
    return pio.to_html(
        fig, full_html=False,
        include_plotlyjs="cdn" if first else False,
        config=CHART_CFG,
        div_id=div_id,
    )


def row(*cols):
    inner = "".join(f'<div class="col-{w} mb-3">{h}</div>' for h, w in cols)
    return f'<div class="row g-3">{inner}</div>'


def kpi_card(label, kpi_id, color):
    return f"""<div class="col-3">
  <div class="kpi-card">
    <p class="kpi-label">{label}</p>
    <h3 class="kpi-value" id="{kpi_id}" style="color:{color}">—</h3>
  </div>
</div>"""


# ── Pre-compute all filter combinations ───────────────────────────────────────

def build_combo_data(df: pl.DataFrame) -> str:
    periods = {
        "todos":   df,
        "dic2025": df.filter((pl.col("anio") == 2025) & (pl.col("mes") == 12)),
        "ene2026": df.filter((pl.col("anio") == 2026) & (pl.col("mes") == 1)),
    }
    traficos = ["todos", "Altura", "Cabotaje", "Transbordo"]
    result = {}

    for period_key, d_period in periods.items():
        d_pas = d_period.filter(pl.col("tipo_carga") == "pasajeros")
        pas_agg = (
            d_pas.group_by("puerto").agg(pl.col("pasajeros").sum())
            .drop_nulls("pasajeros")
            .sort("pasajeros", descending=True).head(TOP_N).sort("pasajeros")
        )

        for trafico in traficos:
            d_cargo = d_period.filter(pl.col("tipo_carga") != "pasajeros")
            if trafico != "todos":
                d_cargo = d_cargo.filter(pl.col("trafico") == trafico)

            key = f"{period_key}|{trafico}"
            print(f"  Computing {key}…")

            # Top buques
            bq = (
                d_cargo.group_by("puerto").agg(pl.col("buques").sum())
                .sort("buques", descending=True).head(TOP_N).sort("buques")
            )
            top_ports = bq["puerto"].to_list()

            # Tipo carga stacked (pct per port)
            totals_map = {r["puerto"]: r["buques"] or 0 for r in bq.iter_rows(named=True)}
            tipo_data = {}
            for tipo in TIPOS:
                sub = (
                    d_cargo.filter(pl.col("tipo_carga") == tipo)
                    .filter(pl.col("puerto").is_in(top_ports))
                    .group_by("puerto").agg(pl.col("buques").sum())
                )
                tipo_map = {r["puerto"]: r["buques"] or 0 for r in sub.iter_rows(named=True)}
                pcts, counts = [], []
                for p in top_ports:
                    t = totals_map.get(p, 0) or 0
                    c = tipo_map.get(p, 0) or 0
                    pcts.append(round(c / t * 100, 1) if t > 0 else 0.0)
                    counts.append(c)
                tipo_data[tipo] = {"pct": pcts, "n": counts}

            # Toneladas
            tn = (
                d_cargo.group_by("puerto")
                .agg((pl.col("entrada").sum() / 1000).round(0).alias("miles"))
                .sort("miles", descending=True).head(TOP_N).sort("miles")
            )

            # TEUs
            te = (
                d_cargo.filter(pl.col("tipo_carga") == "Contenerizada")
                .group_by("puerto").agg(pl.col("teus").sum())
                .drop_nulls("teus")
                .sort("teus", descending=True).head(TOP_N).sort("teus")
            )

            # KPIs
            n_p = d_period["puerto"].n_unique()
            n_b = int(d_cargo["buques"].drop_nulls().sum())
            n_t = int(d_cargo["entrada"].drop_nulls().sum())
            n_pa = int(d_pas["pasajeros"].drop_nulls().sum())

            result[key] = {
                "buques": {"y": bq["puerto"].to_list(),
                           "v": bq["buques"].fill_null(0).to_list()},
                "tipo":   {"ports": top_ports, "data": tipo_data},
                "ton":    {"y": tn["puerto"].to_list(),
                           "v": tn["miles"].fill_null(0).cast(pl.Int64).to_list()},
                "teus":   {"y": te["puerto"].to_list(),
                           "v": te["teus"].fill_null(0).cast(pl.Int64).to_list()},
                "pas":    {"y": pas_agg["puerto"].to_list(),
                           "v": pas_agg["pasajeros"].fill_null(0).cast(pl.Int64).to_list()},
                "kpi":    {"p": n_p, "b": _fmt_num(n_b), "t": _fmt_num(n_t), "pa": _fmt_num(n_pa)},
            }

    return json.dumps(result, ensure_ascii=False, separators=(",", ":"))


# ── Render default charts (todos|todos) ───────────────────────────────────────
d_default       = df
d_cargo_default = df.filter(pl.col("tipo_carga") != "pasajeros")

print("Rendering default charts…")
chart_buques   = chart_div(fig_top_buques(d_cargo_default),   "g-buques",    first=True)
chart_tipo     = chart_div(fig_tipo_carga(d_cargo_default),   "g-tipo-carga")
chart_ton      = chart_div(fig_top_toneladas(d_cargo_default),"g-toneladas")
chart_pas      = chart_div(fig_top_pasajeros(d_default),      "g-pasajeros")
chart_teus     = chart_div(fig_top_teus(d_cargo_default),     "g-teus")

print("Pre-computing all filter combinations…")
combo_json = build_combo_data(df)

# ── JavaScript ────────────────────────────────────────────────────────────────
js = """
const CFG  = {responsive:true, displayModeBar:false};
const BLUE = "#2E86AB", GREEN = "#3BB273", CYAN = "#22D3EE", ORG = "#F4A261";
const LAYOUT_BASE = {
    paper_bgcolor:"rgba(0,0,0,0)", plot_bgcolor:"rgba(0,0,0,0)",
    font:{color:"#CBD5E1"},
    xaxis:{gridcolor:"#334155"}, yaxis:{gridcolor:"#334155"},
    margin:{l:10, r:80, t:40, b:20},
};
const TIPO_COLORS = {
    "Agricola":"#3BB273","Contenerizada":"#2E86AB","Mineral":"#94A3B8",
    "Otros F.":"#F4A261","Petroleo":"#E84855","Suelta":"#9B59B6"
};
const TIPOS = ["Agricola","Contenerizada","Mineral","Otros F.","Petroleo","Suelta"];

function fmtNum(v) {
    if (v >= 1e6) return (v/1e6).toFixed(1)+"M";
    if (v >= 1e3) return (v/1e3).toFixed(1)+"K";
    return v.toString();
}

function layout(title, extraMarginR) {
    return Object.assign({}, LAYOUT_BASE, {
        title:{text:title, font:{color:"#CBD5E1", size:13}},
        margin:{l:10, r:extraMarginR||80, t:40, b:20},
    });
}

function barHeight(n) { return Math.max(300, n*28+80); }

function updateCharts() {
    const period  = document.getElementById("dd-periodo").value;
    const trafico = document.getElementById("dd-trafico").value;
    const key     = period + "|" + trafico;
    const D       = DATA[key];
    if (!D) { console.error("No data for key:", key); return; }

    // KPIs
    document.getElementById("kpi-puertos").textContent   = D.kpi.p;
    document.getElementById("kpi-buques").textContent    = D.kpi.b;
    document.getElementById("kpi-toneladas").textContent = D.kpi.t;
    document.getElementById("kpi-pasajeros").textContent = D.kpi.pa;

    // Top buques
    Plotly.react("g-buques", [{
        type:"bar", orientation:"h", x:D.buques.v, y:D.buques.y,
        marker:{color:BLUE},
        text:D.buques.v.map(fmtNum), textposition:"outside",
        hovertemplate:"%{y}: %{x:,}<extra></extra>",
    }], Object.assign(layout("Top puertos por buques", 80),
        {height:barHeight(D.buques.y.length)}), CFG);

    // Tipo carga stacked
    const tipoTraces = TIPOS.map(tipo => {
        const td = D.tipo.data[tipo];
        if (!td) return null;
        return {
            type:"bar", orientation:"h", name:tipo,
            x:td.pct, y:D.tipo.ports,
            marker:{color:TIPO_COLORS[tipo]},
            text:td.pct.map(p => p>4 ? p.toFixed(0)+"%" : ""),
            textposition:"inside", insidetextanchor:"middle",
            customdata:td.n,
            hovertemplate:"<b>"+tipo+"</b>: %{x:.1f}%  (n=%{customdata:,})<extra></extra>",
        };
    }).filter(Boolean);
    Plotly.react("g-tipo-carga", tipoTraces,
        Object.assign(layout("Tipo de carga por puerto (% buques)", 10), {
            barmode:"stack",
            height:barHeight(D.tipo.ports.length),
            xaxis:{range:[0,100], visible:false, gridcolor:"#334155"},
            legend:{orientation:"h", y:-0.12, x:0},
            margin:{l:10, r:10, t:40, b:60},
        }), CFG);

    // Toneladas
    Plotly.react("g-toneladas", [{
        type:"bar", orientation:"h", x:D.ton.v, y:D.ton.y,
        marker:{color:GREEN},
        text:D.ton.v.map(v => v.toLocaleString()+"K"), textposition:"outside",
        hovertemplate:"%{y}: %{x:,}K ton<extra></extra>",
    }], Object.assign(layout("Top puertos por toneladas de entrada (miles)", 100),
        {height:barHeight(D.ton.y.length)}), CFG);

    // Pasajeros (only period filter applies)
    Plotly.react("g-pasajeros", [{
        type:"bar", orientation:"h", x:D.pas.v, y:D.pas.y,
        marker:{color:CYAN},
        text:D.pas.v.map(fmtNum), textposition:"outside",
        hovertemplate:"%{y}: %{x:,}<extra></extra>",
    }], Object.assign(layout("Top puertos por pasajeros (cruceros)", 80),
        {height:barHeight(D.pas.y.length)}), CFG);

    // TEUs
    Plotly.react("g-teus", [{
        type:"bar", orientation:"h", x:D.teus.v, y:D.teus.y,
        marker:{color:ORG},
        text:D.teus.v.map(fmtNum), textposition:"outside",
        hovertemplate:"%{y}: %{x:,} TEUs<extra></extra>",
    }], Object.assign(layout("Top puertos por TEUs (carga contenerizada)", 80),
        {height:barHeight(D.teus.y.length)}), CFG);
}

function applyWhenReady(attempt) {
    attempt = attempt || 0;
    if (attempt > 120) { updateCharts(); return; }
    const el = document.getElementById("g-buques");
    if (el && el.querySelector("svg.main-svg")) {
        updateCharts();
    } else {
        setTimeout(() => applyWhenReady(attempt+1), 80);
    }
}

document.getElementById("dd-periodo").addEventListener("change", function() {
    const url = new URL(window.location.href);
    url.searchParams.set("periodo", this.value);
    window.location.href = url.toString();
});
document.getElementById("dd-trafico").addEventListener("change", function() {
    const url = new URL(window.location.href);
    url.searchParams.set("trafico", this.value);
    window.location.href = url.toString();
});

window.addEventListener("load", function() {
    const p = new URLSearchParams(window.location.search);
    const periodo  = p.get("periodo")  || "todos";
    const trafico  = p.get("trafico")  || "todos";
    document.getElementById("dd-periodo").value = periodo;
    document.getElementById("dd-trafico").value = trafico;
    applyWhenReady();
});
"""

# ── Full HTML ─────────────────────────────────────────────────────────────────
html = f"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Puertos Marítimos de México</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css"
        rel="stylesheet" crossorigin="anonymous">
  <style>
    body       {{ background:#0F172A; color:#CBD5E1; padding:24px;
                 font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }}
    .kpi-card  {{ background:#1E293B; border:1px solid #334155; border-radius:8px;
                 padding:16px; text-align:center; }}
    .kpi-label {{ color:#94A3B8; font-size:.8rem; margin-bottom:4px; }}
    .kpi-value {{ font-weight:700; margin:0; font-size:1.5rem; }}
    .filter-box {{ background:#1E293B; border:1px solid #334155; border-radius:8px;
                   padding:14px 18px; margin-bottom:20px; }}
    .filter-label {{ color:#94A3B8; font-size:.82rem; margin-bottom:6px; display:block; }}
    .dark-select {{ background-color:#1E293B!important; border-color:#334155!important;
                    color:#CBD5E1!important; }}
    .dark-select:focus {{ border-color:#2E86AB!important;
                          box-shadow:0 0 0 .2rem rgba(46,134,171,.25)!important; }}
    .dark-select option {{ background:#1E293B; color:#CBD5E1; }}
  </style>
</head>
<body>
<div class="container-fluid">

  <div class="mb-4">
    <h2 style="color:#F8FAFC;font-weight:700;margin-bottom:4px;">Puertos Marítimos de México</h2>
    <p style="color:#64748B;font-size:.9rem;">Dic 2025 – Ene 2026 · Fuente: SCT / API Puertos · 44 puertos</p>
  </div>

  <!-- Filters -->
  <div class="filter-box">
    <div class="row g-3">
      <div class="col-12 col-md-3">
        <label class="filter-label">Periodo</label>
        <select id="dd-periodo" class="form-select dark-select">
          <option value="todos">Ambos periodos</option>
          <option value="dic2025">Dic 2025</option>
          <option value="ene2026">Ene 2026</option>
        </select>
      </div>
      <div class="col-12 col-md-3">
        <label class="filter-label">Tráfico (carga)</label>
        <select id="dd-trafico" class="form-select dark-select">
          <option value="todos">Todos</option>
          <option value="Altura">Altura</option>
          <option value="Cabotaje">Cabotaje</option>
          <option value="Transbordo">Transbordo</option>
        </select>
      </div>
    </div>
  </div>

  <!-- KPIs -->
  <div class="row g-3 mb-4">
    {kpi_card("Puertos activos",     "kpi-puertos",   "#F8FAFC")}
    {kpi_card("Buques (carga)",      "kpi-buques",    "#2E86AB")}
    {kpi_card("Toneladas entrada",   "kpi-toneladas", "#3BB273")}
    {kpi_card("Pasajeros cruceros",  "kpi-pasajeros", "#22D3EE")}
  </div>

  <!-- Charts row 1 -->
  {row((chart_buques, 6), (chart_tipo, 6))}
  <!-- Charts row 2 -->
  {row((chart_ton, 6), (chart_pas, 6))}
  <!-- Charts row 3 -->
  {row((chart_teus, 12))}

</div>

<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"
        crossorigin="anonymous"></script>
<script>
const DATA = {combo_json};
{js}
</script>
</body>
</html>"""

out = Path("site/puertos.html")
out.write_text(html, encoding="utf-8")
size_kb = len(html.encode()) / 1024
print(f"\nWritten {out}  ({size_kb:.0f} KB)")
