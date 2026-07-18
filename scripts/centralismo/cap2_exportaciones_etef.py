"""
Cap 2 — Extensión: el motor exportador por entidad productora (ETEF).

INEGI ETEF (data/inegi/etef/) publica las exportaciones de mercancías por entidad
federativa PRODUCTORA × subsector SCIAN, 2007-2025 — el dato que las aduanas (BCMM)
no dan: dónde se produce lo que se exporta, no por dónde cruza. Tres tests:

  A. Descentramiento (H2): shares estatales/regionales del motor exportador,
     su evolución 2007-2025, concentración y β-convergencia de la plataforma.
  B. Asimetría sede-producción (H1): CDMX produce ~0.65% de la exportación pero
     concentra la recaudación por domicilio fiscal (55.7% en 2020, Ríos & Saucedo)
     — contraste con el share de PIB y la recaudación imputada por incidencia.
  C. Factor exportador (H5): mismo diseño que cap5_remesas.py — ¿la intensidad
     exportadora (USD/hab) es el factor omitido de la ecuación de crecimiento?

Trampas ETEF: VAL_USD en MILES de USD; celdas estado-subsector suprimidas
(No disponible/Confidencial = null) — los totales estatales usan la suma de
disponibles y cuadran contra el total publicado; CVE_ENT ya es entero 1-32.

Figuras → centralismo/informe/figuras/cap2_etef_*.png
Run: uv run python scripts/centralismo/cap2_exportaciones_etef.py
"""

import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import polars as pl
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from comun import (CORTO, PIBE_TOTAL, RAIZ, REGION_DE, cargar_poblacion,
                   guardar_fig, leer_pibe)
from cap1_recaudacion_imputada import cargar_aporte_imputado
from cap5_factor_competencia import (A0, A1, componentes, crecimiento_pib,
                                     dependencia_fiscal, distancia_cdmx)
from cap5_remesas import COLS5, indice_pca, ols

ZIP_ETEF = RAIZ / "data/inegi/etef/conjunto_de_datos_eef_csv.zip"
CDMX, CAMPECHE, TABASCO = 9, 4, 27
SCIAN_PETROLEO = 211  # Extracción de petróleo y gas


def cargar_etef() -> pl.DataFrame:
    """ETEF anual → cve_ent, año, x_musd (total) y x_musd_sp (sin petróleo)."""
    with zipfile.ZipFile(ZIP_ETEF) as z:
        df = pl.read_csv(io.BytesIO(z.read(
            "conjunto_de_datos/eef_estatal_anual_tr_cifra_2007_2025.csv")),
            infer_schema_length=5000)
    # VAL_USD en MILES USD; null = suprimido (No disponible/Confidencial) — sum lo omite
    out = (df.group_by("ANIO", "CVE_ENT")
           .agg((pl.col("VAL_USD").sum() / 1e3).alias("x_musd"),
                (pl.col("VAL_USD").filter(pl.col("CODIGO_SCIAN") != SCIAN_PETROLEO)
                 .sum() / 1e3).alias("x_musd_sp"))
           .rename({"ANIO": "año", "CVE_ENT": "cve_ent"}))
    assert out.group_by("año").len()["len"].eq(32).all()
    tot24 = out.filter(pl.col("año") == 2024)["x_musd"].sum()
    assert abs(tot24 / 559_922.280 - 1) < 0.001, tot24
    return out


def base_estatal(etef: pl.DataFrame, pob: pl.DataFrame) -> pl.DataFrame:
    return (etef.join(pob, on=["cve_ent", "año"])
            .with_columns(
                (pl.col("x_musd") / pl.col("x_musd").sum().over("año")).alias("share"),
                (pl.col("x_musd") * 1e6 / pl.col("pob_total")).alias("x_pc"),
                (pl.col("x_musd_sp") * 1e6 / pl.col("pob_total")).alias("x_pc_sp"),
                pl.col("cve_ent").replace_strict(CORTO).alias("estado"),
                pl.col("cve_ent").replace_strict(REGION_DE).alias("region")))


def main():
    pob = cargar_poblacion()
    b = base_estatal(cargar_etef(), pob)
    b24 = b.filter(pl.col("año") == 2024)

    print("=== A. Descentramiento: ¿dónde se produce la exportación? (H2) ===")
    top = b24.sort("share", descending=True)
    print("  top-6 2024:", " · ".join(f"{r['estado']}={r['share'] * 100:.1f}%"
                                      for r in top.head(6).iter_rows(named=True)))
    cdmx = b24.filter(pl.col("cve_ent") == CDMX)
    rk = top.with_row_index("rank").filter(pl.col("cve_ent") == CDMX)
    print(f"  CDMX 2024: {cdmx['share'][0] * 100:.2f}% de la exportación nacional "
          f"(lugar {rk['rank'][0] + 1}/32) — con 24.8% de la actividad gubernamental")
    assert cdmx["share"][0] < 0.01

    reg = (b.group_by("año", "region").agg(pl.sum("x_musd"))
           .with_columns((pl.col("x_musd") / pl.col("x_musd").sum().over("año"))
                         .alias("share")))
    pr = (pob.filter(pl.col("año") == 2024)
          .with_columns(pl.col("cve_ent").replace_strict(REGION_DE).alias("region"))
          .group_by("region").agg(pl.sum("pob_total")))
    pr = pr.with_columns((pl.col("pob_total") / pr["pob_total"].sum()).alias("s_pob"))
    print("  región (share export 2007 → 2024 · share población):")
    for r in (reg.pivot(on="año", index="region", values="share")
              .join(pr, on="region").sort("2024", descending=True)
              .iter_rows(named=True)):
        print(f"    {r['region']:<12} {r['2007'] * 100:>5.1f}% → {r['2024'] * 100:>5.1f}% "
              f"· pob {r['s_pob'] * 100:.1f}%")
    hhi = b.group_by("año").agg((pl.col("share") ** 2).sum().alias("hhi") * 1e4)
    print(f"  HHI estatal: {hhi.filter(pl.col('año') == 2007)['hhi'][0]:.0f} (2007) → "
          f"{hhi.filter(pl.col('año') == 2024)['hhi'][0]:.0f} (2024)")

    # β-convergencia de la plataforma exportadora (sin petróleo, log pc)
    conv = (b.filter(pl.col("año") == 2007).select("cve_ent", "estado", "x_pc_sp")
            .join(b.filter(pl.col("año") == 2025)
                  .select("cve_ent", pl.col("x_pc_sp").alias("x_fin")), on="cve_ent")
            .with_columns((np.log(pl.col("x_fin") / pl.col("x_pc_sp")) / (2025 - 2007)
                           * 100).alias("g")))
    beta, alfa = np.polyfit(np.log(conv["x_pc_sp"].to_numpy()), conv["g"].to_numpy(), 1)
    r_conv, p_conv = pearsonr(np.log(conv["x_pc_sp"].to_numpy()), conv["g"].to_numpy())
    print(f"  β-convergencia export pc sin petróleo 2007→2025: β={beta:+.2f} "
          f"(negativo = los rezagados crecen más rápido) · corr = {r_conv:+.2f} "
          f"(p={p_conv:.2f})")

    print("\n=== B. Asimetría sede-producción (H1) ===")
    pib24 = leer_pibe(PIBE_TOTAL, bloque="Millones de pesos").filter(pl.col("año") == 2024)
    pib24 = pib24.with_columns((pl.col("valor") / pib24["valor"].sum()).alias("s_pib"))
    rec24 = (cargar_aporte_imputado().filter(pl.col("año") == 2024)
             .select("cve_ent", pl.col("share_recaudacion").alias("s_rec")))
    tri = (b24.select("cve_ent", "estado", "share")
           .join(pib24.select("cve_ent", "s_pib"), on="cve_ent")
           .join(rec24, on="cve_ent"))
    print(f"  {'estado':<12} {'export':>7} {'PIB':>7} {'rec.imputada':>13}")
    for r in tri.sort("share", descending=True).head(6).iter_rows(named=True):
        print(f"  {r['estado']:<12} {r['share'] * 100:>6.1f}% {r['s_pib'] * 100:>6.1f}% "
              f"{r['s_rec'] * 100:>12.1f}%")
    rc = tri.filter(pl.col("cve_ent") == CDMX).row(0, named=True)
    print(f"  {'CDMX':<12} {rc['share'] * 100:>6.1f}% {rc['s_pib'] * 100:>6.1f}% "
          f"{rc['s_rec'] * 100:>12.1f}%")
    print("  lectura: la capital produce <1% del bien exportable pero registró 55.7% de "
          "la recaudación por domicilio fiscal (2020, Ríos & Saucedo) — la sede no es "
          "la fábrica; la recaudación imputada por incidencia la baja a "
          f"{rc['s_rec'] * 100:.1f}%")

    print(f"\n=== C. Factor exportador en la ecuación de crecimiento {A0}→{A1} (H5) ===")
    x18 = (b.filter(pl.col("año") == 2018)
           .select("cve_ent", pl.col("x_pc_sp").alias("exp18")))
    xwin = (b.filter(pl.col("año").is_between(A0, A1))
            .group_by("cve_ent").agg(pl.mean("x_pc_sp").alias("exp_win")))
    comp = componentes()
    pc1, _ = indice_pca(comp, COLS5)
    base = (comp.with_columns(pl.Series("indice", pc1))
            .join(x18, on="cve_ent").join(xwin, on="cve_ent")
            .join(crecimiento_pib().select("cve_ent", "crec"), on="cve_ent")
            .join(dependencia_fiscal(), on="cve_ent")
            .join(distancia_cdmx().select("cve_ent", "dist_cdmx"), on="cve_ent")
            .with_columns(pl.col("cve_ent").replace_strict(CORTO).alias("estado")))
    print(f"  corr(export pc 2018, índice de potencial) = "
          f"{np.corrcoef(base['exp18'], base['indice'])[0, 1]:+.2f} · "
          f"corr(export pc win, crec) = "
          f"{np.corrcoef(base['exp_win'], base['crec'])[0, 1]:+.2f}")
    y = base["crec"].to_numpy()
    x_ind = base["indice"].to_numpy()
    x_dep = base["dependencia"].to_numpy() * 100
    x_dis = base["dist_cdmx"].to_numpy() / 100
    print("  M0 — réplica Cap 5 (índice + dependencia + distancia):")
    ols(y, [x_ind, x_dep, x_dis],
        ["índice potencial", "dependencia fiscal (pp)", "distancia CDMX (100 km)"])
    for var in ("exp18", "exp_win"):
        print(f"  M1 — M0 + export pc sin petróleo ({var}, por 1000 USD/hab):")
        ols(y, [x_ind, x_dep, x_dis, base[var].to_numpy() / 1000],
            ["índice potencial", "dependencia fiscal (pp)", "distancia CDMX (100 km)",
             f"export pc ({var})"])
    print("  robustez sin Campeche/Tabasco (plataforma petrolera declinante):")
    sc = base.filter(~pl.col("cve_ent").is_in([CAMPECHE, TABASCO]))
    ols(sc["crec"].to_numpy(),
        [sc["indice"].to_numpy(), sc["dependencia"].to_numpy() * 100,
         sc["dist_cdmx"].to_numpy() / 100, sc["exp_win"].to_numpy() / 1000],
        ["índice potencial", "dependencia fiscal (pp)", "distancia CDMX (100 km)",
         "export pc (exp_win)"])
    print("  caveats: USD nominales (válido transversal); la exportación bruta no es "
          "valor agregado local (contenido importado de la maquila); n=32, correlacional")

    # figuras -----------------------------------------------------------------
    orden_reg = reg.filter(pl.col("año") == 2024).sort("share", descending=True)["region"]
    fig = go.Figure()
    for region in orden_reg:
        d = reg.filter(pl.col("region") == region).sort("año")
        fig.add_trace(go.Scatter(x=d["año"], y=d["share"] * 100, name=region))
    fig.update_layout(
        title="El motor exportador es periférico y estable<br>"
              "<sup>share regional de las exportaciones por entidad productora "
              "(ETEF, USD corrientes) · CDMX 2024 = 0.65%</sup>",
        xaxis_title="año", yaxis_title="% de la exportación nacional")
    guardar_fig(fig, "cap2_etef_shares")

    fig = go.Figure()
    for col, nombre in [("share", "exportación producida (ETEF)"),
                        ("s_pib", "PIB"), ("s_rec", "recaudación imputada")]:
        d = tri.sort("s_pib", descending=True).head(8)
        fig.add_trace(go.Bar(x=d["estado"], y=d[col] * 100, name=nombre))
    fig.update_layout(
        barmode="group",
        title="La sede no es la fábrica: producción exportadora vs peso económico y "
              "fiscal (2024)<br><sup>8 mayores economías estatales · la recaudación por "
              "domicilio fiscal asignaba 55.7% a CDMX (2020, Ríos & Saucedo)</sup>",
        yaxis_title="% del total nacional")
    guardar_fig(fig, "cap2_etef_sede_produccion")

    bb, aa = np.polyfit(base["exp_win"].to_numpy() / 1000, y, 1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=base["exp_win"] / 1000, y=base["crec"], mode="markers+text",
        text=base["estado"], textposition="top center", textfont=dict(size=9),
        showlegend=False,
        marker=dict(size=10, color=base["indice"], colorscale="RdYlGn",
                    colorbar=dict(title="índice de<br>potencial"))))
    xs = np.array([float(base["exp_win"].min()), float(base["exp_win"].max())]) / 1000
    fig.add_trace(go.Scatter(x=xs, y=aa + bb * xs, mode="lines", showlegend=False,
                             line=dict(color="#7f8c8d", dash="dash", width=1)))
    fig.update_layout(
        title=f"¿La plataforma exportadora predice el crecimiento? (test H5)<br>"
              f"<sup>export pc sin petróleo (promedio {A0}-{A1}, ETEF) vs crecimiento "
              f"del PIB pc sin petróleo · color = índice de potencial</sup>",
        xaxis_title=f"exportación por habitante (miles de USD, promedio {A0}-{A1})",
        yaxis_title="crecimiento anual promedio del PIB pc (%)")
    guardar_fig(fig, "cap2_etef_crecimiento")


if __name__ == "__main__":
    main()
