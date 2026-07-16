"""
Cap 2 — ¿Crecimiento acotado por el centro? (H2)

Tests:
  A. Convergencia β: crecimiento del PIB pc real 2003-2024 ~ nivel inicial (OLS).
  B. Convergencia σ: dispersión de ln(PIB pc) entre estados por año.
  C. Concentración: share de CDMX en el PIB nacional, top-5, HHI; share de CDMX en
     "actividades gubernamentales" (PIBE_46) = centralización administrativa medible.
  D. Empleo formal IMSS 1997-2026: share de CDMX y formalización por estado.

PIB real a precios de 2018, con y sin minería petrolera (Campeche/Tabasco distorsionan).
Figuras → centralismo/informe/figuras/cap2_*.png
Run: uv run python scripts/centralismo/cap2_convergencia.py
"""

import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import polars as pl
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from comun import (CORTO, PIBE_GOBIERNO, PIBE_MINERIA_PETROLERA, PIBE_TOTAL,
                   RAIZ, cargar_poblacion, guardar_fig, leer_pibe)

A0, A1 = 2003, 2024


def panel_pib_real() -> pl.DataFrame:
    pib = leer_pibe(PIBE_TOTAL).rename({"valor": "pib"})
    petro = leer_pibe(PIBE_MINERIA_PETROLERA).rename({"valor": "petro"})
    pob = cargar_poblacion()
    return (
        pib.join(petro, on=["cve_ent", "año"], how="left")
        .with_columns((pl.col("pib") - pl.col("petro").fill_null(0)).alias("pib_sp"))
        .join(pob, on=["cve_ent", "año"])
        .with_columns(
            (pl.col("pib") * 1e6 / pl.col("pob_total")).alias("pib_pc"),
            (pl.col("pib_sp") * 1e6 / pl.col("pob_total")).alias("pib_pc_sp"),
        )
    )


def beta_convergencia(panel: pl.DataFrame, col: str, etiqueta: str):
    w = (
        panel.filter(pl.col("año").is_in([A0, A1]))
        .pivot(on="año", index="cve_ent", values=col)
        .with_columns(
            np.log(pl.col(str(A0))).alias("ln0"),
            ((np.log(pl.col(str(A1))) - np.log(pl.col(str(A0)))) / (A1 - A0) * 100).alias("crec"),
        )
    )
    X = sm.add_constant(w["ln0"].to_numpy())
    modelo = sm.OLS(w["crec"].to_numpy(), X).fit()
    b, p = modelo.params[1], modelo.pvalues[1]
    print(f"β-convergencia [{etiqueta}]: β={b:+.3f} (p={p:.3f}, R²={modelo.rsquared:.2f}) "
          f"→ {'CONVERGENCIA' if b < 0 and p < 0.05 else 'divergencia' if b > 0 and p < 0.05 else 'sin patrón significativo'}")
    if b < 0:
        # velocidad anual y vida media del gap (aprox. λ = -b/100)
        lam = -b / 100
        print(f"   velocidad λ≈{lam*100:.2f}%/año; vida media del rezago ≈ {np.log(2)/lam:.0f} años")
    return w, modelo


def main():
    panel = panel_pib_real()

    # ---- A. β-convergencia
    print(f"=== A. Convergencia β ({A0}→{A1}, PIB pc real 2018) ===")
    w_sp, mod_sp = beta_convergencia(panel, "pib_pc_sp", "sin petróleo")
    beta_convergencia(panel, "pib_pc", "con petróleo")

    w = w_sp.with_columns(pl.col("cve_ent").replace_strict(CORTO).alias("estado"))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=w["ln0"], y=w["crec"], mode="markers+text", text=w["estado"],
        textposition="top center", textfont=dict(size=9),
        marker=dict(size=8, color="#1f77b4"), showlegend=False))
    xs = np.linspace(float(w["ln0"].min()), float(w["ln0"].max()), 50)
    fig.add_trace(go.Scatter(x=xs, y=mod_sp.params[0] + mod_sp.params[1] * xs, mode="lines",
                             line=dict(color="#c0392b"),
                             name=f"OLS β={mod_sp.params[1]:+.2f} (p={mod_sp.pvalues[1]:.2f})"))
    fig.update_layout(
        title=f"Convergencia β {A0}–{A1}: ¿crecen más rápido los estados pobres?<br><sup>PIB per cápita real (2018) sin minería petrolera; pendiente negativa = convergencia</sup>",
        xaxis_title=f"ln(PIB pc {A0})", yaxis_title="crecimiento anual promedio (%)",
    )
    guardar_fig(fig, "cap2_beta_convergencia")

    # ---- B. σ-convergencia
    sigma = (
        panel.group_by("año")
        .agg(
            pl.col("pib_pc_sp").log().std().alias("sin petróleo"),
            pl.col("pib_pc").log().std().alias("con petróleo"),
        )
        .sort("año")
    )
    s0 = sigma.filter(pl.col("año") == A0)["sin petróleo"][0]
    s1 = sigma.filter(pl.col("año") == A1)["sin petróleo"][0]
    print(f"\n=== B. Convergencia σ ===\nσ ln(PIB pc sp): {A0}={s0:.3f} → {A1}={s1:.3f} "
          f"({'converge' if s1 < s0 else 'DIVERGE'}, Δ={100*(s1/s0-1):+.1f}%)")
    fig = go.Figure()
    for c in ["sin petróleo", "con petróleo"]:
        fig.add_trace(go.Scatter(x=sigma["año"], y=sigma[c], mode="lines+markers", name=c))
    fig.update_layout(
        title="Convergencia σ: dispersión interestatal del PIB per cápita real<br><sup>desv. estándar de ln(PIB pc) entre los 32 estados; a la baja = los estados se parecen más</sup>",
        xaxis_title="año", yaxis_title="σ de ln(PIB pc)",
    )
    guardar_fig(fig, "cap2_sigma_convergencia")

    # ---- C. concentración del PIB y del aparato de gobierno
    shares = panel.with_columns((pl.col("pib_sp") / pl.col("pib_sp").sum().over("año")).alias("sh"))
    conc = (
        shares.group_by("año")
        .agg(
            (pl.col("sh") ** 2).sum().alias("hhi"),
            pl.col("sh").sort(descending=True).head(5).sum().alias("top5"),
        )
        .sort("año")
    )
    cdmx = shares.filter(pl.col("cve_ent") == 9).select("año", pl.col("sh").alias("cdmx")).sort("año")
    gob = leer_pibe(PIBE_GOBIERNO)
    gob = gob.with_columns((pl.col("valor") / pl.col("valor").sum().over("año")).alias("sh_gob"))
    gob_cdmx = gob.filter(pl.col("cve_ent") == 9).select("año", "sh_gob").sort("año")

    print(f"\n=== C. Concentración (PIB sin petróleo) ===")
    for a in [A0, 2013, A1]:
        r = conc.filter(pl.col("año") == a).row(0, named=True)
        c = cdmx.filter(pl.col("año") == a)["cdmx"][0]
        g = gob_cdmx.filter(pl.col("año") == a)["sh_gob"][0]
        print(f"  {a}: CDMX={c*100:.1f}% del PIB · top5={r['top5']*100:.1f}% · HHI={r['hhi']:.3f} · "
              f"CDMX en act. gubernamentales={g*100:.1f}%")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cdmx["año"], y=cdmx["cdmx"] * 100, name="CDMX — share del PIB nacional"))
    fig.add_trace(go.Scatter(x=gob_cdmx["año"], y=gob_cdmx["sh_gob"] * 100,
                             name="CDMX — share de actividades gubernamentales"))
    fig.add_trace(go.Scatter(x=conc["año"], y=conc["top5"] * 100, name="Top-5 estados — share del PIB",
                             line=dict(dash="dot")))
    fig.update_layout(
        title="Concentración territorial de la economía y del aparato estatal (PIB sin petróleo)<br><sup>PIBE_46 = valor agregado de actividades legislativas/gubernamentales/justicia</sup>",
        xaxis_title="año", yaxis_title="% del total nacional",
    )
    guardar_fig(fig, "cap2_concentracion")

    # ---- D. empleo formal IMSS
    imss = (
        pl.scan_csv(RAIZ / "data/datamx/empleo_formal/empleo_formal.csv")
        .with_columns(pl.col("PERIODO").str.slice(0, 4).cast(pl.Int64).alias("año"))
        .group_by("año", "CVE_ENT", "PERIODO")
        .agg(pl.sum("TOTAL"))
        .group_by("año", "CVE_ENT")
        .agg(pl.mean("TOTAL").alias("empleos"))  # promedio de meses del año
        .collect()
        .rename({"CVE_ENT": "cve_ent"})
    )
    sh_imss = imss.with_columns((pl.col("empleos") / pl.col("empleos").sum().over("año")).alias("sh"))
    print("\n=== D. Empleo formal IMSS ===")
    for a in [1998, 2010, 2025]:
        top = sh_imss.filter(pl.col("año") == a).sort("sh", descending=True).head(3)
        print(f"  {a}: " + " · ".join(f"{CORTO[r['cve_ent']]}={r['sh']*100:.1f}%" for r in top.iter_rows(named=True)))

    fig = go.Figure()
    for cve in [9, 15, 19, 14]:
        s = sh_imss.filter((pl.col("cve_ent") == cve) & pl.col("año").is_between(1998, 2025)).sort("año")
        fig.add_trace(go.Scatter(x=s["año"], y=s["sh"] * 100, mode="lines", name=CORTO[cve]))
    fig.update_layout(
        title="Share del empleo formal nacional (puestos IMSS), 1998–2025",
        xaxis_title="año", yaxis_title="% de los puestos IMSS del país",
    )
    guardar_fig(fig, "cap2_imss_share")

    # formalización: puestos IMSS por 100 habitantes
    pob = cargar_poblacion()
    form = imss.join(pob, on=["cve_ent", "año"]).with_columns(
        (pl.col("empleos") / pl.col("pob_total") * 100).alias("form_pc"))
    f25 = form.filter(pl.col("año") == 2025).sort("form_pc", descending=True)
    print("  formalización 2025 (puestos IMSS por 100 hab): top3 " +
          ", ".join(f"{CORTO[r['cve_ent']]}={r['form_pc']:.1f}" for r in f25.head(3).iter_rows(named=True)) +
          " | bottom3 " +
          ", ".join(f"{CORTO[r['cve_ent']]}={r['form_pc']:.1f}" for r in f25.tail(3).iter_rows(named=True)))

    # ---- E. crecimiento vs dependencia de transferencias (preview cap4/5)
    cp = pl.read_parquet(RAIZ / "informe_data/cp_estado_ramo.parquet")
    tr = (cp.filter(pl.col("id_ramo").is_in([28, 33]) & pl.col("cve_ent").is_between(1, 32))
          .group_by("ciclo", "cve_ent").agg(pl.sum("monto_ejercido").alias("transfer")))
    pibn = leer_pibe(PIBE_TOTAL, bloque="Millones de pesos").rename({"valor": "pib_nom"})
    dep = (tr.rename({"ciclo": "año"}).join(pibn, on=["cve_ent", "año"])
           .with_columns((pl.col("transfer") / (pl.col("pib_nom") * 1e6)).alias("dep_transfer"))
           .filter(pl.col("año").is_between(2013, 2024))
           .group_by("cve_ent").agg(pl.mean("dep_transfer")))
    cruce = w_sp.join(dep, on="cve_ent")
    c = np.corrcoef(cruce["dep_transfer"], cruce["crec"])[0, 1]
    print(f"\n=== E. corr(dependencia transferencias/PIB 2013-24, crecimiento PIB pc) = {c:+.3f} ===")


if __name__ == "__main__":
    main()
