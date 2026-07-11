"""
Cap 7 — Extensión dinámica: ¿la desigualdad sube ANTES de que suba la violencia?

El Cap 7 midió desigualdad y violencia en niveles (transversal); Enamorado et al.
(2016, JDE) encuentran que la desigualdad — no la pobreza — predice la violencia.
Aquí el test dinámico con los dos paneles largos disponibles en el repo:

  A. Desigualdad intra-estatal: Theil salarial ENTRE subsectores IMSS (62 grupos,
     ponderado por puestos; masa salarial diaria / puestos), estado × año 1998-2025.
     Es desigualdad del sector FORMAL inter-industria — un proxy, no un Gini de
     ingreso de hogares (que no existe en panel largo).
  B. Violencia: víctimas de homicidio INEGI (X85-Y09, entidad de ocurrencia),
     2012-2023 (registros 2012-2024; los tardíos caen en el archivo siguiente).
  C. Tests: (i) panel lead-lag con FE de estado y año (k = 0..5, errores por
     estado); (ii) primeras diferencias largas; (iii) nacional: Gini interestatal
     del PIB pc (serie completa 2003-2024, extiende el punto 1 del Cap 7) × tasa.

Figura → centralismo/informe/figuras/cap7_desigualdad_dinamica.png
Run: uv run python scripts/centralismo/cap7_desigualdad_dinamica.py
"""

import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import polars as pl
import statsmodels.formula.api as smf
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).resolve().parent))
from comun import (CORTO, PIBE_MINERIA_PETROLERA, PIBE_TOTAL, RAIZ,
                   cargar_poblacion, guardar_fig, leer_pibe)
from cap7_desigualdad_violencia import gini

A_INEGI = list(range(2012, 2024))  # años de ocurrencia con registro consolidado


def theil_imss(nivel: str = "SUBSECTOR") -> pl.DataFrame:
    """Theil salarial entre grupos sectoriales IMSS por estado-año."""
    agg = (pl.scan_csv(RAIZ / "data/datamx/empleo_formal/empleo_formal.csv")
           .with_columns(pl.col("PERIODO").str.slice(0, 4).cast(pl.Int64).alias("año"))
           .filter(pl.col("año") >= 1998)  # 1997 solo tiene ago-dic
           .group_by("año", "CVE_ENT", nivel)
           .agg(pl.sum("TOTAL").alias("j"), pl.sum("MASA_SALARIAL_TOTAL").alias("m"))
           .filter((pl.col("j") > 0) & (pl.col("m") > 0))
           .collect())
    t = (agg.with_columns((pl.col("m") / pl.col("j")).alias("x"),
                          (pl.col("j") / pl.col("j").sum().over("año", "CVE_ENT")).alias("p"),
                          (pl.col("m").sum().over("año", "CVE_ENT")
                           / pl.col("j").sum().over("año", "CVE_ENT")).alias("mu"))
         .with_columns((pl.col("p") * (pl.col("x") / pl.col("mu"))
                        * (pl.col("x") / pl.col("mu")).log()).alias("term"))
         .group_by("año", "CVE_ENT").agg(pl.sum("term").alias("theil"),
                                         pl.len().alias("grupos"))
         .rename({"CVE_ENT": "cve_ent"}).sort("cve_ent", "año"))
    return t


def homicidios_inegi() -> pl.DataFrame:
    """Víctimas X85-Y09 por entidad de ocurrencia y año de ocurrencia, 2012-2023."""
    archivos = (
        [f"defunciones_base_datos_{a}_csv.zip" for a in range(2012, 2017)]
        + ["conjunto_de_datos_defunciones_generales_2017_csv.zip"]
        + [f"conjunto_de_datos_defunciones_registradas_{a}_csv.zip"
           for a in range(2018, 2024)]
        + ["conjunto_de_datos_edr2024_csv.zip"])
    partes = []
    for zname in archivos:
        z = zipfile.ZipFile(RAIZ / "data/inegi/defunciones" / zname)
        csvs = [n for n in z.namelist()
                if "conjunto_de_datos/" in n and n.lower().endswith(".csv")]
        main = max(csvs, key=lambda n: z.getinfo(n).file_size)
        partes.append(
            pl.read_csv(io.BytesIO(z.read(main)), infer_schema_length=0,
                        columns=["ent_ocurr", "causa_def", "anio_ocur"])
            .filter(pl.col("causa_def").str.contains(r"^(X8[5-9]|X9|Y0)"))
            # 2018-2019 traen un tab colgante en los campos numéricos
            .with_columns(pl.col("ent_ocurr").str.strip_chars().cast(pl.Int64, strict=False),
                          pl.col("anio_ocur").str.strip_chars().cast(pl.Int64, strict=False)))
    hom = (pl.concat(partes)
           .filter(pl.col("anio_ocur").is_in(A_INEGI) & pl.col("ent_ocurr").is_between(1, 32))
           .group_by("ent_ocurr", "anio_ocur").agg(pl.len().alias("victimas"))
           .rename({"ent_ocurr": "cve_ent", "anio_ocur": "año"}))
    return (hom.join(cargar_poblacion(), on=["cve_ent", "año"])
            .with_columns((pl.col("victimas") / pl.col("pob_total") * 1e5).alias("tasa")))


def main():
    print("=== A. Panel de desigualdad salarial formal (IMSS, Theil entre subsectores) ===")
    th = theil_imss("SUBSECTOR")
    th_s = theil_imss("SECTOR").rename({"theil": "theil_sector"})
    chk = th.join(th_s.select("año", "cve_ent", "theil_sector"), on=["año", "cve_ent"])
    print(f"  {th.height} estado-años ({th['año'].min()}-{th['año'].max()}) · "
          f"mediana de grupos por estado-año: {th['grupos'].median():.0f}")
    print(f"  robustez de nivel: corr(Theil subsector, Theil sector 1-díg) = "
          f"{np.corrcoef(chk['theil'], chk['theil_sector'])[0, 1]:+.2f}")
    for a in (1998, 2010, 2025):
        d = th.filter(pl.col("año") == a)
        arriba = d.sort("theil", descending=True).head(1)
        print(f"  {a}: Theil mediano = {d['theil'].median():.3f} · "
              f"máximo {CORTO[arriba['cve_ent'][0]]} ({arriba['theil'][0]:.3f})")

    print("\n=== B. Panel de violencia (víctimas INEGI X85-Y09, 2012-2023) ===")
    hom = homicidios_inegi()
    n12 = hom.filter(pl.col("año") == 2012)["victimas"].sum()
    n23 = hom.filter(pl.col("año") == 2023)["victimas"].sum()
    print(f"  víctimas nacionales: {n12:,} (2012) → {n23:,} (2023)")

    print("\n=== C. Tests dinámicos ===")
    base = hom.select("cve_ent", "año", "tasa")
    mu, sd = th["theil"].mean(), th["theil"].std()
    betas = []
    for k in range(6):
        lagged = th.select("cve_ent", (pl.col("año") + k).alias("año"),
                           ((pl.col("theil") - mu) / sd).alias("theil_z"))
        pan = base.join(lagged, on=["cve_ent", "año"]).to_pandas()
        m = smf.ols("tasa ~ theil_z + C(cve_ent) + C(año)", data=pan).fit(
            cov_type="cluster", cov_kwds={"groups": pan["cve_ent"]})
        betas.append((k, m.params["theil_z"], *m.conf_int().loc["theil_z"],
                      m.pvalues["theil_z"], int(m.nobs)))
        print(f"  (i) FE estado+año, rezago k={k}: β = {betas[-1][1]:+.2f} "
              f"por 100k por +1 DE de Theil (p={betas[-1][4]:.3f}, n={betas[-1][5]})")

    d_th = (th.filter(pl.col("año").is_in([2012, 2017]))
            .pivot(values="theil", index="cve_ent", on="año")
            .with_columns((pl.col("2017") - pl.col("2012")).alias("d_theil")))
    d_h = (base.filter(pl.col("año").is_in([2018, 2023]))
           .pivot(values="tasa", index="cve_ent", on="año")
           .with_columns((pl.col("2023") - pl.col("2018")).alias("d_tasa")))
    dd = d_th.join(d_h, on="cve_ent")
    c_d = np.corrcoef(dd["d_theil"], dd["d_tasa"])[0, 1]
    arriba = dd.sort("d_theil", descending=True)
    print(f"  (ii) diferencias largas: corr(ΔTheil 2012→17, Δtasa 2018→23) = {c_d:+.2f}"
          f" · Δtasa media mitad-alta de ΔTheil = "
          f"{arriba.head(16)['d_tasa'].mean():+.1f} vs mitad-baja = "
          f"{arriba.tail(16)['d_tasa'].mean():+.1f}")

    # (iii) nacional: Gini interestatal del PIB pc (sin petróleo) 2003-2024
    pib = leer_pibe(PIBE_TOTAL).rename({"valor": "pib"})
    petro = leer_pibe(PIBE_MINERIA_PETROLERA).rename({"valor": "petro"})
    p = (pib.join(petro, on=["cve_ent", "año"], how="left")
         .join(cargar_poblacion(), on=["cve_ent", "año"])
         .with_columns(((pl.col("pib") - pl.col("petro").fill_null(0)) * 1e6
                        / pl.col("pob_total")).alias("pc")))
    serie_g = []
    for a in sorted(p["año"].unique().to_list()):
        d = p.filter(pl.col("año") == a)
        serie_g.append((a, gini(d["pc"].to_numpy(), d["pob_total"].to_numpy())))
    gser = pl.DataFrame({"año": [a for a, _ in serie_g], "gini": [g for _, g in serie_g]})
    assert abs(gser.filter(pl.col("año") == 2003)["gini"][0] - 0.242) < 0.002  # ancla Cap 7
    nac = (hom.group_by("año").agg(pl.sum("victimas"), pl.sum("pob_total"))
           .with_columns((pl.col("victimas") / pl.col("pob_total") * 1e5).alias("tasa"))
           .sort("año"))
    print("  (iii) Gini interestatal PIB pc: " + " · ".join(
        f"{r['año']}: {r['gini']:.3f}" for r in gser.iter_rows(named=True)
        if r["año"] in (2003, 2008, 2013, 2018, 2024)))
    g_j = gser.join(nac.select("año", "tasa"), on="año")
    for k in (0, 3, 5):
        gk = (gser.with_columns((pl.col("año") + k).alias("año"))
              .rename({"gini": "gini_k"}).join(nac.select("año", "tasa"), on="año"))
        print(f"        corr(Gini interestatal t−{k}, tasa nacional t) = "
              f"{np.corrcoef(gk['gini_k'], gk['tasa'])[0, 1]:+.2f} (n={gk.height})")
    print("  caveats: Theil = desigualdad FORMAL inter-industria (proxy, sin hogares "
          "ni informales); n=32; el panel FE identifica con variación temporal "
          "dentro de estado — lenta en ambas series")

    # figura: coeficientes por rezago + series nacionales
    fig = make_subplots(
        rows=1, cols=2, column_widths=[0.45, 0.55],
        specs=[[{}, {"secondary_y": True}]],
        subplot_titles=("panel FE: β de Theil(t−k) sobre tasa(t)",
                        "nacional: Gini interestatal vs tasa de homicidio"))
    ks = [b[0] for b in betas]
    fig.add_trace(go.Scatter(
        x=ks, y=[b[1] for b in betas], mode="markers", showlegend=False,
        error_y=dict(type="data", symmetric=False,
                     array=[b[3] - b[1] for b in betas],
                     arrayminus=[b[1] - b[2] for b in betas]),
        marker=dict(size=9, color="#c0392b")), row=1, col=1)
    fig.add_hline(y=0, line=dict(color="#7f8c8d", width=1), row=1, col=1)
    fig.add_trace(go.Scatter(x=g_j["año"], y=g_j["gini"], name="Gini interestatal PIB pc",
                             line=dict(color="#2c3e50")), row=1, col=2)
    fig.add_trace(go.Scatter(x=nac["año"], y=nac["tasa"], name="homicidios por 100k",
                             line=dict(color="#c0392b")), row=1, col=2, secondary_y=True)
    fig.update_xaxes(title_text="rezago k (años)", row=1, col=1)
    fig.update_layout(
        title="¿La desigualdad precede a la violencia? Dentro del estado no; entre "
              "estados, solo co-tendencia<br><sup>izq: β de Theil salarial IMSS "
              "rezagado (FE estado+año, IC95, errores por estado) · der: series "
              "nacionales — el Gini interestatal alcanza su pico ~3 años antes que la "
              "tasa</sup>",
        legend=dict(orientation="h", y=-0.2))
    guardar_fig(fig, "cap7_desigualdad_dinamica", ancho=1100)


if __name__ == "__main__":
    main()
