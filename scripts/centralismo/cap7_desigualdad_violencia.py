"""
Cap 7 — Síntesis: desigualdad de riqueza y geografía de la violencia.

Testea el párrafo-resumen de la hipótesis: "gran desigualdad de la riqueza en los
estados como violencia generalizada, pero que se concentra en ciertos puntos".

  A. Desigualdad: Gini interestatal del PIB pc (2003 vs 2024, sin petróleo).
  B. Violencia: tasa de homicidio doloso por estado (SESNSP 2024) y concentración
     municipal (share del top-50 de municipios); desaparecidos per cápita.
  C. Cruce: ¿la violencia sigue a la pobreza/potencial, o a otra geografía?

Figuras → centralismo/informe/figuras/cap7_*.png
Run: uv run python scripts/centralismo/cap7_desigualdad_violencia.py
"""

import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from comun import (CORTO, PIBE_MINERIA_PETROLERA, PIBE_TOTAL, RAIZ,
                   cargar_pobreza, cargar_poblacion, guardar_fig, leer_pibe)

MESES = ["Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio", "Julio",
         "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"]
AÑO_V = 2024


def gini(x, w=None):
    """Gini ponderado (área bajo la curva de Lorenz por trapecios)."""
    x = np.asarray(x, dtype=float)
    w = np.ones_like(x) if w is None else np.asarray(w, dtype=float)
    orden = np.argsort(x)
    x, w = x[orden], w[orden]
    F = np.concatenate([[0.0], np.cumsum(w) / w.sum()])
    L = np.concatenate([[0.0], np.cumsum(x * w) / np.sum(x * w)])
    return float(1 - np.sum((F[1:] - F[:-1]) * (L[1:] + L[:-1])))


def test_a():
    print("=== A. Desigualdad interestatal (PIB pc sin petróleo) ===")
    pib = leer_pibe(PIBE_TOTAL).rename({"valor": "pib"})
    petro = leer_pibe(PIBE_MINERIA_PETROLERA).rename({"valor": "petro"})
    pob = cargar_poblacion()
    p = (pib.join(petro, on=["cve_ent", "año"], how="left").join(pob, on=["cve_ent", "año"])
         .with_columns(((pl.col("pib") - pl.col("petro").fill_null(0)) * 1e6
                        / pl.col("pob_total")).alias("pc")))
    for a in (2003, 2013, 2024):
        d = p.filter(pl.col("año") == a)
        g = gini(d["pc"].to_numpy(), d["pob_total"].to_numpy())
        arriba = d.sort("pc", descending=True)
        ratio = arriba["pc"][0] / arriba["pc"][-1]
        print(f"  {a}: Gini poblacional={g:.3f} · máx/mín={ratio:.1f}× "
              f"({CORTO[arriba['cve_ent'][0]]} vs {CORTO[arriba['cve_ent'][-1]]})")
    return p


def test_b():
    print("\n=== B. Geografía de la violencia (SESNSP, homicidio doloso) ===")
    inc = (pl.scan_parquet(RAIZ / "data/incidencia_delictiva/incidencia_fuero_comun/"
                           "incidencia_delictiva_fuero_comun.parquet")
           .filter((pl.col("Año") == AÑO_V) & (pl.col("Subtipo de delito") == "Homicidio doloso"))
           .with_columns(pl.sum_horizontal(MESES).alias("casos")))
    por_edo = (inc.group_by("Clave_Ent").agg(pl.sum("casos")).collect()
               .rename({"Clave_Ent": "cve_ent"}).with_columns(pl.col("cve_ent").cast(pl.Int64)))
    pob = cargar_poblacion().filter(pl.col("año") == AÑO_V)
    tasa = (por_edo.join(pob, on="cve_ent")
            .with_columns((pl.col("casos") / pl.col("pob_total") * 1e5).alias("tasa"))
            .with_columns(pl.col("cve_ent").replace_strict(CORTO).alias("estado"))
            .sort("tasa", descending=True))
    nac = por_edo["casos"].sum() / pob["pob_total"].sum() * 1e5
    print(f"  tasa nacional {AÑO_V}: {nac:.1f} por 100k")
    print("  top-5: " + " · ".join(f"{r['estado']}={r['tasa']:.0f}" for r in tasa.head(5).iter_rows(named=True)))
    print("  bottom-3: " + " · ".join(f"{r['estado']}={r['tasa']:.0f}" for r in tasa.tail(3).iter_rows(named=True)))

    por_mun = (inc.group_by("Cve. Municipio", "Municipio").agg(pl.sum("casos")).collect()
               .sort("casos", descending=True))
    tot = por_mun["casos"].sum()
    top50 = por_mun.head(50)["casos"].sum()
    print(f"  concentración municipal: top-50 municipios (de {por_mun.height:,}) = "
          f"{top50/tot*100:.1f}% de los homicidios")

    des = (pl.read_csv(RAIZ / "data/datamx/desaparecidos/desaparecidos.csv",
                       infer_schema_length=0)
           .with_columns(pl.col("CVE_ENT").cast(pl.Int64, strict=False).alias("cve_ent"))
           .filter(pl.col("cve_ent").is_between(1, 32))
           .group_by("cve_ent").agg(pl.len().alias("desaparecidos"))
           .join(pob, on="cve_ent")
           .with_columns((pl.col("desaparecidos") / pl.col("pob_total") * 1e5).alias("des_tasa")))
    d5 = des.sort("des_tasa", descending=True).with_columns(
        pl.col("cve_ent").replace_strict(CORTO).alias("estado")).head(5)
    print("  desaparecidos acumulados por 100k, top-5: " +
          " · ".join(f"{r['estado']}={r['des_tasa']:.0f}" for r in d5.iter_rows(named=True)))
    return tasa, des


def test_c(p, tasa, des):
    print("\n=== C. ¿A qué geografía sigue la violencia? ===")
    dep22 = cargar_pobreza().filter(pl.col("año") == 2022).select("cve_ent", "pct_pobreza")
    base = (tasa.select("cve_ent", "estado", "tasa")
            .join(p.filter(pl.col("año") == 2024).select("cve_ent", "pc"), on="cve_ent")
            .join(dep22, on="cve_ent")
            .join(des.select("cve_ent", "des_tasa"), on="cve_ent"))
    c_pob = np.corrcoef(base["tasa"], base["pct_pobreza"])[0, 1]
    c_pib = np.corrcoef(base["tasa"], base["pc"])[0, 1]
    c_des = np.corrcoef(base["tasa"], base["des_tasa"])[0, 1]
    print(f"  corr(homicidios, pobreza 2022)  = {c_pob:+.2f}")
    print(f"  corr(homicidios, PIB pc)        = {c_pib:+.2f}")
    print(f"  corr(homicidios, desaparecidos) = {c_des:+.2f}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=base["pct_pobreza"], y=base["tasa"], mode="markers+text", text=base["estado"],
        textposition="top center", textfont=dict(size=9),
        marker=dict(size=9, color="#c0392b"), showlegend=False))
    fig.update_layout(
        title=f"La violencia no sigue a la pobreza (corr={c_pob:+.2f})<br><sup>homicidio doloso por 100k ({AÑO_V}) vs pobreza CONEVAL 2022</sup>",
        xaxis_title="% pobreza 2022", yaxis_title=f"homicidios dolosos por 100k, {AÑO_V}",
    )
    guardar_fig(fig, "cap7_violencia_vs_pobreza")

    b = tasa.sort("tasa", descending=True)
    fig = go.Figure(go.Bar(x=b["tasa"], y=b["estado"], orientation="h", marker_color="#c0392b"))
    fig.update_layout(
        title=f"Tasa de homicidio doloso por estado, {AÑO_V} (SESNSP)",
        xaxis_title="homicidios por 100 mil habitantes",
        yaxis=dict(tickfont=dict(size=10), autorange="reversed"),
    )
    guardar_fig(fig, "cap7_tasa_homicidio", alto=800)


def main():
    p = test_a()
    tasa, des = test_b()
    test_c(p, tasa, des)


if __name__ == "__main__":
    main()
