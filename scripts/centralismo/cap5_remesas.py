"""
Cap 5 — Extensión: remesas per cápita, el factor de crecimiento omitido (cierra H5).

El modelo del Cap 5 fue nulo (R²=0.03): ni el índice de potencial, ni la dependencia,
ni la distancia a CDMX explican el crecimiento 2013-2024. El caveat del capítulo intuía
la variable omitida: los expulsores con remesas (Michoacán) crecen per cápita. Se testea:

  A. Remesas pc por estado (Banxico CA79 2003-2025, USD/hab): niveles y medidas
     ex-ante (2018, línea base del índice) y de ventana (promedio 2013-2024).
  B. ¿Componente del índice? corr con el índice de potencial y PCA con 6º componente
     (la lectura literal de la fila pendiente de la nota al pie).
  C. La regresión extendida: M0 (réplica del Cap 5) + remesas pc → β, p, ΔR²;
     test de residuales; robustez sin Chiapas (quiebre post-2020) y con ambas medidas.

Figura → centralismo/informe/figuras/cap5_remesas_crecimiento.png
Run: uv run python scripts/centralismo/cap5_remesas.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import polars as pl
import statsmodels.api as sm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from comun import CORTO, RAIZ, cargar_poblacion, guardar_fig, normalizar_estado
from cap5_factor_competencia import (A0, A1, componentes, crecimiento_pib,
                                     dependencia_fiscal, distancia_cdmx)

CDMX, CHIAPAS = 9, 7
COLS5 = ["formalizacion", "salario_formal", "salud", "educacion", "investigacion"]


def cargar_remesas_pc() -> pl.DataFrame:
    """Banxico CA79 → remesas anuales por estado 2003-2025 en USD por habitante."""
    df = pd.read_csv(RAIZ / "data/banxico/remesas/Consulta_20260615-174730348.csv",
                     encoding="latin-1", skiprows=10, nrows=33)
    # ojo: la fila TOTAL también trae sufijo (", TOTAL") — se excluye por código de serie
    df = df[df["Serie"].str.strip() != "SE29702"].copy()
    pref = "Ingresos por Remesas Familiares, "
    df["cve_ent"] = [normalizar_estado(t[len(pref):], estricto=True) for t in df["Título"]]
    assert df["cve_ent"].notna().all() and df["cve_ent"].nunique() == 32
    fechas = [c for c in df.columns if "/" in c]
    largo = (pl.from_pandas(df[["cve_ent"] + fechas])
             .unpivot(index="cve_ent", variable_name="fecha", value_name="musd")
             .with_columns(pl.col("fecha").str.slice(6, 4).cast(pl.Int64).alias("año")))
    anual = (largo.filter(pl.col("año") <= 2025)  # 2026 solo trae Q1
             .group_by("cve_ent", "año").agg(pl.sum("musd")))
    assert anual.filter(pl.col("año") == 2018).height == 32
    return (anual.join(cargar_poblacion(), on=["cve_ent", "año"])
            .with_columns((pl.col("musd") * 1e6 / pl.col("pob_total")).alias("rem_pc")))


def indice_pca(comp: pl.DataFrame, cols: list[str]):
    """Réplica exacta de la construcción del índice del Cap 5 (PC1, signo fijado)."""
    X = StandardScaler().fit_transform(comp.select(cols).to_numpy())
    pca = PCA(n_components=2).fit(X)
    pc1 = pca.transform(X)[:, 0]
    if np.corrcoef(pc1, comp["formalizacion"])[0, 1] < 0:
        pc1 = -pc1
        pca.components_[0] = -pca.components_[0]
    return pc1, pca


def ols(y, cols_x, nombres):
    m = sm.OLS(y, sm.add_constant(np.column_stack(cols_x))).fit(cov_type="HC3")
    for n, b, p in zip(["const"] + nombres, m.params, m.pvalues):
        print(f"    {n:<28} β={b:+.3f}  p={p:.3f}")
    print(f"    R²={m.rsquared:.2f}  n={int(m.nobs)}")
    return m


def main():
    rem = cargar_remesas_pc()
    r18 = (rem.filter(pl.col("año") == 2018)
           .select("cve_ent", pl.col("rem_pc").alias("rem18")))
    rwin = (rem.filter(pl.col("año").is_between(A0, A1))
            .group_by("cve_ent").agg(pl.mean("rem_pc").alias("rem_win")))

    print("=== A. Remesas per cápita (Banxico CA79, USD/hab nominales) ===")
    top = r18.sort("rem18", descending=True).with_columns(
        pl.col("cve_ent").replace_strict(CORTO).alias("estado"))
    print("  top-4 2018:", ", ".join(f"{r['estado']}={r['rem18']:.0f}"
                                     for r in top.head(4).iter_rows(named=True)))
    print("  bottom-3 2018:", ", ".join(f"{r['estado']}={r['rem18']:.0f}"
                                        for r in top.tail(3).iter_rows(named=True)))
    fc = top.with_row_index("rank").filter(pl.col("cve_ent") == CDMX)
    print(f"  CDMX 2018: {fc['rem18'][0]:.0f} USD/hab (lugar {fc['rank'][0] + 1}/32)")
    sh = (rem.with_columns((pl.col("musd") / pl.col("musd").sum().over("año")).alias("s"))
          .filter(pl.col("cve_ent") == CDMX))
    print(f"  caveat CDMX: su share nacional pasa de "
          f"{sh.filter(pl.col('año') == 2019)['s'][0] * 100:.1f}% (2019) a "
          f"{sh.filter(pl.col('año') == 2025)['s'][0] * 100:.1f}% (2025) — posible "
          f"artefacto de intermediación bancaria (destino del registro, no del hogar)")

    comp = componentes()
    pc1, _ = indice_pca(comp, COLS5)
    base = (comp.with_columns(pl.Series("indice", pc1))
            .join(r18, on="cve_ent").join(rwin, on="cve_ent")
            .join(crecimiento_pib().select("cve_ent", "crec"), on="cve_ent")
            .join(dependencia_fiscal(), on="cve_ent")
            .join(distancia_cdmx().select("cve_ent", "dist_cdmx"), on="cve_ent")
            .with_columns(pl.col("cve_ent").replace_strict(CORTO).alias("estado")))

    print("\n=== B. ¿Componente del índice? ===")
    print(f"  corr(remesas pc 2018, índice de potencial) = "
          f"{np.corrcoef(base['rem18'], base['indice'])[0, 1]:+.2f}")
    pc1_6, pca6 = indice_pca(base, COLS5 + ["rem18"])
    print(f"  PCA con remesas como 6º componente: varianza PC1 = "
          f"{pca6.explained_variance_ratio_[0] * 100:.0f}% (antes 54%)")
    print("  cargas PC1:", {c: f"{v:+.2f}"
                            for c, v in zip(COLS5 + ["rem18"], pca6.components_[0])})
    print(f"  corr(índice de 5, índice de 6) = "
          f"{np.corrcoef(base['indice'].to_numpy(), pc1_6)[0, 1]:+.2f}")

    print(f"\n=== C. Regresión extendida: crecimiento {A0}→{A1} ===")
    y = base["crec"].to_numpy()
    x_ind = base["indice"].to_numpy()
    x_dep = base["dependencia"].to_numpy() * 100
    x_dis = base["dist_cdmx"].to_numpy() / 100
    print("  M0 — réplica Cap 5 (índice + dependencia + distancia):")
    m0 = ols(y, [x_ind, x_dep, x_dis],
             ["índice potencial", "dependencia fiscal (pp)", "distancia CDMX (100 km)"])
    for var in ("rem18", "rem_win"):
        print(f"  M1 — M0 + remesas pc ({var}, por 100 USD/hab):")
        ols(y, [x_ind, x_dep, x_dis, base[var].to_numpy() / 100],
            ["índice potencial", "dependencia fiscal (pp)", "distancia CDMX (100 km)",
             f"remesas pc ({var})"])
    print("  robustez sin Chiapas (quiebre post-2020 del flujo migrante, Cap 9):")
    sc = base.filter(pl.col("cve_ent") != CHIAPAS)
    ols(sc["crec"].to_numpy(),
        [sc["indice"].to_numpy(), sc["dependencia"].to_numpy() * 100,
         sc["dist_cdmx"].to_numpy() / 100, sc["rem_win"].to_numpy() / 100],
        ["índice potencial", "dependencia fiscal (pp)", "distancia CDMX (100 km)",
         "remesas pc (rem_win)"])

    print("  bivariadas: corr(rem18, crec) = "
          f"{np.corrcoef(base['rem18'], y)[0, 1]:+.2f} · corr(rem_win, crec) = "
          f"{np.corrcoef(base['rem_win'], y)[0, 1]:+.2f}")
    resid = y - m0.predict(sm.add_constant(np.column_stack([x_ind, x_dep, x_dis])))
    print(f"  test de residuales: corr(residual M0, rem_win) = "
          f"{np.corrcoef(resid, base['rem_win'])[0, 1]:+.2f}")
    rk = base.with_columns(pl.Series("resid", resid)).with_columns(
        pl.col("rem_win").rank(descending=True).alias("rank_rem"))
    print("  5 residuales más positivos de M0 (crecen más de lo que el modelo predice):")
    for r in rk.sort("resid", descending=True).head(5).iter_rows(named=True):
        print(f"    {r['estado']:<12} resid={r['resid']:+.2f} · remesas pc "
              f"rank {int(r['rank_rem'])}/32 ({r['rem_win']:.0f} USD/hab)")
    print("  caveats: remesas nominales USD (válido transversal, no como nivel real); "
          "el PIB no registra la remesa en sí — el mecanismo es consumo → producción "
          "local de servicios; n=32, correlacional")

    # figura: remesas pc vs crecimiento, color = índice
    b, a = np.polyfit(base["rem_win"].to_numpy(), y, 1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=base["rem_win"], y=base["crec"], mode="markers+text", text=base["estado"],
        textposition="top center", textfont=dict(size=9), showlegend=False,
        marker=dict(size=10, color=base["indice"], colorscale="RdYlGn",
                    colorbar=dict(title="índice de<br>potencial"))))
    xs = np.array([float(base["rem_win"].min()), float(base["rem_win"].max())])
    fig.add_trace(go.Scatter(x=xs, y=a + b * xs, mode="lines", showlegend=False,
                             line=dict(color="#7f8c8d", dash="dash", width=1)))
    fig.update_layout(
        title=f"El factor que el modelo del Cap 5 no veía: las remesas<br>"
              f"<sup>remesas per cápita (promedio anual {A0}-{A1}, Banxico CA79) vs "
              f"crecimiento del PIB pc sin petróleo · color = índice de potencial "
              f"(línea base 2018)</sup>",
        xaxis_title=f"remesas por habitante (USD, promedio {A0}-{A1})",
        yaxis_title="crecimiento anual promedio del PIB pc (%)")
    guardar_fig(fig, "cap5_remesas_crecimiento")


if __name__ == "__main__":
    main()
