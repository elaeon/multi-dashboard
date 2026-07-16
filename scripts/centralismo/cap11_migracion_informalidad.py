"""
Cap 11 — La válvula migratoria: migración interna × informalidad vs las hipótesis.

El dashboard internal_migration_flow.py muestra alta correlación entre migración neta
e informalidad estatal. Este capítulo la cuantifica con la metodología del informe y la
confronta con las hipótesis:

  Migración neta(t) = Pob(t+1) − Pob(t) − Nacimientos(t) + Defunciones(t)  [residual;
  mezcla flujo interno e internacional]. Años limpios 2017-2019 y 2022-2024 — se excluyen
  2020-2021 por el doble artefacto (caída de registros de nacimiento −35% por cierre de
  oficinas + sobremortalidad COVID +27/+57%), documentado en el dashboard.

Correlatos de la tasa migratoria acumulada (‰):
  a) informalidad ENOE (indicadores_laborales, agregado estatal, años limpios)
  b) formalización IMSS 2024 (puestos/100 hab, Cap 2)
  c) índice de potencial competitivo (PC1, Cap 5) — ¿predice flujos aunque no predijo pc?
  d) dependencia fiscal (Cap 4)
  e) razón retorno/aporte (Cap 1) — ¿las transferencias retienen población?

Figuras → centralismo/informe/figuras/cap11_*.png
Run: uv run python scripts/centralismo/cap11_migracion_informalidad.py
"""

import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import polars as pl
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from comun import CORTO, PIBE_MINERIA_PETROLERA, PIBE_TOTAL, RAIZ, cargar_poblacion, guardar_fig, leer_pibe

AÑOS_LIMPIOS = [2017, 2018, 2019, 2022, 2023, 2024]


def migracion_neta() -> pl.DataFrame:
    """Tasa neta acumulada (‰) por estado en los años limpios."""
    nacdef = pl.concat([
        pl.read_csv(RAIZ / f"data/inegi/nacimientos_descesos/{a}.csv")
        for a in AÑOS_LIMPIOS
    ])
    nacdef = (nacdef.filter(pl.col("mun_resid") != 999)
              .group_by("ent_resid", "anio")
              .agg(pl.sum("total_nac").alias("nac"), pl.sum("total_des").alias("des"))
              .rename({"ent_resid": "cve_ent", "anio": "año"}))

    pob = cargar_poblacion()
    pob1 = pob.select("cve_ent", (pl.col("año") - 1).alias("año"), pl.col("pob_total").alias("pob_t1"))
    base = (pob.join(pob1, on=["cve_ent", "año"])
            .join(nacdef, on=["cve_ent", "año"])
            .with_columns((pl.col("pob_t1") - pl.col("pob_total") - pl.col("nac")
                           + pl.col("des")).alias("net_mig")))
    out = (base.group_by("cve_ent")
           .agg(pl.sum("net_mig"), pl.mean("pob_total").alias("pob"))
           .with_columns((pl.col("net_mig") / pl.col("pob") * 1000).alias("mig_rate"),
                         pl.col("cve_ent").replace_strict(CORTO).alias("estado")))
    assert out.height == 32
    return out


def correlatos() -> pl.DataFrame:
    """Reconstruye las medidas de los caps 1/2/4/5 (baratas) a nivel estado."""
    # informalidad ENOE (años limpios disponibles en el panel: 2017-2024)
    lab = (pl.scan_parquet(RAIZ / "data/inegi/indicadores_laborales/indicadores_laborales.parquet")
           .filter((pl.col("cve_mun") == 0) & pl.col("cve_ent").is_between(1, 32)
                   & pl.col("año").is_in(AÑOS_LIMPIOS))
           .group_by("cve_ent").agg(pl.mean("informales").alias("informalidad"))
           .collect())

    # formalización IMSS 2024
    pob = cargar_poblacion()
    imss = (pl.scan_csv(RAIZ / "data/datamx/empleo_formal/empleo_formal.csv")
            .filter(pl.col("PERIODO").str.starts_with("2024"))
            .group_by("CVE_ENT", "PERIODO").agg(pl.sum("TOTAL"))
            .group_by("CVE_ENT").agg(pl.mean("TOTAL").alias("puestos"))
            .collect().rename({"CVE_ENT": "cve_ent"})
            .join(pob.filter(pl.col("año") == 2024), on="cve_ent")
            .with_columns((pl.col("puestos") / pl.col("pob_total") * 100).alias("formalizacion")))

    # dependencia fiscal (Cap 4) y razón retorno/aporte (Cap 1), 2022
    cp = pl.read_parquet(RAIZ / "informe_data/cp_estado_ramo.parquet")
    tr = (cp.filter(pl.col("id_ramo").is_in([28, 33]) & pl.col("cve_ent").is_between(1, 32)
                    & pl.col("ciclo").is_between(2017, 2024))
          .group_by("cve_ent").agg(pl.sum("monto_ejercido").alias("transfer")))
    propios = None
    for archivo, monto in [("mapa_impuestos_estatales.csv", "MONTO_IMPUESTOS"),
                           ("mapa_derechos_estatales.csv", "MONTO_DERECHOS")]:
        d = (pl.read_csv(RAIZ / "data/presupuesto_federacion/recaudacion_local" / archivo,
                         encoding="latin-1", infer_schema_length=0)
             .filter(pl.col("CICLO").cast(pl.Int64).is_between(2017, 2024))
             .select(pl.col("ID_ENTIDAD_FEDERATIVA").cast(pl.Int64).alias("cve_ent"),
                     pl.col(monto).cast(pl.Float64).alias("m"))
             .group_by("cve_ent").agg(pl.sum("m")))
        propios = d if propios is None else propios.join(d, on="cve_ent").with_columns(
            (pl.col("m") + pl.col("m_right")).alias("m")).drop("m_right")
    fiscal = tr.join(propios.rename({"m": "propios"}), on="cve_ent").with_columns(
        (pl.col("transfer") / (pl.col("transfer") + pl.col("propios"))).alias("dependencia"))

    pib = leer_pibe(PIBE_TOTAL, bloque="Millones de pesos").rename({"valor": "pib"})
    petro = leer_pibe(PIBE_MINERIA_PETROLERA, bloque="Millones de pesos").rename({"valor": "petro"})
    razon = (pib.filter(pl.col("año") == 2024)
             .join(petro.filter(pl.col("año") == 2024), on=["cve_ent", "año"], how="left")
             .with_columns((pl.col("pib") - pl.col("petro").fill_null(0)).alias("pib_sp"))
             .join(cp.filter((pl.col("ciclo") == 2024) & pl.col("id_ramo").is_in([28, 33])
                             & pl.col("cve_ent").is_between(1, 32))
                   .group_by("cve_ent").agg(pl.sum("monto_ejercido").alias("t24")), on="cve_ent")
             .with_columns(((pl.col("t24") / pl.col("t24").sum())
                            / (pl.col("pib_sp") / pl.col("pib_sp").sum())).alias("razon")))

    # índice de potencial (Cap 5) — se reconstruye vía import del módulo
    from cap5_factor_competencia import componentes
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    comp = componentes()
    cols = ["formalizacion", "salario_formal", "salud", "educacion", "investigacion"]
    X = StandardScaler().fit_transform(comp.select(cols).to_numpy())
    pc1 = PCA(n_components=1).fit_transform(X)[:, 0]
    if np.corrcoef(pc1, comp["formalizacion"])[0, 1] < 0:
        pc1 = -pc1
    indice = comp.select("cve_ent").with_columns(pl.Series("indice", pc1))

    return (lab.join(imss.select("cve_ent", "formalizacion"), on="cve_ent")
            .join(fiscal.select("cve_ent", "dependencia"), on="cve_ent")
            .join(razon.select("cve_ent", "razon"), on="cve_ent")
            .join(indice, on="cve_ent"))


def main():
    mig = migracion_neta()
    rec = mig.filter(pl.col("mig_rate") > 0).height
    top = mig.sort("mig_rate", descending=True)
    print(f"=== Migración neta residual, años limpios {AÑOS_LIMPIOS} ===")
    print(f"  receptores: {rec} · expulsores: {32 - rec}")
    print("  top receptores: " + " · ".join(f"{r['estado']}={r['mig_rate']:+.0f}‰"
                                            for r in top.head(4).iter_rows(named=True)))
    print("  top expulsores: " + " · ".join(f"{r['estado']}={r['mig_rate']:+.0f}‰"
                                            for r in top.tail(4).iter_rows(named=True)))

    base = mig.join(correlatos(), on="cve_ent")
    assert base.height == 32

    nombres = {"informalidad": "informalidad ENOE (media 17-24)",
               "formalizacion": "formalización IMSS 2024",
               "indice": "índice de potencial (Cap 5)",
               "dependencia": "dependencia fiscal (Cap 4)",
               "razon": "razón retorno/aporte (Cap 1)"}
    print("\n=== Correlatos de la tasa migratoria ===")
    corrs = {}
    for c, n in nombres.items():
        r = np.corrcoef(base["mig_rate"], base[c])[0, 1]
        corrs[c] = r
        print(f"  corr(mig_rate, {n}) = {r:+.2f}")

    X = sm.add_constant(np.column_stack([base["informalidad"].to_numpy(),
                                         np.log(base["pob"].to_numpy())]))
    m = sm.OLS(base["mig_rate"].to_numpy(), X).fit(cov_type="HC3")
    print(f"\n  OLS mig_rate ~ informalidad + ln(pob): β_informalidad={m.params[1]:+.2f}‰ por pp "
          f"(p={m.pvalues[1]:.4f}), R²={m.rsquared:.2f}")

    # ---- F1: scatter cuadrantes informalidad × migración
    med = float(base["informalidad"].median())
    colores = ["#c0392b" if (x < 0 and y >= med) else "#27ae60" if (x >= 0 and y < med)
               else "#e67e22" if x >= 0 else "#7f8c8d"
               for x, y in zip(base["mig_rate"], base["informalidad"])]
    fig = go.Figure()
    fig.add_vline(x=0, line_color="black")
    fig.add_hline(y=med, line_dash="dot", line_color="gray")
    fig.add_trace(go.Scatter(
        x=base["mig_rate"], y=base["informalidad"], mode="markers+text",
        text=base["estado"], textposition="top center", textfont=dict(size=9),
        marker=dict(size=9, color=colores), showlegend=False))
    fig.update_layout(
        title=f"La válvula migratoria: se sale de la informalidad, se llega a la formalidad (corr={corrs['informalidad']:+.2f})<br><sup>tasa neta de migración residual acumulada (‰, años limpios 2017-19/2022-24) vs % de empleo informal (ENOE)</sup>",
        xaxis_title="migración neta acumulada (‰ de la población)",
        yaxis_title="% de ocupados en informalidad",
    )
    guardar_fig(fig, "cap11_migracion_informalidad")

    # ---- F2: barra de correlatos
    orden = ["informalidad", "formalizacion", "indice", "dependencia", "razon"]
    fig = go.Figure(go.Bar(
        x=[corrs[c] for c in orden], y=[nombres[c] for c in orden], orientation="h",
        marker_color=["#c0392b" if corrs[c] < 0 else "#27ae60" for c in orden]))
    fig.add_vline(x=0, line_color="black")
    fig.update_layout(
        title="¿Qué predice hacia dónde se mueve la gente? Correlatos de la migración neta estatal<br><sup>n=32; la migración fluye hacia el potencial/formalidad y abandona la dependencia — las transferencias no retienen</sup>",
        xaxis_title="correlación con la tasa neta de migración",
    )
    guardar_fig(fig, "cap11_correlatos")


if __name__ == "__main__":
    main()
