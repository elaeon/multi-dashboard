"""
Cap 5 — Factor de crecimiento y competencia de los estados (H5).

Construye un índice de potencial competitivo por estado (PCA sobre 5 componentes
medidos al inicio del periodo) y estima:

  crecimiento PIB pc real sin petróleo (2013→2024) ~ índice + dependencia fiscal
                                                     + distancia a CDMX

Componentes del índice (z-scores, 32 estados):
  - formalización 2018 (puestos IMSS / 100 hab)
  - salario formal medio 2018 (masa salarial / puesto, IMSS)
  - acceso a salud 2018 (100 − carencia CONEVAL)
  - capital educativo 2018 (100 − rezago educativo CONEVAL)
  - investigación (SNII por 100 mil hab, padrón 2026 — variable lenta; ver caveat)

H5 pregunta si el crecimiento está "acotado por la CDMX": el coeficiente de la
distancia a CDMX responde directamente (negativo = crecer lejos del centro es más difícil;
positivo = la cercanía al centro FRENA).

Figuras → centralismo/informe/figuras/cap5_*.png
Run: uv run python scripts/centralismo/cap5_factor_competencia.py
"""

import json
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
from comun import (CORTO, PIBE_MINERIA_PETROLERA, PIBE_TOTAL, RAIZ,
                   cargar_pobreza, cargar_poblacion, guardar_fig, leer_pibe,
                   normalizar_estado)

A0, A1 = 2013, 2024


def crecimiento_pib() -> pl.DataFrame:
    pib = leer_pibe(PIBE_TOTAL).rename({"valor": "pib"})
    petro = leer_pibe(PIBE_MINERIA_PETROLERA).rename({"valor": "petro"})
    pob = cargar_poblacion()
    p = (pib.join(petro, on=["cve_ent", "año"], how="left")
         .join(pob, on=["cve_ent", "año"])
         .with_columns(((pl.col("pib") - pl.col("petro").fill_null(0)) * 1e6
                        / pl.col("pob_total")).alias("pc")))
    w = (p.filter(pl.col("año").is_in([A0, A1]))
         .pivot(on="año", index="cve_ent", values="pc"))
    return w.with_columns(
        ((pl.col(str(A1)).log() - pl.col(str(A0)).log()) / (A1 - A0) * 100).alias("crec"))


def componentes() -> pl.DataFrame:
    pob = cargar_poblacion()
    dep = cargar_pobreza()

    imss = (
        pl.scan_csv(RAIZ / "data/datamx/empleo_formal/empleo_formal.csv")
        .filter(pl.col("PERIODO").str.starts_with("2018"))
        .group_by("CVE_ENT", "PERIODO")
        .agg(pl.sum("TOTAL"), pl.sum("MASA_SALARIAL_TOTAL"))
        .group_by("CVE_ENT")
        .agg(pl.mean("TOTAL").alias("puestos"), pl.mean("MASA_SALARIAL_TOTAL").alias("masa"))
        .collect()
        .rename({"CVE_ENT": "cve_ent"})
    )
    base = (imss.join(pob.filter(pl.col("año") == 2018), on="cve_ent")
            .with_columns(
                (pl.col("puestos") / pl.col("pob_total") * 100).alias("formalizacion"),
                (pl.col("masa") / pl.col("puestos")).alias("salario_formal"),
            ))

    d18 = dep.filter(pl.col("año") == 2018).select(
        "cve_ent",
        (100 - pl.col("pct_ic_asalud")).alias("salud"),
        (100 - pl.col("pct_ic_rezedu")).alias("educacion"),
    )

    snii = pd.read_excel(RAIZ / "data/secihti/snii/Padron-SNII-2026-1T.xlsx",
                         sheet_name=0, header=0)
    c_ent = [c for c in snii.columns if "ENTIDAD FINAL" in str(c).upper()][0]
    snii["cve_ent"] = snii[c_ent].map(normalizar_estado)
    cuenta = (pl.from_pandas(snii.dropna(subset=["cve_ent"])[["cve_ent"]])
              .with_columns(pl.col("cve_ent").cast(pl.Int64))
              .group_by("cve_ent").agg(pl.len().alias("snii")))
    inv = (cuenta.join(pob.filter(pl.col("año") == 2024), on="cve_ent")
           .with_columns((pl.col("snii") / pl.col("pob_total") * 1e5).alias("investigacion")))

    return (base.select("cve_ent", "formalizacion", "salario_formal")
            .join(d18, on="cve_ent")
            .join(inv.select("cve_ent", "investigacion"), on="cve_ent"))


def dependencia_fiscal() -> pl.DataFrame:
    cp = pl.read_parquet(RAIZ / "data/presupuesto_federacion/cuenta_publica/cp_estado_ramo.parquet")
    tr = (cp.filter(pl.col("id_ramo").is_in([28, 33]) & pl.col("cve_ent").is_between(1, 32))
          .group_by("ciclo", "cve_ent").agg(pl.sum("monto_ejercido").alias("transfer"))
          .rename({"ciclo": "año"}))
    propios = None
    for archivo, monto in [("mapa_impuestos_estatales.csv", "MONTO_IMPUESTOS"),
                           ("mapa_derechos_estatales.csv", "MONTO_DERECHOS")]:
        df = (pl.read_csv(RAIZ / "data/presupuesto_federacion/recaudacion_local" / archivo,
                          encoding="latin-1", infer_schema_length=0)
              .select(pl.col("CICLO").cast(pl.Int64).alias("año"),
                      pl.col("ID_ENTIDAD_FEDERATIVA").cast(pl.Int64).alias("cve_ent"),
                      pl.col(monto).cast(pl.Float64).alias("m")))
        propios = df if propios is None else propios.join(
            df, on=["año", "cve_ent"], how="full", coalesce=True).with_columns(
            (pl.col("m").fill_null(0) + pl.col("m_right").fill_null(0)).alias("m")).drop("m_right")
    return (tr.join(propios.rename({"m": "propios"}), on=["año", "cve_ent"])
            .filter(pl.col("año").is_between(A0, A1))
            .group_by("cve_ent")
            .agg((pl.sum("transfer") / (pl.sum("transfer") + pl.sum("propios")))
                 .alias("dependencia")))


def distancia_cdmx() -> pl.DataFrame:
    g = json.load(open(RAIZ / "data/mexico_states.geojson"))

    def centroide(geom):
        pts = []

        def rec(c):
            if isinstance(c[0], (int, float)):
                pts.append(c)
            else:
                for x in c:
                    rec(x)
        rec(geom["coordinates"])
        a = np.array(pts)
        return a[:, 0].mean(), a[:, 1].mean()

    filas = []
    for f in g["features"]:
        cve = normalizar_estado(f["properties"]["name"])
        if cve:
            lon, lat = centroide(f["geometry"])
            filas.append((cve, lon, lat))
    df = pl.DataFrame(filas, schema=["cve_ent", "lon", "lat"], orient="row")
    cd = df.filter(pl.col("cve_ent") == 9).row(0, named=True)

    def hav(lon1, lat1, lon2, lat2):
        rl1, rl2 = np.radians(lat1), np.radians(lat2)
        dlat, dlon = rl2 - rl1, np.radians(lon2 - lon1)
        a = np.sin(dlat / 2) ** 2 + np.cos(rl1) * np.cos(rl2) * np.sin(dlon / 2) ** 2
        return 6371 * 2 * np.arcsin(np.sqrt(a))

    return df.with_columns(
        pl.struct("lon", "lat").map_elements(
            lambda s: hav(s["lon"], s["lat"], cd["lon"], cd["lat"]), return_dtype=pl.Float64,
        ).alias("dist_cdmx"))


def main():
    comp = componentes()
    assert comp.height == 32, comp.height
    cols = ["formalizacion", "salario_formal", "salud", "educacion", "investigacion"]

    X = StandardScaler().fit_transform(comp.select(cols).to_numpy())
    pca = PCA(n_components=2).fit(X)
    pc1 = pca.transform(X)[:, 0]
    if np.corrcoef(pc1, comp["formalizacion"])[0, 1] < 0:
        pc1 = -pc1
    print("=== Índice de potencial competitivo (PCA) ===")
    print(f"  varianza explicada PC1: {pca.explained_variance_ratio_[0]*100:.0f}% "
          f"(PC2: {pca.explained_variance_ratio_[1]*100:.0f}%)")
    print("  cargas PC1:", {c: f"{v:+.2f}" for c, v in zip(cols, pca.components_[0])})

    base = (comp.with_columns(pl.Series("indice", pc1))
            .join(crecimiento_pib().select("cve_ent", "crec"), on="cve_ent")
            .join(dependencia_fiscal(), on="cve_ent")
            .join(distancia_cdmx().select("cve_ent", "dist_cdmx"), on="cve_ent")
            .with_columns(pl.col("cve_ent").replace_strict(CORTO).alias("estado")))

    rank = base.sort("indice", descending=True)
    print("  top-5 índice:", ", ".join(f"{r['estado']}={r['indice']:+.2f}" for r in rank.head(5).iter_rows(named=True)))
    print("  bottom-5:", ", ".join(f"{r['estado']}={r['indice']:+.2f}" for r in rank.tail(5).iter_rows(named=True)))

    fig = go.Figure(go.Bar(
        x=rank["indice"], y=rank["estado"], orientation="h",
        marker=dict(color=rank["dependencia"], colorscale="RdYlGn_r",
                    colorbar=dict(title="dependencia<br>fiscal")),
    ))
    fig.update_layout(
        title="Índice de potencial competitivo estatal (PC1 de 5 componentes, línea base 2018)<br><sup>color: dependencia fiscal 2013-2024 — los estados de bajo potencial son los más dependientes</sup>",
        xaxis_title="índice (z)", yaxis=dict(tickfont=dict(size=10), autorange="reversed"),
    )
    guardar_fig(fig, "cap5_indice", alto=800)

    # ---- regresión
    print(f"\n=== OLS: crecimiento PIB pc sin petróleo {A0}→{A1} ~ índice + dependencia + dist_CDMX ===")
    Xr = sm.add_constant(np.column_stack([
        base["indice"].to_numpy(),
        base["dependencia"].to_numpy() * 100,
        base["dist_cdmx"].to_numpy() / 100,
    ]))
    m = sm.OLS(base["crec"].to_numpy(), Xr).fit(cov_type="HC3")
    nombres = ["const", "índice potencial", "dependencia fiscal (pp)", "distancia a CDMX (100 km)"]
    for n, b, p in zip(nombres, m.params, m.pvalues):
        print(f"  {n:<26} β={b:+.3f}  p={p:.3f}")
    print(f"  R²={m.rsquared:.2f}  n={int(m.nobs)}")

    # bivariadas de referencia
    for var, nom in [("indice", "índice"), ("dependencia", "dependencia"), ("dist_cdmx", "dist CDMX")]:
        c = np.corrcoef(base[var], base["crec"])[0, 1]
        print(f"  corr({nom}, crecimiento) = {c:+.2f}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=base["indice"], y=base["crec"], mode="markers+text", text=base["estado"],
        textposition="top center", textfont=dict(size=9),
        marker=dict(size=9, color=base["dependencia"], colorscale="RdYlGn_r",
                    colorbar=dict(title="dependencia")),
        showlegend=False))
    fig.add_hline(y=float(base["crec"].mean()), line_dash="dot", line_color="gray")
    fig.update_layout(
        title=f"Potencial competitivo (2018) vs crecimiento realizado ({A0}–{A1})<br><sup>PIB pc real sin petróleo; color = dependencia fiscal</sup>",
        xaxis_title="índice de potencial (PC1)", yaxis_title="crecimiento anual promedio (%)",
    )
    guardar_fig(fig, "cap5_indice_vs_crecimiento")


if __name__ == "__main__":
    main()
