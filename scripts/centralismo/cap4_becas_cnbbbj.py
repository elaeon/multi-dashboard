"""
Cap 4 (extensión) — Becas CNBBBJ: ¿beneficio medible o solo más círculo? (H4)

El Cap 4 midió el gasto social con el Ramo 20 (Bienestar); las becas Benito
Juárez (básica/media superior/superior) son Ramo 11 (SEP) y quedaron fuera.
Tres piezas con datos locales:

  A. Territorio y focalización — becas CNBBBJ dispersadas por estado 2019-2025
     (data/datos_gob/becas_CNBBBJ, grano localidad; 550.8B MXN acumulados):
     monto per cápita, composición por nivel educativo, corr con pobreza
     (comparable al +0.45 de los subsidios R20) y Gini estatal.
  B. Incidencia — ENIGH tabla `ingresos` por clave de fuente: en 2024 las becas
     CNBBBJ tienen claves propias (P101 básica, P102 media superior, P103
     superior/Jóvenes Escribiendo el Futuro) + P038 becas de gobierno genéricas;
     en 2018 el predecesor es P042 (PROSPERA) + P038. Peso en el ingreso
     corriente por decil nacional, 2018 vs 2024 — ¿el beneficio se movió hacia
     abajo? (La columna `becas` de concentradohogar NO sirve: mezcla privadas y
     excluye los programas Benito Juárez, pesa solo 0.17%.)
  C. Outcome educativo — ENIGH poblacion (`asis_esc`): tasa de asistencia
     escolar 15-17 (media superior, beca universal) y 18-22 (superior) por
     estado, 2018 vs 2024; test Δasistencia ~ becas_pc (análogo del test
     Bienestar→Δpobreza del Cap 4.A).

Trampas aplicadas: parquets becas en lazy (117.6M filas), BECA>0 obligatorio,
CVE_EDO cast a int (2025 viene zero-padded), montos nominales; ENIGH 2018 usa
`ubic_geo` y su tabla poblacion no trae factor/geografía (join a
concentradohogar por folioviv+foliohog); todo ponderado con `factor`.

Figuras → centralismo/informe/figuras/cap4_becas_*.png
Run: uv run python scripts/centralismo/cap4_becas_cnbbbj.py
"""

import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from comun import CORTO, RAIZ, cargar_pobreza, cargar_poblacion, guardar_fig

DIR_BECAS = RAIZ / "data/datos_gob/becas_CNBBBJ"
DIR_ENIGH = RAIZ / "data/inegi/enigh"
AÑOS_BECAS = list(range(2019, 2026))

ENIGH_ZIP = {2018: "conjunto_de_datos_enigh_2018_ns_csv.zip",
             2024: "conjunto_de_datos_enigh2024_ns_csv.zip"}


def gini(x: np.ndarray) -> float:
    x = np.sort(x)
    n = len(x)
    return float((2 * np.arange(1, n + 1) - n - 1) @ x / (n * x.sum()))


# --------------------------------------------------------------- A. becas por estado

def becas_estado_nivel() -> pl.DataFrame:
    partes = []
    for año in AÑOS_BECAS:
        lf = pl.scan_parquet(DIR_BECAS / f"becas_{año}.parquet")
        agg = (lf.filter(pl.col("BECA") > 0)
                 .with_columns(pl.col("CVE_EDO").cast(pl.Int64).alias("cve_ent"))
                 .group_by("cve_ent", "nivel_educativo")
                 .agg(pl.sum("BECA").alias("monto"))
                 .with_columns(pl.lit(año).alias("año"))
                 .collect())
        partes.append(agg)
    return pl.concat(partes)


def leer_enigh(ola: int, tabla: str) -> pd.DataFrame:
    with zipfile.ZipFile(DIR_ENIGH / ENIGH_ZIP[ola]) as zf:
        nombre = next(n for n in zf.namelist()
                      if f"conjunto_de_datos/conjunto_de_datos_{tabla}_enigh" in n)
        with zf.open(nombre) as f:
            return pd.read_csv(io.BytesIO(f.read()), low_memory=False)


def main():
    pob = cargar_poblacion()

    # ================================================================ A
    be = becas_estado_nivel()
    tot_año = be.group_by("año").agg(pl.sum("monto")).sort("año")
    t19 = tot_año.filter(pl.col("año") == 2019)["monto"][0] / 1e9
    t25 = tot_año.filter(pl.col("año") == 2025)["monto"][0] / 1e9
    assert 50 < t19 < 52 and 121 < t25 < 124, (t19, t25)  # cross-check DATA_OVERVIEW

    print("=== A. Becas CNBBBJ dispersadas (nominal) ===")
    for r in tot_año.iter_rows(named=True):
        print(f"  {r['año']}: {r['monto']/1e9:.1f} B MXN")

    niv = (be.group_by("año", "nivel_educativo").agg(pl.sum("monto"))
           .with_columns((pl.col("monto") / pl.col("monto").sum().over("año") * 100).alias("pct"))
           .sort("año", "nivel_educativo"))
    print("\n  composición por nivel (% del monto anual):")
    for año in (2019, 2022, 2025):
        fila = {r["nivel_educativo"]: r["pct"] for r in
                niv.filter(pl.col("año") == año).iter_rows(named=True)}
        print(f"  {año}: básica {fila.get('basica', 0):.0f}% · media_sup {fila.get('media_superior', 0):.0f}% "
              f"· superior {fila.get('superior', 0):.0f}%")

    # per cápita promedio 2022-2024 y focalización
    bpc = (be.filter(pl.col("año").is_between(2022, 2024))
           .group_by("cve_ent").agg((pl.sum("monto") / 3).alias("becas"))
           .join(pob.filter(pl.col("año") == 2023).select("cve_ent", "pob_total"), on="cve_ent")
           .with_columns((pl.col("becas") / pl.col("pob_total")).alias("becas_pc"),
                         pl.col("cve_ent").replace_strict(CORTO).alias("estado")))
    assert bpc.height == 32
    dep22 = cargar_pobreza().filter(pl.col("año") == 2022).select("cve_ent", "pct_pobreza")
    b2 = bpc.join(dep22, on="cve_ent")
    c_foc = np.corrcoef(b2["becas_pc"], b2["pct_pobreza"])[0, 1]
    g = gini(bpc["becas_pc"].to_numpy())
    print(f"\n  focalización: corr(becas_pc 22-24, pobreza 2022) = {c_foc:+.3f}  (subsidios R20: +0.45)")
    print(f"  Gini estatal de becas per cápita: {g:.3f}")

    # F1: scatter focalización
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=b2["pct_pobreza"], y=b2["becas_pc"], mode="markers+text", text=b2["estado"],
        textposition="top center", textfont=dict(size=9),
        marker=dict(size=8, color="#1f77b4"), showlegend=False))
    z = np.polyfit(b2["pct_pobreza"], b2["becas_pc"], 1)
    xs = np.linspace(float(b2["pct_pobreza"].min()), float(b2["pct_pobreza"].max()), 50)
    fig.add_trace(go.Scatter(x=xs, y=np.polyval(z, xs), mode="lines",
                             line=dict(color="#c0392b"), name="tendencia"))
    fig.update_layout(
        title=f"Becas CNBBBJ per cápita (prom. 2022-2024) vs pobreza estatal 2022<br>"
              f"<sup>corr = {c_foc:+.2f} (subsidios R20: +0.45) · Gini estatal {g:.2f}</sup>",
        xaxis_title="% población en pobreza (CONEVAL 2022)",
        yaxis_title="becas per cápita (MXN nominales/año)",
    )
    guardar_fig(fig, "cap4_becas_focalizacion")

    # F2: composición por nivel 2019-2025
    fig = go.Figure()
    for nivel, nombre in [("basica", "Básica"), ("media_superior", "Media superior"),
                          ("superior", "Superior")]:
        s = niv.filter(pl.col("nivel_educativo") == nivel).sort("año")
        fig.add_trace(go.Bar(x=s["año"], y=s["monto"] / 1e9, name=nombre))
    fig.update_layout(
        barmode="stack",
        title="Becas CNBBBJ por nivel educativo, 2019-2025 (nominal)<br>"
              "<sup>2024: caída de media superior · 2025: expansión de básica (Rita Cetina)</sup>",
        xaxis_title="año", yaxis_title="miles de millones MXN",
    )
    guardar_fig(fig, "cap4_becas_nivel")

    # ================================================================ B
    print("\n=== B. Incidencia del ingreso por becas de gobierno (ENIGH ingresos) ===")
    CLAVES = {2018: ["P038", "P042"],           # becas gobierno + PROSPERA
              2024: ["P038", "P101", "P102", "P103"]}  # + becas Benito Juárez por nivel
    deciles = {}
    for ola in (2018, 2024):
        ing = leer_enigh(ola, "ingresos")
        ing["ing_tri"] = pd.to_numeric(ing["ing_tri"], errors="coerce")
        beca_hog = (ing[ing["clave"].isin(CLAVES[ola])]
                    .groupby(["folioviv", "foliohog"])["ing_tri"].sum()
                    .rename("beca_gob").reset_index())
        ch = leer_enigh(ola, "concentradohogar")
        ch = ch.merge(beca_hog, on=["folioviv", "foliohog"], how="left")
        ch["beca_gob"] = ch["beca_gob"].fillna(0)
        ch = ch.sort_values("ing_cor")
        ch["cum_w"] = ch["factor"].cumsum() / ch["factor"].sum()
        ch["decil"] = np.minimum((ch["cum_w"] * 10).astype(int) + 1, 10)
        d = (ch.assign(becas_w=ch["beca_gob"] * ch["factor"], ing_w=ch["ing_cor"] * ch["factor"])
             .groupby("decil")[["becas_w", "ing_w"]].sum())
        d["share"] = d["becas_w"] / d["ing_w"] * 100
        deciles[ola] = d
        cobertura = ((ch["beca_gob"] > 0) * ch["factor"]).sum() / ch["factor"].sum() * 100
        peso_nal = d["becas_w"].sum() / d["ing_w"].sum() * 100
        share_d1_d3 = d.loc[1:3, "becas_w"].sum() / d["becas_w"].sum() * 100
        print(f"  {ola} (claves {'+'.join(CLAVES[ola])}): hogares receptores {cobertura:.1f}% · "
              f"peso nacional {peso_nal:.2f}% · D1-D3 capturan {share_d1_d3:.0f}% del monto")
        print(f"       share por decil (1=más pobre): " +
              " ".join(f"D{i}={d.loc[i, 'share']:.1f}%" for i in range(1, 11)))

    fig = go.Figure()
    for ola, color in [(2018, "#95a5a6"), (2024, "#1f77b4")]:
        d = deciles[ola]
        fig.add_trace(go.Bar(x=[f"D{i}" for i in d.index], y=d["share"], name=str(ola),
                             marker_color=color))
    fig.update_layout(
        barmode="group",
        title="Peso del ingreso por becas de gobierno en el ingreso del hogar, por decil<br>"
              "<sup>ENIGH: 2018 = P038+PROSPERA (P042) · 2024 = P038 + becas Benito Juárez (P101-P103) · deciles ponderados</sup>",
        xaxis_title="decil de ingreso (1 = más pobre)", yaxis_title="becas gob. / ing_cor (%)",
    )
    guardar_fig(fig, "cap4_becas_deciles")

    # ================================================================ C
    print("\n=== C. Outcome: asistencia escolar 2018 vs 2024 (ENIGH poblacion) ===")
    tasas = {}
    for ola in (2018, 2024):
        po = leer_enigh(ola, "poblacion")
        if "factor" not in po.columns:  # 2018: factor/geo viven en concentradohogar
            ch = leer_enigh(ola, "concentradohogar")
            col_geo = "ubica_geo" if "ubica_geo" in ch.columns else "ubic_geo"
            ch["cve_ent"] = ch[col_geo] // 1000
            po = po.merge(ch[["folioviv", "foliohog", "factor", "cve_ent"]],
                          on=["folioviv", "foliohog"], how="left")
        else:
            po["cve_ent"] = po["entidad"]
        # 2018 trae blancos como strings en asis_esc → castear antes de filtrar
        po["asis_esc"] = pd.to_numeric(po["asis_esc"], errors="coerce")
        po = po[po["asis_esc"].isin([1, 2])]
        filas = []
        for nombre, lo, hi in [("15-17", 15, 17), ("18-22", 18, 22)]:
            sub = po[(po["edad"] >= lo) & (po["edad"] <= hi)]
            nal = ((sub["asis_esc"] == 1) * sub["factor"]).sum() / sub["factor"].sum() * 100
            print(f"  {ola} asistencia {nombre}: {nal:.1f}% nacional")
            est = (sub.assign(asiste_w=(sub["asis_esc"] == 1) * sub["factor"])
                   .groupby("cve_ent").agg(asiste_w=("asiste_w", "sum"), w=("factor", "sum")))
            est[f"tasa_{nombre}"] = est["asiste_w"] / est["w"] * 100
            filas.append(est[[f"tasa_{nombre}"]])
        tasas[ola] = pd.concat(filas, axis=1)

    delta = (tasas[2024] - tasas[2018]).reset_index()
    delta = pl.from_pandas(delta).with_columns(pl.col("cve_ent").cast(pl.Int64))
    cruce = delta.join(bpc.select("cve_ent", "estado", "becas_pc"), on="cve_ent")
    assert cruce.height == 32

    for grupo in ("15-17", "18-22"):
        c = np.corrcoef(cruce["becas_pc"], cruce[f"tasa_{grupo}"])[0, 1]
        print(f"  corr(becas_pc, Δasistencia {grupo} 2018→2024) = {c:+.3f}")

    c_15 = np.corrcoef(cruce["becas_pc"], cruce["tasa_15-17"])[0, 1]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=cruce["becas_pc"], y=cruce["tasa_15-17"], mode="markers+text",
        text=cruce["estado"], textposition="top center", textfont=dict(size=9),
        marker=dict(size=8, color="#1f77b4"), showlegend=False))
    z2 = np.polyfit(cruce["becas_pc"], cruce["tasa_15-17"], 1)
    xs2 = np.linspace(float(cruce["becas_pc"].min()), float(cruce["becas_pc"].max()), 50)
    fig.add_trace(go.Scatter(x=xs2, y=np.polyval(z2, xs2), mode="lines",
                             line=dict(color="#c0392b"), name="tendencia"))
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(
        title="¿Donde llegó más beca subió más la asistencia? (15-17 años, 2018→2024)<br>"
              f"<sup>corr = {c_15:+.2f} · y: Δ puntos de asistencia escolar · x: becas CNBBBJ per cápita</sup>",
        xaxis_title="becas per cápita (prom. 2022-2024, MXN/año)",
        yaxis_title="Δ asistencia escolar 15-17 (pp, 2018→2024)",
    )
    guardar_fig(fig, "cap4_becas_asistencia")


if __name__ == "__main__":
    main()
