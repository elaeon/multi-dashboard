"""
Cap 9.B — Tomas clandestinas (CartoCrítica 2008-2016): ¿cierra el ducto la anomalía
Guanajuato?

El Cap 9 dejó al ducto como renta hipotética (Guanajuato ≈40/100k sin puerto, frontera
ni mina) por falta de datos. Aquí se testea con las 8,089 tomas clandestinas
georreferenciadas de CartoCrítica (data/cartocritica/, 2008-2016):

  1. Sobre-representación: municipios con tomas (binario y top-cuartil de intensidad)
     vs share de población — misma métrica del Cap 9.
  2. OLS municipal: log(1+tasa 2023-24) ~ ln(1+tomas) + ln(pob) ± FE de estado.
  3. Falsificación estatal: tomas per cápita 2008-16 × tasa 2023-24 — ¿la renta
     predice la violencia futura en todos lados, o solo donde se disputó?
  4. Secuencia temporal: tomas anuales de Guanajuato × tasas estatales 2015-2024 de
     los 4 estados con más tomas (Ver/Gto/Pue/Hgo).

Caveats: las tomas son detecciones reportadas por Pemex (sesgo de detección/reporte);
la ventana 2008-2016 es previa al pico del huachicol y al operativo federal de 2019
que reorganizó su geografía.

Figuras → centralismo/informe/figuras/cap9_tomas_*.png
Run: uv run python scripts/centralismo/cap9_tomas_clandestinas.py
"""

import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import polars as pl
import statsmodels.formula.api as smf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from comun import CORTO, RAIZ, cargar_poblacion, guardar_fig, normalizar_estado

MESES = ["Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio", "Julio",
         "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"]

CSV_TOMAS = RAIZ / "data/cartocritica/tomas_clandestinas_oleoductos.csv"

# razones share-homicidios/share-población publicadas por cap9_rutas_violencia.py
REF_CAP9 = {"cruce fronterizo": 2.26, "puerto mayor": 1.40, "municipio minero": 1.37}


def _clave(s: str) -> str:
    s = unicodedata.normalize("NFD", str(s)).encode("ascii", "ignore").decode().lower()
    return s.replace("gral.", "general").strip()


def cargar_homicidios(años: list[int], por_año: bool = False) -> pl.DataFrame:
    lf = (pl.scan_parquet(RAIZ / "data/incidencia_delictiva/incidencia_fuero_comun/"
                          "incidencia_delictiva_fuero_comun.parquet")
          .filter(pl.col("Año").is_in(años)
                  & (pl.col("Subtipo de delito") == "Homicidio doloso"))
          .with_columns(pl.sum_horizontal(MESES).alias("casos")))
    if por_año:
        return lf.group_by("Clave_Ent", "Año").agg(pl.sum("casos")).collect()
    inc = (lf.group_by("Cve. Municipio", "Municipio", "Clave_Ent")
           .agg((pl.sum("casos") / len(años)).alias("casos"))  # promedio anual
           .collect())
    return inc.with_columns(pl.col("Municipio").map_elements(_clave, return_dtype=pl.Utf8)
                            .alias("mun_clave"))


def cargar_tomas() -> pd.DataFrame:
    """CSV CartoCrítica → una fila por toma con cve_ent y clave de municipio."""
    t = pd.read_csv(CSV_TOMAS, encoding="utf-8-sig")
    n0 = len(t)
    t = t.dropna(subset=["Estado", "Municipio"]).copy()
    t["cve_ent"] = t["Estado"].map(lambda x: normalizar_estado(x, estricto=True))
    t["mun_clave"] = t["Municipio"].map(_clave)
    print(f"  tomas: {n0:,} registros, {len(t):,} con estado+municipio "
          f"({t['Year of Fecha'].min()}-{t['Year of Fecha'].max()})")
    return t


def resolver_municipios(tomas: pd.DataFrame, hom: pd.DataFrame) -> pd.DataFrame:
    """(cve_ent, municipio) → Cve. Municipio contra el padrón SESNSP; exacto + startswith."""
    agg = tomas.groupby(["cve_ent", "mun_clave"]).size().reset_index(name="tomas")

    def resolver(row):
        sub = hom[(hom["Clave_Ent"] == row.cve_ent) & (hom["mun_clave"] == row.mun_clave)]
        if len(sub) != 1:
            sub = hom[(hom["Clave_Ent"] == row.cve_ent)
                      & hom["mun_clave"].str.startswith(row.mun_clave)]
        return sub["Cve. Municipio"].iloc[0] if len(sub) == 1 else None

    agg["cve_mun"] = agg.apply(resolver, axis=1)
    ok = agg["cve_mun"].notna()
    cobertura = agg.loc[ok, "tomas"].sum() / agg["tomas"].sum()
    print(f"  match municipal: {ok.sum()}/{len(agg)} municipios · "
          f"{agg.loc[ok, 'tomas'].sum():,}/{agg['tomas'].sum():,} tomas ({cobertura*100:.1f}%)")
    assert cobertura >= 0.95, f"cobertura de tomas insuficiente: {cobertura:.3f}"
    return agg[ok].astype({"cve_mun": int})[["cve_mun", "tomas"]]


def main():
    tomas = cargar_tomas()
    hom = cargar_homicidios([2023, 2024]).to_pandas()
    tomas_mun = resolver_municipios(tomas, hom)

    pob = pd.read_csv(RAIZ / "data/conapo/municipios_2020_todos.csv")[["CLAVE", "POB_TOTAL"]]
    base = (hom.merge(pob, left_on="Cve. Municipio", right_on="CLAVE", how="inner")
            .merge(tomas_mun, left_on="Cve. Municipio", right_on="cve_mun", how="left"))
    base["tomas"] = base["tomas"].fillna(0)
    base["tasa"] = base["casos"] / base["POB_TOTAL"] * 1e5
    print(f"=== Cap 9.B: {len(base):,} municipios en base ===")

    # ---- 1. sobre-representación (share homicidios ÷ share población)
    tot_h, tot_p = base["casos"].sum(), base["POB_TOTAL"].sum()
    con = base[base["tomas"] > 0]
    q75 = con["tomas"].quantile(0.75)
    alta = base[base["tomas"] >= q75]
    razones = {}
    print("\n  sobre-representación:")
    for etiqueta, sub in [(f"con tomas (≥1)", con), (f"top-cuartil (≥{q75:.0f} tomas)", alta)]:
        sh, sp = sub["casos"].sum() / tot_h, sub["POB_TOTAL"].sum() / tot_p
        razones[etiqueta] = sh / sp
        print(f"    {etiqueta:<28} n={len(sub):>3} · {sp*100:>5.1f}% pob · "
              f"{sh*100:>5.1f}% homicidios · razón={sh/sp:.2f}×")

    # ---- 2. OLS municipal: intensidad de tomas sin/con FE de estado
    df = base.assign(log_tasa=np.log1p(base["tasa"]), ln_tomas=np.log1p(base["tomas"]),
                     ln_pob=np.log(base["POB_TOTAL"]))
    mA = smf.ols("log_tasa ~ ln_tomas + ln_pob", data=df).fit(cov_type="HC1")
    mB = smf.ols("log_tasa ~ ln_tomas + ln_pob + C(Clave_Ent)", data=df).fit(cov_type="HC1")
    print(f"\n  OLS ln(1+tomas) sobre log(1+tasa 2023-24), n={int(mA.nobs)}:")
    for m, etiqueta in [(mA, "sin FE"), (mB, "+ FE estado")]:
        b, p = m.params["ln_tomas"], m.pvalues["ln_tomas"]
        print(f"    {etiqueta:<12} β={b:+.3f} (p={p:.4f}) → "
              f"duplicar tomas ≈ {(2**b - 1)*100:+.0f}% de tasa · R²={m.rsquared:.2f}")
    gto = base[base["Clave_Ent"] == 11]
    print(f"    dentro de Guanajuato: corr(tomas, tasa) = "
          f"{np.corrcoef(gto['tomas'], gto['tasa'])[0, 1]:+.2f} (n={len(gto)})")

    # ---- 3. falsificación estatal: tomas pc 2008-16 × tasa 2023-24
    # tomas por estado sobre el CSV completo (el estado se conoce aunque el municipio no)
    t_est = tomas.groupby("cve_ent").size().rename("tomas")
    est = (base.groupby("Clave_Ent").agg(casos=("casos", "sum"), pob=("POB_TOTAL", "sum"))
           .join(t_est).fillna({"tomas": 0}))
    est["tasa"] = est["casos"] / est["pob"] * 1e5
    est["tomas_pc"] = est["tomas"] / est["pob"] * 1e5
    corr = np.corrcoef(est["tomas_pc"], est["tasa"])[0, 1]
    print(f"\n  falsificación estatal: corr(tomas pc 2008-16, tasa 2023-24) = {corr:+.2f}")
    for cve in [30, 21, 13, 11]:
        e = est.loc[cve]
        print(f"    {CORTO[cve]:<12} tomas={e['tomas']:>5.0f} · tasa 2023-24={e['tasa']:.1f}")

    # ---- 4. secuencia temporal: tomas Gto × tasas estatales 2015-2024
    top4 = [11, 30, 21, 13]  # Gto, Ver, Pue, Hgo
    anual = cargar_homicidios(list(range(2015, 2025)), por_año=True).to_pandas()
    pob_año = (cargar_poblacion()
               .filter(pl.col("cve_ent").is_in(top4) & pl.col("año").is_in(range(2015, 2025)))
               .to_pandas())
    anual = anual.merge(pob_año, left_on=["Clave_Ent", "Año"], right_on=["cve_ent", "año"])
    anual["tasa"] = anual["casos"] / anual["pob_total"] * 1e5
    tomas_gto = tomas[tomas["cve_ent"] == 11].groupby("Year of Fecha").size()
    print("\n  secuencia Guanajuato: tomas 2011-2015 "
          f"{tomas_gto[2011]}→{tomas_gto[2015]} (×{tomas_gto[2015]/tomas_gto[2011]:.0f}); "
          "tasas 2017→2020: "
          + " · ".join(f"{CORTO[c]} {anual[(anual.Clave_Ent == c) & (anual.Año == 2017)]['tasa'].iloc[0]:.0f}"
                       f"→{anual[(anual.Clave_Ent == c) & (anual.Año == 2020)]['tasa'].iloc[0]:.0f}"
                       for c in top4))

    # ---- figuras
    fig = go.Figure()
    todas = {**{f"ducto: {k}": v for k, v in razones.items()}, **REF_CAP9}
    orden = sorted(todas, key=todas.get)
    fig.add_trace(go.Bar(
        y=orden, x=[todas[k] for k in orden], orientation="h",
        marker_color=["#c0392b" if k.startswith("ducto") else "#95a5a6" for k in orden]))
    fig.add_vline(x=1, line_dash="dash", line_color="black",
                  annotation_text="proporcional a su población")
    fig.update_layout(
        title="El ducto sobre-representa homicidios solo en intensidad alta (2023-2024)<br>"
              "<sup>share de homicidios ÷ share de población; gris = features del Cap 9; "
              "rojo = municipios con tomas clandestinas 2008-2016 (CartoCrítica)</sup>",
        xaxis_title="razón de sobre-representación",
    )
    guardar_fig(fig, "cap9_tomas_sobrerrepresentacion")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=est["tomas_pc"], y=est["tasa"], mode="markers+text",
        text=[CORTO[c] if c in (11, 30, 21, 13, 27, 6, 8) or est.loc[c, "tomas_pc"] > 8
              else "" for c in est.index],
        textposition="top center", marker=dict(size=9, color="#c0392b")))
    fig.update_layout(
        title=f"La renta del ducto NO predice la violencia futura por sí sola (corr = {corr:+.2f})<br>"
              "<sup>tomas clandestinas per cápita 2008-2016 (CartoCrítica) vs tasa de homicidio "
              "2023-2024; Veracruz, Puebla e Hidalgo tuvieron la renta sin la violencia</sup>",
        xaxis_title="tomas clandestinas 2008-2016 por 100 mil hab.",
        yaxis_title="homicidios dolosos por 100 mil hab. (prom. 2023-2024)",
    )
    guardar_fig(fig, "cap9_tomas_falsificacion")

    fig = go.Figure()
    fig.add_trace(go.Bar(x=tomas_gto.index, y=tomas_gto.values, name="tomas en Guanajuato",
                         marker_color="#f0c9a8", yaxis="y2"))
    colores = {11: "#c0392b", 30: "#2c3e50", 21: "#7f8c8d", 13: "#bdc3c7"}
    for cve in top4:
        sub = anual[anual["Clave_Ent"] == cve].sort_values("Año")
        fig.add_trace(go.Scatter(x=sub["Año"], y=sub["tasa"], name=CORTO[cve],
                                 line=dict(color=colores[cve],
                                           width=3 if cve == 11 else 1.5)))
    fig.add_vline(x=2019, line_dash="dot", line_color="black",
                  annotation_text="operativo federal 2019", annotation_position="top left")
    fig.update_layout(
        title="La renta precede a la violencia — pero solo explotó donde se disputó<br>"
              "<sup>barras: tomas clandestinas detectadas en Guanajuato (CartoCrítica); líneas: tasa "
              "de homicidio doloso de los 4 estados con más tomas 2008-2016 (SESNSP/CONAPO)</sup>",
        yaxis=dict(title="homicidios dolosos por 100 mil hab."),
        yaxis2=dict(title="tomas clandestinas (Gto)", overlaying="y", side="right",
                    showgrid=False),
        legend=dict(orientation="h", y=1.02, yanchor="bottom"),
    )
    guardar_fig(fig, "cap9_tomas_timing")


if __name__ == "__main__":
    main()
