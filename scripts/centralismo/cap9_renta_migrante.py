"""
Cap 9.C — La renta migrante: Chiapas como experimento natural del timing.

La secuencia de 9.B (Guanajuato: la renta precede a la violencia) repetida en el caso
más limpio: el estado más pobre del país estuvo entre los menos violentos hasta que el
flujo migrante se volvió una renta tarificable en su frontera. Tres piezas:

  A. La renta, medida: quiebre de remesas de Chiapas post-2020 (Banxico CA79; el
     Cap 9 y la memoria del repo la tratan como renta del flujo migrante, no remesa
     clásica — el trabajador en tránsito/EEUU recién llegado gira a través de Chiapas).
  B. La disputa, medida: SESNSP municipal 2015-2025 — corredor sierra-frontera vs
     resto de Chiapas (homicidio, extorsión, secuestro).
  C. La consecuencia: desplazamiento forzado — IDMC GIDD nacional 2009-2025
     (data/idmc/, CMDPDH no publica microdato) + eventos IDU 2025-26 con estado.

Figuras → centralismo/informe/figuras/cap9_chiapas_{remesas,timing}.png
Run: uv run python scripts/centralismo/cap9_renta_migrante.py
"""

import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).resolve().parent))
from comun import CORTO, RAIZ, guardar_fig
from cap5_remesas import cargar_remesas_pc
from cap9_rutas_violencia import MESES, cargar_homicidios, resolver_catalogo

CHIAPAS = 7
PARES_SUR = {20: "Oaxaca", 12: "Guerrero", 30: "Veracruz", 27: "Tabasco"}
# corredor sierra-frontera (la zona de la disputa CJNG-Sinaloa-"cártel de Chiapas y
# Guatemala") + los dos nodos del corredor costero del Soconusco
CORREDOR = [(7, n) for n in [
    "Frontera Comalapa", "Chicomuselo", "Motozintla", "Amatenango de la Frontera",
    "Mazapa de Madero", "Bejucal de Ocampo", "Bella Vista", "Siltepec",
    "El Porvenir", "La Grandeza", "Tapachula", "Suchiate"]]


def remesas():
    print("=== A. La renta: quiebre de remesas de Chiapas (Banxico CA79) ===")
    rem = cargar_remesas_pc()  # cve_ent × año: musd, pob_total, rem_pc (2003-2025)
    rem = rem.with_columns((pl.col("musd") / pl.col("musd").sum().over("año") * 100)
                           .alias("share"))
    ch = rem.filter(pl.col("cve_ent") == CHIAPAS).sort("año")
    s03 = ch.filter(pl.col("año") == 2003)["share"][0]
    s19 = ch.filter(pl.col("año") == 2019)["share"][0]
    s25 = ch.filter(pl.col("año") == 2025)["share"][0]
    print(f"  share nacional de Chiapas: {s03:.2f}% (2003) · {s19:.2f}% (2019) · "
          f"{s25:.2f}% (2025)")
    d = ch.with_columns(pl.col("share").diff().alias("d_share")).drop_nulls("d_share")
    mx = d.sort("d_share", descending=True).head(3)
    print("  mayores saltos anuales del share:", " · ".join(
        f"{r['año']}: +{r['d_share']:.2f} pp" for r in mx.iter_rows(named=True)))
    m19 = ch.filter(pl.col("año") == 2019)["musd"][0]
    m25 = ch.filter(pl.col("año") == 2025)["musd"][0]
    print(f"  remesas Chiapas: {m19:,.0f} → {m25:,.0f} MUSD (2019→2025, "
          f"×{m25/m19:.1f}); per cápita "
          f"{ch.filter(pl.col('año') == 2019)['rem_pc'][0]:.0f} → "
          f"{ch.filter(pl.col('año') == 2025)['rem_pc'][0]:.0f} USD/hab")
    print("  contraste pares del sur (crecimiento 2019→2025 en USD):")
    for cve, nombre in PARES_SUR.items():
        p = rem.filter(pl.col("cve_ent") == cve)
        f = p.filter(pl.col("año") == 2025)["musd"][0] / p.filter(pl.col("año") == 2019)["musd"][0]
        print(f"    {nombre:<9} ×{f:.1f}")
    nac = (rem.group_by("año").agg(pl.sum("musd")).sort("año"))
    f_nac = nac.filter(pl.col("año") == 2025)["musd"][0] / nac.filter(pl.col("año") == 2019)["musd"][0]
    print(f"    nacional  ×{f_nac:.1f}")
    print("  lectura: la remesa clásica es del expulsor con diáspora (Cap 5); Chiapas "
          "no la tenía — el salto post-2020 coincide con el flujo migrante tarificado, "
          "no con una diáspora nueva (Cap 9 / memoria Banxico)")
    return rem


def violencia():
    print("\n=== B. La disputa: corredor sierra-frontera vs resto de Chiapas ===")
    cat = cargar_homicidios()  # catálogo municipal con mun_clave y Clave_Ent
    cves = resolver_catalogo(CORREDOR, cat, "corredor sierra-frontera")

    inc = (pl.scan_parquet(RAIZ / "data/incidencia_delictiva/incidencia_fuero_comun/"
                           "incidencia_delictiva_fuero_comun.parquet")
           .filter((pl.col("Clave_Ent") == CHIAPAS) & pl.col("Año").is_between(2015, 2025)
                   & pl.col("Subtipo de delito").is_in(
                       ["Homicidio doloso", "Extorsión", "Secuestro"]))
           .with_columns(pl.sum_horizontal(MESES).alias("casos"))
           .group_by("Año", "Cve. Municipio", "Subtipo de delito")
           .agg(pl.sum("casos")).collect()
           .rename({"Cve. Municipio": "cve_mun", "Subtipo de delito": "delito"}))
    pob = (pl.read_csv(RAIZ / "data/conapo/municipios_2020_todos.csv")
           .select(pl.col("CLAVE").cast(pl.Int64).alias("cve_mun"),
                   pl.col("POB_TOTAL").cast(pl.Float64).alias("pob"))
           .filter(pl.col("cve_mun") // 1000 == CHIAPAS))
    pob_c = pob.filter(pl.col("cve_mun").is_in(list(cves)))["pob"].sum()
    pob_r = pob["pob"].sum() - pob_c
    print(f"  corredor: {len(cves)} municipios, {pob_c/1e6:.2f} M hab "
          f"({pob_c/pob['pob'].sum()*100:.0f}% de Chiapas)")

    series = (inc.with_columns(pl.col("cve_mun").is_in(list(cves)).alias("corr"))
              .group_by("Año", "delito", "corr").agg(pl.sum("casos"))
              .with_columns((pl.col("casos")
                             / pl.when(pl.col("corr")).then(pob_c).otherwise(pob_r)
                             * 1e5).alias("tasa"))
              .sort("Año"))
    for delito in ["Homicidio doloso", "Extorsión", "Secuestro"]:
        for corr, etiqueta in [(True, "corredor"), (False, "resto   ")]:
            s = series.filter((pl.col("delito") == delito) & (pl.col("corr") == corr))
            t = {r["Año"]: r["tasa"] for r in s.iter_rows(named=True)}
            pre = np.mean([t.get(a, 0) for a in (2017, 2018, 2019)])
            post = np.mean([t.get(a, 0) for a in (2023, 2024, 2025)])
            print(f"  {delito:<17} {etiqueta}: tasa 2017-19 = {pre:5.1f} → "
                  f"2023-25 = {post:5.1f} por 100k  (×{post/pre if pre else float('nan'):.1f})")
    hc = series.filter((pl.col("delito") == "Homicidio doloso") & pl.col("corr"))
    print("  homicidio corredor por año:", " · ".join(
        f"{r['Año']}: {r['tasa']:.0f}" for r in hc.sort("Año").iter_rows(named=True)
        if r["Año"] >= 2019))
    return series


def subregistro():
    """Método E4 del Cap 12 aplicado a Chiapas: víctimas INEGI ÷ carpetas SESNSP."""
    print("\n=== B-2. ¿El SESNSP ve la guerra? Auditoría de subregistro (E4) ===")
    archivos = {a: f"conjunto_de_datos_defunciones_registradas_{a}_csv.zip"
                for a in range(2018, 2024)} | {2024: "conjunto_de_datos_edr2024_csv.zip"}
    partes = []
    for zname in archivos.values():  # registros tardíos caen en el archivo siguiente
        z = zipfile.ZipFile(RAIZ / "data/inegi/defunciones" / zname)
        main = [n for n in z.namelist()
                if "conjunto_de_datos/" in n and n.lower().endswith(".csv")][0]
        partes.append(
            pl.read_csv(io.BytesIO(z.read(main)), infer_schema_length=0,
                        columns=["ent_ocurr", "causa_def", "anio_ocur"])
            .filter(pl.col("causa_def").str.contains(r"^(X8[5-9]|X9|Y0)"))
            # 2018-2019 traen un tab colgante en los campos numéricos
            .with_columns(pl.col("ent_ocurr").str.strip_chars().cast(pl.Int64, strict=False),
                          pl.col("anio_ocur").str.strip_chars().cast(pl.Int64, strict=False)))
    inegi = (pl.concat(partes)
             .filter(pl.col("anio_ocur").is_between(2018, 2023)
                     & pl.col("ent_ocurr").is_between(1, 32))
             .group_by("ent_ocurr", "anio_ocur").agg(pl.len().alias("inegi")))
    sesnsp = (pl.scan_parquet(RAIZ / "data/incidencia_delictiva/incidencia_fuero_comun/"
                              "incidencia_delictiva_fuero_comun.parquet")
              .filter(pl.col("Año").is_between(2018, 2023)
                      & (pl.col("Subtipo de delito") == "Homicidio doloso"))
              .with_columns(pl.sum_horizontal(MESES).alias("casos"))
              .group_by("Clave_Ent", "Año").agg(pl.sum("casos").alias("sesnsp"))
              .collect().with_columns(pl.col("Clave_Ent").cast(pl.Int64)))
    cmp = inegi.rename({"ent_ocurr": "cve_ent", "anio_ocur": "año"}).join(
        sesnsp.rename({"Clave_Ent": "cve_ent", "Año": "año"}), on=["cve_ent", "año"])
    print("  razón víctimas INEGI (X85-Y09, ocurrencia) ÷ carpetas SESNSP:")
    print("  año        Chiapas   nacional")
    for a in range(2018, 2024):
        c = cmp.filter((pl.col("cve_ent") == CHIAPAS) & (pl.col("año") == a))
        n = cmp.filter(pl.col("año") == a)
        print(f"  {a}:      {c['inegi'][0]/c['sesnsp'][0]:5.2f}      "
              f"{n['inegi'].sum()/n['sesnsp'].sum():5.2f}   "
              f"(Chiapas INEGI {c['inegi'][0]:,} vs SESNSP {c['sesnsp'][0]:,})")


def desplazamiento():
    print("\n=== C. La consecuencia: desplazamiento forzado (IDMC) ===")
    gidd = pl.read_csv(RAIZ / "data/idmc/idmc_gidd_desplazamiento_conflicto_mex.csv")
    assert gidd.filter(pl.col("year") == 2021)["new_displacement"][0] == 28867
    assert gidd.filter(pl.col("year") == 2024)["new_displacement"][0] == 25543
    ult = gidd.filter(pl.col("year") >= 2018).sort("year")
    print("  nuevos desplazamientos por conflicto, nacional (GIDD):",
          " · ".join(f"{r['year']}: {r['new_displacement']:,}"
                     for r in ult.iter_rows(named=True)))
    idu = (pl.read_csv(RAIZ / "data/idmc/mex_idmc_idu_events.csv", infer_schema_length=0)
           .filter(pl.col("displacement_type") == "Conflict")
           .with_columns(pl.col("figure").cast(pl.Int64)))
    chis = idu.filter(pl.col("locations_name").str.contains("Chiapas"))
    print(f"  eventos IDU 2025-26 (ventana móvil): {idu.height} eventos de conflicto; "
          f"Chiapas: {chis.height} eventos, {chis['figure'].sum():,} desplazados")
    print("  cifra citada (no reproducible desde el repo): IDMC GRID 2025 atribuye a "
          "Chiapas el 61.8% de los ~26 mil desplazamientos por conflicto de 2024 "
          "(focos: Tila, Chenalhó, Pantelhó); CMDPDH solo publica informes PDF")
    return gidd


def figuras(rem: pl.DataFrame, series: pl.DataFrame, gidd: pl.DataFrame):
    # 1. share de remesas: Chiapas vs pares del sur
    fig = go.Figure()
    for cve, color in [(CHIAPAS, "#c0392b")] + [(c, None) for c in PARES_SUR]:
        s = rem.filter(pl.col("cve_ent") == cve).sort("año")
        fig.add_trace(go.Scatter(
            x=s["año"], y=s["share"], mode="lines",
            name=CORTO[cve], line=dict(width=3 if cve == CHIAPAS else 1.2,
                                       color=color)))
    fig.add_vline(x=2020, line=dict(color="#7f8c8d", dash="dot"))
    fig.update_layout(
        title="La renta nueva: el share de remesas de Chiapas rompe su serie en 2020"
              "<br><sup>share del total nacional de remesas (Banxico CA79); pares del "
              "sur como contraste — ningún otro estado rompe</sup>",
        xaxis_title="año", yaxis_title="% del total nacional de remesas")
    guardar_fig(fig, "cap9_chiapas_remesas")

    # 2. timing: violencia del corredor vs resto + desplazamiento nacional
    fig = make_subplots(
        rows=1, cols=2, column_widths=[0.55, 0.45],
        subplot_titles=("extorsión en Chiapas: corredor vs resto (por 100k)",
                        "desplazamiento por conflicto, nacional (GIDD)"))
    for delito, dash in [("Extorsión", "solid"), ("Homicidio doloso", "dot")]:
        for corr, nombre, color in [(True, "corredor", "#c0392b"),
                                    (False, "resto", "#7f8c8d")]:
            s = series.filter((pl.col("delito") == delito) & (pl.col("corr") == corr))
            fig.add_trace(go.Scatter(
                x=s["Año"], y=s["tasa"], mode="lines",
                name=f"{delito.lower()} {nombre}", legendgroup=delito,
                line=dict(color=color, dash=dash,
                          width=2.5 if corr else 1.5)), row=1, col=1)
    g = gidd.filter(pl.col("year") >= 2015)
    fig.add_trace(go.Bar(x=g["year"], y=g["new_displacement"], showlegend=False,
                         marker_color="#2c3e50"), row=1, col=2)
    for c in (1, 2):
        fig.add_vline(x=2020.5, line=dict(color="#7f8c8d", dash="dot"), row=1, col=c)
    fig.update_layout(
        title="El timing: la renta rompe en 2020-21, la disputa y el desplazamiento "
              "después<br><sup>izq: SESNSP municipal (corredor sierra-frontera vs resto "
              "de Chiapas) · der: IDMC GIDD — los picos 2021 y 2024 son los años de la "
              "guerra en Chiapas</sup>",
        legend=dict(orientation="h", y=-0.15))
    guardar_fig(fig, "cap9_chiapas_timing", ancho=1100)


def main():
    rem = remesas()
    series = violencia()
    subregistro()
    gidd = desplazamiento()
    figuras(rem, series, gidd)


if __name__ == "__main__":
    main()
