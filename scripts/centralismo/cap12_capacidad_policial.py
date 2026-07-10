"""
Cap 12.B — E2 medido: la capacidad policial per cápita (cierra el mecanismo del Cap 12).

El Cap 12 dejó E2 ("la regulación nunca colapsó en el centro") apoyado en el test ZMVM
y proxies. Con los censos de gobierno ya en el repo se mide la protección directamente:

  A. Serie de fuerza de seguridad estatal per cápita 2016-2024 (CNGSPSPE peredad +
     CNSPE personal) y presupuesto ejercido de seguridad 2024 (CNSPE rec_presup).
  B. Estándar normativo: policía preventiva estatal operativa vs 1.8/1000 hab
     (SESNSP Modelo Óptimo de la Función Policial 2020).
  C. El test ZMVM cerrado: policías por 100k a ambos lados de la frontera
     administrativa (SSC-CDMX vs policía municipal CNGMD + estatal prorrateada).
  D. Test transversal: renta física (frontera/puerto/ducto) × capacidad policial →
     tasa de homicidio — la pregunta del usuario en forma 2×2.

Figuras → centralismo/informe/figuras/cap12_{policia_pc,zmvm_capacidad,renta_capacidad}.png
Run: uv run python scripts/centralismo/cap12_capacidad_policial.py
"""

import sys
import zipfile
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).resolve().parent))
from comun import CORTO, RAIZ, cargar_poblacion, guardar_fig, normalizar_estado
from cap12_paradoja_capital import (CDMX, EDOMEX, HOMICIDIO, EXTORSION,
                                    carga_mun_zmvm, cves_conurbados, tasas_estatales)
from cap9_rutas_violencia import FRONTERA, PUERTOS
from cap9_tomas_clandestinas import cargar_tomas

CNSPE = RAIZ / "data/inegi/cnspe"
CNGSPSPE = RAIZ / "data/inegi/cngspspe"
# edición CNSPE → (tabla de personal, patrón del ZIP); dato = edición - 1
CNSPE_PERSONAL = {2021: ("m1s1p2", "m1_*pers_fun*"), 2022: ("m1s1p2", "m1_*pers_fun*"),
                  2023: ("m1s1p2", "m1_*pers_fun*"), 2024: ("m1s2p2", "m1_*pers_fun*"),
                  2025: ("m1s2p2", "m1_*rec_huma*")}
# edición CNGSPSPE → (tabla, columna de total); dato = edición - 1. La tabla de
# personal cambia de nombre y de dimensión (edad → ingresos) entre ediciones.
CNGSPSPE_PERSONAL = {2017: ("peredad", "perfsptt"), 2018: ("peredad", "perfsptt"),
                     2019: ("pesegpub", "perpeott")}
# anclas de los DATA_OVERVIEW: total nacional por año de dato (validación dura)
ANCLAS = {2017: 217_287, 2018: 217_767, 2020: 225_544, 2021: 221_281,
          2022: 227_892, 2023: 235_832, 2024: 243_643}


def leer_zip_csv(zpath: Path, stem: str) -> pl.DataFrame:
    """CSV llamado `{stem}.csv` o `{stem}_*.csv` dentro del ZIP → polars str."""
    def nombre(n):
        return n.replace("\\", "/").split("/")[-1].lower()
    with zipfile.ZipFile(zpath) as z:
        key = next(n for n in z.namelist()
                   if (nombre(n) == stem + ".csv" or nombre(n).startswith(stem + "_"))
                   and n.lower().endswith(".csv"))
        raw = z.read(key)
    try:
        texto = raw.decode("utf-8-sig")  # BOM en CNSPE 2021-2022
    except UnicodeDecodeError:
        texto = raw.decode("latin-1")
    return pl.read_csv(texto.encode("utf-8"), infer_schema_length=0)


def col_estado(df: pl.DataFrame) -> str:
    return next(c for c in ("entidad_a", "entidad", "ENTIDAD", "cvegeo") if c in df.columns)


# ------------------------------------------------- A. serie de fuerza estatal

def fuerza_estatal(pob: pl.DataFrame) -> pl.DataFrame:
    print("=== A. Fuerza de seguridad estatal per cápita 2016-2024 ===")
    partes = []
    # CNGSPSPE ediciones 2017-2019 (dato 2016-2018)
    for ed, (tabla, tot) in CNGSPSPE_PERSONAL.items():
        zpath = next((CNGSPSPE / str(ed)).glob("m2_*ec_hum_pre_segpub*.zip"))
        df = leer_zip_csv(zpath, tabla)
        ce = col_estado(df)
        partes.append(df.with_columns(pl.col(ce).cast(pl.Int64).alias("cve_ent"),
                                      pl.col(tot).cast(pl.Float64, strict=False))
                      .group_by("cve_ent").agg(pl.sum(tot).alias("fuerza"))
                      .with_columns(pl.lit(ed - 1).alias("año")))
    # CNSPE ediciones 2021-2025 (dato 2020-2024): personal por sexo, total en sexostt
    for ed, (tabla, patron) in CNSPE_PERSONAL.items():
        df = leer_zip_csv(next((CNSPE / str(ed)).glob(patron + "*.zip")), tabla)
        ce = col_estado(df)
        partes.append(df.with_columns(pl.col(ce).cast(pl.Int64).alias("cve_ent"),
                                      pl.col("sexostt").cast(pl.Float64, strict=False))
                      .group_by("cve_ent").agg(pl.sum("sexostt").alias("fuerza"))
                      .with_columns(pl.lit(ed - 1).alias("año")))
    serie = pl.concat(partes).filter(pl.col("cve_ent").is_between(1, 32))
    for año, esperado in ANCLAS.items():
        total = serie.filter(pl.col("año") == año)["fuerza"].sum()
        assert abs(total - esperado) < 1, f"ancla {año}: {total:,.0f} ≠ {esperado:,}"
    print(f"  totales nacionales validados contra los DATA_OVERVIEW: "
          f"{', '.join(str(a) for a in ANCLAS)} ✓ (sin dato 2019: la edición 2020 "
          f"del CNGSPSPE eliminó el módulo de RRHH de seguridad)")

    serie = (serie.join(pob, on=["cve_ent", "año"])
             .with_columns((pl.col("fuerza") / pl.col("pob_total") * 1000).alias("pc")))
    print("  policías/1000 hab (todo el personal de la institución estatal de seguridad):")
    for año in sorted(serie["año"].unique()):
        d = serie.filter(pl.col("año") == año)
        f = d.filter(pl.col("cve_ent") == CDMX)
        rank = d.sort("pc", descending=True).with_row_index("r").filter(
            pl.col("cve_ent") == CDMX)["r"][0] + 1
        nac = d["fuerza"].sum() / d["pob_total"].sum() * 1000
        print(f"    {año}: CDMX={f['pc'][0]:5.2f} (fuerza {f['fuerza'][0]:>7,.0f}, "
              f"lugar {rank}/32) · nacional={nac:.2f} · mediana={d['pc'].median():.2f} "
              f"· CDMX/mediana={f['pc'][0] / d['pc'].median():.1f}×")
    d18 = serie.filter(pl.col("año") == 2018)
    share = d18.filter(pl.col("cve_ent") == CDMX)["fuerza"][0] / d18["fuerza"].sum()
    print(f"  CDMX concentra el {share * 100:.1f}% del personal estatal de seguridad del "
          f"país (dato 2018) con 7% de la población")
    print("  caveat: el total CDMX incluye Policía Auxiliar y Bancaria-Industrial; "
          "por eso B reporta también la medida conservadora (solo preventiva, MOFP)")

    # presupuesto ejercido de seguridad estatal, dato 2024 (CNSPE 2025 rec_presup)
    pre = leer_zip_csv(next((CNSPE / "2025").glob("*rec_presup*.zip")), "m1s4p1")
    pre = (pre.with_columns(pl.col("cve_ent").cast(pl.Int64),
                            pl.col("presup3").cast(pl.Float64, strict=False))
           .join(pob.filter(pl.col("año") == 2024), on="cve_ent"))
    f = pre.filter(pl.col("cve_ent") == CDMX)
    validos = pre.filter(pl.col("presup3").is_not_null())
    print(f"  presupuesto ejercido en seguridad 2024: CDMX={f['presup3'][0] / 1e9:.2f} mmdp "
          f"({f['presup3'][0] / validos['presup3'].sum() * 100:.1f}% del nacional; "
          f"QRoo no reporta) · per cápita CDMX={f['presup3'][0] / f['pob_total'][0]:,.0f} "
          f"MXN vs nacional={validos['presup3'].sum() / validos['pob_total'].sum():,.0f}")
    return serie


# ------------------------------------------------- B. Modelo Óptimo (preventiva)

def mofp(pob: pl.DataFrame) -> pl.DataFrame:
    print("\n=== B. Policía preventiva estatal vs estándar 1.8/1000 (MOFP jun-2020) ===")
    df = pd.read_excel(RAIZ / "data/sesnsp/modelo_funcion_policial/MOFP_2020.xlsx",
                       sheet_name="Junio 2020")
    col0 = df.columns[0]
    df = df[df[col0].notna() & (df[col0].astype(str).str.strip() != "")].copy()
    df["cve_ent"] = [normalizar_estado(str(s).rstrip("1/ ").strip(), estricto=True)
                     for s in df[col0]]
    assert len(df) == 32
    prev = (pl.DataFrame({"cve_ent": df["cve_ent"].to_list(),
                          "oper": df["Elementos Operativos"].astype(float).to_list(),
                          "cup": df["Cuentan con CUP"].astype(float).to_list()})
            .join(pob.filter(pl.col("año") == 2020), on="cve_ent")
            .with_columns((pl.col("oper") / pl.col("pob_total") * 1000).alias("prev_pc")))
    f = prev.filter(pl.col("cve_ent") == CDMX)
    abajo = prev.filter(pl.col("prev_pc") < 1.8).height
    print(f"  CDMX: {f['oper'][0]:,.0f} preventivos operativos = {f['prev_pc'][0]:.2f}/1000 "
          f"hab → {f['prev_pc'][0] / 1.8:.1f}× el estándar del Modelo Óptimo (1.8)")
    print(f"  resto del país: {abajo}/32 estados POR DEBAJO del estándar · "
          f"mediana={prev['prev_pc'].median():.2f}/1000 · "
          f"CDMX/mediana={f['prev_pc'][0] / prev['prev_pc'].median():.1f}×")
    print(f"  nuance de calidad: cobertura CUP de CDMX={f['cup'][0] / f['oper'][0] * 100:.0f}% "
          f"vs mediana estatal={prev.with_columns((pl.col('cup') / pl.col('oper')).alias('r'))['r'].median() * 100:.0f}% "
          f"— la ventaja del centro es de cantidad, no de certificación")
    return prev


# ------------------------------------------------- C. ZMVM cerrado

def zmvm_capacidad(pob: pl.DataFrame, serie: pl.DataFrame, prev: pl.DataFrame):
    print("\n=== C. El test ZMVM, cerrado: policías por 100k a ambos lados ===")
    # policía municipal CNGMD 2025 (dato 2024): m3s1p22, suma de grados jerárquicos
    z = (RAIZ / "data/inegi/cngmd/2025/"
         "datosabiertos_conjunto_de_datos_rec_huma_sp_cngmd2025_csv.zip")
    mun = (leer_zip_csv(z, "m3s1p22")
           .with_columns(pl.col("cvegeo").str.zfill(5).cast(pl.Int64).alias("cve_mun"),
                         pl.col("sexostt").cast(pl.Float64, strict=False))
           .group_by("cve_mun").agg(pl.sum("sexostt").alias("pol_mun"),
                                    pl.col("sexostt").is_not_null().any().alias("reporta")))
    alcaldias = mun.filter(pl.col("cve_mun").is_in(range(9002, 9018)))
    assert not alcaldias["reporta"].any(), "alguna alcaldía reporta policía propia"
    print(f"  hecho institucional: las 16 alcaldías CDMX reportan NA en el CNGMD — "
          f"cero policía propia; toda la fuerza es del gobierno central de la ciudad")

    cves_con = cves_conurbados()
    pob_mun = (pl.read_csv(RAIZ / "data/conapo/municipios_2020_todos.csv")
               .select(pl.col("CLAVE").cast(pl.Int64).alias("cve_mun"),
                       pl.col("POB_TOTAL").cast(pl.Float64).alias("pob")))
    con = mun.filter(pl.col("cve_mun").is_in(list(cves_con))).join(pob_mun, on="cve_mun")
    assert con.height == len(cves_con)
    pob_con = con["pob"].sum()
    pob_mex = pob.filter((pl.col("cve_ent") == EDOMEX) & (pl.col("año") == 2024))["pob_total"][0]
    pob_cdmx = pob.filter((pl.col("cve_ent") == CDMX) & (pl.col("año") == 2024))["pob_total"][0]
    est24 = {r["cve_ent"]: r["fuerza"]
             for r in serie.filter(pl.col("año") == 2024).iter_rows(named=True)}

    pol_mun_con = con["pol_mun"].sum()
    pol_est_con = est24[EDOMEX] * pob_con / pob_mex  # estatal prorrateada per cápita
    lado_con = (pol_mun_con + pol_est_con) / pob_con * 1e5
    lado_cdmx = est24[CDMX] / pob_cdmx * 1e5
    prev_cdmx = prev.filter(pl.col("cve_ent") == CDMX)
    lado_cdmx_prev = prev_cdmx["oper"][0] / prev_cdmx["pob_total"][0] * 1e5
    print(f"  Edomex conurbado ({len(cves_con)} munis, {pob_con / 1e6:.1f}M): "
          f"{pol_mun_con:,.0f} policías municipales + {pol_est_con:,.0f} estatales "
          f"prorrateados = {lado_con:.0f}/100k")
    print(f"  Alcaldías CDMX ({pob_cdmx / 1e6:.1f}M): {est24[CDMX]:,.0f} de la SSC = "
          f"{lado_cdmx:.0f}/100k (medida conservadora, solo preventiva MOFP-2020: "
          f"{lado_cdmx_prev:.0f}/100k)")
    print(f"  razón de capacidad CDMX/conurbado: {lado_cdmx / lado_con:.1f}× "
          f"(conservadora {lado_cdmx_prev / lado_con:.1f}×)")

    # violencia a ambos lados (mismas definiciones del Cap 12 sección D)
    hom = carga_mun_zmvm(HOMICIDIO, "v")
    ext = carga_mun_zmvm(EXTORSION, "v")
    tasas = {}
    for nombre, d in [("hom", hom), ("ext", ext)]:
        cd = d.filter(pl.col("Clave_Ent") == CDMX).join(
            pob_mun, left_on="Cve. Municipio", right_on="cve_mun")
        cn = d.filter(pl.col("Cve. Municipio").is_in(list(cves_con))).join(
            pob_mun, left_on="Cve. Municipio", right_on="cve_mun")
        tasas[nombre] = (cd["v"].sum() / cd["pob"].sum() * 1e5,
                         cn["v"].sum() / cn["pob"].sum() * 1e5)
    print(f"  violencia 2023-24: homicidio {tasas['hom'][0]:.1f} vs {tasas['hom'][1]:.1f} · "
          f"extorsión {tasas['ext'][0]:.1f} vs {tasas['ext'][1]:.1f} (CDMX vs conurbado)")
    print("  caveat: no incluye despliegue federal (GN/Sedena) ni policía de proximidad "
          "de facto; la asimetría medida es la del aparato civil local")

    lados = ["Alcaldías CDMX", "Edomex conurbado"]
    fig = make_subplots(cols=2, column_widths=[0.45, 0.55], horizontal_spacing=0.14,
                        subplot_titles=("capacidad: policías por 100 mil hab",
                                        "violencia por 100 mil hab (2023-2024)"))
    fig.add_trace(go.Bar(x=lados, y=[lado_cdmx, lado_con], showlegend=False,
                         marker_color=["#2980b9", "#c0392b"],
                         text=[f"{lado_cdmx:,.0f}", f"{lado_con:,.0f}"],
                         textposition="outside"), row=1, col=1)
    fig.add_annotation(x=0, y=lado_cdmx_prev, xref="x", yref="y", ax=60, ay=-25,
                       text=f"solo preventiva: {lado_cdmx_prev:,.0f}",
                       font=dict(size=10), arrowcolor="#7f8c8d")
    for i, (nombre, etiqueta) in enumerate([("hom", "homicidio doloso"),
                                            ("ext", "extorsión")]):
        fig.add_trace(go.Bar(x=lados, y=list(tasas[nombre]), name=etiqueta,
                             marker_color=["#95a5a6", "#34495e"][i],
                             text=[f"{v:.1f}" for v in tasas[nombre]],
                             textposition="outside"), row=1, col=2)
    fig.update_layout(
        title="La discontinuidad ZMVM, con el mecanismo medido: la protección cae y la "
              "violencia sube al cruzar la línea<br><sup>policías: SSC-CDMX (CNSPE dato "
              "2024) vs municipales CNGMD 2024 + estatales Edomex prorrateados · misma "
              "metrópoli, distinta jurisdicción</sup>",
        legend=dict(orientation="h", y=-0.12))
    guardar_fig(fig, "cap12_zmvm_capacidad")


# ------------------------------------------------- D. renta física × capacidad

def renta_capacidad(pob: pl.DataFrame, serie: pl.DataFrame):
    print("\n=== D. Test transversal: renta física × capacidad → homicidio ===")
    tomas = cargar_tomas()
    top_tomas = set(tomas.groupby("cve_ent").size().nlargest(8).index)
    fisica = ({c for c, _ in FRONTERA} | {c for c, _ in PUERTOS} | top_tomas)
    print(f"  renta física = frontera ∪ puerto mayor ∪ top-8 tomas clandestinas "
          f"(catálogos Cap 9): {len(fisica)} estados (la mina se excluye: a nivel "
          f"estatal no discrimina)")

    hom = tasas_estatales(HOMICIDIO, [2023, 2024, 2025], pob)
    b = (serie.filter(pl.col("año") == 2024).select("cve_ent", "pc")
         .join(hom, on="cve_ent")
         .with_columns(pl.col("cve_ent").is_in(list(fisica)).alias("fisica")))
    med = b["pc"].median()
    print(f"  mediana de capacidad (policías estatales/1000, dato 2024): {med:.2f}")
    print(f"  tasa media de homicidio 2023-25 por celda:")
    for fis in (True, False):
        for alta in (False, True):
            d = b.filter((pl.col("fisica") == fis)
                         & ((pl.col("pc") >= med) == alta))
            print(f"    renta física={'sí' if fis else 'no ':<2} · capacidad "
                  f"{'alta' if alta else 'baja'}: {d['tasa'].mean():5.1f}/100k "
                  f"(n={d.height:2d})  {d.sort('tasa', descending=True)['estado'].head(4).to_list()}")
    print("  guarda de endogeneidad: la correlación cruda policía↔homicidio está "
          "confundida (los estados violentos contratan más y la federación despliega "
          "GN donde arde); el 2×2 es descriptivo con n=32 — el test limpio del "
          "mecanismo es la discontinuidad ZMVM (sección C)")

    fig = go.Figure()
    for fis, color, nombre in [(True, "#c0392b", "con renta física (frontera/puerto/ducto)"),
                               (False, "#2980b9", "sin renta física disputable")]:
        d = b.filter(pl.col("fisica") == fis)
        fig.add_trace(go.Scatter(
            x=d["pc"], y=d["tasa"], mode="markers+text", text=d["estado"],
            name=nombre, textposition="top center", textfont=dict(size=9),
            marker=dict(size=10, color=color)))
    fig.add_vline(x=med, line_dash="dot", line_color="#95a5a6")
    fig.add_hline(y=b["tasa"].median(), line_dash="dot", line_color="#95a5a6")
    fig.add_annotation(x=1, y=b["tasa"].min() + 6, xref="x", xanchor="right",
                       text="CDMX: máxima protección,<br>renta líquida — sin disputa",
                       showarrow=False, font=dict(size=11, color="#2980b9"))
    fig.update_layout(
        title="La violencia vive donde hay flujo físico y poca protección<br>"
              "<sup>x = policías estatales por mil hab (CNSPE, dato 2024; log) · y = "
              "homicidio doloso por 100 mil (2023-25) · líneas = medianas · descriptivo, "
              "n=32</sup>",
        xaxis_title="policías estatales por 1,000 hab (incluye auxiliar/bancaria en CDMX)",
        xaxis_type="log",
        yaxis_title="homicidios dolosos por 100 mil hab")
    guardar_fig(fig, "cap12_renta_capacidad")
    return b


# ------------------------------------------------- figura A

def figura_capacidad(serie: pl.DataFrame, prev: pl.DataFrame):
    fig = make_subplots(cols=2, column_widths=[0.5, 0.5], horizontal_spacing=0.12)
    p = prev.with_columns(pl.col("cve_ent").replace_strict(CORTO).alias("estado")) \
        .sort("prev_pc")
    fig.add_trace(go.Bar(
        x=p["prev_pc"], y=p["estado"], orientation="h", showlegend=False,
        marker_color=["#c0392b" if c == CDMX else "#95a5a6" for c in p["cve_ent"]]),
        row=1, col=1)
    fig.add_vline(x=1.8, line_dash="dash", line_color="#2c3e50", row=1, col=1,
                  annotation_text="estándar 1.8", annotation_position="bottom right",
                  annotation_font_size=10)
    años = sorted(serie["año"].unique())
    nac = [serie.filter(pl.col("año") == a)["fuerza"].sum()
           / serie.filter(pl.col("año") == a)["pob_total"].sum() * 1000 for a in años]
    cdmx = [serie.filter((pl.col("año") == a) & (pl.col("cve_ent") == CDMX))["pc"][0]
            for a in años]
    mediana = [serie.filter(pl.col("año") == a)["pc"].median() for a in años]
    for y, nombre, color, ancho in [(cdmx, "CDMX", "#c0392b", 3),
                                    (nac, "nacional", "#2c3e50", 2),
                                    (mediana, "mediana estatal", "#95a5a6", 2)]:
        fig.add_trace(go.Scatter(x=años, y=y, name=nombre,
                                 line=dict(color=color, width=ancho)), row=1, col=2)
    fig.update_layout(
        title="La protección está donde está el centro: capacidad policial estatal<br>"
              "<sup>izq: policía preventiva operativa (SESNSP-MOFP 2020) · der: todo el "
              "personal de la institución estatal de seguridad<br>(CNGSPSPE/CNSPE; sin "
              "dato 2019; CDMX incluye Policía Auxiliar y Bancaria-Industrial)</sup>",
        margin=dict(t=90), yaxis=dict(tickfont=dict(size=9)),
        legend=dict(orientation="h", y=-0.1))
    fig.update_xaxes(title_text="preventiva operativa /1000 (jun-2020)",
                     title_font_size=12, row=1, col=1)
    fig.update_xaxes(title_text="personal estatal /1000 (2016-2024)",
                     title_font_size=12, row=1, col=2)
    guardar_fig(fig, "cap12_policia_pc", alto=700)


def main():
    pob = cargar_poblacion()
    serie = fuerza_estatal(pob)
    prev = mofp(pob)
    figura_capacidad(serie, prev)
    zmvm_capacidad(pob, serie, prev)
    renta_capacidad(pob, serie)


if __name__ == "__main__":
    main()
