"""
Cap 12 — La paradoja de la capital (H7): renta máxima, violencia contenida.

Objeción a testear: si la violencia es disputa por rentas (Caps 9-10) y colapso de la
regulación centralizada (Trejo-Ley, casos_de_estudio §1.2), la CDMX — máximo flujo de
capitales y divisas — debería exhibir índices iguales o mayores de violencia. No lo hace.

  A. El hecho: tasa de homicidio doloso CDMX vs nacional 2015-2025; residual de CDMX
     en la regresión flujo de cuota pc → homicidios (Cap 10 T3).
  B. E4 medición (primero, valida el hecho): SESNSP (carpetas) vs INEGI defunciones
     (X85-Y09, ocurrencia) — ¿subregistro diferencial en CDMX?; desaparecidos pc.
  C. E1 tipo de renta: share CDMX del VA financiero/corporativo (PIBE_36/39) vs rentas
     físicas disputables (frontera/puerto/mina EIMM = 0; tomas clandestinas CartoCrítica).
  D. E2 frontera administrativa: alcaldías CDMX vs núcleo conurbado del Edomex (misma
     zona metropolitana y mismas rentas de flujo, distinta jurisdicción).
  E. E3 firma de renta administrada: extorsión y narcomenudeo pc vs homicidio pc por
     estado; robustez con ENVIPE (prevalencia y tasa de denuncia).

Figuras → centralismo/informe/figuras/cap12_*.png
Run: uv run python scripts/centralismo/cap12_paradoja_capital.py
"""

import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from comun import (CORTO, RAIZ, cargar_envipe, cargar_poblacion, guardar_fig,
                   leer_pibe, normalizar_estado)
from cap9_tomas_clandestinas import cargar_tomas

MESES = ["Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio", "Julio",
         "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"]
CDMX, EDOMEX = 9, 15
PARQUET_INC = (RAIZ / "data/incidencia_delictiva/incidencia_fuero_comun/"
               "incidencia_delictiva_fuero_comun.parquet")

# Núcleo urbano conurbado de la ZMVM en el Edomex: municipios contiguos a la CDMX
# con ≥100 mil hab (censo 2020). Se resuelven por nombre contra SESNSP (assert).
CONURBADOS_EDOMEX = [
    "Ecatepec de Morelos", "Nezahualcóyotl", "Naucalpan de Juárez", "Chimalhuacán",
    "Tlalnepantla de Baz", "Cuautitlán Izcalli", "Ixtapaluca", "Atizapán de Zaragoza",
    "Tultitlán", "Chalco", "Coacalco de Berriozábal", "Valle de Chalco Solidaridad",
    "Nicolás Romero", "Tecámac", "La Paz", "Huixquilucan", "Chicoloapan", "Texcoco",
    "Zumpango", "Cuautitlán", "Tultepec", "Huehuetoca", "Tepotzotlán", "Acolman",
]


def casos_estatales(filtro: pl.Expr, años: list[int]) -> pl.DataFrame:
    """SESNSP → cve_ent, año, casos (para el filtro de delito dado)."""
    return (pl.scan_parquet(PARQUET_INC)
            .filter(pl.col("Año").is_in(años) & filtro)
            .with_columns(pl.sum_horizontal(MESES).alias("casos"))
            .group_by("Clave_Ent", "Año").agg(pl.sum("casos"))
            .collect()
            .rename({"Clave_Ent": "cve_ent", "Año": "año"})
            .with_columns(pl.col("cve_ent").cast(pl.Int64)))


def tasas_estatales(filtro: pl.Expr, años: list[int], pob: pl.DataFrame) -> pl.DataFrame:
    """Tasa por 100k promedio de los años pedidos, por estado."""
    c = (casos_estatales(filtro, años).join(pob, on=["cve_ent", "año"])
         .group_by("cve_ent")
         .agg((pl.sum("casos") / pl.sum("pob_total") * 1e5).alias("tasa")))
    return c.with_columns(pl.col("cve_ent").replace_strict(CORTO).alias("estado"))


HOMICIDIO = pl.col("Subtipo de delito") == "Homicidio doloso"
EXTORSION = pl.col("Tipo de delito") == "Extorsión"
NARCOMENUDEO = pl.col("Tipo de delito") == "Narcomenudeo"


# ---------------------------------------------------------------- A. el hecho

def hecho(pob: pl.DataFrame) -> pl.DataFrame:
    print("=== A. El hecho: ¿es la CDMX anómalamente pacífica para su renta? ===")
    serie = (casos_estatales(HOMICIDIO, list(range(2015, 2026)))
             .join(pob, on=["cve_ent", "año"])
             .with_columns((pl.col("casos") / pl.col("pob_total") * 1e5).alias("tasa")))
    nac = (serie.group_by("año")
           .agg((pl.sum("casos") / pl.sum("pob_total") * 1e5).alias("tasa")).sort("año"))
    cdmx = serie.filter(pl.col("cve_ent") == CDMX).sort("año")
    p = (serie.group_by("año").agg(
        pl.col("tasa").quantile(0.25).alias("p25"),
        pl.col("tasa").quantile(0.75).alias("p75")).sort("año"))

    t24 = serie.filter(pl.col("año") == 2024).sort("tasa", descending=True)
    rango = t24.with_row_index("rank").filter(pl.col("cve_ent") == CDMX)
    print(f"  tasa 2024: CDMX={rango['tasa'][0]:.1f} · nacional="
          f"{nac.filter(pl.col('año') == 2024)['tasa'][0]:.1f} · "
          f"lugar {rango['rank'][0] + 1}/32 (1 = más violento)")
    c15 = cdmx.filter(pl.col("año") == 2015)["tasa"][0]
    pico = cdmx.sort("tasa", descending=True).head(1)
    print(f"  serie CDMX: 2015={c15:.1f} · pico {pico['año'][0]}={pico['tasa'][0]:.1f} · "
          f"2025={cdmx.filter(pl.col('año') == 2025)['tasa'][0]:.1f}")
    bajo_p25 = (cdmx.join(p, on="año")
                .filter(pl.col("tasa") < pl.col("p25")).height)
    print(f"  años (de 11) con CDMX por debajo del p25 estatal: {bajo_p25}")
    ranks = (serie.with_columns(pl.col("tasa").rank("ordinal", descending=True)
                                .over("año").alias("rank"))
             .filter(pl.col("cve_ent") == CDMX))
    print(f"  lugar anual de CDMX 2015-2025: entre {ranks['rank'].min()} y "
          f"{ranks['rank'].max()} de 32")

    # residual de CDMX en flujo de cuota pc → homicidios (medida del Cap 10 T3)
    cuota = pl.read_csv(RAIZ / "data/tollbooths/growth_rate_car_2021_2025.csv",
                        infer_schema_length=0)
    mapa = {s: normalizar_estado(s) for s in cuota["state"].drop_nulls().unique().to_list()}
    flujo = (cuota.with_columns(
        pl.col("tdpa_round_2024").cast(pl.Float64, strict=False).alias("tdpa"),
        pl.col("state").replace_strict(mapa, default=None).alias("cve_ent"))
        .filter(pl.col("cve_ent").is_not_null() & (pl.col("tdpa") > 0))
        .group_by("cve_ent").agg(pl.sum("tdpa")))
    b = (flujo.join(pob.filter(pl.col("año") == 2024), on="cve_ent")
         .join(t24.select("cve_ent", "tasa"), on="cve_ent")
         .with_columns((pl.col("tdpa") / pl.col("pob_total") * 1000).alias("flujo_pc")))
    x, y = b["flujo_pc"].to_numpy(), b["tasa"].to_numpy()
    beta, alfa = np.polyfit(x, y, 1)
    res = y - (alfa + beta * x)
    orden = np.argsort(res)
    fila = b.with_columns(pl.Series("residual", res)).filter(pl.col("cve_ent") == CDMX)
    print(f"  flujo de cuota pc 2024: CDMX={fila['flujo_pc'][0]:.0f} cruces/día por mil hab "
          f"(lugar {int(np.where(np.argsort(-x) == list(b['cve_ent']).index(CDMX))[0][0]) + 1}"
          f"/{b.height} de los que reportan)")
    print(f"  residual de CDMX en OLS tasa~flujo: {fila['residual'][0]:+.1f} por 100k "
          f"(lugar {int(np.where(orden == list(b['cve_ent']).index(CDMX))[0][0]) + 1}"
          f"/{b.height} más negativo)")
    return serie, nac, cdmx, p


# ---------------------------------------------------------------- B. E4 medición

def e4_medicion(pob: pl.DataFrame, serie: pl.DataFrame):
    print("\n=== B. E4 — ¿Es artefacto de medición? ===")
    # INEGI defunciones: homicidios (agresiones X85-Y09) por entidad de OCURRENCIA,
    # años de ocurrencia 2022-2023, leídos de los registros 2022-2024 (los registros
    # tardíos de un año caen en el archivo del siguiente).
    archivos = {
        2022: "conjunto_de_datos_defunciones_registradas_2022_csv.zip",
        2023: "conjunto_de_datos_defunciones_registradas_2023_csv.zip",
        2024: "conjunto_de_datos_edr2024_csv.zip",
    }
    partes = []
    for zname in archivos.values():
        z = zipfile.ZipFile(RAIZ / "data/inegi/defunciones" / zname)
        main = [n for n in z.namelist()
                if "conjunto_de_datos/" in n and n.lower().endswith(".csv")][0]
        df = (pl.read_csv(io.BytesIO(z.read(main)), infer_schema_length=0,
                          columns=["ent_ocurr", "causa_def", "anio_ocur"])
              .filter(pl.col("causa_def").str.contains(r"^(X8[5-9]|X9|Y0)"))
              .with_columns(pl.col("ent_ocurr").cast(pl.Int64, strict=False),
                            pl.col("anio_ocur").cast(pl.Int64, strict=False)))
        partes.append(df)
    inegi = (pl.concat(partes)
             .filter(pl.col("anio_ocur").is_in([2022, 2023])
                     & pl.col("ent_ocurr").is_between(1, 32))
             .group_by("ent_ocurr").agg(pl.len().alias("inegi"))
             .rename({"ent_ocurr": "cve_ent"}))
    sesnsp = (serie.filter(pl.col("año").is_in([2022, 2023]))
              .group_by("cve_ent").agg(pl.sum("casos").alias("sesnsp")))
    cmp = inegi.join(sesnsp, on="cve_ent").with_columns(
        (pl.col("inegi") / pl.col("sesnsp")).alias("razon"))
    nac = cmp["inegi"].sum() / cmp["sesnsp"].sum()
    fila = cmp.filter(pl.col("cve_ent") == CDMX)
    print(f"  víctimas INEGI (X85-Y09, ocurrencia 2022-23) ÷ carpetas SESNSP:")
    print(f"    nacional = {nac:.2f} · CDMX = {fila['razon'][0]:.2f} "
          f"(INEGI {fila['inegi'][0]:,} vs SESNSP {fila['sesnsp'][0]:,})")
    mediana = cmp["razon"].median()
    print(f"    mediana estatal = {mediana:.2f} → el subregistro de CDMX "
          f"{'NO ' if fila['razon'][0] <= mediana + 0.15 else ''}es atípico")

    des = (pl.read_csv(RAIZ / "data/datamx/desaparecidos/desaparecidos.csv",
                       infer_schema_length=0)
           .with_columns(pl.col("CVE_ENT").cast(pl.Int64, strict=False).alias("cve_ent"))
           .filter(pl.col("cve_ent").is_between(1, 32))
           .group_by("cve_ent").agg(pl.len().alias("des"))
           .join(pob.filter(pl.col("año") == 2024), on="cve_ent")
           .with_columns((pl.col("des") / pl.col("pob_total") * 1e5).alias("des_pc"))
           .sort("des_pc", descending=True).with_row_index("rank"))
    f = des.filter(pl.col("cve_ent") == CDMX)
    print(f"  desaparecidos acumulados: CDMX={f['des_pc'][0]:.0f}/100k "
          f"(lugar {f['rank'][0] + 1}/32) — sin marcador de guerra territorial")
    return cmp


# ---------------------------------------------------------------- C. E1 tipo de renta

def e1_renta(pob: pl.DataFrame):
    print("\n=== C. E1 — El tipo de renta: líquida y legible, no disputable ===")
    shares = {}
    for etiqueta, n in [("PIB total", 2), ("Serv. financieros y seguros", 36),
                        ("Corporativos", 39), ("Act. gubernamentales", 46)]:
        v = leer_pibe(n, bloque="Millones de pesos")  # corrientes
        v24 = v.filter(pl.col("año") == 2024)
        shares[etiqueta] = v24.filter(pl.col("cve_ent") == CDMX)["valor"][0] / v24["valor"].sum()
    p24 = pob.filter(pl.col("año") == 2024)
    shares["Población"] = (p24.filter(pl.col("cve_ent") == CDMX)["pob_total"][0]
                           / p24["pob_total"].sum())
    for k, v in shares.items():
        print(f"  share CDMX 2024 · {k:<28} {v * 100:5.1f}%")

    tomas = cargar_tomas()
    n_cdmx = int((tomas["cve_ent"] == CDMX).sum())
    print(f"  rentas físicas disputables en CDMX: 0 cruces fronterizos · 0 puertos de carga "
          f"· 0 municipios EIMM (catálogos Cap 9)")
    print(f"  tomas clandestinas 2008-2016 en CDMX: {n_cdmx} de {len(tomas):,} "
          f"({n_cdmx / len(tomas) * 100:.1f}%)")

    orden = ["Serv. financieros y seguros", "Corporativos", "Act. gubernamentales",
             "PIB total", "Población"]
    fig = go.Figure(go.Bar(
        x=[shares[k] * 100 for k in orden], y=orden, orientation="h",
        marker_color=["#c0392b", "#c0392b", "#e67e22", "#7f8c8d", "#95a5a6"],
        text=[f"{shares[k] * 100:.1f}%" for k in orden], textposition="outside"))
    fig.add_annotation(
        x=max(shares.values()) * 100, y=-0.9, showarrow=False, xanchor="right",
        text=("rentas físicas disputables en CDMX: 0 frontera · 0 puerto · 0 mina EIMM · "
              f"{n_cdmx / len(tomas) * 100:.1f}% de las tomas clandestinas"),
        font=dict(size=12, color="#2c3e50"))
    fig.update_layout(
        title="La renta de la CDMX es líquida y contable, no de flujo físico disputable<br>"
              "<sup>share de la CDMX en el total nacional, 2024 (PIBE corrientes, INEGI) "
              "vs sus rentas territoriales</sup>",
        xaxis_title="% del total nacional", yaxis=dict(autorange="reversed"),
        margin=dict(b=90))
    guardar_fig(fig, "cap12_composicion_renta")
    return shares


# ---------------------------------------------------------------- D. E2 frontera ZMVM

def cves_conurbados() -> set[int]:
    """CONURBADOS_EDOMEX resueltos por nombre contra el padrón municipal SESNSP."""
    import unicodedata
    def clave(s):
        return unicodedata.normalize("NFD", str(s)).encode("ascii", "ignore").decode().lower().strip()

    munis = (pl.scan_parquet(PARQUET_INC)
             .filter(pl.col("Clave_Ent") == EDOMEX)
             .select("Cve. Municipio", "Municipio").unique().collect()
             .with_columns(pl.col("Municipio")
                           .map_elements(clave, return_dtype=pl.Utf8).alias("mun_clave")))
    cves = set()
    for nombre in CONURBADOS_EDOMEX:
        m = munis.filter(pl.col("mun_clave") == clave(nombre))
        assert m.height == 1, f"conurbado no resuelto: {nombre!r} ({m.height})"
        cves.add(int(m["Cve. Municipio"][0]))
    return cves


def carga_mun_zmvm(filtro: pl.Expr, nombre: str) -> pl.DataFrame:
    """Casos anuales promedio 2023-2024 por municipio (para el test ZMVM)."""
    return (pl.scan_parquet(PARQUET_INC)
            .filter(pl.col("Año").is_in([2023, 2024]) & filtro)
            .with_columns(pl.sum_horizontal(MESES).alias("casos"))
            .group_by("Cve. Municipio", "Municipio", "Clave_Ent")
            .agg((pl.sum("casos") / 2).alias(nombre))
            .collect())


def e2_zmvm():
    print("\n=== D. E2 — Frontera administrativa: alcaldías CDMX vs Edomex conurbado ===")
    hom = carga_mun_zmvm(HOMICIDIO, "hom")
    ext = carga_mun_zmvm(EXTORSION, "ext").select("Cve. Municipio", "ext")
    pob = (pl.read_csv(RAIZ / "data/conapo/municipios_2020_todos.csv")
           .select(pl.col("CLAVE").cast(pl.Int64).alias("Cve. Municipio"),
                   pl.col("POB_TOTAL").cast(pl.Float64).alias("pob")))
    cves_con = cves_conurbados()

    base = (hom.join(ext, on="Cve. Municipio", how="left")
            .join(pob, on="Cve. Municipio", how="inner")
            .with_columns(pl.col("ext").fill_null(0)))
    lados = {
        "Alcaldías CDMX (16)": base.filter(pl.col("Clave_Ent") == CDMX),
        f"Edomex conurbado ({len(cves_con)})":
            base.filter(pl.col("Cve. Municipio").is_in(list(cves_con))),
    }
    tasas = {}
    for lado, d in lados.items():
        th = d["hom"].sum() / d["pob"].sum() * 1e5
        te = d["ext"].sum() / d["pob"].sum() * 1e5
        tasas[lado] = (th, te, d["pob"].sum())
        print(f"  {lado:<24} pob={d['pob'].sum() / 1e6:.1f}M · "
              f"homicidio={th:.1f}/100k · extorsión={te:.1f}/100k")
    razon = tasas[list(tasas)[1]][0] / tasas[list(tasas)[0]][0]
    razon_ext = tasas[list(tasas)[1]][1] / tasas[list(tasas)[0]][1]
    print(f"  razón de tasas (Edomex conurbado ÷ alcaldías): homicidio = {razon:.2f}× · "
          f"extorsión = {razon_ext:.2f}×")
    for lado, d in lados.items():
        top = (d.with_columns((pl.col("hom") / pl.col("pob") * 1e5).alias("tasa"))
               .sort("tasa", descending=True).head(1))
        print(f"  máximo en {lado}: {top['Municipio'][0]} = {top['tasa'][0]:.1f}/100k")

    detalle = (pl.concat([d.with_columns(pl.lit(lado).alias("lado"))
                          for lado, d in lados.items()])
               .with_columns((pl.col("hom") / pl.col("pob") * 1e5).alias("tasa"))
               .sort("tasa", descending=True))
    fig = go.Figure()
    for lado, color in zip(lados, ["#2980b9", "#c0392b"]):
        d = detalle.filter(pl.col("lado") == lado)
        fig.add_trace(go.Bar(x=d["tasa"], y=d["Municipio"], orientation="h",
                             name=lado, marker_color=color))
    fig.update_layout(
        barmode="overlay",
        title=f"Misma metrópoli, distinta jurisdicción: homicidio doloso 2023-2024 "
              f"(razón {razon:.1f}×)<br><sup>alcaldías de la CDMX vs núcleo conurbado del "
              f"Edomex (≥100 mil hab) — la renta de la ZMVM es compartida; la capacidad "
              f"institucional no</sup>",
        xaxis_title="homicidios dolosos por 100 mil hab (promedio 2023-2024)",
        yaxis=dict(tickfont=dict(size=9), autorange="reversed"),
        legend=dict(orientation="h", y=-0.08))
    guardar_fig(fig, "cap12_zmvm_frontera", alto=850)
    return tasas


# ---------------------------------------------------------------- E. E3 firma

def e3_firma(pob: pl.DataFrame):
    print("\n=== E. E3 — Firma de renta administrada: extorsión sin guerra ===")
    años = [2023, 2024, 2025]
    th = tasas_estatales(HOMICIDIO, años, pob).rename({"tasa": "hom"})
    te = tasas_estatales(EXTORSION, años, pob).rename({"tasa": "ext"}).drop("estado")
    tn = tasas_estatales(NARCOMENUDEO, años, pob).rename({"tasa": "narco"}).drop("estado")
    b = th.join(te, on="cve_ent").join(tn, on="cve_ent")

    for col, nombre in [("ext", "extorsión"), ("narco", "narcomenudeo")]:
        d = b.sort(col, descending=True).with_row_index("rank")
        f = d.filter(pl.col("cve_ent") == CDMX)
        nac = b[col].mean()
        print(f"  {nombre} pc (denunciada, prom. 2023-25): CDMX={f[col][0]:.1f}/100k "
              f"(lugar {f['rank'][0] + 1}/32; media estatal {nac:.1f})")
    f = b.sort("hom", descending=True).with_row_index("rank").filter(pl.col("cve_ent") == CDMX)
    print(f"  homicidio pc: CDMX={f['hom'][0]:.1f}/100k (lugar {f['rank'][0] + 1}/32)")
    fm = b.filter(pl.col("cve_ent") == EDOMEX)
    top_ext = b.sort("ext", descending=True).head(1)
    print(f"  contraste Edomex: extorsión={fm['ext'][0]:.1f}/100k "
          f"(2° nacional, tras {top_ext['estado'][0]}={top_ext['ext'][0]:.1f}) · "
          f"homicidio={fm['hom'][0]:.1f}/100k")
    print(f"  cuadrante opuesto — {top_ext['estado'][0]}: "
          f"homicidio={top_ext['hom'][0]:.1f} · extorsión={top_ext['ext'][0]:.1f} "
          f"(máximo nacional en ambas: renta física del puerto en disputa)")

    env = cargar_envipe().filter(pl.col("año") >= 2023)
    e = env.group_by("cve_ent").agg(pl.mean("vic_envipe"), pl.mean("rep_rate_envipe"))
    fc = e.filter(pl.col("cve_ent") == CDMX)
    rv = e.sort("vic_envipe", descending=True).with_row_index("rank").filter(
        pl.col("cve_ent") == CDMX)["rank"][0] + 1
    print(f"  ENVIPE 2023-25: victimización CDMX={fc['vic_envipe'][0] * 100:.1f}% "
          f"(lugar {rv}/32) · tasa de denuncia CDMX={fc['rep_rate_envipe'][0] * 100:.1f}% "
          f"vs media {e['rep_rate_envipe'].mean() * 100:.1f}%")

    mx, my = b["ext"].median(), b["hom"].median()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=b["ext"], y=b["hom"], mode="markers+text", text=b["estado"],
        textposition="top center", textfont=dict(size=9), showlegend=False,
        marker=dict(size=10, color=["#c0392b" if c == CDMX else "#7f8c8d"
                                    for c in b["cve_ent"]])))
    fig.add_vline(x=mx, line_dash="dot", line_color="#95a5a6")
    fig.add_hline(y=my, line_dash="dot", line_color="#95a5a6")
    fig.add_annotation(x=b["ext"].max(), y=b["hom"].min(), xanchor="right",
                       text="renta administrada:<br>extorsión alta, guerra baja",
                       showarrow=False, font=dict(size=11, color="#c0392b"))
    fig.update_layout(
        title="La firma de la plaza administrada: extorsión sin guerra<br>"
              "<sup>tasas SESNSP prom. 2023-2025 (denunciadas; cifra negra de extorsión "
              ">95% ENVIPE) · líneas = medianas estatales</sup>",
        xaxis_title="extorsión por 100 mil hab", yaxis_title="homicidio doloso por 100 mil hab")
    guardar_fig(fig, "cap12_extorsion_firma")
    return b


# ---------------------------------------------------------------- figura serie

def figura_series(nac, cdmx, p):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(p["año"]) + list(p["año"])[::-1],
        y=list(p["p75"]) + list(p["p25"])[::-1],
        fill="toself", fillcolor="rgba(127,140,141,0.15)",
        line=dict(width=0), name="rango p25-p75 estatal"))
    fig.add_trace(go.Scatter(x=nac["año"], y=nac["tasa"], name="nacional",
                             line=dict(color="#2c3e50", width=2)))
    fig.add_trace(go.Scatter(x=cdmx["año"], y=cdmx["tasa"], name="CDMX",
                             line=dict(color="#c0392b", width=3)))
    fig.update_layout(
        title="La capital vive por debajo del país que administra<br>"
              "<sup>tasa de homicidio doloso por 100 mil hab (SESNSP 2015-2025); "
              "banda = percentiles 25-75 estatales</sup>",
        xaxis_title="año", yaxis_title="homicidios dolosos por 100 mil hab",
        legend=dict(orientation="h", y=1.02, x=0))
    guardar_fig(fig, "cap12_tasas_series")


def main():
    pob = cargar_poblacion()
    serie, nac, cdmx, p = hecho(pob)
    e4_medicion(pob, serie)
    figura_series(nac, cdmx, p)
    e1_renta(pob)
    e2_zmvm()
    e3_firma(pob)


if __name__ == "__main__":
    main()
