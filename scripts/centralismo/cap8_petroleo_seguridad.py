"""
Cap 8.C — Petróleo: Campeche vs Tabasco — ¿déficit de vigilancia o naturaleza de la renta?

El Cap 1 dejó el petróleo como la única "extracción literal" (la renta de Campeche/
Tabasco se nacionaliza); los Caps 9 y 12 establecen que la renta de recursos genera
violencia donde es físicamente disputable. El par petrolero exhibe la anomalía:
Campeche es de los estados menos violentos y Tabasco diverge al alza. Hipótesis a
testear: la disparidad se debe a menor vigilancia policial/militar (¿déficit de
elementos en Campeche, que tiene menos población?).

  A. El hecho: divergencia de homicidio doloso, extorsión y percepción 2015-2025.
  B. Test de la hipótesis: policías estatales pc (serie del Cap 12.B), preventiva
     MOFP, y gasto militar federal localizado por estado (ramo 13 SEMAR como proxy
     de custodia; el ramo 7 es obra localizada, no despliegue).
  C. Hipótesis alternativa (el marco del Cap 12): naturaleza de la renta — offshore
     custodiada e inaccesible (0 tomas) vs onshore disputable (ductos, tomas).

Notas: fuerza_estatal()/mofp() (Cap 12.B) imprimen su narrativa CDMX al llamarse —
aceptado por precedente (16_anexo_equilibrio_de_fuerzas). La columna inseg_envipe
del panel ENVIPE es la fracción que se siente SEGURA en su municipio (AP4_3_2==1,
pese al nombre); aquí se reporta 1−x.

Figuras → centralismo/informe/figuras/cap8_petroleo_{divergencia,capacidad,renta}.png
Run: uv run python scripts/centralismo/cap8_petroleo_seguridad.py
"""

import sys
from pathlib import Path

import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).resolve().parent))
from comun import (PIBE_MINERIA_PETROLERA, PIBE_TOTAL, RAIZ, cargar_envipe,
                   cargar_poblacion, guardar_fig, leer_pibe)
from cap12_capacidad_policial import fuerza_estatal, mofp
from cap12_paradoja_capital import (EXTORSION, HOMICIDIO, casos_estatales,
                                    tasas_estatales)
from cap9_tomas_clandestinas import cargar_tomas

CAMPECHE, TABASCO = 4, 27
AZUL, ROJO, OSCURO, GRIS = "#2980b9", "#c0392b", "#2c3e50", "#95a5a6"


def cargar_cp() -> pl.DataFrame:
    return pl.read_parquet(RAIZ / "informe_data/cp_estado_ramo.parquet")


def _v(df: pl.DataFrame, cve: int, col: str) -> float:
    return df.filter(pl.col("cve_ent") == cve)[col][0]


# ------------------------------------------------- A. el hecho: la divergencia

def seccion_a_divergencia(pob: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    print("=== 8.C-A. El hecho: divergencia Campeche-Tabasco ===")
    hom = (casos_estatales(HOMICIDIO, list(range(2015, 2026)))
           .join(pob, on=["cve_ent", "año"])
           .with_columns((pl.col("casos") / pl.col("pob_total") * 1e5).alias("tasa")))
    assert hom.height == 32 * 11, f"panel de homicidios incompleto: {hom.height}"
    print("  homicidio doloso /100k (SESNSP fuero común):")
    for año in range(2015, 2026):
        d = hom.filter(pl.col("año") == año)
        c, t = _v(d, CAMPECHE, "tasa"), _v(d, TABASCO, "tasa")
        nac = d["casos"].sum() / d["pob_total"].sum() * 1e5
        print(f"    {año}: Campeche={c:5.1f} · Tabasco={t:5.1f} ({t / c:.1f}×) · "
              f"nacional={nac:5.1f}")
    cc = hom.filter(pl.col("cve_ent") == CAMPECHE)
    c15, c25 = [cc.filter(pl.col("año") == a)["casos"][0] for a in (2015, 2025)]
    t24 = hom.filter((pl.col("cve_ent") == TABASCO) & (pl.col("año") == 2024))
    print(f"  pico de Tabasco 2024: {t24['casos'][0]:,} carpetas ({t24['tasa'][0]:.1f}/100k)")
    print(f"  honestidad: Campeche no es idílico — casos {c15:.0f} (2015) → {c25:.0f} "
          f"(2025), ×{c25 / c15:.1f} en la década; el contraste es de nivel y pendiente")

    ext = tasas_estatales(EXTORSION, [2023, 2024, 2025], pob)
    print(f"  extorsión /100k 2023-25 (delito-firma de la disputa, Cap 12 E3): "
          f"Campeche={_v(ext, CAMPECHE, 'tasa'):.1f} · Tabasco={_v(ext, TABASCO, 'tasa'):.1f} "
          f"· mediana={ext['tasa'].median():.1f}")

    env = (cargar_envipe()
           .with_columns(((1 - pl.col("inseg_envipe")) * 100).alias("pct_inseg"))
           .join(pob, on=["cve_ent", "año"]))
    e24 = env.filter(pl.col("año") == 2024)
    nac24 = ((e24["pct_inseg"] * e24["pob_total"]).sum() / e24["pob_total"].sum())
    print(f"  percepción de inseguridad municipal ENVIPE 2024 (1−inseg_envipe; la "
          f"columna trae la fracción que se siente SEGURA): "
          f"Campeche={_v(e24, CAMPECHE, 'pct_inseg'):.0f}% · "
          f"Tabasco={_v(e24, TABASCO, 'pct_inseg'):.0f}% · nacional={nac24:.0f}%")
    return hom, env


def figura_divergencia(hom: pl.DataFrame, env: pl.DataFrame) -> go.Figure:
    fig = make_subplots(cols=2, horizontal_spacing=0.1,
                        subplot_titles=("homicidio doloso por 100 mil hab",
                                        "% que se siente inseguro en su municipio"))
    series = [(CAMPECHE, "Campeche", AZUL), (TABASCO, "Tabasco", ROJO)]
    años_h = sorted(hom["año"].unique().to_list())
    for cve, nombre, color in series:
        d = hom.filter(pl.col("cve_ent") == cve).sort("año")
        fig.add_trace(go.Scatter(x=d["año"], y=d["tasa"], name=nombre,
                                 line=dict(color=color, width=3)), row=1, col=1)
    nac_h = [hom.filter(pl.col("año") == a) for a in años_h]
    fig.add_trace(go.Scatter(
        x=años_h, y=[d["casos"].sum() / d["pob_total"].sum() * 1e5 for d in nac_h],
        name="nacional", line=dict(color=OSCURO, width=2, dash="dot")), row=1, col=1)
    t24 = hom.filter((pl.col("cve_ent") == TABASCO) & (pl.col("año") == 2024))
    fig.add_annotation(x=2024, y=t24["tasa"][0], row=1, col=1, ax=-55, ay=-5,
                       text=f"2024: {t24['casos'][0]:,} carpetas", font=dict(size=10))

    años_e = sorted(env["año"].unique().to_list())
    for cve, nombre, color in series:
        d = env.filter(pl.col("cve_ent") == cve).sort("año")
        fig.add_trace(go.Scatter(x=d["año"], y=d["pct_inseg"], showlegend=False,
                                 name=nombre, line=dict(color=color, width=3)),
                      row=1, col=2)
    nac_e = [env.filter(pl.col("año") == a) for a in años_e]
    fig.add_trace(go.Scatter(
        x=años_e, showlegend=False, name="nacional",
        y=[(d["pct_inseg"] * d["pob_total"]).sum() / d["pob_total"].sum() for d in nac_e],
        line=dict(color=OSCURO, width=2, dash="dot")), row=1, col=2)
    fig.update_layout(
        title="El par petrolero diverge: misma renta extraída, opuesta seguridad<br>"
              "<sup>izq: SESNSP fuero común 2015-2025 · der: ENVIPE 2017-2025, ponderado "
              "por población el nacional</sup>",
        legend=dict(orientation="h", y=-0.12))
    guardar_fig(fig, "cap8_petroleo_divergencia")
    return fig


# ------------------------------------------------- B. test: ¿déficit de vigilancia?

def seccion_b_vigilancia(pob: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    serie_pol = fuerza_estatal(pob)
    prev = mofp(pob)

    print("\n=== 8.C-B. ¿Déficit de vigilancia? El par petrolero ===")
    print("  policías estatales /1000 hab (serie del Cap 12.B):")
    for año in sorted(serie_pol["año"].unique().to_list()):
        d = serie_pol.filter(pl.col("año") == año)
        print(f"    {año}: Campeche={_v(d, CAMPECHE, 'pc'):.2f} · "
              f"Tabasco={_v(d, TABASCO, 'pc'):.2f} · mediana={d['pc'].median():.2f}")
    for cve, nombre in [(CAMPECHE, "Campeche"), (TABASCO, "Tabasco")]:
        f = prev.filter(pl.col("cve_ent") == cve)
        print(f"  MOFP jun-2020 {nombre}: {f['oper'][0]:,.0f} preventivos operativos = "
              f"{f['prev_pc'][0]:.2f}/1000 (estándar 1.8) · "
              f"CUP {f['cup'][0] / f['oper'][0] * 100:.0f}%")
    print("  → la hipótesis del déficit civil predice lo contrario de lo observado: "
          "Campeche tiene MENOS policía pc que Tabasco y menos violencia")

    cp = cargar_cp()
    print("  gasto federal de seguridad localizado por estado (Cuenta Pública, ejercido "
          "2016-2025, % distribuido a cve_ent 1-32):")
    for ramo, nombre in [(7, "SEDENA"), (13, "SEMAR"), (36, "SSPC/GN")]:
        d = cp.filter((pl.col("id_ramo") == ramo) & pl.col("ciclo").is_between(2016, 2025))
        dist = (d.filter(pl.col("cve_ent").is_between(1, 32))["monto_ejercido"].sum()
                / d["monto_ejercido"].sum())
        print(f"    ramo {ramo:>2} ({nombre}): {dist * 100:5.1f}% distribuido"
              + (" (existe desde 2019)" if ramo == 36 else ""))
        if ramo == 13:
            assert dist >= 0.98, f"ramo 13 poco distribuido: {dist:.3f}"

    semar = (cp.filter((pl.col("id_ramo") == 13) & pl.col("ciclo").is_between(2016, 2025)
                       & pl.col("cve_ent").is_between(1, 32))
             .group_by("ciclo", "cve_ent").agg(pl.sum("monto_ejercido").alias("ejercido")))
    años = sorted(semar["ciclo"].unique().to_list())
    grid = pl.DataFrame({"ciclo": [a for a in años for _ in (0, 1)],
                         "cve_ent": [c for _ in años for c in (CAMPECHE, TABASCO)]})
    semar_pc = (grid.join(semar, on=["ciclo", "cve_ent"], how="left")
                .with_columns(pl.col("ejercido").fill_null(0.0))
                .join(pob.rename({"año": "ciclo"}), on=["ciclo", "cve_ent"])
                .with_columns((pl.col("ejercido") / pl.col("pob_total")).alias("pc")))
    print("  SEMAR (ramo 13) ejercido per cápita, MXN — la corporación que custodia la "
          "sonda:")
    for año in años:
        d = semar_pc.filter(pl.col("ciclo") == año)
        n = semar.filter(pl.col("ciclo") == año)["ejercido"].sum()
        pn = pob.filter((pl.col("año") == año) & pl.col("cve_ent").is_between(1, 32))
        print(f"    {año}: Campeche={_v(d, CAMPECHE, 'pc'):7,.0f} · "
              f"Tabasco={_v(d, TABASCO, 'pc'):7,.0f} · nacional={n / pn['pob_total'].sum():5,.0f}")
    print("  guarda de endogeneidad (Cap 12): el salto SEMAR de Tabasco en 2024-25 es "
          "POSTERIOR al pico de violencia 2024 — despliegue reactivo, no protección previa")
    sed = (cp.filter((pl.col("id_ramo") == 7) & (pl.col("cve_ent") == CAMPECHE)
                     & pl.col("ciclo").is_in([2022, 2023]))
           .join(pob.rename({"año": "ciclo"}), on=["ciclo", "cve_ent"])
           .with_columns((pl.col("monto_ejercido") / pl.col("pob_total")).alias("pc"))
           .sort("ciclo"))
    print(f"  caveat ramo 7 (SEDENA): el monto localizado es obra, no despliegue — "
          f"Campeche pasa de {sed['pc'][0]:,.0f} a {sed['pc'][1]:,.1f} MXN pc entre 2022 y "
          f"2023 (Tren Maya); se descarta como proxy de custodia")
    return serie_pol, semar_pc


def figura_capacidad_par(serie_pol: pl.DataFrame, semar_pc: pl.DataFrame) -> go.Figure:
    fig = make_subplots(cols=2, horizontal_spacing=0.1,
                        subplot_titles=("policías estatales por 1,000 hab",
                                        "gasto SEMAR ejercido per cápita (MXN)"))
    años_p = sorted(serie_pol["año"].unique().to_list())
    for cve, nombre, color in [(CAMPECHE, "Campeche", AZUL), (TABASCO, "Tabasco", ROJO)]:
        d = serie_pol.filter(pl.col("cve_ent") == cve).sort("año")
        fig.add_trace(go.Scatter(x=d["año"], y=d["pc"], name=nombre,
                                 line=dict(color=color, width=3)), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=años_p, y=[serie_pol.filter(pl.col("año") == a)["pc"].median() for a in años_p],
        name="mediana estatal", line=dict(color=GRIS, width=2)), row=1, col=1)
    fig.add_hline(y=1.8, line_dash="dash", line_color=OSCURO, row=1, col=1,
                  annotation_text="estándar MOFP 1.8", annotation_position="bottom right",
                  annotation_font_size=10)

    años_s = sorted(semar_pc["ciclo"].unique().to_list())
    for cve, nombre, color in [(CAMPECHE, "Campeche", AZUL), (TABASCO, "Tabasco", ROJO)]:
        d = semar_pc.filter(pl.col("cve_ent") == cve).sort("ciclo")
        fig.add_trace(go.Scatter(x=d["ciclo"], y=d["pc"], name=nombre, showlegend=False,
                                 line=dict(color=color, width=3)), row=1, col=2)
    fig.add_annotation(x=2024, xref="x2", yref="y2", ax=-70, ay=15, font=dict(size=10),
                       y=semar_pc.filter((pl.col("cve_ent") == TABASCO)
                                         & (pl.col("ciclo") == 2024))["pc"][0],
                       text="salto posterior al pico<br>de violencia (reactivo)")
    fig.update_layout(
        title="¿Déficit de vigilancia en Campeche? Civil sí — naval, todo lo contrario<br>"
              "<sup>izq: CNGSPSPE/CNSPE 2016-2024 (sin dato 2019) · der: Cuenta Pública, "
              "ramo 13 localizado por estado (sin filas = 0 distribuido)</sup>",
        legend=dict(orientation="h", y=-0.12))
    guardar_fig(fig, "cap8_petroleo_capacidad")
    return fig


# ------------------------------------------------- C. la naturaleza de la renta

def seccion_c_renta(pob: pl.DataFrame) -> dict:
    print("\n=== 8.C-C. La hipótesis alternativa: la naturaleza de la renta ===")
    pet = leer_pibe(PIBE_MINERIA_PETROLERA, bloque="Millones de pesos")
    tot = leer_pibe(PIBE_TOTAL, bloque="Millones de pesos")
    sh = (pet.join(tot, on=["cve_ent", "año"], suffix="_tot")
          .with_columns((pl.col("valor") / pl.col("valor_tot") * 100).alias("pct")))
    a_max = sh["año"].max()
    s = sh.filter(pl.col("año") == a_max)
    pib_c, pib_t = _v(s, CAMPECHE, "pct"), _v(s, TABASCO, "pct")
    print(f"  PIB petrolero como % del PIB estatal ({a_max}): Campeche={pib_c:.0f}% "
          f"(plataforma marina — sonda) · Tabasco={pib_t:.0f}% (campos terrestres, "
          f"ductos, Dos Bocas)")

    tomas = cargar_tomas()
    por_edo = tomas.groupby("cve_ent").size().sort_values(ascending=False)
    n_c, n_t = int(por_edo.get(CAMPECHE, 0)), int(por_edo.get(TABASCO, 0))
    assert n_c == 0, f"Campeche con tomas: {n_c}"
    rank_t = list(por_edo.index).index(TABASCO) + 1
    print(f"  tomas clandestinas 2008-2016 (CartoCrítica): Campeche={n_c} · "
          f"Tabasco={n_t} (lugar {rank_t} nacional) — la renta offshore no se puede "
          f"ordeñar; la onshore sí")

    cp = cargar_cp()
    tr = (cp.filter(pl.col("id_ramo").is_in([28, 33]) & pl.col("ciclo").is_between(2016, 2025)
                    & pl.col("cve_ent").is_between(1, 32))
          .group_by("ciclo", "cve_ent").agg(pl.sum("monto_ejercido").alias("ejercido"))
          .join(pob.rename({"año": "ciclo"}), on=["ciclo", "cve_ent"])
          .with_columns((pl.col("ejercido") / pl.col("pob_total")).alias("pc")))
    gana = sum(_v(tr.filter(pl.col("ciclo") == a), CAMPECHE, "pc")
               > _v(tr.filter(pl.col("ciclo") == a), TABASCO, "pc")
               for a in range(2016, 2026))
    t24 = tr.filter(pl.col("ciclo") == 2024)
    tr_c, tr_t = _v(t24, CAMPECHE, "pc"), _v(t24, TABASCO, "pc")
    print(f"  transferencias 28+33: Tabasco recibe más SOLO en absoluto (2024: "
          f"{_v(t24, TABASCO, 'ejercido') / 1e9:.1f} vs "
          f"{_v(t24, CAMPECHE, 'ejercido') / 1e9:.1f} mmdp, tiene 2.6× población); per "
          f"cápita Campeche recibe más en {gana}/10 años (2024: {tr_c:,.0f} vs {tr_t:,.0f} "
          f"MXN) — la premisa se corrige")

    hom3 = tasas_estatales(HOMICIDIO, [2023, 2024, 2025], pob)
    return {
        f"PIB petrolero, % del PIB estatal ({a_max})":
            (pib_c, pib_t, "{:.0f}%"),
        "tomas clandestinas 2008-2016":
            (n_c, n_t, "{:.0f}"),
        "transferencias federales pc 2024 (miles MXN)":
            (tr_c / 1000, tr_t / 1000, "{:,.1f}"),
        "homicidio doloso /100k (2023-25)":
            (_v(hom3, CAMPECHE, "tasa"), _v(hom3, TABASCO, "tasa"), "{:.1f}"),
    }


def figura_renta(par: dict) -> go.Figure:
    fig = make_subplots(rows=2, cols=2, subplot_titles=list(par),
                        vertical_spacing=0.18, horizontal_spacing=0.1)
    for i, (vc, vt, fmt) in enumerate(par.values()):
        fig.add_trace(go.Bar(x=["Campeche", "Tabasco"], y=[vc, vt], showlegend=False,
                             marker_color=[AZUL, ROJO],
                             text=[fmt.format(vc), fmt.format(vt)],
                             textposition="outside"),
                      row=i // 2 + 1, col=i % 2 + 1)
    fig.update_layout(
        title="Misma renta petrolera, distinta física: la sonda no se puede ordeñar; "
              "el ducto sí<br><sup>PIBE 14 · CartoCrítica · Cuenta Pública ramos 28+33 · "
              "SESNSP</sup>",
        margin=dict(t=110))
    fig.update_yaxes(rangemode="tozero")
    guardar_fig(fig, "cap8_petroleo_renta", alto=650)
    return fig


def main():
    pob = cargar_poblacion()
    hom, env = seccion_a_divergencia(pob)
    figura_divergencia(hom, env)
    serie_pol, semar_pc = seccion_b_vigilancia(pob)
    figura_capacidad_par(serie_pol, semar_pc)
    par = seccion_c_renta(pob)
    figura_renta(par)
    print("\n=== 8.C veredicto ===")
    print("  el déficit de vigilancia civil no explica la disparidad (la correlación va "
          "al revés);\n  la explica la física de la renta: custodiada e inaccesible "
          "(Campeche) vs disputable (Tabasco).")


if __name__ == "__main__":
    main()
