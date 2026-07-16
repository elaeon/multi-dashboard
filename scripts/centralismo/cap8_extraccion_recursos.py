"""
Cap 8 — Extracción minera y forestal (refuerzo/contraste de H1).

A. Minería — test de enclave:
   1. Valor de producción minera por estado (SGM, filtros documentados: categoria nula,
      productos-agregado, artefacto Hidalgo 'Agregados pétreos', ×1000 en 2020-2024).
   2. Labor share local: masa salarial IMSS SECTOR=1 (Industrias extractivas) anual
      ÷ valor de producción SGM. Razón baja = la renta sale del estado (enclave).
   3. Peso minero en el PIB (PIBE_15/PIBE_2 corrientes) × razón retorno/aporte (cap 1).
   Años de referencia: 2018 y 2022 (2023-2024 SGM son preliminares: ~310 registros vs 450).

B. Forestal — test de potencial no aprovechado:
   1. Balanza comercial forestal (exp_imp_pais, VAL_USD) vs valor de producción nacional.
   2. Concentración de capacidad industrial CAT (solo Unidad CI = Metros cúbicos).
   3. Tenencia del bosque: volumen autorizado por Tipo de propiedad (manejo maderable).
   4. Valor forestal pc por estado vs pobreza CONEVAL.

Figuras → centralismo/informe/figuras/cap8_*.png
Run: uv run python scripts/centralismo/cap8_extraccion_recursos.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from comun import (CORTO, PIBE_MINERIA_NO_PETROLERA, PIBE_MINERIA_PETROLERA,
                   PIBE_TOTAL, RAIZ, cargar_pobreza, cargar_poblacion,
                   guardar_fig, leer_pibe, normalizar_estado)

DIR_SGM = RAIZ / "data/sgm/produccion_minera"
DIR_SEM = RAIZ / "data/semarnat"
AGREGADOS = r"(?i)^\s*(met[áa]licos|no\s+met[áa]licos|total)"


def cargar_sgm_valor() -> pl.DataFrame:
    """Valor de producción minera limpio, pesos corrientes, estado×año."""
    dfs = []
    for f in sorted(DIR_SGM.glob("produccion_minera_entidad_*.csv")):
        df = pl.read_csv(f, encoding="utf8", null_values=["", "NA"])
        df = df.filter(
            pl.col("categoria").is_not_null()
            & ~pl.col("producto").str.contains(AGREGADOS)
            & ~pl.col("producto").str.to_lowercase().str.contains("agregados")  # artefacto Hidalgo 2024
            & (pl.col("tabla") == "valor")
            & (pl.col("valor") > 0)
        ).with_columns(
            pl.when(pl.col("unidad") == "Miles de pesos corrientes")
            .then(pl.col("valor") * 1000).otherwise(pl.col("valor")).alias("valor_pesos"))
        dfs.append(df.select("estado", "año", "valor_pesos"))
    sgm = pl.concat(dfs)
    mapa = {e: normalizar_estado(e) for e in sgm["estado"].unique().to_list()}
    sgm = (sgm.with_columns(pl.col("estado").replace_strict(mapa, default=None).alias("cve_ent"))
           .filter(pl.col("cve_ent").is_not_null())
           .group_by("cve_ent", "año").agg(pl.sum("valor_pesos").alias("valor_mineria")))
    return sgm


def masa_anual(sector: int) -> pl.DataFrame:
    """Masa salarial anualizada por estado-año para un SECTOR IMSS.

    MASA_SALARIAL_TOTAL es masa DIARIA (≈323 MXN/puesto en promedio histórico):
    anual = promedio de los cortes mensuales × 365.
    """
    return (
        pl.scan_csv(RAIZ / "data/datamx/empleo_formal/empleo_formal.csv")
        .filter(pl.col("SECTOR") == sector)
        .with_columns(pl.col("PERIODO").str.slice(0, 4).cast(pl.Int64).alias("año"))
        .group_by("año", "CVE_ENT", "PERIODO")
        .agg(pl.sum("MASA_SALARIAL_TOTAL").alias("masa_dia"))
        .group_by("año", "CVE_ENT")
        .agg((pl.mean("masa_dia") * 365).alias("masa_anual"))
        .collect().rename({"CVE_ENT": "cve_ent"})
    )


def test_a_mineria():
    print("=== A. Minería — test de enclave ===")
    # 1. contexto: ranking de valor de producción SGM (solo 2018: los archivos 2020-2024
    #    mezclan unidades fila a fila — miles y pesos — y la serie monetaria es inutilizable;
    #    verificado 2026-07-06: total 2020 re-escalado = 2.6× el de 2019, Feldespato-Tlaxcala
    #    = 223 mmdp. Ver nota en INFORME).
    sgm18 = cargar_sgm_valor().filter(pl.col("año") == 2018).sort("valor_mineria", descending=True)
    tot18 = sgm18["valor_mineria"].sum()
    print(f"  valor de producción minera 2018 (SGM limpio): {tot18/1e9:.0f} mmdp; top-5: " +
          " · ".join(f"{CORTO[r['cve_ent']]}={r['valor_mineria']/tot18*100:.1f}%"
                     for r in sgm18.head(5).iter_rows(named=True)))

    # 2. labor share del valor agregado minero (PIBE_15, corrientes) vs manufactura (PIBE_18)
    pib_min = leer_pibe(PIBE_MINERIA_NO_PETROLERA, bloque="Millones de pesos").rename({"valor": "va"})
    pib_man = leer_pibe(18, bloque="Millones de pesos").rename({"valor": "va"})
    m_min = masa_anual(1)   # industrias extractivas
    m_man = masa_anual(3)   # industrias de transformación

    print(f"\n  {'':<8}{'labor share minería':>22}{'labor share manufactura':>26}")
    resumen = {}
    for a in (2018, 2022):
        ls = {}
        for nombre, va, masa in [("mineria", pib_min, m_min), ("manufactura", pib_man, m_man)]:
            x = (va.filter(pl.col("año") == a).join(masa.filter(pl.col("año") == a), on=["cve_ent", "año"])
                 .with_columns((pl.col("masa_anual") / (pl.col("va") * 1e6)).alias("ls")))
            ls[nombre] = x
        nal_min = ls["mineria"]["masa_anual"].sum() / (ls["mineria"]["va"].sum() * 1e6)
        nal_man = ls["manufactura"]["masa_anual"].sum() / (ls["manufactura"]["va"].sum() * 1e6)
        resumen[a] = (nal_min, nal_man, ls["mineria"])
        print(f"  {a:<8}{nal_min*100:>21.1f}%{nal_man*100:>25.1f}%")

    nal_min22, nal_man22, ls22 = resumen[2022]
    top_min = (ls22.join(pib_min.filter(pl.col("año") == 2022)
                         .with_columns((pl.col("va") / pl.col("va").sum()).alias("sh")), on="cve_ent")
               .sort("va", descending=True).head(10)
               .with_columns(pl.col("cve_ent").replace_strict(CORTO).alias("estado")))
    print("  labor share minero 2022 por estado (top-10 por VA minero):")
    for r in top_min.iter_rows(named=True):
        print(f"    {r['estado']:<16} VA={r['va']/1e3:>6.0f} mmdp · salarios formales/VA={r['ls']*100:>5.1f}%")

    b = top_min.sort("ls")
    fig = go.Figure(go.Bar(
        x=b["ls"] * 100, y=b["estado"], orientation="h",
        marker_color=["#c0392b" if v < nal_man22 * 0.6 else "#e67e22" for v in b["ls"]]))
    fig.add_vline(x=nal_man22 * 100, line_dash="dash", line_color="black",
                  annotation_text=f"manufactura nacional {nal_man22*100:.0f}%")
    fig.update_layout(
        title="Test de enclave minero, 2022: salarios formales locales como % del valor agregado minero<br><sup>masa salarial IMSS sector extractivo (anualizada) ÷ PIB minero no petrolero del estado (PIBE_15) — 10 estados con más VA minero</sup>",
        xaxis_title="% del valor agregado minero pagado en salarios formales locales",
    )
    guardar_fig(fig, "cap8_enclave_minero")

    # peso minero en PIB vs razón retorno/aporte (2022)
    pib_min = leer_pibe(PIBE_MINERIA_NO_PETROLERA, bloque="Millones de pesos").rename({"valor": "pib_min"})
    pib = leer_pibe(PIBE_TOTAL, bloque="Millones de pesos").rename({"valor": "pib"})
    petro = leer_pibe(PIBE_MINERIA_PETROLERA, bloque="Millones de pesos").rename({"valor": "petro"})
    cp = pl.read_parquet(RAIZ / "informe_data/cp_estado_ramo.parquet")
    tr = (cp.filter((pl.col("ciclo") == 2022) & pl.col("id_ramo").is_in([28, 33])
                    & pl.col("cve_ent").is_between(1, 32))
          .group_by("cve_ent").agg(pl.sum("monto_ejercido").alias("transfer")))
    x = (pib.filter(pl.col("año") == 2022)
         .join(petro.filter(pl.col("año") == 2022), on=["cve_ent", "año"], how="left")
         .join(pib_min.filter(pl.col("año") == 2022), on=["cve_ent", "año"])
         .join(tr, on="cve_ent")
         .with_columns(
             (pl.col("pib") - pl.col("petro").fill_null(0)).alias("pib_sp"),
             (pl.col("pib_min") / pl.col("pib") * 100).alias("peso_minero"))
         .with_columns(
             ((pl.col("transfer") / pl.col("transfer").sum())
              / (pl.col("pib_sp") / pl.col("pib_sp").sum())).alias("razon"),
             pl.col("cve_ent").replace_strict(CORTO).alias("estado")))
    c = np.corrcoef(x["peso_minero"], x["razon"])[0, 1]
    print(f"\n  corr(peso minero en PIB, razón retorno/aporte) 2022 = {c:+.2f}")
    mineros = x.sort("peso_minero", descending=True).head(8)
    print("  estados mineros (peso PIB): " +
          " · ".join(f"{r['estado']}={r['peso_minero']:.1f}% (razón {r['razon']:.2f})"
                     for r in mineros.iter_rows(named=True)))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x["peso_minero"], y=x["razon"], mode="markers+text", text=x["estado"],
        textposition="top center", textfont=dict(size=9),
        marker=dict(size=9, color="#8e6a1a"), showlegend=False))
    fig.add_hline(y=1, line_dash="dash", line_color="gray")
    fig.update_layout(
        title=f"Peso de la minería en el PIB estatal vs razón retorno/aporte federal, 2022 (corr={c:+.2f})<br><sup>razón &lt;1 = aportador neto; el patrón petrolero se repetiría si los mineros quedaran sistemáticamente debajo de 1</sup>",
        xaxis_title="minería no petrolera como % del PIB estatal",
        yaxis_title="share transferencias ÷ share PIB sin petróleo",
    )
    guardar_fig(fig, "cap8_peso_minero_vs_retorno")


def test_b_forestal():
    print("\n=== B. Forestal — potencial no aprovechado ===")
    # 1. balanza comercial
    ei = pd.read_excel(DIR_SEM / "indicadores_economicos/Exportaciones-e-Importaciones-2019-2023.xlsx",
                       sheet_name="exp_imp_pais")
    ei.columns = [c.strip() for c in ei.columns]
    bal = ei.groupby(["ANIO", "TIPO"])["VAL_USD"].sum().unstack()
    print("  balanza comercial forestal (millones USD):")
    for a, r in bal.iterrows():
        print(f"    {a}: exp={r['Exportaciones']/1e6:,.0f} · imp={r['Importaciones']/1e6:,.0f} "
              f"· razón imp/exp={r['Importaciones']/r['Exportaciones']:.2f}")
    razon_total = bal["Importaciones"].sum() / bal["Exportaciones"].sum()
    imp_mxn = ei[ei["TIPO"] == "Importaciones"]["VAL_MNX"].sum()

    # 2. valor de producción nacional (maderable)
    prod = pd.read_excel(DIR_SEM / "produccion_forestal/Produccion_forestal_maderable_de_Mexico_2019-2023.xlsx",
                         sheet_name="Datos")
    prod.columns = [str(c).strip() for c in prod.columns]
    c_ent = [c for c in prod.columns if "Entidad" in c and "Nombre" in c][0]
    c_val = [c for c in prod.columns if "Valor" in c][0]
    prod = prod.dropna(subset=[c_ent])
    val_nal = prod[c_val].sum()
    print(f"  valor producción maderable 2019-2023: {val_nal/1e9:.1f} mmdp MXN "
          f"vs importaciones forestales {imp_mxn/1e9:.0f} mmdp MXN "
          f"({imp_mxn/val_nal:.1f}× la producción nacional)")

    fig = go.Figure()
    fig.add_trace(go.Bar(x=bal.index, y=bal["Exportaciones"] / 1e6, name="exportaciones",
                         marker_color="#27ae60"))
    fig.add_trace(go.Bar(x=bal.index, y=bal["Importaciones"] / 1e6, name="importaciones",
                         marker_color="#c0392b"))
    fig.update_layout(
        barmode="group",
        title=f"Balanza comercial forestal de México (razón importación/exportación = {razon_total:.1f})<br><sup>el país compra fuera {imp_mxn/val_nal:.0f}× el valor de toda su producción maderable</sup>",
        xaxis_title="año", yaxis_title="millones de USD",
    )
    guardar_fig(fig, "cap8_forestal_deficit")

    # 3. concentración de capacidad CAT (solo metros cúbicos)
    cat = pd.read_excel(DIR_SEM / "industria_forestal/Centros-de-Almacenamiento-y-Transformacion-2019-2023.xlsx",
                        sheet_name="Datos")
    cat.columns = [str(c).strip() for c in cat.columns]
    m3 = cat[cat["Unidad CI"] == "Metros cúbicos"].copy()
    tot_ci = m3["Capacidad instalada (CI)"].sum()
    top10 = m3["Capacidad instalada (CI)"].nlargest(10).sum()
    top1 = m3["Capacidad instalada (CI)"].max()
    print(f"  capacidad CAT (m³): total={tot_ci/1e6:.1f} M m³ · top-10 instalaciones={top10/tot_ci*100:.1f}% "
          f"· 1 sola instalación={top1/tot_ci*100:.1f}%")

    # 4. tenencia del bosque (volumen autorizado, manejo maderable)
    man = pd.read_excel(DIR_SEM / "manejo_forestal/Aprovechamientos-forestales-maderables-autorizados-y-vigentes-2019-2023.xlsx",
                        sheet_name="Datos")
    man.columns = [str(c).strip() for c in man.columns]
    man = man.dropna(subset=["Tipo de propiedad"])
    ten_n = man["Tipo de propiedad"].value_counts(normalize=True) * 100
    ten_v = man.groupby("Tipo de propiedad")["Volumen a aprovechar"].sum()
    ten_v = ten_v / ten_v.sum() * 100
    print("  tenencia de aprovechamientos maderables:")
    for t in ["Particular", "Ejidal", "Comunal"]:
        print(f"    {t:<11} permisos={ten_n.get(t, 0):.1f}% · volumen={ten_v.get(t, 0):.1f}%")

    fig = go.Figure()
    cats = ["Particular", "Ejidal", "Comunal"]
    fig.add_trace(go.Bar(x=cats, y=[ten_n.get(t, 0) for t in cats], name="% de permisos",
                         marker_color="#95a5a6"))
    fig.add_trace(go.Bar(x=cats, y=[ten_v.get(t, 0) for t in cats], name="% del volumen autorizado",
                         marker_color="#2c6e49"))
    fig.update_layout(
        barmode="group",
        title="¿De quién es el bosque aprovechado? Tenencia de los permisos maderables 2019-2023<br><sup>SEMARNAT manejo forestal; la capacidad industrial (CAT) está aparte: 1 instalación = 20.7% del m³ nacional</sup>",
        yaxis_title="%",
    )
    guardar_fig(fig, "cap8_tenencia_forestal")

    # 5. estados forestales vs pobreza
    por_edo = prod.groupby(c_ent)[c_val].sum().sort_values(ascending=False)
    pob = cargar_poblacion().filter(pl.col("año") == 2021)
    dep = cargar_pobreza().filter(pl.col("año") == 2022)
    filas = []
    for ent, v in por_edo.items():
        cve = normalizar_estado(ent)
        if cve:
            p = pob.filter(pl.col("cve_ent") == cve)["pob_total"][0]
            pz = dep.filter(pl.col("cve_ent") == cve)["pct_pobreza"][0]
            filas.append((CORTO[cve], v / 5 / p, pz))  # valor anual promedio pc
    top8 = filas[:8]
    print("  top-8 estados forestales (valor anual pc MXN · pobreza 2022):")
    for e, vpc, pz in top8:
        print(f"    {e:<14} {vpc:>7.0f} MXN/hab · pobreza {pz:.0f}%")


def main():
    test_a_mineria()
    test_b_forestal()


if __name__ == "__main__":
    main()
