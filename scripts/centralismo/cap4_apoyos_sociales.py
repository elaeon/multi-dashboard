"""
Cap 4 — Apoyos sociales: ¿combaten la pobreza o alimentan la dependencia? (H4)

Tests:
  A. Focalización y efecto: subsidios con entrega real / Ramo 20 (Bienestar) per cápita
     vs. nivel (2022) y cambio (2018→2022) de la pobreza CONEVAL por estado.
  B. Dependencia fiscal: transferencias federalizadas (R28+R33) vs ingresos propios
     estatales (impuestos + derechos, recaudacion_local) 2013-2024 — ¿crece la dependencia?
  C. ¿Paliativo o estructural?: evolución nacional de pobreza y carencias CONEVAL
     2016-2022 + expansión del padrón atendido (poblaciones PEF 2019-2024).

Figuras → centralismo/informe/figuras/cap4_*.png
Run: uv run python scripts/centralismo/cap4_apoyos_sociales.py
"""

import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import polars as pl
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from comun import CORTO, RAIZ, cargar_pobreza, cargar_poblacion, guardar_fig

DIR_CP = RAIZ / "data/presupuesto_federacion/cuenta_publica"
DIR_RL = RAIZ / "data/presupuesto_federacion/recaudacion_local"
DIR_POB = RAIZ / "data/presupuesto_federacion/poblaciones/atendidas"


def cargar_subsidios_estado(año: int) -> pl.DataFrame:
    z = zipfile.ZipFile(DIR_CP / f"subsidios_poblacion/Subsidios_CP{año}.zip")
    csv = [n for n in z.namelist() if n.endswith(".csv")][0]
    monto = "MONTO_PAGADO" if año == 2022 else "MONTO_EJERCIDO"
    return (
        pl.read_csv(z.read(csv), encoding="windows-1252", infer_schema_length=0)
        .with_columns(
            pl.col("ID_ENTIDAD_FEDERATIVA").cast(pl.Int64, strict=False).alias("cve_ent"),
            pl.col(monto).cast(pl.Float64, strict=False).alias("monto"),
            pl.col("ID_RAMO").cast(pl.Int64, strict=False).alias("id_ramo"),
        )
        .filter(pl.col("cve_ent").is_between(1, 32))
        .with_columns(pl.lit(año).alias("año"))
        .select("año", "cve_ent", "id_ramo", "monto")
    )


def test_a():
    print("=== A. Focalización y efecto sobre la pobreza ===")
    cp = pl.read_parquet(DIR_CP / "cp_estado_ramo.parquet")
    pob = cargar_poblacion()
    dep = cargar_pobreza()

    # validación: ¿la geografía del Ramo 20 en Cuenta Pública refleja entrega real?
    sub22 = cargar_subsidios_estado(2022).filter(pl.col("id_ramo") == 20).group_by("cve_ent").agg(pl.sum("monto"))
    r20cp = (cp.filter((pl.col("id_ramo") == 20) & (pl.col("ciclo") == 2022) & pl.col("cve_ent").is_between(1, 32))
             .group_by("cve_ent").agg(pl.sum("monto_ejercido")))
    val = sub22.join(r20cp, on="cve_ent")
    c_val = np.corrcoef(val["monto"], val["monto_ejercido"])[0, 1]
    print(f"  validación geografía R20: corr(subsidios 2022, cuenta pública 2022) = {c_val:.3f}")

    # tratamiento: gasto Bienestar (R20) per cápita acumulado 2019-2022 (arranque de la política)
    r20 = (cp.filter((pl.col("id_ramo") == 20) & pl.col("ciclo").is_between(2019, 2022)
                     & pl.col("cve_ent").is_between(1, 32))
           .group_by("cve_ent").agg(pl.sum("monto_ejercido").alias("bienestar")))
    r20 = (r20.join(pob.filter(pl.col("año") == 2022), on="cve_ent")
           .with_columns((pl.col("bienestar") / pl.col("pob_total")).alias("bienestar_pc")))

    d18 = dep.filter(pl.col("año") == 2018).select("cve_ent", pl.col("pct_pobreza").alias("p18"))
    d22 = dep.filter(pl.col("año") == 2022).select("cve_ent", pl.col("pct_pobreza").alias("p22"))
    base = (r20.join(d18, on="cve_ent").join(d22, on="cve_ent")
            .with_columns((pl.col("p22") - pl.col("p18")).alias("delta"))
            .with_columns(pl.col("cve_ent").replace_strict(CORTO).alias("estado")))

    # focalización con entrega real (subsidios R20, promedio 2022-2024)
    sub = pl.concat([cargar_subsidios_estado(a) for a in (2022, 2023, 2024)])
    sub20 = (sub.filter(pl.col("id_ramo") == 20).group_by("cve_ent")
             .agg((pl.sum("monto") / 3).alias("sub20"))
             .join(pob.filter(pl.col("año") == 2023), on="cve_ent")
             .with_columns((pl.col("sub20") / pl.col("pob_total")).alias("sub20_pc"))
             .join(dep.filter(pl.col("año") == 2022).select("cve_ent", "pct_pobreza"), on="cve_ent"))
    c_foc_real = np.corrcoef(sub20["sub20_pc"], sub20["pct_pobreza"])[0, 1]
    print(f"  focalización (entrega real): corr(subsidios R20 pc 22-24, pobreza 2022) = {c_foc_real:+.3f}")

    c_foc = np.corrcoef(base["bienestar_pc"], base["p18"])[0, 1]
    X = sm.add_constant(base["bienestar_pc"].to_numpy() / 1e3)
    m = sm.OLS(base["delta"].to_numpy(), X).fit()
    print(f"  focalización: corr(bienestar_pc 19-22, pobreza 2018) = {c_foc:+.3f}")
    print(f"  efecto: Δpobreza(18→22) ~ bienestar_pc → pendiente={m.params[1]:+.3f} pp por mil MXN pc "
          f"(p={m.pvalues[1]:.3f}, R²={m.rsquared:.2f})")
    print(f"  Δpobreza promedio nacional 2018→2022: {base['delta'].mean():+.1f} pp")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=base["bienestar_pc"] / 1e3, y=base["delta"], mode="markers+text",
        text=base["estado"], textposition="top center", textfont=dict(size=9),
        marker=dict(size=8, color="#1f77b4"), showlegend=False))
    xs = np.linspace(float((base["bienestar_pc"] / 1e3).min()), float((base["bienestar_pc"] / 1e3).max()), 50)
    fig.add_trace(go.Scatter(x=xs, y=m.params[0] + m.params[1] * xs, mode="lines",
                             line=dict(color="#c0392b"),
                             name=f"OLS: {m.params[1]:+.2f} pp/mil MXN (p={m.pvalues[1]:.2f})"))
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(
        title="¿Donde se gastó más Bienestar bajó más la pobreza? (2018→2022)<br><sup>x: gasto Ramo 20 per cápita acumulado 2019-2022; y: cambio en % de pobreza CONEVAL</sup>",
        xaxis_title="Bienestar per cápita 2019-2022 (miles de MXN)",
        yaxis_title="Δ pobreza 2018→2022 (puntos %)",
    )
    guardar_fig(fig, "cap4_efecto_pobreza")
    return base


def test_b():
    print("\n=== B. Dependencia fiscal de los estados ===")
    cp = pl.read_parquet(DIR_CP / "cp_estado_ramo.parquet")
    tr = (cp.filter(pl.col("id_ramo").is_in([28, 33]) & pl.col("cve_ent").is_between(1, 32))
          .group_by("ciclo", "cve_ent").agg(pl.sum("monto_ejercido").alias("transfer"))
          .rename({"ciclo": "año"}))

    propios = None
    for archivo, monto in [("mapa_impuestos_estatales.csv", "MONTO_IMPUESTOS"),
                           ("mapa_derechos_estatales.csv", "MONTO_DERECHOS")]:
        df = (pl.read_csv(DIR_RL / archivo, encoding="latin-1", infer_schema_length=0)
              .select(pl.col("CICLO").cast(pl.Int64).alias("año"),
                      pl.col("ID_ENTIDAD_FEDERATIVA").cast(pl.Int64).alias("cve_ent"),
                      pl.col(monto).cast(pl.Float64).alias("m")))
        propios = df if propios is None else propios.join(df, on=["año", "cve_ent"], how="full", coalesce=True).with_columns(
            (pl.col("m").fill_null(0) + pl.col("m_right").fill_null(0)).alias("m")).drop("m_right")
    propios = propios.rename({"m": "propios"})

    panel = (tr.join(propios, on=["año", "cve_ent"])
             .with_columns((pl.col("transfer") / (pl.col("transfer") + pl.col("propios"))).alias("dependencia"))
             .filter(pl.col("año").is_between(2013, 2024)))

    nac = (panel.group_by("año")
           .agg((pl.sum("transfer") / (pl.sum("transfer") + pl.sum("propios"))).alias("dependencia"))
           .sort("año"))
    print("  dependencia nacional (transferencias / [transferencias + ingresos propios estatales]):")
    for r in nac.iter_rows(named=True):
        print(f"    {r['año']}: {r['dependencia']*100:.1f}%")

    d24 = panel.filter(pl.col("año") == 2024).sort("dependencia", descending=True).with_columns(
        pl.col("cve_ent").replace_strict(CORTO).alias("estado"))
    d13 = panel.filter(pl.col("año") == 2013).select("cve_ent", pl.col("dependencia").alias("dep13"))
    d24j = d24.join(d13, on="cve_ent")
    print(f"  2024: más dependientes: " +
          ", ".join(f"{r['estado']}={r['dependencia']*100:.0f}%" for r in d24.head(4).iter_rows(named=True)))
    print(f"  2024: menos dependientes: " +
          ", ".join(f"{r['estado']}={r['dependencia']*100:.0f}%" for r in d24.tail(4).iter_rows(named=True)))
    sube = (d24j["dependencia"] > d24j["dep13"]).sum()
    print(f"  estados donde la dependencia SUBIÓ 2013→2024: {sube}/32")

    fig = go.Figure(go.Bar(
        x=d24j["dependencia"] * 100, y=d24j["estado"], orientation="h",
        marker_color=[("#c0392b" if a > b else "#27ae60") for a, b in zip(d24j["dependencia"], d24j["dep13"])],
    ))
    fig.add_trace(go.Scatter(x=d24j["dep13"] * 100, y=d24j["estado"], mode="markers",
                             marker=dict(symbol="line-ns-open", size=12, color="black"),
                             name="nivel 2013"))
    fig.update_layout(
        title="Dependencia fiscal estatal 2024 (barra) vs 2013 (marca)<br><sup>transferencias R28+R33 ÷ (transferencias + impuestos y derechos estatales propios) · rojo = subió</sup>",
        xaxis_title="% de los ingresos que proviene de la federación",
        yaxis=dict(tickfont=dict(size=10), autorange="reversed"),
    )
    guardar_fig(fig, "cap4_dependencia", alto=800)
    return nac


def test_c():
    print("\n=== C. ¿Paliativo o estructural? ===")
    dep = cargar_pobreza()
    pob = cargar_poblacion()
    w = (dep.join(pob, on=["cve_ent", "año"])
         .group_by("año")
         .agg(*[
             ((pl.col(c) * pl.col("pob_total")).sum() / pl.col("pob_total").sum()).alias(c)
             for c in ["pct_pobreza", "pct_ic_asalud", "pct_ic_segsoc", "pct_ic_rezedu", "pct_ic_ali"]
         ])
         .sort("año"))
    nombres = {"pct_pobreza": "pobreza", "pct_ic_asalud": "carencia acceso a salud",
               "pct_ic_segsoc": "carencia seguridad social", "pct_ic_rezedu": "rezago educativo",
               "pct_ic_ali": "carencia alimentaria"}
    print("  nacional (promedio ponderado por población):")
    for r in w.iter_rows(named=True):
        print("    " + str(r["año"]) + ": " + " · ".join(f"{nombres[c]}={r[c]:.1f}%" for c in nombres))

    fig = go.Figure()
    for c, n in nombres.items():
        fig.add_trace(go.Scatter(x=w["año"], y=w[c], mode="lines+markers", name=n))
    fig.update_layout(
        title="Pobreza y carencias sociales nacionales, CONEVAL 2016-2022<br><sup>la pobreza (ingreso+carencias) baja; la carencia de acceso a salud se dispara tras la desaparición del Seguro Popular</sup>",
        xaxis_title="año", yaxis_title="% de la población",
    )
    guardar_fig(fig, "cap4_carencias")

    # padrón atendido nacional (poblaciones PEF, Persona física)
    print("  padrón atendido (PEF poblaciones, tipo Persona física):")
    filas = []
    for f in sorted(DIR_POB.iterdir()):
        m = re.search(r"(20\d\d)", f.name)
        if not m or f.suffix.lower() != ".xlsx":
            continue
        año = int(m.group(1))
        if año in (2019, 2020):
            raw = pd.read_excel(f, skiprows=1, header=[0, 1])
        elif año == 2021:
            raw = pd.read_excel(f, header=[0, 1])
        else:
            raw = pd.read_excel(f, header=0)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = ["_".join(str(x) for x in c if "Unnamed" not in str(x)).strip("_")
                           for c in raw.columns]
        raw.columns = [str(c).replace("\n", "_") for c in raw.columns]
        c_tipo = [c for c in raw.columns if re.search(r"objetivo.*Tipo", str(c), re.I)] or \
                 [c for c in raw.columns if re.search(r"Tipo", str(c), re.I)]
        c_cuant = [c for c in raw.columns if re.search(r"atendida.*Cuantificaci", str(c), re.I)]
        if not c_tipo or not c_cuant:
            print(f"    {año}: columnas no identificadas — omitido")
            continue
        pf = raw[raw[c_tipo[0]].astype(str).str.strip() == "Persona física"]
        tot = pd.to_numeric(pf[c_cuant[0]], errors="coerce").sum()
        filas.append((año, tot))
        print(f"    {año}: atendida={tot/1e6:.0f} M (contactos-servicio, no personas únicas)")
    if filas:
        s = pl.DataFrame(filas, schema=["año", "atendida"], orient="row")
        fig = go.Figure(go.Bar(x=s["año"], y=s["atendida"] / 1e6, marker_color="#1f77b4"))
        fig.update_layout(
            title="Cuantificación de población atendida por programas sociales (Persona física)<br><sup>suma de contactos-servicio reportados por programa — no personas únicas</sup>",
            xaxis_title="año", yaxis_title="millones de registros atendidos",
        )
        guardar_fig(fig, "cap4_padron")


def main():
    test_a()
    test_b()
    test_c()


if __name__ == "__main__":
    main()
