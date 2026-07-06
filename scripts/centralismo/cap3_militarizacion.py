"""
Cap 3 — Militarización del Estado (H3 operacionalizada).

H3 original ("la clase dominante es la político-militar") NO es testeable con estos
datos; se operacionaliza como militarización presupuestal-funcional:
  A. Gasto ejercido de ramos militares/seguridad (7 SEDENA, 13 SEMAR, 36 SSPC/GN)
     2016-2025: nivel, share del gasto total y CAGR vs. resto.
  B. Sobre-ejercicio (ejercido/aprobado): ¿los ramos militares gastan sistemáticamente
     por encima de lo que el Congreso aprueba?
  C. Plazas APF 2018-2025: share de posiciones militares en el empleo público federal.
  D. SEDENA actos jurídicos (PNT 2023-2025): expansión de funciones económicas.

Figuras → centralismo/informe/figuras/cap3_*.png
Run: uv run python scripts/centralismo/cap3_militarizacion.py
"""

import sys
from pathlib import Path

import plotly.graph_objects as go
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from comun import RAIZ, guardar_fig

RAMOS_MIL = {7: "SEDENA", 13: "SEMAR", 36: "SSPC/GN"}
PLAZAS = {
    2018: "analitico_plazas_apf18.xlsx",
    2019: "analitico_plazas_apf19.xlsx",
    2020: "analitico_plazas_apf_2020.xlsx",
    2021: "analitico_plazas_apf_2021.xlsx",
    2022: "analitico_plazas_apf_PEF_2022.xlsx",
    2023: "analitico_plazas_apf_PEF_2023.xlsx",
    2024: "analitico_plazas_apf_PEF_2024.xlsx",
    2025: "analitico_plazas_apf_PEF_2025.xlsx",
}

# columnas del buscador PNT (ver dashboard/sedena_concesiones.py)
C_AÑO = "Ejercicio"
C_TIPO = "Tipo de acto jurídico (catálogo)"
C_OBJETO = "Objeto de la realización del acto jurídico"
C_MONTO = "Monto total o beneficio, servicio y/o recurso público aprovechado"


def gasto_militar():
    cp = pl.read_parquet(RAIZ / "data/presupuesto_federacion/cuenta_publica/cp_estado_ramo.parquet")
    nac = cp.group_by("ciclo", "id_ramo").agg(
        pl.sum("monto_aprobado").alias("aprobado"), pl.sum("monto_ejercido").alias("ejercido"))
    nac = nac.filter(pl.col("ciclo") >= 2016)  # alcance homogéneo gf_ecd_epe

    tot = nac.group_by("ciclo").agg(pl.sum("ejercido").alias("tot_ej")).sort("ciclo")
    mil = (nac.filter(pl.col("id_ramo").is_in(list(RAMOS_MIL)))
           .group_by("ciclo").agg(pl.sum("ejercido").alias("mil_ej"),
                                  pl.sum("aprobado").alias("mil_ap")).sort("ciclo"))
    serie = mil.join(tot, on="ciclo").with_columns((pl.col("mil_ej") / pl.col("tot_ej")).alias("share"))

    print("=== A. Gasto militar+seguridad (R7+R13+R36), alcance homogéneo 2016-2025 ===")
    for r in serie.iter_rows(named=True):
        print(f"  {r['ciclo']}: {r['mil_ej']/1e9:.0f} mmdp ejercido · {r['share']*100:.2f}% del gasto total "
              f"· ejercido/aprobado={r['mil_ej']/r['mil_ap']:.2f}")
    e16, e25 = serie["mil_ej"][0], serie["mil_ej"][-1]
    t16, t25 = serie["tot_ej"][0], serie["tot_ej"][-1]
    print(f"  CAGR nominal militar 2016-2025: {((e25/e16)**(1/9)-1)*100:.1f}% vs total {((t25/t16)**(1/9)-1)*100:.1f}%")

    fig = go.Figure()
    for ramo, nombre in RAMOS_MIL.items():
        s = (nac.filter(pl.col("id_ramo") == ramo).join(tot, on="ciclo")
             .with_columns((pl.col("ejercido") / pl.col("tot_ej") * 100).alias("sh")).sort("ciclo"))
        fig.add_trace(go.Scatter(x=s["ciclo"], y=s["sh"], mode="lines+markers", name=nombre,
                                 stackgroup="mil"))
    fig.add_vline(x=2019, line_dash="dot", line_color="gray",
                  annotation_text="creación GN", annotation_position="top left")
    fig.update_layout(
        title="Gasto militar y de seguridad como % del gasto federal ejercido (apilado)",
        xaxis_title="año", yaxis_title="% del gasto total ejercido",
    )
    guardar_fig(fig, "cap3_gasto_militar")
    return nac, serie


def sobre_ejercicio(nac: pl.DataFrame):
    print("\n=== B. Sobre-ejercicio (ejercido/aprobado) ===")
    filas = []
    for r in sorted(nac["ciclo"].unique()):
        y = nac.filter((pl.col("ciclo") == r) & (pl.col("aprobado") > 0))
        m = y.filter(pl.col("id_ramo").is_in(list(RAMOS_MIL)))
        o = y.filter(~pl.col("id_ramo").is_in(list(RAMOS_MIL)))
        rm = m["ejercido"].sum() / m["aprobado"].sum()
        ro = o["ejercido"].sum() / o["aprobado"].sum()
        filas.append((r, rm, ro))
        print(f"  {r}: militar={rm:.2f} · resto={ro:.2f}")
    df = pl.DataFrame(filas, schema=["ciclo", "militar", "resto"], orient="row")
    prom_m, prom_o = df["militar"].mean(), df["resto"].mean()
    print(f"  promedio 2016-2025: militar={prom_m:.2f} vs resto={prom_o:.2f}")

    fig = go.Figure()
    fig.add_trace(go.Bar(x=df["ciclo"], y=(df["militar"] - 1) * 100, name="ramos militares (7+13+36)",
                         marker_color="#7f8c1f"))
    fig.add_trace(go.Bar(x=df["ciclo"], y=(df["resto"] - 1) * 100, name="resto de ramos",
                         marker_color="#95a5a6"))
    fig.add_hline(y=0, line_color="black")
    fig.update_layout(
        barmode="group",
        title="Sobre-ejercicio presupuestal: % gastado por encima de lo aprobado por el Congreso",
        xaxis_title="año", yaxis_title="% sobre lo aprobado",
    )
    guardar_fig(fig, "cap3_sobreejercicio")
    return prom_m, prom_o


def plazas_militares():
    print("\n=== C. Plazas APF (personal federal autorizado) ===")
    filas = []
    for año, nombre in PLAZAS.items():
        df = pl.read_excel(RAIZ / "data/presupuesto_federacion/presupuesto/plazas" / nombre,
                           engine="calamine")
        c_ramo = next(c for c in df.columns if c.lower() == "ramo")
        df = df.with_columns(
            pl.col(c_ramo).cast(pl.Utf8).str.extract(r"^(\d+)").cast(pl.Int64).alias("id_ramo"))
        mil = df.filter(pl.col("id_ramo").is_in(list(RAMOS_MIL)))["Plazas"].sum()
        tot = df["Plazas"].sum()
        filas.append((año, mil, tot, mil / tot))
        print(f"  {año}: militares={mil:,.0f} de {tot:,.0f} APF ({mil/tot*100:.1f}%)")
    df = pl.DataFrame(filas, schema=["año", "mil", "tot", "share"], orient="row")

    fig = go.Figure()
    fig.add_trace(go.Bar(x=df["año"], y=df["mil"] / 1e3, name="plazas militares/seguridad (miles)",
                         marker_color="#7f8c1f"))
    fig.add_trace(go.Scatter(x=df["año"], y=df["share"] * 100, name="% del total APF",
                             yaxis="y2", mode="lines+markers", line=dict(color="#c0392b")))
    fig.update_layout(
        title="Plazas de ramos militares y seguridad (7+13+36) en la Administración Pública Federal",
        xaxis_title="año", yaxis_title="miles de plazas",
        yaxis2=dict(title="% del total APF", overlaying="y", side="right"),
    )
    guardar_fig(fig, "cap3_plazas")
    return df


def actos_sedena():
    print("\n=== D. SEDENA — actos jurídicos (PNT, 2023-2025) ===")
    partes = []
    for f in sorted((RAIZ / "data/sedena").glob("buscador_solicitudes_43334_*.csv")):
        df = pl.read_csv(f, encoding="utf-8-sig", infer_schema_length=0)
        razon = [c for c in df.columns if c.startswith("Razón social")][0]
        partes.append(df.select(
            pl.col(C_AÑO).alias("año"), pl.col(C_TIPO).alias("tipo"),
            pl.col(C_OBJETO).alias("objeto"),
            pl.col(C_MONTO).str.replace_all(",", "").str.replace_all(r"\$", "")
            .str.strip_chars().cast(pl.Float64, strict=False).alias("monto"),
        ))
    df = pl.concat(partes)
    por_año = df.group_by("año").agg(pl.len().alias("actos"), pl.sum("monto")).sort("año")
    for r in por_año.iter_rows(named=True):
        print(f"  {r['año']}: {r['actos']:,} actos · monto {r['monto']/1e9 if r['monto'] else 0:.1f} mmdp")

    top = (df.filter(pl.col("tipo").is_not_null())
           .group_by("tipo").agg(pl.len().alias("n"), pl.sum("monto"))
           .sort("n", descending=True).head(8))
    print("  tipos más frecuentes:")
    for r in top.iter_rows(named=True):
        print(f"    {r['tipo'][:60]:<62} n={r['n']:,}  monto={0 if r['monto'] is None else r['monto']/1e9:.2f} mmdp")

    fig = go.Figure(go.Bar(
        x=top["n"], y=[t[:45] for t in top["tipo"]], orientation="h", marker_color="#7f8c1f"))
    fig.update_layout(
        title="SEDENA: actos jurídicos reportados a la PNT por tipo (2023-2025)",
        xaxis_title="número de actos", yaxis=dict(autorange="reversed"),
    )
    guardar_fig(fig, "cap3_sedena_actos", alto=500)


def main():
    nac, _ = gasto_militar()
    sobre_ejercicio(nac)
    plazas_militares()
    actos_sedena()


if __name__ == "__main__":
    main()
