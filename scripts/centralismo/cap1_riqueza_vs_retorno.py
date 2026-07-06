"""
Cap 1 — Riqueza estatal vs. retorno federal (H1: "entidades explotadas").

Aporte económico: share del PIB nominal por estado (PIBE corrientes, con y sin
minería petrolera). Retorno federal: transferencias federalizadas (Ramos 28
participaciones + 33 aportaciones, Cuenta Pública) y subsidios con entrega
geográfica real (subdataset Subsidios 2022-2025).

NO se usa el gasto total de Cuenta Pública por estado como "retorno": 39.6% se
registra en CDMX por domicilio administrativo (artefacto contable, ver DATA_OVERVIEW).

Figuras → centralismo/informe/figuras/cap1_*.png
Run: uv run python scripts/centralismo/cap1_riqueza_vs_retorno.py
"""

import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from comun import (CORTO, PIBE_MINERIA_PETROLERA, PIBE_TOTAL, RAIZ,
                   cargar_pobreza, cargar_poblacion, guardar_fig, leer_pibe)

AÑO = 2024
RAMOS_TRANSFER = [28, 33]  # participaciones + aportaciones federalizadas


def cargar_transferencias() -> pl.DataFrame:
    cp = pl.read_parquet(RAIZ / "data/presupuesto_federacion/cuenta_publica/cp_estado_ramo.parquet")
    return (
        cp.filter(pl.col("id_ramo").is_in(RAMOS_TRANSFER) & pl.col("cve_ent").is_between(1, 32))
        .group_by("ciclo", "cve_ent")
        .agg(pl.sum("monto_ejercido").alias("transfer"))
    )


def cargar_subsidios(año: int) -> pl.DataFrame:
    z = zipfile.ZipFile(RAIZ / f"data/presupuesto_federacion/cuenta_publica/subsidios_poblacion/Subsidios_CP{año}.zip")
    csv = [n for n in z.namelist() if n.endswith(".csv")][0]
    monto = "MONTO_PAGADO" if año == 2022 else "MONTO_EJERCIDO"
    df = pl.read_csv(z.read(csv), encoding="windows-1252", infer_schema_length=0)
    return (
        df.with_columns(
            pl.col("ID_ENTIDAD_FEDERATIVA").cast(pl.Int64, strict=False).alias("cve_ent"),
            pl.col(monto).cast(pl.Float64, strict=False).alias("subsidios"),
        )
        .filter(pl.col("cve_ent").is_between(1, 32))
        .group_by("cve_ent")
        .agg(pl.sum("subsidios"))
    )


def etiqueta_scatter(fig, x, y, textos):
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="markers+text", text=textos, textposition="top center",
        textfont=dict(size=9), marker=dict(size=8, color="#1f77b4"), showlegend=False,
    ))


def main():
    # ---- aporte: PIB nominal por estado (share), con y sin petróleo
    pib = leer_pibe(PIBE_TOTAL, bloque="Millones de pesos").rename({"valor": "pib"})
    petro = leer_pibe(PIBE_MINERIA_PETROLERA, bloque="Millones de pesos").rename({"valor": "pib_petro"})
    pib = pib.join(petro, on=["cve_ent", "año"], how="left").with_columns(
        pl.col("pib_petro").fill_null(0),
        (pl.col("pib") - pl.col("pib_petro").fill_null(0)).alias("pib_sp"),
    )
    pib = pib.with_columns(
        (pl.col("pib") / pl.col("pib").sum().over("año")).alias("share_pib"),
        (pl.col("pib_sp") / pl.col("pib_sp").sum().over("año")).alias("share_pib_sp"),
    )

    # ---- retorno: transferencias federalizadas
    tr = cargar_transferencias().with_columns(
        (pl.col("transfer") / pl.col("transfer").sum().over("ciclo")).alias("share_transfer")
    )

    pob = cargar_poblacion()
    base = (
        pib.filter(pl.col("año") == AÑO)
        .join(tr.filter(pl.col("ciclo") == AÑO), on="cve_ent")
        .join(pob.filter(pl.col("año") == AÑO), on="cve_ent")
        .with_columns(
            (pl.col("share_transfer") / pl.col("share_pib")).alias("razon"),
            (pl.col("share_transfer") / pl.col("share_pib_sp")).alias("razon_sp"),
            (pl.col("transfer") / pl.col("pob_total")).alias("transfer_pc"),
            (pl.col("pib") * 1e6 / pl.col("pob_total")).alias("pib_pc"),  # pesos
            pl.col("cve_ent").replace_strict(CORTO).alias("estado"),
        )
        .sort("razon_sp")
    )
    assert base.height == 32

    tot_tr = base["transfer"].sum() / 1e12
    print(f"\n=== {AÑO}: transferencias federalizadas (R28+R33) = {tot_tr:.2f} T MXN ===")
    print("\nRazón share_transferencias / share_PIB-sin-petróleo (ordenado):")
    for r in base.iter_rows(named=True):
        print(f"  {r['estado']:<22} razón={r['razon_sp']:.2f}  (con petróleo {r['razon']:.2f})  "
              f"share_PIB={r['share_pib_sp']*100:.2f}%  share_transf={r['share_transfer']*100:.2f}%  "
              f"transf_pc={r['transfer_pc']/1e3:.1f} mil MXN")

    aportadores = base.filter(pl.col("razon_sp") < 0.9)
    receptores = base.filter(pl.col("razon_sp") > 1.25)
    print(f"\nAportadores netos (razón<0.9): {aportadores['estado'].to_list()}")
    print(f"Receptores netos (razón>1.25): {receptores['estado'].to_list()}")

    # correlación per cápita: ¿las transferencias van a los estados pobres?
    dep22 = cargar_pobreza().filter(pl.col("año") == 2022).select("cve_ent", "pct_pobreza")
    b2 = base.join(dep22, on="cve_ent")
    c_pobreza = np.corrcoef(b2["transfer_pc"], b2["pct_pobreza"])[0, 1]
    c_pib = np.corrcoef(b2["transfer_pc"], b2["pib_pc"])[0, 1]
    print(f"\ncorr(transferencias_pc, pobreza 2022) = {c_pobreza:+.3f}")
    print(f"corr(transferencias_pc, PIB_pc)       = {c_pib:+.3f}")

    # ---- F1: share PIB (sin petróleo) vs share transferencias
    fig = go.Figure()
    m = float(max(base["share_pib_sp"].max(), base["share_transfer"].max()) * 105)
    fig.add_trace(go.Scatter(x=[0, m], y=[0, m], mode="lines",
                             line=dict(dash="dash", color="gray"), name="paridad (recibe = aporta)"))
    etiqueta_scatter(fig, (base["share_pib_sp"] * 100).to_list(),
                     (base["share_transfer"] * 100).to_list(), base["estado"].to_list())
    fig.update_layout(
        title=f"Peso económico vs. retorno federalizado, {AÑO}<br><sup>Debajo de la diagonal: el estado recibe menos de lo que pesa en la economía (PIB sin minería petrolera)</sup>",
        xaxis_title="Share del PIB nacional sin petróleo (%)",
        yaxis_title="Share de participaciones + aportaciones (%)",
    )
    guardar_fig(fig, "cap1_share_pib_vs_transferencias")

    # ---- F2: ranking de razón retorno/aporte
    b = base.sort("razon_sp")
    colores = ["#c0392b" if v < 1 else "#27ae60" for v in b["razon_sp"]]
    fig = go.Figure(go.Bar(x=b["razon_sp"], y=b["estado"], orientation="h",
                           marker_color=colores))
    fig.add_vline(x=1, line_dash="dash", line_color="black")
    fig.update_layout(
        title=f"Razón retorno/aporte, {AÑO} (share transferencias ÷ share PIB sin petróleo)<br><sup>&lt;1 = aportador neto (rojo) · &gt;1 = receptor neto (verde)</sup>",
        xaxis_title="razón", yaxis=dict(tickfont=dict(size=10)),
    )
    guardar_fig(fig, "cap1_razon_retorno_aporte", alto=800)

    # ---- F3: per cápita — ¿redistribución hacia pobreza o hacia riqueza?
    fig = go.Figure()
    etiqueta_scatter(fig, (b2["pib_pc"] / 1e3).to_list(), (b2["transfer_pc"] / 1e3).to_list(),
                     b2["estado"].to_list())
    z = np.polyfit(b2["pib_pc"] / 1e3, b2["transfer_pc"] / 1e3, 1)
    xs = np.linspace(float((b2["pib_pc"] / 1e3).min()), float((b2["pib_pc"] / 1e3).max()), 50)
    fig.add_trace(go.Scatter(x=xs, y=np.polyval(z, xs), mode="lines",
                             line=dict(color="#c0392b"), name=f"tendencia (pend.={z[0]:.3f})"))
    fig.update_layout(
        title=f"PIB per cápita vs. transferencias federalizadas per cápita, {AÑO}<br><sup>corr(transf_pc, pobreza)={c_pobreza:+.2f}; corr(transf_pc, PIB pc)={c_pib:+.2f}</sup>",
        xaxis_title="PIB per cápita (miles de MXN corrientes)",
        yaxis_title="Transferencias per cápita (miles de MXN)",
    )
    guardar_fig(fig, "cap1_percapita_redistribucion")

    # ---- F4: tendencia de la razón (2013-2025) para casos ilustrativos
    serie = (
        pib.select("cve_ent", "año", "share_pib_sp")
        .join(tr.rename({"ciclo": "año"}), on=["cve_ent", "año"])
        .with_columns((pl.col("share_transfer") / pl.col("share_pib_sp")).alias("razon_sp"))
        .filter(pl.col("año").is_between(2013, 2024))
    )
    casos = {4: "Campeche", 27: "Tabasco", 19: "Nuevo León", 9: "CDMX",
             7: "Chiapas", 12: "Guerrero", 20: "Oaxaca", 15: "Edomex"}
    fig = go.Figure()
    for cve, nombre in casos.items():
        s = serie.filter(pl.col("cve_ent") == cve).sort("año")
        fig.add_trace(go.Scatter(x=s["año"], y=s["razon_sp"], mode="lines+markers", name=nombre))
    fig.add_hline(y=1, line_dash="dash", line_color="gray")
    fig.update_layout(
        title="Razón retorno/aporte 2013-2024 — casos ilustrativos",
        xaxis_title="año", yaxis_title="share transferencias ÷ share PIB sin petróleo",
    )
    guardar_fig(fig, "cap1_tendencia_razon")

    # ---- F5: subsidios con entrega real per cápita vs PIB pc
    sub = cargar_subsidios(AÑO).join(base.select("cve_ent", "estado", "pib_pc", "pob_total"), on="cve_ent")
    sub = sub.with_columns((pl.col("subsidios") / pl.col("pob_total")).alias("sub_pc"))
    c_sub = np.corrcoef(sub.join(dep22, on="cve_ent")["sub_pc"],
                        sub.join(dep22, on="cve_ent")["pct_pobreza"])[0, 1]
    print(f"corr(subsidios_pc, pobreza 2022)      = {c_sub:+.3f}")
    print(f"subsidios con entrega real {AÑO}: {sub['subsidios'].sum()/1e12:.2f} T MXN")
    fig = go.Figure()
    etiqueta_scatter(fig, (sub["pib_pc"] / 1e3).to_list(), (sub["sub_pc"] / 1e3).to_list(),
                     sub["estado"].to_list())
    fig.update_layout(
        title=f"Subsidios federales con entrega geográfica real per cápita, {AÑO}<br><sup>corr(subsidios_pc, pobreza 2022)={c_sub:+.2f} — fuente: Subsidios Cuenta Pública</sup>",
        xaxis_title="PIB per cápita (miles de MXN)",
        yaxis_title="Subsidios per cápita (miles de MXN)",
    )
    guardar_fig(fig, "cap1_subsidios_entrega_real")

    # números clave para el informe
    cam = base.filter(pl.col("cve_ent") == 4).row(0, named=True)
    print(f"\nCampeche: share PIB {cam['share_pib']*100:.2f}% (sin petróleo {cam['share_pib_sp']*100:.2f}%), "
          f"share transferencias {cam['share_transfer']*100:.2f}%")


if __name__ == "__main__":
    main()
