"""
Cap 1 (extensión) — Razón retorno/aporte con el aporte MEDIDO (H1 re-test).

El Cap 1 aproximó el aporte fiscal estatal con el share de PIB porque el SAT no
publica recaudación por entidad. Ahora existe un aporte medido:
informe_data/recaudacion_imputada_estatal.parquet (2018-2024) — recaudación
tributaria federal imputada por incidencia económica (método Ríos & Saucedo 2025,
generado por preparar_recaudacion_imputada.py).

Test: razón = share_transferencias(R28+R33) ÷ share_recaudación_imputada.
  - ¿CDMX sigue siendo el mayor aportador neto (razón mínima)?
  - ¿Los aportadores/receptores netos cambian respecto a la razón basada en PIB?

Nota metodológica: la imputación reparte con pesos de consumo/ingreso de los
hogares (ENIGH) y factores del Censo Económico, así que correlaciona con el PIB
por construcción; lo informativo son las DIVERGENCIAS (estados fronterizos con
consumo alto, Edomex, estados petroleros donde el PIB no es renta gravable local).
Solo conceptos tributarios (las cuotas IMSS/INFONAVIT/ISSSTE traen monto null y
el proxy de ISSSTE hereda el sesgo de sede CDMX — se excluyen).

Figuras → centralismo/informe/figuras/cap1_imputada_*.png
Run: uv run python scripts/centralismo/cap1_recaudacion_imputada.py
"""

import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from comun import (CORTO, PIBE_MINERIA_PETROLERA, PIBE_TOTAL, RAIZ,
                   guardar_fig, leer_pibe)

AÑO = 2024
RAMOS_TRANSFER = [28, 33]  # participaciones + aportaciones federalizadas


def cargar_transferencias() -> pl.DataFrame:
    cp = pl.read_parquet(RAIZ / "informe_data/cp_estado_ramo.parquet")
    return (
        cp.filter(pl.col("id_ramo").is_in(RAMOS_TRANSFER) & pl.col("cve_ent").is_between(1, 32))
        .group_by("ciclo", "cve_ent")
        .agg(pl.sum("monto_ejercido").alias("transfer"))
        .with_columns((pl.col("transfer") / pl.col("transfer").sum().over("ciclo")).alias("share_transfer"))
    )


def cargar_aporte_imputado() -> pl.DataFrame:
    """Share estatal de la recaudación tributaria federal imputada (2018-2024)."""
    rec = pl.read_parquet(RAIZ / "informe_data/recaudacion_imputada_estatal.parquet")
    rec = rec.filter(pl.col("monto_imputado_millones_pesos").is_not_null()).with_columns(
        pl.col("cve_ent").cast(pl.Int64)
    )
    agg = (
        rec.group_by("anio", "cve_ent")
        .agg(pl.sum("monto_imputado_millones_pesos").alias("recaudacion"))
        .with_columns(
            (pl.col("recaudacion") / pl.col("recaudacion").sum().over("anio")).alias("share_recaudacion")
        )
        .rename({"anio": "año"})
        .with_columns(pl.col("año").cast(pl.Int64))
    )
    assert agg.filter(pl.col("año") == AÑO).height == 32
    return agg


def cargar_share_pib_sp() -> pl.DataFrame:
    """Share de PIB sin minería petrolera (la vara del Cap 1 original)."""
    pib = leer_pibe(PIBE_TOTAL, bloque="Millones de pesos").rename({"valor": "pib"})
    petro = leer_pibe(PIBE_MINERIA_PETROLERA, bloque="Millones de pesos").rename({"valor": "pib_petro"})
    pib = pib.join(petro, on=["cve_ent", "año"], how="left").with_columns(
        (pl.col("pib") - pl.col("pib_petro").fill_null(0)).alias("pib_sp")
    )
    return pib.with_columns(
        (pl.col("pib_sp") / pl.col("pib_sp").sum().over("año")).alias("share_pib_sp")
    ).select("cve_ent", "año", "share_pib_sp")


def main():
    aporte = cargar_aporte_imputado()
    tr = cargar_transferencias().rename({"ciclo": "año"})
    pib_sp = cargar_share_pib_sp()

    base = (
        aporte.filter(pl.col("año") == AÑO)
        .join(tr.filter(pl.col("año") == AÑO), on=["cve_ent", "año"])
        .join(pib_sp.filter(pl.col("año") == AÑO), on=["cve_ent", "año"])
        .with_columns(
            (pl.col("share_transfer") / pl.col("share_recaudacion")).alias("razon_imputada"),
            (pl.col("share_transfer") / pl.col("share_pib_sp")).alias("razon_pib"),
            pl.col("cve_ent").replace_strict(CORTO).alias("estado"),
        )
        .sort("razon_imputada")
    )
    assert base.height == 32

    print(f"\n=== {AÑO}: razón retorno/aporte — aporte MEDIDO (recaudación imputada) vs PIB ===")
    print(f"{'Estado':<22}{'razón imputada':>15}{'razón PIB-sp':>14}{'share_recaud':>14}{'share_transf':>14}")
    for r in base.iter_rows(named=True):
        print(f"  {r['estado']:<20}{r['razon_imputada']:>13.2f}{r['razon_pib']:>14.2f}"
              f"{r['share_recaudacion']*100:>13.2f}%{r['share_transfer']*100:>13.2f}%")

    c = np.corrcoef(base["razon_imputada"], base["razon_pib"])[0, 1]
    print(f"\ncorr(razón imputada, razón PIB) = {c:+.3f}")

    # ¿quién cambia de lado de la paridad (1.0)?
    cambian = base.filter(
        ((pl.col("razon_imputada") < 1) & (pl.col("razon_pib") > 1)) |
        ((pl.col("razon_imputada") > 1) & (pl.col("razon_pib") < 1))
    )
    print(f"estados que cambian de lado de la paridad: {cambian['estado'].to_list()}")

    aportadores = base.filter(pl.col("razon_imputada") < 0.9)["estado"].to_list()
    receptores = base.filter(pl.col("razon_imputada") > 1.25)["estado"].to_list()
    print(f"\nAportadores netos (razón<0.9):  {aportadores}")
    print(f"Receptores netos (razón>1.25): {receptores}")

    cdmx = base.filter(pl.col("cve_ent") == 9).row(0, named=True)
    minimo = base.row(0, named=True)
    print(f"\nCDMX: razón imputada = {cdmx['razon_imputada']:.2f} "
          f"(lugar {base['estado'].to_list().index('CDMX')+1}/32, mínimo = {minimo['estado']} {minimo['razon_imputada']:.2f})")

    # ---- F1: comparación de razones (scatter con diagonal)
    fig = go.Figure()
    m = float(max(base["razon_imputada"].max(), base["razon_pib"].max()) * 1.05)
    fig.add_trace(go.Scatter(x=[0, m], y=[0, m], mode="lines",
                             line=dict(dash="dash", color="gray"), name="misma razón"))
    fig.add_hline(y=1, line_dash="dot", line_color="#999")
    fig.add_vline(x=1, line_dash="dot", line_color="#999")
    fig.add_trace(go.Scatter(
        x=base["razon_pib"], y=base["razon_imputada"], mode="markers+text",
        text=base["estado"], textposition="top center", textfont=dict(size=9),
        marker=dict(size=8, color="#1f77b4"), showlegend=False,
    ))
    fig.update_layout(
        title=f"Razón retorno/aporte {AÑO}: aporte por PIB vs aporte medido (recaudación imputada)<br>"
              f"<sup>corr={c:+.2f}; cuadrante inferior-izquierdo = aportador neto en ambas varas</sup>",
        xaxis_title="razón con share de PIB sin petróleo (vara original Cap 1)",
        yaxis_title="razón con share de recaudación imputada",
    )
    guardar_fig(fig, "cap1_imputada_comparacion_razones")

    # ---- F2: ranking razón imputada
    colores = ["#c0392b" if v < 1 else "#27ae60" for v in base["razon_imputada"]]
    fig = go.Figure(go.Bar(x=base["razon_imputada"], y=base["estado"], orientation="h",
                           marker_color=colores))
    fig.add_vline(x=1, line_dash="dash", line_color="black")
    fig.update_layout(
        title=f"Razón retorno/aporte {AÑO} — aporte medido<br>"
              "<sup>share transferencias R28+R33 ÷ share recaudación tributaria imputada · &lt;1 aportador neto</sup>",
        xaxis_title="razón", yaxis=dict(tickfont=dict(size=10)),
    )
    guardar_fig(fig, "cap1_imputada_razon_ranking", alto=800)

    # ---- F3: serie 2018-2024 casos ilustrativos
    serie = (
        aporte.join(tr, on=["cve_ent", "año"])
        .with_columns((pl.col("share_transfer") / pl.col("share_recaudacion")).alias("razon"))
    )
    casos = {9: "CDMX", 4: "Campeche", 27: "Tabasco", 19: "Nuevo León",
             7: "Chiapas", 12: "Guerrero", 15: "Edomex"}
    fig = go.Figure()
    for cve, nombre in casos.items():
        s = serie.filter(pl.col("cve_ent") == cve).sort("año")
        fig.add_trace(go.Scatter(x=s["año"], y=s["razon"], mode="lines+markers", name=nombre))
    fig.add_hline(y=1, line_dash="dash", line_color="gray")
    fig.update_layout(
        title="Razón retorno/aporte con recaudación imputada, 2018-2024 — casos ilustrativos",
        xaxis_title="año", yaxis_title="share transferencias ÷ share recaudación imputada",
    )
    guardar_fig(fig, "cap1_imputada_tendencia")

    # test de veredicto: flujo centro/norte → sur
    sur = [7, 12, 20]  # Chiapas, Guerrero, Oaxaca
    r_sur = base.filter(pl.col("cve_ent").is_in(sur))["razon_imputada"].to_list()
    print(f"\nSur (Chiapas/Guerrero/Oaxaca) razones: {[f'{v:.2f}' for v in r_sur]}")
    assert all(v > 1 for v in r_sur), "el sur dejó de ser receptor neto — revisar"


if __name__ == "__main__":
    main()
