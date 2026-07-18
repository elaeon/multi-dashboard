"""
Cap 4 (extensión) — Dependencia fiscal estatal con la jerarquía COMPLETA de
ingresos (EFIPEM), 2010-2024. Re-test de H4.

El Cap 4.B midió la dependencia con ingresos propios = impuestos + derechos
(recaudacion_local, Transparencia Presupuestaria) — sobreestimación reconocida
en las Limitaciones del INFORME. EFIPEM (INEGI) publica por estado la jerarquía
completa: aquí ingresos propios = Impuestos + Derechos + Productos +
Aprovechamientos + Contribuciones de Mejoras + Cuotas y Aportaciones de
Seguridad Social, y transferencias = Participaciones + Aportaciones federales —
ambos lados de la MISMA fuente.

Se excluyen del denominador: Financiamiento (deuda, no ingreso recurrente),
Disponibilidad inicial (arrastre contable, solo algunos años) y Otros ingresos
(cajón residual con ingresos por cuenta de terceros y extraordinarios — incluirlo
haría la dependencia no comparable entre estados que lo usan y no).

Trampas EFIPEM aplicadas: utf-8 (no latin-1), solo CATEGORIA=='Capítulo' (la
jerarquía anida 5 niveles), CDMX viene en zip aparte del estatal.

Figuras → centralismo/informe/figuras/cap4_dependencia_efipem*.png
Run: uv run python scripts/centralismo/cap4_dependencia_efipem.py
"""

import io
import sys
import zipfile
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from comun import CORTO, RAIZ, guardar_fig

DIR_EFIPEM = RAIZ / "data/inegi/efipem"
AÑOS = list(range(2010, 2025))

PROPIOS = ["Impuestos", "Derechos", "Productos", "Aprovechamientos",
           "Contribuciones de Mejoras", "Cuotas y Aportaciones de Seguridad Social"]
TRANSFER = ["Participaciones federales", "Aportaciones federales"]


def cargar_efipem() -> pl.DataFrame:
    """Panel estado × año × capítulo de ingresos (estatal + cdmx), 2010-2024."""
    partes = []
    for zipname in ("conjunto_de_datos_efipem_estatal_csv.zip",
                    "conjunto_de_datos_efipem_cdmx_csv.zip"):
        with zipfile.ZipFile(DIR_EFIPEM / zipname) as zf:
            csvs = [n for n in zf.namelist()
                    if "conjunto_de_datos" in n and n.endswith(".csv")]
            for nombre in sorted(csvs):
                año = int(nombre.rsplit("_", 1)[-1].removesuffix(".csv"))
                if año not in AÑOS:
                    continue
                with zf.open(nombre) as f:
                    df = pd.read_csv(io.BytesIO(f.read()), encoding="utf-8", low_memory=False)
                df = df[(df["TEMA"] == "Ingresos") & (df["CATEGORIA"] == "Capítulo")]
                df["DESCRIPCION_CATEGORIA"] = df["DESCRIPCION_CATEGORIA"].str.replace("\xa0", " ", regex=False)
                partes.append(pl.from_pandas(
                    df[["ANIO", "CVE_ENT", "DESCRIPCION_CATEGORIA", "VALOR"]]
                ))
    return pl.concat(partes)


def main():
    ef = cargar_efipem()

    panel = (
        ef.with_columns(
            pl.when(pl.col("DESCRIPCION_CATEGORIA").is_in(PROPIOS)).then(pl.lit("propios"))
            .when(pl.col("DESCRIPCION_CATEGORIA").is_in(TRANSFER)).then(pl.lit("transfer"))
            .otherwise(pl.lit("otro")).alias("grupo")
        )
        .filter(pl.col("grupo") != "otro")
        .group_by("ANIO", "CVE_ENT", "grupo")
        .agg(pl.sum("VALOR"))
        .pivot(values="VALOR", index=["ANIO", "CVE_ENT"], on="grupo")
        .fill_null(0)
        .with_columns((pl.col("transfer") / (pl.col("transfer") + pl.col("propios"))).alias("dependencia"))
        .rename({"ANIO": "año", "CVE_ENT": "cve_ent"})
    )
    assert panel.filter(pl.col("año") == 2024)["cve_ent"].n_unique() == 32
    assert panel["dependencia"].is_between(0, 1).all()

    # ---- serie nacional
    nal = (panel.group_by("año")
           .agg((pl.sum("transfer") / (pl.sum("transfer") + pl.sum("propios"))).alias("dependencia"))
           .sort("año"))
    print("\n=== Dependencia fiscal nacional (EFIPEM: transferencias / [transf + propios completos]) ===")
    for r in nal.iter_rows(named=True):
        print(f"  {r['año']}: {r['dependencia']*100:.1f}%")

    # ---- comparación con la vara previa del Cap 4 (impuestos+derechos, recaudacion_local)
    prev = []
    for archivo, monto in [("mapa_impuestos_estatales.csv", "MONTO_IMPUESTOS"),
                           ("mapa_derechos_estatales.csv", "MONTO_DERECHOS")]:
        df = pl.read_csv(RAIZ / "data/presupuesto_federacion/recaudacion_local" / archivo,
                         encoding="latin-1")
        prev.append(df.select(pl.col("CICLO").alias("año"),
                              pl.col("ID_ENTIDAD_FEDERATIVA").cast(pl.Int64).alias("cve_ent"),
                              pl.col(monto).alias("monto")))
    propios_prev = pl.concat(prev).group_by("año", "cve_ent").agg(pl.sum("monto").alias("propios_prev"))

    cmp = (panel.join(propios_prev, on=["año", "cve_ent"])
           .group_by("año")
           .agg((pl.sum("transfer") / (pl.sum("transfer") + pl.sum("propios_prev"))).alias("dep_prev"),
                (pl.sum("transfer") / (pl.sum("transfer") + pl.sum("propios"))).alias("dep_efipem"))
           .sort("año"))
    print("\n=== Vara previa (imp+der) vs EFIPEM completo — nacional ===")
    for r in cmp.iter_rows(named=True):
        print(f"  {r['año']}: previa {r['dep_prev']*100:.1f}%  →  completa {r['dep_efipem']*100:.1f}%  "
              f"(Δ {-(r['dep_prev']-r['dep_efipem'])*100:+.1f} pp)")

    # ---- ranking 2024 y cambio 2013→2024
    d24 = panel.filter(pl.col("año") == 2024).with_columns(
        pl.col("cve_ent").replace_strict(CORTO).alias("estado")).sort("dependencia", descending=True)
    d13 = panel.filter(pl.col("año") == 2013).select("cve_ent", pl.col("dependencia").alias("dep13"))
    d24j = d24.join(d13, on="cve_ent")
    sube = (d24j["dependencia"] > d24j["dep13"]).sum()
    print(f"\n2024 más dependientes: " + ", ".join(
        f"{r['estado']}={r['dependencia']*100:.0f}%" for r in d24.head(4).iter_rows(named=True)))
    print(f"2024 menos dependientes: " + ", ".join(
        f"{r['estado']}={r['dependencia']*100:.0f}%" for r in d24.tail(4).iter_rows(named=True)))
    print(f"estados donde la dependencia SUBIÓ 2013→2024: {sube}/32")

    # ---- F1: serie nacional, ambas varas
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cmp["año"], y=cmp["dep_prev"] * 100, mode="lines+markers",
                             name="vara previa (impuestos+derechos)", line=dict(dash="dot")))
    fig.add_trace(go.Scatter(x=cmp["año"], y=cmp["dep_efipem"] * 100, mode="lines+markers",
                             name="EFIPEM completo (6 capítulos propios)"))
    fig.update_layout(
        title="Dependencia fiscal estatal nacional, 2010-2024 — dos varas<br>"
              "<sup>transferencias R28+R33 (EFIPEM) ÷ (transferencias + ingresos propios) · la vara previa sobreestimaba</sup>",
        xaxis_title="año", yaxis_title="dependencia (%)",
    )
    guardar_fig(fig, "cap4_dependencia_efipem_series")

    # ---- F2: ranking 2024 con marca 2013
    d = d24j.sort("dependencia")
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=d["dependencia"] * 100, y=d["estado"], orientation="h",
        marker_color=[("#c0392b" if a > b else "#27ae60") for a, b in zip(d["dependencia"], d["dep13"])],
        name="2024",
    ))
    fig.add_trace(go.Scatter(x=d["dep13"] * 100, y=d["estado"], mode="markers",
                             marker=dict(symbol="line-ns-open", size=12, color="black"), name="2013"))
    fig.update_layout(
        title="Dependencia fiscal 2024 (barra) vs 2013 (marca) — medición EFIPEM completa<br>"
              "<sup>rojo = subió · ingresos propios: impuestos, derechos, productos, aprovechamientos, mejoras, cuotas SS</sup>",
        xaxis_title="dependencia (%)", yaxis=dict(tickfont=dict(size=10)),
    )
    guardar_fig(fig, "cap4_dependencia_efipem_ranking", alto=800)


if __name__ == "__main__":
    main()
