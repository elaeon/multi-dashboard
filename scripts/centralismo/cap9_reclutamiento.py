"""
Cap 9.D — El canal de oferta: jóvenes sin escuela ni empleo formal (ENOE 2025).

El Cap 9 midió dónde está la renta (la demanda de violencia); este script mide la
oferta: la "población reclutable" (Prieto-Curiel et al. 2023) aproximada como los
jóvenes de 15-29 que ni estudian ni tienen empleo formal ("sin IMSS ni escuela",
puntos.md §5). El test de la reconciliación de casos_de_estudio (ubicación vs
reclutamiento): la bolsa de reclutables NO debería correlacionar con la violencia
local (la pobreza no ubica la violencia) pero SÍ con la expulsión migratoria (la
pobreza exporta mano de obra hacia los corredores).

  A. ENOE sdem 2025 (4 trimestres, fac_tri): % reclutable por estado + NEET estricto.
  B. Cruces n=32: homicidio SESNSP (prom. 2023-24), desaparecidos, migración neta
     (Cap 11), pobreza, índice de potencial (Cap 5).

Figura → centralismo/informe/figuras/cap9_reclutables.png
Run: uv run python scripts/centralismo/cap9_reclutamiento.py
"""

import gc
import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from comun import CORTO, RAIZ, cargar_pobreza, cargar_poblacion, guardar_fig
from cap5_factor_competencia import componentes
from cap5_remesas import COLS5, indice_pca
from cap11_migracion_informalidad import migracion_neta

MESES = ["Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio", "Julio",
         "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"]
TRIMESTRES = [(2025, 1), (2025, 2), (2025, 3), (2025, 4)]
COLS_SDEM = ["ent", "cve_ent", "eda", "cs_p17", "emp_ppal", "clase2", "fac_tri"]


def leer_sdem(año: int, t: int) -> pl.DataFrame:
    """sdem de un trimestre, solo columnas necesarias (ent→cve_ent, fac_tri>0)."""
    stem = f"{año}_{t}t"
    inner = (f"conjunto_de_datos_sdem_enoe_{stem}/conjunto_de_datos/"
             f"conjunto_de_datos_sdem_enoe_{stem}.csv")
    with zipfile.ZipFile(RAIZ / f"data/inegi/enoe/conjunto_de_datos_enoe_{stem}_csv.zip") as z:
        with z.open(inner) as f:
            data = f.read()
    header = pd.read_csv(io.BytesIO(data), encoding="latin-1", nrows=0)
    usar = [c for c in COLS_SDEM if c in header.columns]
    df = pd.read_csv(io.BytesIO(data), encoding="latin-1", usecols=usar,
                     dtype={"eda": str, "cs_p17": str})
    del data
    gc.collect()
    if "ent" in df.columns:  # quiebre de esquema 2025Q3: ent → cve_ent
        df = df.rename(columns={"ent": "cve_ent"})
    out = (pl.from_pandas(df)
           .with_columns(
               pl.col("cve_ent").cast(pl.Int64),
               pl.col("eda").str.strip_chars().cast(pl.Float64, strict=False),
               pl.col("cs_p17").str.strip_chars(),
               pl.col("fac_tri").cast(pl.Float64))
           .filter(pl.col("fac_tri") > 0))
    assert out["cve_ent"].n_unique() == 32
    return out


def cargar_enoe() -> pl.DataFrame:
    """% de jóvenes 15-29 reclutables (ni escuela ni empleo formal) por estado."""
    partes = [leer_sdem(a, t) for a, t in TRIMESTRES]
    sdem = pl.concat(partes)
    del partes
    gc.collect()

    # ancla de validación: informalidad nacional ≈55% (DATA_OVERVIEW / boletín INEGI)
    ocupados = sdem.filter(pl.col("clase2") == 1)
    informalidad = (ocupados.filter(pl.col("emp_ppal") == 1)["fac_tri"].sum()
                    / ocupados["fac_tri"].sum() * 100)
    print(f"  ancla: informalidad nacional 2025 = {informalidad:.1f}% (esperado ~55%)")
    assert 52 < informalidad < 58

    jov = sdem.filter(pl.col("eda").is_between(15, 29)).with_columns(
        # sin escuela (cs_p17: 1=asiste) y sin empleo formal (emp_ppal: 2=formal)
        ((pl.col("cs_p17") != "1") & (pl.col("emp_ppal") != 2)).alias("reclutable"),
        ((pl.col("cs_p17") != "1") & (pl.col("clase2") != 1)).alias("neet"))
    por_edo = (jov.group_by("cve_ent")
               .agg((pl.col("fac_tri").filter(pl.col("reclutable")).sum()
                     / pl.col("fac_tri").sum() * 100).alias("reclutables_pct"),
                    (pl.col("fac_tri").filter(pl.col("neet")).sum()
                     / pl.col("fac_tri").sum() * 100).alias("neet_pct"),
                    (pl.col("fac_tri").sum() / len(TRIMESTRES)).alias("jovenes")))
    nac_rec = (jov.filter(pl.col("reclutable"))["fac_tri"].sum()
               / jov["fac_tri"].sum() * 100)
    nac_neet = (jov.filter(pl.col("neet"))["fac_tri"].sum() / jov["fac_tri"].sum() * 100)
    print(f"  nacional 15-29 ({jov['fac_tri'].sum() / len(TRIMESTRES) / 1e6:.1f} M): "
          f"reclutables (ni escuela ni empleo formal) = {nac_rec:.1f}% · "
          f"NEET estricto (ni escuela ni empleo) = {nac_neet:.1f}%")
    return por_edo


def cargar_violencia() -> pl.DataFrame:
    """Tasa estatal de homicidio doloso (promedio 2023-2024) y desaparecidos pc."""
    hom = (pl.scan_parquet(RAIZ / "data/incidencia_delictiva/incidencia_fuero_comun/"
                           "incidencia_delictiva_fuero_comun.parquet")
           .filter(pl.col("Año").is_in([2023, 2024])
                   & (pl.col("Subtipo de delito") == "Homicidio doloso"))
           .with_columns(pl.sum_horizontal(MESES).alias("casos"))
           .group_by("Clave_Ent").agg((pl.sum("casos") / 2).alias("casos"))
           .collect()
           .rename({"Clave_Ent": "cve_ent"}).with_columns(pl.col("cve_ent").cast(pl.Int64)))
    des = (pl.read_csv(RAIZ / "data/datamx/desaparecidos/desaparecidos.csv",
                       infer_schema_length=0)
           .with_columns(pl.col("CVE_ENT").cast(pl.Int64, strict=False).alias("cve_ent"))
           .filter(pl.col("cve_ent").is_between(1, 32))
           .group_by("cve_ent").agg(pl.len().alias("des")))
    pob = cargar_poblacion().filter(pl.col("año") == 2024)
    return (hom.join(des, on="cve_ent").join(pob, on="cve_ent")
            .with_columns((pl.col("casos") / pl.col("pob_total") * 1e5).alias("tasa_hom"),
                          (pl.col("des") / pl.col("pob_total") * 1e5).alias("tasa_des")))


def main():
    print("=== A. La bolsa de reclutables (ENOE sdem 2025, 4 trimestres) ===")
    enoe = cargar_enoe()

    comp = componentes()
    pc1, _ = indice_pca(comp, COLS5)
    base = (enoe.join(cargar_violencia(), on="cve_ent")
            .join(migracion_neta().select("cve_ent", "mig_rate"), on="cve_ent")
            .join(cargar_pobreza().filter(pl.col("año") == 2022)
                  .select("cve_ent", "pct_pobreza"), on="cve_ent")
            .join(comp.with_columns(pl.Series("indice", pc1))
                  .select("cve_ent", "indice"), on="cve_ent")
            .with_columns(pl.col("cve_ent").replace_strict(CORTO).alias("estado")))
    assert base.height == 32

    top = base.sort("reclutables_pct", descending=True)
    print("  top-5 reclutables:", " · ".join(
        f"{r['estado']}={r['reclutables_pct']:.0f}%" for r in top.head(5).iter_rows(named=True)))
    print("  bottom-3:", " · ".join(
        f"{r['estado']}={r['reclutables_pct']:.0f}%" for r in top.tail(3).iter_rows(named=True)))

    print("\n=== B. ¿A qué correlaciona la oferta? (n=32) ===")
    for col, nombre in [("tasa_hom", "homicidios 2023-24"),
                        ("tasa_des", "desaparecidos acum."),
                        ("mig_rate", "migración neta (Cap 11)"),
                        ("pct_pobreza", "pobreza 2022"),
                        ("indice", "índice de potencial (Cap 5)")]:
        c = np.corrcoef(base["reclutables_pct"], base[col])[0, 1]
        c_neet = np.corrcoef(base["neet_pct"], base[col])[0, 1]
        print(f"  corr(reclutables, {nombre:<26}) = {c:+.2f}   [NEET estricto {c_neet:+.2f}]")

    print("  corr(reclutables, NEET estricto) = "
          f"{np.corrcoef(base['reclutables_pct'], base['neet_pct'])[0, 1]:+.2f}")
    print("  caveats: medida estatal (el reclutamiento opera a nivel municipio/corredor);"
          " transversal 2025; 'reclutable' = condición económica, no propensión")

    c_hom = np.corrcoef(base["reclutables_pct"], base["tasa_hom"])[0, 1]
    c_mig = np.corrcoef(base["reclutables_pct"], base["mig_rate"])[0, 1]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=base["reclutables_pct"], y=base["tasa_hom"], mode="markers+text",
        text=base["estado"], textposition="top center", textfont=dict(size=9),
        showlegend=False,
        marker=dict(size=10, color=base["mig_rate"], colorscale="RdBu",
                    cmid=0, colorbar=dict(title="migración<br>neta (‰)"))))
    fig.update_layout(
        title=f"La oferta de reclutables no está donde está la violencia "
              f"(corr={c_hom:+.2f})<br>"
              f"<sup>jóvenes 15-29 sin escuela ni empleo formal (ENOE 2025) vs homicidio "
              f"doloso · color = migración neta (corr con reclutables {c_mig:+.2f}: "
              f"rojo = expulsor)</sup>",
        xaxis_title="% de jóvenes 15-29 sin escuela ni empleo formal",
        yaxis_title="homicidios dolosos por 100k (prom. 2023-24)")
    guardar_fig(fig, "cap9_reclutables")


if __name__ == "__main__":
    main()
