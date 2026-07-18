"""
Cap 3 (extensión) — Militarización medida con censos federales (H3 re-test).

Dos mediciones independientes de las plazas APF y ramos de Cuenta Pública que
usa el Cap 3:

  A. CNSPF (censo INEGI de la Guardia Nacional, ediciones 2020-2025 = años de
     referencia 2019-2024): headcount censal de GN y su presupuesto
     aprobado/ejercido. Contraste clave con Cap 3.B: los RAMOS militares
     (7/13/36) sobre-ejercen sistemáticamente, pero la GN como institución
     SUB-ejerce ~40-60% de su aprobado — consistente con que su costo real se
     carga al ramo SEDENA. NUNCA fusionar con Policía Federal (2018-2019).

  B. FAP (informes semestrales al Senado, Art. Quinto Transitorio): despliegue
     TERRITORIAL de ejército/marina (2023-2026) y GN (2024-2025) por entidad —
     la dimensión estatal que las plazas APF (nacionales) no tienen. Test:
     ¿el despliegue sigue a la población o a la violencia (homicidios SESNSP)?

Figuras → centralismo/informe/figuras/cap3_censos_*.png
Run: uv run python scripts/centralismo/cap3_censos_federales.py
"""

import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from comun import CORTO, RAIZ, cargar_poblacion, con_cve, guardar_fig

DIR_CNSPF = RAIZ / "data/inegi/cnspf"
DIR_FAP = RAIZ / "data/fap"

# (edición, zip, tabla) de personal GN con columna sexostt; ref = edición - 1
PERSONAL_GN = {
    2020: ("m1_pers_guardia_nac_cnspf2020_csv.zip", "m1p1_5"),
    2021: ("m1_pers_guardia_nac_cnspf2021_csv.zip", "m1s1p5"),
    2022: ("m1_pers_guardia_nac_cnspf2022_csv.zip", "m1s1p4"),
    2023: ("m1_pers_guardia_nac_cnspf2023_csv.zip", "m1s1p4"),
    2024: ("m1_conjunto_de_datos_pers_guardia_nac_cnspf2024_csv.zip", "m1s2p4"),
    2025: ("m1_conjunto_de_datos_rec_huma_cnspf2025_csv.zip", "m1s2p3"),
}

# presupuesto GN (solo ediciones 2023-2025 = ref 2022-2024); presup2=aprobado presup3=ejercido
PRESUP_GN = {
    2023: "m1_rec_presup_cnspf2023_csv.zip",
    2024: "m1_rec_presup_cnspf2024_csv.zip",
    2025: "m1_conjunto_de_datos_rec_presup_cnspf2025_csv.zip",
}


def leer_tabla_cnspf(edicion: int, zipname: str, tabla: str) -> pd.DataFrame:
    with zipfile.ZipFile(DIR_CNSPF / str(edicion) / zipname) as zf:
        nombre = next(n for n in zf.namelist()
                      if f"{tabla}_cnspf{edicion}.csv" in n.lower()
                      and "diccionario" not in n and "0_indice" not in n)
        with zf.open(nombre) as f:
            return pd.read_csv(io.BytesIO(f.read()), encoding="latin-1")


def headcount_gn() -> pl.DataFrame:
    filas = []
    for edicion, (zipname, tabla) in PERSONAL_GN.items():
        df = leer_tabla_cnspf(edicion, zipname, tabla)
        col = next(c for c in df.columns if "sexostt" in c.lower())
        total = pd.to_numeric(df[col], errors="coerce").sum()
        filas.append((edicion - 1, int(total)))
    return pl.DataFrame(filas, schema=["año", "efectivos"], orient="row")


def presupuesto_gn() -> pl.DataFrame:
    filas = []
    for edicion, zipname in PRESUP_GN.items():
        with zipfile.ZipFile(DIR_CNSPF / str(edicion) / zipname) as zf:
            nombre = sorted(n for n in zf.namelist()
                            if n.lower().endswith(".csv") and "conjunto_de_datos" in n
                            and "diccionario" not in n and "0_indice" not in n)[0]
            with zf.open(nombre) as f:
                df = pd.read_csv(io.BytesIO(f.read()), encoding="latin-1")
        filas.append((edicion - 1,
                      float(pd.to_numeric(df["presup2"], errors="coerce").sum()),
                      float(pd.to_numeric(df["presup3"], errors="coerce").sum())))
    return pl.DataFrame(filas, schema=["año", "aprobado", "ejercido"], orient="row")


def despliegue_fap() -> pl.DataFrame:
    """Último dato semestral disponible por fuerza y año-carpeta, por entidad."""
    partes = []
    for carpeta in sorted(DIR_FAP.iterdir()):
        if not carpeta.is_dir():
            continue
        año = int(carpeta.name)
        for csv in carpeta.glob("efectivos_*_desplegados_*.csv"):
            fuerza = ("gn" if "guardia" in csv.name
                      else "marina" if "naval" in csv.name else "ejercito")
            df = pd.read_csv(csv)
            cols_tot = [c for c in df.columns if c.startswith("total")]
            # el informe más reciente no-vacío del archivo (columna octavo 2026 viene vacía)
            col = next(c for c in reversed(cols_tot) if df[c].notna().any())
            partes.append(pl.DataFrame({
                "año": año, "fuerza": fuerza,
                "estado": df["entidad_federativa"].tolist(),
                "efectivos": pd.to_numeric(df[col], errors="coerce").fillna(0).tolist(),
            }))
    fap = pl.concat(partes)
    return con_cve(fap, "estado").drop_nulls("cve_ent")


def homicidios_estatales(año: int) -> pl.DataFrame:
    inc = pl.read_parquet(
        RAIZ / "data/incidencia_delictiva/incidencia_fuero_comun/incidencia_delictiva_fuero_comun.parquet")
    meses = ["Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio", "Julio",
             "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"]
    return (
        inc.filter((pl.col("Año") == año) &
                   (pl.col("Subtipo de delito") == "Homicidio doloso"))
        .group_by("Clave_Ent")
        .agg(pl.sum_horizontal([pl.col(m).sum() for m in meses]).alias("homicidios"))
        .rename({"Clave_Ent": "cve_ent"})
        .with_columns(pl.col("cve_ent").cast(pl.Int64))
    )


def main():
    # ---- A. GN censal: headcount y presupuesto
    hc = headcount_gn().sort("año")
    assert hc.filter(pl.col("año") == 2024)["efectivos"][0] == 132_003  # cross-check overview
    print("=== A. Guardia Nacional según censo CNSPF (ref-año) ===")
    for r in hc.iter_rows(named=True):
        print(f"  {r['año']}: {r['efectivos']:,} efectivos")
    cagr = (hc["efectivos"][-1] / hc["efectivos"][0]) ** (1 / (hc["año"][-1] - hc["año"][0])) - 1
    print(f"  CAGR 2019-2024: {cagr*100:+.2f}%/año")

    pres = presupuesto_gn().sort("año").with_columns(
        (pl.col("ejercido") / pl.col("aprobado")).alias("ejecucion"))
    print("\n  Presupuesto GN censal (miles de millones MXN):")
    for r in pres.iter_rows(named=True):
        print(f"  {r['año']}: aprobado {r['aprobado']/1e9:.1f} → ejercido {r['ejercido']/1e9:.1f} "
              f"({r['ejecucion']*100:.0f}% de ejecución)")
    print("  ↔ contraste Cap 3.B: los ramos 7/13/36 sobre-ejercen; la GN sub-ejerce"
          " — su costo real se carga a SEDENA (ramo 7).")

    # ---- F1: headcount + ejecución presupuestal
    fig = go.Figure()
    fig.add_trace(go.Bar(x=hc["año"], y=hc["efectivos"], name="efectivos GN (censo)",
                         marker_color="#1f77b4"))
    fig.add_trace(go.Scatter(x=pres["año"], y=pres["ejecucion"] * 100, name="ejecución presupuestal (%)",
                             yaxis="y2", mode="lines+markers", line=dict(color="#c0392b")))
    fig.update_layout(
        title=f"Guardia Nacional según su propio censo (CNSPF), 2019-2024<br>"
              f"<sup>efectivos +{cagr*100:.1f}%/año; la GN ejerce solo 36-49% de su presupuesto aprobado — "
              "el costo real opera vía SEDENA</sup>",
        xaxis_title="año de referencia", yaxis_title="efectivos",
        yaxis2=dict(title="ejecución (%)", overlaying="y", side="right", range=[0, 110]),
        legend=dict(orientation="h", y=1.02, x=0),
    )
    guardar_fig(fig, "cap3_censos_gn")

    # ---- B. despliegue territorial FAP
    fap = despliegue_fap()
    ult = fap.filter(pl.col("año") == 2026)  # ejército+marina 2026 (GN no tiene 2026)
    gn25 = fap.filter((pl.col("fuerza") == "gn") & (pl.col("año") == 2025))
    desp = (
        pl.concat([ult, gn25])
        .group_by("cve_ent")
        .agg(pl.sum("efectivos").alias("desplegados"))
    )
    pob = cargar_poblacion().filter(pl.col("año") == 2024).select("cve_ent", "pob_total")
    hom = homicidios_estatales(2024)
    b = (desp.join(pob, on="cve_ent").join(hom, on="cve_ent")
         .with_columns(
             (pl.col("desplegados") / pl.col("pob_total") * 1e5).alias("desp_100k"),
             (pl.col("homicidios") / pl.col("pob_total") * 1e5).alias("hom_100k"),
         ))
    assert b.height == 32

    c_pob = np.corrcoef(b["desplegados"], b["pob_total"])[0, 1]
    c_hom = np.corrcoef(np.log(b["desp_100k"]), np.log(b["hom_100k"]))[0, 1]
    print("\n=== B. Despliegue territorial FAP (ejército+marina 2026 + GN 2025) ===")
    print(f"  total desplegado: {b['desplegados'].sum():,.0f} efectivos")
    print(f"  corr(desplegados, población)                = {c_pob:+.3f}")
    print(f"  corr(log desplegados/100k, log homicidios/100k) = {c_hom:+.3f}")
    b = b.with_columns(pl.col("cve_ent").replace_strict(CORTO).alias("estado"))
    top = b.sort("desp_100k", descending=True).head(6)
    print("  top despliegue per cápita: " + ", ".join(
        f"{r['estado']}={r['desp_100k']:.0f}/100k" for r in top.iter_rows(named=True)))

    # ---- F2: despliegue pc vs homicidios pc
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=b["hom_100k"], y=b["desp_100k"], mode="markers+text", text=b["estado"],
        textposition="top center", textfont=dict(size=9),
        marker=dict(size=8, color="#1f77b4"), showlegend=False,
    ))
    fig.update_layout(
        title=f"Despliegue FAP per cápita vs homicidio doloso per cápita (2024-2026)<br>"
              f"<sup>corr log-log = {c_hom:+.2f} — el despliegue per cápita responde a la violencia "
              "(Colima/Sinaloa/Guerrero al tope); en niveles absolutos también escala con población "
              f"(corr = {c_pob:+.2f})</sup>",
        xaxis_title="homicidios dolosos por 100k hab (2024, SESNSP)",
        yaxis_title="efectivos FAP desplegados por 100k hab",
    )
    guardar_fig(fig, "cap3_censos_fap_despliegue")


if __name__ == "__main__":
    main()
