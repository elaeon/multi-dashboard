"""
Cap 10 — La red de cuota (CAPUFE/SCT): radialidad, renta concesionada y flujos.

Cuatro tests sobre data/tollbooths/growth_rate_car_2021_2025.csv (1,640 casetas):
  T1 Radialidad: ¿la red de cuota está físicamente centrada en la región Centro
     (herencia del diseño radial) — en casetas, km y recaudación?
  T2 Renta concesionada: ¿quién captura el peaje (CAPUFE público vs grupos privados),
     qué tan concentrado (HHI), cuánta red es "rescue" (rescate FARAC 1997), y quién
     sube tarifas sobre la inflación?
  T3 Flujo pagado × violencia: el test de FLUJO que faltaba al Cap 9 — ¿el tráfico
     de cuota per cápita estatal predice homicidios mejor que la pobreza?
  T4 ¿La red se descentra?: CAGR de tráfico 2021-2025 por región.

Guardas: tdpa/vta/toll < 0 son artefactos (filtrar); km dedupe por stretch_id;
TDPA/VTA tienen ~55% null (cobertura declarada; T3 es sugerente, no concluyente).

Figuras → centralismo/informe/figuras/cap10_*.png
Run: uv run python scripts/centralismo/cap10_red_cuota.py
"""

import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from comun import (CORTO, RAIZ, REGION_DE, cargar_pobreza, cargar_poblacion,
                   guardar_fig, normalizar_estado)

MESES = ["Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio", "Julio",
         "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"]
RADIAL = r"(^|_)mexico($|_)"  # 'mexico_queretaro' sí; 'mexicali' no


def cargar() -> pl.DataFrame:
    df = pl.read_csv(RAIZ / "data/tollbooths/growth_rate_car_2021_2025.csv",
                     infer_schema_length=0)
    num = ["stretch_length_km", "tdpa_round_2024", "vta_round_2024",
           "tdpa_cagr_growth_rate_2021_2025", "toll_inflation_diff"]
    df = df.with_columns([pl.col(c).cast(pl.Float64, strict=False) for c in num])
    df = df.with_columns([  # negativos = artefactos documentados
        pl.when(pl.col(c) < 0).then(None).otherwise(pl.col(c)).alias(c)
        for c in ["tdpa_round_2024", "vta_round_2024"]])
    mapa = {s: normalizar_estado(s) for s in df["state"].drop_nulls().unique().to_list()}
    assert all(v is not None for v in mapa.values()), mapa
    df = df.with_columns(
        pl.col("state").replace_strict(mapa, default=None).alias("cve_ent"))
    df = df.with_columns(
        pl.col("cve_ent").replace_strict(REGION_DE, default=None).alias("region"),
        (pl.col("road_name").fill_null("") + "_" + pl.col("stretch_name").fill_null("")
         + "_" + pl.col("stretch_way").fill_null("")).str.contains(RADIAL).alias("radial"))
    return df


def t1_radialidad(df: pl.DataFrame):
    print("=== T1. Radialidad de la red de cuota ===")
    con_edo = df.filter(pl.col("region").is_not_null())
    km = (con_edo.unique(subset="stretch_id")
          .group_by("region").agg(pl.sum("stretch_length_km").alias("km")))
    met = (con_edo.group_by("region")
           .agg(pl.len().alias("casetas"), pl.sum("vta_round_2024").alias("vta"))
           .join(km, on="region"))
    tot = met.select(pl.sum("casetas"), pl.sum("km"), pl.sum("vta")).row(0)
    print(f"  {'región':<12}{'casetas':>9}{'km':>9}{'recaudación 2024':>18}")
    for r in met.sort("casetas", descending=True).iter_rows(named=True):
        print(f"  {r['region']:<12}{r['casetas']/tot[0]*100:>8.1f}%{r['km']/tot[1]*100:>8.1f}%"
              f"{r['vta']/tot[2]*100:>17.1f}%")
    centro = met.filter(pl.col("region") == "Centro").row(0, named=True)
    pob = cargar_poblacion().filter(pl.col("año") == 2024)
    pob_centro = (pob.with_columns(pl.col("cve_ent").replace_strict(REGION_DE).alias("r"))
                  .filter(pl.col("r") == "Centro")["pob_total"].sum() / pob["pob_total"].sum())
    print(f"  región Centro: {centro['casetas']/tot[0]*100:.0f}% de casetas · "
          f"{centro['km']/tot[1]*100:.0f}% de km · {centro['vta']/tot[2]*100:.0f}% de recaudación "
          f"— con {pob_centro*100:.0f}% de la población")

    rad = df.group_by("radial").agg(
        pl.len().alias("casetas"), pl.sum("vta_round_2024").alias("vta"),
        pl.col("stretch_length_km").sum().alias("km"))
    r1 = rad.filter(pl.col("radial")).row(0, named=True)
    print(f"  corredores 'México-*' (radiales por nombre): {r1['casetas']/df.height*100:.0f}% "
          f"de casetas · {r1['vta']/df['vta_round_2024'].sum()*100:.0f}% de la recaudación")

    fig = go.Figure()
    orden = met.sort("casetas", descending=True)
    for col, nombre in [("casetas", "% casetas"), ("km", "% km"), ("vta", "% recaudación 2024")]:
        t = {"casetas": tot[0], "km": tot[1], "vta": tot[2]}[col]
        fig.add_trace(go.Bar(x=orden["region"], y=orden[col] / t * 100, name=nombre))
    fig.add_trace(go.Scatter(
        x=orden["region"],
        y=[(pob.with_columns(pl.col("cve_ent").replace_strict(REGION_DE).alias("r"))
            .filter(pl.col("r") == reg)["pob_total"].sum() / pob["pob_total"].sum() * 100)
           for reg in orden["region"]],
        name="% población", mode="markers", marker=dict(symbol="line-ew-open", size=24, color="black")))
    fig.update_layout(
        barmode="group",
        title="La red de cuota está centrada: distribución regional de casetas, km y recaudación<br><sup>marca negra = % de la población nacional de la región</sup>",
        yaxis_title="% del total nacional",
    )
    guardar_fig(fig, "cap10_radialidad")


def t2_operadores(df: pl.DataFrame):
    print("\n=== T2. La renta del movimiento: operadores y rescate ===")
    op = (df.filter(pl.col("parent_tb_manage").is_not_null())
          .group_by("parent_tb_manage")
          .agg(pl.len().alias("casetas"), pl.sum("vta_round_2024").alias("vta"),
               pl.mean("toll_inflation_diff").alias("dif_inflacion"))
          .sort("vta", descending=True))
    tot_vta = op["vta"].sum()
    op = op.with_columns((pl.col("vta") / tot_vta).alias("sh"))
    hhi = float((op["sh"] ** 2).sum())
    pub = op.filter(pl.col("parent_tb_manage") == "capufe")["sh"][0]
    print(f"  recaudación 2024 reportada: {tot_vta/1e9:.1f} mmdp · HHI por grupo = {hhi:.3f}")
    print(f"  CAPUFE (público) = {pub*100:.0f}% · privados = {(1-pub)*100:.0f}%")
    for r in op.head(6).iter_rows(named=True):
        d = r["dif_inflacion"]
        print(f"    {r['parent_tb_manage']:<12} {r['sh']*100:>5.1f}% de la recaudación · "
              f"tarifa vs inflación {'+' if d and d > 0 else ''}{d:.1f} pp" if d is not None else
              f"    {r['parent_tb_manage']:<12} {r['sh']*100:>5.1f}%")
    far = df.filter(pl.col("farac").is_not_null()).group_by("farac").agg(pl.len().alias("n"))
    print("  FARAC:", {r["farac"]: r["n"] for r in far.iter_rows(named=True)},
          "— 'rescue' = tramos del rescate carretero de 1997 (deuda pública, operación re-concesionada)")

    top = op.head(8).sort("sh")
    fig = go.Figure(go.Bar(
        x=top["sh"] * 100, y=top["parent_tb_manage"], orientation="h",
        marker_color=["#27ae60" if n == "capufe" else "#c0392b" for n in top["parent_tb_manage"]]))
    fig.update_layout(
        title=f"¿Quién cobra el peaje? Share de la recaudación de cuota 2024 (HHI={hhi:.2f})<br><sup>verde = público (CAPUFE) · rojo = grupos privados · 269 casetas provienen del rescate FARAC de 1997</sup>",
        xaxis_title="% de la recaudación reportada",
    )
    guardar_fig(fig, "cap10_operadores")


def t3_flujo_violencia(df: pl.DataFrame):
    print("\n=== T3. Flujo pagado × violencia (estatal) ===")
    flujo = (df.filter(pl.col("cve_ent").is_not_null())
             .group_by("cve_ent")
             .agg(pl.sum("tdpa_round_2024").alias("tdpa"),
                  (pl.col("tdpa_round_2024").is_not_null().mean() * 100).alias("cobertura")))
    pob = cargar_poblacion().filter(pl.col("año") == 2024)
    hom = (pl.scan_parquet(RAIZ / "data/incidencia_delictiva/incidencia_fuero_comun/"
                           "incidencia_delictiva_fuero_comun.parquet")
           .filter((pl.col("Año") == 2024) & (pl.col("Subtipo de delito") == "Homicidio doloso"))
           .with_columns(pl.sum_horizontal(MESES).alias("casos"))
           .group_by("Clave_Ent").agg(pl.sum("casos")).collect()
           .rename({"Clave_Ent": "cve_ent"}).with_columns(pl.col("cve_ent").cast(pl.Int64)))
    des = (pl.read_csv(RAIZ / "data/datamx/desaparecidos/desaparecidos.csv", infer_schema_length=0)
           .with_columns(pl.col("CVE_ENT").cast(pl.Int64, strict=False).alias("cve_ent"))
           .filter(pl.col("cve_ent").is_between(1, 32))
           .group_by("cve_ent").agg(pl.len().alias("des")))
    dep = cargar_pobreza().filter(pl.col("año") == 2022).select("cve_ent", "pct_pobreza")

    base = (flujo.join(pob, on="cve_ent").join(hom, on="cve_ent")
            .join(des, on="cve_ent").join(dep, on="cve_ent")
            .with_columns(
                (pl.col("tdpa") / pl.col("pob_total") * 1000).alias("flujo_pc"),  # cruces/día por mil hab
                (pl.col("casos") / pl.col("pob_total") * 1e5).alias("tasa_hom"),
                (pl.col("des") / pl.col("pob_total") * 1e5).alias("tasa_des"),
                pl.col("cve_ent").replace_strict(CORTO).alias("estado")))
    print(f"  estados con flujo TDPA reportado: {base.height}/32 · "
          f"cobertura mediana de casetas con dato: {base['cobertura'].median():.0f}%")

    for y, ny in [("tasa_hom", "homicidios 2024"), ("tasa_des", "desaparecidos acum.")]:
        c_f = np.corrcoef(base["flujo_pc"], base[y])[0, 1]
        c_p = np.corrcoef(base["pct_pobreza"], base[y])[0, 1]
        print(f"  corr(flujo pc, {ny}) = {c_f:+.2f}   vs corr(pobreza, {ny}) = {c_p:+.2f}")
    sin_centro = base.filter(pl.col("cve_ent").replace_strict(REGION_DE) != "Centro")
    c_sc = np.corrcoef(sin_centro["flujo_pc"], sin_centro["tasa_hom"])[0, 1]
    print(f"  corr(flujo pc, homicidios) excluyendo región Centro = {c_sc:+.2f} "
          f"(el tráfico de conmuters del Centro diluye la señal de corredor)")

    c_f = np.corrcoef(base["flujo_pc"], base["tasa_hom"])[0, 1]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=base["flujo_pc"], y=base["tasa_hom"], mode="markers+text", text=base["estado"],
        textposition="top center", textfont=dict(size=9),
        marker=dict(size=9, color="#1f77b4"), showlegend=False))
    fig.update_layout(
        title=f"Flujo carretero de cuota vs violencia, 2024 (corr={c_f:+.2f}; sin Centro {c_sc:+.2f})<br><sup>TDPA per cápita (cruces pagados/día por mil hab; cobertura ~45% de casetas, sesgo por operador) vs homicidio doloso</sup>",
        xaxis_title="cruces de cuota por día por mil habitantes",
        yaxis_title="homicidios dolosos por 100k, 2024",
    )
    guardar_fig(fig, "cap10_flujo_violencia")


def t4_descentramiento(df: pl.DataFrame):
    print("\n=== T4. ¿La red se descentra? (CAGR de tráfico 2021-2025) ===")
    g = (df.filter(pl.col("region").is_not_null()
                   & pl.col("tdpa_cagr_growth_rate_2021_2025").is_not_null()
                   & pl.col("tdpa_cagr_growth_rate_2021_2025").is_between(-50, 50))
         .group_by("region")
         .agg(pl.median("tdpa_cagr_growth_rate_2021_2025").alias("cagr"), pl.len().alias("n"))
         .sort("cagr", descending=True))
    for r in g.iter_rows(named=True):
        print(f"  {r['region']:<12} CAGR mediano tráfico = {r['cagr']:+.1f}% (n={r['n']})")

    fig = go.Figure(go.Bar(
        x=g["region"], y=g["cagr"],
        marker_color=["#c0392b" if reg == "Centro" else "#1f77b4" for reg in g["region"]],
        text=[f"n={n}" for n in g["n"]], textposition="outside"))
    fig.add_hline(y=0, line_color="black")
    fig.update_layout(
        title="Crecimiento del tráfico de cuota por región, 2021-2025 (CAGR mediano por caseta)<br><sup>casetas con dato de tráfico y CAGR en ±50% (excluye artefactos documentados)</sup>",
        yaxis_title="% anual",
    )
    guardar_fig(fig, "cap10_crecimiento_region")


def main():
    df = cargar()
    print(f"casetas: {df.height} · con estado: {df['cve_ent'].is_not_null().sum()} "
          f"· con TDPA 2024: {df['tdpa_round_2024'].is_not_null().sum()}\n")
    t1_radialidad(df)
    t2_operadores(df)
    t3_flujo_violencia(df)
    t4_descentramiento(df)


if __name__ == "__main__":
    main()
