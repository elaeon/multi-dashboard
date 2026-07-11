"""
Anexo equilibrio_de_fuerzas — tipología de equilibrios K/L/T por estado.

Operacionaliza el modelo de Koyama (2026, Public Choice, "Adam Smith and the role of
the towns in feudal Europe") sobre los 32 estados, con las variables del corpus:

  - W  (riqueza de la "ciudad")      = PIBE per cápita sin petróleo, 2024
  - murallas (protección local)      = policías estatales por mil hab (CNSPE, dato 2024)
  - carta (autonomía fiscal)         = 1 − dependencia de transferencias (2013–2024)
  - renta del señor (peajeable)      = frontera ∪ puerto ∪ ducto (catálogos Cap 9/9.B)
  - disputa                          = tasa de homicidio doloso (2023–2025) y extorsión

Cuadrantes W × protección (corte en mediana) ≈ configuraciones del modelo:
  I   W≥med, pol≥med → renta protegida (el rincón KT/absolutismo local)
  II  W≥med, pol<med → ciudad rica sin murallas (peaje señorial)
  III W<med, pol<med → sin renta ni Estado
  IV  W<med, pol≥med → protegida sin renta

Sin descargas: todo sale de datasets ya catalogados en data/DATA_HAWK_INDEX.md.
Run: uv run python scripts/centralismo/eq_tipologia.py
"""

import sys
from pathlib import Path

import plotly.graph_objects as go
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from comun import (CORTO, PIBE_MINERIA_PETROLERA, PIBE_TOTAL, cargar_poblacion,
                   guardar_fig, leer_pibe)
from cap12_capacidad_policial import fuerza_estatal
from cap12_paradoja_capital import (EXTORSION, HOMICIDIO, casos_estatales,
                                    tasas_estatales)
from cap5_factor_competencia import dependencia_fiscal
from cap9_rutas_violencia import FRONTERA, PUERTOS
from cap9_tomas_clandestinas import cargar_tomas

CDMX = 9
CUADRANTES = {(True, True): "I renta protegida", (True, False): "II rica sin murallas",
              (False, False): "III sin renta ni Estado", (False, True): "IV protegida sin renta"}


def construir(pob: pl.DataFrame) -> pl.DataFrame:
    """Base estado × {W pc, policías pc, autonomía, renta física, homicidio, extorsión}."""
    print("\n=== Tipología: construcción de la base (n=32) ===")
    # W: PIBE pc 2024 sin petróleo (misma definición del Cap 5/7)
    pib = leer_pibe(PIBE_TOTAL).rename({"valor": "pib"})
    petro = leer_pibe(PIBE_MINERIA_PETROLERA).rename({"valor": "petro"})
    w = (pib.join(petro, on=["cve_ent", "año"], how="left")
         .filter(pl.col("año") == 2024)
         .join(pob.filter(pl.col("año") == 2024), on=["cve_ent", "año"])
         .with_columns(((pl.col("pib") - pl.col("petro").fill_null(0)) * 1e6
                        / pl.col("pob_total")).alias("w_pc"))
         .select("cve_ent", "w_pc"))

    # murallas: personal estatal de seguridad /1000, dato 2024 (imprime la sección A
    # del cap12 con sus anclas — se reutiliza sin modificar)
    pol = (fuerza_estatal(pob).filter(pl.col("año") == 2024)
           .select("cve_ent", pl.col("pc").alias("pol_pc")))

    # carta: autonomía fiscal = 1 − dependencia (Cap 5, 2013–2024)
    auton = dependencia_fiscal().with_columns(
        (1 - pl.col("dependencia")).alias("autonomia")).select("cve_ent", "autonomia")

    # renta física peajeable: misma definición del test D del cap12
    tomas = cargar_tomas()
    top_tomas = set(tomas.groupby("cve_ent").size().nlargest(8).index)
    fisica = {c for c, _ in FRONTERA} | {c for c, _ in PUERTOS} | top_tomas
    print(f"  renta física = frontera ∪ puerto ∪ top-8 ductos: {len(fisica)} estados")

    hom = tasas_estatales(HOMICIDIO, [2023, 2024, 2025], pob).rename({"tasa": "hom"})
    ext = (tasas_estatales(EXTORSION, [2023, 2024, 2025], pob)
           .select("cve_ent", pl.col("tasa").alias("ext")))

    b = (w.join(pol, on="cve_ent").join(auton, on="cve_ent")
         .join(hom, on="cve_ent").join(ext, on="cve_ent")
         .with_columns(pl.col("cve_ent").is_in(list(fisica)).alias("fisica")))
    assert b.height == 32, f"base con {b.height} estados"

    # anclas duras
    nac = (casos_estatales(HOMICIDIO, [2023, 2024, 2025])
           .join(pob, on=["cve_ent", "año"]))
    tasa_nac = nac["casos"].sum() / nac["pob_total"].sum() * 1e5
    assert 14 < tasa_nac < 26, tasa_nac
    assert b.sort("pol_pc", descending=True)["cve_ent"][0] == CDMX
    assert b.sort("w_pc", descending=True)["cve_ent"][0] == CDMX
    print(f"  anclas: homicidio nacional 2023-25 = {tasa_nac:.1f}/100k · CDMX lugar 1 "
          f"en policías pc y en PIBE pc sin petróleo ✓")
    return b


def tipologia(b: pl.DataFrame) -> pl.DataFrame:
    print("\n=== Cuadrantes W × protección (cortes en mediana) ===")
    med_w, med_p = b["w_pc"].median(), b["pol_pc"].median()
    print(f"  medianas: W pc = {med_w:,.0f} MXN-2018 · policías = {med_p:.2f}/1000 "
          f"(estándar MOFP: 1.8/1000 — solo CDMX lo cumple) · "
          f"autonomía fiscal = {b['autonomia'].median() * 100:.1f}%")
    b = b.with_columns(
        pl.struct(["w_pc", "pol_pc"]).map_elements(
            lambda s: CUADRANTES[(s["w_pc"] >= med_w, s["pol_pc"] >= med_p)],
            return_dtype=pl.String).alias("cuadrante"))

    print(f"\n  {'estado':<20} {'W pc':>9} {'pol':>5} {'auton':>6} {'fis':>4} "
          f"{'hom':>6} {'ext':>5}  cuadrante")
    for r in b.sort("cuadrante", "hom", descending=[False, True]).iter_rows(named=True):
        print(f"  {r['estado']:<20} {r['w_pc']:>9,.0f} {r['pol_pc']:>5.2f} "
              f"{r['autonomia'] * 100:>5.1f}% {'sí' if r['fisica'] else 'no':>4} "
              f"{r['hom']:>6.1f} {r['ext']:>5.1f}  {r['cuadrante']}")

    print("\n  resumen por cuadrante (homicidio y extorsión = tasas medias /100k):")
    for cu in sorted(CUADRANTES.values()):
        d = b.filter(pl.col("cuadrante") == cu)
        fis = d.filter(pl.col("fisica")).height
        print(f"    {cu:<24} n={d.height:2d} · hom={d['hom'].mean():5.1f} · "
              f"ext={d['ext'].mean():4.1f} · renta física {fis}/{d.height} · "
              f"autonomía media={d['autonomia'].mean() * 100:.1f}%")

    # el corte fino: dentro de los estados CON renta física, ¿protege la muralla?
    print("\n  solo estados con renta física peajeable:")
    for alta, nombre in [(False, "protección baja"), (True, "protección alta")]:
        d = b.filter(pl.col("fisica") & ((pl.col("pol_pc") >= med_p) == alta))
        print(f"    {nombre}: hom media={d['hom'].mean():5.1f}/100k (n={d.height}) "
              f"{d.sort('hom', descending=True)['estado'].head(4).to_list()}")

    print("\n  correlaciones (n=32, descriptivo):")
    for a, bb, et in [("pol_pc", "hom", "murallas × homicidio"),
                      ("w_pc", "hom", "W × homicidio"),
                      ("autonomia", "hom", "carta (autonomía) × homicidio"),
                      ("autonomia", "pol_pc", "carta × murallas"),
                      ("autonomia", "w_pc", "carta × W")]:
        c = pl.DataFrame({"a": b[a], "b": b[bb]}).select(pl.corr("a", "b"))[0, 0]
        print(f"    {et:<32} r = {c:+.2f}")
    print("  guarda de endogeneidad: correlaciones crudas confundidas (los estados "
          "violentos contratan más policía y reciben GN); el test limpio del mecanismo "
          "es la discontinuidad ZMVM (Cap 12 C-2)")
    return b


def figura(b: pl.DataFrame):
    med_w, med_p = b["w_pc"].median(), b["pol_pc"].median()
    fig = go.Figure()
    for fis, simbolo, nombre in [(True, "diamond", "con renta física peajeable"),
                                 (False, "circle", "sin renta física")]:
        d = b.filter(pl.col("fisica") == fis)
        fig.add_trace(go.Scatter(
            x=d["w_pc"], y=d["pol_pc"], mode="markers+text", text=d["estado"],
            textposition="top center", textfont=dict(size=8), name=nombre,
            marker=dict(size=11, symbol=simbolo, color=d["hom"],
                        colorscale="Reds", cmin=0, cmax=50,
                        line=dict(width=1, color="#7f8c8d"),
                        colorbar=dict(title="homicidio<br>/100k", x=1.12),
                        showscale=fis)))
    fig.add_vline(x=med_w, line_dash="dot", line_color="#95a5a6")
    fig.add_hline(y=med_p, line_dash="dot", line_color="#95a5a6")
    for x, y, texto in [(0.99, 0.88, "I. renta protegida"),
                        (0.99, 0.02, "II. rica sin murallas"),
                        (0.01, 0.02, "III. sin renta ni Estado"),
                        (0.01, 0.99, "IV. protegida sin renta")]:
        fig.add_annotation(x=x, y=y, xref="x domain", yref="y domain",
                           xanchor="right" if x > 0.5 else "left", showarrow=False,
                           text=texto, font=dict(size=11, color="#7f8c8d"))
    fig.update_layout(
        title="La tipología del equilibrio: riqueza (W) × murallas (protección local)<br>"
              "<sup>x = PIBE pc sin petróleo 2024 (log) · y = policías estatales /1000 "
              "hab, dato 2024 (log; CDMX incluye auxiliar/bancaria) · color = homicidio "
              "2023-25 (escala truncada en 50; Colima=92) · rombo = frontera/puerto/"
              "ducto · líneas = medianas</sup>",
        xaxis_title="PIBE per cápita sin petróleo, 2024 (MXN de 2018)",
        yaxis_title="policías estatales por 1,000 hab (2024)",
        xaxis_type="log", yaxis_type="log",
        legend=dict(orientation="h", y=-0.15))
    guardar_fig(fig, "eq_tipologia", ancho=1000, alto=700)


def main():
    pob = cargar_poblacion()
    b = construir(pob)
    b = tipologia(b)
    figura(b)


if __name__ == "__main__":
    main()
