"""
Cap 10 — Extensión: las casetas externas (aduanas del comercio exterior, BCMM).

La red de cuota interna es radial (Centro: 63% de casetas con 33% de población).
El BCMM por aduana (data/inegi/balanza_comercial/, ZIP mtra) da el otro lado:
por dónde cruza físicamente el comercio exterior — 24 aduanas nombradas con
estado, panel mensual estable 2012-2026, "Otras aduanas" acota 11-13%.

  A. Geografía del flujo externo (H2/Cap 10): shares por estado/región vs
     población, concentración (Nuevo Laredo), corredores frontera/puerto/interior
     y su crecimiento 2012→2025 (nearshoring).
  B. Renta física continua (H7/anexo equilibrio): el flujo aduanal (X+M) per
     cápita vuelve continua la dummy frontera∪puerto∪ductos de la tipología —
     corr con homicidio/extorsión 2023-2025 (mismas fuentes del Cap 12).
  C. Transnacional, petróleo y era-arancel: declive de la exportación petrolera
     (ZIP mensual), par Dos Bocas/Cd. del Carmen (Cap 8.C), y el quiebre
     2025-2026 (aranceles EEUU): flujo mensual, share USA y ETEF 2026-T1.

Trampas BCMM-mtra: VAL_USD en MILLONES USD; TIPO singular ("Exportación");
las filas por aduana vienen SOLO desglosadas por modo (sumar MTRA = total de
la aduana; MTRA null solo existe en las filas de total nacional); "Colombia,
N.L." es la aduana del puente Colombia en Nuevo León, no el país.

Figuras → centralismo/informe/figuras/cap10_aduanas_*.png
Run: uv run python scripts/centralismo/cap10_aduanas_comercio.py
"""

import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import polars as pl
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from comun import CORTO, RAIZ, REGION_DE, cargar_poblacion, guardar_fig
from cap12_paradoja_capital import EXTORSION, HOMICIDIO, tasas_estatales
from cap9_rutas_violencia import FRONTERA, PUERTOS

DIR_BCMM = RAIZ / "data/inegi/balanza_comercial"
CDMX, COLIMA, YUCATAN, MICHOACAN = 9, 6, 31, 16

# aduana → (cve_ent, corredor); "Colombia, N.L." es de Nuevo León (puente Colombia)
ADUANAS = {
    "Nuevo Laredo, Tamps.": (28, "frontera"),
    "Cd. Juárez, Chih.": (8, "frontera"),
    "Tijuana, B.C.": (2, "frontera"),
    "Colombia, N.L.": (19, "frontera"),
    "Cd. Reynosa, Tamps.": (28, "frontera"),
    "Piedras Negras, Coah.": (5, "frontera"),
    "Nogales, Son.": (26, "frontera"),
    "Mexicali, B.C.": (2, "frontera"),
    "Matamoros,Tamps.": (28, "frontera"),
    "Veracruz, Ver.": (30, "puerto"),
    "Altamira, Tamps.": (28, "puerto"),
    "Manzanillo, Col.": (6, "puerto"),
    "Lazaro Cardenas, Mich.": (16, "puerto"),
    "Coatzacoalcos, Ver.": (30, "puerto"),
    "Tuxpan, Ver.": (30, "puerto"),
    "Dos Bocas, Tab.": (27, "puerto"),
    "Cd. del Carmen, Camp.": (4, "puerto"),
    "Progreso, Yuc.": (31, "puerto"),
    "Aeropuerto Internacional de la Cd. de México, D.F.": (9, "interior"),
    "Guadalajara, Jal.": (14, "interior"),
    "Monterrey, N.L.": (19, "interior"),
    "Toluca, Mex.": (15, "interior"),
    "Chihuahua, Chih.": (8, "interior"),
    "Puebla, Pue.": (21, "interior"),
}
TOTALES = {"Exportación total", "Importación total"}
AGREGADOS = TOTALES | {"Aéreo", "Marítimo", "Ferroviario", "Carretero",
                       "Otros modos", "Otras aduanas"}


def leer_zip(nombre_zip: str, nombre_csv: str) -> pl.DataFrame:
    with zipfile.ZipFile(DIR_BCMM / nombre_zip) as z:
        return pl.read_csv(io.BytesIO(z.read(nombre_csv)))


def cargar_aduanas() -> pl.DataFrame:
    """mtra-aduana → anio, mes, tipo (Exportación/Importación), aduana, cve_ent,
    corredor, musd. Solo aduanas nombradas (filas modo sumadas por aduana-mes)."""
    df = leer_zip("conjunto_de_datos_bcmm_mensual_mtra_csv.zip",
                  "conjunto_de_datos/bcmm_mtra_aduana_mensual_tr_cifra_2012_2026.csv")
    assert set(df["CONCEPTO"].unique()) == set(ADUANAS) | AGREGADOS
    adu = (df.filter(pl.col("CONCEPTO").is_in(list(ADUANAS)))
           .group_by("ANIO", "MES", "TIPO", "CONCEPTO")
           .agg(pl.sum("VAL_USD").alias("musd"))
           .rename({"ANIO": "año", "MES": "mes", "TIPO": "tipo", "CONCEPTO": "aduana"})
           .with_columns(
               pl.col("aduana").replace_strict({a: c for a, (c, _) in ADUANAS.items()})
               .alias("cve_ent"),
               pl.col("aduana").replace_strict({a: t for a, (_, t) in ADUANAS.items()})
               .alias("corredor")))
    # reconciliación: nombradas + Otras aduanas == total publicado (2024)
    tot = df.filter(pl.col("CONCEPTO").is_in(TOTALES) & (pl.col("ANIO") == 2024))
    otras = (df.filter((pl.col("CONCEPTO") == "Otras aduanas") & (pl.col("ANIO") == 2024))
             .group_by("TIPO").agg(pl.sum("VAL_USD")))
    for tipo, esperado in [("Exportación", 617_677.1), ("Importación", 636_218.0)]:
        suma = (adu.filter((pl.col("año") == 2024) & (pl.col("tipo") == tipo))["musd"].sum()
                + otras.filter(pl.col("TIPO") == tipo)["VAL_USD"][0])
        publicado = tot.filter(pl.col("TIPO") == tipo)["VAL_USD"].sum()
        assert abs(suma / publicado - 1) < 0.001 and abs(publicado / esperado - 1) < 0.001
    assert adu.group_by("año").agg(pl.col("aduana").n_unique())["aduana"].eq(24).all()
    return adu


def main():
    pob = cargar_poblacion()
    adu = cargar_aduanas()
    a24 = (adu.filter(pl.col("año") == 2024)
           .group_by("aduana", "cve_ent", "corredor", "tipo")
           .agg(pl.sum("musd")))

    print("=== A. Geografía del flujo externo (H2/Cap 10) ===")
    x24 = a24.filter(pl.col("tipo") == "Exportación")
    tot_x = 617_677.1  # total publicado, incluye Otras aduanas
    nl = x24.filter(pl.col("aduana") == "Nuevo Laredo, Tamps.")["musd"].sum()
    print(f"  Nuevo Laredo 2024: {nl / tot_x * 100:.1f}% de toda la exportación · "
          f"HHI de aduanas nombradas = "
          f"{((x24['musd'] / x24['musd'].sum()) ** 2).sum() * 1e4:,.0f}")
    reg = (a24.with_columns(pl.col("cve_ent").replace_strict(REGION_DE).alias("region"))
           .group_by("region").agg(pl.sum("musd").alias("flujo")))
    pr = (pob.filter(pl.col("año") == 2024)
          .with_columns(pl.col("cve_ent").replace_strict(REGION_DE).alias("region"))
          .group_by("region").agg(pl.sum("pob_total")))
    pr = pr.with_columns((pl.col("pob_total") / pr["pob_total"].sum()).alias("s_pob"))
    reg = reg.with_columns((pl.col("flujo") / reg["flujo"].sum()).alias("s"))
    print("  región (share del flujo X+M en aduanas nombradas 2024 · población):")
    for r in reg.join(pr, on="region").sort("s", descending=True).iter_rows(named=True):
        print(f"    {r['region']:<12} {r['s'] * 100:>5.1f}% · pob {r['s_pob'] * 100:.1f}%")
    centro = reg.filter(pl.col("region") == "Centro")["s"][0]
    print(f"  contraste con la red interna: Centro = {centro * 100:.1f}% del flujo "
          f"aduanal vs 63% de las casetas de cuota — las casetas externas son "
          f"periféricas por construcción, y la renta que cruza también")

    corr_g = (adu.filter(pl.col("año").is_in([2012, 2025]))
              .group_by("año", "corredor").agg(pl.sum("musd"))
              .pivot(on="año", index="corredor", values="musd")
              .with_columns(((pl.col("2025") / pl.col("2012")) ** (1 / 13) - 1)
                            .alias("cagr") * 100))
    print("  corredores 2012→2025 (X+M, CAGR nominal USD):")
    for r in corr_g.sort("cagr", descending=True).iter_rows(named=True):
        print(f"    {r['corredor']:<10} {r['2012']:>9,.0f} → {r['2025']:>9,.0f} M "
              f"({r['cagr']:+.1f}%/año)")

    print("\n=== B. Renta física continua (H7/anexo equilibrio) ===")
    flujo = (adu.filter(pl.col("año").is_between(2023, 2025))
             .group_by("cve_ent").agg((pl.sum("musd") / 3).alias("flujo")))
    p24 = pob.filter(pl.col("año") == 2024).select("cve_ent", "pob_total")
    b = (pl.DataFrame({"cve_ent": list(range(1, 33))})
         .join(flujo, on="cve_ent", how="left")
         .with_columns(pl.col("flujo").fill_null(0.0))
         .join(p24, on="cve_ent")
         .with_columns((pl.col("flujo") * 1e6 / pl.col("pob_total")).alias("flujo_pc"),
                       pl.col("cve_ent").replace_strict(CORTO).alias("estado")))
    hom = tasas_estatales(HOMICIDIO, [2023, 2024, 2025], pob).rename({"tasa": "hom"})
    ext = (tasas_estatales(EXTORSION, [2023, 2024, 2025], pob)
           .select("cve_ent", pl.col("tasa").alias("ext")))
    b = b.join(hom.select("cve_ent", "hom"), on="cve_ent").join(ext, on="cve_ent")
    assert b.height == 32
    n_con = b.filter(pl.col("flujo") > 0).height
    lf = np.log1p(b["flujo_pc"].to_numpy())
    print(f"  {n_con}/32 estados con aduana nombrada (resto flujo=0 — eso ES la variable)")
    for var in ("hom", "ext"):
        rp, pp_ = pearsonr(lf, b[var].to_numpy())
        rs, ps = spearmanr(b["flujo_pc"].to_numpy(), b[var].to_numpy())
        print(f"  corr(log1p flujo pc, {var}): Pearson {rp:+.2f} (p={pp_:.2f}) · "
              f"Spearman {rs:+.2f} (p={ps:.2f})")
    dummy = np.array([c in ({c for c, _ in FRONTERA} | {c for c, _ in PUERTOS})
                      for c in b["cve_ent"]])
    rd, pd_ = pearsonr(dummy.astype(float), b["hom"].to_numpy())
    print(f"  la dummy frontera∪puerto del anexo: corr con homicidio {rd:+.2f} "
          f"(p={pd_:.2f}) — comparar poder discriminante")
    for cve, aduana in [(COLIMA, "Manzanillo"), (YUCATAN, "Progreso"),
                        (MICHOACAN, "Lázaro Cárdenas")]:
        r = b.filter(pl.col("cve_ent") == cve).row(0, named=True)
        print(f"    {r['estado']:<10} flujo {r['flujo']:>7,.0f} M USD/año "
              f"({r['flujo_pc']:>6,.0f} pc) · hom {r['hom']:5.1f} · ext {r['ext']:4.1f}"
              f"  [{aduana}]")
    print("  guarda: n=32 correlacional; el flujo mide la renta que cruza, no quién "
          "la disputa — la tipología (W × murallas) sigue siendo la interacción")

    print("\n=== C. Transnacional, petróleo y era-arancel ===")
    dfs = []
    with zipfile.ZipFile(DIR_BCMM / "conjunto_de_datos_bcmm_mensual_csv.zip") as z:
        for n in z.namelist():
            if n.startswith("conjunto_de_datos/") and n.endswith(".csv"):
                dfs.append(pl.read_csv(io.BytesIO(z.read(n))))
    men = pl.concat(dfs)
    xt = (men.filter(pl.col("CONCEPTO") == "Exportaciones Totales")
          .group_by("ANIO").agg(pl.sum("VAL_USD").alias("x"), pl.len().alias("m")))
    pet = (men.filter(pl.col("CONCEPTO") == "Petroleras")
           .group_by("ANIO").agg(pl.sum("VAL_USD").alias("p")))
    sp = (xt.join(pet, on="ANIO").with_columns((pl.col("p") / pl.col("x") * 100)
                                               .alias("s_pet")).sort("ANIO"))
    s12 = sp.filter(pl.col("ANIO") == 2012).row(0, named=True)
    s25 = sp.filter(pl.col("ANIO") == 2025).row(0, named=True)
    print(f"  exportación petrolera: {s12['s_pet']:.1f}% (2012) → {s25['s_pet']:.1f}% "
          f"(2025) del total — la renta petrolera exportada se apagó")
    par = (adu.filter(pl.col("aduana").is_in(["Dos Bocas, Tab.", "Cd. del Carmen, Camp."])
                      & (pl.col("tipo") == "Exportación"))
           .group_by("año", "aduana").agg(pl.sum("musd")).sort("año"))
    for a in ["Dos Bocas, Tab.", "Cd. del Carmen, Camp."]:
        d = par.filter(pl.col("aduana") == a)
        print(f"  {a}: {d.filter(pl.col('año') == 2014)['musd'][0]:,.0f} (2014) → "
              f"{d.filter(pl.col('año') == 2025)['musd'][0]:,.0f} M USD (2025)")

    # era-arancel: YoY mensual ene-may 2026 vs 2025 y share USA mensual
    may = (men.filter((pl.col("CONCEPTO") == "Exportaciones Totales")
                      & pl.col("MES").is_between(1, 5)
                      & pl.col("ANIO").is_in([2025, 2026]))
           .group_by("ANIO").agg(pl.sum("VAL_USD")).sort("ANIO"))
    yoy = (may["VAL_USD"][1] / may["VAL_USD"][0] - 1) * 100
    print(f"  era-arancel: exportación ene-may 2026 vs 2025 = {yoy:+.1f}% "
          f"({may['VAL_USD'][0]:,.0f} → {may['VAL_USD'][1]:,.0f} M USD) — sin quiebre "
          f"a la baja; el agregado nominal acelera (cifras revisadas, sin deflactar)")
    dfs = []
    with zipfile.ZipFile(DIR_BCMM
                         / "conjunto_de_datos_bcmm_mensual_paises_bien_csv.zip") as z:
        for n in z.namelist():
            if n.startswith("conjunto_de_datos/") and n.endswith(".csv"):
                dfs.append(pl.read_csv(io.BytesIO(z.read(n))))
    pb = (pl.concat(dfs)
          .filter((pl.col("TIPO") == "Exportaciones") & pl.col("PAIS_O_D").is_not_null()))
    usa = (pb.group_by("ANIO", "MES")
           .agg((pl.col("VAL_USD").filter(pl.col("PAIS_O_D") == "USA").sum()
                 / pl.col("VAL_USD").sum() * 100).alias("s_usa")).sort("ANIO", "MES"))
    u = {a: usa.filter((pl.col("ANIO") == a) & pl.col("MES").is_between(1, 3))["s_usa"]
         .mean() for a in (2024, 2025, 2026)}
    print(f"  share USA (T1): {u[2024]:.1f}% (2024) → {u[2025]:.1f}% (2025) → "
          f"{u[2026]:.1f}% (2026) — el arancel no ha roto la dependencia del destino")
    with zipfile.ZipFile(RAIZ / "data/inegi/etef/conjunto_de_datos_eef_trimestral_csv.zip") as z:
        tri = pl.read_csv(io.BytesIO(z.read(
            "conjunto_de_datos/eef_trimestral_tr_cifra_2007_2026.csv")),
            infer_schema_length=5000)
    t1 = (tri.filter((pl.col("TRIMESTRE") == "I") & pl.col("ANIO").is_in([2025, 2026]))
          .group_by("ANIO", "CVE_ENT").agg(pl.sum("VAL_USD"))
          .pivot(on="ANIO", index="CVE_ENT", values="VAL_USD")
          .with_columns(((pl.col("2026") / pl.col("2025") - 1) * 100).alias("d"))
          .with_columns(pl.col("CVE_ENT").replace_strict(CORTO).alias("estado")))
    print("  ETEF 2026-T1 vs 2025-T1, extremos estatales:",
          " · ".join(f"{r['estado']}={r['d']:+.0f}%" for r in
                     t1.sort("d").head(3).iter_rows(named=True)), "| top:",
          " · ".join(f"{r['estado']}={r['d']:+.0f}%" for r in
                     t1.sort("d", descending=True).head(3).iter_rows(named=True)))

    # figuras -----------------------------------------------------------------
    fig = go.Figure()
    orden = reg.join(pr, on="region").sort("s", descending=True)
    fig.add_trace(go.Bar(x=orden["region"], y=orden["s"] * 100,
                         name="flujo aduanal (X+M 2024)"))
    fig.add_trace(go.Bar(x=orden["region"], y=orden["s_pob"] * 100, name="población"))
    fig.add_trace(go.Bar(x=["Centro"], y=[63], name="casetas de cuota (Cap 10)",
                         marker_color="#c0392b", opacity=0.6))
    fig.update_layout(
        barmode="group",
        title="Las casetas externas son periféricas; las internas, radiales<br>"
              "<sup>share regional del flujo por aduanas nombradas (BCMM 2024) vs "
              "población · la barra roja: 63% de las casetas de cuota están en el "
              "Centro</sup>",
        yaxis_title="% del total nacional")
    guardar_fig(fig, "cap10_aduanas_regiones")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=np.log1p(b["flujo_pc"].to_numpy()), y=b["hom"], mode="markers+text",
        text=b["estado"], textposition="top center", textfont=dict(size=9),
        showlegend=False,
        marker=dict(size=10, color=b["ext"], colorscale="OrRd",
                    colorbar=dict(title="extorsión<br>/100k"))))
    fig.update_layout(
        title="La renta que cruza, medida: flujo aduanal per cápita vs violencia<br>"
              "<sup>X+M promedio 2023-2025 por estado de la aduana (USD/hab, log1p) vs "
              "homicidio doloso · color = extorsión — la renta física ahora es "
              "continua, no dummy</sup>",
        xaxis_title="log(1 + flujo aduanal por habitante, USD)",
        yaxis_title="homicidio doloso /100k (2023-2025)")
    guardar_fig(fig, "cap10_aduanas_flujo_violencia")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sp["ANIO"], y=sp["s_pet"], name="share petrolero (%)"))
    usa_a = (usa.group_by("ANIO").agg(pl.mean("s_usa")).sort("ANIO")
             .filter(pl.col("ANIO") < 2026))
    fig.add_trace(go.Scatter(x=usa_a["ANIO"], y=usa_a["s_usa"],
                             name="share destino USA (%)", yaxis="y2"))
    fig.update_layout(
        title="Doble dependencia en movimiento: el petróleo se apaga, EEUU se "
              "concentra<br><sup>% petrolero de la exportación (izq) y % con destino "
              "EEUU (der, desde 2015) · BCMM</sup>",
        yaxis=dict(title="% petrolero de la exportación"),
        yaxis2=dict(title="% destino EEUU", overlaying="y", side="right",
                    range=[75, 90]))
    guardar_fig(fig, "cap10_aduanas_transnacional")


if __name__ == "__main__":
    main()
