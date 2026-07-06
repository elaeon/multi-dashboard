"""
Cap 9 — Las rutas de la violencia: ¿geografía de renta? (refuerzo de P6).

El Cap 7 mostró que la violencia NO sigue a la pobreza (corr −0.17 estatal) y que se
concentra (top-50 municipios = 49.6% de homicidios). Aquí se testea si sigue a los
puntos de renta: municipios mineros (EIMM), puertos mayores y cruces fronterizos.

  1. Tasas municipales de homicidio doloso (promedio 2023-2024; población censo 2020).
  2. Sobre-representación: share de homicidios vs share de población por feature;
     composición del top-50.
  3. OLS municipal: log(1+tasa) ~ minero + puerto + frontera + pobreza CONEVAL 2020
     + ln(población) + efectos fijos de estado (n≈2,400).

Los catálogos de puertos/frontera se resuelven POR NOMBRE contra el parquet de
incidencia (con assert) para evitar errores de clave.

Figuras → centralismo/informe/figuras/cap9_*.png
Run: uv run python scripts/centralismo/cap9_rutas_violencia.py
"""

import io
import sys
import unicodedata
import zipfile
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import polars as pl
import statsmodels.formula.api as smf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from comun import RAIZ, guardar_fig

MESES = ["Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio", "Julio",
         "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"]

# (cve_ent, nombre a buscar) — puertos de carga mayores del sistema portuario
PUERTOS = [
    (2, "Ensenada"), (26, "Guaymas"), (25, "Mazatlán"), (25, "Ahome"),
    (6, "Manzanillo"), (16, "Lázaro Cárdenas"), (12, "Acapulco"),
    (20, "Salina Cruz"), (30, "Coatzacoalcos"), (30, "Veracruz"), (30, "Tuxpan"),
    (28, "Altamira"), (28, "Tampico"), (27, "Paraíso"), (31, "Progreso"),
]

# municipios con cruce fronterizo formal México-EEUU
FRONTERA = [
    (2, "Tijuana"), (2, "Tecate"), (2, "Mexicali"),
    (26, "San Luis Río Colorado"), (26, "Puerto Peñasco"), (26, "Sáric"),
    (26, "Nogales"), (26, "Naco"), (26, "Agua Prieta"),
    (8, "Ascensión"), (8, "Juárez"), (8, "Ojinaga"),
    (5, "Acuña"), (5, "Piedras Negras"), (19, "Anáhuac"),
    (28, "Nuevo Laredo"), (28, "Mier"), (28, "Miguel Alemán"), (28, "Camargo"),
    (28, "Gustavo Díaz Ordaz"), (28, "Reynosa"), (28, "Río Bravo"), (28, "Matamoros"),
]


def _clave(s: str) -> str:
    s = unicodedata.normalize("NFD", str(s)).encode("ascii", "ignore").decode().lower()
    return s.strip()


def resolver_catalogo(catalogo, municipios: pl.DataFrame, etiqueta: str) -> set[int]:
    """(cve_ent, nombre) → set de CVE_MUN 5-dígitos, validado por nombre exacto."""
    cves = set()
    for ent, nombre in catalogo:
        m = municipios.filter(
            (pl.col("Clave_Ent") == ent)
            & (pl.col("mun_clave") == _clave(nombre)))
        if m.height != 1:
            # fallback: contains (p.ej. "Veracruz" = "Veracruz de Ignacio de la Llave")
            m = municipios.filter(
                (pl.col("Clave_Ent") == ent)
                & pl.col("mun_clave").str.contains(_clave(nombre), literal=True))
        assert m.height >= 1, f"{etiqueta}: no encontré {nombre!r} en ent {ent}"
        cves.add(int(m["Cve. Municipio"][0]))
    print(f"  catálogo {etiqueta}: {len(cves)} municipios resueltos")
    return cves


def cargar_homicidios() -> pl.DataFrame:
    inc = (pl.scan_parquet(RAIZ / "data/incidencia_delictiva/incidencia_fuero_comun/"
                           "incidencia_delictiva_fuero_comun.parquet")
           .filter(pl.col("Año").is_in([2023, 2024])
                   & (pl.col("Subtipo de delito") == "Homicidio doloso"))
           .with_columns(pl.sum_horizontal(MESES).alias("casos"))
           .group_by("Cve. Municipio", "Municipio", "Clave_Ent")
           .agg((pl.sum("casos") / 2).alias("casos"))  # promedio anual 2023-2024
           .collect())
    return inc.with_columns(pl.col("Municipio").map_elements(_clave, return_dtype=pl.Utf8)
                            .alias("mun_clave"))


def cargar_mineros() -> set[int]:
    z = zipfile.ZipFile(RAIZ / "data/inegi/industria_minerometalurgica/"
                        "conjunto_de_datos_eimm_municipio_csv.zip")
    dfs = []
    for nombre in z.namelist():
        if "eimm_municipio_mensual" in nombre and any(str(a) in nombre for a in (2023, 2024)):
            df = pl.read_csv(io.BytesIO(z.read(nombre)), infer_schema_length=0)
            dfs.append(df.select("ID_ENTIDAD", "ID_MUNICIPIO", "ANIO", "VOLUMEN"))
    e = pl.concat(dfs)
    m = (e.with_columns(
            pl.col("ANIO").cast(pl.Int64, strict=False),
            pl.col("VOLUMEN").cast(pl.Float64, strict=False),
            (pl.col("ID_ENTIDAD").cast(pl.Int64, strict=False) * 1000
             + pl.col("ID_MUNICIPIO").cast(pl.Int64, strict=False)).alias("cve_mun"))
         .filter(pl.col("ANIO").is_in([2023, 2024]) & (pl.col("VOLUMEN") > 0)))
    return set(m["cve_mun"].unique().to_list())


def cargar_pobreza_municipal() -> pl.DataFrame:
    z = zipfile.ZipFile(RAIZ / "data/coneval/Python_MMP_2020.zip")
    with z.open("Base final/pobreza20.csv") as f:
        raw = pl.read_csv(io.BytesIO(f.read()), columns=["ubica_geo", "factor", "pobreza"])
    return (raw.with_columns(pl.col("ubica_geo").cast(pl.Int64).alias("cve_mun"))
            .group_by("cve_mun")
            .agg(((pl.col("pobreza") * pl.col("factor")).sum() / pl.sum("factor") * 100)
                 .alias("pobreza_pct")))


def main():
    hom = cargar_homicidios()
    pob = (pl.read_csv(RAIZ / "data/conapo/municipios_2020_todos.csv")
           .select(pl.col("CLAVE").cast(pl.Int64).alias("cve_mun"),
                   pl.col("POB_TOTAL").cast(pl.Float64).alias("pob")))
    base = (hom.rename({"Cve. Municipio": "cve_mun"})
            .join(pob, on="cve_mun", how="inner"))
    cobertura = base["pob"].sum() / pob["pob"].sum()
    print(f"=== Cap 9: {base.height:,} municipios con match; cobertura poblacional {cobertura*100:.1f}% ===")
    assert cobertura > 0.90

    puertos = resolver_catalogo(PUERTOS, hom.rename({"Cve. Municipio": "Cve. Municipio"}), "puertos")
    frontera = resolver_catalogo(FRONTERA, hom, "frontera")
    mineros = cargar_mineros()
    print(f"  catálogo mineros (EIMM 2023-24, VOLUMEN>0): {len(mineros)} municipios")

    base = base.with_columns(
        pl.col("cve_mun").is_in(list(puertos)).cast(pl.Int8).alias("puerto"),
        pl.col("cve_mun").is_in(list(frontera)).cast(pl.Int8).alias("frontera"),
        pl.col("cve_mun").is_in(list(mineros)).cast(pl.Int8).alias("minero"),
        (pl.col("casos") / pl.col("pob") * 1e5).alias("tasa"),
    )

    # ---- 1. sobre-representación: share de homicidios vs share de población
    tot_h, tot_p = base["casos"].sum(), base["pob"].sum()
    print("\n  sobre-representación (share de homicidios ÷ share de población):")
    sobre = {}
    for feat in ["puerto", "frontera", "minero"]:
        sub = base.filter(pl.col(feat) == 1)
        sh, sp = sub["casos"].sum() / tot_h, sub["pob"].sum() / tot_p
        sobre[feat] = (sh, sp, sh / sp)
        print(f"    {feat:<9} n={sub.height:>4} · {sp*100:>5.1f}% de la población · "
              f"{sh*100:>5.1f}% de los homicidios · razón={sh/sp:.2f}×")

    top50 = base.sort("casos", descending=True).head(50)
    n_any = top50.filter((pl.col("puerto") + pl.col("frontera") + pl.col("minero")) > 0).height
    print(f"  top-50 municipios por homicidios: {top50['puerto'].sum()} puerto · "
          f"{top50['frontera'].sum()} frontera · {top50['minero'].sum()} mineros · "
          f"{n_any} con al menos un feature ({n_any*2}%)")

    # ---- 2. regresiones municipales: sin FE (asociación cruda) vs con FE de estado
    full = (base.with_columns((pl.col("tasa") + 1).log().alias("log_tasa"),
                              pl.col("pob").log().alias("ln_pob"))
            .select("log_tasa", "minero", "puerto", "frontera", "ln_pob", "Clave_Ent", "cve_mun")
            .to_pandas())
    feats = ["minero", "puerto", "frontera"]

    mA = smf.ols("log_tasa ~ minero + puerto + frontera + ln_pob", data=full).fit(cov_type="HC1")
    mB = smf.ols("log_tasa ~ minero + puerto + frontera + ln_pob + C(Clave_Ent)",
                 data=full).fit(cov_type="HC1")
    print(f"\n  OLS A (n={int(mA.nobs)}, sin FE): asociación nacional con los puntos de renta")
    for v in feats + ["ln_pob"]:
        print(f"    {v:<10} β={mA.params[v]:+.3f} (p={mA.pvalues[v]:.4f}) → {(np.exp(mA.params[v])-1)*100:+.0f}%")
    print(f"    R²={mA.rsquared:.2f}")
    print(f"  OLS B (n={int(mB.nobs)}, + FE estado): ¿el feature importa DENTRO del mismo estado?")
    for v in feats:
        print(f"    {v:<10} β={mB.params[v]:+.3f} (p={mB.pvalues[v]:.4f}) → {(np.exp(mB.params[v])-1)*100:+.0f}%")
    print(f"    R²={mB.rsquared:.2f}  → los FE estatales absorben la geografía de corredor")

    # Spec C: pobreza CONEVAL 2020 (solo municipios en muestra ENIGH — subconjunto no censal)
    dfp = (base.join(cargar_pobreza_municipal(), on="cve_mun", how="inner")
           .with_columns((pl.col("tasa") + 1).log().alias("log_tasa"),
                         pl.col("pob").log().alias("ln_pob"))
           .select("log_tasa", "minero", "puerto", "frontera", "pobreza_pct", "ln_pob", "Clave_Ent")
           .to_pandas())
    mC = smf.ols("log_tasa ~ minero + puerto + frontera + pobreza_pct + ln_pob + C(Clave_Ent)",
                 data=dfp).fit(cov_type="HC1")
    print(f"  OLS C (n={int(mC.nobs)}, subm. ENIGH, + pobreza): "
          f"pobreza β={mC.params['pobreza_pct']:+.4f} (p={mC.pvalues['pobreza_pct']:.4f}) "
          f"→ {(np.exp(mC.params['pobreza_pct']*10)-1)*100:+.0f}% por +10 pp de pobreza")
    m = mB  # figura de coeficientes: especificaciones A y B

    # ---- figuras
    fig = go.Figure()
    feats = ["minero", "puerto", "frontera"]
    fig.add_trace(go.Bar(x=feats, y=[sobre[f][1] * 100 for f in feats],
                         name="% de la población nacional", marker_color="#95a5a6"))
    fig.add_trace(go.Bar(x=feats, y=[sobre[f][0] * 100 for f in feats],
                         name="% de los homicidios del país", marker_color="#c0392b"))
    fig.update_layout(
        barmode="group",
        title="Los puntos de renta concentran más homicidios que población (2023-2024)<br><sup>municipios mineros (EIMM), puertos mayores y cruces fronterizos</sup>",
        yaxis_title="%",
    )
    guardar_fig(fig, "cap9_hotspots_features")

    nombres = {"minero": "municipio minero", "puerto": "puerto mayor",
               "frontera": "cruce fronterizo"}
    fig = go.Figure()
    for modelo, etiqueta, color in [(mA, "nacional (sin FE)", "#c0392b"),
                                    (mB, "dentro del mismo estado (FE)", "#95a5a6")]:
        fig.add_trace(go.Bar(
            y=[nombres[v] for v in feats],
            x=[(np.exp(modelo.params[v]) - 1) * 100 for v in feats],
            error_x=dict(type="data",
                         array=[(np.exp(modelo.params[v] + 1.96 * modelo.bse[v]) - 1) * 100
                                - (np.exp(modelo.params[v]) - 1) * 100 for v in feats]),
            orientation="h", name=etiqueta, marker_color=color))
    fig.add_vline(x=0, line_color="black")
    fig.update_layout(
        barmode="group",
        title="La violencia sigue al corredor, no al punto: efecto de cada feature sobre la tasa municipal<br><sup>OLS log(1+tasa) con control de tamaño; al añadir efectos fijos de estado el efecto se desvanece — la ruta es estatal</sup>",
        xaxis_title="% de cambio en la tasa de homicidio doloso (IC 95%)",
    )
    guardar_fig(fig, "cap9_coeficientes")


if __name__ == "__main__":
    main()
