"""
Cap 10 T5 — El valor del corredor, medido en el municipio (SECTUR 7_10, 1994-2024).

El T3 del Cap 10 midió flujo × violencia a nivel ESTATAL (corr +0.28, tollbooths
2021-25, ~45% de cobertura). Aquí se usa la serie larga de SECTUR (7_10.xlsx del
Compendio CETM2024: ~200 tramos de cuota × 31 años, todas las clases de vehículo,
suma exacta al total nacional) para bajar el test al MUNICIPIO — la unidad donde el
Cap 9 mostró que los features de renta se desvanecen con efectos fijos estatales.

  A. Parseo tramo × año + mapeo endpoints → municipios (match por nombre contra el
     catálogo SESNSP + alias manual; el trazado completo no existe — nota al pie).
  B. Cross-section: sobre-representación de municipios-corredor en homicidios
     2023-24 y regresiones del marco Cap 9 (± FE estatales; intensidad de tráfico).
  C. Dinámico: Δ tráfico 2015→2024 × Δ tasa de homicidio (2015-16 → 2023-24).

Figura → centralismo/informe/figuras/cap10_corredor_municipal.png
Run: uv run python scripts/centralismo/cap10_corredor_municipal.py
"""

import io
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import polars as pl
import statsmodels.formula.api as smf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from comun import RAIZ, guardar_fig
from cap9_rutas_violencia import MESES, _clave

SUBFILAS = {"Automóviles", "Autobuses", "Camiones de Carga", "Otros Vehículos",
            "T o t a l"}

# alias por TRAMO completo (casos no separables por endpoint) — None = se excluye
ALIAS_TRAMO = {
    "Puente Int. Camargo": [28007],            # Camargo, Tamaulipas (no Chihuahua)
    "Puente Int. Juárez - Lincoln": [28027],   # puente internacional de Nuevo Laredo
    "Cuauhtémoc - Osiris": None,               # casetas urbanas ZMVM sin municipio claro
    "Rayón - Vicente Guerrero (Aut. Río Verde - Cd. Valles)": [24025],
    "Puente San Miguel": None,                 # localidad no identificable con certeza
    "Arco Norte Cd. De México (Caseta Tula)": [13076],  # Tula de Allende
    "Viaducto elevado de Tlalpan": [9012],
}
# alias por endpoint (localidades ≠ cabecera municipal, nombres viejos, ambigüedades
# resueltas por el corredor del tramo) — None = punto metropolitano sin municipio único
ALIAS = {
    "México": None, "Norte de la Ciudad de México": None,
    "El Sueco": None, "La Gloria": None, "La Carbonera": None,
    "Ecatepec": 15033, "Pachuca": 13048, "La Marquesa": 15062, "Chamapa": 15057,
    "Lechería": 15109, "Peñón": 9017, "Pirámides": 15092, "Constituyentes": 9016,
    "Cardel": 30016, "Oriente de San Luis Potosí": 24028, "La Rumorosa": 2003,
    "Víctor Rosales": 32005, "Nororiente de Querétaro": 22014, "Puerto México": 30039,
    "Grijalva": 27004, "Villahermosa": 27004, "Oriente de Saltillo": 5030,
    "Colorado": 22011, "Nueva Italia": 16053, "Cadereyta": 19009, "Dovalí Jaime": 30039,
    "La Tinaja": 30047, "Coseoleacaque": 30048, "El Centinela": 2002, "La Pera": 17020,
    "Nepantla": 15084, "Tulancingo": 13077, "Nuevo Necaxa": 21089, "Yerbanis": 10005,
    "Poniente de Tampico": 28038, "Sur de Guadalajara": 14097, "Estación Don": 26042,
    "Villa Ahumada": 8001, "Cuacnopalan": 21110, "Camargo": 8011, "Jiménez": 8032,
    "Matamoros": 28022, "Pánuco": 30123, "Lázaro Cárdenas": 16052, "Santa Ana": 26058,
    "Tuxpan": 30189, "Tuxpan Puente Tuxpan": 30189, "Nogales": 26043,
    "San Cristobal": 7078, "San Fernando": 28035, "Cárdenas": 27002, "Agua Dulce": 30010,
    "Cuautla": 17006, "Int. Reynosa": 28032, "Int. Juárez": 8037,
}


def leer_710() -> pd.DataFrame:
    """7_10.xlsx → tramo × año (total de vehículos), validado contra la fila total."""
    z = zipfile.ZipFile(RAIZ / "data/sectur/compendio_estadistico/CETM2024.zip")
    df = pd.read_excel(io.BytesIO(z.read("CETM2024/7_10.xlsx")), sheet_name=0, header=None)
    años = {c: int(df.iloc[2, c]) for c in range(3, df.shape[1])}
    filas = df[df[2].notna()].copy()
    filas["tramo"] = filas[2].astype(str).str.strip()
    tr = filas[~filas["tramo"].isin(SUBFILAS)].copy()
    for c, a in años.items():
        tr[a] = pd.to_numeric(tr[c], errors="coerce").fillna(0.0)
    tot = df[df[2] == "T o t a l"].index[0]
    for a, c in [(1994, 3), (2024, 33)]:
        assert abs(tr[a].sum() - float(df.iloc[tot, c])) < 1, a  # suma = fila total
    print(f"  7_10: {len(tr)} tramos · total nacional 2024 = {tr[2024].sum()/1e6:.0f} M "
          f"de cruces (validado contra la fila 'T o t a l')")
    return tr[["tramo"] + list(años.values())]


def endpoints(tramo: str) -> list[str]:
    if tramo in ALIAS_TRAMO:
        return []  # resuelto a nivel tramo
    n = re.sub(r"\(.*?\)", "", tramo).strip()
    m = re.match(r"(?i)libramiento\s+(?:de\s+)?(?:la\s+)?(.+)", n)
    eps = [m.group(1)] if (m and " - " not in n) else n.split(" - ")
    out = []
    for e in eps:
        e = re.sub(r"(?i)\s+y\s+(lib\.?|libramiento|ramal|acceso).*", "", e.strip())
        e = re.sub(r"(?i)^(libramiento\s+(de\s+)?|ent\.|entronque|cd\.|ciudad|puente)\s*", "", e)
        e = re.sub(r"(?i)-ent\..*", "", e).strip()
        out.append(e)
    return out


def mapear_tramos(tr: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """tramo → lista de cve_mun (endpoints); imprime cobertura de tráfico 2024."""
    cat_df = (pl.scan_parquet(RAIZ / "data/incidencia_delictiva/incidencia_fuero_comun/"
                              "incidencia_delictiva_fuero_comun.parquet")
              .filter(pl.col("Año") == 2024)
              .select("Cve. Municipio", "Municipio").unique().collect()
              .with_columns(pl.col("Municipio").map_elements(_clave, return_dtype=pl.Utf8)
                            .alias("mc")))
    cat: dict[str, list[int]] = {}
    for r in cat_df.iter_rows(named=True):
        cat.setdefault(r["mc"], []).append(int(r["Cve. Municipio"]))

    def resolver(e: str):
        if e in ALIAS:
            return ALIAS[e]
        hit = cat.get(_clave(e), [])
        if len(hit) == 1:
            return hit[0]
        if len(hit) > 1:
            return None  # ambiguo nacional sin alias: no se asigna
        cont = [k for k in cat if _clave(e) in k]  # fallback contains-único
        if len(cont) == 1 and len(cat[cont[0]]) == 1:
            return cat[cont[0]][0]
        return None

    munis, n_ep, n_ok = [], 0, 0
    for _, r in tr.iterrows():
        if r["tramo"] in ALIAS_TRAMO:
            munis.append(ALIAS_TRAMO[r["tramo"]] or [])
            continue
        res = [resolver(e) for e in endpoints(r["tramo"])]
        n_ep += len(res)
        n_ok += sum(1 for x in res if x is not None)
        munis.append(sorted({x for x in res if x is not None}))
    tr = tr.copy()
    tr["munis"] = munis
    con = tr[tr["munis"].map(len) > 0]
    cobertura = con[2024].sum() / tr[2024].sum()
    print(f"  mapeo endpoints→municipio: {n_ok}/{n_ep} endpoints resueltos · "
          f"{len(con)}/{len(tr)} tramos con ≥1 municipio · "
          f"{cobertura*100:.1f}% del tráfico 2024 asignado")
    assert cobertura > 0.70
    return tr, cobertura


def trafico_municipal(tr: pd.DataFrame, año: int) -> pl.DataFrame:
    reg = {}
    for _, r in tr.iterrows():
        for m in r["munis"]:
            reg[m] = reg.get(m, 0.0) + float(r[año])
    return pl.DataFrame({"cve_mun": list(reg), f"traf_{año}": list(reg.values())})


def homicidios_mun(años: list[int]) -> pl.DataFrame:
    return (pl.scan_parquet(RAIZ / "data/incidencia_delictiva/incidencia_fuero_comun/"
                            "incidencia_delictiva_fuero_comun.parquet")
            .filter(pl.col("Año").is_in(años)
                    & (pl.col("Subtipo de delito") == "Homicidio doloso"))
            .with_columns(pl.sum_horizontal(MESES).alias("casos"))
            .group_by("Cve. Municipio", "Clave_Ent")
            .agg((pl.sum("casos") / len(años)).alias("casos"))
            .collect().rename({"Cve. Municipio": "cve_mun"}))


def main():
    print("=== A. La serie larga de la cuota (SECTUR 7_10) ===")
    tr = leer_710()
    tr, _ = mapear_tramos(tr)

    pob = (pl.read_csv(RAIZ / "data/conapo/municipios_2020_todos.csv")
           .select(pl.col("CLAVE").cast(pl.Int64).alias("cve_mun"),
                   pl.col("POB_TOTAL").cast(pl.Float64).alias("pob")))
    base = (homicidios_mun([2023, 2024]).join(pob, on="cve_mun", how="inner")
            .join(trafico_municipal(tr, 2024), on="cve_mun", how="left")
            .with_columns(pl.col("traf_2024").fill_null(0.0))
            .with_columns((pl.col("traf_2024") > 0).cast(pl.Int8).alias("corredor"),
                          (pl.col("casos") / pl.col("pob") * 1e5).alias("tasa")))

    print("\n=== B. Cross-section municipal (marco del Cap 9) ===")
    tot_h, tot_p = base["casos"].sum(), base["pob"].sum()
    cor = base.filter(pl.col("corredor") == 1)
    sh, sp = cor["casos"].sum() / tot_h, cor["pob"].sum() / tot_p
    print(f"  municipios-corredor (endpoint de tramo de cuota): n={cor.height} · "
          f"{sp*100:.1f}% de la población · {sh*100:.1f}% de los homicidios 2023-24 · "
          f"razón = {sh/sp:.2f}×")
    q3 = cor["traf_2024"].quantile(0.75)
    top = cor.filter(pl.col("traf_2024") >= q3)
    sh_t, sp_t = top["casos"].sum() / tot_h, top["pob"].sum() / tot_p
    print(f"  top-cuartil por tráfico (≥{q3/1e6:.1f} M de cruces/año): n={top.height} · "
          f"{sp_t*100:.1f}% de la población · {sh_t*100:.1f}% de los homicidios · "
          f"razón = {sh_t/sp_t:.2f}×")

    full = (base.with_columns((pl.col("tasa") + 1).log().alias("log_tasa"),
                              pl.col("pob").log().alias("ln_pob"))
            .select("log_tasa", "corredor", "ln_pob", "Clave_Ent", "traf_2024", "pob")
            .to_pandas())
    mA = smf.ols("log_tasa ~ corredor + ln_pob", data=full).fit(cov_type="HC1")
    mB = smf.ols("log_tasa ~ corredor + ln_pob + C(Clave_Ent)", data=full).fit(cov_type="HC1")
    print(f"  OLS binario (n={int(mA.nobs)}): corredor β={mA.params['corredor']:+.3f} "
          f"(p={mA.pvalues['corredor']:.4f}) → {(np.exp(mA.params['corredor'])-1)*100:+.0f}% "
          f"sin FE; con FE estado β={mB.params['corredor']:+.3f} "
          f"(p={mB.pvalues['corredor']:.4f}) → {(np.exp(mB.params['corredor'])-1)*100:+.0f}%")

    dentro = full[full["traf_2024"] > 0].copy()
    dentro["ln_traf"] = np.log(dentro["traf_2024"])
    mC = smf.ols("log_tasa ~ ln_traf + ln_pob", data=dentro).fit(cov_type="HC1")
    mD = smf.ols("log_tasa ~ ln_traf + ln_pob + C(Clave_Ent)", data=dentro).fit(cov_type="HC1")
    print(f"  OLS intensidad, solo corredor (n={int(mC.nobs)}): duplicar tráfico → "
          f"{(2**mC.params['ln_traf']-1)*100:+.1f}% (p={mC.pvalues['ln_traf']:.4f}) sin FE; "
          f"{(2**mD.params['ln_traf']-1)*100:+.1f}% (p={mD.pvalues['ln_traf']:.4f}) con FE")
    zmvm = dentro[~dentro["Clave_Ent"].isin([9, 15])]
    mE = smf.ols("log_tasa ~ ln_traf + ln_pob + C(Clave_Ent)", data=zmvm).fit(cov_type="HC1")
    print(f"  robustez sin ZMVM (ent 9/15, casetas urbanas; n={int(mE.nobs)}): "
          f"duplicar tráfico → {(2**mE.params['ln_traf']-1)*100:+.1f}% "
          f"(p={mE.pvalues['ln_traf']:.4f}) con FE")

    print("\n=== C. Dinámico: Δ tráfico 2015→2024 × Δ tasa (2015-16 → 2023-24) ===")
    t15 = trafico_municipal(tr, 2015)
    h15 = (homicidios_mun([2015, 2016]).select("cve_mun", pl.col("casos").alias("casos0")))
    din = (base.filter(pl.col("traf_2024") > 0)
           .join(t15, on="cve_mun", how="inner").filter(pl.col("traf_2015") > 0)
           .join(h15, on="cve_mun", how="left").with_columns(pl.col("casos0").fill_null(0.0))
           .with_columns(
               (pl.col("traf_2024") / pl.col("traf_2015")).log().alias("d_traf"),
               ((pl.col("casos") - pl.col("casos0")) / pl.col("pob") * 1e5).alias("d_tasa")))
    c_din = np.corrcoef(din["d_traf"], din["d_tasa"])[0, 1]
    print(f"  municipios con tráfico en ambas puntas: n={din.height} · "
          f"corr(Δlog tráfico, Δ tasa homicidio) = {c_din:+.2f}")
    dfd = din.select("d_traf", "d_tasa", "Clave_Ent").to_pandas()
    mF = smf.ols("d_tasa ~ d_traf", data=dfd).fit(cov_type="HC1")
    mG = smf.ols("d_tasa ~ d_traf + C(Clave_Ent)", data=dfd).fit(cov_type="HC1")
    print(f"  OLS: β={mF.params['d_traf']:+.2f} (p={mF.pvalues['d_traf']:.3f}) sin FE; "
          f"β={mG.params['d_traf']:+.2f} (p={mG.pvalues['d_traf']:.3f}) con FE estado "
          f"[Δ tasa por 100k por duplicación ≈ β×ln2]")
    print("  caveats: tráfico total (todas las clases; T3 usaba solo autos), asignación "
          "por endpoints (no trazado), tramos que abren/cierran entre 2015 y 2024")

    # figura: intensidad de tráfico vs tasa municipal
    cifra = base.filter(pl.col("corredor") == 1).with_columns(
        (pl.col("traf_2024") / 1e6).alias("traf_m"))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=cifra["traf_m"], y=cifra["tasa"], mode="markers", showlegend=False,
        marker=dict(size=7, color="#1f77b4", opacity=0.65),
        text=[f"{m}" for m in cifra["cve_mun"]], hoverinfo="text+x+y"))
    b, a = np.polyfit(np.log10(cifra["traf_m"]), cifra["tasa"], 1)
    xs = np.logspace(np.log10(float(cifra["traf_m"].min())),
                     np.log10(float(cifra["traf_m"].max())), 50)
    fig.add_trace(go.Scatter(x=xs, y=a + b * np.log10(xs), mode="lines", showlegend=False,
                             line=dict(color="#7f8c8d", dash="dash", width=1)))
    fig.update_layout(
        title="La renta que circula no marca el punto: el valor del corredor no gradúa "
              "la violencia municipal<br><sup>municipios endpoint de tramos de cuota: "
              "cruces anuales 2024 (SECTUR 7_10, log) vs tasa de homicidio 2023-24</sup>",
        xaxis=dict(type="log", title="cruces de cuota 2024 (millones, escala log)"),
        yaxis_title="homicidios dolosos por 100k (prom. 2023-24)")
    guardar_fig(fig, "cap10_corredor_municipal")


if __name__ == "__main__":
    main()
