"""
Cap 4 — Test H4 fuera de muestra: la ENIGH 2024 (medición INEGI post-CONEVAL).

Pregunta pendiente del INFORME: ¿siguió cayendo la pobreza de ingreso mientras la
carencia de acceso a salud empeora? El veredicto "paliativo" del Cap 4 se estimó con
CONEVAL 2016-2022; la medición 2024 la hizo el INEGI (transferencia de funciones
vigente desde julio 2025) y su insumo, la ENIGH 2024, ya está en el repo.

Diseño (sin descargas, todo de data/coneval/ y data/inegi/enigh/):
  A. Serie oficial 2016-2022 desde los microdatos MMP del CONEVAL ("Base final/
     pobrezaYY.csv"): pobreza multidimensional, ingreso < LPI (plp), carencia de
     salud (ic_asalud). Las líneas de pobreza por ingresos se RECUPERAN de los
     propios microdatos (frontera de ictpc entre plp=1 y plp=0, por rururb) — no se
     hardcodea ninguna cifra externa.
  B. Constructo comparable desde la ENIGH cruda en las 5 olas: ingreso pc mensual =
     (ing_cor − estim_alqu)/3/tot_integ (el ICTPC de CONEVAL excluye renta imputada);
     headcount bajo la línea de cada ola; la brecha vs. oficial por ola es el control
     de validez. Línea 2024 = línea 2022 × deflactor, en 3 escenarios (el deflactor
     PIB subestimó la inflación de la canasta en 2020-22 — se documenta en A).
  C. Carencia de salud: proxy "sin afiliación" replicable en las 5 olas (2016/18:
     ni segpop ni atemed; 2020/22: ni pop_insabi ni atemed; 2024: inst_9 explícito),
     con brecha vs. ic_asalud oficial por ola.
  D. Robustez distribucional: ingreso pc real por decil 2022→2024 (no depende de
     ninguna línea).

Trampas ENIGH nuevas (además de las 8 de la memoria del dataset):
  - poblacion 2016 NO trae factor y su folioviv pierde ceros a la izquierda frente a
    concentradohogar: unir SOLO con llaves casteadas a entero.
  - El cuestionario de salud se reestructuró en 2024: desaparece pop_insabi y atemed
    cambia de filtro (63% nulos); la afiliación pasa a inst_1..inst_9 multiselect,
    con inst_9 = "Sin afiliación".
  - El bloque PIBE "Índice de precios implícitos" trae dos filas por año (índice y
    variación): deduplicar con keep="first".

Run: uv run python scripts/centralismo/cap4_h4_fuera_muestra.py
"""

import io
import sys
import zipfile
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from comun import RAIZ, PIBE_TOTAL, guardar_fig, leer_pibe

OLAS = [2016, 2018, 2020, 2022, 2024]
ZIPS_ENIGH = {
    2016: "conjunto_de_datos_enigh2016_nueva_serie_csv.zip",
    2018: "conjunto_de_datos_enigh_2018_ns_csv.zip",
    2020: "conjunto_de_datos_enigh_ns_2020_csv.zip",
    2022: "conjunto_de_datos_enigh_ns_2022_csv.zip",
    2024: "conjunto_de_datos_enigh2024_ns_csv.zip",
}
# escenarios de actualización de la línea 2022→2024: deflactor PIB (calculado), y dos
# techos que cubren con holgura el sesgo canasta-vs-deflactor observado en 2016-2022
ESCENARIOS_2024 = {"deflactor PIB": None, "+10%": 1.10, "+15%": 1.15}


def leer_tabla(año: int, tabla: str, columnas: list[str]) -> pl.DataFrame:
    z = zipfile.ZipFile(RAIZ / "data/inegi/enigh" / ZIPS_ENIGH[año])
    ruta = next(n for n in z.namelist()
                if n.rsplit("/", 1)[-1].startswith(f"conjunto_de_datos_{tabla}")
                and n.endswith(".csv") and "bitacora" not in n)
    with z.open(ruta) as f:
        return pl.read_csv(io.BytesIO(f.read()), columns=columnas, infer_schema_length=0)


def deflactor_pibe() -> dict[int, float]:
    d = leer_pibe(PIBE_TOTAL, bloque="Índice de precios implícitos base 2018=100",
                  incluir_nacional=True)
    nac = (d.filter(pl.col("cve_ent") == 0)
           .unique(subset=["año"], keep="first", maintain_order=True))
    return {r["año"]: r["valor"] for r in nac.iter_rows(named=True)}


def oficial_mmp() -> tuple[pl.DataFrame, dict[int, dict[str, float]]]:
    """Serie oficial CONEVAL 2016-2022 + líneas LPI implícitas por ola."""
    print("\n=== A. Serie oficial CONEVAL (microdatos MMP en repo) ===")
    filas, lineas = [], {}
    for año in OLAS[:-1]:
        z = zipfile.ZipFile(RAIZ / f"data/coneval/Python_MMP_{año}.zip")
        with z.open(f"Base final/pobreza{str(año)[2:]}.csv") as f:
            m = pl.read_csv(io.BytesIO(f.read()),
                            columns=["factor", "rururb", "ictpc", "plp", "pobreza",
                                     "ic_asalud"],
                            infer_schema_length=0)
        m = m.with_columns(pl.all().cast(pl.Float64, strict=False))

        def tasa(v):
            ok = m.filter(pl.col(v).is_not_null())
            return (ok[v] * ok["factor"]).sum() / ok["factor"].sum() * 100

        lin = {}
        for cod, nom in [(0.0, "urbana"), (1.0, "rural")]:
            d = m.filter(pl.col("rururb") == cod)
            hi = d.filter(pl.col("plp") == 1)["ictpc"].max()
            lo = d.filter(pl.col("plp") == 0)["ictpc"].min()
            assert lo - hi < 5, (año, nom, hi, lo)  # frontera nítida = línea recuperada
            lin[nom] = (hi + lo) / 2
        lineas[año] = lin
        filas.append((año, m["factor"].sum() / 1e6, tasa("pobreza"), tasa("plp"),
                      tasa("ic_asalud")))
        print(f"  {año}: pob {filas[-1][1]:5.1f}M · pobreza {filas[-1][2]:5.2f}% · "
              f"ingreso<LPI {filas[-1][3]:5.2f}% · carencia salud {filas[-1][4]:5.2f}% · "
              f"LPI urbana={lin['urbana']:,.2f} rural={lin['rural']:,.2f} MXN/mes")

    ofi = pl.DataFrame(filas, schema=["año", "pob", "pobreza", "plp", "ic_asalud"],
                       orient="row")
    # anclas duras (cifras publicadas CONEVAL 2022: 36.3 / 43.5 / 39.1; LPI ago-2022)
    f22 = ofi.filter(pl.col("año") == 2022)
    assert 35.8 < f22["pobreza"][0] < 36.8, f22
    assert 43.0 < f22["plp"][0] < 44.0, f22
    assert 38.5 < f22["ic_asalud"][0] < 39.6, f22
    assert 4150 < lineas[2022]["urbana"] < 4170 and 2965 < lineas[2022]["rural"] < 2975
    return ofi, lineas


def validar_deflactor(lineas: dict, defl: dict) -> None:
    print("\n  ¿el deflactor PIB reproduce el crecimiento histórico de la LPI urbana?")
    sesgos = []
    for a, b in [(2016, 2018), (2018, 2020), (2020, 2022)]:
        g_lin = lineas[b]["urbana"] / lineas[a]["urbana"]
        g_def = defl[b] / defl[a]
        sesgos.append(g_lin / g_def - 1)
        print(f"    {a}→{b}: línea ×{g_lin:.3f} vs deflactor ×{g_def:.3f} "
              f"(sesgo {sesgos[-1]:+.1%})")
    print(f"  → sesgo histórico máximo {max(sesgos):+.1%} (2020-22): los escenarios "
          f"+10%/+15% de la línea 2024 lo cubren con holgura de 2-3×")


def pobreza_ingreso(lineas: dict, defl: dict, ofi: pl.DataFrame) -> dict:
    """Headcount bajo LPI con constructo propio en las 5 olas."""
    print("\n=== B. Pobreza de ingreso: constructo ENIGH vs oficial ===")
    print("  constructo: (ing_cor − estim_alqu)/3/tot_integ, peso = factor×tot_integ, "
          "rural = tam_loc 4")
    resultados = {}
    for año in OLAS:
        c = leer_tabla(año, "concentradohogar",
                       ["factor", "tot_integ", "ing_cor", "estim_alqu", "tam_loc"])
        c = (c.with_columns(pl.col(x).cast(pl.Float64, strict=False)
                            for x in ["factor", "tot_integ", "ing_cor", "estim_alqu"])
             .with_columns(
                 ((pl.col("ing_cor") - pl.col("estim_alqu")) / 3
                  / pl.col("tot_integ")).alias("ing_pc"),
                 (pl.col("factor") * pl.col("tot_integ")).alias("peso"),
                 (pl.col("tam_loc").str.strip_chars() == "4").alias("rural")))
        tot = c["peso"].sum()
        assert 115e6 < tot < 135e6, (año, tot)

        def headcount(lin_urb, lin_rur):
            pobre = c.filter(pl.col("ing_pc")
                             < pl.when(pl.col("rural")).then(lin_rur).otherwise(lin_urb))
            return pobre["peso"].sum() / tot * 100

        if año < 2024:
            h = headcount(lineas[año]["urbana"], lineas[año]["rural"])
            o = ofi.filter(pl.col("año") == año)["plp"][0]
            resultados[año] = {"constructo": h}
            print(f"  {año}: constructo {h:5.2f}% vs oficial {o:5.2f}% "
                  f"(brecha {h - o:+.2f})")
        else:
            base = defl[2024] / defl[2022]
            resultados[año] = {}
            for nombre, factor in ESCENARIOS_2024.items():
                factor = base if factor is None else factor
                h = headcount(lineas[2022]["urbana"] * factor,
                              lineas[2022]["rural"] * factor)
                resultados[año][nombre] = h
                print(f"  2024 línea {nombre} (×{factor:.3f} → urbana "
                      f"{lineas[2022]['urbana'] * factor:,.0f}): {h:5.2f}%")
        resultados[año]["base"] = c
    return resultados


def carencia_salud(ofi: pl.DataFrame) -> dict[int, float]:
    """Proxy sin-afiliación replicable en las 5 olas, vs ic_asalud oficial."""
    print("\n=== C. Carencia de salud: proxy sin afiliación vs oficial ===")
    proxy = {}
    for año in OLAS:
        extra = (["segpop"] if año <= 2018 else
                 ["pop_insabi"] if año <= 2022 else ["inst_9"])
        p = leer_tabla(año, "poblacion", ["folioviv", "foliohog", "atemed"] + extra)
        c = leer_tabla(año, "concentradohogar", ["folioviv", "foliohog", "factor"])
        p = (p.with_columns(pl.col("folioviv").cast(pl.Int64),
                            pl.col("foliohog").cast(pl.Int64))
             .join(c.with_columns(pl.col("folioviv").cast(pl.Int64),
                                  pl.col("foliohog").cast(pl.Int64),
                                  pl.col("factor").cast(pl.Float64)),
                   on=["folioviv", "foliohog"], how="inner"))
        assert p.height > 200_000 and 115e6 < p["factor"].sum() < 135e6, año
        if año <= 2022:
            sp = extra[0]
            sin = p.filter((pl.col(sp).str.strip_chars() != "1")
                           & (pl.col("atemed").str.strip_chars() != "1"))
        else:
            sin = p.filter(pl.col("inst_9").str.strip_chars() == "9")
        proxy[año] = sin["factor"].sum() / p["factor"].sum() * 100
        o = ("" if año == 2024 else
             f" vs oficial {ofi.filter(pl.col('año') == año)['ic_asalud'][0]:5.2f}% "
             f"(brecha {proxy[año] - ofi.filter(pl.col('año') == año)['ic_asalud'][0]:+.2f})")
        print(f"  {año}: sin afiliación {proxy[año]:5.2f}%{o}")
    return proxy


def deciles_reales(res_ing: dict, defl: dict) -> None:
    """Crecimiento real del ingreso pc por decil 2022→2024 (sin líneas de por medio)."""
    print("\n=== D. Robustez: ingreso pc real por decil, 2022→2024 (MXN de 2018) ===")
    medias = {}
    for año in [2022, 2024]:
        c = (res_ing[año]["base"].sort("ing_pc")
             .with_columns((pl.col("peso").cum_sum() / pl.col("peso").sum())
                           .alias("acum")))
        medias[año] = [
            (d.filter(pl.col("decil") == i)["ing_pc"].dot(
                d.filter(pl.col("decil") == i)["peso"])
             / d.filter(pl.col("decil") == i)["peso"].sum()) * 100 / defl[año]
            for d in [c.with_columns(((pl.col("acum") * 10).ceil()
                                      .clip(1, 10)).alias("decil"))]
            for i in range(1, 11)]
    print("  decil:   " + " ".join(f"{i:>6d}" for i in range(1, 11)))
    print("  2022:    " + " ".join(f"{v:6,.0f}" for v in medias[2022]))
    print("  2024:    " + " ".join(f"{v:6,.0f}" for v in medias[2024]))
    print("  Δ real:  " + " ".join(f"{b / a - 1:+6.1%}" for a, b in
                                   zip(medias[2022], medias[2024])))


def figura(ofi: pl.DataFrame, res_ing: dict, proxy: dict) -> None:
    años_c = OLAS
    constructo = [res_ing[a]["constructo"] for a in OLAS[:-1]]
    esc = [v for k, v in res_ing[2024].items() if k != "base"]
    constructo.append(sum(esc) / len(esc))
    fig = make_subplots(rows=1, cols=2, subplot_titles=(
        "pobreza de ingreso (% con ingreso < LPI)",
        "carencia de acceso a servicios de salud (%)"))
    fig.add_trace(go.Scatter(x=ofi["año"], y=ofi["plp"], mode="lines+markers",
                             name="oficial CONEVAL (MMP)", line=dict(color="#c0392b")),
                  row=1, col=1)
    fig.add_trace(go.Scatter(
        x=años_c, y=constructo, mode="lines+markers", name="constructo ENIGH",
        line=dict(color="#7f8c8d", dash="dash"),
        error_y=dict(type="data", visible=True,
                     array=[0] * 4 + [max(esc) - constructo[-1]],
                     arrayminus=[0] * 4 + [constructo[-1] - min(esc)])),
        row=1, col=1)
    fig.add_trace(go.Scatter(x=ofi["año"], y=ofi["ic_asalud"], mode="lines+markers",
                             name="oficial CONEVAL (MMP)", line=dict(color="#c0392b"),
                             showlegend=False),
                  row=1, col=2)
    fig.add_trace(go.Scatter(x=años_c, y=[proxy[a] for a in años_c],
                             mode="lines+markers", name="proxy sin afiliación",
                             line=dict(color="#7f8c8d", dash="dash")),
                  row=1, col=2)
    fig.update_layout(
        title="H4 fuera de muestra: la ENIGH 2024 (medición INEGI) extiende ambas series"
              "<br><sup>izq: % de personas bajo la línea de pobreza por ingresos "
              "(2024 = línea 2022 actualizada, banda = escenarios deflactor/+10%/+15%) · "
              "der: % sin afiliación a servicios de salud vs carencia oficial</sup>",
        legend=dict(orientation="h", y=-0.15))
    fig.update_yaxes(rangemode="tozero")
    guardar_fig(fig, "h4_fuera_de_muestra", ancho=1100, alto=520)


def main():
    defl = deflactor_pibe()
    print(f"deflactor PIB implícito 2022→2024: ×{defl[2024] / defl[2022]:.4f}")
    ofi, lineas = oficial_mmp()
    validar_deflactor(lineas, defl)
    res_ing = pobreza_ingreso(lineas, defl, ofi)
    proxy = carencia_salud(ofi)
    deciles_reales(res_ing, defl)

    print("\n=== E. Veredicto H4 fuera de muestra ===")
    esc = {k: v for k, v in res_ing[2024].items() if k != "base"}
    c22 = res_ing[2022]["constructo"]
    print(f"  pobreza de ingreso (constructo): 2022 {c22:.2f}% → 2024 "
          f"{min(esc.values()):.2f}–{max(esc.values()):.2f}% según escenario de línea "
          f"→ {'CAE en todos los escenarios' if max(esc.values()) < c22 else 'NO cae en todos'}")
    print(f"  carencia de salud (proxy): 2018 {proxy[2018]:.2f}% → 2022 {proxy[2022]:.2f}% "
          f"→ 2024 {proxy[2024]:.2f}% — retrocede desde el pico pero sigue "
          f"~{proxy[2024] / proxy[2018]:.1f}× el nivel pre-2020")
    figura(ofi, res_ing, proxy)


if __name__ == "__main__":
    main()
