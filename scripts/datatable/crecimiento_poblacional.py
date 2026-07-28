"""
Tasa de crecimiento poblacional por entidad, censos generales 1895–2020.

Fuente: data/inegi/ccpv/poblacion/poblacion_historica.csv — serie histórica de
población total por entidad en los 14 censos generales de México. Reporta, censo
por censo: población, cambio absoluto y % respecto al censo anterior, y la tasa
de crecimiento media anual (TCMA) — el estándar demográfico de INEGI/CONAPO para
comparar intercensales de duración distinta (los censos de esta serie van de 5 a
11 años de diferencia entre sí, así que un % de cambio simple no es comparable
de un período a otro).

Con --proyeccion, complementa la serie con la reconstrucción/proyección anual de
CONAPO (data/conapo/proyecciones_poblacion/pob_estado_año.parquet, 1990–2040):
llena los huecos intercensales de 1990 a 2020 y agrega los años 2021–2040. El
censo manda en los años censales (CONAPO reporta una cifra "a mitad de año" con
ajuste de subenumeración que no coincide con el conteo censal crudo — por eso se
usa sólo para los años que el censo no cubre, nunca para reemplazarlo).

--proyeccion admite un paso opcional en años (default 1 = anual): --proyeccion 5
sólo agrega los años de CONAPO múltiplos de 5 (1990, 1995, 2000, ...), en vez de
cada año. Los años censales siempre aparecen, sin importar el paso.

Sin --entidad, muestra en su lugar una tabla nacional: las 32 entidades de un
censo (--año, default el último, 2020) con su población, participación (share)
sobre el total nacional, y la diferencia poblacional contra el censo anterior.

--clusters K agrupa las 32 entidades en K regiones de estados VECINOS (por
contigüidad geográfica) y reporta población y share por región — para ver qué
tan concentrada está la población en ciertas zonas del país. Usa --año (default
2020) y es incompatible con --entidad (es un reporte nacional). Baja California
y Baja California Sur no tienen frontera terrestre con el continente (Golfo de
California de por medio); se tratan como vecinas de Sonora y Sinaloa para que
el país quede como un solo bloque conexo.

--mapa (junto con --clusters) genera un choropleth HTML interactivo, coloreado
por la población/share de cada cluster (escala de calor continua — todos los
estados de un mismo cluster comparten color).

--regiones K reporta, para las mismas K regiones de estados vecinos que
--clusters, la trayectoria histórica de población en los 14 censos (cambio,
% y TCMA) — la agrupación geográfica es la misma para todos los censos (no
depende del año), así que a diferencia de --clusters (una foto de un año) esto
muestra el crecimiento de cada región a través del tiempo. Incompatible con
--entidad.

Run: uv run python scripts/datatable/crecimiento_poblacional.py --entidad Jalisco
     uv run python scripts/datatable/crecimiento_poblacional.py --entidad Jalisco --proyeccion
     uv run python scripts/datatable/crecimiento_poblacional.py --entidad Jalisco --proyeccion 5
     uv run python scripts/datatable/crecimiento_poblacional.py
     uv run python scripts/datatable/crecimiento_poblacional.py --año 1990
     uv run python scripts/datatable/crecimiento_poblacional.py --clusters 5
     uv run python scripts/datatable/crecimiento_poblacional.py --clusters 5 --mapa
     uv run python scripts/datatable/crecimiento_poblacional.py --regiones 5
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import plotly.express as px
import polars as pl
from sklearn.cluster import AgglomerativeClustering

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "scripts" / "centralismo"))
from comun import NOMBRE, normalizar_estado

CSV_PATH = RAIZ / "data" / "inegi" / "ccpv" / "poblacion" / "poblacion_historica.csv"
CONAPO_PATH = RAIZ / "data" / "conapo" / "proyecciones_poblacion" / "pob_estado_año.parquet"
GEOJSON_PATH = RAIZ / "data" / "mexico_states.geojson"
DIR = RAIZ / "dashboard_data"
CENSOS = [1895, 1900, 1910, 1921, 1930, 1940, 1950, 1960, 1970, 1980, 1990, 2000, 2010, 2020]

# Vecindad terrestre real de las 32 entidades (curada a mano — no hay geopandas/
# shapely instalado ni tabla de adyacencia ya preparada en el repo). Baja
# California y Baja California Sur no tienen frontera terrestre con el
# continente (separadas por el Golfo de California); se agrega un puente hacia
# Sonora y Sinaloa (sus vecinos más cercanos) para que el país quede como un
# solo bloque conexo — sin ese puente, K=1 sería geográficamente imposible.
VECINOS = {
    1: {14, 32}, 2: {3, 25, 26}, 3: {2, 25, 26}, 4: {7, 23, 27, 31},
    5: {8, 10, 19, 32}, 6: {14, 16}, 7: {4, 20, 27, 30}, 8: {5, 10, 25, 26},
    9: {15, 17}, 10: {5, 8, 18, 25, 32}, 11: {14, 16, 22, 24, 32},
    12: {15, 16, 17, 20, 21}, 13: {15, 21, 22, 24, 29, 30},
    14: {1, 6, 11, 16, 18, 32}, 15: {9, 12, 13, 16, 17, 21, 22, 29},
    16: {6, 11, 12, 14, 15, 22}, 17: {9, 12, 15, 21}, 18: {10, 14, 25, 32},
    19: {5, 24, 28, 32}, 20: {7, 12, 21, 30}, 21: {12, 13, 15, 17, 20, 29, 30},
    22: {11, 13, 15, 16, 24}, 23: {4, 31}, 24: {11, 13, 19, 22, 28, 30, 32},
    25: {2, 3, 8, 10, 18, 26}, 26: {2, 3, 8, 25}, 27: {4, 7, 30},
    28: {19, 24, 30}, 29: {13, 15, 21}, 30: {7, 13, 20, 21, 24, 27, 28},
    31: {4, 23}, 32: {1, 5, 10, 11, 14, 18, 19, 24},
}
for _cve, _vec in VECINOS.items():
    for _v in _vec:
        assert _cve in VECINOS[_v], f"VECINOS no simétrico: {_cve} -> {_v}"


def _entidad(valor):
    """Acepta clave (1-32) o nombre en cualquier régimen ('cdmx', 'Distrito Federal')."""
    if valor.isdigit() and 1 <= int(valor) <= 32:
        return int(valor)
    cve = normalizar_estado(valor)
    if cve is None:
        raise argparse.ArgumentTypeError(
            f"entidad no reconocida: {valor!r} (usa el nombre o la clave 1-32)"
        )
    return cve


def _cve(etiqueta):
    """Nombre de la fuente → CVE_ENT 1-32, o None para las pseudo-filas (Islas
    Marías, la fila combinada de Baja California pre-1930)."""
    return normalizar_estado(etiqueta.split(" (")[0])  # 'Nayarit (Territorio de Tepic)'


def _valor(s):
    """Celda → int, o None si estaba vacía o llevaba el centinela '-'. El
    .replace(' ', '') limpia un residuo de separador de miles con espacio que
    sobrevivió en al menos una celda (Aguascalientes 1940)."""
    s = s.strip().replace(" ", "")
    return None if s in ("", "-") else int(s)


def leer_poblacion_historica() -> pl.DataFrame:
    filas = []
    with open(CSV_PATH, encoding="utf-8", newline="") as f:
        lector = csv.reader(f)
        next(lector)  # encabezado: etiquetas censo_N/poblacion_N genéricas (y una duplicada)
        for row in lector:
            cve = _cve(row[0])
            if cve is None:
                continue
            resto = row[1:]
            for año_str, val_str in zip(resto[0::2], resto[1::2]):
                filas.append({"cve_ent": cve, "entidad": NOMBRE[cve],
                             "año": int(año_str), "poblacion": _valor(val_str),
                             "fuente": "censo"})

    df = pl.DataFrame(filas, schema={"cve_ent": pl.Int16, "entidad": pl.Utf8,
                                     "año": pl.Int16, "poblacion": pl.Int64,
                                     "fuente": pl.Utf8})
    assert df["cve_ent"].n_unique() == 32, f"{df['cve_ent'].n_unique()} entidades, se esperaban 32"
    assert set(df["año"].unique().to_list()) == set(CENSOS)
    return df


def leer_conapo() -> pl.DataFrame:
    """Reconstrucción 1990–2019 + proyección 2020–2040 de CONAPO, anual, sin
    huecos, por entidad (data/conapo/proyecciones_poblacion/pob_estado_año.parquet,
    generado por scripts/prepare_conapo_pob.py)."""
    df = pl.read_parquet(CONAPO_PATH)
    cve_por_estado = {e: normalizar_estado(e) for e in df["estado"].unique().to_list()}
    sin_resolver = [e for e, cve in cve_por_estado.items() if cve is None]
    assert not sin_resolver, f"estados de CONAPO sin resolver: {sin_resolver}"
    return (
        df.with_columns(
            pl.col("estado").replace(cve_por_estado).cast(pl.Int16).alias("cve_ent"),
            pl.when(pl.col("año") >= 2020).then(pl.lit("conapo (proyección)"))
              .otherwise(pl.lit("conapo (reconstrucción)")).alias("fuente"),
        )
        .select(["cve_ent", pl.col("año").cast(pl.Int16), pl.col("pob_total").alias("poblacion"), "fuente"])
    )


def combinar(censo_df, conapo_df, cve, paso=1) -> pl.DataFrame:
    """Serie censal + CONAPO de una entidad: el censo manda en los años
    censales; CONAPO sólo rellena los años que el censo no cubre (huecos
    1990–2020 y la extensión 2021–2040) — ver docstring del módulo. `paso`
    submuestrea los años de CONAPO a sus múltiplos (p. ej. paso=5 → sólo años
    terminados en 0 o 5); los años censales no se ven afectados por `paso`."""
    censo = censo_df.filter(pl.col("cve_ent") == cve).select(["cve_ent", "año", "poblacion", "fuente"])
    conapo = (conapo_df.filter((pl.col("cve_ent") == cve) & ~pl.col("año").is_in(CENSOS)
                               & (pl.col("año") % paso == 0))
              .select(["cve_ent", "año", "poblacion", "fuente"]))
    return pl.concat([censo, conapo]).sort("año")


def calcular(df, cve):
    """Serie de una entidad con cambio absoluto, % y TCMA respecto al año
    anterior CON DATO (no necesariamente el renglón inmediato anterior — algunas
    entidades tienen huecos, p. ej. Quintana Roo no tiene censo propio en 1895 ni
    1900)."""
    salida = []
    año_prev = pob_prev = None
    for r in df.filter(pl.col("cve_ent") == cve).sort("año").iter_rows(named=True):
        año, pob = r["año"], r["poblacion"]
        fila = {"año": año, "poblacion": pob, "fuente": r["fuente"], "años": None,
                "cambio_absoluto": None, "cambio_pct": None, "tcma_pct": None}
        if pob is not None and pob_prev is not None:
            años = año - año_prev
            fila["años"] = años
            fila["cambio_absoluto"] = pob - pob_prev
            fila["cambio_pct"] = (pob - pob_prev) / pob_prev * 100
            fila["tcma_pct"] = ((pob / pob_prev) ** (1 / años) - 1) * 100
        salida.append(fila)
        if pob is not None:
            año_prev, pob_prev = año, pob
    return pl.DataFrame(salida, schema={
        "año": pl.Int16, "poblacion": pl.Int64, "fuente": pl.Utf8, "años": pl.Int16,
        "cambio_absoluto": pl.Int64, "cambio_pct": pl.Float64, "tcma_pct": pl.Float64,
    })


def _fmt(v, decimales=0, signo=False, sufijo=""):
    if v is None:
        return "—"
    return f"{v:{'+' if signo else ''},.{decimales}f}{sufijo}"


def _paso(valor):
    """Paso en años para --proyeccion (entero positivo)."""
    try:
        n = int(valor)
    except ValueError:
        raise argparse.ArgumentTypeError(f"paso inválido: {valor!r} (usa un entero, p. ej. 5)")
    if n < 1:
        raise argparse.ArgumentTypeError(f"paso inválido: {n} (debe ser >= 1)")
    return n


def _censo(valor):
    """Año censal para --año (debe ser uno de los 14 censos generales)."""
    try:
        n = int(valor)
    except ValueError:
        raise argparse.ArgumentTypeError(f"año inválido: {valor!r}")
    if n not in CENSOS:
        raise argparse.ArgumentTypeError(f"censo no disponible: {n} (elige entre {CENSOS})")
    return n


def tabla_nacional(df, año):
    """Población, share nacional y diferencia vs. el censo anterior, para las
    32 entidades, en un año censal dado."""
    idx = CENSOS.index(año)
    año_prev = CENSOS[idx - 1] if idx > 0 else None

    actual = df.filter(pl.col("año") == año).select(["cve_ent", "entidad", "poblacion"])
    total_actual = actual["poblacion"].sum()

    if año_prev is not None:
        anterior = df.filter(pl.col("año") == año_prev).select(
            ["cve_ent", pl.col("poblacion").alias("poblacion_prev")])
        total_prev = anterior["poblacion_prev"].sum()
    else:
        anterior = pl.DataFrame(schema={"cve_ent": pl.Int16, "poblacion_prev": pl.Int64})
        total_prev = None

    tabla = (
        actual.join(anterior, on="cve_ent", how="left")
        .with_columns(
            (pl.col("poblacion") / total_actual * 100).alias("share_pct"),
            (pl.col("poblacion") - pl.col("poblacion_prev")).alias("dif_absoluta"),
            ((pl.col("poblacion") - pl.col("poblacion_prev")) / pl.col("poblacion_prev") * 100)
            .alias("dif_pct"),
        )
        .sort("poblacion", descending=True)
    )
    return tabla, año_prev, total_actual, total_prev


def _k(valor):
    """Número de clusters para --clusters (entero entre 1 y 32)."""
    try:
        n = int(valor)
    except ValueError:
        raise argparse.ArgumentTypeError(f"clusters inválido: {valor!r}")
    if not (1 <= n <= 32):
        raise argparse.ArgumentTypeError(f"clusters inválido: {n} (debe estar entre 1 y 32)")
    return n


def _centroide_anillo(coords):
    """Centroide y área (con signo) de un anillo de polígono — fórmula shoelace."""
    area = cx = cy = 0.0
    for i in range(len(coords) - 1):
        x0, y0 = coords[i]
        x1, y1 = coords[i + 1]
        cruz = x0 * y1 - x1 * y0
        area += cruz
        cx += (x0 + x1) * cruz
        cy += (y0 + y1) * cruz
    area /= 2
    return cx / (6 * area), cy / (6 * area), abs(area)


def centroides() -> dict:
    """Centroide (lon, lat) por entidad, a partir de data/mexico_states.geojson
    — área-ponderado entre las partes de un MultiPolygon, sin proyectar (grados
    simples). Suficiente para una regionalización gruesa a escala país, no para
    precisión GIS."""
    data = json.loads(GEOJSON_PATH.read_text(encoding="utf-8"))
    out = {}
    for feat in data["features"]:
        cve = normalizar_estado(feat["properties"]["name"])
        if cve is None:
            continue
        geom = feat["geometry"]
        polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
        sx = sy = sa = 0.0
        for poly in polys:
            cx, cy, area = _centroide_anillo(poly[0])  # anillo exterior únicamente
            sx += cx * area
            sy += cy * area
            sa += area
        out[cve] = (sx / sa, sy / sa)
    assert len(out) == 32, f"{len(out)} entidades resueltas de {GEOJSON_PATH.name}, se esperaban 32"
    return out


def _conectado(cves, vecinos) -> bool:
    """BFS: ¿el subconjunto de entidades forma un solo componente conexo en VECINOS?"""
    cves = set(cves)
    inicio = next(iter(cves))
    visitados = {inicio}
    pendientes = [inicio]
    while pendientes:
        actual = pendientes.pop()
        for v in vecinos[actual] & cves:
            if v not in visitados:
                visitados.add(v)
                pendientes.append(v)
    return visitados == cves


def agrupar(k) -> dict:
    """Asigna las 32 entidades a k grupos de vecinos contiguos: clustering
    jerárquico (ward) sobre el centroide geográfico de cada entidad,
    restringido a fusionar sólo entidades vecinas (VECINOS). La asignación es
    puramente geográfica — no depende del año censal, así que el mismo
    agrupamiento sirve para reportar cualquier censo (ver clusters()) o para
    seguir la trayectoria histórica de cada región (ver serie_regiones())."""
    cent = centroides()
    cves = list(range(1, 33))
    X = np.array([cent[c] for c in cves])
    adj = np.array([[1 if b in VECINOS[a] else 0 for b in cves] for a in cves])

    modelo = AgglomerativeClustering(n_clusters=k, connectivity=adj, linkage="ward")
    etiquetas = modelo.fit_predict(X)

    grupos = {}
    for cve, etiqueta in zip(cves, etiquetas):
        grupos.setdefault(int(etiqueta), []).append(cve)
    for miembros in grupos.values():
        assert _conectado(miembros, VECINOS), f"cluster desconectado: {miembros}"
    return grupos


def _pob_grupo(censo_df, miembros, año):
    """Población de un grupo de entidades en un censo. Entidades sin dato ese
    censo (p. ej. Quintana Roo en 1895) se tratan como 0, siempre que el grupo
    tenga algún integrante con dato — si ninguno tiene dato, devuelve None."""
    valores = (censo_df.filter((pl.col("año") == año) & pl.col("cve_ent").is_in(miembros))
               ["poblacion"].to_list())
    disponibles = [v for v in valores if v is not None]
    return sum(disponibles) if disponibles else None


def clusters(censo_df, año, k):
    """Agrupa las 32 entidades en k regiones de estados vecinos (ver
    agrupar()) y reporta la población de cada región en un censo dado."""
    pob = censo_df.filter(pl.col("año") == año).select(["cve_ent", "entidad", "poblacion"])
    total_nacional = pob["poblacion"].sum()
    nombre_por_cve = dict(zip(pob["cve_ent"].to_list(), pob["entidad"].to_list()))

    filas = []
    for miembros in agrupar(k).values():
        poblacion = _pob_grupo(censo_df, miembros, año)
        filas.append({
            "entidades": sorted(nombre_por_cve[c] for c in miembros),
            "poblacion": poblacion,
            "share_pct": poblacion / total_nacional * 100,
        })
    filas.sort(key=lambda f: f["poblacion"], reverse=True)
    return filas, total_nacional


def serie_regiones(censo_df, k, conapo_df=None, paso=1) -> pl.DataFrame:
    """Trayectoria histórica de población por región — mismo cálculo de
    cambio/TCMA que calcular() para una entidad, pero sumando la región
    completa. La agrupación (agrupar()) se calcula una sola vez y es la misma
    para todos los años, ya que no depende del año.

    Con conapo_df, extiende la serie más allá del último censo (2020) con la
    reconstrucción/proyección de CONAPO (combinar()), sujeta al mismo `paso`
    opcional que --proyeccion — sin conapo_df, sólo los 14 censos."""
    if conapo_df is not None:
        df = pl.concat([combinar(censo_df, conapo_df, cve, paso) for cve in range(1, 33)])
    else:
        df = censo_df
    nombre_por_cve = dict(censo_df.select(["cve_ent", "entidad"]).unique().iter_rows())
    total_por_año = dict(df.group_by("año").agg(pl.sum("poblacion")).iter_rows())
    años = sorted(df["año"].unique().to_list())
    grupos = list(agrupar(k).values())
    grupos.sort(key=lambda m: _pob_grupo(df, m, CENSOS[-1]) or 0, reverse=True)

    salida = []
    for cluster_id, miembros in enumerate(grupos, start=1):
        entidades = ", ".join(sorted(nombre_por_cve[c] for c in miembros))
        año_prev = pob_prev = None
        for año in años:
            pob = _pob_grupo(df, miembros, año)
            fila = {"cluster_id": cluster_id, "entidades": entidades, "año": año,
                    "poblacion": pob,
                    "share_pct": pob / total_por_año[año] * 100 if pob is not None else None,
                    "años": None, "cambio_absoluto": None,
                    "cambio_pct": None, "tcma_pct": None}
            if pob is not None and pob_prev is not None:
                años_dif = año - año_prev
                fila["años"] = años_dif
                fila["cambio_absoluto"] = pob - pob_prev
                fila["cambio_pct"] = (pob - pob_prev) / pob_prev * 100
                fila["tcma_pct"] = ((pob / pob_prev) ** (1 / años_dif) - 1) * 100
            salida.append(fila)
            if pob is not None:
                año_prev, pob_prev = año, pob
    return pl.DataFrame(salida, schema={
        "cluster_id": pl.Int64, "entidades": pl.Utf8, "año": pl.Int16,
        "poblacion": pl.Int64, "share_pct": pl.Float64, "años": pl.Int16,
        "cambio_absoluto": pl.Int64, "cambio_pct": pl.Float64, "tcma_pct": pl.Float64,
    })


def mapa_clusters(filas, año, k, modo="poblacion", union=False) -> Path:
    """Choropleth HTML de --clusters: colorea cada entidad según `modo` —
    "poblacion" (default): escala de calor continua por el share de su cluster
    (todas las entidades de un cluster comparten color); "cluster": un color
    categórico por cluster (más fácil de distinguir cuántos clusters hay).
    Mismo patrón que dashboard/pibe.py:fig_concentration_map(), con
    data/mexico_states.geojson.

    `union=True` (--mapa-union) quita el trazo de borde de los polígonos: como
    los estados de un mismo cluster ya comparten el mismo color de relleno,
    sin línea divisoria se perciben como una sola masa territorial — no hace
    falta una fusión geométrica real (no hay shapely/geopandas instalado).

    El geojson usa "México" (no "Estado de México", el nombre canónico de
    comun.py) para esa entidad — sin corregirlo, plotly no encuentra el
    feature y lo deja sin colorear (visualmente "ajeno" al resto del mapa)."""
    NOMBRE_GEOJSON = {"Estado de México": "México"}
    filas_por_entidad = [
        {"entidad": entidad, "entidad_geojson": NOMBRE_GEOJSON.get(entidad, entidad),
         "cluster_id": i, "poblacion": f["poblacion"], "share_pct": f["share_pct"]}
        for i, f in enumerate(filas, start=1)
        for entidad in f["entidades"]
    ]
    df = pl.DataFrame(filas_por_entidad).to_pandas()
    geo = json.loads(GEOJSON_PATH.read_text(encoding="utf-8"))

    if modo == "cluster":
        df["cluster_id"] = df["cluster_id"].astype(str)
        color_col = "cluster_id"
        color_kwargs = dict(color_discrete_sequence=px.colors.qualitative.Set3,
                            labels={"cluster_id": "Cluster"})
    else:
        color_col = "share_pct"
        color_kwargs = dict(
            color_continuous_scale=[[0, "#1E293B"], [0.5, "#2E86AB"], [1, "#F4A261"]],
            labels={"share_pct": "% del cluster"},
        )

    fig = px.choropleth_map(
        df, geojson=geo, locations="entidad_geojson", featureidkey="properties.name",
        color=color_col, hover_name="entidad", custom_data=["cluster_id", "poblacion", "share_pct"],
        map_style="carto-darkmatter", center={"lat": 23.6, "lon": -102.5}, zoom=4,
        **color_kwargs,
    )
    fig.update_traces(hovertemplate="<b>%{hovertext}</b><br>Cluster %{customdata[0]}"
                                    "<br>Población del cluster: %{customdata[1]:,}"
                                    "<br>%{customdata[2]:.1f}% del total nacional<extra></extra>")
    if union:
        fig.update_traces(marker_line_width=0)
    subtitulo = "coloreado por población" if modo == "poblacion" else "coloreado por cluster"
    fig.update_layout(
        title=dict(text=f"Clusters de estados vecinos — censo {año} ({k} clusters), {subtitulo}"),
        margin=dict(t=60, b=0, l=0, r=0), height=580,
    )

    sufijo = "mapa_union" if union else "mapa"
    ruta = DIR / f"crecimiento_clusters_{año}_{k}_{sufijo}_{modo}.html"
    fig.write_html(ruta)
    return ruta


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--entidad", type=_entidad, default=None,
                   help="Nombre o clave INEGI (1-32) de la entidad federativa. Si se omite, "
                        "se muestra la tabla nacional de las 32 entidades para --año")
    p.add_argument("--año", type=_censo, default=CENSOS[-1],
                   help="Censo a usar en la tabla nacional o --clusters (sin --entidad); "
                        f"default el último censo, {CENSOS[-1]}")
    p.add_argument("--clusters", type=_k, default=None,
                   help="Agrupa las 32 entidades en N regiones de estados vecinos (1-32) y "
                        "reporta población/share por región, para --año. Incompatible con "
                        "--entidad")
    p.add_argument("--regiones", type=_k, default=None,
                   help="Igual que --clusters, pero reporta la trayectoria histórica (los 14 "
                        "censos) de población de cada región, en vez de una foto de un año. "
                        "Admite --proyeccion para extenderla más allá de 2020. Incompatible "
                        "con --entidad")
    p.add_argument("--proyeccion", nargs="?", type=_paso, const=1, default=None,
                   help="Complementa la serie con la reconstrucción/proyección de CONAPO "
                        "(1990-2040): llena los huecos intercensales de 1990 a 2020 y agrega "
                        "los años 2021-2040. Admite un paso opcional en años (default 1 = "
                        "anual); p. ej. --proyeccion 5 sólo agrega múltiplos de 5")
    p.add_argument("--guardar", action="store_true",
                   help="Escribe el resultado a dashboard_data/crecimiento_<entidad>.parquet "
                        "(o dashboard_data/crecimiento_nacional_<año>.parquet sin --entidad)")
    p.add_argument("--mapa", nargs="?", choices=["poblacion", "cluster"], const="poblacion",
                   default=None,
                   help="Con --clusters, genera un choropleth HTML interactivo en "
                        "dashboard_data/crecimiento_clusters_<año>_<k>_mapa_<modo>.html. "
                        "--mapa poblacion (default): escala de calor por el share del cluster. "
                        "--mapa cluster: un color categórico por cluster")
    p.add_argument("--mapa-union", dest="mapa_union", nargs="?",
                   choices=["poblacion", "cluster"], const="poblacion", default=None,
                   help="Igual que --mapa, pero sin bordes entre estados del mismo cluster "
                        "(se ven como una sola masa territorial) — "
                        "dashboard_data/crecimiento_clusters_<año>_<k>_mapa_union_<modo>.html")
    a = p.parse_args()

    if (a.mapa or a.mapa_union) and a.clusters is None:
        p.error("--mapa/--mapa-union requieren --clusters")

    censo_df = leer_poblacion_historica()

    if a.regiones is not None:
        if a.entidad is not None:
            p.error("--regiones es incompatible con --entidad (es un reporte nacional)")
        conapo_df = leer_conapo() if a.proyeccion else None
        tabla = serie_regiones(censo_df, a.regiones, conapo_df, a.proyeccion or 1)

        print(f"\n{'═' * 78}")
        print(f"  Crecimiento poblacional por región — {a.regiones} clusters de estados vecinos")
        rango = "1895–2020, censos" if not a.proyeccion else (
            "1895–2040, censo + CONAPO" if a.proyeccion == 1
            else f"1895–2040, censo + CONAPO, paso {a.proyeccion} años")
        print(f"  ({rango} — agrupación geográfica fija, no depende del año)")
        print("═" * 78)
        etiqueta_col = "Año" if a.proyeccion else "Censo"
        for cluster_id in dict.fromkeys(tabla["cluster_id"].to_list()):
            sub = tabla.filter(pl.col("cluster_id") == cluster_id)
            print(f"\n  Cluster {cluster_id}: {sub['entidades'][0]}")
            print(f"  {etiqueta_col:<7} {'Población':>13} {'Share':>7} {'Δ Población':>14} {'Δ % (total)':>13}"
                  f" {'Años':>6} {'TCMA % anual':>13}")
            for r in sub.iter_rows(named=True):
                print(f"  {r['año']:<7} {_fmt(r['poblacion']):>13}"
                      f" {_fmt(r['share_pct'], 1, sufijo='%'):>7}"
                      f" {_fmt(r['cambio_absoluto'], signo=True):>14}"
                      f" {_fmt(r['cambio_pct'], 1, signo=True, sufijo='%'):>13}"
                      f" {_fmt(r['años']):>6}"
                      f" {_fmt(r['tcma_pct'], 2, signo=True, sufijo='%'):>13}")
        print()
        if a.proyeccion:
            print("Nota: en los años censales la población es la del censo (suma de la región).")
            print("CONAPO llena el resto de los años, pero su cifra difiere del censo en los años")
            print("censales (ajuste de subenumeración + población a mitad de año vs. fecha censal)")
            print("— por eso puede verse un salto en el primer año inmediato tras cada censo.\n")

        if a.guardar:
            ruta = DIR / f"crecimiento_regiones_{a.regiones}.parquet"
            tabla.write_parquet(ruta)
            print(f"Saved → {ruta}  ({tabla.height:,} filas)\n")
        return

    if a.clusters is not None:
        if a.entidad is not None:
            p.error("--clusters es incompatible con --entidad (es un reporte nacional)")
        filas, total_nacional = clusters(censo_df, a.año, a.clusters)

        print(f"\n{'═' * 68}")
        print(f"  Clusters de estados vecinos — censo {a.año} ({a.clusters} clusters)")
        print("═" * 68)
        acumulado = 0.0
        for i, f in enumerate(filas, start=1):
            acumulado += f["share_pct"]
            print(f"\n  Cluster {i}: {_fmt(f['poblacion']):>12}  "
                  f"({_fmt(f['share_pct'], 1, sufijo='%')}, {_fmt(acumulado, 1, sufijo='%')} acum.)")
            print(f"    {', '.join(f['entidades'])}")
        print(f"\n  {'TOTAL NACIONAL':<16} {_fmt(total_nacional):>12}  (100.0%)\n")

        if a.guardar:
            acumulado = 0.0
            filas_guardar = []
            for i, f in enumerate(filas, start=1):
                acumulado += f["share_pct"]
                for entidad in f["entidades"]:
                    filas_guardar.append({
                        "cluster_id": i, "entidad": entidad,
                        "cve_ent": normalizar_estado(entidad),
                        "poblacion_cluster": f["poblacion"], "share_pct": f["share_pct"],
                        "share_pct_acum": acumulado,
                    })
            salida = pl.DataFrame(filas_guardar).with_columns(pl.col("cve_ent").cast(pl.Int16))
            ruta = DIR / f"crecimiento_clusters_{a.año}_{a.clusters}.parquet"
            salida.write_parquet(ruta)
            print(f"Saved → {ruta}  ({salida.height:,} filas)\n")

        if a.mapa:
            ruta = mapa_clusters(filas, a.año, a.clusters, a.mapa)
            print(f"Saved → {ruta}\n")
        if a.mapa_union:
            ruta = mapa_clusters(filas, a.año, a.clusters, a.mapa_union, union=True)
            print(f"Saved → {ruta}\n")
        return

    if a.entidad is None:
        tabla, año_prev, total_actual, total_prev = tabla_nacional(censo_df, a.año)

        print(f"\n{'═' * 68}")
        sub = f" (vs. censo {año_prev})" if año_prev else " (primer censo, sin comparación previa)"
        print(f"  Población por entidad — censo {a.año}{sub}")
        print("═" * 68)
        print(f"  {'#':>3}  {'Entidad':<24} {'Población':>12} {'Share':>7}"
              f" {'Δ Población':>14} {'Δ %':>9}")
        for i, r in enumerate(tabla.iter_rows(named=True), start=1):
            print(f"  {i:>3}  {r['entidad']:<24} {_fmt(r['poblacion']):>12}"
                  f" {_fmt(r['share_pct'], 1, sufijo='%'):>7}"
                  f" {_fmt(r['dif_absoluta'], signo=True):>14}"
                  f" {_fmt(r['dif_pct'], 1, signo=True, sufijo='%'):>9}")
        dif_total = (total_actual - total_prev) if total_prev is not None else None
        dif_total_pct = (dif_total / total_prev * 100) if total_prev is not None else None
        print(f"  {'':>3}  {'TOTAL NACIONAL':<24} {_fmt(total_actual):>12} {'100.0%':>7}"
              f" {_fmt(dif_total, signo=True):>14} {_fmt(dif_total_pct, 1, signo=True, sufijo='%'):>9}")
        print()

        if a.guardar:
            salida = tabla.select(["cve_ent", "entidad", "poblacion", "share_pct",
                                   "dif_absoluta", "dif_pct"]).with_columns(pl.lit(a.año).alias("año"))
            ruta = DIR / f"crecimiento_nacional_{a.año}.parquet"
            salida.write_parquet(ruta)
            print(f"Saved → {ruta}  ({salida.height:,} filas)\n")
        return

    df = combinar(censo_df, leer_conapo(), a.entidad, a.proyeccion) if a.proyeccion else censo_df
    tabla = calcular(df, a.entidad)

    print(f"\n{'═' * 68}")
    if a.proyeccion:
        rango = ("1895–2040 (censo + CONAPO)" if a.proyeccion == 1
                 else f"1895–2040 (censo + CONAPO, paso {a.proyeccion} años)")
    else:
        rango = "1895–2020"
    print(f"  {NOMBRE[a.entidad]} — crecimiento poblacional, {rango}")
    print("═" * 68)
    etiqueta_col = "Año" if a.proyeccion else "Censo"
    print(f"  {etiqueta_col:<7} {'Población':>12} {'Δ Población':>14} {'Δ % (total)':>13}"
          f" {'Años':>6} {'TCMA % anual':>13}" + ("  Fuente" if a.proyeccion else ""))
    for r in tabla.iter_rows(named=True):
        linea = (f"  {r['año']:<7} {_fmt(r['poblacion']):>12}"
                 f" {_fmt(r['cambio_absoluto'], signo=True):>14}"
                 f" {_fmt(r['cambio_pct'], 1, signo=True, sufijo='%'):>13}"
                 f" {_fmt(r['años']):>6}"
                 f" {_fmt(r['tcma_pct'], 2, signo=True, sufijo='%'):>13}")
        if a.proyeccion:
            linea += f"  {r['fuente']}"
        print(linea)
    print()

    if a.proyeccion:
        print("Nota: en los años censales la población es la del censo (fuente 'censo').")
        print("CONAPO llena el resto de los años, pero su cifra difiere del censo en los años")
        print("censales (ajuste de subenumeración + población a mitad de año vs. fecha censal)")
        print("— por eso puede verse un salto en el primer año inmediato tras cada censo.\n")

    if a.guardar:
        cols = ["cve_ent", "entidad", "año", "poblacion", "años",
                "cambio_absoluto", "cambio_pct", "tcma_pct"]
        if a.proyeccion:
            cols.append("fuente")
        salida = tabla.with_columns(
            pl.lit(a.entidad).cast(pl.Int16).alias("cve_ent"),
            pl.lit(NOMBRE[a.entidad]).alias("entidad"),
        ).select(cols)
        ruta = DIR / f"crecimiento_{NOMBRE[a.entidad].lower().replace(' ', '_')}.parquet"
        salida.write_parquet(ruta)
        print(f"Saved → {ruta}  ({salida.height:,} filas)\n")


if __name__ == "__main__":
    main()
