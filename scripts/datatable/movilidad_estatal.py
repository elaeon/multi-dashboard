"""
Movilidad y migración interestatal — dos subcomandos:

  nacimiento  Migración interestatal ACUMULADA por lugar de nacimiento, censos 1950–2020.
              Mide el stock de toda la vida: cuántos residentes de una entidad nacieron en
              otra, y cuántos nacidos en ella residen hoy en otra.
  movilidad   Movilidad DECLARADA del quinquenio previo (residencia actual vs. hace 5 años),
              censos 1990–2020.

Ambos miden cosas distintas y no son comparables entre sí — en 2020, 3.8 M de personas
cambiaron de entidad en 5 años (movilidad), pero 21.6 M residen en una entidad distinta a
la de su nacimiento (nacimiento).

── nacimiento ────────────────────────────────────────────────────────────────
Fuente: dashboard_data/ccpv_nacimiento_estatal.parquet
        (generada por scripts/prepare_ccpv_nacimiento.py)

--convergencia agrupa las 32 entidades por similitud de PERFIL migratorio en
vez de reportar una sola entidad: estados de origen que mandan a sus
emigrantes preferentemente a los mismos destinos ("emigración"), y entidades
de destino que reciben inmigrantes de las mismas fuentes ("inmigración"). Se
agrupa por preferencia (% del flujo de cada estado), no por volumen — así un
estado chico y uno grande con el mismo patrón de destino caen en el mismo
grupo. El número de grupos K se fija con --k, o se autodetecta (silhouette
score) si se omite. Por default cubre los 4 censos 1950-1980.

--inmigracion/--emigracion no requieren --map: filtran el reporte a mostrar
sólo esa dirección. Con --entidad, filtran cuál de las dos tablas (llegadas/
salidas) se imprime; sin --entidad, imprimen el ranking nacional de
entidades receptoras/emisoras (mismos datos que --convergencia, agregados
por entidad en vez de por perfil).

--map genera un diagrama de flujo HTML (dashboard_data/), siempre para un
censo suelto. --map geo (default) dibuja arcos sobre el mapa real de México
(mismo estilo que --mapa de crecimiento_poblacional.py); --map sankey dibuja
un diagrama de bandas — ambos requieren --entidad y exactamente uno de
--inmigracion (orígenes que le mandan migrantes) o --emigracion (destinos
hacia donde emigra). --map chord también acepta --entidad + dirección (la
misma entidad y sus --top contrapartes, dibujadas como cuerdas en círculo en
vez de bandas), pero a diferencia de geo/sankey --entidad es OPCIONAL: si se
omite, genera un chord diagram NACIONAL con los --top flujos más grandes
entre cualquier par de entidades.

── movilidad ─────────────────────────────────────────────────────────────────
Reporta, según el censo/encuesta elegido (1990, 2000, 2005, 2010, 2015 o 2020):
  · Inmigrantes — quiénes llegaron desde otra entidad en el quinquenio previo.
  · Emigrantes  — quiénes salieron hacia otra entidad en el mismo quinquenio.
  · Saldo neto migratorio interestatal y tasa por mil habitantes.
  · Extranjeros — flujo (residían en otro país hace 5 años, con desglose por país
    en 2020) y stock (nacidos en el extranjero).

2005 (II Conteo) y 2015 (Encuesta Intercensal) rellenan parcialmente los huecos
entre los 4 censos "completos". 2005 tiene la misma matriz origen-destino que los
censos; 2015 es una encuesta MUESTRAL cuya fuente pública sólo trae agregados
(población/inmigrantes/emigrantes/saldo) sin desglose por estado de origen ni
por país — para ese año, las tablas de top-estados y la sección de extranjeros
se muestran como "no disponible en esta fuente".

La movilidad es la declarada en el censo (residencia actual vs. residencia 5 años
antes), no una estimación indirecta como la de scripts/prepare_migracion_interna.py.

Fuente: dashboard_data/ccpv_migracion_estatal.parquet y
        dashboard_data/ccpv_extranjeros_pais_2020.parquet
        (generadas por scripts/prepare_ccpv_migracion.py)

--convergencia agrupa las 32 entidades por similitud de PERFIL de movilidad
del quinquenio (no por volumen): estados de origen que mandan a sus
emigrantes preferentemente a los mismos destinos ("emigración"), y entidades
de destino que reciben inmigrantes de las mismas fuentes ("inmigración"). K
se fija con --k, o se autodetecta (silhouette score) si se omite. 2015
(Encuesta Intercensal) no trae matriz origen-destino y se excluye del
default; --año 2015 con --convergencia se rechaza explícitamente.

--inmigracion/--emigracion no requieren --map: filtran el reporte a mostrar
sólo esa dirección. Con --entidad, filtran cuál de las dos tablas (llegadas/
salidas) se imprime; sin --entidad, imprimen el ranking nacional de
entidades receptoras/emisoras del quinquenio (mismos datos que
--convergencia, agregados por entidad en vez de por perfil).

--map genera un diagrama de flujo HTML (dashboard_data/), siempre para un
censo suelto (2015 excluido, sin matriz origen-destino). --map geo (default)
dibuja arcos sobre el mapa real de México; --map sankey dibuja un diagrama
de bandas — ambos requieren --entidad y exactamente uno de --inmigracion
(orígenes que le mandan gente en el quinquenio) o --emigracion (destinos
hacia donde se fue). --map chord también acepta --entidad + dirección (la
misma entidad y sus --top contrapartes, dibujadas como cuerdas en círculo en
vez de bandas), pero a diferencia de geo/sankey --entidad es OPCIONAL: si se
omite, genera un chord diagram NACIONAL con los --top flujos más grandes
entre cualquier par de entidades.

Run: uv run python scripts/datatable/movilidad_estatal.py nacimiento --entidad Jalisco --año 1970
     uv run python scripts/datatable/movilidad_estatal.py nacimiento --entidad cdmx --serie
     uv run python scripts/datatable/movilidad_estatal.py nacimiento --entidad Jalisco --año 1950-1980
     uv run python scripts/datatable/movilidad_estatal.py nacimiento --convergencia
     uv run python scripts/datatable/movilidad_estatal.py nacimiento --convergencia --año 1960 --k 5
     uv run python scripts/datatable/movilidad_estatal.py nacimiento --entidad Jalisco --año 1970 --map --inmigracion
     uv run python scripts/datatable/movilidad_estatal.py nacimiento --entidad Jalisco --año 1970 --map sankey --emigracion
     uv run python scripts/datatable/movilidad_estatal.py nacimiento --entidad Jalisco --año 1970 --map chord --emigracion
     uv run python scripts/datatable/movilidad_estatal.py nacimiento --año 1970 --map chord
     uv run python scripts/datatable/movilidad_estatal.py nacimiento --entidad Jalisco --año 1970 --inmigracion
     uv run python scripts/datatable/movilidad_estatal.py nacimiento --año 1970 --emigracion
     uv run python scripts/datatable/movilidad_estatal.py movilidad --entidad Jalisco --año 2020
     uv run python scripts/datatable/movilidad_estatal.py movilidad --entidad Jalisco --año 1990-2020
     uv run python scripts/datatable/movilidad_estatal.py movilidad --convergencia
     uv run python scripts/datatable/movilidad_estatal.py movilidad --convergencia --año 2020 --k 5
     uv run python scripts/datatable/movilidad_estatal.py movilidad --entidad Jalisco --año 2020 --map --inmigracion
     uv run python scripts/datatable/movilidad_estatal.py movilidad --entidad Jalisco --año 2020 --map sankey --emigracion
     uv run python scripts/datatable/movilidad_estatal.py movilidad --entidad Jalisco --año 2020 --map chord --emigracion
     uv run python scripts/datatable/movilidad_estatal.py movilidad --año 2020 --map chord
     uv run python scripts/datatable/movilidad_estatal.py movilidad --entidad Jalisco --año 2020 --inmigracion
     uv run python scripts/datatable/movilidad_estatal.py movilidad --año 2020 --emigracion
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
import plotly.graph_objects as go
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(RAIZ / "scripts"))
sys.path.insert(0, str(RAIZ / "scripts" / "centralismo"))
from comun import NOMBRE, normalizar_estado
from prepare_ccpv_nacimiento import TERRITORIOS
from prepare_ccpv_migracion import EMIGRANTES_AGREGADO
from crecimiento_poblacional import centroides

DIR = RAIZ / "dashboard_data"

CENSOS_NACIMIENTO = [1950, 1960, 1970, 1980, 1990, 2000, 2010, 2020]
CENSOS_CONVERGENCIA_NACIMIENTO = (1950, 1960, 1970, 1980)

CENSOS_MOVILIDAD = [1990, 2000, 2005, 2010, 2015, 2020]
CENSOS_CONVERGENCIA_MOVILIDAD = tuple(c for c in CENSOS_MOVILIDAD if c != 2015)
VENTANA = {1990: "1985 → 1990", 2000: "enero 1995 → 2000",
           2005: "octubre 2000 → octubre 2005", 2010: "junio 2005 → 2010",
           2015: "marzo 2010 → marzo 2015", 2020: "marzo 2015 → marzo 2020"}
EXTRANJERO = ["En los Estados Unidos de América", "En otro país"]
SIN_DESGLOSE = "(sin desglose por estado en esta fuente)"


# ── Compartido: validadores CLI ────────────────────────────────────────────

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


def _hacer_año_o_rango(censos):
    """Fábrica de validador --año: un censo suelto ('2010') o un rango
    inclusivo ('1990-2020') → int o tupla, validado contra la lista `censos`
    del subcomando correspondiente."""
    def _validador(valor):
        if "-" in valor:
            partes = valor.split("-")
            if len(partes) != 2:
                raise argparse.ArgumentTypeError(f"rango inválido: {valor!r} (usa 'AAAA-AAAA')")
            try:
                lo, hi = sorted(int(p) for p in partes)
            except ValueError:
                raise argparse.ArgumentTypeError(f"rango inválido: {valor!r} (usa 'AAAA-AAAA')")
            rango = tuple(c for c in censos if lo <= c <= hi)
            if not rango:
                raise argparse.ArgumentTypeError(
                    f"ningún censo cae en el rango {lo}-{hi} (censos disponibles: {censos})")
            return rango
        try:
            censo = int(valor)
        except ValueError:
            raise argparse.ArgumentTypeError(f"año inválido: {valor!r}")
        if censo not in censos:
            raise argparse.ArgumentTypeError(f"censo no disponible: {censo} (elige entre {censos})")
        return censo
    return _validador


_año_o_rango_nacimiento = _hacer_año_o_rango(CENSOS_NACIMIENTO)
_año_o_rango_movilidad = _hacer_año_o_rango(CENSOS_MOVILIDAD)


def _k(valor):
    try:
        n = int(valor)
    except ValueError:
        raise argparse.ArgumentTypeError(f"k inválido: {valor!r} (usa un entero, p. ej. 5)")
    if n < 2:
        raise argparse.ArgumentTypeError(f"k inválido: {n} (debe ser >= 2)")
    return n


def _tabla(titulo, df, col, total, top, poblacion=None):
    """Imprime top-N filas de (col, personas) con su % y % acumulado del total.
    Si se pasa `poblacion` (población total de la entidad), agrega qué % de
    ella representa `total`."""
    encabezado = f"\n{titulo}: {total:,}"
    if poblacion:
        encabezado += f"  ({total / poblacion * 100:.2f}% de la población de la entidad)"
    print(encabezado)
    if total == 0:
        return
    print(f"  {'#':>3}  {'':<34} {'Personas':>12} {'%':>7} {'% acum.':>8}")
    acumulado = 0
    for i, r in enumerate(df.head(top).iter_rows(named=True), start=1):
        acumulado += r["personas"]
        print(f"  {i:>3}  {r[col]:<34} {r['personas']:>12,} {r['personas']/total*100:>6.1f}%"
              f" {acumulado/total*100:>7.1f}%")
    resto = df.slice(top)
    if resto.height:
        suma = resto["personas"].sum()
        acumulado += suma
        print(f"  {'':>3}  {f'Resto ({resto.height})':<34} {suma:>12,} {suma/total*100:>6.1f}%"
              f" {acumulado/total*100:>7.1f}%")


# ── Compartido: diagrama de flujo HTML (arcos geo / Sankey de 3 niveles) ───

def mapa_flujo_geo(cve, censo, direccion, tabla, top, prefijo, ventana=None) -> Path:
    """Arcos sobre el mapa de México entre el centroide de `cve` y los
    centroides de sus contrapartes (orígenes si direccion='inmigracion',
    destinos si 'emigracion'), grosor de línea proporcional a `personas`.
    Mismo estilo (carto-darkmatter, centroides de crecimiento_poblacional.py)
    que --mapa de ese script. `prefijo` fija el nombre del archivo
    (migracion_flujo_* / movilidad_flujo_*); `ventana` agrega el rango de
    fechas del quinquenio al título (sólo lo usa el subcomando movilidad)."""
    cent = centroides()
    lon0, lat0 = cent[cve]
    filas = tabla.head(top)
    max_personas = filas["personas"].max()

    fig = go.Figure()
    for cve_c, personas in filas.select(["cve_contraparte", "personas"]).iter_rows():
        lon1, lat1 = cent[cve_c]
        fig.add_trace(go.Scattermap(
            lon=[lon0, lon1], lat=[lat0, lat1], mode="lines",
            line=dict(width=1 + 6 * personas / max_personas, color="#F4A261"),
            opacity=0.6, hoverinfo="skip", showlegend=False,
        ))

    contrapartes = filas["cve_contraparte"].to_list()
    fig.add_trace(go.Scattermap(
        lon=[cent[c][0] for c in contrapartes], lat=[cent[c][1] for c in contrapartes],
        mode="markers", marker=dict(size=9, color="#2E86AB"),
        text=[NOMBRE[c] for c in contrapartes], customdata=filas["personas"].to_list(),
        hovertemplate="<b>%{text}</b><br>%{customdata:,} personas<extra></extra>",
        showlegend=False,
    ))
    fig.add_trace(go.Scattermap(
        lon=[lon0], lat=[lat0], mode="markers", marker=dict(size=14, color="#F4A261"),
        text=[NOMBRE[cve]], hovertemplate="<b>%{text}</b><extra></extra>", showlegend=False,
    ))

    titulo = "Inmigración hacia" if direccion == "inmigracion" else "Emigración desde"
    cabecera = (f"Censo {censo} ({ventana}) — top {filas.height} de {tabla.height}"
                if ventana else f"Censo {censo} (top {filas.height} de {tabla.height})")
    fig.update_layout(
        title=dict(text=f"{titulo} {NOMBRE[cve]} — {cabecera}"),
        map=dict(style="carto-darkmatter", center=dict(lat=lat0, lon=lon0), zoom=4),
        margin=dict(t=60, b=0, l=0, r=0), height=580,
        paper_bgcolor="rgba(0,0,0,0)", font_color="#CBD5E1",
    )
    sufijo = NOMBRE[cve].lower().replace(" ", "_")
    ruta = DIR / f"{prefijo}_flujo_{sufijo}_{censo}_{direccion}_geo.html"
    fig.write_html(ruta)
    return ruta


def mapa_flujo_sankey(cve, censo, direccion, tabla, top, prefijo, ventana=None) -> Path:
    """Diagrama de bandas (Sankey) origen→destino para una sola entidad, con
    las `top` contrapartes de mayor volumen. Mismo estilo base (nodo azul,
    banda semitransparente) que dashboard/natalidad_defunciones.py:fig_sankey().
    `prefijo`/`ventana` como en mapa_flujo_geo."""
    filas = tabla.head(top)
    contrapartes = filas["cve_contraparte"].to_list()
    personas = filas["personas"].to_list()

    if direccion == "inmigracion":
        labels = [NOMBRE[c] for c in contrapartes] + [NOMBRE[cve]]
        source = list(range(len(contrapartes)))
        target = [len(contrapartes)] * len(contrapartes)
    else:
        labels = [NOMBRE[cve]] + [NOMBRE[c] for c in contrapartes]
        source = [0] * len(contrapartes)
        target = list(range(1, len(contrapartes) + 1))

    fig = go.Figure(go.Sankey(
        node=dict(pad=10, thickness=14, label=labels, color="#2E86AB",
                  hovertemplate="<b>%{label}</b><extra></extra>"),
        link=dict(source=source, target=target, value=personas,
                  color="rgba(46,134,171,0.35)",
                  hovertemplate="<b>%{source.label} → %{target.label}</b><br>"
                                "%{value:,} personas<extra></extra>"),
    ))
    titulo = "Inmigración hacia" if direccion == "inmigracion" else "Emigración desde"
    cabecera = (f"Censo {censo} ({ventana}) — top {filas.height} de {tabla.height}"
                if ventana else f"Censo {censo} (top {filas.height} de {tabla.height})")
    fig.update_layout(
        title=dict(text=f"{titulo} {NOMBRE[cve]} — {cabecera}"),
        height=max(400, len(labels) * 22 + 100),
        paper_bgcolor="rgba(0,0,0,0)", font_color="#CBD5E1",
        margin=dict(t=65, b=10, l=10, r=10),
    )
    sufijo = NOMBRE[cve].lower().replace(" ", "_")
    ruta = DIR / f"{prefijo}_flujo_{sufijo}_{censo}_{direccion}_sankey.html"
    fig.write_html(ruta)
    return ruta


def _click_relaciones_js(pares_od, node_idx, opacidad_base) -> str:
    """JS (post_script de fig.write_html) para los chord diagrams: click en un
    nodo atenúa las cuerdas y los demás nodos que no se relacionan con él
    (resaltados en foco); un segundo click sobre el mismo nodo restaura la
    vista completa. `pares_od` es la lista [[origen, destino], ...] en el
    mismo orden en que se agregaron las cuerdas (2 trazas por cuerda: línea +
    marcador de hover), `node_idx` el índice de la traza de nodos."""
    plantilla = """
(function() {
    var gd = document.getElementById('{plot_id}');
    var od = __OD__;
    var nodeIdx = __NODE_IDX__;
    var opacidadBase = __OP_BASE__;
    var foco = null;
    gd.on('plotly_click', function(ev) {
        var pt = ev.points && ev.points[0];
        if (!pt || pt.curveNumber !== nodeIdx) return;
        var cve = pt.customdata[0];
        var n = od.length;
        var lineIdx = [], lineOp = [];
        var nodeCustom = gd.data[nodeIdx].customdata;
        var nodeOp = [];
        if (foco === cve) {
            for (var i = 0; i < n; i++) { lineIdx.push(2 * i); lineOp.push(opacidadBase); }
            for (var j = 0; j < nodeCustom.length; j++) { nodeOp.push(1); }
            foco = null;
        } else {
            var relacionados = {};
            relacionados[cve] = true;
            for (var i = 0; i < n; i++) {
                var toca = od[i][0] === cve || od[i][1] === cve;
                lineIdx.push(2 * i);
                lineOp.push(toca ? 0.9 : 0.04);
                if (toca) { relacionados[od[i][0]] = true; relacionados[od[i][1]] = true; }
            }
            for (var j = 0; j < nodeCustom.length; j++) {
                nodeOp.push(relacionados[nodeCustom[j][0]] ? 1 : 0.12);
            }
            foco = cve;
        }
        Plotly.restyle(gd, {opacity: lineOp}, lineIdx);
        Plotly.restyle(gd, {'marker.opacity': [nodeOp]}, [nodeIdx]);
    });
})();
"""
    return (plantilla
            .replace("__OD__", json.dumps(pares_od))
            .replace("__NODE_IDX__", str(node_idx))
            .replace("__OP_BASE__", str(opacidad_base)))


def mapa_flujo_chord_entidad(cve, censo, direccion, tabla, top, prefijo, ventana=None) -> Path:
    """Chord diagram para una sola entidad: mismos datos que mapa_flujo_sankey
    (nodo `cve` + sus `top` contrapartes de mayor volumen), pero dibujado como
    cuerdas Bezier cuadráticas en círculo en vez de bandas Sankey. Grosor de
    cuerda proporcional a `personas`, hover con el par origen→destino. Click
    en un nodo aísla sus relaciones (atenúa el resto); un segundo click
    restaura la vista completa. `prefijo`/`ventana` como en mapa_flujo_geo."""
    filas = tabla.head(top)
    contrapartes = filas["cve_contraparte"].to_list()
    personas = filas["personas"].to_list()

    nodos = [cve] + contrapartes
    n = len(nodos)
    angulos = {c: -np.pi / 2 + 2 * np.pi * i / n for i, c in enumerate(nodos)}
    pos = {c: (np.cos(a), np.sin(a)) for c, a in angulos.items()}
    max_personas = max(personas)

    fig = go.Figure()
    pares_od = []
    for c_contra, p_val in zip(contrapartes, personas):
        origen, destino = (c_contra, cve) if direccion == "inmigracion" else (cve, c_contra)
        pares_od.append([origen, destino])
        p0, p1 = pos[origen], pos[destino]
        cx, cy = (p0[0] + p1[0]) * 0.25, (p0[1] + p1[1]) * 0.25
        t = np.linspace(0, 1, 40)
        x = (1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * cx + t ** 2 * p1[0]
        y = (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * cy + t ** 2 * p1[1]
        fig.add_trace(go.Scatter(
            x=x, y=y, mode="lines",
            line=dict(width=1.5 + 16 * p_val / max_personas, color="#2E86AB"),
            opacity=0.5, hoverinfo="skip", showlegend=False,
        ))
        m = len(t) // 2
        fig.add_trace(go.Scatter(
            x=[x[m]], y=[y[m]], mode="markers",
            marker=dict(size=12, color="#2E86AB", opacity=0),
            hovertemplate=f"<b>{NOMBRE[origen]} → {NOMBRE[destino]}</b><br>"
                          f"{p_val:,.0f} personas<extra></extra>",
            showlegend=False,
        ))

    fig.add_trace(go.Scatter(
        x=[pos[c][0] for c in nodos], y=[pos[c][1] for c in nodos],
        mode="markers",
        marker=dict(size=[26] + [10 + 16 * p / max_personas for p in personas],
                    color=["#F4A261"] + ["#2E86AB"] * len(contrapartes),
                    line=dict(color="#0F172A", width=1)),
        customdata=[[c, NOMBRE[c]] for c in nodos],
        hovertemplate="<b>%{customdata[1]}</b><extra></extra>",
        showlegend=False,
    ))
    node_idx = 2 * len(contrapartes)

    r_label = 1.18
    for c in nodos:
        x0, y0 = pos[c]
        fig.add_annotation(
            x=x0 * r_label, y=y0 * r_label, text=NOMBRE[c], showarrow=False,
            font=dict(size=10, color="#CBD5E1"),
            xanchor="left" if x0 >= 0 else "right", yanchor="middle",
        )

    titulo = "Inmigración hacia" if direccion == "inmigracion" else "Emigración desde"
    cabecera = (f"Censo {censo} ({ventana}) — top {filas.height} de {tabla.height}"
                if ventana else f"Censo {censo} (top {filas.height} de {tabla.height})")
    fig.update_layout(
        title=dict(text=f"{titulo} {NOMBRE[cve]} — {cabecera}"),
        xaxis=dict(visible=False, range=[-1.5, 1.5]),
        yaxis=dict(visible=False, range=[-1.5, 1.5], scaleanchor="x"),
        height=700, width=700,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#CBD5E1", margin=dict(t=65, b=10, l=10, r=10),
    )
    sufijo = NOMBRE[cve].lower().replace(" ", "_")
    ruta = DIR / f"{prefijo}_flujo_{sufijo}_{censo}_{direccion}_chord.html"
    fig.write_html(ruta, post_script=_click_relaciones_js(pares_od, node_idx, 0.5))
    return ruta


def mapa_flujo_chord(cves, matriz, censo, top, prefijo, ventana=None) -> Path:
    """Chord diagram NACIONAL: TODAS las entidades de la matriz origen-destino
    (misma que usa --convergencia) como nodos en círculo — tamaño del
    marcador proporcional a su volumen total (entrante + saliente), color
    naranja si participa en alguno de los `top` flujos más grandes dibujados,
    gris si no. Las cuerdas (Bezier cuadrática hacia el centro) tienen
    grosor proporcional a `personas` y hover con el par origen→destino. Click
    en un nodo aísla sus relaciones (atenúa el resto); un segundo click
    restaura la vista completa. Se usa con --map chord cuando no se pasa
    --entidad. `prefijo`/`ventana` como en mapa_flujo_geo."""
    idx = {c: i for i, c in enumerate(cves)}
    total_flujo = {c: float(matriz[idx[c], :].sum() + matriz[:, idx[c]].sum()) for c in cves}
    nodos = sorted(cves, key=lambda c: total_flujo[c], reverse=True)
    n = len(nodos)
    angulos = {c: -np.pi / 2 + 2 * np.pi * i / n for i, c in enumerate(nodos)}
    pos = {c: (np.cos(a), np.sin(a)) for c, a in angulos.items()}

    flujos = [(cves[i], cves[j], float(matriz[i, j]))
              for i in range(len(cves)) for j in range(len(cves))
              if i != j and matriz[i, j] > 0]
    flujos.sort(key=lambda f: f[2], reverse=True)
    flujos = flujos[:top]
    max_personas = max(v for _, _, v in flujos)
    max_total = max(total_flujo.values())

    fig = go.Figure()
    for origen, destino, personas in flujos:
        p0, p1 = pos[origen], pos[destino]
        cx, cy = (p0[0] + p1[0]) * 0.25, (p0[1] + p1[1]) * 0.25
        t = np.linspace(0, 1, 40)
        x = (1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * cx + t ** 2 * p1[0]
        y = (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * cy + t ** 2 * p1[1]
        fig.add_trace(go.Scatter(
            x=x, y=y, mode="lines",
            line=dict(width=1.5 + 16 * personas / max_personas, color="#2E86AB"),
            opacity=0.45, hoverinfo="skip", showlegend=False,
        ))
        # Marcador invisible en el punto medio, sólo para el hover del flujo.
        m = len(t) // 2
        fig.add_trace(go.Scatter(
            x=[x[m]], y=[y[m]], mode="markers",
            marker=dict(size=12, color="#2E86AB", opacity=0),
            hovertemplate=f"<b>{NOMBRE[origen]} → {NOMBRE[destino]}</b><br>"
                          f"{personas:,.0f} personas<extra></extra>",
            showlegend=False,
        ))

    tocados = {c for o, d, _ in flujos for c in (o, d)}
    fig.add_trace(go.Scatter(
        x=[pos[c][0] for c in nodos], y=[pos[c][1] for c in nodos],
        mode="markers",
        marker=dict(size=[8 + 22 * total_flujo[c] / max_total for c in nodos],
                    color=["#F4A261" if c in tocados else "#475569" for c in nodos],
                    line=dict(color="#0F172A", width=1)),
        customdata=[[c, NOMBRE[c], total_flujo[c]] for c in nodos],
        hovertemplate="<b>%{customdata[1]}</b><br>Flujo total: %{customdata[2]:,.0f}<extra></extra>",
        showlegend=False,
    ))
    node_idx = 2 * len(flujos)

    r_label = 1.18
    for c in nodos:
        x0, y0 = pos[c]
        fig.add_annotation(
            x=x0 * r_label, y=y0 * r_label, text=NOMBRE[c], showarrow=False,
            font=dict(size=9, color="#CBD5E1"),
            xanchor="left" if x0 >= 0 else "right", yanchor="middle",
        )

    cabecera = f"Censo {censo} ({ventana})" if ventana else f"Censo {censo}"
    fig.update_layout(
        title=dict(text=f"Top {len(flujos)} flujos migratorios entre {n} entidades — {cabecera} "
                        f"(tamaño del nodo = volumen total)"),
        xaxis=dict(visible=False, range=[-1.6, 1.6]),
        yaxis=dict(visible=False, range=[-1.6, 1.6], scaleanchor="x"),
        height=800, width=800,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#CBD5E1", margin=dict(t=60, b=10, l=10, r=10),
    )
    pares_od = [[o, d] for o, d, _ in flujos]
    ruta = DIR / f"{prefijo}_flujo_nacional_{censo}_chord.html"
    fig.write_html(ruta, post_script=_click_relaciones_js(pares_od, node_idx, 0.45))
    return ruta


def _tabla_flujo_nacional(cves, matriz, direccion, top):
    """Ranking nacional de entidades por volumen total de inmigración (suma
    de columna: todo lo que entra) o emigración (suma de fila: todo lo que
    sale), a partir de la misma matriz origen-destino que usa --convergencia
    y mapa_flujo_chord. Se imprime con --map chord sin --entidad cuando se
    pasa --inmigracion/--emigracion."""
    personas = matriz.sum(axis=0) if direccion == "inmigracion" else matriz.sum(axis=1)
    df = (pl.DataFrame({"entidad": [NOMBRE[c] for c in cves], "personas": personas.astype(int)})
          .sort("personas", descending=True))
    titulo = ("INMIGRACIÓN NACIONAL — entidades que más reciben" if direccion == "inmigracion"
              else "EMIGRACIÓN NACIONAL — entidades que más envían")
    _tabla(titulo, df, "entidad", int(matriz.sum()), top)


# ── Compartido: matemática de clustering por perfil de preferencia ─────────

def _perfiles(matriz, eje):
    """Normaliza cada fila (eje='origen': % de los emigrantes de ese estado
    que va a cada destino) o cada columna (eje='destino': % de los
    inmigrantes de ese estado que viene de cada origen) para que sumen 1 —
    perfil de PREFERENCIA, no de volumen."""
    m = matriz if eje == "origen" else matriz.T
    sumas = m.sum(axis=1, keepdims=True)
    sumas[sumas == 0] = 1
    return m / sumas


def _clusterizar(perfiles, k):
    """Agrupa filas de `perfiles` por similitud coseno (distancia = 1 -
    similitud). Con k dado, clustering jerárquico manual; con k=None, prueba
    k=2..10 y elige el que maximiza el silhouette score (automático)."""
    dist = np.clip(1 - cosine_similarity(perfiles), 0, None)
    np.fill_diagonal(dist, 0)
    if k is not None:
        etiquetas = AgglomerativeClustering(
            n_clusters=k, metric="precomputed", linkage="average").fit_predict(dist)
        return etiquetas, k, silhouette_score(dist, etiquetas, metric="precomputed")

    mejor = None
    for candidato in range(2, min(10, len(perfiles) - 1) + 1):
        etiquetas = AgglomerativeClustering(
            n_clusters=candidato, metric="precomputed", linkage="average").fit_predict(dist)
        score = silhouette_score(dist, etiquetas, metric="precomputed")
        if mejor is None or score > mejor[2]:
            mejor = (etiquetas, candidato, score)
    return mejor


# ════════════════════════════════════════════════════════════════════════════
# Subcomando: nacimiento
# ════════════════════════════════════════════════════════════════════════════

def cargar_nacimiento():
    ruta = DIR / "ccpv_nacimiento_estatal.parquet"
    if not ruta.exists():
        raise SystemExit(f"Falta {ruta}.\n"
                         "Corre primero: uv run python scripts/prepare_ccpv_nacimiento.py")
    return pl.read_parquet(ruta)


def flujos_nacimiento(df, cve, censo):
    """(inmigrantes, emigrantes, población total) de una entidad en un censo."""
    g = df.filter(pl.col("censo") == censo)
    desglose = g.filter(~pl.col("total_categoria"))
    inm = (desglose.filter(pl.col("cve_destino") == cve)
           .select(["origen", "personas"]).sort("personas", descending=True))
    emi = (desglose.filter((pl.col("cve_origen") == cve) & (pl.col("cve_destino") > 0))
           .select(["destino", "personas"]).sort("personas", descending=True))
    poblacion = g.filter(pl.col("total_categoria") & (pl.col("cve_destino") == cve)
                         & (pl.col("categoria") == "Total"))["personas"].sum()
    return inm, emi, poblacion


def detalle_nacimiento(df, cve, censo, top, direccion=None):
    """`direccion=None` imprime ambas tablas (INMIGRANTES y EMIGRANTES);
    'inmigracion'/'emigracion' imprime sólo la correspondiente."""
    inm, emi, poblacion = flujos_nacimiento(df, cve, censo)
    n_inm, n_emi = inm["personas"].sum(), emi["personas"].sum()

    print(f"\n{'═' * 72}")
    print(f"  {NOMBRE[cve]} — Censo {censo}")
    print("  Migración interestatal acumulada, por lugar de nacimiento")
    print(f"  Población total: {poblacion:,}")
    if cve in TERRITORIOS.get(censo, ()):
        print("  Nota: en este censo aún era territorio federal, no estado.")
    print("═" * 72)

    if direccion != "emigracion":
        _tabla("INMIGRANTES — residen aquí pero nacieron en otra entidad",
               inm, "origen", n_inm, top, poblacion)
    if direccion != "inmigracion":
        _tabla("EMIGRANTES — nacieron aquí pero residen en otra entidad",
               emi, "destino", n_emi, top, poblacion)

    neto = n_inm - n_emi
    print(f"\nSALDO NETO ACUMULADO: {neto:+,}"
          f"  ({neto / poblacion * 1000:+.2f} por mil habitantes)")

    print(f"\n{'─' * 72}")
    otras = (df.filter((pl.col("censo") == censo) & (pl.col("cve_destino") == cve)
                       & pl.col("total_categoria")
                       & ~pl.col("categoria").is_in(["Total", "En la entidad", "En otra entidad"]))
             .sort("personas", descending=True))
    for r in otras.iter_rows(named=True):
        print(f"  {r['categoria']:<44} {r['personas']:>12,}")
    print()
    return inm, emi


def acumulado_nacimiento(df, cve, censos, top, direccion=None):
    """Reporte de un rango de censos, usando el CENSO DESTINO (el más reciente
    del rango) como cifra reportada — es un stock, ya consistente con la
    población real de ese año, así que sumarlo con censos anteriores duplicaría
    personas que nunca cambiaron de entidad. Los censos anteriores del rango se
    muestran como trayectoria de contexto, sin sumarse. `direccion` como en
    detalle_nacimiento.
    """
    censo_destino = censos[-1]
    print(f"\n{'═' * 72}")
    print(f"  Rango solicitado: {censos[0]}–{censos[-1]} → usando censo destino {censo_destino}")
    print("  (el stock de migración por nacimiento es acumulado a cada fecha censal;")
    print("   sumar varios censos duplicaría personas. Se reporta el más reciente del")
    print("   rango, ya consistente con la población real de ese año.)")

    inm, emi = detalle_nacimiento(df, cve, censo_destino, top, direccion)

    if len(censos) > 1:
        print(f"{'─' * 72}")
        print("Trayectoria dentro del rango (censos anteriores, no se suman):")
        print(f"  {'Censo':<7} {'Población':>13} {'Inmigrantes':>13} {'Emigrantes':>13}"
              f" {'Saldo':>13} {'% nac. fuera':>13}")
        for censo in censos:
            i, e, poblacion = flujos_nacimiento(df, cve, censo)
            n_i, n_e = i["personas"].sum(), e["personas"].sum()
            nota = " *" if cve in TERRITORIOS.get(censo, ()) else ""
            print(f"  {censo:<7} {poblacion:>13,} {n_i:>13,} {n_e:>13,}"
                  f" {n_i - n_e:>+13,} {n_i / poblacion * 100:>12.2f}%{nota}")
        if any(cve in TERRITORIOS.get(c, ()) for c in censos):
            print("  * territorio federal en ese censo, no estado")
        print()

    return inm, emi


def serie_nacimiento(df, cve):
    print(f"\n{'═' * 78}")
    print(f"  {NOMBRE[cve]} — migración interestatal acumulada, 1950–2020")
    print("═" * 78)
    print(f"  {'Censo':<7} {'Población':>13} {'Inmigrantes':>13} {'Emigrantes':>13}"
          f" {'Saldo':>13} {'% nac. fuera':>13}")
    for censo in CENSOS_NACIMIENTO:
        inm, emi, poblacion = flujos_nacimiento(df, cve, censo)
        n_inm, n_emi = inm["personas"].sum(), emi["personas"].sum()
        nota = " *" if cve in TERRITORIOS.get(censo, ()) else ""
        print(f"  {censo:<7} {poblacion:>13,} {n_inm:>13,} {n_emi:>13,}"
              f" {n_inm - n_emi:>+13,} {n_inm / poblacion * 100:>12.2f}%{nota}")
    if any(cve in TERRITORIOS.get(c, ()) for c in CENSOS_NACIMIENTO):
        print("\n  * territorio federal en ese censo, no estado")
    print()


def _matriz_flujo_nacimiento(df, censo):
    """Matriz cuadrada origen×destino (personas) de un censo: sólo migración
    a otra entidad con origen conocido (excluye "Entidad no especificada").
    Devuelve la lista ordenada de claves (índice de la matriz) y la matriz."""
    g = (df.filter((pl.col("censo") == censo) & (pl.col("categoria") == "En otra entidad")
                   & ~pl.col("total_categoria") & pl.col("cve_origen").is_not_null()
                   & (pl.col("cve_destino") > 0))
         .select(["cve_origen", "cve_destino", "personas"]))
    cves = sorted(set(g["cve_origen"].to_list()) | set(g["cve_destino"].to_list()))
    idx = {c: i for i, c in enumerate(cves)}
    matriz = np.zeros((len(cves), len(cves)))
    for o, d, personas in g.iter_rows():
        matriz[idx[o], idx[d]] = personas
    return cves, matriz


def _flujo_entidad_nacimiento(df, cve, censo, direccion) -> pl.DataFrame:
    """Tabla de flujo de una sola entidad en un censo: direccion='inmigracion'
    devuelve (cve_contraparte, personas) de todo lo que entra a `cve`;
    direccion='emigracion' devuelve lo que sale de `cve`. Mismo filtro limpio
    que _matriz_flujo_nacimiento (excluye agregados nacionales y "Entidad no
    especificada", que no se pueden ubicar en un mapa)."""
    g = (df.filter((pl.col("censo") == censo) & (pl.col("categoria") == "En otra entidad")
                   & ~pl.col("total_categoria") & pl.col("cve_origen").is_not_null()
                   & (pl.col("cve_destino") > 0)))
    if direccion == "inmigracion":
        g = g.filter(pl.col("cve_destino") == cve).select(["cve_origen", "personas"])
        g = g.rename({"cve_origen": "cve_contraparte"})
    else:
        g = g.filter(pl.col("cve_origen") == cve).select(["cve_destino", "personas"])
        g = g.rename({"cve_destino": "cve_contraparte"})
    return g.sort("personas", descending=True)


def convergencia_nacimiento(df, censo, k, top):
    """Agrupa las entidades de un censo por similitud de perfil migratorio:
    'emigracion' (mismos destinos preferentes) e 'inmigracion' (mismas
    fuentes preferentes). Imprime, por grupo, sus miembros y el/los
    destino(s)/origen(es) compartido(s) dominante(s). Devuelve
    {direccion: {cve_ent: cluster_id}} para --guardar."""
    cves, matriz = _matriz_flujo_nacimiento(df, censo)
    nombre = {c: NOMBRE[c] for c in cves}
    resultado = {}

    print(f"\n{'═' * 78}")
    print(f"  Convergencia de flujos migratorios — Censo {censo}")
    print("  Agrupamiento por perfil de preferencia (no por volumen)")
    print("═" * 78)

    direcciones = [
        ("origen", "emigracion",
         "EMIGRACIÓN — estados que comparten los mismos destinos preferentes",
         "Destino(s) compartido(s)"),
        ("destino", "inmigracion",
         "INMIGRACIÓN — estados que comparten las mismas fuentes preferentes",
         "Origen(es) compartido(s)"),
    ]
    for eje, clave, titulo, etiqueta_contraparte in direcciones:
        perfiles = _perfiles(matriz, eje)
        etiquetas, k_usado, score = _clusterizar(perfiles, k)
        resultado[clave] = {int(c): int(e) for c, e in zip(cves, etiquetas)}

        print(f"\n  {titulo}")
        if k is None:
            print(f"  (k={k_usado} elegido automáticamente vía silhouette score = {score:.3f})")
        grupos = {}
        for cve, etq in zip(cves, etiquetas):
            grupos.setdefault(int(etq), []).append(cve)

        for cluster_id in sorted(grupos):
            miembros = grupos[cluster_id]
            print(f"\n    Grupo {cluster_id + 1}: {', '.join(sorted(nombre[c] for c in miembros))}")
            perfil_medio = perfiles[[cves.index(c) for c in miembros]].mean(axis=0)
            print(f"      {etiqueta_contraparte} (promedio del grupo):")
            for i in np.argsort(perfil_medio)[::-1][:top]:
                if perfil_medio[i] <= 0:
                    continue
                print(f"        {nombre[cves[i]]:<24} {perfil_medio[i] * 100:>6.1f}%")
            if any(c in TERRITORIOS.get(censo, ()) for c in miembros):
                print("      * incluye entidad(es) que aún eran territorio federal en este censo")
    print()
    return resultado


def main_nacimiento(a):
    p = a._subparser
    if a.convergencia and a.entidad is not None:
        p.error("--convergencia es incompatible con --entidad (es un reporte cruzado de las 32 entidades)")
    if a.map and a.convergencia:
        p.error("--map es incompatible con --convergencia")
    if a.map and a.serie:
        p.error("--map es incompatible con --serie (usa un censo suelto en --año)")
    if a.serie and a.entidad is None:
        p.error("--serie requiere --entidad")
    if a.map and a.año is None:
        p.error("--map requiere --año (censo suelto)")
    if a.map and isinstance(a.año, tuple):
        p.error("--map requiere un censo suelto en --año, no un rango")
    if a.inmigracion and a.emigracion:
        p.error("--inmigracion y --emigracion son mutuamente excluyentes")
    if a.map in ("geo", "sankey") and a.entidad is None:
        p.error(f"--map {a.map} requiere --entidad")
    if a.map and a.entidad is not None and not a.inmigracion and not a.emigracion:
        p.error("--map con --entidad requiere --inmigracion o --emigracion")
    if not a.convergencia and a.entidad is None and isinstance(a.año, tuple):
        p.error("el reporte/chord nacional requiere un censo suelto en --año, no un rango")
    if (not a.convergencia and not a.map and a.entidad is None
            and not (a.inmigracion or a.emigracion)):
        p.error("se requiere --entidad (o usa --convergencia, o --inmigracion/--emigracion "
                 "para el ranking nacional)")
    if not a.convergencia and not a.serie and a.año is None:
        p.error("se requiere --año (o usa --serie)")

    df = cargar_nacimiento()

    if a.convergencia:
        censos = (list(a.año) if isinstance(a.año, tuple)
                  else [a.año] if a.año else list(CENSOS_CONVERGENCIA_NACIMIENTO))
        for censo in censos:
            resultado = convergencia_nacimiento(df, censo, a.k, a.top)
            if a.guardar:
                for direccion, asignacion in resultado.items():
                    salida = pl.DataFrame({
                        "censo": [censo] * len(asignacion),
                        "cve_ent": list(asignacion.keys()),
                        "entidad": [NOMBRE[c] for c in asignacion],
                        "direccion": [direccion] * len(asignacion),
                        "cluster_id": list(asignacion.values()),
                    })
                    ruta = DIR / f"convergencia_{direccion}_{censo}.parquet"
                    salida.write_parquet(ruta)
                    print(f"Saved → {ruta}  ({salida.height:,} filas)\n")
        return

    direccion = "inmigracion" if a.inmigracion else "emigracion" if a.emigracion else None

    if a.entidad is None:
        cves, matriz = _matriz_flujo_nacimiento(df, a.año)
        if direccion:
            _tabla_flujo_nacional(cves, matriz, direccion, a.top)
        if a.map == "chord":
            ruta = mapa_flujo_chord(cves, matriz, a.año, a.top, "migracion")
            print(f"Saved → {ruta}\n")
        return

    cve = a.entidad
    if a.serie:
        serie_nacimiento(df, cve)
        return

    if isinstance(a.año, tuple):
        inm, emi = acumulado_nacimiento(df, cve, list(a.año), a.top, direccion)
        etiqueta_año = f"{a.año[0]}-{a.año[-1]}"
    else:
        inm, emi = detalle_nacimiento(df, cve, a.año, a.top, direccion)
        etiqueta_año = str(a.año)

    if a.guardar:
        salida = pl.concat([
            inm.rename({"origen": "contraparte"}).with_columns(pl.lit("inmigracion").alias("flujo")),
            emi.rename({"destino": "contraparte"}).with_columns(pl.lit("emigracion").alias("flujo")),
        ]).with_columns(
            pl.lit(etiqueta_año).alias("censo"),
            pl.lit(cve).cast(pl.Int16).alias("cve_ent"),
            pl.lit(NOMBRE[cve]).alias("entidad"),
        ).select(["censo", "cve_ent", "entidad", "flujo", "contraparte", "personas"])
        ruta = DIR / f"nacimiento_{NOMBRE[cve].lower().replace(' ', '_')}_{etiqueta_año}.parquet"
        salida.write_parquet(ruta)
        print(f"Saved → {ruta}  ({salida.height:,} filas)\n")

    if a.map:
        tabla = _flujo_entidad_nacimiento(df, cve, a.año, direccion)
        fn = {"geo": mapa_flujo_geo, "sankey": mapa_flujo_sankey,
              "chord": mapa_flujo_chord_entidad}[a.map]
        ruta = fn(cve, a.año, direccion, tabla, a.top, "migracion")
        print(f"Saved → {ruta}\n")


# ════════════════════════════════════════════════════════════════════════════
# Subcomando: movilidad
# ════════════════════════════════════════════════════════════════════════════

def cargar_movilidad():
    return pl.read_parquet(DIR / "ccpv_migracion_estatal.parquet")


def _agregado(df, cve_ent, categoria):
    f = df.filter(pl.col("total_categoria") & (pl.col("cve_destino") == cve_ent)
                  & (pl.col("categoria") == categoria))
    return f["personas"].sum()


def flujos_movilidad(mig, cve, censo):
    """(inmigrantes, emigrantes, población, tabla flujo completa) de un censo."""
    flujo = mig.filter((pl.col("censo") == censo) & (pl.col("concepto") == "residencia_5a"))
    desglose = flujo.filter(~pl.col("total_categoria"))

    if desglose.filter(pl.col("cve_destino") == cve).height == 0:
        # Fuente sin matriz origen-destino (Encuesta Intercensal 2015): sólo hay
        # totales agregados. Se representa como una única fila "placeholder" para
        # que las tablas top-N y el total (personas.sum()) sigan funcionando sin
        # ramificar el resto del archivo.
        inm = pl.DataFrame({"origen": [SIN_DESGLOSE],
                            "personas": [_agregado(flujo, cve, "En otra entidad")]},
                           schema={"origen": pl.Utf8, "personas": pl.Int64})
        emi = pl.DataFrame({"destino": [SIN_DESGLOSE],
                            "personas": [_agregado(flujo, cve, EMIGRANTES_AGREGADO)]},
                           schema={"destino": pl.Utf8, "personas": pl.Int64})
    else:
        # Incluye la fila 'Entidad no especificada' (sólo 1990) para que el total
        # cuadre con el agregado publicado de 'En otra entidad'. Esos casos no son
        # atribuibles a un origen, así que no aparecen del lado de los emigrantes.
        inm = (desglose.filter(pl.col("cve_destino") == cve)
               .select(["origen", "personas"]).sort("personas", descending=True))
        emi = (desglose.filter((pl.col("cve_origen") == cve) & (pl.col("cve_destino") > 0))
               .select(["destino", "personas"]).sort("personas", descending=True))
        assert inm["personas"].sum() == _agregado(flujo, cve, "En otra entidad")

    poblacion = _agregado(flujo, cve, "Total")
    return inm, emi, poblacion, flujo


def paises_2020(cve, top):
    """Desglose por país del flujo extranjero de 2020 (único censo con este detalle)."""
    df = (
        pl.read_parquet(DIR / "ccpv_extranjeros_pais_2020.parquet")
        .filter((pl.col("cve_destino") == cve) & (pl.col("cod_pais") < 997))
        .select(["pais", "personas"]).sort("personas", descending=True)
    )
    _tabla("  Flujo por país de residencia en marzo de 2015", df, "pais",
           df["personas"].sum(), top)
    return df


def detalle_movilidad(mig, cve, censo, top, direccion=None):
    """`direccion=None` imprime ambas tablas (INMIGRANTES y EMIGRANTES);
    'inmigracion'/'emigracion' imprime sólo la correspondiente."""
    inm, emi, poblacion, flujo = flujos_movilidad(mig, cve, censo)
    n_inm, n_emi = inm["personas"].sum(), emi["personas"].sum()

    print(f"\n{'═' * 72}")
    print(f"  {NOMBRE[cve]} — Censo {censo}")
    print(f"  Movilidad interestatal declarada, {VENTANA[censo]}")
    print(f"  Población de 5 años y más: {poblacion:,}")
    print("═" * 72)

    if direccion != "emigracion":
        _tabla("INMIGRANTES — llegaron desde otra entidad", inm, "origen", n_inm, top, poblacion)
    if direccion != "inmigracion":
        _tabla("EMIGRANTES — salieron hacia otra entidad", emi, "destino", n_emi, top, poblacion)

    neto = n_inm - n_emi
    print(f"\nSALDO NETO INTERESTATAL: {neto:+,}"
          f"  ({neto / poblacion * 1000:+.2f} por mil habitantes)")

    # ── Extranjeros ──────────────────────────────────────────────────────────
    print(f"\n{'─' * 72}\nEXTRANJEROS")

    if flujo.filter(pl.col("categoria").is_in(EXTRANJERO)).height == 0:
        print("\n  Flujo — no disponible en esta fuente (sólo migración interestatal).")
    else:
        f_ext = {c: _agregado(flujo, cve, c) for c in EXTRANJERO}
        print(f"\n  Flujo — residían en otro país en {VENTANA[censo].split(' → ')[0]}: "
              f"{sum(f_ext.values()):,}")
        if censo >= 2005:  # 1990 y 2000 no separan Estados Unidos del resto
            for c, v in f_ext.items():
                print(f"      {c:<38} {v:>12,}")

    paises = pl.DataFrame(schema={"pais": pl.Utf8, "personas": pl.Int64})
    if censo == 2020:
        paises = paises_2020(cve, top)

    stock = mig.filter((pl.col("censo") == censo) & (pl.col("concepto") == "nacimiento"))
    if stock.height == 0:
        print("\n  Stock — no disponible en esta fuente (no releva lugar de nacimiento).")
    else:
        s_ext = {c: _agregado(stock, cve, c) for c in EXTRANJERO}
        s_tot = sum(s_ext.values())
        pob_total = _agregado(stock, cve, "Total")
        print(f"\n  Stock — nacidos en el extranjero: {s_tot:,}"
              f"  ({s_tot / pob_total * 100:.2f}% de la población total)")
        if censo >= 2005:
            for c, v in s_ext.items():
                print(f"      {c:<38} {v:>12,}")
        sexo = stock.filter(pl.col("total_categoria") & (pl.col("cve_destino") == cve)
                            & pl.col("categoria").is_in(EXTRANJERO))
        h, m = sexo["hombres"].sum(), sexo["mujeres"].sum()
        if h and m:
            print(f"      {'Hombres / Mujeres':<38} {h:>12,} / {m:,}")
    print()
    return inm, emi, paises


def flujos_acumulados_movilidad(mig, cve, censos):
    """Suma inmigrantes/emigrantes de una entidad a través de varios censos.

    Cada censo mide un quinquenio DISTINTO y no solapado (1985-1990, 1995-2000,
    2005-2010, 2015-2020), así que sumarlos no duplica el mismo movimiento dos
    veces — pero tampoco cubre los años intermedios (1990-1995, 2000-2005,
    2010-2015), que ningún censo captura.
    """
    inms, emis = [], []
    for censo in censos:
        inm, emi, _, _ = flujos_movilidad(mig, cve, censo)
        inms.append(inm)
        emis.append(emi)
    inm = (pl.concat(inms).group_by("origen").agg(pl.col("personas").sum())
           .sort("personas", descending=True))
    emi = (pl.concat(emis).group_by("destino").agg(pl.col("personas").sum())
           .sort("personas", descending=True))
    return inm, emi


def acumulado_movilidad(mig, cve, censos, top, direccion=None):
    """`direccion` como en detalle_movilidad. El % de población de las tablas
    se calcula sobre la población del censo destino (el más reciente del
    rango), igual criterio que acumulado_nacimiento."""
    inm, emi = flujos_acumulados_movilidad(mig, cve, censos)
    n_inm, n_emi = inm["personas"].sum(), emi["personas"].sum()
    _, _, poblacion_destino, _ = flujos_movilidad(mig, cve, censos[-1])

    print(f"\n{'═' * 72}")
    print(f"  {NOMBRE[cve]} — Censos {censos[0]}–{censos[-1]} (acumulado)")
    print("  Movilidad interestatal declarada, suma de quinquenios no solapados")
    print(f"  Ventanas sumadas: {' · '.join(VENTANA[c] for c in censos)}")
    print("═" * 72)

    if direccion != "emigracion":
        _tabla(f"INMIGRANTES — suma de {len(censos)} censos", inm, "origen", n_inm, top,
               poblacion_destino)
    if direccion != "inmigracion":
        _tabla(f"EMIGRANTES — suma de {len(censos)} censos", emi, "destino", n_emi, top,
               poblacion_destino)

    print(f"\nSALDO NETO INTERESTATAL (suma de los {len(censos)} censos): {n_inm - n_emi:+,}")
    print("\nNota: estos censos cubren quinquenios no solapados, pero dejan años sin medir")
    print("entre uno y otro (p. ej. 1990-1995, 2000-2005, 2010-2015). La suma es la")
    print("migración observada en las ventanas sumadas, no la migración continua del período.")

    print(f"\n{'─' * 72}")
    print("Balance por censo — migración, inmigración y población total:")
    print(f"  {'Censo':<7} {'Población':>13} {'Inmigrantes':>13} {'Emigrantes':>13}"
          f" {'Saldo neto':>13} {'Saldo ‰':>10}")
    for censo in censos:
        i, e, poblacion, _ = flujos_movilidad(mig, cve, censo)
        n_i, n_e = i["personas"].sum(), e["personas"].sum()
        saldo = n_i - n_e
        print(f"  {censo:<7} {poblacion:>13,} {n_i:>13,} {n_e:>13,}"
              f" {saldo:>+13,} {saldo / poblacion * 1000:>9.2f}")

    # ── Extranjeros ──────────────────────────────────────────────────────────
    print(f"\n{'─' * 72}\nEXTRANJEROS")

    flujo = mig.filter(pl.col("censo").is_in(censos) & (pl.col("concepto") == "residencia_5a")
                       & pl.col("total_categoria") & (pl.col("cve_destino") == cve)
                       & pl.col("categoria").is_in(EXTRANJERO))
    ext = flujo.group_by("categoria").agg(pl.col("personas").sum())
    total_ext = ext["personas"].sum()
    print(f"\n  Flujo — residían en otro país, suma de {len(censos)} censos: {total_ext:,}")
    sin_flujo_ext = [c for c in censos
                     if mig.filter((pl.col("censo") == c) & (pl.col("concepto") == "residencia_5a")
                                   & pl.col("categoria").is_in(EXTRANJERO)).height == 0]
    if sin_flujo_ext:
        print(f"      (sin flujo de extranjero en esta fuente para: "
              f"{', '.join(str(c) for c in sin_flujo_ext)})")
    # 1990/2000 no separan EEUU del resto, así que desglosar mezclado con 2005+
    # atribuiría de más a 'En otro país'. Sólo se desglosa si todos los censos separan.
    if all(c >= 2005 for c in censos if c not in sin_flujo_ext):
        for r in ext.sort("personas", descending=True).iter_rows(named=True):
            print(f"      {r['categoria']:<38} {r['personas']:>12,}")

    paises = pl.DataFrame(schema={"pais": pl.Utf8, "personas": pl.Int64})
    if 2020 in censos:
        paises = paises_2020(cve, top)

    # El stock (nacidos en el extranjero) es un corte a cada censo, no un flujo: una
    # misma persona puede seguir ahí en el siguiente censo. Se muestra por censo, sin
    # sumar, para no fingir un conteo de personas únicas.
    print(f"\n  Stock — nacidos en el extranjero, por censo (no se suma entre censos):")
    for censo in censos:
        stock = mig.filter((pl.col("censo") == censo) & (pl.col("concepto") == "nacimiento"))
        if stock.height == 0:
            print(f"      {censo:<10} no disponible en esta fuente")
            continue
        s_tot = sum(_agregado(stock, cve, c) for c in EXTRANJERO)
        pob_total = _agregado(stock, cve, "Total")
        print(f"      {censo:<10} {s_tot:>12,}   ({s_tot / pob_total * 100:.2f}% de la población)")
    print()
    return inm, emi, paises


def _matriz_flujo_movilidad(mig, censo):
    """Matriz cuadrada origen×destino (personas) del quinquenio de un censo:
    sólo migración a otra entidad con origen conocido (excluye "Entidad no
    especificada"). Devuelve la lista ordenada de claves y la matriz."""
    if censo not in CENSOS_CONVERGENCIA_MOVILIDAD:
        raise SystemExit(
            f"El censo {censo} no trae matriz origen-destino en esta fuente "
            f"(sólo agregados) — --convergencia/--map sin --entidad no aplican. "
            f"Censos disponibles: {list(CENSOS_CONVERGENCIA_MOVILIDAD)}")
    g = (mig.filter((pl.col("censo") == censo) & (pl.col("concepto") == "residencia_5a")
                    & (pl.col("categoria") == "En otra entidad") & ~pl.col("total_categoria")
                    & pl.col("cve_origen").is_not_null() & (pl.col("cve_destino") > 0))
         .select(["cve_origen", "cve_destino", "personas"]))
    cves = sorted(set(g["cve_origen"].to_list()) | set(g["cve_destino"].to_list()))
    idx = {c: i for i, c in enumerate(cves)}
    matriz = np.zeros((len(cves), len(cves)))
    for o, d, personas in g.iter_rows():
        matriz[idx[o], idx[d]] = personas
    return cves, matriz


def _flujo_entidad_movilidad(mig, cve, censo, direccion) -> pl.DataFrame:
    """Tabla de flujo de una sola entidad en el quinquenio de un censo:
    direccion='inmigracion' devuelve (cve_contraparte, personas) de quienes
    llegaron a `cve`; direccion='emigracion' lo que salió de `cve`. Mismo
    filtro limpio que _matriz_flujo_movilidad (excluye agregados y "Entidad no
    especificada", y rechaza censos sin matriz origen-destino como 2015)."""
    if censo not in CENSOS_CONVERGENCIA_MOVILIDAD:
        raise SystemExit(
            f"El censo {censo} no trae matriz origen-destino en esta fuente "
            f"(sólo agregados) — --map no aplica. Censos disponibles: "
            f"{list(CENSOS_CONVERGENCIA_MOVILIDAD)}")
    g = (mig.filter((pl.col("censo") == censo) & (pl.col("concepto") == "residencia_5a")
                    & (pl.col("categoria") == "En otra entidad") & ~pl.col("total_categoria")
                    & pl.col("cve_origen").is_not_null() & (pl.col("cve_destino") > 0)))
    if direccion == "inmigracion":
        g = g.filter(pl.col("cve_destino") == cve).select(["cve_origen", "personas"])
        g = g.rename({"cve_origen": "cve_contraparte"})
    else:
        g = g.filter(pl.col("cve_origen") == cve).select(["cve_destino", "personas"])
        g = g.rename({"cve_destino": "cve_contraparte"})
    return g.sort("personas", descending=True)


def convergencia_movilidad(mig, censo, k, top):
    """Agrupa las entidades de un censo por similitud de perfil de movilidad
    del quinquenio: 'emigracion' (mismos destinos preferentes) e
    'inmigracion' (mismas fuentes preferentes). Imprime, por grupo, sus
    miembros y el/los destino(s)/origen(es) compartido(s) dominante(s).
    Devuelve {direccion: {cve_ent: cluster_id}} para --guardar."""
    cves, matriz = _matriz_flujo_movilidad(mig, censo)
    nombre = {c: NOMBRE[c] for c in cves}
    resultado = {}

    print(f"\n{'═' * 78}")
    print(f"  Convergencia de movilidad interestatal — Censo {censo} ({VENTANA[censo]})")
    print("  Agrupamiento por perfil de preferencia (no por volumen)")
    print("═" * 78)

    direcciones = [
        ("origen", "emigracion",
         "EMIGRACIÓN — estados que comparten los mismos destinos preferentes",
         "Destino(s) compartido(s)"),
        ("destino", "inmigracion",
         "INMIGRACIÓN — estados que comparten las mismas fuentes preferentes",
         "Origen(es) compartido(s)"),
    ]
    for eje, clave, titulo, etiqueta_contraparte in direcciones:
        perfiles = _perfiles(matriz, eje)
        etiquetas, k_usado, score = _clusterizar(perfiles, k)
        resultado[clave] = {int(c): int(e) for c, e in zip(cves, etiquetas)}

        print(f"\n  {titulo}")
        if k is None:
            print(f"  (k={k_usado} elegido automáticamente vía silhouette score = {score:.3f})")
        grupos = {}
        for cve, etq in zip(cves, etiquetas):
            grupos.setdefault(int(etq), []).append(cve)

        for cluster_id in sorted(grupos):
            miembros = grupos[cluster_id]
            print(f"\n    Grupo {cluster_id + 1}: {', '.join(sorted(nombre[c] for c in miembros))}")
            perfil_medio = perfiles[[cves.index(c) for c in miembros]].mean(axis=0)
            print(f"      {etiqueta_contraparte} (promedio del grupo):")
            for i in np.argsort(perfil_medio)[::-1][:top]:
                if perfil_medio[i] <= 0:
                    continue
                print(f"        {nombre[cves[i]]:<24} {perfil_medio[i] * 100:>6.1f}%")
    print()
    return resultado


def main_movilidad(a):
    p = a._subparser
    if a.convergencia and a.entidad is not None:
        p.error("--convergencia es incompatible con --entidad (es un reporte cruzado de las 32 entidades)")
    if a.map and a.convergencia:
        p.error("--map es incompatible con --convergencia")
    if a.map and a.año is None:
        p.error("--map requiere --año (censo suelto)")
    if a.map and isinstance(a.año, tuple):
        p.error("--map requiere un censo suelto en --año, no un rango")
    if a.inmigracion and a.emigracion:
        p.error("--inmigracion y --emigracion son mutuamente excluyentes")
    if a.map in ("geo", "sankey") and a.entidad is None:
        p.error(f"--map {a.map} requiere --entidad")
    if a.map and a.entidad is not None and not a.inmigracion and not a.emigracion:
        p.error("--map con --entidad requiere --inmigracion o --emigracion")
    if not a.convergencia and a.entidad is None and isinstance(a.año, tuple):
        p.error("el reporte/chord nacional requiere un censo suelto en --año, no un rango")
    if (not a.convergencia and not a.map and a.entidad is None
            and not (a.inmigracion or a.emigracion)):
        p.error("se requiere --entidad (o usa --convergencia, o --inmigracion/--emigracion "
                 "para el ranking nacional)")
    if not a.convergencia and a.año is None:
        p.error("se requiere --año")

    mig = cargar_movilidad()

    if a.convergencia:
        censos = (list(a.año) if isinstance(a.año, tuple)
                  else [a.año] if a.año else list(CENSOS_CONVERGENCIA_MOVILIDAD))
        for censo in censos:
            resultado = convergencia_movilidad(mig, censo, a.k, a.top)
            if a.guardar:
                for direccion, asignacion in resultado.items():
                    salida = pl.DataFrame({
                        "censo": [censo] * len(asignacion),
                        "cve_ent": list(asignacion.keys()),
                        "entidad": [NOMBRE[c] for c in asignacion],
                        "direccion": [direccion] * len(asignacion),
                        "cluster_id": list(asignacion.values()),
                    })
                    ruta = DIR / f"convergencia_movilidad_{direccion}_{censo}.parquet"
                    salida.write_parquet(ruta)
                    print(f"Saved → {ruta}  ({salida.height:,} filas)\n")
        return

    direccion = "inmigracion" if a.inmigracion else "emigracion" if a.emigracion else None

    if a.entidad is None:
        cves, matriz = _matriz_flujo_movilidad(mig, a.año)
        if direccion:
            _tabla_flujo_nacional(cves, matriz, direccion, a.top)
        if a.map == "chord":
            ruta = mapa_flujo_chord(cves, matriz, a.año, a.top, "movilidad", ventana=VENTANA[a.año])
            print(f"Saved → {ruta}\n")
        return

    cve, top = a.entidad, a.top
    if isinstance(a.año, tuple):
        inm, emi, paises = acumulado_movilidad(mig, cve, list(a.año), top, direccion)
        etiqueta_año = f"{a.año[0]}-{a.año[-1]}"
    else:
        inm, emi, paises = detalle_movilidad(mig, cve, a.año, top, direccion)
        etiqueta_año = str(a.año)

    if a.guardar:
        salida = pl.concat([
            inm.rename({"origen": "contraparte"}).with_columns(pl.lit("inmigracion").alias("flujo")),
            emi.rename({"destino": "contraparte"}).with_columns(pl.lit("emigracion").alias("flujo")),
            paises.rename({"pais": "contraparte"}).with_columns(pl.lit("extranjero").alias("flujo")),
        ]).with_columns(
            pl.lit(etiqueta_año).alias("censo"),
            pl.lit(cve).cast(pl.Int16).alias("cve_ent"),
            pl.lit(NOMBRE[cve]).alias("entidad"),
        ).select(["censo", "cve_ent", "entidad", "flujo", "contraparte", "personas"])
        ruta = DIR / f"movilidad_{NOMBRE[cve].lower().replace(' ', '_')}_{etiqueta_año}.parquet"
        salida.write_parquet(ruta)
        print(f"Saved → {ruta}  ({salida.height:,} filas)\n")

    if a.map:
        tabla = _flujo_entidad_movilidad(mig, cve, a.año, direccion)
        fn = {"geo": mapa_flujo_geo, "sankey": mapa_flujo_sankey,
              "chord": mapa_flujo_chord_entidad}[a.map]
        ruta = fn(cve, a.año, direccion, tabla, top, "movilidad", ventana=VENTANA[a.año])
        print(f"Saved → {ruta}\n")


# ── CLI ──────────────────────────────────────────────────────────────────

def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    p_nac = sub.add_parser("nacimiento",
                            help="Migración acumulada por lugar de nacimiento, censos 1950–2020")
    p_nac.set_defaults(_subparser=p_nac)
    p_nac.add_argument("--entidad", type=_entidad, default=None,
                   help="Nombre o clave INEGI (1-32) de la entidad federativa "
                        "(obligatorio salvo con --convergencia)")
    p_nac.add_argument("--año", type=_año_o_rango_nacimiento,
                   help="Año censal, o rango 'AAAA-AAAA' para el acumulado de varios "
                        "censos (obligatorio salvo con --serie; con --convergencia, "
                        "default 1950-1980)")
    p_nac.add_argument("--top", type=int, default=10, help="Filas por tabla (default 10)")
    p_nac.add_argument("--serie", action="store_true",
                   help="Trayectoria de la entidad en los 8 censos, en vez del detalle")
    p_nac.add_argument("--convergencia", action="store_true",
                   help="Agrupa las 32 entidades por similitud de perfil migratorio "
                        "(destinos preferentes en emigración, fuentes preferentes en "
                        "inmigración) en vez de reportar una sola entidad. Incompatible "
                        "con --entidad")
    p_nac.add_argument("--k", type=_k, default=None,
                   help="Número de grupos para --convergencia. Si se omite, se "
                        "autodetecta vía silhouette score (k=2..10)")
    p_nac.add_argument("--map", nargs="?", choices=["geo", "sankey", "chord"], const="geo", default=None,
                   help="Genera un diagrama de flujo HTML en dashboard_data/. Requiere --año "
                        "(censo suelto). --map geo (default) arcos sobre el mapa de México, "
                        "--map sankey diagrama de bandas: ambos requieren --entidad y "
                        "exactamente uno de --inmigracion/--emigracion. --map chord también "
                        "acepta --entidad y una dirección (dibuja esa entidad + sus --top "
                        "contrapartes como cuerdas en círculo), pero --entidad es opcional: "
                        "si se omite, genera un chord diagram NACIONAL con los --top flujos "
                        "más grandes entre cualquier par de entidades")
    p_nac.add_argument("--inmigracion", action="store_true",
                   help="Filtra el reporte a sólo inmigración (con --entidad: orígenes que le "
                        "mandan migrantes; sin --entidad: ranking nacional de entidades "
                        "receptoras). No requiere --map; con --map y --entidad, además fija la "
                        "dirección del diagrama")
    p_nac.add_argument("--emigracion", action="store_true",
                   help="Filtra el reporte a sólo emigración (con --entidad: destinos hacia "
                        "donde emigra; sin --entidad: ranking nacional de entidades emisoras). "
                        "No requiere --map; con --map y --entidad, además fija la dirección "
                        "del diagrama")
    p_nac.add_argument("--guardar", action="store_true",
                   help="Escribe el resultado a dashboard_data/nacimiento_<entidad>_<año>.parquet "
                        "(o dashboard_data/convergencia_<direccion>_<censo>.parquet con --convergencia)")

    p_mov = sub.add_parser("movilidad",
                            help="Movilidad declarada del quinquenio previo, censos 1990–2020")
    p_mov.set_defaults(_subparser=p_mov)
    p_mov.add_argument("--entidad", type=_entidad, default=None,
                   help="Nombre o clave INEGI (1-32) de la entidad federativa "
                        "(obligatorio salvo con --convergencia)")
    p_mov.add_argument("--año", type=_año_o_rango_movilidad,
                   help="Año censal, o rango 'AAAA-AAAA' para el acumulado de varios censos "
                        "(obligatorio salvo con --convergencia; con --convergencia, default "
                        f"{CENSOS_CONVERGENCIA_MOVILIDAD[0]}-{CENSOS_CONVERGENCIA_MOVILIDAD[-1]}, "
                        "excluye 2015)")
    p_mov.add_argument("--top", type=int, default=10, help="Filas por tabla (default 10)")
    p_mov.add_argument("--convergencia", action="store_true",
                   help="Agrupa las 32 entidades por similitud de perfil de movilidad "
                        "(destinos preferentes en emigración, fuentes preferentes en "
                        "inmigración) en vez de reportar una sola entidad. Incompatible "
                        "con --entidad; no aplica a 2015 (sin matriz origen-destino)")
    p_mov.add_argument("--k", type=_k, default=None,
                   help="Número de grupos para --convergencia. Si se omite, se "
                        "autodetecta vía silhouette score (k=2..10)")
    p_mov.add_argument("--map", nargs="?", choices=["geo", "sankey", "chord"], const="geo", default=None,
                   help="Genera un diagrama de flujo HTML en dashboard_data/. Requiere --año "
                        "(censo suelto, 2015 excluido). --map geo (default) arcos sobre el "
                        "mapa de México, --map sankey diagrama de bandas: ambos requieren "
                        "--entidad y exactamente uno de --inmigracion/--emigracion. --map "
                        "chord también acepta --entidad y una dirección (dibuja esa entidad "
                        "+ sus --top contrapartes como cuerdas en círculo), pero --entidad es "
                        "opcional: si se omite, genera un chord diagram NACIONAL con los "
                        "--top flujos más grandes entre cualquier par de entidades")
    p_mov.add_argument("--inmigracion", action="store_true",
                   help="Filtra el reporte a sólo inmigración (con --entidad: orígenes que le "
                        "mandan gente; sin --entidad: ranking nacional de entidades receptoras). "
                        "No requiere --map; con --map y --entidad, además fija la dirección "
                        "del diagrama")
    p_mov.add_argument("--emigracion", action="store_true",
                   help="Filtra el reporte a sólo emigración (con --entidad: destinos hacia "
                        "donde se fue; sin --entidad: ranking nacional de entidades emisoras). "
                        "No requiere --map; con --map y --entidad, además fija la dirección "
                        "del diagrama")
    p_mov.add_argument("--guardar", action="store_true",
                   help="Escribe el resultado a dashboard_data/movilidad_<entidad>_<año>.parquet "
                        "(o dashboard_data/convergencia_movilidad_<direccion>_<censo>.parquet "
                        "con --convergencia)")
    return p


def main():
    a = build_parser().parse_args()
    {"nacimiento": main_nacimiento, "movilidad": main_movilidad}[a.cmd](a)


if __name__ == "__main__":
    main()
