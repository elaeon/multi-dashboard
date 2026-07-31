"""
Regionalización estable de las 32 entidades por movilidad interestatal.

Corre el agrupamiento de movilidad_estatal.py --convergencia (regiones de
entidades vecinas que maximizan el intercambio migratorio interno respecto al
esperado por su tamaño) sobre los 5 censos con matriz origen-destino —1990,
2000, 2005, 2010 y 2020; 2015 es Encuesta Intercensal y sólo trae agregados—
y consolida los 5 agrupamientos en UNA sola asignación entidad→región, para
k=5 y k=6 (ver PRESETS).

Con k=5, 29 de las 32 entidades caen en la misma región los 5 censos; las 3
que no (Guerrero, Zacatecas, Coahuila) se asignan por mayoría de censos. Con
k=6 el grupo Noroeste de k=5 se divide en Noroeste + Norte-centro, lo que dis-
tingue 3 entidades inestables distintas (Guerrero, Aguascalientes, Zacatecas).
Ojo con la mayoría: en ambos presets hay entidades que están SALIENDO de la
región que les toca —cambiaron justo en el censo más reciente, 2020— así que
la mayoría favorece el pasado sobre la foto actual. Ver el reporte de cada
preset al correr el script para el detalle censo por censo.

Los nombres de región son geográficos y se asignan aquí: el algoritmo sólo
produce IDs numéricos, y además los renumera en cada censo (van ordenados por
tamaño de grupo). NUCLEOS de cada preset es la llave de nombrado — cada grupo
de cada censo se bautiza con el núcleo con el que más entidades comparte.

Fuente: dashboard_data/ccpv_migracion_estatal.parquet
        (generada por scripts/prepare_ccpv_migracion.py)

Output: dashboard_data/regiones_movilidad.csv     (k=5, columnas: region, entidad)
        dashboard_data/regiones_movilidad_k6.csv  (k=6, columnas: region, entidad)
Run: uv run python scripts/prepare_regiones_movilidad.py
"""

import sys
from collections import Counter, defaultdict
from pathlib import Path

import polars as pl

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "scripts" / "datatable"))
sys.path.insert(0, str(RAIZ / "scripts" / "centralismo"))
from comun import NOMBRE
import movilidad_estatal as mov

OUT_DIR = RAIZ / "dashboard_data"
CENSOS = mov.CENSOS_CONVERGENCIA_MOVILIDAD

# Un preset por K: núcleo invariante de cada región (claves INEGI) → nombre,
# y el CSV de salida. Los núcleos sirven sólo para nombrar: cada grupo de
# cada censo se empareja con el núcleo con el que más entidades comparte. El
# orden de las llaves es el de salida del CSV.
PRESETS = {
    5: {
        "ruta": OUT_DIR / "regiones_movilidad.csv",
        "nucleos": {
            "Noroeste":        {2, 3, 8, 10, 25, 26},         # BC, BCS, Chih, Dgo, Sin, Son
            "Noreste":         {19, 20, 21, 24, 28, 29, 30},  # NL, Oax, Pue, SLP, Tamps, Tlax, Ver
            "Occidente":       {1, 6, 11, 14, 16, 18},        # Ags, Col, Gto, Jal, Mich, Nay
            "Valle de México": {9, 13, 15, 17, 22},           # CDMX, Hgo, Edomex, Mor, Qro
            "Sureste":         {4, 7, 23, 27, 31},            # Camp, Chis, QRoo, Tab, Yuc
        },
    },
    6: {
        "ruta": OUT_DIR / "regiones_movilidad_k6.csv",
        "nucleos": {
            "Noroeste":        {2, 3, 25, 26},                # BC, BCS, Sin, Son
            "Norte-centro":    {5, 8, 10},                    # Coahuila, Chihuahua, Durango
            "Noreste":         {19, 20, 21, 24, 28, 29, 30},  # NL, Oax, Pue, SLP, Tamps, Tlax, Ver
            "Occidente":       {6, 11, 14, 16, 18},           # Col, Gto, Jal, Mich, Nay
            "Valle de México": {9, 13, 15, 17, 22},           # CDMX, Hgo, Edomex, Mor, Qro
            "Sureste":         {4, 7, 23, 27, 31},            # Camp, Chis, QRoo, Tab, Yuc
        },
    },
}


def agrupar(mig, censo, k) -> list[set[int]]:
    """Regiones de un censo en k grupos, como sets de claves INEGI. Usa la
    maquinaria de movilidad_estatal.py directamente en vez del reporte de
    --convergencia, que además imprime."""
    cves, matriz = mov._matriz_flujo_movilidad(mig, censo)
    grupos = mov._agrupar_contiguo(mov._afinidad_intercambio(matriz),
                                   mov._intercambio(matriz),
                                   mov._adyacencia(cves), k)
    return [{cves[i] for i in g} for g in grupos]


def nombrar(grupos, censo, nucleos) -> dict[str, set[int]]:
    """Bautiza cada grupo con el núcleo (de `nucleos`) con el que más
    entidades comparte, resolviendo conflictos por traslape descendente."""
    pares = sorted(((len(g & nuc), gi, nombre)
                    for gi, g in enumerate(grupos) for nombre, nuc in nucleos.items()),
                   reverse=True)
    asignado, usados = {}, set()
    for _, gi, nombre in pares:
        if gi in asignado or nombre in usados:
            continue
        asignado[gi], _ = nombre, usados.add(nombre)
    if len(asignado) != len(grupos) or len(usados) != len(nucleos):
        raise RuntimeError(
            f"Censo {censo}: no se pudo emparejar cada grupo con un núcleo distinto. "
            f"Revisa NUCLEOS de este preset — el agrupamiento cambió lo suficiente "
            f"como para invalidar los nombres de región.")
    return {asignado[gi]: g for gi, g in enumerate(grupos)}


def generar(mig, k, ruta, nucleos):
    votos = defaultdict(Counter)
    for censo in CENSOS:
        for nombre, entidades in nombrar(agrupar(mig, censo, k), censo, nucleos).items():
            for cve in entidades:
                votos[cve][nombre] += 1

    orden = {nombre: i for i, nombre in enumerate(nucleos)}
    tabla = (
        pl.DataFrame([{"region": votos[cve].most_common(1)[0][0], "entidad": NOMBRE[cve]}
                      for cve in range(1, 33)])
        .with_columns(pl.col("region").replace_strict(orden).alias("_orden"))
        .sort(["_orden", "entidad"])
        .drop("_orden")
    )
    assert tabla.height == 32, f"{tabla.height} entidades, se esperaban 32"

    ruta.parent.mkdir(parents=True, exist_ok=True)
    tabla.write_csv(ruta)
    print(f"\n{'═' * 60}\nGuardado → {ruta}  (k={k}, {tabla.height} entidades, "
          f"{len(CENSOS)} censos: {', '.join(map(str, CENSOS))})\n" + "═" * 60)

    for nombre in nucleos:
        miembros = tabla.filter(pl.col("region") == nombre)["entidad"].to_list()
        print(f"  {nombre:<16} ({len(miembros)}): {', '.join(miembros)}")

    inestables = {cve: v for cve, v in votos.items() if len(v) > 1}
    print(f"\n  Entidades que cambian de región entre censos: {len(inestables)}/32")
    for cve, v in sorted(inestables.items(), key=lambda kv: NOMBRE[kv[0]]):
        reparto = ", ".join(f"{nombre} {n}" for nombre, n in v.most_common())
        print(f"    {NOMBRE[cve]:<16} {reparto}  → {v.most_common(1)[0][0]}")


def main():
    mig = mov.cargar_movilidad()
    for k, preset in PRESETS.items():
        generar(mig, k, preset["ruta"], preset["nucleos"])
    print()


if __name__ == "__main__":
    main()
