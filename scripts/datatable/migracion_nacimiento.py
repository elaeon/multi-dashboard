"""
Migración interestatal acumulada por LUGAR DE NACIMIENTO, censos 1950–2020.

Mide el stock de toda la vida: cuántos residentes de una entidad nacieron en otra, y
cuántos nacidos en ella residen hoy en otra. NO es comparable con
scripts/datatable/movilidad_estatal.py, que mide el flujo del quinquenio previo — en
2020, 3.8 M de personas cambiaron de entidad en 5 años, pero 21.6 M residen en una
entidad distinta a la de su nacimiento.

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

Run: uv run python scripts/datatable/migracion_nacimiento.py --entidad Jalisco --año 1970
     uv run python scripts/datatable/migracion_nacimiento.py --entidad cdmx --serie
     uv run python scripts/datatable/migracion_nacimiento.py --entidad Jalisco --año 1950-1980
     uv run python scripts/datatable/migracion_nacimiento.py --convergencia
     uv run python scripts/datatable/migracion_nacimiento.py --convergencia --año 1960 --k 5
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(RAIZ / "scripts"))
sys.path.insert(0, str(RAIZ / "scripts" / "centralismo"))
from comun import NOMBRE
from movilidad_estatal import _entidad, _tabla
from prepare_ccpv_nacimiento import TERRITORIOS

DIR = RAIZ / "dashboard_data"
CENSOS = [1950, 1960, 1970, 1980, 1990, 2000, 2010, 2020]
CENSOS_CONVERGENCIA = (1950, 1960, 1970, 1980)


def _año_o_rango(valor):
    """Un censo suelto ('1970') o un rango inclusivo ('1950-1980') → int o tupla."""
    if "-" in valor:
        partes = valor.split("-")
        if len(partes) != 2:
            raise argparse.ArgumentTypeError(f"rango inválido: {valor!r} (usa 'AAAA-AAAA')")
        try:
            lo, hi = sorted(int(p) for p in partes)
        except ValueError:
            raise argparse.ArgumentTypeError(f"rango inválido: {valor!r} (usa 'AAAA-AAAA')")
        censos = tuple(c for c in CENSOS if lo <= c <= hi)
        if not censos:
            raise argparse.ArgumentTypeError(
                f"ningún censo cae en el rango {lo}-{hi} (censos disponibles: {CENSOS})")
        return censos
    try:
        censo = int(valor)
    except ValueError:
        raise argparse.ArgumentTypeError(f"año inválido: {valor!r}")
    if censo not in CENSOS:
        raise argparse.ArgumentTypeError(f"censo no disponible: {censo} (elige entre {CENSOS})")
    return censo


def _k(valor):
    try:
        n = int(valor)
    except ValueError:
        raise argparse.ArgumentTypeError(f"k inválido: {valor!r} (usa un entero, p. ej. 5)")
    if n < 2:
        raise argparse.ArgumentTypeError(f"k inválido: {n} (debe ser >= 2)")
    return n


def cargar():
    ruta = DIR / "ccpv_nacimiento_estatal.parquet"
    if not ruta.exists():
        raise SystemExit(f"Falta {ruta}.\n"
                         "Corre primero: uv run python scripts/prepare_ccpv_nacimiento.py")
    return pl.read_parquet(ruta)


def flujos(df, cve, censo):
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


def detalle(df, cve, censo, top):
    inm, emi, poblacion = flujos(df, cve, censo)
    n_inm, n_emi = inm["personas"].sum(), emi["personas"].sum()

    print(f"\n{'═' * 72}")
    print(f"  {NOMBRE[cve]} — Censo {censo}")
    print("  Migración interestatal acumulada, por lugar de nacimiento")
    print(f"  Población total: {poblacion:,}")
    if cve in TERRITORIOS.get(censo, ()):
        print("  Nota: en este censo aún era territorio federal, no estado.")
    print("═" * 72)

    _tabla("INMIGRANTES — residen aquí pero nacieron en otra entidad",
           inm, "origen", n_inm, top)
    _tabla("EMIGRANTES — nacieron aquí pero residen en otra entidad",
           emi, "destino", n_emi, top)

    neto = n_inm - n_emi
    print(f"\nSALDO NETO ACUMULADO: {neto:+,}"
          f"  ({neto / poblacion * 1000:+.2f} por mil habitantes)")
    print(f"Residentes nacidos en otra entidad: {n_inm / poblacion * 100:.2f}% de la población")

    print(f"\n{'─' * 72}")
    otras = (df.filter((pl.col("censo") == censo) & (pl.col("cve_destino") == cve)
                       & pl.col("total_categoria")
                       & ~pl.col("categoria").is_in(["Total", "En la entidad", "En otra entidad"]))
             .sort("personas", descending=True))
    for r in otras.iter_rows(named=True):
        print(f"  {r['categoria']:<44} {r['personas']:>12,}")
    print()
    return inm, emi


def acumulado(df, cve, censos, top):
    """Reporte de un rango de censos, usando el CENSO DESTINO (el más reciente
    del rango) como cifra reportada — es un stock, ya consistente con la
    población real de ese año, así que sumarlo con censos anteriores duplicaría
    personas que nunca cambiaron de entidad. Los censos anteriores del rango se
    muestran como trayectoria de contexto, sin sumarse.
    """
    censo_destino = censos[-1]
    print(f"\n{'═' * 72}")
    print(f"  Rango solicitado: {censos[0]}–{censos[-1]} → usando censo destino {censo_destino}")
    print("  (el stock de migración por nacimiento es acumulado a cada fecha censal;")
    print("   sumar varios censos duplicaría personas. Se reporta el más reciente del")
    print("   rango, ya consistente con la población real de ese año.)")

    inm, emi = detalle(df, cve, censo_destino, top)

    if len(censos) > 1:
        print(f"{'─' * 72}")
        print("Trayectoria dentro del rango (censos anteriores, no se suman):")
        print(f"  {'Censo':<7} {'Población':>13} {'Inmigrantes':>13} {'Emigrantes':>13}"
              f" {'Saldo':>13} {'% nac. fuera':>13}")
        for censo in censos:
            i, e, poblacion = flujos(df, cve, censo)
            n_i, n_e = i["personas"].sum(), e["personas"].sum()
            nota = " *" if cve in TERRITORIOS.get(censo, ()) else ""
            print(f"  {censo:<7} {poblacion:>13,} {n_i:>13,} {n_e:>13,}"
                  f" {n_i - n_e:>+13,} {n_i / poblacion * 100:>12.2f}%{nota}")
        if any(cve in TERRITORIOS.get(c, ()) for c in censos):
            print("  * territorio federal en ese censo, no estado")
        print()

    return inm, emi


def serie(df, cve):
    print(f"\n{'═' * 78}")
    print(f"  {NOMBRE[cve]} — migración interestatal acumulada, 1950–2020")
    print("═" * 78)
    print(f"  {'Censo':<7} {'Población':>13} {'Inmigrantes':>13} {'Emigrantes':>13}"
          f" {'Saldo':>13} {'% nac. fuera':>13}")
    for censo in CENSOS:
        inm, emi, poblacion = flujos(df, cve, censo)
        n_inm, n_emi = inm["personas"].sum(), emi["personas"].sum()
        nota = " *" if cve in TERRITORIOS.get(censo, ()) else ""
        print(f"  {censo:<7} {poblacion:>13,} {n_inm:>13,} {n_emi:>13,}"
              f" {n_inm - n_emi:>+13,} {n_inm / poblacion * 100:>12.2f}%{nota}")
    if any(cve in TERRITORIOS.get(c, ()) for c in CENSOS):
        print("\n  * territorio federal en ese censo, no estado")
    print()


def _matriz_flujo(df, censo):
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


def convergencia(df, censo, k, top):
    """Agrupa las entidades de un censo por similitud de perfil migratorio:
    'emigracion' (mismos destinos preferentes) e 'inmigracion' (mismas
    fuentes preferentes). Imprime, por grupo, sus miembros y el/los
    destino(s)/origen(es) compartido(s) dominante(s). Devuelve
    {direccion: {cve_ent: cluster_id}} para --guardar."""
    cves, matriz = _matriz_flujo(df, censo)
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


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--entidad", type=_entidad, default=None,
                   help="Nombre o clave INEGI (1-32) de la entidad federativa "
                        "(obligatorio salvo con --convergencia)")
    p.add_argument("--año", type=_año_o_rango,
                   help="Año censal, o rango 'AAAA-AAAA' para el acumulado de varios "
                        "censos (obligatorio salvo con --serie; con --convergencia, "
                        "default 1950-1980)")
    p.add_argument("--top", type=int, default=10, help="Filas por tabla (default 10)")
    p.add_argument("--serie", action="store_true",
                   help="Trayectoria de la entidad en los 8 censos, en vez del detalle")
    p.add_argument("--convergencia", action="store_true",
                   help="Agrupa las 32 entidades por similitud de perfil migratorio "
                        "(destinos preferentes en emigración, fuentes preferentes en "
                        "inmigración) en vez de reportar una sola entidad. Incompatible "
                        "con --entidad")
    p.add_argument("--k", type=_k, default=None,
                   help="Número de grupos para --convergencia. Si se omite, se "
                        "autodetecta vía silhouette score (k=2..10)")
    p.add_argument("--guardar", action="store_true",
                   help="Escribe el resultado a dashboard_data/nacimiento_<entidad>_<año>.parquet "
                        "(o dashboard_data/convergencia_<direccion>_<censo>.parquet con --convergencia)")
    a = p.parse_args()
    if a.convergencia and a.entidad is not None:
        p.error("--convergencia es incompatible con --entidad (es un reporte cruzado de las 32 entidades)")
    if not a.convergencia and a.entidad is None:
        p.error("se requiere --entidad (o usa --convergencia)")
    if not a.convergencia and not a.serie and a.año is None:
        p.error("se requiere --año (o usa --serie)")

    df = cargar()

    if a.convergencia:
        censos = list(a.año) if isinstance(a.año, tuple) else [a.año] if a.año else list(CENSOS_CONVERGENCIA)
        for censo in censos:
            resultado = convergencia(df, censo, a.k, a.top)
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

    cve = a.entidad
    if a.serie:
        serie(df, cve)
        return

    if isinstance(a.año, tuple):
        inm, emi = acumulado(df, cve, list(a.año), a.top)
        etiqueta_año = f"{a.año[0]}-{a.año[-1]}"
    else:
        inm, emi = detalle(df, cve, a.año, a.top)
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


if __name__ == "__main__":
    main()
