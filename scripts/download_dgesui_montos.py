#!/usr/bin/env python3
"""
Download per-university DGESUI "Subsidio Ordinario" data (JSON).

Fuente: https://dgesui.ses.sep.gob.mx/sep.subsidioentransparencia.mx/{año}/subsidio-ordinario/universidad/{SIGLA}/exportar
A diferencia del PEF (que solo agrega el subsidio por entidad), este endpoint
da el detalle por institución: Monto Federal, Monto Estatal, Monto Público,
Subsidio por Alumno, % Participación Federación/Estado, matrícula, rector/a,
dirección, etc.

La sigla es una de las claves de UNIVERSIDADES_DGESUI_2025 (en
scripts/prepare_comparacion_universidades.py) -- se valida contra ese mapeo
antes de tocar la red. El rango de años válido lo determina el servidor y
cambia con el tiempo (verificado 2019-2026 disponibles, no se hardcodea).
Sigla inválida o año sin dato → HTTP 404 con una página HTML "Not Found", no
JSON -- se detecta por status code, nunca se guarda HTML como si fuera JSON.

Usage:
  # Una universidad
  python scripts/download_dgesui_montos.py BUAP --year 2025 --output data/dgesui/montos/BUAP_2025.json

  # Todas las universidades del mapeo (sigla omitida)
  python scripts/download_dgesui_montos.py --year 2025 --output data/dgesui/montos/2025
"""

import argparse
import json
import sys
import time
from pathlib import Path

import requests
import urllib3

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_comparacion_universidades import UNIVERSIDADES_DGESUI_2025

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

BASE_URL = "https://dgesui.ses.sep.gob.mx/sep.subsidioentransparencia.mx"
DELAY = 0.3  # seconds between requests en modo lote


def resolver_sigla(sigla: str) -> str:
    """Sigla canónica (tal como está en UNIVERSIDADES_DGESUI_2025), buscando
    sin distinguir mayúsculas/minúsculas. Sale con error si no matchea."""
    por_mayusculas = {s.upper(): s for s in UNIVERSIDADES_DGESUI_2025}
    canonica = por_mayusculas.get(sigla.upper())
    if canonica is None:
        sys.exit(f"'{sigla}' no está en UNIVERSIDADES_DGESUI_2025 (scripts/prepare_comparacion_universidades.py). Siglas válidas: {sorted(UNIVERSIDADES_DGESUI_2025)}")
    return canonica


def descargar_universidad(session: requests.Session, sigla: str, year: int, destino: Path) -> bool:
    """Descarga el JSON de una universidad/año y lo guarda en `destino`.
    Devuelve True si tuvo éxito, False si el servidor respondió con error
    (no aborta -- quien llama decide si eso es fatal o se omite)."""
    url = f"{BASE_URL}/{year}/subsidio-ordinario/universidad/{sigla}/exportar"
    r = session.get(url, timeout=30, verify=False)
    if r.status_code != 200:
        print(f"  [aviso] {sigla} ({year}): HTTP {r.status_code} en {url}, se omite")
        return False
    data = r.json()

    destino.parent.mkdir(parents=True, exist_ok=True)
    with destino.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    montos = data.get("Montos", {})
    federal = montos.get("Monto Federal", {}).get("Número", "?")
    estatal = montos.get("Monto Estatal", {}).get("Número", "?")
    publico = montos.get("Monto Público", {}).get("Número", "?")
    print(f"  OK {sigla} ({year}) → {destino}  [Federal {federal} | Estatal {estatal} | Público {publico}]")
    return True


def main():
    parser = argparse.ArgumentParser(description="Download DGESUI Subsidio Ordinario data (JSON) por universidad")
    parser.add_argument("sigla", nargs="?", default=None, help="Sigla de la universidad (ver UNIVERSIDADES_DGESUI_2025). Si se omite, descarga todas.")
    parser.add_argument("--year", type=int, required=True, help="Año del programa Subsidio Ordinario")
    parser.add_argument("--output", type=Path, required=True, help="Con sigla: ruta del archivo a escribir. Sin sigla: directorio donde se guarda un archivo SIGLA_AÑO.json por universidad.")
    args = parser.parse_args()

    session = requests.Session()
    session.headers["User-Agent"] = "Mozilla/5.0 (compatible)"

    if args.sigla is not None:
        sigla = resolver_sigla(args.sigla)
        url = f"{BASE_URL}/{args.year}/subsidio-ordinario/universidad/{sigla}/exportar"
        r = session.get(url, timeout=30, verify=False)
        if r.status_code != 200:
            sys.exit(f"HTTP {r.status_code} en {url} -- revisa que '{sigla}' tenga datos para el año {args.year}.")
        data = r.json()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        montos = data.get("Montos", {})
        print(f"OK {sigla} ({args.year}) → {args.output}")
        print(f"  Monto Federal: {montos.get('Monto Federal', {}).get('Número', '?')}")
        print(f"  Monto Estatal: {montos.get('Monto Estatal', {}).get('Número', '?')}")
        print(f"  Monto Público: {montos.get('Monto Público', {}).get('Número', '?')}")
        return

    args.output.mkdir(parents=True, exist_ok=True)
    print(f"Descargando {len(UNIVERSIDADES_DGESUI_2025)} universidades para {args.year} → {args.output}/")
    ok, omitidas = 0, 0
    for i, sigla in enumerate(UNIVERSIDADES_DGESUI_2025):
        if i > 0:
            time.sleep(DELAY)
        destino = args.output / f"{sigla}_{args.year}.json"
        if descargar_universidad(session, sigla, args.year, destino):
            ok += 1
        else:
            omitidas += 1

    print(f"\n{ok} guardadas, {omitidas} omitidas (sin dato para {args.year} o error HTTP).")


if __name__ == "__main__":
    main()
