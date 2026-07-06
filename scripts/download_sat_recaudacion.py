#!/usr/bin/env python3
"""
Download SAT open-data files relevant to federal tax collection.

IMPORTANT — recaudación POR ENTIDAD FEDERATIVA is NOT published by SAT
as structured open data (verified 2026-07-06: minisitio DatosAbiertos,
cifras_sat, ITG PDF annexes 2014/2023, Azure blob probes). The informe
therefore approximates state-level fiscal contribution with PIBE shares
(see centralismo/informe/). What SAT does publish, downloaded here:

  - IngresosTributarios.xls  — recaudación nacional mensual por impuesto, 2010+
  - PorEntFed.xls            — padrón de contribuyentes activos POR ESTADO, mensual 2010+
                               (única serie estatal del SAT; proxy de base fiscal)

Output:
  data/sat/IngresosTributarios.xls
  data/sat/PorEntFed.xls

Usage:
  uv run python scripts/download_sat_recaudacion.py
"""

from pathlib import Path

import requests

BLOB = "https://wu1agsprosta001.blob.core.windows.net/agsc-publicaciones/Datos_abiertos/Cifras_SAT/Documents"
OUTPUT_DIR = Path("data/sat")

FILES = {
    "IngresosTributarios.xls": f"{BLOB}/IngresosTributarios.xls",
    "PorEntFed.xls": f"{BLOB}/PorEntFed.xls",
}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers["User-Agent"] = "Mozilla/5.0 (compatible)"

    for name, url in FILES.items():
        dest = OUTPUT_DIR / name
        if dest.exists():
            print(f"SKIP: {name}")
            continue
        r = session.get(url, timeout=120)
        r.raise_for_status()
        if not r.content.startswith(b"\xd0\xcf\x11\xe0"):  # OLE2 magic (legacy .xls)
            raise RuntimeError(f"{name}: unexpected content (not .xls) — source moved?")
        dest.write_bytes(r.content)
        print(f"OK ({len(r.content) // 1024} KB): {name}")


if __name__ == "__main__":
    main()
