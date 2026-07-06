#!/usr/bin/env python3
"""
Download INEGI PIBE (PIB por Entidad Federativa, base 2018) tabulados.

Source pattern:
  https://www.inegi.org.mx/contenidos/programas/pibent/2018/tabulados/ori/PIBE_<n>.xlsx

Notes:
  - Series: 2003-2024R, millones de pesos a precios constantes de 2018.
  - INEGI serves soft-404s (HTML with status 200) for missing numbers
    (e.g. PIBE_1, PIBE_11, PIBE_47) — files are validated by ZIP magic bytes.
  - A manifest.csv (numero, titulo) is written from each file's Tabulado sheet.

Output structure:
  data/inegi/pibe/PIBE_<n>.xlsx
  data/inegi/pibe/manifest.csv

Usage:
  uv run python scripts/download_pibe.py            # download everything
  uv run python scripts/download_pibe.py --dry-run  # print URLs only
"""

import argparse
import csv
import time
from pathlib import Path

import openpyxl
import requests

BASE_URL = "https://www.inegi.org.mx/contenidos/programas/pibent/2018/tabulados/ori"
OUTPUT_DIR = Path("data/inegi/pibe")
MAX_N = 85  # probe range; real files observed: 2-80 with gaps
DELAY = 0.3  # seconds between requests


def is_xlsx(payload: bytes) -> bool:
    return payload[:4] == b"PK\x03\x04"


def tabulado_title(path: Path) -> str:
    wb = openpyxl.load_workbook(path, read_only=True)
    try:
        rows = wb["Tabulado"].iter_rows(min_row=3, max_row=3, max_col=1, values_only=True)
        return str(next(rows)[0] or "").strip()
    finally:
        wb.close()


def main():
    parser = argparse.ArgumentParser(description="Download INEGI PIBE tabulados")
    parser.add_argument("--dry-run", action="store_true", help="Print URLs without downloading")
    args = parser.parse_args()

    session = requests.Session()
    session.headers["User-Agent"] = "Mozilla/5.0 (compatible)"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = []
    for n in range(1, MAX_N + 1):
        url = f"{BASE_URL}/PIBE_{n}.xlsx"
        dest = OUTPUT_DIR / f"PIBE_{n}.xlsx"

        if dest.exists():
            manifest.append((n, tabulado_title(dest)))
            print(f"SKIP: {dest.name}")
            continue
        if args.dry_run:
            print(f"DRY: {url}")
            continue

        time.sleep(DELAY)
        try:
            r = session.get(url, timeout=60)
            r.raise_for_status()
        except Exception as e:
            print(f"ERROR PIBE_{n}: {e}")
            continue

        if not is_xlsx(r.content):
            print(f"MISSING (soft-404): PIBE_{n}")
            continue

        dest.write_bytes(r.content)
        title = tabulado_title(dest)
        manifest.append((n, title))
        print(f"OK ({len(r.content) // 1024} KB): {dest.name} — {title[:80]}")

    if manifest and not args.dry_run:
        with open(OUTPUT_DIR / "manifest.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["numero", "titulo"])
            w.writerows(manifest)
        print(f"\nmanifest.csv: {len(manifest)} tabulados")


if __name__ == "__main__":
    main()
