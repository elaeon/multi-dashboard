#!/usr/bin/env python3
"""
Download INEGI datos abiertos (CSV) from DescargaMasiva ZIP manifests.

Works with any INEGI census program directory containing DescargaMasiva*.zip files.

Workflow:
  1. Place DescargaMasiva*.zip files (downloaded from INEGI) in the target directory
  2. Run this script — it reads the DescargaMasivaOD.xml inside each ZIP,
     extracts all file URLs, and downloads them to {base_dir}/{year}/.

INEGI serves HTTP 200 with ~2 KB of HTML for missing files (soft-404).
Valid ZIP files start with magic bytes b'PK\x03\x04' — used to detect real downloads.

Output structure:
  {base_dir}/{year}/{filename}.zip   ← datos abiertos CSV packages

Usage:
  uv run python scripts/download_inegi.py data/inegi/cngmd
  uv run python scripts/download_inegi.py data/inegi/cnge
  uv run python scripts/download_inegi.py data/inegi/cnge --years 2025
  uv run python scripts/download_inegi.py data/inegi/cnge --dry-run
"""
import argparse
import re
import time
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import requests

DELAY = 0.3
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible)"}


def is_zip(data: bytes) -> bool:
    return data[:4] == b"PK\x03\x04"


def year_from_url(url: str) -> str | None:
    m = re.search(r"/(\d{4})/", url)
    return m.group(1) if m else None


def load_manifests(cngmd_dir: Path, filter_years: set[str] | None) -> dict[str, list[str]]:
    """Parse all DescargaMasiva*.zip files and return {year: [url, ...]}."""
    by_year: dict[str, list[str]] = {}
    zips = sorted(cngmd_dir.glob("DescargaMasiva*.zip"))
    if not zips:
        print(f"No DescargaMasiva*.zip files found in {cngmd_dir}")
        return by_year
    for zpath in zips:
        with zipfile.ZipFile(zpath) as z:
            xml_names = [n for n in z.namelist() if n.endswith(".xml")]
            if not xml_names:
                print(f"WARNING: no XML in {zpath.name}")
                continue
            root = ET.fromstring(z.read(xml_names[0]).decode("utf-8-sig"))
        urls = [a.text.strip() for a in root.iter("Archivo") if a.text]
        for url in urls:
            year = year_from_url(url)
            if year is None:
                continue
            if filter_years and year not in filter_years:
                continue
            by_year.setdefault(year, []).append(url)
        total_mb = root.find("Descarga").get("totalMb", "?")
        years_in_zip = sorted({year_from_url(u) for u in urls if year_from_url(u)})
        print(f"Loaded {zpath.name}: {len(urls)} files, years={years_in_zip}, total={total_mb}")
    return by_year


def download_year(
    session: requests.Session,
    base_dir: Path,
    year: str,
    urls: list[str],
    dry_run: bool,
    ok_offset: int,
    grand_total: int,
) -> tuple[int, list[str]]:
    """Return (ok_count, errors). ok_offset/grand_total are for cross-year progress display."""
    dest_dir = base_dir / year
    dest_dir.mkdir(parents=True, exist_ok=True)
    ok = 0
    errors: list[str] = []
    for url in urls:
        parts = url.split("/")
        filename = f"{parts[-2]}_{parts[-1]}"
        dest = dest_dir / filename
        if dest.exists():
            print(f"SKIP  {year}/{filename}")
            continue
        if dry_run:
            print(f"DRY   {url}")
            continue
        try:
            r = session.get(url, timeout=120)
            r.raise_for_status()
        except Exception as e:
            msg = f"{year}/{filename}: {e}"
            print(f"ERROR {msg}")
            errors.append(msg)
            continue
        if not is_zip(r.content):
            msg = f"{year}/{filename} (soft-404, {len(r.content)} bytes)"
            print(f"MISS  {msg}")
            errors.append(msg)
            continue
        dest.write_bytes(r.content)
        ok += 1
        print(f"OK    {year}/{filename} ({len(r.content) // 1024:,} KB)  [{ok_offset + ok}/{grand_total}]")
        time.sleep(DELAY)
    return ok, errors


def check_downloads(base_dir: Path, by_year: dict[str, list[str]]) -> None:
    """Print which files are present/missing on disk."""
    total = missing = 0
    for year in sorted(by_year):
        dest_dir = base_dir / year
        year_missing = []
        for url in by_year[year]:
            parts = url.split("/")
            filename = f"{parts[-2]}_{parts[-1]}"
            total += 1
            if not (dest_dir / filename).exists():
                missing += 1
                year_missing.append(filename)
        status = "OK" if not year_missing else f"MISSING {len(year_missing)}"
        print(f"\n--- {year} [{status}] ---")
        for name in year_missing:
            print(f"  MISSING  {year}/{name}")
    print(f"\n{total - missing}/{total} files present")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download INEGI datos abiertos from DescargaMasiva XML manifests")
    parser.add_argument("base_dir", help="Directory containing DescargaMasiva*.zip files (e.g. data/inegi/cnge)")
    parser.add_argument("--years", nargs="+", help="Filter to specific year(s), e.g. --years 2025 2023")
    parser.add_argument("--dry-run", action="store_true", help="Print URLs without downloading")
    parser.add_argument("--check", action="store_true", help="Check which files are missing without downloading")
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    filter_years = set(args.years) if args.years else None

    by_year = load_manifests(base_dir, filter_years)
    if not by_year:
        return

    if args.check:
        check_downloads(base_dir, by_year)
        return

    session = requests.Session()
    session.headers.update(HEADERS)

    grand_total = sum(len(v) for v in by_year.values())
    total_ok = 0
    all_errors: list[str] = []
    for year in sorted(by_year):
        urls = by_year[year]
        print(f"\n--- {year} ({len(urls)} files) ---")
        ok, errors = download_year(session, base_dir, year, urls, args.dry_run, total_ok, grand_total)
        total_ok += ok
        all_errors.extend(errors)
    print(f"\nDownloaded {total_ok}/{grand_total} files")
    if all_errors:
        print(f"\nErrors ({len(all_errors)}):")
        for err in all_errors:
            print(f"  {err}")


if __name__ == "__main__":
    main()
