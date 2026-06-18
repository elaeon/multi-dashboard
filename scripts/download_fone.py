#!/usr/bin/env python3
"""
Download FONE quarterly payroll data from dgsanef.sep.gob.mx.

Downloads 4 priority tables for all states across 9 periods:
  2024 Q1–Q4, 2025 Q1–Q4, 2026 Q1

Tables:
  - AnaliticoPlazas
  - PlazasDocAdmtvasDirec
  - MovimientosPlaza
  - PersonalLicencias

Output structure:
  data/fone/<year>/Q<n>/<state>/

Usage:
  python scripts/download_fone.py              # download everything
  python scripts/download_fone.py --dry-run    # print URLs only
  python scripts/download_fone.py --year 2025 --quarter 1
"""

import re
import time
import argparse
import unicodedata
from pathlib import Path
from urllib.parse import urljoin

import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

BASE_URL = "https://dgsanef.sep.gob.mx"
OUTPUT_DIR = Path("data/fone")
DELAY = 0.5  # seconds between requests

# Keyword fragments that appear in priority table page slugs/filenames (lowercase)
TABLE_KEYWORDS = [
    "analitico_de_plazas",
    "plazas_docentes",
    "movimientos_de_plazas",
    "personal_con_licencia",
]

# Substrings that disqualify a match (to exclude personal_con_licencia_prejubilatoria)
TABLE_EXCLUDE = ["prejubilatoria"]

DYNAMIC_PERIODS = [
    {"year": 2024, "quarter": 2, "index_url": f"{BASE_URL}/segundo_trimestre_2024"},
    {"year": 2024, "quarter": 3, "index_url": f"{BASE_URL}/tercer_trimestre_2024"},
    {"year": 2024, "quarter": 4, "index_url": f"{BASE_URL}/cuarto_trimestre_2024"},
    {"year": 2025, "quarter": 1, "index_url": f"{BASE_URL}/primer_trimestre_2025"},
    {"year": 2025, "quarter": 2, "index_url": f"{BASE_URL}/Segundo_Trimestre_2025"},
    {"year": 2025, "quarter": 3, "index_url": f"{BASE_URL}/Tercer_Trimestre_2025"},
    {"year": 2025, "quarter": 4, "index_url": f"{BASE_URL}/Cuarto_Trimestre_2025"},
]


def normalize_name(name: str) -> str:
    """Convert 'Baja California Sur' → 'baja_california_sur'."""
    nfkd = unicodedata.normalize("NFD", name)
    ascii_str = nfkd.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "_", ascii_str.lower()).strip("_")


def get_soup(session: requests.Session, url: str):
    from bs4 import BeautifulSoup
    try:
        r = session.get(url, timeout=30, verify=False)
        r.raise_for_status()
        # Use raw bytes so BeautifulSoup detects encoding from <meta charset>
        # rather than relying on requests' often-wrong latin-1 default
        return BeautifulSoup(r.content, "html.parser")
    except Exception as e:
        print(f"  ERROR fetching {url}: {e}")
        return None


def is_priority_table(href: str) -> bool:
    h = href.lower()
    if any(ex in h for ex in TABLE_EXCLUDE):
        return False
    return any(kw in h for kw in TABLE_KEYWORDS)


def find_zip_url(soup, page_url: str) -> str | None:
    for a in soup.find_all("a", href=True):
        if a["href"].lower().endswith(".zip"):
            return urljoin(page_url, a["href"])
    return None


def download_zip(session: requests.Session, url: str, dest: Path, dry_run: bool) -> None:
    if dest.exists():
        print(f"    SKIP: {dest.name}")
        return
    if dry_run:
        print(f"    DRY: {url}")
        print(f"      → {dest}")
        return
    try:
        r = session.get(url, timeout=120, stream=True, verify=False)
        r.raise_for_status()
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=65536):
                f.write(chunk)
        print(f"    OK ({dest.stat().st_size // 1024} KB): {dest.name}")
    except Exception as e:
        print(f"    ERROR: {e}")


def process_table_page(session, table_url: str, dest_dir: Path, dry_run: bool) -> None:
    time.sleep(DELAY)
    soup = get_soup(session, table_url)
    if not soup:
        return
    zip_url = find_zip_url(soup, table_url)
    if not zip_url:
        print(f"  NO ZIP at {table_url}")
        return
    filename = zip_url.split("/")[-1].split("?")[0]
    download_zip(session, zip_url, dest_dir / filename, dry_run)


def process_dynamic_period(session, period: dict, dry_run: bool) -> None:
    year, quarter = period["year"], period["quarter"]
    print(f"\n{'='*60}")
    print(f"{year} Q{quarter} (dynamic)")
    print(f"{'='*60}")

    soup = get_soup(session, period["index_url"])
    if not soup:
        return

    qt = f"_{quarter}t_{year}"
    state_links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if qt.lower() in href.lower() and href.startswith("/") and len(href) > 6:
            state_links.append((a.get_text(strip=True), f"{BASE_URL}{href}"))

    if not state_links:
        print("  No state links found")
        return

    print(f"  {len(state_links)} states")
    for state_name, state_url in state_links:
        time.sleep(DELAY)
        state_soup = get_soup(session, state_url)
        if not state_soup:
            continue

        dest_dir = OUTPUT_DIR / str(year) / f"Q{quarter}" / normalize_name(state_name)
        print(f"\n  [{state_name}]")

        for a in state_soup.find_all("a", href=True):
            href = a["href"]
            if is_priority_table(href):
                table_url = f"{BASE_URL}{href}" if href.startswith("/") else urljoin(state_url, href)
                print(f"    → {href.split('/')[-1]}")
                process_table_page(session, table_url, dest_dir, dry_run)


def process_static_period(
    session,
    year: int,
    quarter: int,
    index_url: str,
    dry_run: bool,
) -> None:
    print(f"\n{'='*60}")
    print(f"{year} Q{quarter} (static HTML)")
    print(f"{'='*60}")

    soup = get_soup(session, index_url)
    if not soup:
        return

    qt = f"1t_{year}"
    state_urls = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if qt in href.lower() and href.lower().endswith(".html"):
            full_url = urljoin(index_url, href)
            if "storage" in full_url:
                state_urls.append(full_url)

    if not state_urls:
        print("  No state pages found")
        return

    print(f"  {len(state_urls)} states")
    for state_url in state_urls:
        # Derive state name from the directory above the HTML file
        parts = state_url.rstrip("/").split("/")
        state_name = parts[-2] if len(parts) >= 2 else "unknown"
        dest_dir = OUTPUT_DIR / str(year) / f"Q{quarter}" / normalize_name(state_name)

        time.sleep(DELAY)
        state_soup = get_soup(session, state_url)
        if not state_soup:
            continue

        print(f"\n  [{state_name}]")
        for a in state_soup.find_all("a", href=True):
            href = a["href"]
            if is_priority_table(href):
                table_url = urljoin(state_url, href)
                print(f"    → {href}")
                process_table_page(session, table_url, dest_dir, dry_run)


def main():
    parser = argparse.ArgumentParser(description="Download FONE data from SEP DGSANEF")
    parser.add_argument("--dry-run", action="store_true", help="Print URLs without downloading")
    parser.add_argument("--year", type=int, help="Filter to specific year (2024, 2025, 2026)")
    parser.add_argument("--quarter", type=int, choices=[1, 2, 3, 4], help="Filter to specific quarter")
    args = parser.parse_args()

    session = requests.Session()
    session.headers["User-Agent"] = "Mozilla/5.0 (compatible)"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    def want(year, quarter):
        if args.year and args.year != year:
            return False
        if args.quarter and args.quarter != quarter:
            return False
        return True

    # 2024 Q1 — static HTML (historical archive)
    if want(2024, 1):
        process_static_period(
            session, 2024, 1,
            index_url=f"{BASE_URL}/historico_primer_trimestre_2024",
            dry_run=args.dry_run,
        )

    # 2024 Q2–Q4 and 2025 Q1–Q4 — dynamic pages
    for period in DYNAMIC_PERIODS:
        if want(period["year"], period["quarter"]):
            process_dynamic_period(session, period, args.dry_run)

    # 2026 Q1 — static HTML
    if want(2026, 1):
        process_static_period(
            session, 2026, 1,
            index_url=f"{BASE_URL}/storage/recursos/art73lgcg/Primer_Trimestre_2026/Primer_Trimestre_2026.html",
            dry_run=args.dry_run,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
