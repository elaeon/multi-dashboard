#!/usr/bin/env python3
"""
Download SECTUR DataTur weekly hotel-occupancy Excel files.

Scrapes https://datatur.sectur.gob.mx/SitePages/hoteleria.aspx for weekly
report ZIPs ({YEAR}_Semana_{WW}.zip), downloads each in memory, and extracts
only the Excel file into data/sectur/hoteleria/. Monthly reports and PDFs
are ignored; no ZIP is written to disk.

Usage:
  python scripts/download_sectur_hoteleria.py
"""

import io
import re
import time
import zipfile
from pathlib import Path
from urllib.parse import urljoin

import requests

PAGE_URL = "https://datatur.sectur.gob.mx/SitePages/hoteleria.aspx"
OUTPUT_DIR = Path("data/sectur/hoteleria")
DELAY = 0.5  # seconds between requests


def find_weekly_zip_urls(session: requests.Session) -> list[str]:
    r = session.get(PAGE_URL, timeout=30)
    r.raise_for_status()
    hrefs = re.findall(r'href="([^"]*Semana[^"]*\.zip)"', r.text, flags=re.IGNORECASE)
    return sorted({urljoin(PAGE_URL, h) for h in hrefs})


def extract_excel(zip_bytes: bytes, target: Path) -> bool:
    """Extract the Excel member to `target` (zip member names can differ from the zip stem)."""
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        for member in zf.namelist():
            if member.lower().endswith((".xls", ".xlsx")):
                target.write_bytes(zf.read(member))
                return True
    return False


def main():
    session = requests.Session()
    session.headers["User-Agent"] = "Mozilla/5.0 (compatible)"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    urls = find_weekly_zip_urls(session)
    print(f"{len(urls)} weekly zip links found")

    for url in urls:
        stem = url.split("/")[-1].removesuffix(".zip")
        target = OUTPUT_DIR / f"{stem}.xlsx"
        if target.exists():
            print(f"  SKIP: {target.name}")
            continue
        try:
            time.sleep(DELAY)
            r = session.get(url, timeout=120)
            r.raise_for_status()
            if extract_excel(r.content, target):
                print(f"  OK ({target.stat().st_size // 1024} KB): {target.name}")
            else:
                print(f"  NO EXCEL in {url.split('/')[-1]}")
        except Exception as e:
            print(f"  ERROR {url.split('/')[-1]}: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
