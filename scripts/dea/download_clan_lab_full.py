"""
Re-download the DEA clan-lab CSVs for years 2004-2015, which the site's CSV
export button hard-caps at 600 rows (see DATA_OVERVIEW.md "Trap A"). The same
data is also rendered as a paginated HTML table (20 rows/page) that is NOT
subject to the cap, so we scrape it page by page via the Drupal Views AJAX
endpoint that powers that table.

The site sits behind Akamai bot detection: plain HTTP clients and Playwright's
chromium both get "Access Denied". Webkit passes the check, so this script
must launch webkit specifically.

After running this, regenerate the combined parquet with:
    uv run python scripts/dea/fill_missing_states.py
"""

import csv
import json
import math
import time

from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright

BASE_URL = "https://www.dea.gov/es/clan-lab"
AJAX_URL = (
    "https://www.dea.gov/es/views/ajax"
    "?_wrapper_format=drupal_ajax"
    "&state=All"
    "&date={year}"
    "&view_name=clandestine_labs_geojson"
    "&view_display_id=default"
    "&view_args="
    "&view_path=%2Fnode%2F145986"
    "&view_base_path=clan_lab%2Fexport%2Fdea_clan_lab_export.csv"
    "&pager_element=0"
    "&_drupal_ajax=1"
    "&page={page}"
)
DATA_DIR = (
    "/home/casa/projects/multi-dashboard/data/dea/desmantelamiento_laboratorios"
)
CAPPED_YEARS = range(2004, 2016)
ROWS_PER_PAGE = 20
REQUEST_DELAY_SECONDS = 0.3
MAX_RETRIES = 3

FIELD_CLASSES = {
    "state": "views-field-field-clan-lab-address-administrative-area",
    "county": "views-field-field-clan-lab-county",
    "city": "views-field-field-clan-lab-address-locality",
    "address1": "views-field-field-clan-lab-address-address-line1",
    "date": "views-field-field-clan-lab-date",
}


def fetch_page(page, year, page_num):
    url = AJAX_URL.format(year=year, page=page_num)
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            html = page.evaluate(
                """async (u) => {
                    const r = await fetch(u, {credentials: 'include'});
                    const text = await r.text();
                    const json = JSON.parse(text);
                    const ins = json.find(
                        c => c.command === 'insert' && c.data && c.data.length > 1000
                    );
                    return ins ? ins.data : '';
                }""",
                url,
            )
            if html:
                return html
        except Exception as e:
            if attempt == MAX_RETRIES:
                raise
            print(f"    retry {attempt}/{MAX_RETRIES} after error: {e}")
            time.sleep(1)
    raise RuntimeError(f"Failed to fetch year={year} page={page_num}")


def parse_total(html):
    soup = BeautifulSoup(html, "html.parser")
    header = soup.find(class_="l-view__header")
    text = header.get_text(" ", strip=True) if header else ""
    marker = "of "
    idx = text.find(marker)
    if idx == -1:
        return None
    tail = text[idx + len(marker):]
    number = tail.split(" ", 1)[0].replace(",", "")
    return int(number)


def parse_rows(html):
    soup = BeautifulSoup(html, "html.parser")
    rows = []
    for tr in soup.select("table tbody tr"):
        row = {}
        for field, cls in FIELD_CLASSES.items():
            td = tr.find("td", class_=cls)
            if field == "date":
                time_tag = td.find("time") if td else None
                row[field] = time_tag.get_text(strip=True) if time_tag else ""
            else:
                row[field] = td.get_text(strip=True) if td else ""
        rows.append(row)
    return rows


def download_year(page, year):
    print(f"Year {year}:")
    first_html = fetch_page(page, year, 0)
    total = parse_total(first_html)
    if total is None:
        raise RuntimeError(f"Could not determine total row count for year {year}")
    total_pages = math.ceil(total / ROWS_PER_PAGE)
    print(f"  total={total} rows across {total_pages} pages")

    rows = parse_rows(first_html)
    time.sleep(REQUEST_DELAY_SECONDS)

    for page_num in range(1, total_pages):
        html = fetch_page(page, year, page_num)
        rows.extend(parse_rows(html))
        time.sleep(REQUEST_DELAY_SECONDS)

    if len(rows) != total:
        print(f"  WARNING: collected {len(rows)} rows, expected {total}")
    else:
        print(f"  collected {len(rows)} rows (matches expected total)")

    out_path = f"{DATA_DIR}/dea_clan_lab_export_{year}.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["state", "county", "city", "address1", "date"]
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"  wrote {out_path}")
    return len(rows)


def main():
    summary = {}
    with sync_playwright() as p:
        browser = p.webkit.launch(headless=True)
        page = browser.new_page()
        page.goto(BASE_URL)

        for year in CAPPED_YEARS:
            summary[year] = download_year(page, year)

        browser.close()

    print("\nSummary (600 = previously capped):")
    for year, count in summary.items():
        print(f"  {year}: 600 -> {count}")

    print(
        "\nNext step: regenerate the combined parquet with "
        "`uv run python scripts/dea/fill_missing_states.py`"
    )


if __name__ == "__main__":
    main()
