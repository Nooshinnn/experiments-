import time
import csv
import os
import signal
import requests
from bs4 import BeautifulSoup
from datetime import datetime

OUTPUT_FILE   = "phishing_urls_dataset.csv"
DELAY_SECONDS = 1.5
BASE_URL      = "https://www.phishtank.net/phish_search.php?page={page}&valid=y&Search=Search"
START_PAGE    = 0

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}


stop_requested = False

def handle_sigint(sig, frame):
    global stop_requested
    print("\n\n[!] Ctrl+C detected — finishing current page then stopping...")
    stop_requested = True

signal.signal(signal.SIGINT, handle_sigint)


def open_csv_writer(filepath: str):
    file_exists = os.path.isfile(filepath) and os.path.getsize(filepath) > 0
    f = open(filepath, "a", newline="", encoding="utf-8-sig")
    writer = csv.writer(f)
    if not file_exists:
        writer.writerow(["url", "label"])
    return f, writer


def load_existing_urls(filepath: str) -> set:
    seen = set()
    if not os.path.isfile(filepath):
        return seen
    with open(filepath, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            seen.add(row["url"])
    print(f"  Loaded {len(seen)} existing URLs (resuming previous run)")
    return seen


def fetch_page(url: str) -> BeautifulSoup | None:
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        return BeautifulSoup(resp.text, "html.parser")
    except requests.RequestException as e:
        print(f"  [ERROR] {e}")
        return None


def extract_urls(soup: BeautifulSoup) -> list[str]:
    urls = []
    table = soup.find("table")
    if not table:
        return urls
    for row in table.find_all("tr")[1:]:   # skip header
        cells = row.find_all("td")
        if len(cells) < 2:
            continue
        raw_text   = cells[1].get_text(separator="\n").strip()
        first_line = raw_text.split("\n")[0].strip()
        if first_line.startswith("http"):
            urls.append(first_line)
    return urls


def is_last_page(soup: BeautifulSoup) -> bool:
    table = soup.find("table")
    if not table:
        return True
    rows = table.find_all("tr")
    return len(rows) <= 1   # only header row = no results



def main():
    global stop_requested

    print("=" * 55)
    print("  PhishTank Valid Phishes — Unlimited Scraper")
    print("  Press Ctrl+C at any time to stop and save.")
    print("=" * 55)

    seen        = load_existing_urls(OUTPUT_FILE)
    total       = len(seen)
    page        = START_PAGE
    retry_count = 0
    MAX_RETRIES = 5

    csv_file, writer = open_csv_writer(OUTPUT_FILE)

    try:
        while not stop_requested:
            url       = BASE_URL.format(page=page)
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"\n[{timestamp}] Page {page} | Total collected: {total}")
            print(f"  {url}")

            soup = fetch_page(url)

            # ── Retry on network failure ───────────────
            if soup is None:
                retry_count += 1
                if retry_count >= MAX_RETRIES:
                    print(f"  [!] {MAX_RETRIES} consecutive failures — stopping.")
                    break
                wait = DELAY_SECONDS * retry_count * 2
                print(f"  Retrying in {wait:.0f}s... (attempt {retry_count}/{MAX_RETRIES})")
                time.sleep(wait)
                continue

            retry_count = 0

            # ── Detect end of archive ──────────────────
            if is_last_page(soup):
                print("\n  Reached the end of the PhishTank archive.")
                print("  Restarting from page 0 in 60s to catch new submissions...")
                time.sleep(60)
                page = START_PAGE
                continue

            # ── Extract & save ─────────────────────────
            page_urls = extract_urls(soup)
            new_count = 0

            for url in page_urls:
                if url not in seen:
                    seen.add(url)
                    writer.writerow([url, 1])
                    new_count += 1

            csv_file.flush()    # write to disk immediately
            total += new_count
            print(f"  +{new_count} new URLs written | Running total: {total}")

            page += 1
            time.sleep(DELAY_SECONDS)

    finally:
        csv_file.close()
        print(f"\n{'=' * 55}")
        print(f"  Stopped. {total} total URLs saved to: {OUTPUT_FILE}")
        print(f"{'=' * 55}")


if __name__ == "__main__":
    main()
